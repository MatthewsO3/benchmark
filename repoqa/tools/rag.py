"""Semantic search tool: embedding RAG over the repo (Qdrant + sentence-transformers).

Vectors are stored in a **local, on-disk Qdrant** collection (``QdrantClient(path=...)``)
rather than an in-memory FAISS index. Qdrant runs embedded in-process — it opens
no network port and needs no server or root, which suits a shared machine — while
scaling to the millions of chunks a large C codebase like the Linux kernel
produces, far past what a brute-force flat index handles comfortably.

The embedding model is :data:`config.EMBED_MODEL` (swap via the ``EMBED_MODEL``
env var — never by editing the installed package). The collection is keyed on a
fingerprint of (embed model, chunking parameters, repo contents), so changing
the model, the chunk size or the repo transparently rebuilds instead of
colliding with a stale collection. The build writes a completion marker only
after every chunk is upserted, so a run that dies mid-build is rebuilt next time
rather than being searched as a half-populated (and silently wrong) index.

Concurrency note: embedded Qdrant takes an exclusive lock on its on-disk
directory, so only **one process** can open the store at a time. Run benchmark
sweeps sequentially; starting two at once makes the second fail to open the
store. (The fingerprint also uses file mtime, so a fresh checkout or copy of the
same repo rebuilds the index even though its contents are unchanged.)
"""
import os
import logging

from langchain_core.tools import tool

from .. import config
from .common import walk_repo, iter_chunks_with_lines

log = logging.getLogger("repoqa.tools.rag")

#: Directory holding the embedded Qdrant storage (persists across runs).
_CACHE_DIR = ".qdrant_cache"
#: Number of chunks returned per query (default; overridable per call).
_TOP_K = 3
#: Number of points per Qdrant upsert request (bounds a single payload size).
_BATCH = 512
#: Number of texts handed to one ``encode`` call. Larger spans let the GPU-backed
#: encoder keep its batches full (it sub-batches internally by EMBED_BATCH);
#: also bounds peak host memory for the vectors AND texts of one span.
_EMBED_SPAN = 8192


class RagIndex:
    """Lazy embedded-Qdrant index over the repo using a sentence-transformers model.

    The collection is persisted on disk and keyed on repo contents + embedding
    model + chunking parameters, so repeated runs skip re-embedding while a
    change to any of those transparently rebuilds.

    A collection is only treated as a cache hit once a **completion marker** has
    been written (see :meth:`_marker_path`). A run that dies partway through the
    upserts leaves the marker absent, so the next run rebuilds instead of
    silently searching a half-populated index.
    """

    def __init__(self):
        self._ready = False
        self._model = None
        self._client = None
        self._collection = None
        self._empty = False  # repo produced no chunks

    def _fingerprint(self, repo: str) -> str:
        """Compute a cache key from the embed model, chunking params and sources.

        The digest folds in the chunking parameters (max chars / overlap) in
        addition to the model and each file's (path, size, mtime). Changing the
        chunk size therefore yields a new collection rather than reusing an
        index cut at the old size.

        :param repo: Root directory of the target repo.
        :return: A short hex digest.
        """
        import hashlib

        digest = hashlib.sha256()
        digest.update(config.EMBED_MODEL.encode())
        # Chunking params: a different chunk size must not collide with a stale
        # index built at the previous size.
        digest.update(f"chunk={config.CHUNK_MAX_CHARS}:"
                      f"{config.CHUNK_OVERLAP_LINES}".encode())
        for path in sorted(walk_repo(repo)):
            try:
                stat = os.stat(path)
            except OSError:
                continue
            digest.update(path.encode("utf-8", "ignore"))
            digest.update(str(stat.st_size).encode())
            digest.update(str(int(stat.st_mtime)).encode())
        return digest.hexdigest()[:16]

    def _marker_path(self) -> str:
        """Path of the sentinel file marking a fully-built collection.

        The marker is written only after every chunk has been upserted, so its
        presence means the collection for this fingerprint is complete. Keyed on
        the collection name so distinct fingerprints have distinct markers.
        """
        return os.path.join(_CACHE_DIR, f"{self._collection}.complete")

    def _resolve_device(self) -> str:
        """Resolve :data:`config.EMBED_DEVICE` to a concrete torch device string.

        ``"auto"`` becomes ``"cuda"`` when a GPU is visible, else ``"cpu"``. An
        explicit value is honored as-is (and a requested CUDA device falls back
        to CPU with a warning if no GPU is present, so a run never hard-fails on
        a misconfigured device).

        :return: A device string such as ``"cuda"`` or ``"cpu"``.
        """
        requested = (config.EMBED_DEVICE or "auto").lower()
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except Exception:  # noqa: BLE001
            has_cuda = False
        if requested == "auto":
            return "cuda" if has_cuda else "cpu"
        if requested.startswith("cuda") and not has_cuda:
            log.warning("RAG: EMBED_DEVICE=%s requested but no CUDA GPU is "
                        "available; falling back to CPU", requested)
            return "cpu"
        return requested

    def _load_model(self):
        """Load (and cache) the sentence-transformers embedding model on the
        configured device (GPU when available).

        :return: The loaded model.
        """
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            device = self._resolve_device()
            log.info("RAG: loading embed model %s on %s (first run downloads it)...",
                     config.EMBED_MODEL, device)
            self._model = SentenceTransformer(config.EMBED_MODEL, device=device)
            log.info("RAG: embed model ready on device=%s (batch_size=%d)",
                     device, config.EMBED_BATCH)
        return self._model

    def _embed(self, texts):
        """Embed a list of texts into normalized float vectors.

        Uses the configured batch size so GPU throughput stays high while VRAM
        remains bounded on a huge repo.

        :param texts: List of strings to embed.
        :return: A numpy array of unit-normalized embeddings.
        """
        model = self._load_model()
        return model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=config.EMBED_BATCH,
        )

    def _max_seq_tokens(self) -> int:
        """Return the embedding model's max input length in tokens (0 if unknown).

        all-MiniLM-L6-v2 defaults to 256; anything past that is silently
        truncated before embedding, so the tail of an over-long chunk is stored
        as payload text but never contributes to its vector.
        """
        model = self._load_model()
        try:
            return int(getattr(model, "max_seq_length", 0) or 0)
        except Exception:  # noqa: BLE001
            return 0

    def _warn_if_chunks_truncate(self, sample_texts) -> None:
        """Warn once if sampled chunks exceed the model's max sequence length.

        We tokenize a small sample rather than every chunk (tokenizing the whole
        kernel would be wasteful) and compare against :meth:`_max_seq_tokens`.
        If chunks routinely exceed it, their tails never influence retrieval and
        ``CHUNK_MAX_CHARS`` should be lowered.
        """
        limit = self._max_seq_tokens()
        if not limit or not sample_texts:
            return
        model = self._load_model()
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            return
        try:
            over = 0
            longest = 0
            for text in sample_texts:
                n = len(tokenizer.tokenize(text))
                longest = max(longest, n)
                if n > limit:
                    over += 1
        except Exception:  # noqa: BLE001
            return
        if over:
            log.warning(
                "RAG: %d/%d sampled chunks exceed the embed model's max "
                "sequence length (%d tokens; longest sampled=%d). Their tails "
                "are stored but NOT embedded, so retrieval ignores them. Lower "
                "CHUNK_MAX_CHARS (currently %d) so chunks fit within %d tokens.",
                over, len(sample_texts), limit, longest,
                config.CHUNK_MAX_CHARS, limit)

    def _iter_repo_chunks(self, repo):
        """Yield ``(path, text, start_line, end_line)`` for every chunk in ``repo``.

        Streams file by file so the whole corpus of chunk texts is never held in
        memory at once — only the current span's worth is retained by
        :meth:`build`. Important on a repo the size of the Linux kernel.
        """
        for path in walk_repo(repo):
            try:
                with open(path, encoding="utf-8", errors="ignore") as handle:
                    content = handle.read()
            except OSError:
                continue
            for chunk, start, end in iter_chunks_with_lines(content):
                yield path, chunk, start, end

    def build(self) -> None:
        """Build (or open from cache) the embedded Qdrant collection. Idempotent.

        A collection counts as a cache hit only when both the collection exists
        **and** its completion marker is present. A partial collection left by a
        crashed run (marker absent) is dropped and rebuilt, so the tool never
        searches a half-populated index.
        """
        if self._ready:
            return
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct

        repo = config.TARGET_REPO
        fingerprint = self._fingerprint(repo)
        self._collection = f"repo_{fingerprint}"

        os.makedirs(_CACHE_DIR, exist_ok=True)
        # Embedded, on-disk Qdrant: in-process, no server, no open port.
        self._client = QdrantClient(path=os.path.join(_CACHE_DIR, "store"))

        marker = self._marker_path()
        # Reuse an existing collection only if the completion marker confirms
        # the previous build finished every upsert (cache hit).
        if self._client.collection_exists(self._collection):
            if os.path.exists(marker):
                count = self._client.count(self._collection).count
                log.info("RAG: reusing complete Qdrant collection %s (%d points)",
                         self._collection, count)
                self._ready = True
                return
            # Unmarked collection: empty, or a crash left it partial. Drop it so
            # we rebuild cleanly rather than searching an incomplete index.
            log.warning("RAG: collection %s has no completion marker "
                        "(partial/aborted build); rebuilding", self._collection)
            self._client.delete_collection(self._collection)

        # A stale marker without its collection would be misleading; clear it.
        if os.path.exists(marker):
            os.remove(marker)

        # Determine the vector dimension from the model itself.
        dim = int(self._load_model().get_sentence_embedding_dimension())

        log.info("RAG: creating collection %s (dim=%d) and embedding chunks "
                 "from %s...", self._collection, dim, repo)
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        # Stream chunks and embed in spans: both vectors AND texts for only one
        # span are held at a time (peak memory bounded by _EMBED_SPAN), rather
        # than materializing every chunk of the repo up front.
        chunk_iter = self._iter_repo_chunks(repo)
        next_id = 0
        total = 0
        truncation_checked = False
        while True:
            span = []  # (path, text, start, end)
            for item in chunk_iter:
                span.append(item)
                if len(span) >= _EMBED_SPAN:
                    break
            if not span:
                break

            if not truncation_checked:
                self._warn_if_chunks_truncate([text for _p, text, _s, _e in span[:256]])
                truncation_checked = True

            vectors = self._embed([text for _p, text, _s, _e in span])
            for begin in range(0, len(span), _BATCH):
                end = min(begin + _BATCH, len(span))
                points = [
                    PointStruct(
                        id=next_id + offset,
                        vector=vectors[begin + offset].tolist(),
                        payload={
                            "path": span[begin + offset][0],
                            "text": span[begin + offset][1],
                            "start_line": span[begin + offset][2],
                            "end_line": span[begin + offset][3],
                        },
                    )
                    for offset in range(end - begin)
                ]
                self._client.upsert(collection_name=self._collection, points=points)
            next_id += len(span)
            total += len(span)
            log.info("RAG: embedded %d chunks", total)

        if total == 0:
            # No source chunks. Drop the empty collection; mark empty so search
            # returns a clear status rather than rebuilding every call.
            log.warning("RAG: no source chunks found under %s", repo)
            self._client.delete_collection(self._collection)
            self._empty = True
            self._ready = True
            return

        # Write the completion marker only now that every chunk is upserted.
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write(str(total))

        self._ready = True
        log.info("RAG: index ready (%d chunks) in collection %s",
                 total, self._collection)

    def search(self, query: str, *, k: int = _TOP_K) -> str:
        """Return the top-``k`` most relevant code/doc chunks for ``query``.

        :param query: The natural-language or code query.
        :param k: Number of chunks to return.
        :return: Formatted ``[path:start-end]`` + chunk blocks, or a status
            string. The line range tells the agent exactly where each chunk sits
            so it can open the surrounding source.
        """
        if not self._ready:
            self.build()
        if self._empty:
            return "empty index"
        query_vec = self._embed([query])[0].tolist()
        result = self._client.query_points(
            collection_name=self._collection,
            query=query_vec,
            limit=k,
            with_payload=True,
        )
        blocks = []
        for point in result.points:
            payload = point.payload or {}
            path = payload.get("path", "?")
            start = payload.get("start_line")
            end = payload.get("end_line")
            location = f"{path}:{start}-{end}" if start and end else path
            blocks.append(f"[{location}]\n{payload.get('text', '')}")
        return "\n---\n".join(blocks) if blocks else "no matches"

    def close(self) -> None:
        """Close the embedded Qdrant client, releasing its on-disk lock.

        Safe to call more than once. Called at interpreter exit to avoid a noisy
        destructor warning during shutdown.
        """
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
        self._ready = False


#: Shared index instance used by the tool and warmup.
rag_index = RagIndex()

# Close the embedded Qdrant store cleanly at exit (silences a harmless
# "Python is likely shutting down" destructor warning from qdrant-client).
import atexit  # noqa: E402

atexit.register(rag_index.close)


@tool
def rag_tool(query: str) -> str:
    """Semantic search over the repo (embedding RAG). Returns top relevant code/doc chunks."""
    return rag_index.search(query)
