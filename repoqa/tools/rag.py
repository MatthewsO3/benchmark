"""Semantic search tool: embedding RAG over the repo (Qdrant + sentence-transformers).

Vectors are stored in a **local, on-disk Qdrant** collection (``QdrantClient(path=...)``)
rather than an in-memory FAISS index. Qdrant runs embedded in-process — it opens
no network port and needs no server or root, which suits a shared machine — while
scaling to the millions of chunks a large C codebase like the Linux kernel
produces, far past what a brute-force flat index handles comfortably.

The embedding model is :data:`config.EMBED_MODEL` (swap via the ``EMBED_MODEL``
env var — never by editing the installed package). The collection is keyed on a
fingerprint of (embed model, vector dimension, repo contents), so changing the
model or the repo transparently rebuilds at the correct vector size instead of
colliding with a stale collection.
"""
import os
import logging

from langchain_core.tools import tool

from .. import config
from .common import walk_repo, chunk_by_lines

log = logging.getLogger("repoqa.tools.rag")

#: Directory holding the embedded Qdrant storage (persists across runs).
_CACHE_DIR = ".qdrant_cache"
#: Number of chunks returned per query.
_TOP_K = 3
#: Number of points per Qdrant upsert request (bounds a single payload size).
_BATCH = 512
#: Number of texts handed to one ``encode`` call. Larger spans let the GPU-backed
#: encoder keep its batches full (it sub-batches internally by EMBED_BATCH);
#: also bounds peak host memory for the vectors of one span.
_EMBED_SPAN = 8192


class RagIndex:
    """Lazy embedded-Qdrant index over the repo using a sentence-transformers model.

    The collection is persisted on disk and keyed on repo contents + embedding
    model, so repeated runs skip re-embedding.
    """

    def __init__(self):
        self._ready = False
        self._model = None
        self._client = None
        self._collection = None
        self._empty = False  # repo produced no chunks

    def _fingerprint(self, repo: str) -> str:
        """Compute a cache key from the embed model and all source files.

        :param repo: Root directory of the target repo.
        :return: A short hex digest over (model, path, size, mtime) of each file.
        """
        import hashlib

        digest = hashlib.sha256()
        digest.update(config.EMBED_MODEL.encode())
        for path in sorted(walk_repo(repo)):
            try:
                stat = os.stat(path)
            except OSError:
                continue
            digest.update(path.encode("utf-8", "ignore"))
            digest.update(str(stat.st_size).encode())
            digest.update(str(int(stat.st_mtime)).encode())
        return digest.hexdigest()[:16]

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

    def build(self) -> None:
        """Build (or open from cache) the embedded Qdrant collection. Idempotent."""
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

        # Reuse an existing collection for this fingerprint (cache hit).
        if self._client.collection_exists(self._collection):
            count = self._client.count(self._collection).count
            if count > 0:
                log.info("RAG: reusing Qdrant collection %s (%d points)",
                         self._collection, count)
                self._ready = True
                return
            # Empty/partial collection: drop and rebuild cleanly.
            self._client.delete_collection(self._collection)

        # Determine the vector dimension from the model itself.
        dim = int(self._load_model().get_sentence_embedding_dimension())

        log.info("RAG: scanning repo %s for chunks...", repo)
        paths, texts = [], []
        for path in walk_repo(repo):
            try:
                with open(path, encoding="utf-8", errors="ignore") as handle:
                    content = handle.read()
            except OSError:
                continue
            for chunk in chunk_by_lines(content):
                paths.append(path)
                texts.append(chunk)

        if not texts:
            log.warning("RAG: no source chunks found under %s", repo)
            self._empty = True
            self._ready = True
            return

        log.info("RAG: creating collection %s (dim=%d) and embedding %d chunks...",
                 self._collection, dim, len(texts))
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

        # Embed in large spans (so the GPU-backed encoder can batch efficiently
        # via EMBED_BATCH) and upsert in smaller sub-batches (so a single upsert
        # payload stays bounded). Peak memory is bounded by the span size.
        total_n = len(texts)
        for span_begin in range(0, total_n, _EMBED_SPAN):
            span_end = min(span_begin + _EMBED_SPAN, total_n)
            vectors = self._embed(texts[span_begin:span_end])
            for begin in range(span_begin, span_end, _BATCH):
                end = min(begin + _BATCH, span_end)
                points = [
                    PointStruct(
                        id=idx,
                        vector=vectors[idx - span_begin].tolist(),
                        payload={"path": paths[idx], "text": texts[idx]},
                    )
                    for idx in range(begin, end)
                ]
                self._client.upsert(collection_name=self._collection, points=points)
            log.info("RAG: embedded %d/%d chunks", span_end, total_n)

        self._ready = True
        log.info("RAG: index ready (%d chunks) in collection %s",
                 len(texts), self._collection)

    def search(self, query: str, *, k: int = _TOP_K) -> str:
        """Return the top-``k`` most relevant code/doc chunks for ``query``.

        :param query: The natural-language or code query.
        :param k: Number of chunks to return.
        :return: Formatted ``[path]`` + chunk blocks, or a status string.
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
            blocks.append(f"[{payload.get('path', '?')}]\n{payload.get('text', '')}")
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
