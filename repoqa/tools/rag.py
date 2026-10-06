"""Semantic search tool: embedding RAG over the repo (FAISS + MiniLM)."""
import os
import logging

from langchain_core.tools import tool

from .. import config
from .common import walk_repo, chunk_by_lines

log = logging.getLogger("repoqa.tools.rag")

#: Directory where the FAISS index and chunk list are cached between runs.
_CACHE_DIR = ".rag_cache"
#: Number of chunks returned per query.
_TOP_K = 3


class RagIndex:
    """Lazy in-memory RAG index over the repo using a small embedding model.

    The index is cached to disk keyed on repo contents, so repeated runs skip
    the embedding step.
    """

    def __init__(self):
        self._ready = False
        self._chunks = []  # list of (path, chunk_text)
        self._model = None
        self._index = None

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

    def _cache_paths(self, fingerprint: str):
        """Return the ``(index_path, chunks_path)`` for a fingerprint.

        :param fingerprint: The repo fingerprint from :meth:`_fingerprint`.
        :return: Tuple of the FAISS index path and the pickled-chunks path.
        """
        os.makedirs(_CACHE_DIR, exist_ok=True)
        base = os.path.join(_CACHE_DIR, fingerprint)
        return base + ".faiss", base + ".pkl"

    def _load_cache(self, fingerprint: str) -> bool:
        """Load a previously built index from disk if present.

        :param fingerprint: The repo fingerprint identifying the cache entry.
        :return: ``True`` if the cache was loaded, ``False`` otherwise.
        """
        import pickle

        import faiss

        index_path, chunks_path = self._cache_paths(fingerprint)
        if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
            return False
        try:
            self._index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as handle:
                self._chunks = pickle.load(handle)
            log.info("RAG: loaded index from cache (%d chunks)", len(self._chunks))
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("RAG: cache load failed (%s), rebuilding", exc)
            return False

    def _save_cache(self, fingerprint: str) -> None:
        """Persist the current index and chunks to disk.

        :param fingerprint: The repo fingerprint identifying the cache entry.
        """
        import pickle

        import faiss

        index_path, chunks_path = self._cache_paths(fingerprint)
        try:
            faiss.write_index(self._index, index_path)
            with open(chunks_path, "wb") as handle:
                pickle.dump(self._chunks, handle)
            log.info("RAG: index cached to %s", index_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("RAG: cache save failed (%s)", exc)

    def build(self) -> None:
        """Build (or load from cache) the embedding index. Idempotent."""
        from sentence_transformers import SentenceTransformer
        import faiss

        repo = config.TARGET_REPO
        fingerprint = self._fingerprint(repo)

        log.info("RAG: loading embed model %s (first run downloads it)...", config.EMBED_MODEL)
        self._model = SentenceTransformer(config.EMBED_MODEL)

        # Try the disk cache first (persists across runs; keyed on repo contents).
        if self._load_cache(fingerprint):
            self._ready = True
            return

        log.info("RAG: scanning repo %s for chunks...", repo)
        for path in walk_repo(repo):
            try:
                with open(path, encoding="utf-8", errors="ignore") as handle:
                    text = handle.read()
            except OSError:
                continue
            for chunk in chunk_by_lines(text):
                self._chunks.append((path, chunk))
        if not self._chunks:
            self._ready = True
            return

        log.info("RAG: embedding %d chunks...", len(self._chunks))
        embeddings = self._model.encode([chunk for _, chunk in self._chunks], convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(embeddings)
        self._ready = True
        log.info("RAG: index ready (%d chunks)", len(self._chunks))
        self._save_cache(fingerprint)

    def search(self, query: str, *, k: int = _TOP_K) -> str:
        """Return the top-``k`` most relevant code/doc chunks for ``query``.

        :param query: The natural-language or code query.
        :param k: Number of chunks to return.
        :return: Formatted ``[path]`` + chunk blocks, or a status string.
        """
        import faiss

        if not self._ready:
            self.build()
        if not self._chunks:
            return "empty index"
        query_vec = self._model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        _, indices = self._index.search(query_vec, k)
        blocks = []
        for i in indices[0]:
            if 0 <= i < len(self._chunks):
                path, text = self._chunks[i]
                blocks.append(f"[{path}]\n{text}")
        return "\n---\n".join(blocks)


#: Shared index instance used by the tool and warmup.
rag_index = RagIndex()


@tool
def rag_tool(query: str) -> str:
    """Semantic search over the repo (embedding RAG). Returns top relevant code/doc chunks."""
    return rag_index.search(query)
