"""Indexed lexical search tool: ripgrep over a prebuilt file index.

This is the *indexed* grep condition, the counterpart to the online GNU-grep
tool in :mod:`repoqa.tools.grep`. ripgrep has no native persistent index, so the
"index" here is the thing ripgrep can actually be driven by: a list of every
in-scope C source file (``*.c``/``*.h``) under ``TARGET_REPO``, collected once at
**warmup** and cached on disk. Each query then runs ``rg`` against that cached
file list instead of re-walking the directory tree, so the per-query cost drops
the filesystem-traversal component (which dominates on a tree the size of the
Linux kernel) and pays only the match scan.

The index is keyed on repo contents (path/size/mtime of every source file) plus
the source-extension scope, so adding/removing/changing files transparently
rebuilds it rather than searching a stale list. The on-disk list is written only
after the full walk completes, so a crashed build is rebuilt next run rather
than used half-populated.

Concurrency: the index file is per-fingerprint, so independent targets/scopes do
not collide. Compared with the online grep tool, the ONLY difference is index
vs. live traversal — the pattern semantics, scope and output format are kept
identical so accuracy stays attributable to the indexing, not to search
behavior.
"""
import os
import logging
import subprocess

from langchain_core.tools import tool

from .. import config
from .common import walk_repo
from ..ladder import active_scope, active_label, scope_file_list

log = logging.getLogger("repoqa.tools.rg_index")

#: Maximum characters of output returned to the model.
_MAX_OUTPUT_CHARS = 4000
#: Number of file paths passed per ``rg`` invocation. Bounds the argv length so
#: an arbitrarily large (kernel-sized) index is searched in chunks without ever
#: exceeding the OS command-line limit.
_FILES_PER_CALL = 4000
#: Directory holding the cached file-list index (persists across runs).
_CACHE_DIR = ".rg_index_cache"


class RgIndex:
    """Lazy, on-disk file-list index that ripgrep searches instead of the tree.

    Walks the repo once (at :meth:`build`, invoked by warmup) to collect every
    in-scope source file, writes the list to a fingerprinted cache file, and
    serves each query by pointing ``rg`` at that list. The fingerprint folds in
    repo contents and the source-extension scope so a change rebuilds rather
    than reusing a stale list.
    """

    def __init__(self):
        self._ready = False
        self._list_path = None
        self._count = 0

    def reset(self) -> None:
        """Force the next :meth:`build` to re-resolve (e.g. after a scope change).

        The index is keyed on the active scope's fingerprint, so re-building
        picks up a different rung's file set rather than reusing the previous
        rung's cached list within the same process.
        """
        self._ready = False
        self._list_path = None
        self._count = 0

    def _fingerprint(self, repo: str) -> str:
        """Compute a cache key from the source scope and repo contents.

        :param repo: Root directory of the target repo.
        :return: A short hex digest covering the extension scope plus each
            source file's (path, size, mtime).
        """
        import hashlib

        digest = hashlib.sha256()
        digest.update(",".join(config.SOURCE_EXTENSIONS).encode())
        # Fold the active ladder scope into the key so each rung gets its own
        # index rather than colliding with a whole-repo one.
        label = active_label()
        if label:
            digest.update(f"scope={label}".encode())
        source_paths = scope_file_list() if active_scope() is not None else sorted(walk_repo(repo))
        for path in source_paths:
            try:
                stat = os.stat(path)
            except OSError:
                continue
            digest.update(path.encode("utf-8", "ignore"))
            digest.update(str(stat.st_size).encode())
            digest.update(str(int(stat.st_mtime)).encode())
        return digest.hexdigest()[:16]

    def build(self) -> None:
        """Build (or reuse) the on-disk file-list index. Idempotent.

        Collects every in-scope source file under :data:`config.TARGET_REPO` and
        writes a NUL-separated list to a fingerprinted cache file. A list is
        reused only when its fingerprinted file already exists (a completed
        build); a partial write never happens because the list is written to a
        temp file and atomically renamed into place.
        """
        if self._ready:
            return
        repo = config.TARGET_REPO
        fingerprint = self._fingerprint(repo)
        os.makedirs(_CACHE_DIR, exist_ok=True)
        self._list_path = os.path.join(_CACHE_DIR, f"files_{fingerprint}.lst")

        if os.path.exists(self._list_path):
            with open(self._list_path, "rb") as handle:
                self._count = handle.read().count(b"\x00")
            log.info("rg_index: reusing file index %s (%d files)",
                     self._list_path, self._count)
            self._ready = True
            return

        log.info("rg_index: building file index for %s ...", repo)
        tmp_path = self._list_path + ".tmp"
        count = 0
        # When a ladder scope is active, index EXACTLY the rung's files; else
        # walk the whole repo. NUL-separate paths so filenames with
        # spaces/newlines survive.
        source_paths = scope_file_list() if active_scope() is not None else walk_repo(repo)
        with open(tmp_path, "wb") as handle:
            for path in source_paths:
                handle.write(os.fsencode(path))
                handle.write(b"\x00")
                count += 1
        os.replace(tmp_path, self._list_path)  # atomic: no partial index served
        self._count = count
        self._ready = True
        log.info("rg_index: index ready (%d files) at %s", count, self._list_path)

    def _read_file_list(self) -> list:
        """Return the cached index as a list of file path strings.

        :return: The in-scope source file paths recorded at build time.
        """
        with open(self._list_path, "rb") as handle:
            raw = handle.read()
        return [os.fsdecode(p) for p in raw.split(b"\x00") if p]

    def search(self, pattern: str) -> str:
        """Search the prebuilt file index for ``pattern`` with ripgrep.

        Runs ``rg`` against the cached file list (paths passed explicitly as
        arguments) rather than traversing the directory tree. The list is sent
        in chunks so an arbitrarily large kernel-sized index never exceeds the
        OS command-line length limit; results are concatenated in order and
        capped. ``rg`` with an explicit file list performs no directory walk,
        which is the point of the index.

        :param pattern: The regex/text pattern to search for.
        :return: Matching lines (``file:line:text``), a status, or an error.
        """
        if not self._ready:
            self.build()
        if not self._count:
            return "no indexed files (empty index)"

        files = self._read_file_list()
        base = ["rg", "-n", "-i", "--no-heading", "--no-messages", "-e", pattern, "--"]
        collected = []
        truncated = False
        error = None
        # Chunk the file list to stay well under the OS argv limit (~128K bytes
        # on Linux). 4000 kernel paths per call is comfortably within that.
        for start in range(0, len(files), _FILES_PER_CALL):
            chunk = files[start:start + _FILES_PER_CALL]
            try:
                completed = subprocess.run(
                    base + chunk, capture_output=True, timeout=120,
                )
            except FileNotFoundError:
                return "rg_index error: ripgrep (rg) not installed"
            if completed.returncode not in (0, 1):  # 1 == no matches (ok)
                error = completed.stderr.decode("utf-8", "replace")[:500]
                continue
            out = completed.stdout.decode("utf-8", "replace")
            if out:
                collected.append(out)
            joined_len = sum(len(c) for c in collected)
            if joined_len >= _MAX_OUTPUT_CHARS:
                truncated = True
                break

        if not collected:
            return f"rg_index error: {error}" if error else "no matches"
        result = "".join(collected)[:_MAX_OUTPUT_CHARS]
        if truncated and len(result) >= _MAX_OUTPUT_CHARS:
            result = result[:_MAX_OUTPUT_CHARS]
        return result


#: Shared index instance used by the tool and warmup.
rg_index = RgIndex()


@tool
def rg_index_tool(pattern: str) -> str:
    """Search the repo's C sources for a regex/text PATTERN using ripgrep over a
    prebuilt file index (built once at warmup). Returns matching lines with
    file:line. Same scope/semantics as the online grep tool, but queries hit the
    cached file list instead of re-walking the directory tree."""
    return rg_index.search(pattern)
