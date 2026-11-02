"""Shared helpers for the repo-scanning tools (grep, LSP, RAG)."""
import os
from typing import Iterator
from urllib.parse import urlparse, unquote
from urllib.request import url2pathname

from .. import config

#: Directories skipped when scanning the target repo (VCS metadata, caches,
#: vendored code, build artefacts) so results stay focused on first-party
#: source. Covers both Python/JS projects and large C repos like the Linux
#: kernel (whose build output and git metadata we never want to index).
EXCLUDE_DIRS = {
    # VCS / editors / generic caches
    ".git", ".svn", ".hg", ".idea", ".vscode", ".cache", "data",
    # python / js tooling
    ".venv", "venv", "env", "__pycache__", "node_modules",
    ".mypy_cache", ".ruff_cache", ".pytest_cache", "dist", "build",
    "site-packages",
    # kernel / C build output (harmless if absent)
    ".tmp_versions", "include/generated", "include/config",
}


def _matches_source(name: str) -> bool:
    """Return whether a filename is an in-scope source file.

    :param name: The bare filename.
    :return: ``True`` if it ends with one of :data:`config.SOURCE_EXTENSIONS`.
    """
    return name.endswith(config.SOURCE_EXTENSIONS)


def walk_repo(repo: str) -> Iterator[str]:
    """Yield source file paths under ``repo``, skipping :data:`EXCLUDE_DIRS`.

    "Source file" is defined by :data:`config.SOURCE_EXTENSIONS` (``.c``/``.h``
    for the kernel by default, ``.py`` for a Python target), so this walk is
    language-agnostic and controlled by configuration.

    :param repo: Root directory to walk.
    :return: Iterator over in-scope source file paths.
    """
    for root, dirs, files in os.walk(repo):
        # Prune excluded directories in place so os.walk won't descend into them.
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for name in files:
            if _matches_source(name):
                yield os.path.join(root, name)


def uri_to_path(uri: str) -> str:
    """Convert a ``file://`` URI to a local filesystem path, cross-platform.

    Handles Windows (``file:///C:/...``) and POSIX (``file:///home/...``)
    correctly, including percent-encoded characters. Non-file inputs are
    returned unchanged.

    :param uri: The URI (or raw path) to convert.
    :return: A local filesystem path.
    """
    if not uri or not uri.startswith("file:"):
        return uri
    parsed = urlparse(uri)
    return url2pathname(unquote(parsed.path))


def chunk_by_lines(text: str, *, max_chars: int = 800, overlap_lines: int = 3) -> list:
    """Split ``text`` into chunks that respect line boundaries.

    Whole lines are accumulated up to roughly ``max_chars`` per chunk (a single
    over-long line becomes its own chunk). The last ``overlap_lines`` lines are
    carried into the next chunk so context spanning a boundary is not lost.

    :param text: The text to split.
    :param max_chars: Approximate maximum characters per chunk.
    :param overlap_lines: Number of trailing lines repeated in the next chunk.
    :return: List of chunk strings.
    """
    lines = text.splitlines(keepends=True)
    chunks = []
    current, current_len = [], 0
    for line in lines:
        if current and current_len + len(line) > max_chars:
            chunks.append("".join(current))
            current = current[-overlap_lines:] if overlap_lines else []
            current_len = sum(len(part) for part in current)
        current.append(line)
        current_len += len(line)
    if current and "".join(current).strip():
        chunks.append("".join(current))
    return chunks
