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


#: Default chunking parameters, shared by :func:`chunk_by_lines` and
#: :func:`iter_chunks_with_lines`. Sourced from :mod:`config` so there is one
#: overridable definition. The RAG cache fingerprint folds these in, so
#: changing them (via ``CHUNK_MAX_CHARS`` / ``CHUNK_OVERLAP_LINES`` in the env)
#: invalidates a stale index instead of silently reusing chunks cut at a
#: different size.
CHUNK_MAX_CHARS = config.CHUNK_MAX_CHARS
CHUNK_OVERLAP_LINES = config.CHUNK_OVERLAP_LINES


def iter_chunks_with_lines(text, *, max_chars=CHUNK_MAX_CHARS,
                           overlap_lines=CHUNK_OVERLAP_LINES):
    """Split ``text`` into line-boundary chunks, yielding each with its line span.

    Same accumulation rule as :func:`chunk_by_lines` (whole lines up to roughly
    ``max_chars``, a single over-long line standing alone, ``overlap_lines``
    trailing lines carried into the next chunk), but implemented as a generator
    that also reports where each chunk sits in the file. This lets a caller
    stream chunks without materializing them all, and show line numbers in
    output.

    :param text: The text to split.
    :param max_chars: Approximate maximum characters per chunk.
    :param overlap_lines: Number of trailing lines repeated in the next chunk.
    :yield: ``(chunk_text, start_line, end_line)`` with 1-based inclusive lines.
    """
    lines = text.splitlines(keepends=True)
    current = []          # (line_number, line_text) pending in this chunk
    current_len = 0
    for lineno, line in enumerate(lines, 1):
        if current and current_len + len(line) > max_chars:
            chunk = "".join(part for _, part in current)
            yield chunk, current[0][0], current[-1][0]
            current = current[-overlap_lines:] if overlap_lines else []
            current_len = sum(len(part) for _, part in current)
        current.append((lineno, line))
        current_len += len(line)
    if current:
        chunk = "".join(part for _, part in current)
        if chunk.strip():
            yield chunk, current[0][0], current[-1][0]


def chunk_by_lines(text: str, *, max_chars: int = CHUNK_MAX_CHARS,
                   overlap_lines: int = CHUNK_OVERLAP_LINES) -> list:
    """Split ``text`` into chunks that respect line boundaries.

    Whole lines are accumulated up to roughly ``max_chars`` per chunk (a single
    over-long line becomes its own chunk). The last ``overlap_lines`` lines are
    carried into the next chunk so context spanning a boundary is not lost.

    Thin wrapper over :func:`iter_chunks_with_lines` that drops the line spans.

    :param text: The text to split.
    :param max_chars: Approximate maximum characters per chunk.
    :param overlap_lines: Number of trailing lines repeated in the next chunk.
    :return: List of chunk strings.
    """
    return [chunk for chunk, _start, _end in
            iter_chunks_with_lines(text, max_chars=max_chars,
                                   overlap_lines=overlap_lines)]
