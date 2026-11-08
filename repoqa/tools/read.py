"""Shared file-reading tool offered alongside each search tool.

The three search tools (grep/LSP/RAG) locate evidence but each exposes only its
own slice of a file: grep returns ``file:line`` hits, LSP a symbol definition,
RAG embedding chunks. This tool gives all of them the same, identical follow-up
capability: open a file at a chosen line range and read it.

Keeping the read capability *shared and identical* across those tools means the
only variable between the grep/LSP/RAG conditions stays the search tool itself,
so accuracy differences remain attributable to it. ``shell_tool`` deliberately
does NOT get this tool — it can already ``cat``/``head``/``tail`` natively, and
bolting a second reader onto it would double-count the same capability.

Like the shell sandbox, every path is confined to :data:`config.TARGET_REPO`
(absolute paths outside it and ``..`` traversal are rejected) and the output is
bounded: at most :data:`_MAX_LINES` lines per call.
"""
import os

from langchain_core.tools import tool

from .. import config
from ..ladder import in_scope, active_scope

#: Maximum number of lines returned by a single read (per teammate request).
_MAX_LINES = 50
#: Maximum characters of output returned to the model (a safety cap on top of
#: the line cap, for pathologically long lines).
_MAX_OUTPUT_CHARS = 4000


def _confined(path: str, repo: str) -> bool:
    """Return whether ``path`` resolves inside ``repo``.

    Expands ``~``, joins relatives against the repo, resolves ``..`` and
    symlinks, then checks containment. Mirrors the shell tool's sandbox so the
    two tools confine paths identically.

    :param path: The path to check (absolute or relative to the repo).
    :param repo: Absolute path to the target repo root.
    :return: ``True`` if the resolved path is the repo or lives under it.
    """
    expanded = os.path.expanduser(path)
    candidate = expanded if os.path.isabs(expanded) else os.path.join(repo, expanded)
    resolved = os.path.realpath(candidate)
    repo_real = os.path.realpath(repo)
    return resolved == repo_real or resolved.startswith(repo_real + os.sep)


def read_path(path: str, start_line: int = 1, end_line: int = None) -> str:
    """Read a confined slice of a file under the target repo.

    :param path: File path, absolute or relative to :data:`config.TARGET_REPO`.
    :param start_line: 1-based first line to return (clamped to ``>= 1``).
    :param end_line: 1-based last line to return (inclusive). When omitted, reads
        up to :data:`_MAX_LINES` lines from ``start_line``.
    :return: The requested lines prefixed with ``file:line``, or an error string.
    """
    if not path or not str(path).strip():
        return "read error: empty path"
    repo = os.path.abspath(config.TARGET_REPO)
    if not _confined(path, repo):
        return f"read error: path {path!r} is outside the target repo"

    expanded = os.path.expanduser(path)
    abs_path = expanded if os.path.isabs(expanded) else os.path.join(repo, expanded)
    # Ladder containment (decision ii): a file outside the active rung scope is
    # treated exactly as if it did not exist, so a reduced rung is
    # indistinguishable from a kernel that only contained the scoped subsystems.
    # No message reveals that a larger tree is present.
    if active_scope() is not None and not in_scope(abs_path):
        return f"read error: no such file {path!r}"
    if not os.path.isfile(abs_path):
        return f"read error: no such file {path!r}"

    try:
        start = max(1, int(start_line))
    except (TypeError, ValueError):
        start = 1
    try:
        end = int(end_line) if end_line is not None else None
    except (TypeError, ValueError):
        end = None

    # Bound the window to at most _MAX_LINES lines regardless of the requested
    # end, so a read stays small and identical in cost across tools.
    if end is None or end < start:
        end = start + _MAX_LINES - 1
    else:
        end = min(end, start + _MAX_LINES - 1)

    try:
        with open(abs_path, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError as exc:
        return f"read error: {exc}"

    total = len(lines)
    if start > total:
        return f"read error: start_line {start} past end of file ({total} lines)"
    end = min(end, total)

    selected = lines[start - 1:end]
    rel = os.path.relpath(abs_path, repo)
    numbered = [f"{rel}:{start + i}:{line.rstrip(chr(10)).rstrip(chr(13))}"
                for i, line in enumerate(selected)]
    body = "\n".join(numbered)
    if len(body) > _MAX_OUTPUT_CHARS:
        body = body[:_MAX_OUTPUT_CHARS] + "\n... (truncated)"
    return body or "(empty range)"


@tool
def read_file(path: str, start_line: int = 1, end_line: int = None) -> str:
    """Read a slice of a file in the repo. PATH is relative to the repo root
    (or absolute inside it). Returns lines from START_LINE to END_LINE (1-based,
    inclusive) prefixed with file:line. At most 50 lines are returned per call;
    omit END_LINE to read 50 lines from START_LINE. Paths outside the repo and
    '..' traversal are rejected. Use this to read the files search results point to."""
    return read_path(path, start_line, end_line)
