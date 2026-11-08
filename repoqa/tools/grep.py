"""Online lexical search tool: GNU ``grep`` over the target repo (live scan).

This is the *online* grep condition: every query shells out to GNU ``grep -rn``
and walks the ``TARGET_REPO`` filesystem fresh — there is no index and no warmup.
Cost is paid per query at search time, proportional to repo size. Its
index-backed counterpart is :mod:`repoqa.tools.rg_index` (ripgrep over a
prebuilt file list).

Scope is restricted to in-scope C sources (``*.c``/``*.h`` via
:data:`config.SOURCE_GLOBS`) and the excluded build/VCS dirs are pruned, so both
lexical tools search exactly the same file set. A pure-Python regex scan is used
as a last-resort fallback only when GNU ``grep`` is not installed.
"""
import re
import subprocess

from langchain_core.tools import tool

from .. import config
from .common import EXCLUDE_DIRS, walk_repo
from ..ladder import active_scope, scope_file_list

#: Maximum characters of output returned to the model.
_MAX_OUTPUT_CHARS = 4000
#: Maximum matches collected by the Python fallback scan.
_MAX_FALLBACK_HITS = 100
#: File paths passed per grep invocation when a ladder scope restricts the set
#: (bounds argv length so a large rung is searched in chunks).
_FILES_PER_CALL = 4000


def _gnu_grep_scoped(pattern: str, files: list):
    """Run GNU ``grep`` over an explicit ``files`` list (ladder scope active).

    When a rung scope is active the search must be confined to exactly the
    rung's files — never the whole tree — so out-of-scope subsystems are
    invisible. Paths are passed explicitly (chunked to bound argv length) and
    grep performs no directory traversal.

    :param pattern: The regex/text pattern to search for.
    :param files: The in-scope file paths to search.
    :return: Matching lines (``file:line:text``), an error string, or ``None``
        if GNU ``grep`` is not installed.
    """
    collected = []
    error = None
    for start in range(0, len(files), _FILES_PER_CALL):
        chunk = files[start:start + _FILES_PER_CALL]
        command = ["grep", "-niHE", "--no-messages", "-e", pattern, "--"] + chunk
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=120,
                encoding="utf-8", errors="replace",
            )
        except FileNotFoundError:
            return None
        if completed.returncode not in (0, 1):
            error = completed.stderr[:500]
            continue
        if completed.stdout:
            collected.append(completed.stdout)
        if sum(len(c) for c in collected) >= _MAX_OUTPUT_CHARS:
            break
    if not collected:
        return f"grep error: {error}" if error else "no matches"
    return "".join(collected)[:_MAX_OUTPUT_CHARS]


def _gnu_grep(pattern: str, repo: str):
    """Run GNU ``grep`` recursively over ``repo`` for ``pattern`` (online scan).

    Builds ``grep -rni`` with ``--include`` globs mirroring
    :data:`config.SOURCE_GLOBS` (``*.c``/``*.h``) and ``--exclude-dir`` for each
    entry in :data:`EXCLUDE_DIRS`, so the search set matches the other lexical
    tool exactly. The filesystem is walked live on every call — this is the
    "online" (un-indexed) condition. Used only when NO ladder scope is active;
    a scoped run goes through :func:`_gnu_grep_scoped`.

    :param pattern: The regex/text pattern to search for.
    :param repo: Root directory to search.
    :return: Matching lines (``file:line:text``), an error string, or ``None``
        if GNU ``grep`` is not installed.
    """
    command = ["grep", "-rniE", "--no-messages"]
    # Restrict to in-scope source files (``*.c``/``*.h`` for the kernel).
    for glob in config.SOURCE_GLOBS:
        command += [f"--include={glob}"]
    for excluded in EXCLUDE_DIRS:
        command += [f"--exclude-dir={excluded}"]
    command += ["-e", pattern, repo]
    try:
        # grep emits bytes in the file's encoding; force UTF-8 decoding with
        # replacement so non-ASCII source does not crash or get mangled.
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        return None
    if completed.returncode in (0, 1):  # 1 == "no matches", not an error
        return (completed.stdout or "no matches")[:_MAX_OUTPUT_CHARS]
    return f"grep error: {completed.stderr[:500]}"


def _python_scan(pattern: str, repo: str) -> str:
    """Fallback regex scan used when GNU ``grep`` is unavailable.

    Mirrors grep's case-insensitive semantics; an invalid regex is treated as a
    literal string. Honors an active ladder scope: when one is set, only the
    rung's files are scanned (never the whole tree).

    :param pattern: The regex/text pattern to search for.
    :param repo: Root directory to search.
    :return: Matching lines (``file:line:text``) or ``"no matches"``.
    """
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
    scoped = scope_file_list()
    paths = scoped if scoped else walk_repo(repo)
    hits = []
    for path in paths:
        try:
            with open(path, encoding="utf-8", errors="ignore") as handle:
                for line_no, line in enumerate(handle, 1):
                    if regex.search(line):
                        hits.append(f"{path}:{line_no}:{line.rstrip()}")
                        if len(hits) >= _MAX_FALLBACK_HITS:
                            return "\n".join(hits)[:_MAX_OUTPUT_CHARS]
        except OSError:
            continue
    return "\n".join(hits)[:_MAX_OUTPUT_CHARS] if hits else "no matches"


@tool
def grep_tool(pattern: str) -> str:
    """Search the repo's C sources for a regex/text PATTERN using GNU grep (online,
    un-indexed live scan). Returns matching lines with file:line."""
    repo = config.TARGET_REPO
    # When a ladder rung scope is active, confine the search to exactly the
    # rung's files so out-of-scope subsystems are invisible.
    if active_scope() is not None:
        files = scope_file_list()
        result = _gnu_grep_scoped(pattern, files)
        if result is not None:
            return result
        return _python_scan(pattern, repo)
    result = _gnu_grep(pattern, repo)
    if result is not None:
        return result
    return _python_scan(pattern, repo)
