"""Lexical search tool: ripgrep over the target repo (regex fallback)."""
import re
import subprocess

from langchain_core.tools import tool

from .. import config
from .common import EXCLUDE_DIRS, walk_repo

#: Maximum characters of output returned to the model.
_MAX_OUTPUT_CHARS = 4000
#: Maximum matches collected by the Python fallback scan.
_MAX_FALLBACK_HITS = 100


def _ripgrep(pattern: str, repo: str):
    """Run ripgrep over ``repo`` for ``pattern``.

    :param pattern: The regex/text pattern to search for.
    :param repo: Root directory to search.
    :return: Matching lines (``file:line:text``), an error string, or ``None``
        if ripgrep is not installed.
    """
    command = ["rg", "-n", "-i", "--no-heading", "-g", "*.py"]
    for excluded in EXCLUDE_DIRS:
        command += ["-g", f"!**/{excluded}/**"]
    command += [pattern, repo]
    try:
        # ripgrep emits UTF-8; force UTF-8 decoding instead of the Windows
        # locale (cp1252), otherwise non-ASCII text (e.g. Hungarian) is mangled.
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=30,
            encoding="utf-8", errors="replace",
        )
    except FileNotFoundError:
        return None
    if completed.returncode in (0, 1):  # 1 == "no matches", not an error
        return (completed.stdout or "no matches")[:_MAX_OUTPUT_CHARS]
    return f"grep error: {completed.stderr[:500]}"


def _python_scan(pattern: str, repo: str) -> str:
    """Fallback regex scan used when ripgrep is unavailable.

    Mirrors ripgrep's case-insensitive semantics; an invalid regex is treated
    as a literal string.

    :param pattern: The regex/text pattern to search for.
    :param repo: Root directory to search.
    :return: Matching lines (``file:line:text``) or ``"no matches"``.
    """
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error:
        regex = re.compile(re.escape(pattern), re.IGNORECASE)
    hits = []
    for path in walk_repo(repo):
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
    """Search the target repo for a regex/text PATTERN. Returns matching lines with file:line."""
    repo = config.TARGET_REPO
    result = _ripgrep(pattern, repo)
    if result is not None:
        return result
    return _python_scan(pattern, repo)
