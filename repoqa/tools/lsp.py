"""Semantic search tool: symbol lookup via a real language server (jedi/LSP)."""
import os
import logging

from langchain_core.tools import tool

from .. import config
from .common import uri_to_path

log = logging.getLogger("repoqa.tools.lsp")

#: Maximum number of symbol hits included in the output.
_MAX_HITS = 10
#: Maximum lines read for a single definition snippet.
_MAX_SNIPPET_LINES = 80


def _leading_spaces(line: str) -> int:
    """Return the indentation width (leading-whitespace length) of ``line``.

    :param line: The source line.
    :return: Number of leading whitespace characters.
    """
    return len(line) - len(line.lstrip())


def _bracket_depth(depth: int, line: str) -> int:
    """Update a running bracket-nesting depth with the brackets on ``line``.

    :param depth: The depth before this line.
    :param line: The source line to scan.
    :return: The depth after accounting for ``()[]{}`` on the line.
    """
    for char in line:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
    return depth


def _definition_snippet(abs_path: str, line: int, *, max_lines: int = _MAX_SNIPPET_LINES) -> str:
    """Return the full definition starting at ``line`` (0-indexed).

    Includes any decorators above the definition, the (possibly multi-line)
    signature, and the whole body up to the dedent boundary. For non-``def``
    symbols (constants/assignments) the full statement is returned, following
    bracket and backslash continuations. Capped at ``max_lines``.

    Returning the full body (not just a few lines around the signature) is what
    lets the answerer confirm behaviour questions like "does X do Y?".

    :param abs_path: Absolute path to the source file.
    :param line: Zero-based line index where the definition starts.
    :param max_lines: Maximum number of lines to return.
    :return: The definition's source text, or ``""`` on error.
    """
    try:
        with open(abs_path, encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
    except OSError:
        return ""
    total = len(lines)
    if not (0 <= line < total):
        return ""

    # Include decorator lines directly above the definition.
    start = line
    above = line - 1
    while above >= 0 and lines[above].lstrip().startswith("@"):
        start = above
        above -= 1

    head = lines[line].lstrip()
    is_block = head.startswith(("def ", "async def ", "class "))

    if not is_block:
        # Statement/assignment: extend over bracket or backslash continuations.
        end = line + 1
        depth = _bracket_depth(0, lines[line])
        while (end < total and (end - start) < max_lines
               and (depth > 0 or lines[end - 1].rstrip().endswith("\\"))):
            depth = _bracket_depth(depth, lines[end])
            end += 1
        return "".join(lines[start:end]).rstrip()

    base_indent = _leading_spaces(lines[line])
    # Phase 1: consume the signature up to the block-opening ':' at depth 0.
    depth = _bracket_depth(0, lines[line])
    signature_done = depth <= 0 and lines[line].rstrip().endswith(":")
    end = line + 1
    while not signature_done and end < total and (end - start) < max_lines:
        depth = _bracket_depth(depth, lines[end])
        if depth <= 0 and lines[end].rstrip().endswith(":"):
            signature_done = True
        end += 1
    # Phase 2: consume the body until a non-blank line dedents to <= base indent.
    while end < total and (end - start) < max_lines:
        current = lines[end]
        if current.strip() and _leading_spaces(current) <= base_indent:
            break
        end += 1
    return "".join(lines[start:end]).rstrip()


class LspServer:
    """Lazy singleton wrapping a multilspy language server over the target repo."""

    def __init__(self):
        self._server = None
        self._context = None

    def ensure_started(self) -> None:
        """Start the jedi language server if it is not already running.

        The heavy ``multilspy`` import and server start-up happen here (lazily)
        so a grep-only run never pays for them.
        """
        if self._server is not None:
            return
        from multilspy import SyncLanguageServer
        from multilspy.multilspy_config import MultilspyConfig
        from multilspy.multilspy_logger import MultilspyLogger

        repo = os.path.abspath(config.TARGET_REPO)
        log.info("LSP: starting jedi language server for %s ...", repo)
        server_config = MultilspyConfig.from_dict({"code_language": "python"})
        self._server = SyncLanguageServer.create(server_config, MultilspyLogger(), repo)
        self._context = self._server.start_server()
        self._context.__enter__()
        log.info("LSP: server ready")

    def lookup(self, symbol: str) -> str:
        """Look up ``symbol`` and return each definition's location and source.

        :param symbol: The function/class/variable name to resolve.
        :return: Formatted ``[path:line] name`` + definition snippet blocks, or
            a short status string (``"symbol not found"``, ``"lsp ..."``).
        """
        try:
            self.ensure_started()
        except Exception as exc:  # noqa: BLE001
            log.warning("LSP: server start failed (%s)", exc)
            return f"lsp unavailable: {exc}"

        try:
            hits = self._server.request_workspace_symbol(symbol) or []
        except Exception as exc:  # noqa: BLE001
            return f"lsp query error: {exc}"
        if not hits:
            return "symbol not found"

        blocks = []
        for hit in hits[:_MAX_HITS]:
            name = hit.get("name", symbol)
            location = hit.get("location", {})
            path = uri_to_path(location.get("uri", ""))
            line = location.get("range", {}).get("start", {}).get("line", 0)
            snippet = _definition_snippet(path, line)
            blocks.append(f"[{path}:{line + 1}] {name}\n{snippet}")
        return "\n---\n".join(blocks)


#: Shared language-server instance used by the tool and warmup.
lsp_server = LspServer()


@tool
def lsp_tool(symbol: str) -> str:
    """Look up a SYMBOL (function/class/variable) in the repo using a real Language
    Server (jedi via LSP). Returns each definition's location and source snippet.
    Unlike grep, this is AST/semantic-aware, not text matching."""
    return lsp_server.lookup(symbol)
