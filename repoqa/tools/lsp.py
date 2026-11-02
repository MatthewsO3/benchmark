"""Semantic search tool: symbol lookup via a real language server (clangd/LSP).

Drives **clangd** (the standard C/C++ language server) over an LSP stdio
connection using ``pylspclient``, and issues ``workspace/symbol`` queries — the
semantic, AST-aware analogue of grep. clangd is auto-detected on ``PATH``: if it
is absent the tool degrades to a clear ``"lsp unavailable: clangd not found"``
status so the rest of the benchmark still runs.

Historical note: this tool previously used ``multilspy`` + jedi for a Python
target. multilspy has no C/clangd backend, so for the kernel (C) it was replaced
with a direct clangd client. Definition-snippet extraction was likewise rewritten
from Python indentation rules to C brace-matching.
"""
import os
import shutil
import logging
import threading

from langchain_core.tools import tool

from .. import config
from .common import uri_to_path

log = logging.getLogger("repoqa.tools.lsp")

#: Maximum number of symbol hits included in the output.
_MAX_HITS = 10
#: Maximum lines read for a single definition snippet.
_MAX_SNIPPET_LINES = 80
#: Seconds to wait for clangd to answer a request.
_LSP_TIMEOUT = 30.0


def _bracket_delta(line: str) -> int:
    """Return the net change in brace depth contributed by ``line``.

    Counts ``{`` as +1 and ``}`` as -1. A lightweight scan (it does not strip
    strings/comments); adequate for locating a definition's body in practice.

    :param line: The source line to scan.
    :return: ``line.count('{') - line.count('}')``.
    """
    return line.count("{") - line.count("}")


def _definition_snippet(abs_path: str, line: int, *, max_lines: int = _MAX_SNIPPET_LINES) -> str:
    """Return the full C definition starting at ``line`` (0-indexed).

    Handles the common C shapes:

    * **Braced definitions** (functions, ``struct``/``union``/``enum`` bodies):
      the signature (which may span several lines up to the first ``{``) plus
      the body, ending when brace depth returns to zero.
    * **One-line declarations / prototypes / macros / typedefs**: the statement
      is returned up to its terminating ``;`` (or a trailing ``\\`` continuation
      for multi-line ``#define`` macros).

    Capped at ``max_lines``. Returning the whole body (not just the signature)
    is what lets the answerer confirm behaviour questions like "does X do Y?".

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

    start = line
    end = line + 1

    # Phase 1: find the first '{' at or after the start line (the body opener),
    # unless the statement terminates first with ';' (a prototype/declaration)
    # or is a line-continued macro.
    brace_seen = False
    scan = line
    while scan < total and (scan - start) < max_lines:
        text = lines[scan]
        if "{" in text:
            brace_seen = True
            break
        stripped = text.rstrip()
        # A macro that continues onto the next line ends with a backslash.
        if stripped.endswith("\\"):
            scan += 1
            continue
        if ";" in text:
            # Prototype / declaration / typedef / one-line statement.
            return "".join(lines[start:scan + 1]).rstrip()
        scan += 1

    if not brace_seen:
        # No brace and no clear terminator within the window: return what we saw.
        return "".join(lines[start:min(scan + 1, total)]).rstrip()

    # Phase 2: consume the braced body until depth returns to zero.
    depth = 0
    end = scan
    while end < total and (end - start) < max_lines:
        depth += _bracket_delta(lines[end])
        end += 1
        if depth <= 0:
            break
    return "".join(lines[start:end]).rstrip()


class LspServer:
    """Lazy singleton wrapping a clangd language server over the target repo."""

    def __init__(self):
        self._client = None
        self._process = None
        self._unavailable = None  # cached reason string once known unavailable

    def ensure_started(self) -> None:
        """Start clangd and initialize the LSP session if not already running.

        The heavy subprocess start-up happens here (lazily) so a non-LSP run
        never pays for it.

        :raises RuntimeError: If clangd is not installed or fails to start.
        """
        if self._client is not None:
            return
        if self._unavailable:
            raise RuntimeError(self._unavailable)

        clangd_path = shutil.which("clangd")
        if not clangd_path:
            self._unavailable = "clangd not found on PATH"
            raise RuntimeError(self._unavailable)

        import subprocess
        from pylspclient import LspClient, LspEndpoint, JsonRpcEndpoint

        repo = os.path.abspath(config.TARGET_REPO)
        log.info("LSP: starting clangd for %s ...", repo)
        # --background-index lets clangd build its index so workspace/symbol has
        # data to search; -j bounds parallelism on a shared box.
        self._process = subprocess.Popen(
            [clangd_path, "--background-index", "-j=4", "--log=error"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        rpc = JsonRpcEndpoint(self._process.stdin, self._process.stdout)
        endpoint = LspEndpoint(rpc, timeout=_LSP_TIMEOUT)
        self._client = LspClient(endpoint)

        root_uri = f"file://{repo}"
        self._client.initialize(
            processId=os.getpid(),
            rootPath=repo,
            rootUri=root_uri,
            initializationOptions=None,
            capabilities={"workspace": {"symbol": {"dynamicRegistration": True}}},
            trace="off",
            workspaceFolders=[{"uri": root_uri, "name": "repo"}],
        )
        self._client.initialized()
        log.info("LSP: clangd ready")

    def _workspace_symbol(self, symbol: str):
        """Issue a raw ``workspace/symbol`` request (not wrapped by pylspclient).

        :param symbol: The symbol name to search for.
        :return: The raw list of SymbolInformation dicts (possibly empty/None).
        """
        return self._client.lsp_endpoint.call_method("workspace/symbol", query=symbol)

    def lookup(self, symbol: str) -> str:
        """Look up ``symbol`` and return each definition's location and source.

        :param symbol: The function/struct/macro/variable name to resolve.
        :return: Formatted ``[path:line] name`` + definition snippet blocks, or
            a short status string (``"symbol not found"``, ``"lsp ..."``).
        """
        try:
            self.ensure_started()
        except Exception as exc:  # noqa: BLE001
            log.warning("LSP: server start failed (%s)", exc)
            return f"lsp unavailable: {exc}"

        try:
            hits = self._workspace_symbol(symbol) or []
        except Exception as exc:  # noqa: BLE001
            return f"lsp query error: {exc}"
        if not hits:
            return "symbol not found"

        blocks = []
        for hit in hits[:_MAX_HITS]:
            name = _get(hit, "name", symbol)
            location = _get(hit, "location", {}) or {}
            path = uri_to_path(_get(location, "uri", ""))
            start = _get(_get(location, "range", {}) or {}, "start", {}) or {}
            line = _get(start, "line", 0) or 0
            snippet = _definition_snippet(path, line)
            blocks.append(f"[{path}:{line + 1}] {name}\n{snippet}")
        return "\n---\n".join(blocks)


def _get(obj, key, default=None):
    """Read ``key`` from a dict or an attribute from a pydantic/obj result.

    pylspclient may return either plain dicts or typed objects depending on the
    method path, so normalize both.

    :param obj: Dict or object to read from.
    :param key: Key / attribute name.
    :param default: Value if missing.
    :return: The value or ``default``.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


#: Shared language-server instance used by the tool and warmup.
lsp_server = LspServer()


@tool
def lsp_tool(symbol: str) -> str:
    """Look up a SYMBOL (function/struct/macro/variable) in the repo using a real
    Language Server (clangd via LSP). Returns each definition's location and source
    snippet. Unlike grep, this is AST/semantic-aware, not text matching."""
    return lsp_server.lookup(symbol)
