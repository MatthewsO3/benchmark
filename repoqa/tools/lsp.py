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
import subprocess

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
#: Max candidate files opened (didOpen) to force-index a symbol on a miss.
_MAX_OPEN_CANDIDATES = 5
#: Seconds to wait for clangd to index freshly-opened files before retrying.
_INDEX_SETTLE_S = 2.0


def _find_clangd():
    """Locate a clangd executable, preferring PATH then the active venv.

    The ``clangd`` PyPI wheel installs the binary into the environment's
    ``bin``/``Scripts`` dir, which is on PATH under ``uv run`` but not always
    otherwise. Check PATH first, then the running interpreter's bin dir so the
    tool works even when the venv is not activated.

    :return: Absolute path to clangd, or ``None`` if not found.
    """
    found = shutil.which("clangd")
    if found:
        return found
    import sys
    bindir = os.path.dirname(sys.executable)
    for name in ("clangd", "clangd.exe"):
        candidate = os.path.join(bindir, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


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
        self._opened = set()      # file URIs already sent via didOpen

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

        clangd_path = _find_clangd()
        if not clangd_path:
            self._unavailable = ("clangd not found (install it with "
                                 "`uv pip install clangd`, or put it on PATH)")
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

    def preindex_scope(self) -> int:
        """Force-index EVERY in-scope source file at warmup (``didOpen`` each).

        Fairness invariant: clangd's ``--background-index`` warms lazily, so
        without this the index grows *during* the run — a symbol lookup that
        misses early (and must fall back to the slow candidate ``didOpen``
        retry) may hit cleanly later, once background indexing or an earlier
        question's fallback has parsed its file. That makes a question's result
        and latency depend on run order and biases the LSP condition
        optimistically versus the stateless grep/ripgrep tools.

        Opening every in-scope file up front removes that: when the first
        question runs, the whole rung is already parsed and in the index, so
        every question faces the SAME index state and the same chance of a
        first-try hit — the semantic analogue of ripgrep's "build the file
        index once at warmup, then every query sees it".

        Scope-aware: when a ladder rung is active this opens exactly the rung's
        files (never the whole tree), so each rung pre-indexes only its own
        scope. Idempotent per file (``_did_open`` skips already-open URIs), so
        the per-question fallback adds nothing new afterwards.

        :return: The number of files opened (``didOpen`` sent).
        """
        self.ensure_started()
        from ..ladder import active_scope, scope_file_list
        from .common import walk_repo
        if active_scope() is not None:
            files = scope_file_list()
        else:
            files = list(walk_repo(os.path.abspath(config.TARGET_REPO)))
        log.info("LSP: pre-indexing %d in-scope file(s) at warmup ...", len(files))
        opened = 0
        for path in files:
            if self._did_open(path):
                opened += 1
        # Let clangd finish parsing the just-opened TUs before queries start, so
        # the first question sees a settled index like every later one.
        if opened:
            import time
            time.sleep(_INDEX_SETTLE_S)
        log.info("LSP: pre-index complete (%d file(s) opened)", opened)
        return opened

    def _workspace_symbol(self, symbol: str):
        """Issue a raw ``workspace/symbol`` request (not wrapped by pylspclient).

        :param symbol: The symbol name to search for.
        :return: The raw list of SymbolInformation dicts (possibly empty/None).
        """
        return self._client.lsp_endpoint.call_method("workspace/symbol", query=symbol)

    def _did_open(self, path: str) -> bool:
        """Send ``textDocument/didOpen`` for ``path`` so clangd indexes it now.

        clangd's ``--background-index`` warms lazily, so a fresh
        ``workspace/symbol`` query can miss a symbol whose file has not been
        parsed yet. Opening the file forces an immediate parse+index, which is
        what makes the subsequent lookup deterministic. Already-opened files are
        skipped.

        :param path: Absolute path to a source file to open.
        :return: ``True`` if a didOpen was sent (or already open), ``False`` on
            read error.
        """
        uri = f"file://{path}"
        if uri in self._opened:
            return True
        try:
            with open(path, encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
        except OSError:
            return False
        try:
            self._client.didOpen({
                "uri": uri,
                "languageId": config.SOURCE_LANGUAGE or "c",
                "version": 1,
                "text": text,
            })
        except Exception as exc:  # noqa: BLE001
            log.warning("LSP: didOpen failed for %s (%s)", path, exc)
            return False
        self._opened.add(uri)
        return True

    def _candidate_files(self, symbol: str) -> list:
        """Find source files likely to define ``symbol`` (a quick lexical scan).

        Used only to force-index on a ``workspace/symbol`` miss: a tiny ripgrep
        (or Python fallback) for the bare symbol name, scoped to in-scope C
        sources, yields the files to ``didOpen`` so clangd parses them and the
        retry query can resolve the symbol semantically.

        :param symbol: The symbol name to locate candidate files for.
        :return: Up to :data:`_MAX_OPEN_CANDIDATES` absolute file paths.
        """
        repo = os.path.abspath(config.TARGET_REPO)
        from ..ladder import in_scope
        command = ["rg", "-l", "-w", "--no-messages"]
        for glob in config.SOURCE_GLOBS:
            command += ["-g", glob]
        command += ["-e", symbol, repo]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=60,
                encoding="utf-8", errors="replace",
            )
            if completed.returncode in (0, 1):
                # Containment: never open a file outside the active rung scope,
                # or the fallback could pull in (and reveal) out-of-scope code.
                paths = [p for p in completed.stdout.splitlines()
                         if p.strip() and in_scope(p)]
                return paths[:_MAX_OPEN_CANDIDATES]
        except (FileNotFoundError, OSError):
            pass
        # Fallback: Python scan (rg missing). Bounded to the first few matches.
        from .common import walk_repo
        import re
        rx = re.compile(rf"\b{re.escape(symbol)}\b")
        hits = []
        for path in walk_repo(repo):
            if not in_scope(path):
                continue
            try:
                with open(path, encoding="utf-8", errors="ignore") as handle:
                    if rx.search(handle.read()):
                        hits.append(path)
                        if len(hits) >= _MAX_OPEN_CANDIDATES:
                            break
            except OSError:
                continue
        return hits

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
            # Background index may not have reached this symbol's file yet.
            # Force-index the files that lexically mention it (didOpen), let
            # clangd settle briefly, then retry the semantic query. This keeps
            # the result AST-accurate while making the first lookup reliable on
            # a freshly-started server over a large tree like the kernel.
            candidates = self._candidate_files(symbol)
            opened = 0
            for path in candidates:
                if self._did_open(path):
                    opened += 1
            if opened:
                import time
                time.sleep(_INDEX_SETTLE_S)
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

    def close(self) -> None:
        """Shut down clangd cleanly, releasing its subprocess.

        Without an explicit shutdown the clangd child keeps its stdio open and
        the interpreter can block at exit. We skip the LSP ``shutdown``/``exit``
        handshake (it can itself hang waiting on a reply) and just close our
        pipes and terminate the child, which is reliable. Safe to call more than
        once; registered at interpreter exit.
        """
        client, self._client = self._client, None
        process, self._process = self._process, None
        self._opened.clear()
        if process is not None:
            # Close our ends of the pipes so clangd sees EOF on stdin and we do
            # not block on its stdout, then terminate/kill the child.
            for stream in (process.stdin, process.stdout):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:  # noqa: BLE001
                    pass
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    process.kill()
                except Exception:  # noqa: BLE001
                    pass


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

# Shut clangd down cleanly at interpreter exit (prevents a hang waiting on the
# child process's stdin, mirroring the RAG index's atexit close).
import atexit  # noqa: E402

atexit.register(lsp_server.close)


@tool
def lsp_tool(symbol: str) -> str:
    """Look up a SYMBOL (function/struct/macro/variable) in the repo using a real
    Language Server (clangd via LSP). Returns each definition's location and source
    snippet. Unlike grep, this is AST/semantic-aware, not text matching."""
    return lsp_server.lookup(symbol)
