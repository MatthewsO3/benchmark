"""The four retrieval tools under benchmark, plus a small registry.

Each benchmarked *condition* is a primary search tool. The three search tools
(grep/LSP/RAG) are additionally offered a shared :func:`read_file` tool so the
model can open the files their results point to; ``shell_tool`` is not, because
it can already read files via its own commands (``cat``/``head``/``tail``).

Exposes :data:`ALL_TOOLS`, :data:`TOOL_NAMES`, :func:`get_tool`,
:func:`get_secondary_tools` and :func:`warmup`.
"""
import logging

from .grep import grep_tool
from .lsp import lsp_tool, lsp_server
from .rag import rag_tool, rag_index
from .shell import shell_tool
from .read import read_file

log = logging.getLogger("repoqa.tools")

#: All benchmarked (primary) tools, in display order.
ALL_TOOLS = [grep_tool, lsp_tool, rag_tool, shell_tool]
#: Names of all benchmarked (primary) tools.
TOOL_NAMES = [t.name for t in ALL_TOOLS]

#: Shared read tool offered alongside the search tools (not itself benchmarked).
SHARED_READ_TOOL = read_file

#: Which primary tools additionally get the shared read tool. ``shell_tool`` is
#: excluded: it can already read files natively, so adding read_file would
#: double-count the same capability and blur its metrics.
_TOOLS_WITH_READ = {"grep_tool", "lsp_tool", "rag_tool"}

_BY_NAME = {t.name: t for t in ALL_TOOLS}


def get_tool(name: str):
    """Return the registered primary tool with the given name.

    :param name: Tool name (e.g. ``"grep_tool"``).
    :return: The langchain tool object.
    :raises ValueError: If no tool with that name is registered.
    """
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(f"unknown tool: {name}. available: {TOOL_NAMES}") from None


def get_secondary_tools(name: str) -> list:
    """Return the extra tools offered alongside the primary tool ``name``.

    For grep/LSP/RAG this is the shared :func:`read_file`; for ``shell_tool``
    it is empty.

    :param name: The primary tool name.
    :return: List of additional langchain tool objects (possibly empty).
    """
    return [SHARED_READ_TOOL] if name in _TOOLS_WITH_READ else []


def warmup(name: str) -> None:
    """Prime a tool's heavy one-time cost before latency is measured.

    Builds the RAG index / loads the embedding model, or starts the LSP server,
    so the first question's timing reflects steady-state cost. ``grep`` has no
    warmup cost.

    :param name: Tool name to warm up.
    """
    if name == "rag_tool":
        rag_index.build()
    elif name == "lsp_tool":
        try:
            lsp_server.ensure_started()
        except Exception as exc:  # noqa: BLE001
            log.warning("LSP warmup failed (%s)", exc)
