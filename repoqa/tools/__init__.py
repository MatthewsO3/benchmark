"""The three retrieval tools under benchmark, plus a small registry.

Exposes :data:`ALL_TOOLS`, :data:`TOOL_NAMES`, :func:`get_tool` and
:func:`warmup`.
"""
import logging

from .grep import grep_tool
from .lsp import lsp_tool, lsp_server
from .rag import rag_tool, rag_index

log = logging.getLogger("repoqa.tools")

#: All benchmarked tools, in display order.
ALL_TOOLS = [grep_tool, lsp_tool, rag_tool]
#: Names of all benchmarked tools.
TOOL_NAMES = [t.name for t in ALL_TOOLS]

_BY_NAME = {t.name: t for t in ALL_TOOLS}


def get_tool(name: str):
    """Return the registered tool with the given name.

    :param name: Tool name (e.g. ``"grep_tool"``).
    :return: The langchain tool object.
    :raises ValueError: If no tool with that name is registered.
    """
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(f"unknown tool: {name}. available: {TOOL_NAMES}") from None


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
