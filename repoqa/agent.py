"""The two-call benchmark loop for a single question.

Phase 1 (agent): the model is forced to call exactly one tool, and its
reasoning trace is captured. Phase 2 (answerer): a fresh, tool-free call
collapses the question + reasoning + tool output into one ``yes``/``no`` word.
Keeping the tool as the only variable makes accuracy differences attributable
to the tool rather than to multi-step reasoning.
"""
import re
import time
import logging
from dataclasses import dataclass

from langchain_core.utils.function_calling import convert_to_openai_tool

from . import config
from .llm import LLMClient
from .transcript import Transcript
from .usage import TokenUsage
from .tools import get_tool

log = logging.getLogger("repoqa.agent")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def normalize_answer(text: str) -> str:
    """Reduce raw model output to a single lowercase word.

    Strips ``<think>`` reasoning blocks, then prefers an explicit ``yes``/``no``;
    otherwise falls back to the first word-like token.

    :param text: Raw model output.
    :return: A single lowercase word (may be empty).
    """
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    match = _YES_NO_RE.search(text)
    if match:
        return match.group(1).lower()
    match = _WORD_RE.search(text)
    return match.group(0).lower() if match else text.strip().lower()


@dataclass
class QuestionResult:
    """Outcome of running one question through the loop.

    :param question: The question text.
    :param tool: The tool under test.
    :param cache_mode: The active cache mode.
    :param answer: The normalized one-word answer.
    :param reasoning: The agent's reasoning trace.
    :param tool_output: The raw tool output.
    :param usage: Combined token/cost accounting for both calls.
    :param latency_s: Wall-clock time for both calls, in seconds.
    """

    question: str
    tool: str
    cache_mode: str
    answer: str
    reasoning: str
    tool_output: str
    usage: TokenUsage
    latency_s: float


def answer_question(*, question: str, tool_name: str, client: LLMClient,
                    transcript: Transcript, cache_mode: str = None) -> QuestionResult:
    """Run the two-call loop for one question and record it to the transcript.

    :param question: The question to answer.
    :param tool_name: Name of the single tool to use (``grep_tool`` etc.).
    :param client: The :class:`~repoqa.llm.LLMClient` to use.
    :param transcript: The :class:`~repoqa.transcript.Transcript` to log to.
    :param cache_mode: ``"on"``/``"off"``; defaults to :data:`config.CACHE_MODE`.
    :return: The :class:`QuestionResult`.
    """
    tool = get_tool(tool_name)
    cache_mode = (cache_mode or config.CACHE_MODE).lower()
    tool_schema = convert_to_openai_tool(tool)

    transcript.question(tool=tool_name, cache_mode=cache_mode, question=question)
    start = time.perf_counter()

    # Phase 1: the agent is forced to call the single tool under test.
    log.info("[%s] LLM call #1 (forced tool) model=%s", tool_name, client.model)
    agent = client.call_agent(
        system_prompt=config.SYSTEM_PROMPT,
        question=question,
        tool_schema=tool_schema,
        tool_name=tool_name,
        cache_mode=cache_mode,
    )
    transcript.agent_call(tool=tool_name, question=question, agent=agent)

    tool_output = ""
    if agent.tool_name:
        log.info("[%s] tool call args=%s", tool_name, agent.tool_args)
        tool_output = str(tool.invoke(agent.tool_args or {}))
        log.info("[%s] tool result: %d chars", tool_name, len(tool_output))
        transcript.tool_execution(tool=tool_name, args=agent.tool_args, output=tool_output)
    else:
        log.warning("[%s] model did not call the tool", tool_name)

    # Phase 2: tool-free answerer. The agent's own reasoning is forwarded so it
    # still contributes to the final answer, alongside the raw tool output.
    context = (
        f"QUESTION:\n{question}\n\n"
        f"AGENT REASONING:\n{agent.reasoning or '(none)'}\n\n"
        f"TOOL CONTEXT:\n{tool_output or '(none)'}"
    )
    log.info("[%s] LLM call #2 (answerer, no tools)", tool_name)
    answerer = client.call_answerer(
        system_prompt=config.ANSWERER_SYSTEM_PROMPT,
        context=context,
        cache_mode=cache_mode,
    )
    transcript.answerer_call(tool=tool_name, answerer=answerer)

    result = QuestionResult(
        question=question,
        tool=tool_name,
        cache_mode=cache_mode,
        answer=normalize_answer(answerer.text),
        reasoning=agent.reasoning,
        tool_output=tool_output,
        usage=agent.usage + answerer.usage,
        latency_s=round(time.perf_counter() - start, 3),
    )
    transcript.final(answer=result.answer, usage=result.usage, latency_s=result.latency_s)
    log.info("[%s] answer=%r tokens(in/out/total)=%d/%d/%d cached=%d cost=$%.6f time=%.2fs",
             tool_name, result.answer, result.usage.input_tokens, result.usage.output_tokens,
             result.usage.total_tokens, result.usage.cached_tokens, result.usage.cost_usd,
             result.latency_s)
    return result
