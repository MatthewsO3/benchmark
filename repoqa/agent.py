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
from dataclasses import dataclass, field

from langchain_core.utils.function_calling import convert_to_openai_tool

from . import config
from .llm import LLMClient, apply_cache_mode
from .transcript import Transcript
from .usage import TokenUsage
from .tools import get_tool, get_secondary_tools

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
    :param rounds_used: Number of tool-call rounds actually used (the agent
        stops on its own when it has enough evidence; the open loop is only
        bounded by ``max_rounds`` as a safety cap).
    :param max_rounds: Safety cap on tool-call rounds for the open loop.
    :param category: The question's category (metadata for per-category scoring;
        never shown to the model).
    :param tool_time_s: Cumulative LOCAL tool execution time across rounds, in
        seconds. Measures only ``tool.invoke`` (ripgrep/Qdrant/clangd/subprocess)
        and excludes the LLM API round-trip and any rate-limit backoff.
    :param tool_calls: Total number of tool calls the agent actually made across
        the open loop (task 3: now that the loop is open, this is a first-class
        metric, not a constant).
    :param tool_call_counts: Per-tool count of calls (e.g. how many primary-tool
        vs ``read_file`` calls), keyed by tool name.
    :param tool_call_time_s: Per-tool cumulative LOCAL execution time, keyed by
        tool name (task 4: tool-use time for every tool).
    :param tool_call_tokens: Per-tool cumulative token usage attributed to the
        agent turn that issued each call, keyed by tool name (task 3).
    :param reasoning_tokens: Total hidden reasoning tokens across all calls.
    """

    question: str
    tool: str
    cache_mode: str
    answer: str
    reasoning: str
    tool_output: str
    usage: TokenUsage
    latency_s: float
    rounds_used: int = 0
    max_rounds: int = 0
    category: str = "uncategorized"
    tool_time_s: float = 0.0
    tool_calls: int = 0
    tool_call_counts: dict = field(default_factory=dict)
    tool_call_time_s: dict = field(default_factory=dict)
    tool_call_tokens: dict = field(default_factory=dict)
    reasoning_tokens: int = 0


def answer_question(*, question: str, tool_name: str, client: LLMClient,
                    transcript: Transcript, cache_mode: str = None,
                    max_rounds: int = None, question_id: str = None,
                    expected: str = None, category: str = None) -> QuestionResult:
    """Run the OPEN agent loop for one question and record it.

    The loop is open-ended: each round the agent may call the single tool under
    test (or the shared ``read_file``), and it keeps going until it decides it
    has enough evidence and stops emitting tool calls — at which point the final
    tool-free answerer/verifier call collapses the question + accumulated
    reasoning + tool outputs into one ``yes``/``no`` word. No round is forced;
    the agent is free from the first turn. ``max_rounds`` is only a SAFETY CAP
    that bounds a runaway/looping model, not a target.

    Per-tool-call metrics are recorded now that the loop is open: the number of
    tool calls, and per-tool counts, local execution time, and token usage.

    :param question: The question to answer.
    :param tool_name: Name of the single tool to use (``grep_tool`` etc.).
    :param client: The :class:`~repoqa.llm.LLMClient` to use.
    :param transcript: The :class:`~repoqa.transcript.Transcript` to log to.
    :param cache_mode: ``"on"``/``"off"``; defaults to :data:`config.CACHE_MODE`.
    :param max_rounds: Safety cap on tool-call rounds; defaults to
        :data:`config.AGENT_MAX_ROUNDS`.
    :return: The :class:`QuestionResult`.
    """
    tool = get_tool(tool_name)
    cache_mode = (cache_mode or config.CACHE_MODE).lower()
    max_rounds = max_rounds or config.AGENT_MAX_ROUNDS

    # The primary (benchmarked) tool, plus any shared secondary tools offered
    # alongside it (e.g. read_file for grep/lsp/rag; none for shell). Build a
    # name -> tool map so a round can dispatch to whichever one the model picks.
    secondary_tools = get_secondary_tools(tool_name)
    callable_tools = {tool.name: tool}
    for extra in secondary_tools:
        callable_tools[extra.name] = extra
    tool_schemas = [convert_to_openai_tool(t) for t in (tool, *secondary_tools)]

    start = time.perf_counter()

    # Build the running chat history the agent loop extends each round.
    # CACHE INVARIANT: the stable system prompt goes FIRST so the provider can
    # cache it as a shared prefix across all questions; the per-question content
    # (the question, then each round's tool result) is only ever appended AFTER
    # it. Never prepend question- or retrieval-specific text ahead of the system
    # prompt, or the cached prefix breaks and every call pays full price.
    system_prompt = apply_cache_mode(config.SYSTEM_PROMPT, cache_mode)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    usage = TokenUsage()
    reasoning_parts = []
    tool_output_parts = []
    rounds_used = 0
    tool_time_s = 0.0  # cumulative LOCAL tool execution time (excludes LLM API)
    rounds_log = []  # structured per-round detail for the JSONL record
    tool_calls = 0  # total tool calls across the open loop (task 3)
    tool_call_counts = {}      # per-tool call count (task 3)
    tool_call_time_s = {}      # per-tool cumulative local time (task 4)
    tool_call_tokens = {}      # per-tool cumulative TokenUsage (task 3)
    hit_cap = False

    for round_no in range(1, max_rounds + 1):
        # OPEN loop: no round is forced. The agent chooses whether to call a
        # tool (tool_choice="auto") every round and stops on its own when done;
        # max_rounds is only a safety cap.
        log.info("[%s] LLM call (agent round %d, cap %d, auto) model=%s",
                 tool_name, round_no, max_rounds, client.model)
        agent = client.call_agent(
            messages=messages,
            tool_schemas=tool_schemas,
            tool_name=tool_name,
            force=False,
        )
        usage = usage + agent.usage
        if agent.reasoning:
            reasoning_parts.append(f"[round {round_no}] {agent.reasoning}")

        round_entry = {
            "round": round_no,
            "forced": False,
            "reasoning": agent.reasoning,
            "tool_name": agent.tool_name,
            "tool_args": agent.tool_args,
            "tool_output": None,
            "usage": agent.usage.as_dict(),
        }

        # Keep the assistant turn in history for the next round.
        messages.append(agent.assistant_message)

        if not agent.tool_name:
            # Agent decided it is done (no tool call) -> exit the open loop and
            # proceed to the final answerer/verifier call.
            round_entry["stopped_early"] = True
            rounds_log.append(round_entry)
            log.info("[%s] agent finished after %d tool-call round(s) (said done)",
                     tool_name, rounds_used)
            # Drop the trailing assistant turn that carried no tool call.
            messages.pop()
            break

        rounds_used = round_no
        tool_calls += 1
        called_name = agent.tool_name
        tool_call_counts[called_name] = tool_call_counts.get(called_name, 0) + 1
        # Attribute this agent turn's token usage to the tool it called (task 3).
        tool_call_tokens[called_name] = (
            tool_call_tokens.get(called_name, TokenUsage()) + agent.usage
        )
        log.info("[%s] round %d tool call %s args=%s", tool_name, round_no,
                 agent.tool_name, agent.tool_args)
        called_tool = callable_tools.get(agent.tool_name)
        if called_tool is None:
            # Model named a tool that wasn't offered (shouldn't happen since
            # only our schemas are passed). Record and feed back so it recovers.
            round_output = (f"tool error: unknown tool {agent.tool_name!r}; "
                            f"available: {sorted(callable_tools)}")
            round_entry["tool_error"] = round_output
            log.warning("[%s] round %d named unknown tool %r",
                        tool_name, round_no, agent.tool_name)
        else:
            tool_start = time.perf_counter()
            try:
                round_output = str(called_tool.invoke(agent.tool_args or {}))
            except Exception as exc:  # noqa: BLE001
                # A weak model sometimes emits a malformed tool call (e.g. empty
                # args missing a required field). Don't crash the whole run:
                # record the failure as this round's "output" and feed it back
                # so the model can correct itself on the next round.
                round_output = f"tool error: invalid arguments {agent.tool_args!r}: {exc}"
                round_entry["tool_error"] = str(exc)
                log.warning("[%s] round %d tool call failed (%s)", tool_name, round_no, exc)
            else:
                log.info("[%s] round %d (%s) tool result: %d chars",
                         tool_name, round_no, agent.tool_name, len(round_output))
            # Local tool execution time ONLY (ripgrep/Qdrant/clangd/subprocess).
            # Deliberately excludes the LLM API round-trip and any rate-limit
            # backoff, which live in the client calls outside this block.
            round_tool_time = round(time.perf_counter() - tool_start, 4)
            round_entry["tool_time_s"] = round_tool_time
            tool_time_s += round_tool_time
            # Per-tool cumulative local time (task 4).
            tool_call_time_s[called_name] = round(
                tool_call_time_s.get(called_name, 0.0) + round_tool_time, 4)
        round_entry["tool_output"] = round_output
        rounds_log.append(round_entry)
        tool_output_parts.append(f"[round {round_no}] {round_output}")

        # Feed the tool result back so the next round can build on it.
        messages.append({
            "role": "tool",
            "tool_call_id": agent.tool_call_id,
            "content": round_output,
        })
    else:
        # Loop exhausted the safety cap without the agent saying "done".
        hit_cap = True
        log.warning("[%s] hit safety cap of %d rounds without the agent "
                    "stopping; proceeding to answerer", tool_name, max_rounds)

    reasoning = "\n\n".join(reasoning_parts)
    tool_output = "\n\n".join(tool_output_parts)

    # Final call: tool-free answerer. All accumulated reasoning and tool output
    # is forwarded so every round contributes to the final answer.
    # CACHE INVARIANT: the stable ANSWERER_SYSTEM_PROMPT is sent as the system
    # message (the cacheable prefix); all variable content (question, reasoning,
    # tool/RAG output) is packed into this user-message context that follows it.
    # Keep the variable content here, after the system prompt — never before.
    context = (
        f"QUESTION:\n{question}\n\n"
        f"AGENT REASONING:\n{reasoning or '(none)'}\n\n"
        f"TOOL CONTEXT:\n{tool_output or '(none)'}"
    )
    log.info("[%s] LLM call (answerer, no tools) after %d round(s)",
             tool_name, rounds_used)
    answerer = client.call_answerer(
        system_prompt=config.ANSWERER_SYSTEM_PROMPT,
        context=context,
        cache_mode=cache_mode,
    )
    usage = usage + answerer.usage

    answer = normalize_answer(answerer.text)
    latency_s = round(time.perf_counter() - start, 3)
    result = QuestionResult(
        question=question,
        tool=tool_name,
        cache_mode=cache_mode,
        answer=answer,
        reasoning=reasoning,
        tool_output=tool_output,
        usage=usage,
        latency_s=latency_s,
        rounds_used=rounds_used,
        max_rounds=max_rounds,
        category=category or "uncategorized",
        tool_time_s=round(tool_time_s, 4),
        tool_calls=tool_calls,
        tool_call_counts=tool_call_counts,
        tool_call_time_s=tool_call_time_s,
        tool_call_tokens={name: tu.as_dict() for name, tu in tool_call_tokens.items()},
        reasoning_tokens=usage.reasoning_tokens,
    )

    # One self-contained JSONL record with everything the old text log held.
    expected_norm = normalize_answer(expected) if expected is not None else None
    transcript.question_record(record={
        "id": question_id,
        "category": category or "uncategorized",
        "tool": tool_name,
        "cache_mode": cache_mode,
        "max_rounds": max_rounds,
        "rounds_used": rounds_used,
        "hit_cap": hit_cap,
        "tool_calls": tool_calls,
        "tool_call_counts": tool_call_counts,
        "tool_call_time_s": tool_call_time_s,
        "tool_call_tokens": {name: tu.as_dict() for name, tu in tool_call_tokens.items()},
        "question": question,
        "expected": expected,
        "expected_normalized": expected_norm,
        "answer": answer,
        "correct": (answer == expected_norm) if expected_norm is not None else None,
        "prompts": {
            "agent_system": system_prompt,
            "answerer_system": answerer.system_prompt,
        },
        "rounds": rounds_log,
        "answerer": {
            "context": answerer.context,
            "output": answerer.text,
            "usage": answerer.usage.as_dict(),
        },
        "usage_total": usage.as_dict(),
        "latency_s": latency_s,
        "tool_time_s": round(tool_time_s, 4),
    })

    log.info("[%s] answer=%r rounds=%d (cap %d) tool_calls=%d "
             "tokens(in/out/total)=%d/%d/%d cached=%d reasoning=%d cost=$%.6f time=%.2fs",
             tool_name, result.answer, rounds_used, max_rounds, tool_calls,
             result.usage.input_tokens, result.usage.output_tokens,
             result.usage.total_tokens, result.usage.cached_tokens,
             result.usage.reasoning_tokens, result.usage.cost_usd,
             result.latency_s)
    return result
