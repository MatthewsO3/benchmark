"""Human-readable transcript logging for a benchmark run.

Each run writes one timestamped file under ``logs/`` containing the full
prompt, reasoning, tool output and answer for every question, so you can see
exactly why a tool produced a given answer.
"""
import os
from datetime import datetime

_WIDE = 80


class Transcript:
    """Append-only writer for a single run's transcript.

    :param directory: Directory to create the log file in (created if missing).
    """

    def __init__(self, *, directory: str = "logs"):
        os.makedirs(directory, exist_ok=True)
        self.path = os.path.join(directory, f"run_{datetime.now():%Y%m%d_%H%M%S}.log")

    def write(self, line: str = "") -> None:
        """Append one line to the transcript.

        :param line: Text to write; a trailing newline is added.
        """
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def question(self, *, tool: str, cache_mode: str, question: str) -> None:
        """Write the banner that opens a question's section.

        :param tool: The tool under test.
        :param cache_mode: The active cache mode.
        :param question: The question text.
        """
        self.write("\n\n" + "#" * _WIDE)
        self.write(f"# QUESTION (tool={tool}, cache={cache_mode}): {question}")
        self.write("#" * _WIDE)

    def agent_call(self, *, tool: str, question: str, agent) -> None:
        """Record the forced-tool agent call (call #1).

        :param tool: The tool under test.
        :param question: The question text.
        :param agent: The :class:`~repoqa.llm.AgentCall` returned.
        """
        usage = agent.usage
        self.write("\n" + "=" * _WIDE)
        self.write(f"TOOL={tool} | LLM CALL #1 | agent, forced tool (raw SDK, reasoning on)")
        self.write(f"TOKENS: input={usage.input_tokens} output={usage.output_tokens} "
                   f"total={usage.total_tokens} cached={usage.cached_tokens} "
                   f"cost=${usage.cost_usd:.6f}")
        self.write("-" * _WIDE)
        self.write("INPUT MESSAGES:")
        self.write(f"  [system]\n    {agent.system_prompt}")
        self.write(f"  [user]\n    {question}")
        self.write("-" * _WIDE)
        self.write("REASONING:")
        self.write(f"  {agent.reasoning or '(none)'}")
        self.write(f"  tool_call={{'name': {agent.tool_name!r}, 'args': {agent.tool_args!r}}}")
        self.write("=" * _WIDE)

    def tool_execution(self, *, tool: str, args, output: str) -> None:
        """Record a tool invocation and its raw output.

        :param tool: The tool name.
        :param args: The arguments the tool was called with.
        :param output: The tool's textual output.
        """
        self.write("\n" + "-" * _WIDE)
        self.write(f"TOOL EXECUTION: {tool} args={args}")
        self.write(f"TOOL OUTPUT ({len(output)} chars):")
        self.write(output)
        self.write("-" * _WIDE)

    def answerer_call(self, *, tool: str, answerer) -> None:
        """Record the tool-free answerer call (call #2).

        :param tool: The tool under test.
        :param answerer: The :class:`~repoqa.llm.AnswererCall` returned.
        """
        usage = answerer.usage
        self.write("\n" + "=" * _WIDE)
        self.write(f"TOOL={tool} | LLM CALL #2 | answerer, no tools")
        self.write(f"TOKENS: input={usage.input_tokens} output={usage.output_tokens} "
                   f"total={usage.total_tokens} cached={usage.cached_tokens} "
                   f"cost=${usage.cost_usd:.6f}")
        self.write("-" * _WIDE)
        self.write("INPUT MESSAGES:")
        self.write(f"  [system]\n    {answerer.system_prompt}")
        self.write(f"  [user]\n    {answerer.context}")
        self.write("-" * _WIDE)
        self.write("OUTPUT:")
        self.write(f"  {answerer.text}")
        self.write("=" * _WIDE)

    def final(self, *, answer: str, usage, latency_s: float) -> None:
        """Write the one-line summary that closes a question's section.

        :param answer: The normalized one-word answer.
        :param usage: Combined :class:`~repoqa.usage.TokenUsage` for both calls.
        :param latency_s: Wall-clock time for both calls, in seconds.
        """
        self.write(f"\n>>> FINAL answer={answer!r} "
                   f"tokens in/out/total={usage.input_tokens}/{usage.output_tokens}/"
                   f"{usage.total_tokens} cached={usage.cached_tokens} "
                   f"cost=${usage.cost_usd:.6f} time={latency_s}s")
