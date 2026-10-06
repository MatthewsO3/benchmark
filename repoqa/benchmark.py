"""Run the benchmark over a question set and aggregate per-tool metrics.

For every tool, every question is run through the two-call loop
(:func:`repoqa.agent.answer_question`); accuracy, token usage, cache hits,
cost and latency are aggregated and printed, and a machine-readable
``BENCHMARK_JSON`` block is written to the transcript for the plotter.
"""
import json
import logging
import statistics
from dataclasses import dataclass, field

from . import config, configure_logging
from .agent import answer_question, normalize_answer, QuestionResult
from .llm import LLMClient
from .transcript import Transcript
from .tools import TOOL_NAMES, warmup

log = logging.getLogger("repoqa.benchmark")


def load_questions(path: str) -> list:
    """Load a ``.jsonl`` question set.

    :param path: Path to the file; one JSON object (``id``, ``question``,
        ``expected``) per line.
    :return: List of question dicts.
    """
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass
class ToolAccumulator:
    """Collects per-question metrics for one tool and computes the summary.

    :param correct: Number of correct answers so far.
    :param results: The individual :class:`~repoqa.agent.QuestionResult` records.
    """

    correct: int = 0
    results: list = field(default_factory=list)

    def record(self, result: QuestionResult, *, is_correct: bool) -> None:
        """Add one question's result.

        :param result: The question result to record.
        :param is_correct: Whether the answer matched the expected value.
        """
        self.correct += int(is_correct)
        self.results.append(result)

    def summary(self, *, total_questions: int) -> dict:
        """Compute mean metrics across all recorded questions.

        :param total_questions: Number of questions in the set (for accuracy).
        :return: A summary dict (accuracy, mean tokens/cached/cost/time, totals).
        """
        def mean(values):
            return statistics.mean(values) if values else 0

        inputs = [r.usage.input_tokens for r in self.results]
        outputs = [r.usage.output_tokens for r in self.results]
        totals = [r.usage.total_tokens for r in self.results]
        cacheds = [r.usage.cached_tokens for r in self.results]
        costs = [r.usage.cost_usd for r in self.results]
        times = [r.latency_s for r in self.results]
        return {
            "n": total_questions,
            "correct": self.correct,
            "accuracy": self.correct / total_questions if total_questions else 0,
            "mean_input_tokens": mean(inputs),
            "mean_output_tokens": mean(outputs),
            "mean_total_tokens": mean(totals),
            "mean_cached_tokens": mean(cacheds),
            "mean_cost_usd": mean(costs),
            "total_cost_usd": sum(costs),
            "mean_time_s": mean(times),
        }


def _emit(transcript: Transcript, line: str = "") -> None:
    """Print a line to the console and append it to the transcript.

    :param transcript: The run transcript.
    :param line: The line to emit.
    """
    print(line, flush=True)
    transcript.write(line)


def _print_summary_table(transcript: Transcript, summaries: dict) -> None:
    """Print the aggregated per-tool comparison table.

    :param transcript: The run transcript (table is echoed to it).
    :param summaries: Mapping of tool name to its summary dict.
    """
    _emit(transcript, "\n\n================ BENCHMARK ================")
    header = (f"{'tool':<12} {'accuracy':<14} {'mean_in':<10} {'mean_out':<10} "
              f"{'mean_tot':<10} {'mean_cached':<12} {'mean_cost$':<12} {'mean_time_s':<12}")
    _emit(transcript, header)
    _emit(transcript, "-" * 90)
    for tool, s in summaries.items():
        _emit(transcript,
              f"{tool:<12} {s['correct']}/{s['n']} = {s['accuracy']:<6.0%}  "
              f"{s['mean_input_tokens']:<10.1f} {s['mean_output_tokens']:<10.1f} "
              f"{s['mean_total_tokens']:<10.1f} {s['mean_cached_tokens']:<12.1f} "
              f"{s['mean_cost_usd']:<12.6f} {s['mean_time_s']:<12.3f}")


def run_benchmark(*, questions_path: str = "questions.jsonl", cache_mode: str = None) -> dict:
    """Benchmark every tool over the question set and report the results.

    :param questions_path: Path to the ``.jsonl`` question set.
    :param cache_mode: ``"on"``/``"off"``; defaults to :data:`config.CACHE_MODE`.
    :return: Mapping of tool name to its summary dict.
    """
    configure_logging()
    cache_mode = (cache_mode or config.CACHE_MODE).lower()
    questions = load_questions(questions_path)
    total = len(questions)

    client = LLMClient()
    transcript = Transcript()
    _emit(transcript, f"CACHE_MODE = {cache_mode}")

    summaries = {}
    for tool_name in TOOL_NAMES:
        print(f"\n########## TOOL: {tool_name} ##########", flush=True)
        # Prime heavy one-time cost (index build / server start) so it doesn't
        # pollute the first question's measured latency.
        print(f"[{tool_name}] warming up...", flush=True)
        warmup(tool_name)

        accumulator = ToolAccumulator()
        for i, question in enumerate(questions, 1):
            print(f"\n--- [{tool_name}] [{i}/{total}] {question['id']} ---", flush=True)
            result = answer_question(
                question=question["question"],
                tool_name=tool_name,
                client=client,
                transcript=transcript,
                cache_mode=cache_mode,
            )
            expected = normalize_answer(question.get("expected", ""))
            is_correct = result.answer == expected
            accumulator.record(result, is_correct=is_correct)
            print(f"    answer={result.answer!r} expected={expected!r} "
                  f"{'OK' if is_correct else 'X'} tokens(in/out/tot)="
                  f"{result.usage.input_tokens}/{result.usage.output_tokens}/"
                  f"{result.usage.total_tokens} cached={result.usage.cached_tokens} "
                  f"cost=${result.usage.cost_usd:.6f} time={result.latency_s}s")

        summaries[tool_name] = accumulator.summary(total_questions=total)

    _print_summary_table(transcript, summaries)

    # Machine-readable block for the plot script.
    transcript.write("\n===BENCHMARK_JSON===")
    transcript.write(json.dumps({"cache_mode": cache_mode, "tools": summaries}))
    transcript.write("===END_BENCHMARK_JSON===")
    print(f"\nTranscript + benchmark written to: {transcript.path}", flush=True)
    return summaries
