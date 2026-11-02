"""Run the benchmark over a question set and aggregate per-tool metrics.

For every tool, every question is run through the two-call loop
(:func:`repoqa.agent.answer_question`); accuracy, token usage, cache hits,
cost and latency are aggregated and printed, and a machine-readable
``BENCHMARK_JSON`` block is written to the transcript for the plotter.
"""
import json
import logging
import statistics
from collections import defaultdict
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
    :param cat_total: Per-category question counts.
    :param cat_correct: Per-category correct-answer counts.
    """

    correct: int = 0
    results: list = field(default_factory=list)
    cat_total: dict = field(default_factory=lambda: defaultdict(int))
    cat_correct: dict = field(default_factory=lambda: defaultdict(int))

    def record(self, result: QuestionResult, *, is_correct: bool) -> None:
        """Add one question's result.

        :param result: The question result to record.
        :param is_correct: Whether the answer matched the expected value.
        """
        self.correct += int(is_correct)
        self.results.append(result)
        category = result.category or "uncategorized"
        self.cat_total[category] += 1
        self.cat_correct[category] += int(is_correct)

    def summary(self, *, total_questions: int) -> dict:
        """Compute mean metrics across all recorded questions.

        :param total_questions: Number of questions in the set (for accuracy).
        :return: A summary dict (accuracy, mean tokens/cached/cost/time, totals,
            and a per-category accuracy breakdown).
        """
        def mean(values):
            return statistics.mean(values) if values else 0

        inputs = [r.usage.input_tokens for r in self.results]
        outputs = [r.usage.output_tokens for r in self.results]
        totals = [r.usage.total_tokens for r in self.results]
        cacheds = [r.usage.cached_tokens for r in self.results]
        costs = [r.usage.cost_usd for r in self.results]
        times = [r.latency_s for r in self.results]
        tool_times = [r.tool_time_s for r in self.results]
        rounds = [r.rounds_used for r in self.results]
        max_rounds = max((r.max_rounds for r in self.results), default=0)
        # Per-category breakdown: correct / total / accuracy for each category
        # seen in this run (sorted for stable output).
        by_category = {}
        for category in sorted(self.cat_total):
            cat_n = self.cat_total[category]
            cat_ok = self.cat_correct[category]
            by_category[category] = {
                "correct": cat_ok,
                "n": cat_n,
                "accuracy": cat_ok / cat_n if cat_n else 0,
            }
        return {
            "n": total_questions,
            "correct": self.correct,
            "accuracy": self.correct / total_questions if total_questions else 0,
            "by_category": by_category,
            "mean_input_tokens": mean(inputs),
            "mean_output_tokens": mean(outputs),
            "mean_total_tokens": mean(totals),
            "mean_cached_tokens": mean(cacheds),
            "mean_cost_usd": mean(costs),
            "total_cost_usd": sum(costs),
            "mean_time_s": mean(times),
            "mean_tool_time_s": mean(tool_times),
            "mean_rounds": mean(rounds),
            "max_rounds": max_rounds,
        }


def _emit(line: str = "") -> None:
    """Print a line to the console.

    :param line: The line to print.
    """
    print(line, flush=True)


def _print_summary_table(summaries: dict) -> None:
    """Print the aggregated per-tool comparison table to the console.

    :param summaries: Mapping of tool name to its summary dict.
    """
    _emit("\n\n================ BENCHMARK ================")
    header = (f"{'tool':<12} {'accuracy':<14} {'mean_in':<10} {'mean_out':<10} "
              f"{'mean_tot':<10} {'mean_cached':<12} {'mean_cost$':<12} "
              f"{'mean_time_s':<12} {'mean_tool_s':<12} {'mean_rounds':<12}")
    _emit(header)
    _emit("-" * 114)
    for tool, s in summaries.items():
        _emit(f"{tool:<12} {s['correct']}/{s['n']} = {s['accuracy']:<6.0%}  "
              f"{s['mean_input_tokens']:<10.1f} {s['mean_output_tokens']:<10.1f} "
              f"{s['mean_total_tokens']:<10.1f} {s['mean_cached_tokens']:<12.1f} "
              f"{s['mean_cost_usd']:<12.6f} {s['mean_time_s']:<12.3f} "
              f"{s.get('mean_tool_time_s', 0):<12.4f} "
              f"{s.get('mean_rounds', 0):<4.2f}/{s.get('max_rounds', 0)}")

    _print_category_table(summaries)


def _print_category_table(summaries: dict) -> None:
    """Print a per-category accuracy breakdown, one column per category.

    Overall accuracy aggregates all questions; this shows how each tool does on
    each question category (e.g. local_fact vs complex_reasoning vs
    cross_subsystem), so differences between categories are visible.

    :param summaries: Mapping of tool name to its summary dict.
    """
    # Union of categories across tools (identical in practice, but be safe).
    categories = []
    for s in summaries.values():
        for cat in s.get("by_category", {}):
            if cat not in categories:
                categories.append(cat)
    if not categories:
        return
    categories.sort()

    _emit("\n---------------- ACCURACY BY CATEGORY ----------------")
    header = f"{'tool':<12} {'overall':<14} " + " ".join(f"{c:<20}" for c in categories)
    _emit(header)
    _emit("-" * (26 + 21 * len(categories)))
    for tool, s in summaries.items():
        cells = [f"{tool:<12} {s['correct']}/{s['n']} = {s['accuracy']:<6.0%}"]
        by_cat = s.get("by_category", {})
        for cat in categories:
            entry = by_cat.get(cat)
            if entry:
                cells.append(f"{entry['correct']}/{entry['n']} = {entry['accuracy']:<5.0%}".ljust(20))
            else:
                cells.append(f"{'-':<20}")
        _emit(" ".join(cells))


def run_benchmark(*, questions_path: str = "questions.jsonl", cache_mode: str = None,
                  max_rounds: int = None, log_path: str = None) -> dict:
    """Benchmark every tool over the question set and report the results.

    :param questions_path: Path to the ``.jsonl`` question set.
    :param cache_mode: ``"on"``/``"off"``; defaults to :data:`config.CACHE_MODE`.
    :param max_rounds: Max agent tool-call rounds (``n``); defaults to
        :data:`config.AGENT_ROUNDS`.
    :param log_path: Explicit transcript file path; defaults to a timestamped
        file under ``logs/``.
    :return: Mapping of tool name to its summary dict.
    """
    configure_logging()
    cache_mode = (cache_mode or config.CACHE_MODE).lower()
    max_rounds = max_rounds or config.AGENT_ROUNDS
    questions = load_questions(questions_path)
    total = len(questions)

    client = LLMClient()
    transcript = Transcript(path=log_path)
    transcript.meta(cache_mode=cache_mode, max_rounds=max_rounds,
                    questions_path=questions_path, total_questions=total,
                    model=client.model)
    _emit(f"CACHE_MODE = {cache_mode}")
    _emit(f"AGENT_ROUNDS (n) = {max_rounds}")

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
                max_rounds=max_rounds,
                question_id=question.get("id"),
                expected=question.get("expected", ""),
                category=question.get("category"),
            )
            expected = normalize_answer(question.get("expected", ""))
            is_correct = result.answer == expected
            accumulator.record(result, is_correct=is_correct)
            print(f"    answer={result.answer!r} expected={expected!r} "
                  f"{'OK' if is_correct else 'X'} rounds={result.rounds_used}/{result.max_rounds} "
                  f"tokens(in/out/tot)="
                  f"{result.usage.input_tokens}/{result.usage.output_tokens}/"
                  f"{result.usage.total_tokens} cached={result.usage.cached_tokens} "
                  f"cost=${result.usage.cost_usd:.6f} time={result.latency_s}s")

        summaries[tool_name] = accumulator.summary(total_questions=total)

    _print_summary_table(summaries)

    # Aggregated summary record for the plot script (last line of the JSONL).
    transcript.summary(cache_mode=cache_mode, max_rounds=max_rounds, tools=summaries)
    print(f"\nTranscript (JSONL) written to: {transcript.path}", flush=True)
    return summaries
