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
    :param cat_results: Per-category lists of results (for per-category metrics).
    """

    correct: int = 0
    results: list = field(default_factory=list)
    cat_results: dict = field(default_factory=lambda: defaultdict(list))

    def record(self, result: QuestionResult, *, is_correct: bool) -> None:
        """Add one question's result.

        :param result: The question result to record.
        :param is_correct: Whether the answer matched the expected value.
        """
        self.correct += int(is_correct)
        # Tag correctness onto the result so per-category aggregation can reuse
        # the exact same metric computation as the overall summary.
        result._is_correct = is_correct  # noqa: SLF001 (local annotation)
        self.results.append(result)
        category = result.category or "uncategorized"
        self.cat_results[category].append(result)

    @staticmethod
    def _metrics(results: list, *, n: int) -> dict:
        """Compute the full metric block for a list of results.

        Shared by the overall summary and every per-category block (task 5) so
        the two never drift. Covers accuracy, every token category (task 2),
        tool-call counts (task 3), tool time (task 4), cost, latency and rounds
        — each as a mean plus the sum where a sum is meaningful.

        :param results: The question results to aggregate.
        :param n: Denominator for accuracy (number of questions in this bucket).
        :return: A metrics dict.
        """
        def mean(values):
            return statistics.mean(values) if values else 0

        correct = sum(int(getattr(r, "_is_correct", False)) for r in results)

        # Every token category from TokenUsage (task 2).
        token_fields = ("input_tokens", "output_tokens", "total_tokens",
                        "cached_tokens", "cache_write_tokens", "reasoning_tokens",
                        "uncached_input_tokens")
        token_means = {f"mean_{f}": mean([getattr(r.usage, f) for r in results])
                       for f in token_fields}
        token_sums = {f"total_{f}": sum(getattr(r.usage, f) for r in results)
                      for f in token_fields}

        costs = [r.usage.cost_usd for r in results]
        times = [r.latency_s for r in results]
        tool_times = [r.tool_time_s for r in results]
        rounds = [r.rounds_used for r in results]
        tool_calls = [r.tool_calls for r in results]

        # Per-tool breakdown (task 3 + 4): aggregate counts/time/tokens by the
        # tool name the agent called, across all results in this bucket.
        per_tool = {}
        for r in results:
            for name, cnt in (r.tool_call_counts or {}).items():
                slot = per_tool.setdefault(
                    name, {"calls": 0, "time_s": 0.0, "tokens": 0})
                slot["calls"] += cnt
            for name, t in (r.tool_call_time_s or {}).items():
                per_tool.setdefault(
                    name, {"calls": 0, "time_s": 0.0, "tokens": 0})["time_s"] += t
            for name, tu in (r.tool_call_tokens or {}).items():
                per_tool.setdefault(
                    name, {"calls": 0, "time_s": 0.0, "tokens": 0})["tokens"] += \
                    tu.get("total", 0)
        for slot in per_tool.values():
            slot["time_s"] = round(slot["time_s"], 4)

        return {
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else 0,
            **token_means,
            **token_sums,
            "mean_cost_usd": mean(costs),
            "total_cost_usd": sum(costs),
            "mean_time_s": mean(times),
            "mean_tool_time_s": mean(tool_times),
            "total_tool_time_s": round(sum(tool_times), 4),
            "mean_rounds": mean(rounds),
            "max_rounds": max((r.max_rounds for r in results), default=0),
            "mean_tool_calls": mean(tool_calls),
            "total_tool_calls": sum(tool_calls),
            "per_tool": per_tool,
        }

    def summary(self, *, total_questions: int) -> dict:
        """Compute all metrics overall and per category (task 5).

        :param total_questions: Number of questions in the set (for accuracy).
        :return: A summary dict: the overall metric block, plus a ``by_category``
            map holding the SAME metric block computed for each category.
        """
        overall = self._metrics(self.results, n=total_questions)
        by_category = {}
        for category in sorted(self.cat_results):
            bucket = self.cat_results[category]
            by_category[category] = self._metrics(bucket, n=len(bucket))
        overall["by_category"] = by_category
        return overall


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
    header = (f"{'tool':<14} {'accuracy':<14} {'mean_in':<9} {'mean_out':<9} "
              f"{'mean_tot':<9} {'mean_cached':<11} {'mean_reason':<11} "
              f"{'mean_cost$':<11} {'mean_time_s':<11} {'mean_tool_s':<11} "
              f"{'mean_calls':<10} {'mean_rounds':<12}")
    _emit(header)
    _emit("-" * len(header))
    for tool, s in summaries.items():
        _emit(f"{tool:<14} {s['correct']}/{s['n']} = {s['accuracy']:<6.0%}  "
              f"{s['mean_input_tokens']:<9.1f} {s['mean_output_tokens']:<9.1f} "
              f"{s['mean_total_tokens']:<9.1f} {s['mean_cached_tokens']:<11.1f} "
              f"{s.get('mean_reasoning_tokens', 0):<11.1f} "
              f"{s['mean_cost_usd']:<11.6f} {s['mean_time_s']:<11.3f} "
              f"{s.get('mean_tool_time_s', 0):<11.4f} "
              f"{s.get('mean_tool_calls', 0):<10.2f} "
              f"{s.get('mean_rounds', 0):<4.2f}/{s.get('max_rounds', 0)}")

    _print_per_tool_table(summaries)
    _print_category_table(summaries)


def _print_per_tool_table(summaries: dict) -> None:
    """Print per-tool-call counts, time and tokens (tasks 3 & 4).

    For each benchmarked condition, the agent may call its primary tool and the
    shared ``read_file`` several times across the open loop. This breaks the
    totals down by the tool actually called, so grep vs read_file (etc.) cost is
    visible.

    :param summaries: Mapping of tool name to its summary dict.
    """
    _emit("\n---------------- PER TOOL-CALL (summed over run) ----------------")
    header = f"{'condition':<14} {'called_tool':<16} {'calls':<8} {'time_s':<10} {'tokens':<10}"
    _emit(header)
    _emit("-" * len(header))
    for tool, s in summaries.items():
        per_tool = s.get("per_tool", {})
        if not per_tool:
            _emit(f"{tool:<14} {'(no tool calls)':<16}")
            continue
        for called, slot in sorted(per_tool.items()):
            _emit(f"{tool:<14} {called:<16} {slot['calls']:<8} "
                  f"{slot['time_s']:<10.4f} {slot['tokens']:<10}")


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

    _print_full_category_metrics(summaries)


def _print_full_category_metrics(summaries: dict) -> None:
    """Print every key metric broken down per category, per tool (task 5).

    The accuracy-by-category table above is the quick view; this prints the full
    metric block (tokens, cost, time, tool time, tool calls, rounds) for each
    category so differences by question type are visible for every measure, not
    just accuracy.

    :param summaries: Mapping of tool name to its summary dict.
    """
    categories = []
    for s in summaries.values():
        for cat in s.get("by_category", {}):
            if cat not in categories:
                categories.append(cat)
    if not categories:
        return
    categories.sort()

    _emit("\n---------------- ALL METRICS BY CATEGORY ----------------")
    header = (f"{'category':<18} {'tool':<14} {'acc':<12} {'in':<8} {'out':<8} "
              f"{'tot':<8} {'cached':<8} {'reason':<8} {'cost$':<11} "
              f"{'time_s':<9} {'tool_s':<9} {'calls':<7} {'rounds':<7}")
    for cat in categories:
        _emit(f"\n[{cat}]")
        _emit(header)
        _emit("-" * len(header))
        for tool, s in summaries.items():
            c = s.get("by_category", {}).get(cat)
            if not c:
                continue
            _emit(f"{'':<18} {tool:<14} "
                  f"{c['correct']}/{c['n']} = {c['accuracy']:<4.0%} "
                  f"{c['mean_input_tokens']:<8.0f} {c['mean_output_tokens']:<8.0f} "
                  f"{c['mean_total_tokens']:<8.0f} {c['mean_cached_tokens']:<8.0f} "
                  f"{c.get('mean_reasoning_tokens', 0):<8.0f} "
                  f"{c['mean_cost_usd']:<11.6f} {c['mean_time_s']:<9.3f} "
                  f"{c.get('mean_tool_time_s', 0):<9.4f} "
                  f"{c.get('mean_tool_calls', 0):<7.2f} "
                  f"{c.get('mean_rounds', 0):<7.2f}")


def run_benchmark(*, questions_path: str = "questions.jsonl", cache_mode: str = None,
                  max_rounds: int = None, log_path: str = None,
                  tools: list = None, questions: list = None,
                  meta_extra: dict = None) -> dict:
    """Benchmark the selected tools over the question set and report the results.

    :param questions_path: Path to the ``.jsonl`` question set. Ignored when
        ``questions`` is supplied directly (e.g. a ladder rung's subset).
    :param cache_mode: ``"on"``/``"off"``; defaults to :data:`config.CACHE_MODE`.
    :param max_rounds: Safety cap on the open-loop tool-call rounds; defaults to
        :data:`config.AGENT_MAX_ROUNDS`.
    :param log_path: Explicit transcript file path; defaults to a timestamped
        file under ``logs/``.
    :param tools: Which primary tools to benchmark (names). Defaults to all of
        :data:`TOOL_NAMES`. Pass e.g. ``["grep_tool"]`` to run a single tool.
    :param questions: Pre-loaded question dicts to run (overrides
        ``questions_path``); used by the ladder runner to pass a rung's subset.
    :param meta_extra: Extra key/values folded into the transcript ``meta``
        record and every summary (e.g. ladder rung + size), for aggregation.
    :return: Mapping of tool name to its summary dict.
    """
    configure_logging()
    cache_mode = (cache_mode or config.CACHE_MODE).lower()
    max_rounds = max_rounds or config.AGENT_MAX_ROUNDS
    if questions is None:
        questions = load_questions(questions_path)
    total = len(questions)
    tool_names = tools or TOOL_NAMES
    meta_extra = meta_extra or {}

    client = LLMClient()
    transcript = Transcript(path=log_path)
    transcript.meta(cache_mode=cache_mode, max_rounds=max_rounds,
                    questions_path=questions_path, total_questions=total,
                    model=client.model, tools=tool_names, **meta_extra)
    _emit(f"CACHE_MODE = {cache_mode}")
    _emit(f"AGENT round safety cap = {max_rounds} (open loop: agent stops when done)")
    if meta_extra.get("rung"):
        _emit(f"LADDER rung = {meta_extra['rung']}  "
              f"(scoped files = {meta_extra.get('scoped_files', '?')})")
    _emit(f"TOOLS = {tool_names}")

    summaries = {}
    for tool_name in tool_names:
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

        summary = accumulator.summary(total_questions=total)
        if meta_extra:
            summary = {**summary, **meta_extra}
        summaries[tool_name] = summary

    _print_summary_table(summaries)

    # Aggregated summary record for the plot script (last line of the JSONL).
    transcript.summary(cache_mode=cache_mode, max_rounds=max_rounds, tools=summaries)
    print(f"\nTranscript (JSONL) written to: {transcript.path}", flush=True)
    return summaries
