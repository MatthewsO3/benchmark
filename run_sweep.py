"""Run the benchmark for a range of round budgets ``n`` back-to-back.

Runs ``run_benchmark`` once per ``n`` in ``[1..max]`` without manual babysitting,
saving each run's transcript as ``logs/sweep_<timestamp>/n{k}.jsonl`` and a
``sweep.json`` manifest, then draws a comparison chart across all ``n``.

Usage::

    uv run run_sweep.py                                   # n=1..3 on default questions
    uv run run_sweep.py data/benchmark_questions.jsonl --max-n 5
    uv run run_sweep.py data/benchmark_questions.jsonl --max-n 5 --cache off
    uv run run_sweep.py --ns 1 2 4                         # explicit n values

The resulting folder can be re-plotted any time with::

    uv run plot_sweep.py logs/sweep_<timestamp>
"""
import os
import json
import argparse
from datetime import datetime

from repoqa.benchmark import run_benchmark
from repoqa.plotting import plot_sweep


def main() -> None:
    """Parse CLI arguments and run the sweep across round budgets."""
    parser = argparse.ArgumentParser(
        description="Run the benchmark for several round budgets n and compare.")
    parser.add_argument(
        "questions", nargs="?", default="data/benchmark_questions.jsonl",
        help="path to questions .jsonl (default: data/benchmark_questions.jsonl)")
    parser.add_argument(
        "--max-n", type=int, default=3, dest="max_n",
        help="run n=1..max-n (default: 3). Ignored if --ns is given.")
    parser.add_argument(
        "--ns", type=int, nargs="+", default=None,
        help="explicit list of n values to run, e.g. --ns 1 2 4")
    parser.add_argument(
        "--cache", choices=["on", "off"], default=None, dest="cache_mode",
        help="prompt caching mode; defaults to config.CACHE_MODE.")
    parser.add_argument(
        "--no-plot", action="store_true",
        help="skip drawing the comparison chart at the end.")
    args = parser.parse_args()

    ns = args.ns if args.ns else list(range(1, args.max_n + 1))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = os.path.join("logs", f"sweep_{stamp}")
    os.makedirs(sweep_dir, exist_ok=True)

    print(f"SWEEP: n values = {ns}")
    print(f"SWEEP: questions = {args.questions}")
    print(f"SWEEP: output dir = {sweep_dir}\n")

    manifest = {
        "created": stamp,
        "questions": args.questions,
        "cache_mode": args.cache_mode,
        "ns": ns,
        "runs": {},
    }

    for n in ns:
        log_path = os.path.join(sweep_dir, f"n{n}.jsonl")
        print(f"\n{'=' * 70}\nSWEEP: running n={n} -> {log_path}\n{'=' * 70}")
        summaries = run_benchmark(
            questions_path=args.questions,
            cache_mode=args.cache_mode,
            max_rounds=n,
            log_path=log_path,
        )
        manifest["runs"][str(n)] = {"log": os.path.basename(log_path),
                                    "tools": summaries}
        # Persist after each n so a crash still leaves a usable manifest.
        with open(os.path.join(sweep_dir, "sweep.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    print(f"\nSWEEP: done. logs + manifest in {sweep_dir}")
    if not args.no_plot:
        plot_sweep(sweep_dir)


if __name__ == "__main__":
    main()
