"""CLI entry point: benchmark the grep / LSP / RAG tools over a question set.

Usage::

    uv run run_eval.py questions.jsonl
    uv run run_eval.py questions.jsonl --cache off
"""
import argparse

from repoqa.benchmark import run_benchmark


def main() -> None:
    """Parse CLI arguments and run the benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark grep/lsp/rag tools on a question set.")
    parser.add_argument(
        "questions", nargs="?", default="questions.jsonl",
        help="path to questions .jsonl (default: questions.jsonl)")
    parser.add_argument(
        "--cache", choices=["on", "off"], default=None, dest="cache_mode",
        help="prompt caching: 'on' (stable prefix, cheaper re-runs) or 'off' "
             "(nonce prefix, reproducible cold cost). Defaults to config.CACHE_MODE.")
    parser.add_argument(
        "--rounds", "-n", type=int, default=None, dest="max_rounds",
        help="max agent tool-call rounds n (round 1 forces the tool, rounds "
             "2..n are optional and may stop early). Defaults to config.AGENT_ROUNDS.")
    parser.add_argument(
        "--log", "-o", default=None, dest="log_path",
        help="transcript output file path (default: logs/run_<timestamp>.log)")
    args = parser.parse_args()
    run_benchmark(questions_path=args.questions, cache_mode=args.cache_mode,
                  max_rounds=args.max_rounds, log_path=args.log_path)


if __name__ == "__main__":
    main()
