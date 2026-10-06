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
    args = parser.parse_args()
    run_benchmark(questions_path=args.questions, cache_mode=args.cache_mode)


if __name__ == "__main__":
    main()
