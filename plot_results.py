"""CLI entry point: plot a tool benchmark from a run log.

Usage::

    uv run plot_results.py logs/run_YYYYMMDD_HHMMSS.log
    uv run plot_results.py            # uses the newest log in logs/
"""
import sys

from repoqa.plotting import plot_log


def main() -> None:
    """Plot the benchmark from the given log, or the newest log in ``logs/``."""
    plot_log(sys.argv[1] if len(sys.argv) > 1 else None)


if __name__ == "__main__":
    main()
