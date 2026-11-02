"""CLI entry point: plot a round-budget sweep comparison.

Usage::

    uv run plot_sweep.py logs/sweep_YYYYMMDD_HHMMSS
    uv run plot_sweep.py            # uses the newest sweep_* dir in logs/
"""
import os
import sys
import glob

from repoqa.plotting import plot_sweep


def _newest_sweep(directory: str = "logs") -> str:
    """Return the most recently modified ``sweep_*`` directory.

    :param directory: Directory to search.
    :return: Path to the newest sweep directory.
    :raises SystemExit: If none exist.
    """
    dirs = [d for d in glob.glob(os.path.join(directory, "sweep_*")) if os.path.isdir(d)]
    if not dirs:
        raise SystemExit(f"no sweep_* directories found in {directory}/")
    return max(dirs, key=os.path.getmtime)


def main() -> None:
    """Plot the given sweep directory, or the newest one in ``logs/``."""
    sweep_dir = sys.argv[1] if len(sys.argv) > 1 else _newest_sweep()
    plot_sweep(sweep_dir)


if __name__ == "__main__":
    main()
