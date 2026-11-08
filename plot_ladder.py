"""CLI entry point: plot ladder (codebase-scaling) benchmark results.

Draws every metric as a function of codebase size (the ladder rungs), with one
line per tool, from a ladder experiment directory produced by ``run_ladder.py``
(a directory of ``<rung>__<tool>.jsonl`` cells plus a ``manifest.json``). The
final picture compares, e.g., ladder_1/2/3 × grep/rg_index/lsp/rag/shell on
accuracy (overall + per category), tokens, reasoning tokens, cost, latency,
tool time and tool-call count — showing how each tool scales as the indexed
codebase grows.

Usage::

    # point at the experiment directory
    uv run plot_ladder.py logs/ladder_fixed

    # or point at the whole logs/ folder: every subdir with a manifest.json
    # is plotted (each gets its own ladder_comparison.png)
    uv run plot_ladder.py logs

    # default: the newest experiment dir under logs/
    uv run plot_ladder.py
"""
import os
import sys
import glob

from repoqa.plotting import plot_ladder


def _is_experiment_dir(path: str) -> bool:
    """Return whether ``path`` looks like a ladder experiment directory.

    :param path: Candidate directory.
    :return: ``True`` if it holds a ``manifest.json`` or any ``*__*.jsonl`` cell.
    """
    if not os.path.isdir(path):
        return False
    if os.path.exists(os.path.join(path, "manifest.json")):
        return True
    return bool(glob.glob(os.path.join(path, "*__*.jsonl")))


def _find_experiment_dirs(root: str) -> list:
    """Find ladder experiment directories under ``root``.

    If ``root`` itself is an experiment dir, returns just it. Otherwise returns
    every immediate subdirectory that looks like one.

    :param root: A directory (an experiment dir, or a parent like ``logs/``).
    :return: List of experiment directory paths.
    """
    if _is_experiment_dir(root):
        return [root]
    subdirs = [os.path.join(root, d) for d in os.listdir(root)
               if os.path.isdir(os.path.join(root, d))]
    return [d for d in sorted(subdirs) if _is_experiment_dir(d)]


def _newest_experiment_dir(root: str = "logs") -> str:
    """Return the most recently modified experiment dir under ``root``.

    :param root: Directory to search.
    :return: Path to the newest experiment directory.
    :raises SystemExit: If none exist.
    """
    dirs = _find_experiment_dirs(root)
    if not dirs:
        raise SystemExit(f"no ladder experiment directories found under {root}/")
    return max(dirs, key=os.path.getmtime)


def main() -> None:
    """Plot the given ladder experiment dir(s), or the newest under ``logs/``."""
    target = sys.argv[1] if len(sys.argv) > 1 else _newest_experiment_dir()

    if _is_experiment_dir(target):
        dirs = [target]
    else:
        dirs = _find_experiment_dirs(target)
        if not dirs:
            raise SystemExit(f"{target!r} is not a ladder experiment dir and "
                             "contains none. Point me at a dir with a "
                             "manifest.json or <rung>__<tool>.jsonl cells.")
    for exp_dir in dirs:
        print(f"\n=== plotting {exp_dir} ===")
        plot_ladder(exp_dir)


if __name__ == "__main__":
    main()
