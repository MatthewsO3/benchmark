"""Draw a tool-benchmark chart from a run log.

Reads the ``BENCHMARK_JSON`` block that :func:`repoqa.benchmark.run_benchmark`
writes into a log file and draws a side-by-side comparison (accuracy, mean
tokens, mean latency).
"""
import os
import glob
import json

import matplotlib.pyplot as plt

#: Bar colors, one per tool.
_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def newest_log(*, directory: str = "logs") -> str:
    """Return the most recently modified log file in ``directory``.

    :param directory: Directory to search for ``*.log`` files.
    :return: Path to the newest log file.
    :raises SystemExit: If no log files are found.
    """
    files = glob.glob(os.path.join(directory, "*.log"))
    if not files:
        raise SystemExit(f"no log files found in {directory}/")
    return max(files, key=os.path.getmtime)


def load_summary(log_path: str):
    """Extract the machine-readable benchmark block from a log file.

    :param log_path: Path to the run log.
    :return: Tuple of (per-tool summary dict, cache_mode or ``None``).
    :raises SystemExit: If the log has no ``BENCHMARK_JSON`` block.
    """
    with open(log_path, encoding="utf-8") as handle:
        text = handle.read()
    start = text.find("===BENCHMARK_JSON===")
    end = text.find("===END_BENCHMARK_JSON===")
    if start == -1 or end == -1:
        raise SystemExit(f"no BENCHMARK_JSON block in {log_path}. "
                         "Run the benchmark to completion first.")
    payload = text[start + len("===BENCHMARK_JSON==="):end].strip()
    data = json.loads(payload)
    # Current shape: {"cache_mode": ..., "tools": {...}}. Old shape: {tool: {...}}.
    if isinstance(data, dict) and "tools" in data:
        return data["tools"], data.get("cache_mode")
    return data, None


def plot(summary: dict, log_path: str, *, cache_mode: str = None) -> str:
    """Render and save the three-panel comparison chart.

    :param summary: Per-tool summary dict (from :func:`load_summary`).
    :param log_path: Path to the source log (used for the title and output name).
    :param cache_mode: Cache mode to show in the title, if known.
    :return: Path to the saved PNG.
    """
    tools = list(summary.keys())
    accuracy = [summary[t]["accuracy"] * 100 for t in tools]
    mean_tokens = [summary[t]["mean_total_tokens"] for t in tools]
    mean_time = [summary[t]["mean_time_s"] for t in tools]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    title = f"Tool benchmark — {os.path.basename(log_path)}"
    if cache_mode:
        title += f"  (cache={cache_mode})"
    fig.suptitle(title, fontsize=13)

    panels = [
        (axes[0], accuracy, "Accuracy (%)", "%.0f%%", (0, 100)),
        (axes[1], mean_tokens, "Mean total tokens / question", "%.0f", None),
        (axes[2], mean_time, "Mean latency (s) / question", "%.1fs", None),
    ]
    for axis, values, label, fmt, ylim in panels:
        bars = axis.bar(tools, values, color=_COLORS[:len(tools)])
        axis.set_title(label)
        axis.set_ylabel(label)
        if ylim:
            axis.set_ylim(*ylim)
        axis.grid(axis="y", alpha=0.3)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                      fmt % value, ha="center", va="bottom", fontsize=9)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    out_png = os.path.splitext(log_path)[0] + "_benchmark.png"
    plt.savefig(out_png, dpi=130)
    print(f"saved plot -> {out_png}")
    plt.show()
    return out_png


def plot_log(log_path: str = None) -> str:
    """Plot a benchmark from a log file (newest in ``logs/`` if unspecified).

    :param log_path: Path to the run log, or ``None`` to use the newest.
    :return: Path to the saved PNG.
    """
    log_path = log_path or newest_log()
    summary, cache_mode = load_summary(log_path)
    return plot(summary, log_path, cache_mode=cache_mode)
