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

    :param directory: Directory to search for run files.
    :return: Path to the newest run file (``.jsonl`` preferred, else ``.log``).
    :raises SystemExit: If no run files are found.
    """
    files = glob.glob(os.path.join(directory, "*.jsonl")) + \
        glob.glob(os.path.join(directory, "*.log"))
    if not files:
        raise SystemExit(f"no run files (*.jsonl / *.log) found in {directory}/")
    return max(files, key=os.path.getmtime)


def load_summary(log_path: str):
    """Extract the aggregated benchmark summary from a run file.

    Supports the current JSONL format (a record with ``type == "summary"``) and
    the legacy text ``.log`` format (a ``===BENCHMARK_JSON===`` block).

    :param log_path: Path to the run file (``.jsonl`` or legacy ``.log``).
    :return: Tuple of (per-tool summary dict, cache_mode or ``None``,
        max_rounds or ``None``).
    :raises SystemExit: If no summary can be found.
    """
    # JSONL: scan for the summary record (it is written last).
    if log_path.endswith(".jsonl"):
        summary_rec = None
        with open(log_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "summary":
                    summary_rec = rec
        if summary_rec is None:
            raise SystemExit(f"no summary record in {log_path}. "
                             "Run the benchmark to completion first.")
        return (summary_rec["tools"], summary_rec.get("cache_mode"),
                summary_rec.get("max_rounds"))

    # Legacy text log.
    with open(log_path, encoding="utf-8") as handle:
        text = handle.read()
    start = text.find("===BENCHMARK_JSON===")
    end = text.find("===END_BENCHMARK_JSON===")
    if start == -1 or end == -1:
        raise SystemExit(f"no BENCHMARK_JSON block in {log_path}. "
                         "Run the benchmark to completion first.")
    payload = text[start + len("===BENCHMARK_JSON==="):end].strip()
    data = json.loads(payload)
    if isinstance(data, dict) and "tools" in data:
        return data["tools"], data.get("cache_mode"), data.get("max_rounds")
    return data, None, None


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
    summary, cache_mode, _ = load_summary(log_path)
    return plot(summary, log_path, cache_mode=cache_mode)



def _marker(i: int) -> str:
    """Return a line marker for the ``i``-th tool.

    :param i: Zero-based tool index.
    :return: A matplotlib marker character.
    """
    return "osD^v*"[i % 6]


def plot_sweep(sweep_dir: str) -> str:
    """Plot how each metric varies with the round budget ``n`` across a sweep.

    Reads every ``n{k}.log`` in ``sweep_dir`` (written by ``run_sweep.py``),
    then draws one figure with a panel per metric (accuracy, mean total tokens,
    mean latency, mean cost, mean rounds used); within each panel there is one
    line per tool plotted against ``n``.

    :param sweep_dir: Directory holding the per-n logs and ``sweep.json``.
    :return: Path to the saved comparison PNG.
    """
    logs = glob.glob(os.path.join(sweep_dir, "n*.jsonl")) or \
        glob.glob(os.path.join(sweep_dir, "n*.log"))
    logs = sorted(logs, key=lambda p: int(os.path.basename(p)[1:].split(".")[0]))
    if not logs:
        raise SystemExit(f"no n*.jsonl / n*.log files found in {sweep_dir}/")

    # n -> {tool -> summary}
    per_n = {}
    cache_mode = None
    for log_path in logs:
        n = int(os.path.basename(log_path)[1:].split(".")[0])
        summary, cm, _ = load_summary(log_path)
        per_n[n] = summary
        cache_mode = cache_mode or cm

    ns = sorted(per_n)
    tools = list(next(iter(per_n.values())).keys())

    # (metric key, panel title, scale factor, value format)
    metrics = [
        ("accuracy", "Accuracy (%)", 100.0, "%.0f"),
        ("mean_total_tokens", "Mean total tokens / q", 1.0, "%.0f"),
        ("mean_time_s", "Mean latency (s) / q", 1.0, "%.1f"),
        ("mean_cost_usd", "Mean cost ($) / q", 1.0, "%.5f"),
        ("mean_rounds", "Mean rounds used", 1.0, "%.2f"),
    ]

    cols = 3
    rows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    title = f"Round-budget sweep (n={ns[0]}..{ns[-1]})"
    if cache_mode:
        title += f"  (cache={cache_mode})"
    fig.suptitle(title, fontsize=14)

    for axis, (key, label, scale, fmt) in zip(axes, metrics):
        for ti, tool in enumerate(tools):
            ys = [per_n[n].get(tool, {}).get(key, 0) * scale for n in ns]
            axis.plot(ns, ys, marker=_marker(ti), color=_COLORS[ti % len(_COLORS)],
                      label=tool, linewidth=1.8)
            for x, y in zip(ns, ys):
                axis.annotate(fmt % y, (x, y), textcoords="offset points",
                              xytext=(0, 6), ha="center", fontsize=7)
        axis.set_title(label)
        axis.set_xlabel("n (max tool-call rounds)")
        axis.set_xticks(ns)
        axis.grid(alpha=0.3)
        if key == "accuracy":
            axis.set_ylim(0, 100)
    # Hide any unused panels.
    for axis in axes[len(metrics):]:
        axis.axis("off")

    # One shared legend.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=len(tools), fontsize=9)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    out_png = os.path.join(sweep_dir, "sweep_comparison.png")
    plt.savefig(out_png, dpi=130)
    print(f"saved sweep comparison -> {out_png}")
    plt.show()
    return out_png
