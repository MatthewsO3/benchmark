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
    """Render and save the comparison chart (accuracy, tokens, latency, tool time).

    :param summary: Per-tool summary dict (from :func:`load_summary`).
    :param log_path: Path to the source log (used for the title and output name).
    :param cache_mode: Cache mode to show in the title, if known.
    :return: Path to the saved PNG.
    """
    tools = list(summary.keys())
    accuracy = [summary[t]["accuracy"] * 100 for t in tools]
    mean_tokens = [summary[t]["mean_total_tokens"] for t in tools]
    mean_time = [summary[t]["mean_time_s"] for t in tools]
    mean_tool_time = [summary[t].get("mean_tool_time_s", 0) for t in tools]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    title = f"Tool benchmark — {os.path.basename(log_path)}"
    if cache_mode:
        title += f"  (cache={cache_mode})"
    fig.suptitle(title, fontsize=13)

    panels = [
        (axes[0], accuracy, "Accuracy (%)", "%.0f%%", (0, 100)),
        (axes[1], mean_tokens, "Mean total tokens / question", "%.0f", None),
        (axes[2], mean_time, "Mean latency (s) / question", "%.1fs", None),
        (axes[3], mean_tool_time, "Mean tool time (s) / question", "%.3fs", None),
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

    Reads every ``n{k}.jsonl`` in ``sweep_dir`` (written by ``run_sweep.py``),
    then draws one figure with a panel per metric; within each panel there is
    one line per tool plotted against ``n``. Accuracy is broken out into an
    **overall** panel plus **one panel per question category** found in the run
    (e.g. ``local_fact``, ``complex_reasoning``, ``cross_subsystem``), so you can
    see how each question type responds to more rounds; the remaining panels are
    mean total tokens, mean latency, mean tool time, mean cost, and mean rounds used.

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

    # Discover which question categories appear in the run (from any tool's
    # by_category block). Lets the chart adapt to whatever the question set
    # used; an un-categorized run simply yields no extra accuracy panels.
    present_categories = []
    for summary in per_n.values():
        for tool_summary in summary.values():
            for cat in (tool_summary.get("by_category") or {}):
                if cat not in present_categories:
                    present_categories.append(cat)
    # Stable, readable order; drop the "uncategorized" bucket from the
    # per-category panels (it is already covered by the overall panel).
    present_categories = [c for c in sorted(present_categories) if c != "uncategorized"]

    def _value(tool_summary, key, category):
        """Pull a metric from a tool's summary (flat key or category accuracy).

        :param tool_summary: One tool's summary dict for a given ``n``.
        :param key: The flat metric key, or ``"accuracy"`` when ``category`` set.
        :param category: A category name to read per-category accuracy for, or
            ``None`` for the flat/overall metric.
        :return: The numeric value (0 if missing).
        """
        if category is not None:
            entry = (tool_summary.get("by_category") or {}).get(category) or {}
            return entry.get("accuracy", 0)
        return tool_summary.get(key, 0)

    # (metric key, panel title, scale factor, value format, category-or-None).
    # Accuracy is split into one OVERALL panel plus one panel per category so
    # you can see how each question type responds to the round budget n.
    metrics = [("accuracy", "Accuracy — overall (%)", 100.0, "%.0f", None)]
    for cat in present_categories:
        metrics.append(("accuracy", f"Accuracy — {cat} (%)", 100.0, "%.0f", cat))
    metrics += [
        ("mean_total_tokens", "Mean total tokens / q", 1.0, "%.0f", None),
        ("mean_reasoning_tokens", "Mean reasoning tokens / q", 1.0, "%.0f", None),
        ("mean_time_s", "Mean latency (s) / q", 1.0, "%.1f", None),
        ("mean_tool_time_s", "Mean tool time (s) / q", 1.0, "%.3f", None),
        ("mean_cost_usd", "Mean cost ($) / q", 1.0, "%.5f", None),
        ("mean_tool_calls", "Mean tool calls / q", 1.0, "%.2f", None),
        ("mean_rounds", "Mean rounds used", 1.0, "%.2f", None),
    ]

    cols = 3
    rows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    title = f"Round-budget sweep (n={ns[0]}..{ns[-1]})"
    if cache_mode:
        title += f"  (cache={cache_mode})"
    fig.suptitle(title, fontsize=14)

    for axis, (key, label, scale, fmt, category) in zip(axes, metrics):
        for ti, tool in enumerate(tools):
            ys = [_value(per_n[n].get(tool, {}), key, category) * scale for n in ns]
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


# ---------------------------------------------------------------------------
# Ladder (codebase-scaling) plots
# ---------------------------------------------------------------------------

def _load_cell_summary(cell_path: str):
    """Return the per-tool summary dict from one ladder cell transcript.

    Each cell file is a benchmark JSONL run of exactly one (rung, tool); its
    ``summary`` record holds a single tool whose summary carries the metrics
    plus ladder metadata (``rung``, ``rung_size``, ``by_category``).

    :param cell_path: Path to the cell ``.jsonl``.
    :return: The single tool's summary dict, or ``None`` if absent.
    """
    summary_rec = None
    with open(cell_path, encoding="utf-8") as handle:
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
    if not summary_rec or not summary_rec.get("tools"):
        return None
    # One tool per cell; return its summary.
    return next(iter(summary_rec["tools"].values()))


def _collect_ladder(exp_dir: str):
    """Collect every (rung, tool) cell under a ladder experiment directory.

    Prefers the ``manifest.json`` index; falls back to globbing
    ``<rung>__<tool>.jsonl`` files if no manifest is present.

    :param exp_dir: The experiment directory (holds cells + manifest).
    :return: Tuple ``(cells, ladder_name)`` where ``cells`` is a list of dicts
        ``{rung, tool, ch_loc, ch_files, summary}`` and ``ladder_name`` is the
        ladder file name (or ``None``).
    :raises SystemExit: If no cells are found.
    """
    manifest_path = os.path.join(exp_dir, "manifest.json")
    cell_files = []
    ladder_name = None
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        ladder_name = manifest.get("ladder")
        for cell in manifest.get("cells", []):
            cell_files.append(os.path.join(exp_dir, cell["path"]))
    else:
        cell_files = sorted(glob.glob(os.path.join(exp_dir, "*__*.jsonl")))
    if not cell_files:
        raise SystemExit(f"no ladder cells (*.jsonl / manifest) found in {exp_dir}/")

    cells = []
    for path in cell_files:
        if not os.path.exists(path):
            continue
        summary = _load_cell_summary(path)
        if summary is None:
            continue
        rung_size = summary.get("rung_size") or {}
        cells.append({
            "rung": summary.get("rung"),
            "tool": summary.get("tool") or _tool_from_name(path),
            "ch_loc": rung_size.get("ch_loc") or summary.get("ch_loc"),
            "ch_files": rung_size.get("ch_files"),
            "summary": summary,
        })
    if not cells:
        raise SystemExit(f"no readable summaries in the cells under {exp_dir}/")
    return cells, ladder_name


def _tool_from_name(cell_path: str) -> str:
    """Infer the tool name from a ``<rung>__<tool>.jsonl`` filename.

    :param cell_path: Path to the cell file.
    :return: The tool portion of the name (best-effort).
    """
    base = os.path.splitext(os.path.basename(cell_path))[0]
    return base.split("__", 1)[1] if "__" in base else base


def _ladder_categories(cells: list) -> list:
    """Return the sorted question categories present across cells.

    :param cells: The collected cell dicts.
    :return: Category names (excluding the ``uncategorized`` bucket).
    """
    cats = []
    for c in cells:
        for cat in (c["summary"].get("by_category") or {}):
            if cat not in cats:
                cats.append(cat)
    return [c for c in sorted(cats) if c != "uncategorized"]


def plot_ladder(exp_dir: str, *, out_path: str = None) -> str:
    """Plot every metric vs codebase size, one line per tool, for a ladder run.

    The subsystem ladder grows the indexed codebase across rungs while asking
    the same questions, so the natural view is each metric as a function of
    codebase size (``ch_loc``), with one line per tool. Accuracy is broken out
    into an overall panel plus one panel per question category; the remaining
    panels cover token usage (total + reasoning), cost, latency, tool time and
    tool-call count — so the whole picture of "how does each tool scale as the
    haystack grows?" is one figure.

    :param exp_dir: The ladder experiment directory (cells + ``manifest.json``).
    :param out_path: Output PNG path; defaults to ``ladder_comparison.png`` in
        ``exp_dir``.
    :return: Path to the saved PNG.
    """
    cells, ladder_name = _collect_ladder(exp_dir)

    # Organize: tool -> sorted list of (ch_loc, rung, summary).
    tools = []
    by_tool = {}
    for c in cells:
        by_tool.setdefault(c["tool"], []).append(c)
        if c["tool"] not in tools:
            tools.append(c["tool"])
    tools.sort()
    for tool in tools:
        by_tool[tool].sort(key=lambda c: (c["ch_loc"] or 0))

    categories = _ladder_categories(cells)

    # Discover every sub-tool that was actually invoked across the run (e.g.
    # the search tool under test plus ``read_file``), so the per-sub-tool
    # call-count and time panels adapt to whatever tools the setup exposed.
    sub_tools = []
    for c in cells:
        for name in (c["summary"].get("per_tool") or {}):
            if name not in sub_tools:
                sub_tools.append(name)
    sub_tools.sort()

    # (summary key, panel title, scale, value format, category-or-None).
    # Accuracy first (overall + per category), then the full token breakdown
    # (input / output / cached / reasoning / total), cost, wall-clock latency,
    # rounds, and aggregate tool-use. Per-sub-tool call/time panels are appended
    # afterwards so every metric the benchmark records is visualized.
    metrics = [("accuracy", "Accuracy — overall (%)", 100.0, "%.0f", None)]
    for cat in categories:
        metrics.append(("accuracy", f"Accuracy — {cat} (%)", 100.0, "%.0f", cat))
    metrics += [
        ("mean_input_tokens", "Mean input tokens / q", 1.0, "%.0f", None),
        ("mean_output_tokens", "Mean output tokens / q", 1.0, "%.0f", None),
        ("mean_reasoning_tokens", "Mean reasoning tokens / q", 1.0, "%.0f", None),
        ("mean_cached_tokens", "Mean cached tokens / q", 1.0, "%.0f", None),
        ("mean_total_tokens", "Mean total tokens / q", 1.0, "%.0f", None),
        ("mean_cost_usd", "Mean cost ($) / q", 1.0, "%.5f", None),
        ("mean_time_s", "Mean wall-clock latency (s) / q", 1.0, "%.1f", None),
        ("mean_rounds", "Mean rounds used / q", 1.0, "%.2f", None),
        ("mean_tool_calls", "Mean tool calls / q (all tools)", 1.0, "%.2f", None),
        ("mean_tool_time_s", "Mean tool time (s) / q (all tools)", 1.0, "%.3f", None),
    ]
    # Per-sub-tool breakdown: how many calls and how much wall time each
    # individual tool (search tool, read_file, ...) accounted for, averaged
    # per question. ``("__subtool__", name, field)`` is a sentinel the value
    # reader below understands.
    for name in sub_tools:
        metrics.append(("__subtool__", f"{name}: calls / q", 1.0, "%.2f",
                        (name, "calls")))
        metrics.append(("__subtool__", f"{name}: time (s) / q", 1.0, "%.3f",
                        (name, "time_s")))

    def _value(summary, key, category):
        """Pull a metric from a summary.

        Handles three cases: per-category accuracy (``category`` is a string),
        a per-sub-tool mean (``key == "__subtool__"`` and ``category`` is a
        ``(tool_name, field)`` tuple, divided by ``n`` to get a per-question
        mean), and a flat top-level metric otherwise.
        """
        if key == "__subtool__":
            name, field = category
            entry = (summary.get("per_tool") or {}).get(name) or {}
            n = summary.get("n") or 1
            return entry.get(field, 0) / n
        if category is not None:
            entry = (summary.get("by_category") or {}).get(category) or {}
            return entry.get("accuracy", 0)
        return summary.get(key, 0)

    cols = 3
    rows = (len(metrics) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten()

    title = "Ladder — metrics vs codebase size"
    if ladder_name:
        title += f"  ({ladder_name})"
    fig.suptitle(title, fontsize=15)

    for axis, (key, label, scale, fmt, category) in zip(axes, metrics):
        for ti, tool in enumerate(tools):
            series = by_tool[tool]
            xs = [c["ch_loc"] for c in series]
            ys = [_value(c["summary"], key, category) * scale for c in series]
            axis.plot(xs, ys, marker=_marker(ti), color=_COLORS[ti % len(_COLORS)],
                      label=tool, linewidth=1.9, markersize=7)
            for x, y in zip(xs, ys):
                axis.annotate(fmt % y, (x, y), textcoords="offset points",
                              xytext=(0, 7), ha="center", fontsize=7)
        axis.set_title(label)
        axis.set_xlabel("codebase size (lines of C/H)")
        axis.set_xscale("log")  # rungs span ~145k -> 1M -> 4M; log reads better
        axis.grid(alpha=0.3, which="major")
        if key == "accuracy":
            axis.set_ylim(0, 100)
        # Show ONLY the rung ticks (suppress matplotlib's dense log minor/major
        # ticks, which otherwise collide with the rung labels).
        all_x = sorted({c["ch_loc"] for c in cells if c["ch_loc"]})
        if all_x:
            from matplotlib.ticker import NullLocator
            axis.xaxis.set_minor_locator(NullLocator())
            axis.set_xticks(all_x)
            rung_by_x = {c["ch_loc"]: c["rung"] for c in cells if c["ch_loc"]}
            axis.set_xticklabels(
                [f"{rung_by_x.get(x, '')}\n{x/1e6:.2f}M LOC" for x in all_x],
                fontsize=8)
            # A little horizontal padding so edge markers/labels aren't clipped.
            axis.set_xlim(min(all_x) * 0.8, max(all_x) * 1.25)

    for axis in axes[len(metrics):]:
        axis.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=max(1, len(tools)), fontsize=10)

    plt.tight_layout(rect=(0, 0.02, 1, 0.95))
    out_path = out_path or os.path.join(exp_dir, "ladder_comparison.png")
    plt.savefig(out_path, dpi=130)
    print(f"saved ladder plot -> {out_path}")
    plt.close(fig)
    return out_path
