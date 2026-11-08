"""Run one ladder rung over selected tools, into a stable experiment directory.

The subsystem ladder (``ladder-fixed.yaml``) asks the SAME fixed question set at
each rung while the indexed codebase grows, isolating codebase size as the only
variable. This runner executes ONE rung for ONE-or-more tools, writing a
self-describing JSONL transcript per (rung, tool) cell plus a ``manifest.json``
that indexes every completed cell — so cells accumulate into one experiment
directory and can be aggregated later.

Containment: the rung's subsystems are resolved to an exact set of ``.c``/``.h``
files and installed as the process-wide tool scope; any file outside it is
invisible to the tools (treated as nonexistent), so a reduced rung is
indistinguishable from a kernel that only contained those subsystems.

Examples::

    # ladder_1 with only the online grep tool (the first planned run)
    uv run run_ladder.py --rung ladder_1 --tools grep_tool

    # add another tool/rung into the SAME experiment dir (accumulates)
    uv run run_ladder.py --rung ladder_1 --tools rg_index_tool \\
        --out-dir logs/ladder_fixed

    # dry run: resolve + report scope and questions, make NO API calls
    uv run run_ladder.py --rung ladder_1 --tools grep_tool --dry-run
"""
import os
import json
import argparse
from datetime import datetime

from repoqa import config, configure_logging
from repoqa.benchmark import run_benchmark, load_questions
from repoqa import ladder as L

#: Default ladder / subsystems / question files (the kernel QA data set).
_DATA_DIR = "data/linux-kernel-qa"
_DEFAULT_LADDER = os.path.join(_DATA_DIR, "ladder-fixed.yaml")
_DEFAULT_SUBSYSTEMS = os.path.join(_DATA_DIR, "subsystems.yaml")
_DEFAULT_QUESTIONS = os.path.join(_DATA_DIR, "linux-kernel-qa.jsonl")


def _manifest_path(out_dir: str) -> str:
    """Return the manifest path for an experiment directory.

    :param out_dir: The experiment output directory.
    :return: Path to ``manifest.json`` inside it.
    """
    return os.path.join(out_dir, "manifest.json")


def _load_manifest(out_dir: str) -> dict:
    """Load the experiment manifest, or start a fresh one.

    :param out_dir: The experiment output directory.
    :return: The manifest dict (``{"cells": [...]}`` at minimum).
    """
    path = _manifest_path(out_dir)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    return {"ladder": None, "created": datetime.now().isoformat(timespec="seconds"),
            "cells": []}


def _save_manifest(out_dir: str, manifest: dict) -> None:
    """Write the manifest atomically.

    :param out_dir: The experiment output directory.
    :param manifest: The manifest dict to persist.
    """
    path = _manifest_path(out_dir)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    os.replace(tmp, path)


def _upsert_cell(manifest: dict, cell: dict) -> None:
    """Insert or replace a (rung, tool) cell record in the manifest.

    A re-run of the same (rung, tool) overwrites its prior entry so the manifest
    always points at the latest transcript for that cell.

    :param manifest: The manifest dict (mutated in place).
    :param cell: The cell record to upsert.
    """
    key = (cell["rung"], cell["tool"])
    manifest["cells"] = [c for c in manifest["cells"]
                         if (c["rung"], c["tool"]) != key]
    manifest["cells"].append(cell)


def main() -> None:
    """Parse arguments, resolve the rung, run the tools, update the manifest."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rung", required=True,
                        help="rung name from the ladder file, e.g. ladder_1")
    parser.add_argument("--tools", nargs="+", default=["grep_tool"],
                        help="which tools to run (default: grep_tool)")
    parser.add_argument("--ladder", default=_DEFAULT_LADDER,
                        help=f"ladder YAML (default: {_DEFAULT_LADDER})")
    parser.add_argument("--subsystems", default=_DEFAULT_SUBSYSTEMS,
                        help=f"subsystems YAML (default: {_DEFAULT_SUBSYSTEMS})")
    parser.add_argument("--questions", default=_DEFAULT_QUESTIONS,
                        help=f"question .jsonl (default: {_DEFAULT_QUESTIONS})")
    parser.add_argument("--out-dir", default=None,
                        help="experiment dir to accumulate cells into "
                             "(default: logs/ladder_fixed)")
    parser.add_argument("--cache", default=None, choices=["on", "off"],
                        help="prompt cache mode (default: config.CACHE_MODE)")
    parser.add_argument("--max-rounds", type=int, default=None,
                        help="open-loop safety cap (default: config.AGENT_MAX_ROUNDS)")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve scope + questions and report; make NO API calls")
    parser.add_argument("--limit", type=int, default=None,
                        help="run only the first N of the rung's questions "
                             "(trial run); default: all")
    args = parser.parse_args()

    configure_logging()

    # Resolve the rung: its scope files and its question subset.
    subsystems = L.load_subsystems(args.subsystems)
    subsystems_ordered = L.load_subsystems_ordered(args.subsystems)
    rung = L.get_rung(args.ladder, args.rung)
    scope_files = L.resolve_rung_files(config.TARGET_REPO, rung,
                                       subsystems, subsystems_ordered)
    all_questions = load_questions(args.questions)
    rung_questions = L.filter_questions(all_questions, rung)
    if args.limit is not None:
        rung_questions = rung_questions[:args.limit]

    rung_size = rung.get("size", {})
    out_dir = args.out_dir or os.path.join("logs", os.path.splitext(
        os.path.basename(args.ladder))[0].replace("-", "_"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Ladder file     : {args.ladder}")
    print(f"Rung            : {rung['name']}")
    print(f"Subsystems ({len(rung.get('subsystems', []))}) : {rung.get('subsystems')}")
    print(f"Scope files     : {len(scope_files)} "
          f"(declared ch_files = {rung_size.get('ch_files', '?')})")
    print(f"Questions       : {len(rung_questions)} "
          f"(declared = {len(rung.get('questions', []))}"
          f"{f'; limited to first {args.limit}' if args.limit is not None else ''})")
    print(f"Tools           : {args.tools}")
    print(f"Target repo     : {config.TARGET_REPO}")
    print(f"Experiment dir  : {out_dir}")
    print(f"Model           : {config.LLM_MODEL}")

    if args.dry_run:
        print("\n[dry-run] scope + questions resolved; no tools run, no API calls.")
        # Sanity: confirm questions' subsystems are all inside the rung (so they
        # are answerable from this reduced tree).
        rung_subs = set(rung.get("subsystems", []))
        unanswerable = [q["id"] for q in rung_questions
                        if not set(q.get("subsystems", [])) <= rung_subs]
        if unanswerable:
            print(f"[dry-run] WARNING: {len(unanswerable)} question(s) reference "
                  f"subsystems outside the rung: {unanswerable[:10]}")
        else:
            print("[dry-run] all questions are answerable within the rung scope.")
        return

    if not config.LLM_API_KEY:
        raise SystemExit("LLM_API_KEY is not set (put it in .env). Aborting before "
                         "any API calls.")

    # Install the rung scope so every tool is confined to it.
    L.activate_scope(scope_files, label=rung["name"])

    meta_extra = {
        "ladder": os.path.basename(args.ladder),
        "rung": rung["name"],
        "rung_subsystems": rung.get("subsystems", []),
        "rung_size": rung_size,
        "scoped_files": len(scope_files),
    }

    manifest = _load_manifest(out_dir)
    manifest["ladder"] = os.path.basename(args.ladder)

    # Run each tool as its own cell -> its own transcript file, updating the
    # manifest after each completes (crash-safe: a dead run leaves the manifest
    # pointing only at finished cells).
    for tool_name in args.tools:
        cell_path = os.path.join(out_dir, f"{rung['name']}__{tool_name}.jsonl")
        print(f"\n==== CELL: rung={rung['name']} tool={tool_name} -> {cell_path} ====")
        summaries = run_benchmark(
            questions=rung_questions,
            tools=[tool_name],
            cache_mode=args.cache,
            max_rounds=args.max_rounds,
            log_path=cell_path,
            meta_extra=meta_extra,
        )
        summary = summaries.get(tool_name, {})
        _upsert_cell(manifest, {
            "rung": rung["name"],
            "tool": tool_name,
            "path": os.path.relpath(cell_path, out_dir),
            "ch_loc": rung_size.get("ch_loc"),
            "ch_files": rung_size.get("ch_files"),
            "scoped_files": len(scope_files),
            "n": summary.get("n"),
            "accuracy": summary.get("accuracy"),
            "finished": datetime.now().isoformat(timespec="seconds"),
        })
        _save_manifest(out_dir, manifest)

    L.clear_scope()
    print(f"\nManifest updated: {_manifest_path(out_dir)}")


if __name__ == "__main__":
    main()
