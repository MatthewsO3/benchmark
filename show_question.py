"""Render one question's full trace from a JSONL run into a readable Markdown file.

Reconstructs the old human-readable transcript view for a single question:
the prompts, every agent round (reasoning, tool call, full tool output), the
tool-free answerer call, and the final answer — for each tool that answered
that question in the run.

Usage::

    uv run show_question.py q21 logs/sweep_20261015_175650/n2.jsonl
    uv run show_question.py q21 logs/run_20261015_180000.jsonl --tool grep_tool
    uv run show_question.py q21 logs/.../n2.jsonl -o custom_name.md

The Markdown file is written next to the JSONL (same folder) unless ``-o`` is
given, and its path is printed.
"""
import os
import json
import argparse


def _load_records(jsonl_path: str):
    """Load all JSON records from a JSONL run file.

    :param jsonl_path: Path to the ``.jsonl`` run file.
    :return: Tuple of (meta dict or ``{}``, list of question records).
    """
    meta, questions = {}, []
    with open(jsonl_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = rec.get("type")
            if kind == "meta":
                meta = rec
            elif kind == "question":
                questions.append(rec)
    return meta, questions


def _fence(text: str) -> str:
    """Wrap text in a fenced code block, picking a fence that won't collide.

    :param text: The text to fence.
    :return: A Markdown fenced block.
    """
    text = text if text is not None else ""
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}\n{text}\n{fence}"


def _render_round(round_rec: dict) -> str:
    """Render one agent round as Markdown.

    :param round_rec: A single entry from a question record's ``rounds`` list.
    :return: Markdown text for the round.
    """
    n = round_rec.get("round")
    forced = round_rec.get("forced")
    choice = "forced tool" if forced else "auto tool"
    usage = round_rec.get("usage", {})
    lines = [f"#### Round {n} ({choice})", ""]
    lines.append(
        f"- tokens: input={usage.get('input')} output={usage.get('output')} "
        f"total={usage.get('total')} cached={usage.get('cached')} "
        f"cost=${usage.get('cost_usd', 0):.6f}")
    if round_rec.get("stopped_early"):
        lines.append("- **model stopped early (no tool call this round)**")
    lines.append("")
    lines.append("**Reasoning**")
    lines.append(_fence(round_rec.get("reasoning") or "(none)"))
    lines.append("")
    lines.append(f"**Tool call** `{round_rec.get('tool_name')}` "
                 f"args: `{json.dumps(round_rec.get('tool_args'))}`")
    lines.append("")
    output = round_rec.get("tool_output")
    if output is not None:
        lines.append(f"**Tool output** ({len(output)} chars)")
        lines.append(_fence(output))
    else:
        lines.append("**Tool output**: (none — no tool was called)")
    lines.append("")
    return "\n".join(lines)


def _render_record(rec: dict) -> str:
    """Render one question record (one tool's run) as Markdown.

    :param rec: A ``type="question"`` record.
    :return: Markdown text for the record.
    """
    usage = rec.get("usage_total", {})
    ans = rec.get("answerer", {})
    ans_usage = ans.get("usage", {})
    prompts = rec.get("prompts", {})

    correct = rec.get("correct")
    verdict = "[CORRECT]" if correct else ("[WRONG]" if correct is False else "[N/A]")

    lines = []
    lines.append(f"## Tool: `{rec.get('tool')}`  {verdict}")
    lines.append("")
    lines.append(f"- answer: **{rec.get('answer')!r}**  "
                 f"expected: **{rec.get('expected')!r}**  correct: **{correct}**")
    lines.append(f"- rounds used: {rec.get('rounds_used')} / {rec.get('max_rounds')}  "
                 f"cache: {rec.get('cache_mode')}")
    lines.append(f"- total tokens: input={usage.get('input')} output={usage.get('output')} "
                 f"total={usage.get('total')} cached={usage.get('cached')} "
                 f"cost=${usage.get('cost_usd', 0):.6f}  latency={rec.get('latency_s')}s")
    lines.append("")

    lines.append("### Prompts")
    lines.append("**Agent system prompt**")
    lines.append(_fence(prompts.get("agent_system")))
    lines.append("**Answerer system prompt**")
    lines.append(_fence(prompts.get("answerer_system")))
    lines.append("")

    lines.append("### Agent loop")
    rounds = rec.get("rounds") or []
    if not rounds:
        lines.append("(no agent rounds recorded)")
        lines.append("")
    for round_rec in rounds:
        lines.append(_render_round(round_rec))

    lines.append("### Answerer call (tool-free)")
    lines.append(
        f"- tokens: input={ans_usage.get('input')} output={ans_usage.get('output')} "
        f"total={ans_usage.get('total')} cached={ans_usage.get('cached')} "
        f"cost=${ans_usage.get('cost_usd', 0):.6f}")
    lines.append("")
    lines.append("**Context sent to answerer**")
    lines.append(_fence(ans.get("context")))
    lines.append("**Answerer raw output**")
    lines.append(_fence(ans.get("output")))
    lines.append("")
    lines.append(f"> **FINAL answer: `{rec.get('answer')}`**")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def render(question_id: str, jsonl_path: str, *, tool: str = None,
           out_path: str = None) -> str:
    """Render a question's trace to a Markdown file.

    :param question_id: The question id to render (e.g. ``"q21"``).
    :param jsonl_path: Path to the ``.jsonl`` run file.
    :param tool: Optional tool name to restrict output to a single tool.
    :param out_path: Optional output path; defaults to a file next to the JSONL.
    :return: Path to the written Markdown file.
    :raises SystemExit: If no matching record is found.
    """
    meta, questions = _load_records(jsonl_path)
    matches = [r for r in questions if r.get("id") == question_id]
    if tool:
        matches = [r for r in matches if r.get("tool") == tool]
    if not matches:
        raise SystemExit(
            f"no record for id={question_id!r}"
            + (f" tool={tool!r}" if tool else "")
            + f" in {jsonl_path}")

    question_text = matches[0].get("question", "")

    header = [
        f"# Question `{question_id}`",
        "",
        f"> {question_text}",
        "",
        f"- source: `{os.path.basename(jsonl_path)}`",
        f"- model: `{meta.get('model', '?')}`  cache: `{meta.get('cache_mode', '?')}`  "
        f"n (max rounds): `{meta.get('max_rounds', '?')}`",
        f"- tools answering this question: {', '.join('`' + r.get('tool', '?') + '`' for r in matches)}",
        "",
        "---",
        "",
    ]
    body = [_render_record(r) for r in matches]
    content = "\n".join(header) + "\n".join(body)

    if out_path is None:
        base = os.path.splitext(os.path.basename(jsonl_path))[0]
        suffix = f"_{tool}" if tool else ""
        out_path = os.path.join(os.path.dirname(os.path.abspath(jsonl_path)),
                                f"{base}_{question_id}{suffix}.md")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    print(f"wrote -> {out_path}")
    return out_path


def main() -> None:
    """Parse CLI arguments and render the question trace."""
    parser = argparse.ArgumentParser(
        description="Render one question's full trace from a JSONL run to Markdown.")
    parser.add_argument("question_id", help="question id to render, e.g. q21")
    parser.add_argument("jsonl", help="path to the .jsonl run file")
    parser.add_argument("--tool", default=None,
                        help="restrict to a single tool (grep_tool/lsp_tool/rag_tool/shell_tool)")
    parser.add_argument("--out", "-o", default=None, dest="out_path",
                        help="output .md path (default: next to the JSONL)")
    args = parser.parse_args()
    render(args.question_id, args.jsonl, tool=args.tool, out_path=args.out_path)


if __name__ == "__main__":
    main()
