# onlab — multi-round LangChain repo Q&A benchmark

Benchmarks three (now four) repo-search tools head-to-head on yes/no questions
about a target codebase. Each question runs a multi-round agent loop:

1. **Agent loop** (up to `n` rounds): round 1 *forces* the single tool under
   test; rounds 2..n are optional (`tool_choice="auto"`) so the model may call
   the same tool again or stop early. Only one tool schema is offered, so it can
   never call a different tool. Rounds actually used are counted as a metric.
2. **Answerer** (one tool-free call): a fresh session receives the question plus
   all accumulated reasoning and tool output, and returns exactly one word
   (`yes`/`no`).

Keeping the tool as the only variable makes accuracy differences attributable to
the tool. Metrics per tool: accuracy, token usage (in/out/total/cached), cost,
latency, and mean rounds used.

Tools benchmarked: `grep_tool` (ripgrep), `lsp_tool` (jedi via LSP),
`rag_tool` (FAISS + MiniLM embeddings), `shell_tool` (sandboxed read-only shell).

## Setup (uv)

```bash
uv sync
copy .env.example .env   # Windows  (cp .env.example .env on Linux/macOS)
```

Then edit `.env`:

- `OPENROUTER_API_KEY` — required (falls back to `OPENAI_API_KEY`).
- `TARGET_REPO` — path to the codebase the questions are about.
- `LLM_MODEL` — optional; a tool-calling, instruction-following model. The free
  default rate-limits and often ignores the yes/no format, so use a paid model
  for meaningful accuracy.

## Commands

All scripts are run with `uv run`. Paths below use forward slashes; on Windows
`logs\sweep_...` also works.

### `run_eval.py` — one benchmark run (single `n`)

Runs every tool over a question set at one round budget, prints the metrics
table, and writes a JSONL transcript.

```bash
uv run run_eval.py                                   # default questions.jsonl, n from config
uv run run_eval.py data/benchmark_questions.jsonl    # pick a question set
uv run run_eval.py data/benchmark_questions.jsonl -n 3
uv run run_eval.py data/benchmark_questions.jsonl -n 3 --cache off
uv run run_eval.py data/test.jsonl -n 2 -o logs/my_run.jsonl
```

Flags:

| flag | alias | default | meaning |
|------|-------|---------|---------|
| `questions` (positional) | | `questions.jsonl` | path to the `.jsonl` question set |
| `--rounds` | `-n` | `config.AGENT_ROUNDS` | max agent tool-call rounds `n` |
| `--cache` | | `config.CACHE_MODE` | `on` (stable prefix, cheaper re-runs) or `off` (nonce prefix, reproducible cold cost) |
| `--log` | `-o` | `logs/run_<timestamp>.jsonl` | transcript output path |

### `run_sweep.py` — sweep several `n` back-to-back

Runs `run_eval` once per `n` unattended, saving each to
`logs/sweep_<timestamp>/n{k}.jsonl` plus a `sweep.json` manifest (rewritten after
each `n`, so a crash still leaves usable data), then draws the comparison chart.

```bash
uv run run_sweep.py                                           # n=1..3, data/benchmark_questions.jsonl
uv run run_sweep.py data/benchmark_questions.jsonl --max-n 5
uv run run_sweep.py data/benchmark_questions.jsonl --max-n 5 --cache off
uv run run_sweep.py data/benchmark_questions.jsonl --ns 1 2 4 # explicit n values
uv run run_sweep.py data/test.jsonl --max-n 2 --no-plot
```

Flags:

| flag | default | meaning |
|------|---------|---------|
| `questions` (positional) | `data/benchmark_questions.jsonl` | path to the `.jsonl` question set |
| `--max-n` | `3` | run `n=1..max-n` (ignored if `--ns` is given) |
| `--ns` | — | explicit list of `n` values, e.g. `--ns 1 2 4` |
| `--cache` | `config.CACHE_MODE` | `on` / `off` (see above) |
| `--no-plot` | off | skip drawing the comparison chart at the end |

### `plot_results.py` — chart one run

Bar chart (accuracy, mean tokens, mean latency) for a single run file.

```bash
uv run plot_results.py                               # newest run in logs/
uv run plot_results.py logs/run_20261015_180000.jsonl
uv run plot_results.py logs/sweep_20261015_180407/n2.jsonl
```

Saves `<run>_benchmark.png` next to the input and opens a window. Accepts the
current `.jsonl` format and legacy `.log` files.

### `plot_sweep.py` — compare across `n`

Line chart with one panel per metric (accuracy, mean total tokens, mean latency,
mean cost, mean rounds used) and one line per tool vs `n`.

```bash
uv run plot_sweep.py                                 # newest sweep_* dir in logs/
uv run plot_sweep.py logs/sweep_20261015_180407
```

Reads every `n{k}.jsonl` in the directory and saves `sweep_comparison.png` there.

### `show_question.py` — readable trace for one question

Reconstructs a human-readable Markdown view of everything that happened for a
single question in a given run: prompts, each agent round (reasoning, tool call,
full tool output), the answerer call, and the final answer — per tool.

```bash
uv run show_question.py q13 logs/sweep_20261015_180407/n3.jsonl
uv run show_question.py q13 logs/sweep_20261015_180407/n3.jsonl --tool grep_tool
uv run show_question.py q13 logs/run_20261015_180000.jsonl -o trace.md
```

| arg / flag | alias | default | meaning |
|------------|-------|---------|---------|
| `question_id` (positional) | | — | question id to render, e.g. `q13` |
| `jsonl` (positional) | | — | path to the `.jsonl` run file |
| `--tool` | | all tools | restrict to one tool (`grep_tool`/`lsp_tool`/`rag_tool`/`shell_tool`) |
| `--out` | `-o` | `<run>_<id>.md` next to the JSONL | output Markdown path |

For a different `n`, point it at that `n`'s JSONL (e.g. `n2.jsonl`).

## Typical workflow

```bash
# 1. Smoke-test the pipeline on a tiny set
uv run run_sweep.py data/test.jsonl --max-n 2

# 2. Full sweep on the real question set
uv run run_sweep.py data/benchmark_questions.jsonl --max-n 4 --cache off

# 3. Inspect a surprising result
uv run show_question.py q13 logs/sweep_<ts>/n3.jsonl

# 4. Re-plot later without re-running
uv run plot_sweep.py logs/sweep_<ts>
```

## Output format

Each run is one JSONL file (one JSON object per line):

- `type:"meta"` — cache mode, `n`, questions path, total, model.
- `type:"question"` — one per question: both system prompts, every round's
  `{reasoning, tool_name, tool_args, full tool_output, usage, tool_error?}`,
  early-stop flag, the answerer's context + output, per-call and total usage,
  latency, and `id/expected/answer/correct`.
- `type:"summary"` — aggregated per-tool metrics (consumed by the plotters).

JSONL keeps records queryable, e.g. list wrong answers:

```bash
uv run python -c "import json; [print(r['id'],r['tool']) for l in open('logs/sweep_x/n2.jsonl',encoding='utf-8') for r in [json.loads(l)] if r.get('type')=='question' and r.get('correct') is False]"
```

## Package layout (`repoqa/`)

- `config.py` — backend, prompts, `TARGET_REPO`, `CACHE_MODE`, `AGENT_ROUNDS`,
  output token caps (`AGENT_MAX_TOKENS`, `ANSWERER_MAX_TOKENS`), embedding model.
- `llm.py` — chat-completions client (agent call via raw OpenAI SDK to capture
  the `reasoning` field; answerer via `ChatOpenAI`). Retries on rate limits,
  request timeouts, output token caps, cache-mode nonce.
- `usage.py` — token/cost accounting.
- `transcript.py` — JSONL transcript writer.
- `agent.py` — the multi-round agent loop + answerer, emits one JSONL record
  per question.
- `benchmark.py` — runs a question set, aggregates per-tool metrics.
- `plotting.py` — single-run and sweep charts.
- `tools/` — plain LangChain tools (NOT MCP): `grep.py`, `lsp.py`, `rag.py`,
  `shell.py` (+ `common.py` helpers).

## Config / environment variables

| var | default | meaning |
|-----|---------|---------|
| `OPENROUTER_API_KEY` | — | API key (falls back to `OPENAI_API_KEY`) |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `deepseek/deepseek-v4-flash-0731:free` | tool-calling model |
| `TARGET_REPO` | `.` | codebase the questions are about |
| `CACHE_MODE` | `on` | `on`/`off` prompt caching (see `--cache`) |
| `AGENT_ROUNDS` | `2` | default max tool-call rounds `n` |
| `AGENT_MAX_TOKENS` | `4096` | cap on agent *output* tokens (does not touch context) |
| `ANSWERER_MAX_TOKENS` | `512` | cap on answerer *output* tokens |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | RAG embedding model |

## Notes

- Backend: OpenRouter (OpenAI-compatible). Swap models via `LLM_MODEL`.
- `--cache off` prepends a per-run nonce so no prompt prefix is cached, giving
  reproducible cold cost/latency; `on` keeps a stable prefix for cheaper re-runs.
- Output token caps limit only generation, not the input/context, so they do not
  change results; they prevent the provider pre-authorizing the full context
  window (a source of spurious 402 errors) and bound cost.
- `shell_tool` is sandboxed: whitelisted read-only programs only, no
  pipes/redirects/chaining, every path argument confined to `TARGET_REPO`. On
  Windows some Unix coreutils (`cat`, Unix-style `find`) are unavailable, which
  can handicap it; a Linux/WSL environment gives it a fair shot.
- RAG embedding `all-MiniLM-L6-v2` downloads on first use and caches to
  `.rag_cache/` keyed on repo contents.
