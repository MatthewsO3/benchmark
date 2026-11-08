# onlab — repo-search tool benchmark (subsystem ladder)

Benchmarks repo-search tools head-to-head on yes/no questions about a target
codebase. Each question runs an **open agent loop**: the model is offered a
single search tool and calls it as many times as it wants (up to a safety cap),
then a tool-free **answerer** call returns exactly one word (`yes`/`no`).
Keeping the tool as the only variable makes accuracy differences attributable to
the tool.

The flagship experiment is the **subsystem ladder**: the SAME fixed question set
is asked at each rung while the indexed codebase grows (e.g. ~145k → ~1M → ~4M
lines of C/H code). This isolates *codebase size* as the only variable, so any
change in accuracy or cost across rungs is caused by scale alone.

Tools benchmarked (`repoqa/tools/`):

| tool | kind | what it is |
|------|------|------------|
| `grep_tool` | lexical, online | GNU `grep -rni`, live scan, no index |
| `rg_index_tool` | lexical, indexed | ripgrep over a prebuilt file list (warmup builds the index) |
| `lsp_tool` | semantic | clangd `workspace/symbol` (+ `didOpen` fallback) for C/C++ |
| `rag_tool` | semantic | FAISS + MiniLM embeddings |
| `shell_tool` | agentic | sandboxed read-only shell (`cat`/`head`/`tail`/…) |

The three search tools (`grep`/`rg_index`/`lsp`/`rag`) are additionally offered a
shared `read_file` tool so the model can open files their results point to.
`shell_tool` is not — it can already read files natively.

## Setup (uv)

```bash
uv sync                 # installs deps incl. the `clangd` wheel for lsp_tool
cp .env.example .env     # then edit .env (see below)
```

Edit `.env`:

- `LLM_API_KEY` — required (falls back to `OPENROUTER_API_KEY` / `OPENAI_API_KEY`).
- `LLM_BASE_URL` — OpenAI-compatible endpoint (e.g. a self-hosted vLLM server).
- `LLM_MODEL` — a tool-calling, instruction-following model.
- `TARGET_REPO` — path to the codebase the questions are about
  (e.g. `/home/mczap/benchmark/kernel/linux-7.2.9`).
- `SOURCE_EXTENSIONS` — file types in scope (`.c,.h` for the kernel runs).

## Running the ladder

The ladder is driven by three data files in `data/linux-kernel-qa/`:

- `ladder-fixed.yaml` — the rungs (`ladder_1`, `ladder_2`, `ladder_3`), each a
  growing set of subsystems asking the same 100 questions.
- `subsystems.yaml` — the 74 subsystems, their file-path patterns (priority
  ordered), and sizes.
- `linux-kernel-qa.jsonl` — the question set (q1..q414; each rung uses the
  100-question subset answerable within rung 1).

### `run_ladder.py` — run one rung over one or more tools

Resolves the rung's subsystems to an exact set of `.c`/`.h` files, installs that
as the **process-wide tool scope** (files outside it are invisible — treated as
nonexistent, so a reduced rung is indistinguishable from a smaller kernel), then
runs each tool and writes:

- one JSONL transcript per cell: `<rung>__<tool>.jsonl`
- a `manifest.json` indexing every completed cell (updated after each cell, so a
  crash leaves the manifest pointing only at finished cells).

```bash
# dry run first: resolve scope + questions, report, make NO API calls
uv run run_ladder.py --rung ladder_1 --tools grep_tool --dry-run

# ladder_1 with the online grep tool, into a stable experiment dir
uv run run_ladder.py --rung ladder_1 --tools grep_tool --out-dir logs/ladder_fixed

# add another tool (or rung) into the SAME dir — cells accumulate
uv run run_ladder.py --rung ladder_1 --tools rg_index_tool --out-dir logs/ladder_fixed
```

Flags:

| flag | default | meaning |
|------|---------|---------|
| `--rung` (required) | — | rung name from the ladder file, e.g. `ladder_1` |
| `--tools` | `grep_tool` | one or more tools to run (space-separated) |
| `--out-dir` | `logs/ladder_fixed` | experiment dir to accumulate cells into |
| `--ladder` | `data/linux-kernel-qa/ladder-fixed.yaml` | ladder YAML |
| `--subsystems` | `data/linux-kernel-qa/subsystems.yaml` | subsystems YAML |
| `--questions` | `data/linux-kernel-qa/linux-kernel-qa.jsonl` | question `.jsonl` |
| `--cache` | `config.CACHE_MODE` | `on` / `off` prompt caching |
| `--max-rounds` | `config.AGENT_MAX_ROUNDS` | open-loop safety cap (default 20) |
| `--limit N` | all | run only the first N questions (trial run) |
| `--dry-run` | off | resolve + report scope/questions; no API calls |

### Running all three rungs for one tool

Each rung is ~hours of wall-clock time, so run them in a detached `tmux` session
and tee the output to a log. Note the brace grouping — without it the pipe binds
only to the last command:

```bash
tmux new -s ladder_grep
{ uv run run_ladder.py --rung ladder_1 --tools grep_tool --out-dir logs/ladder_fixed \
  && uv run run_ladder.py --rung ladder_2 --tools grep_tool --out-dir logs/ladder_fixed \
  && uv run run_ladder.py --rung ladder_3 --tools grep_tool --out-dir logs/ladder_fixed ; } \
  2>&1 | tee logs/ladder_grep.out
# detach: Ctrl-b d   |   reattach: tmux attach -t ladder_grep
```

Point all tools at the **same** `--out-dir` so they land in one experiment
directory and can be plotted together. The typical plan: run `grep_tool` for all
three rungs first, then come back and run `rg_index_tool` (and later
`lsp_tool` / `rag_tool` / `shell_tool`) into the same dir.

### Watching progress

A rung only writes its `summary` record and updates `manifest.json` when it
finishes, so during a run check the per-question records directly:

```bash
# how many questions done in the current cell
grep -c '"type": "question"' logs/ladder_fixed/ladder_1__grep_tool.jsonl

# running accuracy from partial records
uv run python -c "import json; rs=[json.loads(l) for l in open('logs/ladder_fixed/ladder_1__grep_tool.jsonl',encoding='utf-8')]; qs=[r for r in rs if r.get('type')=='question']; print(sum(r.get('correct') is True for r in qs),'/',len(qs))"
```

## Plotting

### `plot_ladder.py` — metrics vs codebase size

Draws every metric as a function of codebase size (the ladder rungs), one line
per tool: accuracy (overall + per category), tokens, reasoning tokens, cost,
latency, tool time, and tool-call count. Reads the experiment dir's cells +
`manifest.json`.

```bash
uv run plot_ladder.py logs/ladder_fixed     # a specific experiment dir
uv run plot_ladder.py logs                  # every experiment dir under logs/
uv run plot_ladder.py                        # newest experiment dir under logs/
```

Saves `ladder_comparison.png` inside the experiment directory. Run it after at
least one rung has finished (a cell with no `summary` record yet is skipped).

## Output format

Each cell is one JSONL file (one JSON object per line):

- `type:"meta"` — cache mode, questions path, model, and ladder metadata
  (`ladder`, `rung`, `rung_subsystems`, `rung_size`, `scoped_files`).
- `type:"question"` — one per question: both system prompts, every agent round's
  `{reasoning, tool_name, tool_args, full tool_output, usage, tool_error?}`,
  whether the safety cap was hit, the answerer's context + output, per-call and
  total usage, latency, per-tool-call time, and `id/expected/answer/correct`.
- `type:"summary"` — aggregated per-tool metrics, including per-category
  breakdowns (`local_fact` / `complex_reasoning` / `cross_subsystem`). Consumed
  by `plot_ladder.py`.

List wrong answers in a cell:

```bash
uv run python -c "import json; [print(r['id']) for l in open('logs/ladder_fixed/ladder_1__grep_tool.jsonl',encoding='utf-8') for r in [json.loads(l)] if r.get('type')=='question' and r.get('correct') is False]"
```

## Package layout (`repoqa/`)

- `config.py` — backend, prompts, `TARGET_REPO`, `SOURCE_EXTENSIONS`,
  `CACHE_MODE`, `AGENT_MAX_ROUNDS`, `LLM_TIMEOUT_S`, output token caps, embedding
  model.
- `llm.py` — chat-completions client (agent call via raw OpenAI SDK to capture
  the reasoning field via `reasoning_content` / `<think>`; answerer via
  `ChatOpenAI`). Retries on rate limits / timeouts.
- `usage.py` — token/cost accounting (`TokenUsage`: input/output/total/cached/
  cache_write/reasoning/uncached_input/cost_usd).
- `transcript.py` — JSONL transcript writer.
- `agent.py` — the open agent loop + answerer, emits one JSONL record per
  question with per-tool-call token and time metrics.
- `benchmark.py` — runs a question set, aggregates per-tool and per-category
  metrics.
- `ladder.py` — loads the ladder/subsystems, builds the file partition, resolves
  a rung to its exact file set, filters questions, and manages the process-wide
  tool scope (`activate_scope`/`clear_scope`/`in_scope`).
- `plotting.py` — `plot_ladder` (metrics vs codebase size).
- `tools/` — plain LangChain tools (NOT MCP): `grep.py`, `rg_index.py`,
  `lsp.py`, `rag.py`, `shell.py`, `read.py` (+ `common.py` helpers).

## Config / environment variables

| var | default | meaning |
|-----|---------|---------|
| `LLM_API_KEY` | — | API key (falls back to `OPENROUTER_API_KEY` / `OPENAI_API_KEY`) |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | — | tool-calling model |
| `LLM_TIMEOUT_S` | `600` | per-request timeout (seconds) |
| `TARGET_REPO` | `.` | codebase the questions are about |
| `SOURCE_EXTENSIONS` | `.c,.h` | file extensions in scope |
| `CACHE_MODE` | `on` | `on`/`off` prompt caching |
| `AGENT_MAX_ROUNDS` | `20` | open-loop safety cap on tool-call rounds |
| `AGENT_MAX_TOKENS` | `4096` | cap on agent *output* tokens |
| `ANSWERER_MAX_TOKENS` | `512` | cap on answerer *output* tokens |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | RAG embedding model |

## Notes

- **Open loop**: the agent is never forced to call the tool; it signals "done" by
  returning no tool call. `AGENT_MAX_ROUNDS` is only a safety cap.
- **Scope containment**: out-of-scope file reads return `"read error: no such
  file"` — identical to a genuine miss — so a reduced rung behaves exactly like a
  smaller kernel.
- `lsp_tool` uses the `clangd` PyPI wheel (installed by `uv sync`); it falls back
  to a `clangd` on `PATH`.
- RAG embedding `all-MiniLM-L6-v2` downloads on first use and caches keyed on repo
  contents. On a shared GPU box, set `EMBED_DEVICE=cpu` to avoid contending for
  the GPU (slower).
- `shell_tool` is sandboxed: whitelisted read-only programs only, no
  pipes/redirects/chaining, every path argument confined to `TARGET_REPO`.
