# onlab — single-round LangChain repo Q&A

One-round agent loop. LLM (+ max 1 tool call) -> 1-word answer, then a fresh
tool-free session gets full context and returns a single word. Measures
accuracy (yes/no), token usage, and latency.

## Setup (uv)

```bash
uv sync
copy .env.example .env   # add OPENROUTER_API_KEY
```

## Run

```bash
uv run run_eval.py questions.jsonl
uv run run_eval.py questions.jsonl --cache off   # reproducible cold cost
uv run plot_results.py                           # chart the newest run log
```

## Files

- `run_eval.py`   — CLI: run the question set, print accuracy/tokens/cost/time
- `plot_results.py` — CLI: chart a run log
- `repoqa/`       — the package:
  - `config.py`      — backend, prompts, target repo, cache mode, embedding model
  - `llm.py`         — chat-completions client (agent + answerer calls)
  - `usage.py`       — token/cost accounting
  - `transcript.py`  — human-readable run log writer
  - `agent.py`       — two-call loop (forced tool -> tool-free answerer)
  - `benchmark.py`   — runs the question set and aggregates per-tool metrics
  - `plotting.py`    — benchmark chart from a run log
  - `tools/`         — plain LangChain tools (NOT MCP): `grep.py`, `lsp.py`, `rag.py`
- `questions.jsonl` — `{id, question, expected}` per line

## Notes

- Backend: OpenRouter (OpenAI-compatible endpoint). Default model
  `deepseek/deepseek-v4-flash-0731:free`. Swap via `LLM_MODEL` in `.env` or `repoqa/config.py`.
- Auth: set `OPENROUTER_API_KEY` in `.env` (falls back to `OPENAI_API_KEY`).
- `TARGET_REPO` points at the codebase the questions are about (set in `.env`).
- `CACHE_MODE` (`on`/`off`, or `--cache`) toggles prompt-cache-friendly prompts.
- RAG embedding: `all-MiniLM-L6-v2` (small local model). "ErlangBERT" is not a
  real model; for code use e.g. `jinaai/jina-embeddings-v2-base-code`.
