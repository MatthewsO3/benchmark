"""Central configuration for the benchmark.

Every value can be overridden with an environment variable (loaded from a
local ``.env`` file). Import this module and read the module-level constants.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# -- LLM backend (OpenAI-compatible endpoint) --------------------------------

#: Base URL of the OpenAI-compatible chat-completions endpoint.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
#: API key for the endpoint. Prefers the backend-neutral ``LLM_API_KEY``; falls
#: back to ``OPENROUTER_API_KEY`` then ``OPENAI_API_KEY`` for older configs.
#: The key is sent as a Bearer token to whatever ``LLM_BASE_URL`` points at, so
#: a self-hosted OpenAI-compatible server (e.g. vLLM) works the same way.
LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("OPENROUTER_API_KEY")
    or os.getenv("OPENAI_API_KEY", "")
)
#: Model identifier (tool-calling capable). Override in ``.env``.
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-v4-flash-0731:free")
#: Sampling temperature. Kept at 0 for reproducible benchmarking.
LLM_TEMPERATURE = 0.0
#: Per-request timeout in seconds. Generous by default because the target model
#: is a reasoning model whose long chain-of-thought can make a single call take
#: minutes on a hard question; the timeout only exists to turn a genuinely stuck
#: call (dead server, hung socket, GPU OOM) into a loud error rather than an
#: infinite hang during an unattended run — not to cut off slow-but-progressing
#: generation. Override via ``LLM_TIMEOUT_S`` in ``.env``.
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_S", "600"))

# -- Agent loop ---------------------------------------------------------------

#: Safety cap on the number of agent tool-call rounds in the OPEN loop. The
#: agent keeps calling tools until it decides it is done (it stops emitting tool
#: calls), then the final answerer/verifier call runs. This cap only bounds a
#: runaway/looping model so a single question cannot spin forever or run up
#: unbounded cost; a well-behaved agent stops well before it. Override via
#: ``AGENT_MAX_ROUNDS`` in ``.env`` or the ``--max-rounds`` CLI flag.
AGENT_MAX_ROUNDS = int(os.getenv("AGENT_MAX_ROUNDS", "40"))

#: Backwards-compatible alias. The loop used to run a fixed number of rounds
#: (``AGENT_ROUNDS``); it is now open-ended, so this just seeds the safety cap
#: when ``AGENT_MAX_ROUNDS`` is not set explicitly.
AGENT_ROUNDS = int(os.getenv("AGENT_ROUNDS", str(AGENT_MAX_ROUNDS)))

# -- Output token caps --------------------------------------------------------

#: Cap on *generated* tokens per call. This never limits the input/context fed
#: to the model — only how much it may produce — so it does not change results
#: as long as it stays above what the model actually generates. It mainly stops
#: the provider from pre-authorizing the model's full context window (which can
#: trigger spurious 402 "insufficient credits" errors) and bounds cost.
#: Set generously; raise via ``.env`` if you ever see truncated output.
AGENT_MAX_TOKENS = int(os.getenv("AGENT_MAX_TOKENS", "4096"))
#: Cap for the answerer call, which only needs a single ``yes``/``no`` word
#: (kept well above that to tolerate a verbose/non-compliant model).
ANSWERER_MAX_TOKENS = int(os.getenv("ANSWERER_MAX_TOKENS", "512"))

# -- Target repository --------------------------------------------------------

#: Path to the codebase the questions are about.
TARGET_REPO = os.getenv("TARGET_REPO", ".")

# -- Source language / file scope ---------------------------------------------

#: Source file extensions the tools consider, as a comma-separated env override
#: (e.g. ``SOURCE_EXTENSIONS=".c,.h"``). This is the single place that defines
#: "what counts as a source file" for grep, RAG and the fallback scan; the code
#: is otherwise language-agnostic. Defaults to C (``.c``/``.h``) to match the
#: Linux-kernel question set, whose scope is explicitly C sources and headers
#: only. For the old Python target set this to ``.py``.
SOURCE_EXTENSIONS = tuple(
    ext if ext.startswith(".") else f".{ext}"
    for ext in (
        e.strip() for e in os.getenv("SOURCE_EXTENSIONS", ".c,.h").split(",")
    )
    if ext.strip()
)

#: ripgrep glob(s) mirroring :data:`SOURCE_EXTENSIONS` (``*.c``, ``*.h`` ...),
#: used to restrict the lexical search to source files.
SOURCE_GLOBS = tuple(f"*{ext}" for ext in SOURCE_EXTENSIONS)

#: Language id for the clangd/LSP tool. Informational for now (clangd infers
#: C/C++ from file contents); kept so the LSP backend is not hardcoded.
SOURCE_LANGUAGE = os.getenv("SOURCE_LANGUAGE", "c")

# -- Prompt caching -----------------------------------------------------------

#: ``"on"`` keeps a stable prompt prefix so the provider can serve repeated
#: system-prompt/tool-schema tokens from cache (cheaper, faster, but cost
#: drifts down across re-runs). ``"off"`` prepends a unique nonce to every
#: prompt so no prefix ever matches, giving reproducible "cold" cost/latency.
#: Override via ``CACHE_MODE`` in ``.env`` or the ``--cache`` CLI flag.
CACHE_MODE = os.getenv("CACHE_MODE", "on").lower()

# -- RAG ----------------------------------------------------------------------

#: Sentence-transformers embedding model used by the RAG tool.
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

#: Device the RAG embedding model runs on. ``"auto"`` picks CUDA when a GPU is
#: available (falling back to CPU), which is strongly preferred for a large
#: codebase — embedding the whole kernel on CPU is hours, on GPU is minutes.
#: Override with ``EMBED_DEVICE=cpu`` to force CPU (e.g. when the GPU is busy),
#: or ``EMBED_DEVICE=cuda``/``cuda:0`` to pin a specific device.
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "auto")

#: Batch size for the embedding ``encode`` call. Larger batches raise GPU
#: throughput at the cost of VRAM; 256 is a safe default for a small model like
#: MiniLM on a mid-range GPU. Raise (e.g. 512/1024) on a free, larger GPU.
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "256"))

#: Chunking parameters for the RAG index. These are folded into the index cache
#: key (see ``RagIndex._fingerprint``), so changing either here rebuilds the
#: index at the new chunk size instead of silently reusing chunks cut at the
#: old one — otherwise any chunk-size comparison would be comparing against a
#: stale index. ``CHUNK_MAX_CHARS`` is deliberately kept small enough that a
#: chunk fits inside the embedding model's max sequence length (MiniLM
#: truncates at 256 tokens); the RAG build warns if chunks look too large.
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "800"))
#: Number of trailing lines repeated into the next chunk (boundary overlap).
CHUNK_OVERLAP_LINES = int(os.getenv("CHUNK_OVERLAP_LINES", "3"))

# -- Prompts ------------------------------------------------------------------

#: System prompt for the agent call. Kept stable (identical across all
#: questions) so the provider can serve it from the prompt/KV cache as a shared
#: prefix — the per-question content (the question, tool results) always comes
#: AFTER this in the message history, never before, so the cached prefix holds.
#: Do not prepend per-question or retrieved content ahead of this text.
SYSTEM_PROMPT = (
    "You are a precise software engineer gathering evidence about a large C "
    "codebase. Each question has a definite yes/no answer that is verifiable "
    "by reading the source. A separate verifier will decide the final yes/no "
    "from the evidence you collect — your job is to FIND and REPORT the "
    "relevant source, not to state the verdict.\n\n"
    "You have one search tool to gather evidence; some setups also give you a "
    "read_file tool to open the exact lines a result points to. Work in rounds: "
    "use the search tool to locate the relevant file/symbol, then, when "
    "available, read the surrounding source to confirm the precise behavior the "
    "question asks about — do not rely on memory or assumptions about how such "
    "code usually works. Questions often describe a file or symbol by its role "
    "rather than naming it, so translate the description into concrete search "
    "terms.\n\n"
    "Be economical: gather only the evidence you actually need. Prefer the "
    "fewest tool calls and the smallest, most targeted queries that settle the "
    "question, and stop as soon as you have the source that shows the behavior. "
    "Keep your notes concise to minimize tokens.\n\n"
    "Report what the source actually shows — cite the concrete file, symbol and "
    "lines and describe the behavior they establish. Do NOT announce a final "
    "yes/no answer or say 'therefore the answer is ...'; leave the verdict to "
    "the verifier. A plausible-sounding behavior is not the same as what the "
    "code actually does, so ground every claim in the lines you read."
)

#: System prompt for the answerer call: fresh session, full context in, no
#: tools, a single ``yes``/``no`` word out. Also kept stable so it caches as a
#: shared prefix; the variable context (question + reasoning + tool output) is
#: supplied as the user message AFTER this, never ahead of it.
ANSWERER_SYSTEM_PROMPT = (
    "You are given a question about a C codebase and the full context another "
    "agent gathered from the source (its reasoning and tool output). Decide the "
    "answer using ONLY that context; do not rely on outside knowledge. The "
    "answer is strictly binary. Output exactly one lowercase word and nothing "
    "else: \"yes\" or \"no\". No punctuation, no explanation, no other text. "
    "If the context is insufficient to confirm the statement, answer \"no\"."
)
