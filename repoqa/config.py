"""Central configuration for the benchmark.

Every value can be overridden with an environment variable (loaded from a
local ``.env`` file). Import this module and read the module-level constants.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# -- LLM backend (OpenRouter: an OpenAI-compatible endpoint) ------------------

#: Base URL of the OpenAI-compatible chat-completions endpoint.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
#: API key; falls back to ``OPENAI_API_KEY`` when ``OPENROUTER_API_KEY`` is unset.
LLM_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
#: Model identifier (tool-calling capable). Override in ``.env``.
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-v4-flash-0731:free")
#: Sampling temperature. Kept at 0 for reproducible benchmarking.
LLM_TEMPERATURE = 0.0

# -- Agent loop ---------------------------------------------------------------

#: Maximum number of agent tool-call rounds (``n``). Round 1 forces the tool;
#: rounds 2..n let the model call it again or stop early. Override via
#: ``AGENT_ROUNDS`` in ``.env`` or the ``--rounds`` CLI flag.
AGENT_ROUNDS = int(os.getenv("AGENT_ROUNDS", "2"))

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

# -- Prompts ------------------------------------------------------------------

#: System prompt for the agent call. Kept stable (identical across all
#: questions) so the provider can serve it from the prompt/KV cache as a shared
#: prefix — the per-question content (the question, tool results) always comes
#: AFTER this in the message history, never before, so the cached prefix holds.
#: Do not prepend per-question or retrieved content ahead of this text.
SYSTEM_PROMPT = (
    "You are a precise software engineer answering questions about a large C "
    "codebase. Each question has a definite yes/no answer that is verifiable "
    "by reading the source.\n\n"
    "You have one search tool to gather evidence; some setups also give you a "
    "read_file tool to open the exact lines a result points to. Work in rounds: "
    "use the search tool to locate the relevant file/symbol, then, when "
    "available, read the surrounding source to confirm the precise behavior the "
    "question asks about — do not answer from memory or assumptions about how "
    "such code usually works. Questions often describe a file or symbol by its "
    "role rather than naming it, so translate the description into concrete "
    "search terms.\n\n"
    "Be economical: gather only the evidence you actually need. Prefer the "
    "fewest tool calls and the smallest, most targeted queries that settle the "
    "question, and stop as soon as you have enough to answer — do not spend "
    "extra rounds or pull large outputs once the answer is clear. Keep your "
    "reasoning concise to minimize tokens.\n\n"
    "Reason explicitly but briefly about what the source shows, then give a "
    "complete, evidence-grounded answer. A plausible-sounding behavior is not "
    "the same as what the code actually does."
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
