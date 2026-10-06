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

# -- Target repository --------------------------------------------------------

#: Path to the codebase the questions are about.
TARGET_REPO = os.getenv("TARGET_REPO", ".")

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

# -- Prompts ------------------------------------------------------------------

#: System prompt for the agent call. It may reason freely but must call exactly
#: one tool, once. Not required to be short.
SYSTEM_PROMPT = (
    "You answer questions about a code repository. "
    "You may call at most ONE tool, once, to gather evidence. "
    "Explain your reasoning and give a complete answer."
)

#: System prompt for the answerer call: fresh session, full context in, no
#: tools, a single ``yes``/``no`` word out.
ANSWERER_SYSTEM_PROMPT = (
    "You are given the full context produced by another agent and a question. "
    "Based only on that context, answer with exactly one word: 'yes' or 'no'. "
    "No tools. No explanation."
)
