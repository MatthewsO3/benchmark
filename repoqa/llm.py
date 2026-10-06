"""Chat-completions client for the two calls the benchmark makes.

The *agent* call (call #1) is forced to call a single tool. It uses the raw
OpenAI SDK because this model returns its chain-of-thought in a separate
``reasoning`` field that langchain's ``ChatOpenAI`` discards on tool-call turns.

The *answerer* call (call #2) is a plain, tool-free completion. It uses
``ChatOpenAI`` exactly as the original implementation did, so results are
unchanged.
"""
import json
import time
import uuid
import logging

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from . import config
from .usage import TokenUsage

log = logging.getLogger("repoqa.llm")

#: Substrings that mark a rate-limit error worth retrying.
_RATE_LIMIT_MARKERS = ("429", "rate", "temporarily")


def apply_cache_mode(system_prompt: str, cache_mode: str) -> str:
    """Return the system prompt, optionally nonce-prefixed to defeat caching.

    :param system_prompt: The base system prompt.
    :param cache_mode: ``"on"`` to keep a stable prefix (the provider may serve
        it from cache) or ``"off"`` to prepend a unique nonce so no prefix ever
        matches and no cache hit occurs.
    :return: The (possibly nonce-prefixed) system prompt.
    """
    if cache_mode == "off":
        return f"[nocache {uuid.uuid4().hex}]\n{system_prompt}"
    return system_prompt


class AgentCall:
    """Result of the forced-tool agent call.

    :param reasoning: The model's reasoning trace (empty if none returned).
    :param tool_name: Name of the tool the model called, or ``None``.
    :param tool_args: Parsed arguments for the tool call, or ``None``.
    :param system_prompt: The system prompt actually sent (for the transcript).
    :param usage: Token/cost accounting for the call.
    """

    def __init__(self, *, reasoning, tool_name, tool_args, system_prompt, usage):
        self.reasoning = reasoning
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.system_prompt = system_prompt
        self.usage = usage


class AnswererCall:
    """Result of the tool-free answerer call.

    :param text: The assistant's raw text output.
    :param system_prompt: The system prompt actually sent (for the transcript).
    :param context: The user context actually sent (for the transcript).
    :param usage: Token/cost accounting for the call.
    """

    def __init__(self, *, text, system_prompt, context, usage):
        self.text = text
        self.system_prompt = system_prompt
        self.context = context
        self.usage = usage


class LLMClient:
    """Reusable client for the configured chat-completions backend.

    :param max_retries: Maximum retries on rate-limit errors.
    :param base_delay: Initial back-off delay in seconds (doubled per retry).
    """

    def __init__(self, *, max_retries: int = 5, base_delay: float = 8.0):
        self.model = config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE
        self.max_retries = max_retries
        self.base_delay = base_delay
        # Raw SDK client for the agent call (reads the `reasoning` field).
        self._openai = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
        # langchain client for the answerer call (we handle retries ourselves).
        self._answerer = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            max_retries=0,
        )

    def _retry(self, operation):
        """Run ``operation`` with exponential back-off on rate-limit errors.

        :param operation: A zero-argument callable performing the request.
        :return: Whatever ``operation`` returns on success.
        :raises Exception: Re-raises the last error if not a retryable rate
            limit, or once retries are exhausted.
        """
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except Exception as exc:  # noqa: BLE001
                message = str(exc).lower()
                rate_limited = any(marker in message for marker in _RATE_LIMIT_MARKERS)
                if not rate_limited or attempt == self.max_retries:
                    raise
                delay = self.base_delay * (2 ** attempt)
                log.warning("rate-limited, retry %d/%d in %.0fs",
                            attempt + 1, self.max_retries, delay)
                time.sleep(delay)

    def call_agent(self, *, system_prompt, question, tool_schema, tool_name, cache_mode):
        """Force exactly one call to ``tool_name`` and capture the reasoning.

        :param system_prompt: System prompt for the agent.
        :param question: The user question.
        :param tool_schema: OpenAI-format JSON schema for the tool.
        :param tool_name: Name of the tool the model is forced to call.
        :param cache_mode: ``"on"`` or ``"off"`` (see :func:`apply_cache_mode`).
        :return: An :class:`AgentCall`.
        """
        sent_prompt = apply_cache_mode(system_prompt, cache_mode)
        response = self._retry(lambda: self._openai.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": sent_prompt},
                {"role": "user", "content": question},
            ],
            tools=[tool_schema],
            tool_choice={"type": "function", "function": {"name": tool_name}},
            extra_body={"reasoning": {"enabled": True}, "usage": {"include": True}},
        ))

        message = response.choices[0].message
        called_name, called_args = None, None
        if message.tool_calls:
            call = message.tool_calls[0]
            called_name = call.function.name
            try:
                called_args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                called_args = {}
        return AgentCall(
            reasoning=(getattr(message, "reasoning", None) or "").strip(),
            tool_name=called_name,
            tool_args=called_args,
            system_prompt=sent_prompt,
            usage=TokenUsage.from_openai_usage(response.usage),
        )

    def call_answerer(self, *, system_prompt, context, cache_mode):
        """Run the tool-free answerer and return its one-word output.

        :param system_prompt: System prompt for the answerer.
        :param context: Full user context (question + reasoning + tool output).
        :param cache_mode: ``"on"`` or ``"off"`` (see :func:`apply_cache_mode`).
        :return: An :class:`AnswererCall`.
        """
        sent_prompt = apply_cache_mode(system_prompt, cache_mode)
        messages = [
            SystemMessage(content=sent_prompt),
            HumanMessage(content=context),
        ]
        message = self._retry(lambda: self._answerer.invoke(messages))
        text = message.content if isinstance(message.content, str) else str(message.content)
        return AnswererCall(
            text=text,
            system_prompt=sent_prompt,
            context=context,
            usage=TokenUsage.from_langchain_message(message),
        )
