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
    """Result of a single agent turn (one LLM call in the tool-call loop).

    :param reasoning: The model's reasoning trace (empty if none returned).
    :param tool_name: Name of the tool the model called, or ``None`` if it
        chose not to call a tool (only possible on non-forced rounds).
    :param tool_args: Parsed arguments for the tool call, or ``None``.
    :param tool_call_id: Provider id of the tool call, needed to attach the
        tool result back into the message history, or ``None``.
    :param assistant_message: The raw assistant message dict to append to the
        running message history for the next round.
    :param system_prompt: The system prompt actually sent (for the transcript).
    :param usage: Token/cost accounting for the call.
    """

    def __init__(self, *, reasoning, tool_name, tool_args, tool_call_id,
                 assistant_message, system_prompt, usage):
        self.reasoning = reasoning
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_call_id = tool_call_id
        self.assistant_message = assistant_message
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
        # timeout so a hung round surfaces as an error instead of blocking.
        self._openai = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY,
                              timeout=120.0)
        # langchain client for the answerer call (we handle retries ourselves).
        self._answerer = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            max_retries=0,
            timeout=120.0,
            max_tokens=config.ANSWERER_MAX_TOKENS,
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

    def _request_agent(self, messages, tool_schemas, tool_choice):
        """Make one raw agent completion and validate the response shape.

        OpenRouter surfaces provider errors (incl. rate limits) in the response
        body with ``choices=None`` rather than an HTTP error, which would crash
        on ``response.choices[0]``. Detect that and raise so :meth:`_retry` can
        back off on rate limits or fail loudly otherwise.

        :param tool_schemas: List of OpenAI-format tool schemas to offer.
        :raises RuntimeError: If the provider returned no choices.
        """
        response = self._openai.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=config.AGENT_MAX_TOKENS,
            messages=messages,
            tools=list(tool_schemas),
            tool_choice=tool_choice,
            extra_body={"reasoning": {"enabled": True}, "usage": {"include": True}},
        )
        if not getattr(response, "choices", None):
            err = getattr(response, "error", None) or getattr(response, "model_extra", {})
            raise RuntimeError(f"provider returned no choices: {err}")
        return response

    def call_agent(self, *, messages, tool_schemas, tool_name, force):
        """Run one agent turn over ``messages`` and capture the reasoning.

        One LLM call in the tool-call loop. On the first round ``force`` is
        ``True`` so the model must call ``tool_name`` (the primary search tool);
        on later rounds it is ``False`` (``tool_choice="auto"``) so the model
        may call any offered tool again, call the shared read tool, or stop
        (return no tool call). Only ``tool_schemas`` are offered, so ``auto``
        can never pick a tool outside the intended set.

        :param messages: The running chat history (list of message dicts). Not
            mutated; the caller appends :attr:`AgentCall.assistant_message` and
            any tool result itself.
        :param tool_schemas: List of OpenAI-format JSON schemas (the primary
            tool, plus any shared secondary tools such as ``read_file``).
        :param tool_name: Name of the primary tool (forced on round 1).
        :param force: Whether to force the primary tool call this round.
        :return: An :class:`AgentCall`.
        """
        tool_choice = (
            {"type": "function", "function": {"name": tool_name}} if force else "auto"
        )
        response = self._retry(lambda: self._request_agent(messages, tool_schemas, tool_choice))

        message = response.choices[0].message
        called_name, called_args, call_id = None, None, None
        # Reconstruct a minimal, replay-safe assistant message. The raw
        # model_dump() carries provider-only fields (reasoning,
        # reasoning_details, annotations) that are invalid in a request
        # message and can make the backend hang/reject on the next round.
        assistant_message = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            call = message.tool_calls[0]
            called_name = call.function.name
            call_id = call.id
            try:
                called_args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                called_args = {}
            assistant_message["tool_calls"] = [{
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments or "{}",
                },
            }]
        return AgentCall(
            reasoning=(getattr(message, "reasoning", None) or "").strip(),
            tool_name=called_name,
            tool_args=called_args,
            tool_call_id=call_id,
            assistant_message=assistant_message,
            system_prompt=None,
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
