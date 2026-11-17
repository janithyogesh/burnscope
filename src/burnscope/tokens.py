"""Estimate the tokens of each model call from a recorded conversation.

A chat model has no memory between calls: every call re-sends the whole
conversation so far. So for each assistant message i in a transcript:

    input  = tool definitions + hidden tool prompt
             + every message before i (each with a small formatting overhead)
             + reply priming
    output = assistant message i

The token counter is passed in as a plain function, so callers choose the
tokenizer and tests can use a simple fake one.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from burnscope.models import ChatMessage, TokenUsage

TokenCounter = Callable[[str], int]
"""Returns the number of tokens in a piece of text."""


@dataclass(frozen=True)
class ChatFormat:
    """How a provider wraps messages and tools into tokens."""

    tokens_per_message: int
    """Hidden tokens around every message (role markers)."""
    tokens_per_name: int
    """Extra tokens when a message has a `name` field."""
    reply_priming: int
    """Tokens that start the model's reply, added once per call."""
    tool_prompt: int
    """Hidden system prompt the provider adds whenever tools are given."""
    tool_schema_style: Literal["openai", "json"]
    """'openai': OpenAI's compact tool format; 'json': raw JSON schema text."""
    count_tool_calls: bool = True
    """False when the transcript already contains each tool call in the message
    text, so counting `tool_calls` as well would count the call twice."""


# Source: OpenAI cookbook "How to count tokens with tiktoken", commit db31449
# (2024-10-08): tokens_per_message = 3, tokens_per_name = 1, and "every reply
# is primed with <|start|>assistant<|message|>" (3 tokens). OpenAI adds no
# separate hidden tool prompt beyond the tool definitions themselves.
OPENAI_FORMAT = ChatFormat(
    tokens_per_message=3,
    tokens_per_name=1,
    reply_priming=3,
    tool_prompt=0,
    tool_schema_style="openai",
)

# Source: archived Anthropic tool-use docs (2024-11-15): Claude 3.5 Sonnet (Oct)
# adds a 346-token tool use system prompt when tool_choice is "auto" (the
# default), and is shown tool definitions "in JSONSchema format". Anthropic does
# not publish per-message overhead, so OpenAI's values are used as a proxy.
CLAUDE_35_SONNET_FORMAT = ChatFormat(
    tokens_per_message=3,
    tokens_per_name=1,
    reply_priming=3,
    tool_prompt=346,
    tool_schema_style="json",
)

# Per-function constants for gpt-4o from the same OpenAI cookbook.
_FUNC_INIT = 7
_PROP_INIT = 3
_PROP_KEY = 3
_ENUM_INIT = -3
_ENUM_ITEM = 3
_FUNC_END = 12


def _openai_tool_tokens(tools: Sequence[dict[str, Any]], count: TokenCounter) -> int:
    """Tool definition tokens using the OpenAI cookbook formula for gpt-4o.

    The formula covers top-level parameters only; nested object fields are
    not counted, so this slightly underestimates tools with nested schemas.
    """
    if not tools:
        return 0
    total = 0
    for tool in tools:
        function = tool["function"]
        total += _FUNC_INIT
        description = function.get("description", "").removesuffix(".")
        total += count(f"{function['name']}:{description}")
        properties: dict[str, Any] = function.get("parameters", {}).get("properties", {})
        if properties:
            total += _PROP_INIT
            for key, spec in properties.items():
                total += _PROP_KEY
                if "enum" in spec:
                    total += _ENUM_INIT
                    for item in spec["enum"]:
                        total += _ENUM_ITEM + count(str(item))
                prop_description = spec.get("description", "").removesuffix(".")
                total += count(f"{key}:{spec.get('type', '')}:{prop_description}")
    return total + _FUNC_END


def tool_definition_tokens(
    tools: Sequence[dict[str, Any]], count: TokenCounter, fmt: ChatFormat
) -> int:
    """Tokens the tool definitions add to every call's input."""
    if fmt.tool_schema_style == "openai":
        return _openai_tool_tokens(tools, count)
    return count(json.dumps(list(tools))) if tools else 0


def message_tokens(message: ChatMessage, count: TokenCounter, fmt: ChatFormat) -> int:
    """Tokens of one message's text and tool calls, without formatting overhead."""
    total = count(message.content)
    if message.name is not None:
        total += count(message.name) + fmt.tokens_per_name
    if fmt.count_tool_calls:
        for call in message.tool_calls:
            total += count(call.name) + count(call.arguments)
    return total


def call_usages(
    messages: Sequence[ChatMessage],
    tools: Sequence[dict[str, Any]],
    count: TokenCounter,
    fmt: ChatFormat,
) -> list[TokenUsage]:
    """Estimate the token usage of every model call in a conversation.

    Each assistant message is one call. No prompt caching is assumed, so all
    input is counted as uncached.
    """
    fixed_input = fmt.reply_priming  # sent with every call, whatever the history
    if tools:
        fixed_input += tool_definition_tokens(tools, count, fmt) + fmt.tool_prompt
    history = 0  # tokens of every message so far, with formatting overhead
    usages: list[TokenUsage] = []
    for message in messages:
        tokens = message_tokens(message, count, fmt)
        if message.role == "assistant":
            usages.append(TokenUsage(uncached_input=fixed_input + history, output=tokens))
        history += fmt.tokens_per_message + tokens
    return usages
