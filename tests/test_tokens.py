"""Tests for rebuilding per-call token usage from a conversation.

A fake counter (one token per character) keeps the expected numbers easy to
work out by hand.
"""

import json
from typing import Any

from burnscope.models import ChatMessage, TokenUsage, ToolCall
from burnscope.tokens import (
    CLAUDE_35_SONNET_FORMAT,
    OPENAI_FORMAT,
    ChatFormat,
    call_usages,
    message_tokens,
    tool_definition_tokens,
)


def chars(text: str) -> int:
    return len(text)


# Simple numbers: 1 overhead per message, 2 for reply priming, 5 for the tool prompt.
FMT = ChatFormat(
    tokens_per_message=1,
    tokens_per_name=1,
    reply_priming=2,
    tool_prompt=5,
    tool_schema_style="json",
)

TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Find an order.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "The order id."}},
            "required": ["order_id"],
        },
    },
}

CONVERSATION = [
    ChatMessage(role="system", content="policy"),  # 6
    ChatMessage(role="user", content="hi"),  # 2
    ChatMessage(
        role="assistant", tool_calls=(ToolCall(name="lookup", arguments='{"id":1}'),)
    ),  # 6 + 8 = 14
    ChatMessage(role="tool", name="lookup", content="found"),  # 5 + (6 + 1) = 12
    ChatMessage(role="assistant", content="done"),  # 4
]


def test_one_usage_per_assistant_message() -> None:
    assert len(call_usages(CONVERSATION, [], chars, FMT)) == 2


def test_input_is_everything_before_the_reply() -> None:
    first, second = call_usages(CONVERSATION, [], chars, FMT)

    # No tools: priming 2 + system (1+6) + user (1+2) = 12
    assert first == TokenUsage(uncached_input=12, output=14)
    # ... + assistant (1+14) + tool (1+12) = 40
    assert second == TokenUsage(uncached_input=40, output=4)


def test_input_grows_with_every_call() -> None:
    first, second = call_usages(CONVERSATION, [], chars, FMT)

    assert second.uncached_input > first.uncached_input


def test_tools_add_definitions_and_hidden_prompt_to_every_call() -> None:
    without = call_usages(CONVERSATION, [], chars, FMT)
    with_tools = call_usages(CONVERSATION, [TOOL], chars, FMT)
    extra = len(json.dumps([TOOL])) + FMT.tool_prompt

    assert [
        u.uncached_input - w.uncached_input for u, w in zip(with_tools, without, strict=True)
    ] == [
        extra,
        extra,
    ]


def test_tool_calls_not_counted_when_already_in_the_text() -> None:
    fmt = ChatFormat(**{**FMT.__dict__, "count_tool_calls": False})
    message = ChatMessage(
        role="assistant",
        content="<call>lookup</call>",
        tool_calls=(ToolCall(name="lookup", arguments="{}"),),
    )

    assert message_tokens(message, chars, fmt) == len("<call>lookup</call>")
    assert message_tokens(message, chars, FMT) == len("<call>lookup</call>") + 6 + 2


def test_conversation_without_replies_has_no_calls() -> None:
    messages = [ChatMessage(role="system", content="x"), ChatMessage(role="user", content="y")]

    assert call_usages(messages, [TOOL], chars, FMT) == []


def test_json_style_counts_the_schema_text() -> None:
    assert tool_definition_tokens([TOOL], chars, FMT) == len(json.dumps([TOOL]))


def test_openai_style_follows_the_cookbook_formula() -> None:
    # 7 (function) + "lookup:Find an order" (20) + 3 (properties)
    # + 3 (property) + "order_id:string:The order id" (28) + 12 (end) = 73
    assert tool_definition_tokens([TOOL], chars, OPENAI_FORMAT) == 73


def test_openai_style_counts_enums() -> None:
    tool = json.loads(json.dumps(TOOL))
    tool["function"]["parameters"]["properties"]["order_id"]["enum"] = ["a", "bb"]

    # enum: -3 + (3 + 1) + (3 + 2) = 6 more than the plain tool
    assert tool_definition_tokens([tool], chars, OPENAI_FORMAT) == 73 + 6


def test_no_tools_means_no_tool_tokens() -> None:
    assert tool_definition_tokens([], chars, OPENAI_FORMAT) == 0
    assert tool_definition_tokens([], chars, CLAUDE_35_SONNET_FORMAT) == 0


def test_claude_format_adds_documented_tool_prompt() -> None:
    assert CLAUDE_35_SONNET_FORMAT.tool_prompt == 346
    assert OPENAI_FORMAT.tool_prompt == 0
