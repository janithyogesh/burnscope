"""Tests for the tau-bench importer, using tiny hand-written fixtures."""

import json
from pathlib import Path
from typing import Any

import pytest

from burnscope.importers.taubench import TauBenchError, load_runs, load_tools
from burnscope.models import ChatMessage, ToolCall

OPENAI_STYLE_RUN: dict[str, Any] = {
    "task_id": 3,
    "trial": 1,
    "reward": 1.0,
    "info": {"user_cost": 0.002, "task": {}},
    "traj": [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"id": 7}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "lookup", "content": "found"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "###STOP###"},
    ],
}

CLAUDE_STYLE_RUN: dict[str, Any] = {
    "task_id": 3,
    "trial": 0,
    "reward": 0.0,
    "info": {"user_cost": None},
    "traj": [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "Checking.<function_calls>lookup</function_calls>",
            "tool_calls": [{"id": "0", "function": {"name": "lookup", "arguments": {"id": 7}}}],
            "tool_call_id": None,
        },
        {
            "role": "tool",
            "content": "found",
            "name": "lookup",
            "tool_call_id": "0",
            "tool_calls": None,
        },
        {"role": "assistant", "content": "done", "tool_calls": None, "tool_call_id": None},
    ],
}


def write_runs(tmp_path: Path, runs: list[dict[str, Any]]) -> Path:
    path = tmp_path / "runs.json"
    path.write_text(json.dumps(runs), encoding="utf-8")
    return path


def test_reads_openai_style_run(tmp_path: Path) -> None:
    (run,) = load_runs(write_runs(tmp_path, [OPENAI_STYLE_RUN]))

    assert (run.task_id, run.trial, run.success, run.user_cost) == (3, 1, True, 0.002)
    assert run.messages[2] == ChatMessage(
        role="assistant", tool_calls=(ToolCall(name="lookup", arguments='{"id": 7}'),)
    )
    assert run.messages[3] == ChatMessage(role="tool", name="lookup", content="found")


def test_reads_claude_style_run(tmp_path: Path) -> None:
    (run,) = load_runs(write_runs(tmp_path, [CLAUDE_STYLE_RUN]))

    assert run.success is False
    assert run.user_cost is None
    # Dict arguments become JSON text, and the text content is kept as written.
    assert run.messages[2].tool_calls == (ToolCall(name="lookup", arguments='{"id": 7}'),)
    assert run.messages[2].content.startswith("Checking.")
    assert run.messages[4] == ChatMessage(role="assistant", content="done")


def test_counts_agent_and_tool_calls(tmp_path: Path) -> None:
    openai_run, claude_run = load_runs(write_runs(tmp_path, [OPENAI_STYLE_RUN, CLAUDE_STYLE_RUN]))

    assert (openai_run.n_agent_calls, openai_run.n_tool_calls) == (2, 1)
    assert (claude_run.n_agent_calls, claude_run.n_tool_calls) == (2, 1)


def test_rejects_file_in_wrong_format(tmp_path: Path) -> None:
    with pytest.raises(TauBenchError, match="not a tau-bench trajectory file"):
        load_runs(write_runs(tmp_path, [{"task_id": 1}]))


INIT_PY = """\
from .lookup import Lookup
from .unused_helper import Helper
from .cancel import Cancel

ALL_TOOLS = [
    Cancel,
    Lookup,
]
"""

LOOKUP_PY = """\
from typing import Any, Dict

class Lookup:
    @staticmethod
    def invoke(data: Dict[str, Any], id: int) -> str:
        raise RuntimeError("must never run")

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {"type": "function", "function": {"name": "lookup", "description": "Find."}}
"""

CANCEL_PY = LOOKUP_PY.replace("lookup", "cancel").replace("Lookup", "Cancel")


@pytest.fixture
def tools_dir(tmp_path: Path) -> Path:
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "__init__.py").write_text(INIT_PY, encoding="utf-8")
    (tools / "lookup.py").write_text(LOOKUP_PY, encoding="utf-8")
    (tools / "cancel.py").write_text(CANCEL_PY, encoding="utf-8")
    (tools / "unused_helper.py").write_text(
        "raise RuntimeError('must never run')", encoding="utf-8"
    )
    return tools


def test_loads_tools_in_all_tools_order(tools_dir: Path) -> None:
    tools = load_tools(tools_dir)

    assert [tool["function"]["name"] for tool in tools] == ["cancel", "lookup"]


def test_tool_without_plain_get_info_is_an_error(tools_dir: Path) -> None:
    (tools_dir / "cancel.py").write_text(
        "def get_info():\n    return build_info()\n", encoding="utf-8"
    )

    with pytest.raises(TauBenchError, match=r"cancel\.py"):
        load_tools(tools_dir)
