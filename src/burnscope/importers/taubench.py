"""Read tau-bench's published agent trajectories.

tau-bench (https://github.com/sierra-research/tau-bench) stores each run as
{task_id, trial, reward, info, traj}, where traj is the full message list.
Two message formats appear in the published files:

- OpenAI style: tool call arguments are JSON text.
- Claude style: tool call arguments are a dict, and the call is also written
  out as text inside the assistant message's content.

Both are normalised into ChatMessage. Tool definitions are read from the
tau-bench source with `ast`, so no tau-bench code is ever run.
"""

import ast
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from burnscope.models import ChatMessage, ToolCall


class TauBenchError(Exception):
    """A tau-bench file is not in the expected format."""


class TauRun(BaseModel):
    """One recorded attempt at one task."""

    model_config = ConfigDict(frozen=True)

    task_id: int
    trial: int
    reward: float
    messages: tuple[ChatMessage, ...]
    user_cost: float | None
    """The simulated user's cost exactly as recorded (may be missing)."""

    @property
    def success(self) -> bool:
        """tau-bench counts a run as solved only when the reward is 1.0."""
        return self.reward == 1.0

    @property
    def n_agent_calls(self) -> int:
        """Model calls made by the agent: one per assistant message."""
        return sum(message.role == "assistant" for message in self.messages)

    @property
    def n_tool_calls(self) -> int:
        """Tool calls the agent made across the whole run."""
        return sum(len(message.tool_calls) for message in self.messages)


def _tool_call(raw: dict[str, Any]) -> ToolCall:
    function = raw["function"]
    arguments = function["arguments"]
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return ToolCall(name=function["name"], arguments=arguments)


def _message(raw: dict[str, Any]) -> ChatMessage:
    return ChatMessage(
        role=raw["role"],
        content=raw.get("content") or "",
        name=raw.get("name") if raw["role"] == "tool" else None,
        tool_calls=tuple(_tool_call(call) for call in raw.get("tool_calls") or ()),
    )


def load_runs(path: Path) -> list[TauRun]:
    """Read every run from one historical trajectory file."""
    try:
        raw_runs = json.loads(path.read_text(encoding="utf-8"))
        return [
            TauRun(
                task_id=raw["task_id"],
                trial=raw["trial"],
                reward=raw["reward"],
                messages=tuple(_message(message) for message in raw["traj"]),
                user_cost=raw["info"].get("user_cost"),
            )
            for raw in raw_runs
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise TauBenchError(f"{path} is not a tau-bench trajectory file: {error!r}") from None


def _tool_order(init_file: Path) -> list[str]:
    """Return the module names of the tools in ALL_TOOLS, in order."""
    tree = ast.parse(init_file.read_text(encoding="utf-8"))
    module_of: dict[str, str] = {}
    all_tools: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                module_of[alias.asname or alias.name] = node.module
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "ALL_TOOLS" for target in node.targets
        ):
            if not isinstance(node.value, ast.List):
                raise TauBenchError(f"ALL_TOOLS in {init_file} is not a plain list")
            all_tools = [element.id for element in node.value.elts if isinstance(element, ast.Name)]
    if not all_tools:
        raise TauBenchError(f"No ALL_TOOLS list found in {init_file}")
    return [module_of[name] for name in all_tools]


def _get_info(tool_file: Path) -> dict[str, Any]:
    """Return the dict that a tool's get_info() returns, without running it."""
    tree = ast.parse(tool_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_info":
            last = node.body[-1]
            if isinstance(last, ast.Return) and last.value is not None:
                try:
                    info = ast.literal_eval(last.value)
                except ValueError:
                    break  # not plain data, e.g. it calls a function
                if isinstance(info, dict):
                    return info
    raise TauBenchError(f"No get_info() returning a plain dict in {tool_file}")


def load_tools(tools_dir: Path) -> list[dict[str, Any]]:
    """Read the tool definitions given to the agent in one tau-bench domain.

    `tools_dir` is e.g. tau_bench/envs/airline/tools.
    """
    return [
        _get_info(tools_dir / f"{module}.py") for module in _tool_order(tools_dir / "__init__.py")
    ]
