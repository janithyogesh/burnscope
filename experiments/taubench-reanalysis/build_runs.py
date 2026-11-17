"""Estimate the agent cost of every run in tau-bench's historical trajectories.

Writes results/runs.csv (one row per run) and prints sanity checks.
Run download.py first.

Usage:
    uv run python experiments/taubench-reanalysis/build_runs.py
"""

import csv
import statistics
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import tiktoken

from burnscope.importers.taubench import TauRun, load_runs, load_tools
from burnscope.models import TokenUsage
from burnscope.pricing import ModelPrice, cost_of, load_pricing
from burnscope.tokens import (
    CLAUDE_35_SONNET_FORMAT,
    OPENAI_FORMAT,
    ChatFormat,
    TokenCounter,
    call_usages,
    tool_definition_tokens,
)

HERE = Path(__file__).parent
TAU_BENCH = HERE / "data" / "tau-bench"
RESULTS = HERE / "results" / "runs.csv"
DOMAINS = ["airline", "retail"]


@dataclass(frozen=True)
class ModelSetup:
    """How to read and price one model's runs."""

    name: str
    file_prefix: str
    fmt: ChatFormat
    main_price: tuple[str, str]
    alt_price: tuple[str, str] | None
    user_cost_is_dollars: bool
    trials: int


MODELS = [
    ModelSetup(
        name="gpt-4o",
        file_prefix="gpt-4o",
        fmt=OPENAI_FORMAT,
        main_price=("openai", "gpt-4o-2024-05-13"),
        alt_price=("openai", "gpt-4o-2024-08-06"),
        user_cost_is_dollars=True,
        trials=4,
    ),
    ModelSetup(
        name="claude-3-5-sonnet-20241022",
        file_prefix="sonnet-35-new",
        # The Claude transcripts repeat each tool call as text in the message
        # content, so the separate tool_calls copy is not counted again.
        fmt=replace(CLAUDE_35_SONNET_FORMAT, count_tool_calls=False),
        main_price=("anthropic", "claude-3-5-sonnet-20241022"),
        alt_price=None,
        # In these files user_cost is 0.1 x the number of user messages: a
        # placeholder, not dollars. It is left empty in the results.
        user_cost_is_dollars=False,
        trials=8,
    ),
]

EXPECTED_TASKS = {"airline": 50, "retail": 115}
PUBLISHED_SUCCESS = {
    ("gpt-4o", "airline"): 0.42,
    ("gpt-4o", "retail"): 0.60,
    ("claude-3-5-sonnet-20241022", "airline"): 0.46,
    ("claude-3-5-sonnet-20241022", "retail"): 0.69,
}
SUCCESS_TOLERANCE = 0.01

COLUMNS = [
    "model",
    "domain",
    "task_id",
    "trial",
    "success",
    "n_agent_calls",
    "n_tool_calls",
    "input_tokens",
    "output_tokens",
    "cost_usd_main",
    "cost_usd_alt",
    "cost_usd_uniform",
    "user_cost_usd",
]


def uniform(fmt: ChatFormat) -> ChatFormat:
    """The same format, but with the tool list counted as JSON text.

    Used for cost_usd_uniform, so both models' tool lists are counted the same way.
    """
    return replace(fmt, tool_schema_style="json")


def o200k_counter() -> TokenCounter:
    """GPT-4o's tokenizer. Special-token text in transcripts is counted as plain text."""
    encoding = tiktoken.get_encoding("o200k_base")
    return lambda text: len(encoding.encode_ordinary(text))


def run_usage(
    run: TauRun, tools: list[dict[str, Any]], count: TokenCounter, fmt: ChatFormat
) -> TokenUsage:
    """Total estimated agent tokens over every call in one run."""
    calls = call_usages(run.messages, tools, count, fmt)
    return TokenUsage(
        uncached_input=sum(call.uncached_input for call in calls),
        output=sum(call.output for call in calls),
    )


def run_row(
    run: TauRun,
    setup: ModelSetup,
    domain: str,
    usage: TokenUsage,
    uniform_usage: TokenUsage,
    main: ModelPrice,
    alt: ModelPrice | None,
) -> dict[str, object]:
    """One CSV row for one run. Token columns use the main counting method."""
    user_cost = run.user_cost if setup.user_cost_is_dollars else None
    return {
        "model": setup.name,
        "domain": domain,
        "task_id": run.task_id,
        "trial": run.trial,
        "success": int(run.success),
        "n_agent_calls": run.n_agent_calls,
        "n_tool_calls": run.n_tool_calls,
        "input_tokens": usage.uncached_input,
        "output_tokens": usage.output,
        "cost_usd_main": f"{cost_of(usage, main).total:.8f}",
        "cost_usd_alt": "" if alt is None else f"{cost_of(usage, alt).total:.8f}",
        "cost_usd_uniform": f"{cost_of(uniform_usage, main).total:.8f}",
        "user_cost_usd": "" if user_cost is None else f"{user_cost:.8f}",
    }


def build() -> list[dict[str, object]]:
    """Estimate tokens and cost for every run and return the CSV rows."""
    if not TAU_BENCH.exists():
        sys.exit("No data found. Run download.py first.")
    pricing = load_pricing(HERE / "pricing.yaml")
    count = o200k_counter()
    rows: list[dict[str, object]] = []
    for setup in MODELS:
        main = pricing.price_for(*setup.main_price)
        alt = None if setup.alt_price is None else pricing.price_for(*setup.alt_price)
        for domain in DOMAINS:
            tools = load_tools(TAU_BENCH / "tau_bench" / "envs" / domain / "tools")
            main_tokens = tool_definition_tokens(tools, count, setup.fmt)
            uniform_tokens = tool_definition_tokens(tools, count, uniform(setup.fmt))
            print(
                f"{setup.name} {domain}: {len(tools)} tools = {main_tokens} tokens per call "
                f"(uniform: {uniform_tokens})"
            )
            path = TAU_BENCH / "historical_trajectories" / f"{setup.file_prefix}-{domain}.json"
            for run in load_runs(path):
                usage = run_usage(run, tools, count, setup.fmt)
                uniform_usage = run_usage(run, tools, count, uniform(setup.fmt))
                rows.append(run_row(run, setup, domain, usage, uniform_usage, main, alt))
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {RESULTS.relative_to(HERE)}")


def sanity_checks(rows: list[dict[str, object]]) -> bool:
    """Print checks against known facts. Return True if all pass."""
    all_ok = True
    print(f"\nTotal runs: {len(rows)} (expected 1980)")
    all_ok &= len(rows) == 1980
    header = (
        f"{'model':<28}{'domain':<9}{'runs':>5}{'tasks':>6}{'trials':>7}"
        f"{'success':>9}{'paper':>7}{'calls p50':>10}{'max':>5}{'cost p50 $':>12}"
        f"{'uniform p50 $':>15}  check"
    )
    print(header)
    print("-" * len(header))
    for setup in MODELS:
        for domain in DOMAINS:
            group = [r for r in rows if r["model"] == setup.name and r["domain"] == domain]
            tasks = {r["task_id"] for r in group}
            trials = {r["trial"] for r in group}
            success = statistics.mean(int(str(r["success"])) for r in group)
            calls = [int(str(r["n_agent_calls"])) for r in group]
            cost = statistics.median(float(str(r["cost_usd_main"])) for r in group)
            uniform_cost = statistics.median(float(str(r["cost_usd_uniform"])) for r in group)
            published = PUBLISHED_SUCCESS[(setup.name, domain)]
            ok = (
                len(tasks) == EXPECTED_TASKS[domain]
                and len(trials) == setup.trials
                and len(group) == EXPECTED_TASKS[domain] * setup.trials
                and abs(success - published) <= SUCCESS_TOLERANCE
            )
            all_ok &= ok
            print(
                f"{setup.name:<28}{domain:<9}{len(group):>5}{len(tasks):>6}{len(trials):>7}"
                f"{success:>9.3f}{published:>7.2f}{statistics.median(calls):>10.0f}"
                f"{max(calls):>5}{cost:>12.4f}{uniform_cost:>15.4f}  {'OK' if ok else 'MISMATCH'}"
            )
    print(f"\nAll sanity checks passed: {all_ok}")
    return all_ok


if __name__ == "__main__":
    result_rows = build()
    write_csv(result_rows)
    sys.exit(0 if sanity_checks(result_rows) else 1)
