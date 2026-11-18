"""Analyse the agent cost of tau-bench runs.

Answers four questions from results/runs.csv:
1. How much does cost vary across repeats of the same task?
2. Do failed runs cost more than successful ones?
3. What is the cost per successful task per model, with a 95% confidence interval?
4. Comparing models on only k trials per task, how often is the cheaper model misjudged?

Writes analysis/summary.md and analysis/fig1.png - fig3.png. Run build_runs.py first.

Usage:
    uv run python experiments/taubench-reanalysis/analyze.py
"""

import csv
import itertools
import math
import random
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from burnscope.analysis.stats import bootstrap_ci, cost_per_success, cv, max_min_ratio, percentile

HERE = Path(__file__).parent
RUNS_CSV = HERE / "results" / "runs.csv"
OUT_DIR = HERE / "analysis"

SEED = 20261117
N_RESAMPLES = 2000  # bootstrap resamples, and random 4-of-8 subsets in the Claude check
N_FLIP_DRAWS = 20000  # draws per flip simulation (question 4)
DPI = 200
GPT = "gpt-4o"
CLAUDE = "claude-3-5-sonnet-20241022"
DOMAINS = ("airline", "retail")
ALL_DOMAINS = (*DOMAINS, "both")
MATCHED_TRIALS = 4  # GPT-4o has 4 trials per task; Claude's first 4 match it
K_VALUES = (1, 2, 3)

COST_DEFINITIONS = {
    "main": (
        "`cost_usd_main`: each provider's documented tool counting; GPT-4o at 2024-05-13 prices"
    ),
    "alt": "`cost_usd_alt`: as main, but GPT-4o at 2024-08-06 prices (Claude unchanged)",
    "uniform": "`cost_usd_uniform`: tool list counted the same way (JSON) for both models",
    "alt_uniform": (
        "`cost_usd_alt_uniform`: tool list counted as in uniform, GPT-4o at 2024-08-06 prices"
    ),
}
DEFINITION_COLORS = {
    "main": "#1f77b4",
    "alt": "#ff7f0e",
    "uniform": "#2ca02c",
    "alt_uniform": "#9467bd",
}
N_DEFINITIONS = {1: "one", 2: "two", 3: "three", 4: "four"}[len(COST_DEFINITIONS)]

Task = tuple[str, int]  # (domain, task_id)


@dataclass(frozen=True)
class Run:
    """One row of runs.csv."""

    model: str
    domain: str
    task_id: int
    trial: int
    success: bool
    n_agent_calls: int
    costs: dict[str, float]  # cost per definition in COST_DEFINITIONS


@dataclass(frozen=True)
class View:
    """A model's runs, optionally limited to its first trials."""

    label: str
    model: str
    color: str
    max_trials: int | None = None


GPT_VIEW = View("GPT-4o (4 trials)", GPT, "#1f77b4")
CLAUDE_ALL = View("Claude (8 trials)", CLAUDE, "#d62728")
CLAUDE_MATCHED = View("Claude (first 4 trials)", CLAUDE, "#ff9896", MATCHED_TRIALS)
VIEWS = (GPT_VIEW, CLAUDE_ALL, CLAUDE_MATCHED)


# --- loading and selecting ---------------------------------------------------


def load_runs(path: Path = RUNS_CSV) -> list[Run]:
    """Read runs.csv. Claude has no alt price, so its alt cost is its main cost."""
    runs = []
    with path.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            main = float(row["cost_usd_main"])
            runs.append(
                Run(
                    model=row["model"],
                    domain=row["domain"],
                    task_id=int(row["task_id"]),
                    trial=int(row["trial"]),
                    success=row["success"] == "1",
                    n_agent_calls=int(row["n_agent_calls"]),
                    costs={
                        "main": main,
                        "alt": float(row["cost_usd_alt"]) if row["cost_usd_alt"] else main,
                        "uniform": float(row["cost_usd_uniform"]),
                        "alt_uniform": float(row["cost_usd_alt_uniform"]),
                    },
                )
            )
    return runs


def by_task(runs: Sequence[Run], view: View, domain: str) -> dict[Task, list[Run]]:
    """The view's runs grouped by task, each task's runs in trial order."""
    tasks: dict[Task, list[Run]] = defaultdict(list)
    for run in runs:
        if run.model != view.model or domain not in (run.domain, "both"):
            continue
        if view.max_trials is not None and run.trial >= view.max_trials:
            continue
        tasks[(run.domain, run.task_id)].append(run)
    return {task: sorted(group, key=lambda r: r.trial) for task, group in sorted(tasks.items())}


def cps(groups: Sequence[Sequence[Run]], definition: str) -> float:
    """Cost per success over every run in the given task groups."""
    flat = [run for group in groups for run in group]
    return cost_per_success([r.costs[definition] for r in flat], [r.success for r in flat])


# --- question 1: run-to-run spread ---------------------------------------------


@dataclass(frozen=True)
class Spread:
    view: View
    domain: str
    tasks: int
    cv_p50: float
    cv_p90: float
    ratio_p50: float
    ratio_p90: float
    ratio_worst: float


def spreads(runs: Sequence[Run]) -> list[Spread]:
    results = []
    for view in VIEWS:
        for domain in DOMAINS:
            tasks = by_task(runs, view, domain)
            costs = [[r.costs["main"] for r in group] for group in tasks.values()]
            cvs = [cv(c) for c in costs]
            ratios = [max_min_ratio(c) for c in costs]
            results.append(
                Spread(
                    view,
                    domain,
                    len(tasks),
                    percentile(cvs, 0.5),
                    percentile(cvs, 0.9),
                    percentile(ratios, 0.5),
                    percentile(ratios, 0.9),
                    max(ratios),
                )
            )
    return results


def success_ratio(group: Sequence[Run]) -> float | None:
    """Max/min cost among a task's successful runs; None if fewer than 2 succeeded."""
    costs = [r.costs["main"] for r in group if r.success]
    return max_min_ratio(costs) if len(costs) >= 2 else None


@dataclass(frozen=True)
class SuccessSpread:
    view: View
    domain: str
    tasks: int  # tasks with at least 2 successful runs
    all_tasks: int
    ratio_p50: float
    ratio_p90: float
    ratio_max: float


def success_spreads(runs: Sequence[Run]) -> list[SuccessSpread]:
    """Run-to-run spread using successful runs only, on 4 trials per task."""
    results = []
    for view in (GPT_VIEW, CLAUDE_MATCHED):
        for domain in DOMAINS:
            tasks = by_task(runs, view, domain)
            ratios = [r for g in tasks.values() if (r := success_ratio(g)) is not None]
            results.append(
                SuccessSpread(
                    view,
                    domain,
                    len(ratios),
                    len(tasks),
                    percentile(ratios, 0.5),
                    percentile(ratios, 0.9),
                    max(ratios),
                )
            )
    return results


# --- question 2: cost of failure -------------------------------------------------


@dataclass(frozen=True)
class Failure:
    view: View
    domain: str
    n_ok: int
    n_failed: int
    median_ok: float
    median_failed: float
    mean_ok: float
    mean_failed: float
    tasks: int
    within_tasks: int
    within_median: float
    within_q1: float
    within_q3: float
    within_share_costlier: float

    @property
    def overall_ratio(self) -> float:
        """Median failed-run cost / median successful-run cost, all runs pooled."""
        return self.median_failed / self.median_ok

    @property
    def overall_mean_ratio(self) -> float:
        """Mean failed-run cost / mean successful-run cost, all runs pooled."""
        return self.mean_failed / self.mean_ok


def failures(runs: Sequence[Run]) -> list[Failure]:
    results = []
    for view in (GPT_VIEW, CLAUDE_ALL):
        for domain in DOMAINS:
            tasks = by_task(runs, view, domain)
            flat = [r for group in tasks.values() for r in group]
            ok = [r.costs["main"] for r in flat if r.success]
            failed = [r.costs["main"] for r in flat if not r.success]
            ratios = []
            for group in tasks.values():
                task_ok = [r.costs["main"] for r in group if r.success]
                task_failed = [r.costs["main"] for r in group if not r.success]
                if task_ok and task_failed:
                    ratios.append(statistics.median(task_failed) / statistics.median(task_ok))
            results.append(
                Failure(
                    view,
                    domain,
                    len(ok),
                    len(failed),
                    statistics.median(ok),
                    statistics.median(failed),
                    statistics.fmean(ok),
                    statistics.fmean(failed),
                    len(tasks),
                    len(ratios),
                    statistics.median(ratios),
                    percentile(ratios, 0.25),
                    percentile(ratios, 0.75),
                    sum(x > 1 for x in ratios) / len(ratios),
                )
            )
    return results


# --- question 3: cost per success with confidence intervals ----------------------


@dataclass(frozen=True)
class CpsResult:
    view: View
    domain: str
    definition: str
    estimate: float
    ci: tuple[float, float]


def cps_results(runs: Sequence[Run]) -> list[CpsResult]:
    results = []
    for view in VIEWS:
        for domain in ALL_DOMAINS:
            groups = list(by_task(runs, view, domain).values())
            for definition in COST_DEFINITIONS:

                def statistic(sample: list[list[Run]], d: str = definition) -> float:
                    return cps(sample, d)

                ci = bootstrap_ci(groups, statistic, n_resamples=N_RESAMPLES, seed=SEED)
                results.append(CpsResult(view, domain, definition, cps(groups, definition), ci))
    return results


@dataclass(frozen=True)
class PairedResult:
    domain: str
    definition: str
    ratio: float
    ci: tuple[float, float]

    @property
    def includes_one(self) -> bool:
        return self.ci[0] <= 1.0 <= self.ci[1]

    @property
    def cheaper(self) -> str:
        return "GPT-4o" if self.ratio < 1 else "Claude"


Pair = tuple[list[Run], list[Run]]


def paired_tasks(runs: Sequence[Run], domain: str) -> list[Pair]:
    """(GPT-4o runs, Claude's first 4 runs) for every task both models ran."""
    gpt = by_task(runs, GPT_VIEW, domain)
    claude = by_task(runs, CLAUDE_MATCHED, domain)
    return [(gpt[task], claude[task]) for task in gpt if task in claude]


def paired_ratio(pairs: Sequence[Pair], definition: str) -> float:
    """GPT-4o cost per success divided by Claude's, over the same tasks."""
    return cps([g for g, _ in pairs], definition) / cps([c for _, c in pairs], definition)


def paired_results(runs: Sequence[Run]) -> list[PairedResult]:
    results = []
    for domain in ALL_DOMAINS:
        pairs = paired_tasks(runs, domain)
        for definition in COST_DEFINITIONS:

            def statistic(sample: list[Pair], d: str = definition) -> float:
                return paired_ratio(sample, d)

            ci = bootstrap_ci(pairs, statistic, n_resamples=N_RESAMPLES, seed=SEED)
            results.append(PairedResult(domain, definition, paired_ratio(pairs, definition), ci))
    return results


@dataclass(frozen=True)
class SubsetCheck:
    """Claude's cost per success (main) from all 8 trials vs from 4 of them."""

    domain: str
    all_trials: float
    first_four: float
    random_median: float
    random_range: tuple[float, float]  # 2.5th and 97.5th percentile
    ranking_changes: int  # random subsets that flip GPT-4o vs Claude under any definition


def subset_checks(runs: Sequence[Run]) -> list[SubsetCheck]:
    """Does using Claude's first 4 trials (to match GPT-4o) bias the comparison?

    Draws N_RESAMPLES random 4-of-8 subsets per task and compares the resulting cost per
    success with the all-8 value, and the cheaper model under every cost definition.
    """
    results = []
    for domain in ALL_DOMAINS:
        claude = list(by_task(runs, CLAUDE_ALL, domain).values())
        gpt = list(by_task(runs, GPT_VIEW, domain).values())
        gpt_cps = {d: cps(gpt, d) for d in COST_DEFINITIONS}
        all_cps = {d: cps(claude, d) for d in COST_DEFINITIONS}
        rng = random.Random(SEED)
        main_values = []
        changes = 0
        for _ in range(N_RESAMPLES):
            sample = [rng.sample(group, MATCHED_TRIALS) for group in claude]
            values = {d: cps(sample, d) for d in COST_DEFINITIONS}
            main_values.append(values["main"])
            changes += any(
                (gpt_cps[d] < values[d]) != (gpt_cps[d] < all_cps[d]) for d in COST_DEFINITIONS
            )
        first = list(by_task(runs, CLAUDE_MATCHED, domain).values())
        results.append(
            SubsetCheck(
                domain,
                all_cps["main"],
                cps(first, "main"),
                percentile(main_values, 0.5),
                (percentile(main_values, 0.025), percentile(main_values, 0.975)),
                changes,
            )
        )
    return results


# --- question 4: single-run risk ---------------------------------------------------


@dataclass(frozen=True)
class FlipResult:
    domain: str
    definition: str
    reference_ratio: float
    flip_probability: dict[int, float]  # k -> share of draws that flip the conclusion
    k1_median: float  # GPT-4o / Claude ratio across the k=1 draws
    k1_range: tuple[float, float]  # its 2.5th and 97.5th percentile


Totals = tuple[float, ...]  # cost under each definition, then the number of successes


def subset_totals(group: Sequence[Run], k: int) -> list[Totals]:
    """Summed costs and successes for every k-run subset of one task's runs."""
    return [
        (
            *(sum(r.costs[d] for r in subset) for d in COST_DEFINITIONS),
            sum(r.success for r in subset),
        )
        for subset in itertools.combinations(group, k)
    ]


def ratio_from_totals(gpt: Totals, claude: Totals, i: int) -> float:
    """GPT-4o / Claude cost per success for definition index i, from summed totals."""
    gpt_cps = gpt[i] / gpt[-1] if gpt[-1] else math.inf
    claude_cps = claude[i] / claude[-1] if claude[-1] else math.inf
    if math.isinf(claude_cps):
        return 0.0 if not math.isinf(gpt_cps) else 1.0
    return gpt_cps / claude_cps


def flip_results(runs: Sequence[Run]) -> list[FlipResult]:
    """How often k random trials per task reverse the full-data 'cheaper model'.

    Each draw picks one k-trial subset per task and model, uniformly at random (the same
    as sampling k trials without replacement), and is scored under every definition.
    """
    definitions = list(COST_DEFINITIONS)
    flips: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    k1_ratios: dict[tuple[str, str], list[float]] = defaultdict(list)
    references: dict[tuple[str, str], float] = {}
    for domain in ALL_DOMAINS:
        pairs = paired_tasks(runs, domain)
        for d in definitions:
            references[(domain, d)] = paired_ratio(pairs, d)
        rng = random.Random(SEED)
        for k in K_VALUES:
            options = [(subset_totals(g, k), subset_totals(c, k)) for g, c in pairs]
            counts = dict.fromkeys(definitions, 0)
            for _ in range(N_FLIP_DRAWS):
                gpt = tuple(map(sum, zip(*(rng.choice(g) for g, _ in options), strict=True)))
                claude = tuple(map(sum, zip(*(rng.choice(c) for _, c in options), strict=True)))
                for i, d in enumerate(definitions):
                    ratio = ratio_from_totals(gpt, claude, i)
                    counts[d] += (ratio > 1) != (references[(domain, d)] > 1)
                    if k == 1:
                        k1_ratios[(domain, d)].append(ratio)
            for d in definitions:
                flips[(domain, d)][k] = counts[d] / N_FLIP_DRAWS
    return [
        FlipResult(
            domain,
            d,
            references[(domain, d)],
            flips[(domain, d)],
            percentile(k1_ratios[(domain, d)], 0.5),
            (
                percentile(k1_ratios[(domain, d)], 0.025),
                percentile(k1_ratios[(domain, d)], 0.975),
            ),
        )
        for domain in ALL_DOMAINS
        for d in definitions
    ]


# --- tables ---------------------------------------------------------------------------


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """A markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def with_ci(estimate: float, ci: tuple[float, float], fmt: str = "${:.3f}") -> str:
    return f"{fmt.format(estimate)} [{fmt.format(ci[0])}, {fmt.format(ci[1])}]"


def spread_table(results: Sequence[Spread]) -> str:
    return table(
        ["view", "domain", "tasks", "CV p50", "CV p90", "max/min p50", "max/min p90", "worst task"],
        [
            [
                s.view.label,
                s.domain,
                s.tasks,
                f"{s.cv_p50:.2f}",
                f"{s.cv_p90:.2f}",
                f"{s.ratio_p50:.2f}x",
                f"{s.ratio_p90:.2f}x",
                f"{s.ratio_worst:.1f}x",
            ]
            for s in results
        ],
    )


def success_spread_table(results: Sequence[SuccessSpread]) -> str:
    return table(
        ["view", "domain", "tasks with 2+ successes", "max/min p50", "max/min p90", "max/min max"],
        [
            [
                s.view.label,
                s.domain,
                f"{s.tasks} of {s.all_tasks}",
                f"{s.ratio_p50:.2f}x",
                f"{s.ratio_p90:.2f}x",
                f"{s.ratio_max:.1f}x",
            ]
            for s in results
        ],
    )


def failure_tables(results: Sequence[Failure]) -> tuple[str, str]:
    overall = table(
        [
            "view",
            "domain",
            "successful runs",
            "failed runs",
            "median cost, successful",
            "median cost, failed",
            "failed / successful (medians)",
            "mean cost, successful",
            "mean cost, failed",
            "failed / successful (means)",
        ],
        [
            [
                f.view.label,
                f.domain,
                f.n_ok,
                f.n_failed,
                f"${f.median_ok:.3f}",
                f"${f.median_failed:.3f}",
                f"{f.overall_ratio:.2f}x",
                f"${f.mean_ok:.3f}",
                f"${f.mean_failed:.3f}",
                f"{f.overall_mean_ratio:.2f}x",
            ]
            for f in results
        ],
    )
    within = table(
        [
            "view",
            "domain",
            "tasks with both outcomes",
            "median per-task ratio (failed / successful)",
            "middle 50% of tasks",
            "tasks where failures cost more",
        ],
        [
            [
                f.view.label,
                f.domain,
                f"{f.within_tasks} of {f.tasks}",
                f"{f.within_median:.2f}x",
                f"{f.within_q1:.2f}x to {f.within_q3:.2f}x",
                f"{f.within_share_costlier:.0%}",
            ]
            for f in results
        ],
    )
    return overall, within


def cps_table(results: Sequence[CpsResult]) -> str:
    rows: list[list[object]] = []
    for view in VIEWS:
        for domain in ALL_DOMAINS:
            cells = {
                r.definition: with_ci(r.estimate, r.ci)
                for r in results
                if r.view == view and r.domain == domain
            }
            rows.append([view.label, domain, *(cells[d] for d in COST_DEFINITIONS)])
    return table(["view", "domain", *COST_DEFINITIONS], rows)


def paired_table(results: Sequence[PairedResult]) -> str:
    rows: list[list[object]] = []
    for domain in ALL_DOMAINS:
        cells = []
        for definition in COST_DEFINITIONS:
            (r,) = [x for x in results if x.domain == domain and x.definition == definition]
            note = "includes 1.0" if r.includes_one else "excludes 1.0"
            cells.append(f"{with_ci(r.ratio, r.ci, '{:.2f}')} ({note})")
        rows.append([domain, *cells])
    return table(["domain", *COST_DEFINITIONS], rows)


def flip_table(results: Sequence[FlipResult]) -> str:
    return table(
        [
            "domain",
            "cost definition",
            "GPT-4o / Claude, all 4 trials",
            *(f"flip chance, k={k}" for k in K_VALUES),
            "ratio at k=1: median [95% range]",
        ],
        [
            [
                r.domain,
                r.definition,
                f"{r.reference_ratio:.2f} ({'GPT-4o' if r.reference_ratio < 1 else 'Claude'} "
                "cheaper)",
                *(f"{r.flip_probability[k]:.1%}" for k in K_VALUES),
                f"{r.k1_median:.2f} [{r.k1_range[0]:.2f}, {r.k1_range[1]:.2f}]",
            ]
            for r in results
        ],
    )


def subset_table(results: Sequence[SubsetCheck]) -> str:
    return table(
        [
            "domain",
            "Claude, all 8 trials",
            "Claude, first 4",
            "Claude, random 4 of 8: median [95% range]",
            "subsets that change the cheaper model",
        ],
        [
            [
                r.domain,
                f"${r.all_trials:.3f}",
                f"${r.first_four:.3f}",
                f"${r.random_median:.3f} [${r.random_range[0]:.3f}, ${r.random_range[1]:.3f}]",
                f"{r.ranking_changes:,} of {N_RESAMPLES:,}",
            ]
            for r in results
        ],
    )


# --- charts -----------------------------------------------------------------------------


def fig_spread(runs: Sequence[Run], path: Path) -> None:
    """Fig 1: every run's cost per task, tasks sorted by median cost."""
    fig = Figure(figsize=(13, 8), layout="constrained")
    axes = fig.subplots(2, 2, sharey=True)
    for row, view in enumerate((GPT_VIEW, CLAUDE_ALL)):
        for col, domain in enumerate(DOMAINS):
            ax = axes[row][col]
            groups = sorted(
                by_task(runs, view, domain).values(),
                key=lambda g: statistics.median(r.costs["main"] for r in g),
            )
            for x, group in enumerate(groups):
                costs = [r.costs["main"] for r in group]
                colors = ["#2ca02c" if r.success else "#d62728" for r in group]
                ax.scatter([x] * len(costs), costs, s=9, c=colors, alpha=0.6, linewidths=0)
                ax.hlines(statistics.median(costs), x - 0.45, x + 0.45, color="black", lw=0.7)
            ax.set_yscale("log")
            ax.set_title(f"{view.label}, {domain} ({len(groups)} tasks)")
            ax.set_xlabel("Task, sorted by median cost per run")
            ax.set_ylabel("Agent cost per run (US$, log scale)")
    handles = [
        Line2D([], [], marker="o", ls="", color="#2ca02c", label="successful run"),
        Line2D([], [], marker="o", ls="", color="#d62728", label="failed run"),
        Line2D([], [], color="black", lw=0.7, label="task median"),
    ]
    fig.legend(handles=handles, loc="outside upper right", ncols=3)
    fig.suptitle("Cost of repeated runs of the same task (cost_usd_main)", x=0.02, ha="left")
    fig.savefig(path, dpi=DPI)


def fig_cps(results: Sequence[CpsResult], path: Path) -> None:
    """Fig 2: cost per successful task with 95% bootstrap intervals."""
    fig = Figure(figsize=(18, 4.8), layout="constrained")
    axes = fig.subplots(1, len(COST_DEFINITIONS), sharey=True)
    for ax, definition in zip(axes, COST_DEFINITIONS, strict=True):
        for i, view in enumerate(VIEWS):
            points = [
                next(
                    r
                    for r in results
                    if r.view == view and r.domain == domain and r.definition == definition
                )
                for domain in ALL_DOMAINS
            ]
            xs = [j + (i - 1) * 0.22 for j in range(len(ALL_DOMAINS))]
            ys = [p.estimate for p in points]
            errors = [
                [p.estimate - p.ci[0] for p in points],
                [p.ci[1] - p.estimate for p in points],
            ]
            ax.errorbar(xs, ys, yerr=errors, fmt="o", capsize=4, color=view.color, label=view.label)
        ax.set_xticks(range(len(ALL_DOMAINS)), ALL_DOMAINS)
        ax.set_xlabel("Domain")
        ax.set_title(f"cost_usd_{definition}")
        ax.grid(axis="y", alpha=0.3)
    # Shared y axis: make room for the highest interval in any panel.
    axes[0].set_ylim(0, 1.05 * max(r.ci[1] for r in results))
    axes[0].set_ylabel("Cost per successful task (US$), 95% CI")
    axes[0].legend(loc="lower left")
    fig.suptitle("Cost per successful task under each cost definition", x=0.02, ha="left")
    fig.savefig(path, dpi=DPI)


def fig_flip(results: Sequence[FlipResult], path: Path) -> None:
    """Fig 3: chance of picking the wrong cheaper model vs trials per task."""
    fig = Figure(figsize=(14, 4.8), layout="constrained")
    axes = fig.subplots(1, 3, sharey=True)
    for ax, domain in zip(axes, ALL_DOMAINS, strict=True):
        for definition in COST_DEFINITIONS:
            (r,) = [x for x in results if x.domain == domain and x.definition == definition]
            ax.plot(
                K_VALUES,
                [100 * r.flip_probability[k] for k in K_VALUES],
                marker="o",
                color=DEFINITION_COLORS[definition],
                label=f"cost_usd_{definition}",
            )
        ax.plot(
            [MATCHED_TRIALS],
            [0],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            ls="",
            label="k=4: the full data (0% by definition)",
        )
        ax.set_xticks([*K_VALUES, MATCHED_TRIALS])
        ax.set_xlabel("Trials per task used to compare the models (k)")
        ax.set_title(domain)
        ax.set_ylim(-2, 50)
        ax.grid(alpha=0.3)
        domain_results = [x for x in results if x.domain == domain]
        if all(p == 0 for x in domain_results for p in x.flip_probability.values()):
            ax.text(
                0.5,
                0.5,
                f"All {N_DEFINITIONS} cost definitions:\n0% at every k (lines overlap)",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
    axes[0].set_ylabel("Chance of naming the wrong cheaper model (%)")
    axes[0].legend(loc="upper right")
    fig.suptitle(
        f"Comparing GPT-4o and Claude on fewer trials per task ({N_FLIP_DRAWS:,} random draws)",
        x=0.02,
        ha="left",
    )
    fig.savefig(path, dpi=DPI)


BLOG_SOURCE_NOTE = (
    "Data: tau-bench published runs (Sierra Research). Agent cost estimated from transcripts "
    "at late-2024 list prices; GPT-4o at the gpt-4o-2024-05-13 rate.\n"
    "If the newer gpt-4o-2024-08-06 was used, GPT-4o costs would be about half. "
    "The spread within each task is the same either way. github.com/janithyogesh/burnscope"
)
BLOG_DOLLAR_TICKS = [0.025, 0.05, 0.10, 0.25, 0.50, 1.00]
BLOG_SUCCESS = {"color": "#1f77b4", "marker": "o", "label": "Successful run"}
BLOG_FAILED = {"color": "#e8710a", "marker": "^", "label": "Failed run"}


def dollars(value: float, _position: object = None) -> str:
    """Tick label like $0.05, keeping a third decimal only when needed ($0.025)."""
    return f"${value:.3f}" if round(value, 2) != round(value, 3) else f"${value:.2f}"


def highlight_widest_task(ax: Axes, groups: Sequence[Sequence[Run]]) -> None:
    """Mark the task whose costliest successful run is the most times its cheapest one.

    Only successful runs count, so the example can't be a run that was cheap
    because it gave up early.
    """
    ratios = [success_ratio(group) or 0.0 for group in groups]
    x = max(range(len(groups)), key=ratios.__getitem__)
    succeeded = sorted((r for r in groups[x] if r.success), key=lambda r: r.costs["main"])
    cheapest, costliest = succeeded[0], succeeded[-1]
    low, high = cheapest.costs["main"], costliest.costs["main"]
    ax.vlines(x, low, high, color="#444444", lw=0.8, ls="--", zorder=0)
    # Put the label on the emptier side: upper left if the task is on the right.
    on_right = x > len(groups) / 2
    ax.annotate(
        # "\$" keeps matplotlib from reading the two dollar signs as a math formula.
        f"Task {cheapest.task_id}: \\${low:.2f} vs \\${high:.2f} "
        f"({cheapest.n_agent_calls} vs {costliest.n_agent_calls} calls),\n"
        f"both succeeded: {ratios[x]:.1f}x",
        xy=(x, (low * high) ** 0.5),  # middle of the line on a log scale
        xytext=(0.04, 0.95) if on_right else (0.96, 0.05),
        textcoords="axes fraction",
        ha="left" if on_right else "right",
        va="top" if on_right else "bottom",
        multialignment="left",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#999999"},
        arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1},
    )


def fig_blog(runs: Sequence[Run], path: Path) -> None:
    """Simplified airline version of fig 1 for a general audience (1600 x 900 px)."""
    fig = Figure(figsize=(8, 4.5), layout="constrained", facecolor="white")
    axes = fig.subplots(1, 2, sharey=True)
    panels = ((GPT_VIEW, "GPT-4o"), (CLAUDE_MATCHED, "Claude 3.5 Sonnet (Oct 2024)"))
    for ax, (view, title) in zip(axes, panels, strict=True):
        groups = sorted(
            by_task(runs, view, "airline").values(),
            key=lambda g: statistics.median(r.costs["main"] for r in g),
        )
        for style, success in ((BLOG_SUCCESS, True), (BLOG_FAILED, False)):
            points = [
                (x, r.costs["main"])
                for x, g in enumerate(groups)
                for r in g
                if r.success == success
            ]
            ax.scatter(
                [x for x, _ in points],
                [cost for _, cost in points],
                s=14,
                color=style["color"],
                marker=style["marker"],
                alpha=0.75,
                linewidths=0,
            )
        for x, group in enumerate(groups):
            median = statistics.median(r.costs["main"] for r in group)
            ax.hlines(median, x - 0.45, x + 0.45, color="black", lw=0.9)
        if view == CLAUDE_MATCHED:
            highlight_widest_task(ax, groups)
        ax.set_yscale("log")
        ax.set_ylim(0.021, 1.25)
        ax.yaxis.set_major_locator(FixedLocator(BLOG_DOLLAR_TICKS))
        ax.yaxis.set_major_formatter(FuncFormatter(dollars))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([])
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("50 airline tasks, sorted by\nmedian cost per run", fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Cost of one run (US$, log scale)", fontsize=13)
    handles = [
        Line2D(
            [],
            [],
            ls="",
            color=BLOG_SUCCESS["color"],
            marker=BLOG_SUCCESS["marker"],
            label=BLOG_SUCCESS["label"],
        ),
        Line2D(
            [],
            [],
            ls="",
            color=BLOG_FAILED["color"],
            marker=BLOG_FAILED["marker"],
            label=BLOG_FAILED["label"],
        ),
        Line2D([], [], color="black", lw=0.9, label="Task median"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=10, frameon=False)
    fig.suptitle("Same task, different bill: 4 runs of each airline task", fontsize=18)
    fig.supxlabel(BLOG_SOURCE_NOTE, fontsize=7, color="#555555")
    fig.savefig(path, dpi=DPI, facecolor="white")


# --- summary -----------------------------------------------------------------------------


def robustness_lines(paired: Sequence[PairedResult]) -> list[str]:
    """Which 'cheaper model' conclusions hold under every cost definition."""
    lines = []
    for domain in ALL_DOMAINS:
        rs = [r for r in paired if r.domain == domain]
        parts = ", ".join(
            f"{r.cheaper} under `{r.definition}`"
            + ("" if not r.includes_one else " (CI includes 1.0: not distinguishable)")
            for r in rs
        )
        holds = len({r.cheaper for r in rs}) == 1 and not any(r.includes_one for r in rs)
        verdict = (
            f"**holds under all {N_DEFINITIONS}**" if holds else "**depends on the assumption**"
        )
        lines.append(f"- {domain}: cheaper per success is {parts}. The conclusion {verdict}.")
    return lines


def summary(
    spread: Sequence[Spread],
    success_spread: Sequence[SuccessSpread],
    failure: Sequence[Failure],
    per_model: Sequence[CpsResult],
    paired: Sequence[PairedResult],
    subsets: Sequence[SubsetCheck],
    flips: Sequence[FlipResult],
) -> str:
    matched = [s for s in spread if s.view in (GPT_VIEW, CLAUDE_MATCHED)]
    worst = max(spread, key=lambda s: s.ratio_worst)
    overall_ratios = [f.overall_ratio for f in failure]
    within_medians = [f.within_median for f in failure]
    mean_ratios = [f.overall_mean_ratio for f in failure]
    changes = "; ".join(
        f"{c.domain}: "
        + (
            "none"
            if c.ranking_changes == 0
            else f"{c.ranking_changes:,} of {N_RESAMPLES:,} ({c.ranking_changes / N_RESAMPLES:.1%})"
        )
        for c in subsets
    )
    first_inside = all(c.random_range[0] <= c.first_four <= c.random_range[1] for c in subsets)
    subset_line = f"- Random 4-of-8 picks that change the cheaper model: {changes}.\n" + (
        "- Claude's first-4 value is inside the random-4 95% range in every domain, so"
        " using the first 4 trials is not an unusual pick."
        if first_inside
        else "- Claude's first-4 value falls outside the random-4 95% range in at least"
        " one domain, so the choice of trials matters there."
    )
    biggest_flip = max(flips, key=lambda r: r.flip_probability[1])
    overall_tables = failure_tables(failure)
    cv_by = {(s.view, s.domain): s.cv_p50 for s in spread}
    airline_varies_more = all(cv_by[(v, "airline")] > cv_by[(v, "retail")] for v in VIEWS)
    domain_line = (
        "- Airline costs vary more than retail for every view (higher median CV)."
        if airline_varies_more
        else "- Neither domain varies more for every view."
    )
    gap_line = (
        "Much of the pooled gap comes from failures being concentrated in harder, longer "
        "tasks, not from failing itself."
        if max(within_medians) < min(overall_ratios)
        else "Failing itself accounts for a large part of the pooled gap."
    )
    return f"""# tau-bench re-analysis: results

Generated by `analyze.py` from `results/runs.csv`. Random seed {SEED}, {N_RESAMPLES:,}
bootstrap resamples, {N_RESAMPLES:,} random subsets in the Claude 4-of-8 check and
{N_FLIP_DRAWS:,} draws per flip simulation. See `../README.md` for how the costs were
estimated and their limitations.

## Two ideas used below

- **Coefficient of variation (CV):** the standard deviation of a task's run costs divided
  by their mean. A CV of 0.3 means a run typically costs about 30% more or less than that
  task's average run. Because it is relative, cheap and expensive tasks can be compared.
- **Bootstrap confidence interval:** we resample tasks with replacement 2,000 times
  (keeping each task's trials together, since runs of the same task are alike), recompute
  the number each time, and report the middle 95% of the results. It shows how much the
  number could move if tau-bench had contained a different mix of similar tasks.

## Cost definitions

{chr(10).join(f"- **{name}**: {text}" for name, text in COST_DEFINITIONS.items())}

Questions 1 and 2 look within one model and use `main`. Questions 3 and 4 compare models
and use all {N_DEFINITIONS}.

## 1. How much does cost vary across repeats of the same task?

Per task: CV and max/min ratio of run cost across trials; then the median (p50) and 90th
percentile (p90) across tasks. Models should be compared on the 4-trial views: with 8
trials there are more chances to draw an extreme run, so max/min grows by construction.

{spread_table(spread)}

- With 4 trials, the costliest run of a typical task costs
  **{min(s.ratio_p50 for s in matched):.2f}x to {max(s.ratio_p50 for s in matched):.2f}x**
  the cheapest (median max/min across tasks), and
  **{min(s.ratio_p90 for s in matched):.2f}x to {max(s.ratio_p90 for s in matched):.2f}x**
  for the 10% most variable tasks.
- The most extreme task: **{worst.ratio_worst:.1f}x** ({worst.view.label}, {worst.domain}).
{domain_line}

**Successful runs only** (4 trials per task, tasks with at least 2 successful runs). A
failed run can be cheap simply because the agent gave up early, so this checks that cost
still varies when the agent completes the same task correctly.

{success_spread_table(success_spread)}

![Fig 1](fig1.png)

## 2. Do failed runs cost more than successful ones?

Both headline ratios use medians, so a few very long runs cannot dominate them:

- **Pooled:** median cost of a failed run / median cost of a successful run, all runs
  of a model and domain together.
- **Within the same task:** for each task, median failed-run cost / median
  successful-run cost; then the median of those ratios across tasks.

The pooled ratio of means is shown alongside for comparison.

**Overall** (all runs pooled):

{overall_tables[0]}

**Within the same task** (only tasks with both successful and failed runs):

{overall_tables[1]}

- Pooled, failed runs cost **{min(overall_ratios):.2f}x to {max(overall_ratios):.2f}x**
  as much as successful ones (medians), or
  **{min(mean_ratios):.2f}x to {max(mean_ratios):.2f}x** using means.
- Within the same task, the median ratio is only
  **{min(within_medians):.2f}x to {max(within_medians):.2f}x**. {gap_line}

## 3. Cost per successful task

Total cost of all runs (failed ones included) divided by the number of successes, with
95% bootstrap confidence intervals (resampling tasks).

{cps_table(per_model)}

**Paired comparison:** GPT-4o's cost per success divided by Claude's, on the same tasks
(GPT-4o's 4 trials vs Claude's first 4), resampling tasks together for both models.
Below 1.0 means GPT-4o is cheaper per success; above 1.0, Claude.

{paired_table(paired)}

**Sensitivity check: which 4 Claude trials?** The paired comparison uses Claude's first 4
trials to match GPT-4o's 4. Here Claude's cost per success (`main`) is computed from all
8 trials, from the first 4, and from {N_RESAMPLES:,} random picks of 4 of the 8 trials per
task. The last column counts random picks that would change the cheaper model (GPT-4o's
all-trials cost per success vs Claude's) under any of the {N_DEFINITIONS} cost definitions.

{subset_table(subsets)}

{subset_line}

![Fig 2](fig2.png)

## 4. How often does comparing on fewer trials pick the wrong cheaper model?

For k trials per task (picked at random, without replacement, {N_FLIP_DRAWS:,} times), how
often does the GPT-4o vs Claude comparison reach the opposite conclusion from all 4
trials? k = 4 is the full data itself, so its flip chance is 0% by definition. The last
column shows how widely the GPT-4o / Claude cost-per-success ratio itself moves when
each task contributes only one trial per model (median and the middle 95% of draws).

{flip_table(flips)}

- The largest risk is **{biggest_flip.flip_probability[1]:.1%} at k=1**
  ({biggest_flip.domain}, `{biggest_flip.definition}`), where the full-data ratio is
  {biggest_flip.reference_ratio:.2f}, close to 1.0. Where the models differ clearly, a
  single trial per task almost never flips the answer.
- This measures only trial-to-trial noise for this fixed set of tasks, not the
  uncertainty from the cost assumptions themselves (see below).

![Fig 3](fig3.png)

## What holds under all {N_DEFINITIONS} cost definitions

{chr(10).join(robustness_lines(paired))}

Within-model findings (questions 1 and 2) do not depend on the cost definition in the
same way: each definition adds a fixed amount per call to every run of a model.
"""


def main() -> None:
    runs = load_runs()
    spread = spreads(runs)
    failure = failures(runs)
    per_model = cps_results(runs)
    paired = paired_results(runs)
    flips = flip_results(runs)

    OUT_DIR.mkdir(exist_ok=True)
    fig_spread(runs, OUT_DIR / "fig1.png")
    fig_cps(per_model, OUT_DIR / "fig2.png")
    fig_flip(flips, OUT_DIR / "fig3.png")
    fig_blog(runs, OUT_DIR / "blog_fig1.png")
    text = summary(
        spread, success_spreads(runs), failure, per_model, paired, subset_checks(runs), flips
    )
    (OUT_DIR / "summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
