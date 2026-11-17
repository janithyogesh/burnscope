"""Small statistics helpers for cost analysis, in plain Python."""

import math
import random
import statistics
from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def cv(values: Sequence[float]) -> float:
    """Coefficient of variation: sample standard deviation divided by the mean.

    A CV of 0.3 means values typically sit about 30% above or below their mean.
    Needs at least two values and a positive mean.
    """
    if len(values) < 2:
        raise ValueError("cv needs at least two values")
    mean = statistics.fmean(values)
    if mean <= 0:
        raise ValueError("cv needs a positive mean")
    return statistics.stdev(values) / mean


def max_min_ratio(values: Sequence[float]) -> float:
    """Largest value divided by the smallest (e.g. costliest run / cheapest run)."""
    if not values:
        raise ValueError("max_min_ratio needs at least one value")
    smallest = min(values)
    if smallest <= 0:
        raise ValueError("max_min_ratio needs positive values")
    return max(values) / smallest


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th quantile (0 <= q <= 1) with linear interpolation.

    Same rule as numpy's default: percentile([1, 2, 3, 4], 0.9) == 3.7.
    """
    if not values:
        raise ValueError("percentile needs at least one value")
    if not 0 <= q <= 1:
        raise ValueError("q must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def cost_per_success(costs: Sequence[float], successes: Sequence[bool]) -> float:
    """Total cost of all runs (failed ones included) divided by the number of successes.

    Returns infinity when no run succeeded.
    """
    if len(costs) != len(successes):
        raise ValueError("costs and successes must have the same length")
    n_successes = sum(successes)
    return math.inf if n_successes == 0 else math.fsum(costs) / n_successes


def bootstrap_ci(
    groups: Sequence[T],
    statistic: Callable[[list[T]], float],
    *,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Confidence interval for `statistic` by resampling whole groups.

    Each resample draws len(groups) groups with replacement, keeping every group
    intact (e.g. all trials of one task stay together), and recomputes the
    statistic. The interval is the middle `level` share of those results.
    """
    if not groups:
        raise ValueError("bootstrap_ci needs at least one group")
    rng = random.Random(seed)
    results = [statistic(rng.choices(groups, k=len(groups))) for _ in range(n_resamples)]
    tail = (1 - level) / 2
    return percentile(results, tail), percentile(results, 1 - tail)
