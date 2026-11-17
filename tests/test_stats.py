"""Tests for the statistics helpers, on inputs small enough to check by hand."""

import math

import pytest

from burnscope.analysis.stats import (
    bootstrap_ci,
    cost_per_success,
    cv,
    max_min_ratio,
    percentile,
)


def test_cv_is_sample_std_over_mean() -> None:
    # mean 2, sample standard deviation 1
    assert cv([1.0, 2.0, 3.0]) == pytest.approx(0.5)


def test_cv_of_identical_values_is_zero() -> None:
    assert cv([4.0, 4.0, 4.0]) == 0.0


def test_cv_needs_two_values() -> None:
    with pytest.raises(ValueError, match="at least two"):
        cv([1.0])


def test_max_min_ratio() -> None:
    assert max_min_ratio([0.5, 2.0, 1.0]) == pytest.approx(4.0)


def test_max_min_ratio_rejects_zero() -> None:
    with pytest.raises(ValueError, match="positive"):
        max_min_ratio([0.0, 1.0])


@pytest.mark.parametrize(
    ("q", "expected"),
    [(0.0, 1.0), (0.5, 2.5), (0.9, 3.7), (1.0, 4.0)],
)
def test_percentile_interpolates_like_numpy(q: float, expected: float) -> None:
    assert percentile([4.0, 1.0, 3.0, 2.0], q) == pytest.approx(expected)


def test_percentile_of_one_value() -> None:
    assert percentile([7.0], 0.9) == 7.0


def test_cost_per_success_counts_failed_runs_cost() -> None:
    # $3 spent in total, 2 successes
    assert cost_per_success([1.0, 1.0, 1.0], [True, False, True]) == pytest.approx(1.5)


def test_cost_per_success_without_successes_is_infinite() -> None:
    assert cost_per_success([1.0], [False]) == math.inf


def test_bootstrap_of_identical_groups_has_no_spread() -> None:
    groups = [[2.0, 2.0]] * 10

    low, high = bootstrap_ci(groups, lambda gs: sum(sum(g) for g in gs) / len(gs))

    assert (low, high) == pytest.approx((4.0, 4.0))


def test_bootstrap_is_repeatable_with_a_seed() -> None:
    groups = [[float(i)] for i in range(20)]

    def mean(gs: list[list[float]]) -> float:
        return sum(g[0] for g in gs) / len(gs)

    assert bootstrap_ci(groups, mean, seed=1) == bootstrap_ci(groups, mean, seed=1)
    assert bootstrap_ci(groups, mean, seed=1) != bootstrap_ci(groups, mean, seed=2)


def test_bootstrap_interval_surrounds_the_estimate() -> None:
    groups = [[float(i)] for i in range(20)]

    low, high = bootstrap_ci(groups, lambda gs: sum(g[0] for g in gs) / len(gs))

    assert low < 9.5 < high


def test_bootstrap_keeps_groups_together() -> None:
    # Every group has exactly 3 items; if groups were split, sizes would vary.
    groups = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    sizes: set[int] = set()

    def record(gs: list[list[float]]) -> float:
        sizes.update(len(g) for g in gs)
        return 0.0

    bootstrap_ci(groups, record, n_resamples=50)

    assert sizes == {3}
