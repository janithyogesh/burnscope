"""Tests for shared data types."""

import pytest
from pydantic import ValidationError

from burnscope.models import TokenUsage


def test_token_usage_defaults_to_zero() -> None:
    usage = TokenUsage()

    assert usage.total == 0


def test_total_adds_every_kind_once() -> None:
    usage = TokenUsage(
        uncached_input=100, cached_input=200, cache_write=50, output=30, reasoning=20
    )

    assert usage.total == 400


def test_rejects_negative_counts() -> None:
    with pytest.raises(ValidationError):
        TokenUsage(output=-1)


def test_rejects_unknown_fields() -> None:
    # A typo like "outptu" must fail loudly instead of being silently dropped.
    with pytest.raises(ValidationError):
        TokenUsage.model_validate({"outptu": 5})


@pytest.mark.parametrize("bad_value", [1.5, "10", True])
def test_rejects_non_integer_counts(bad_value: object) -> None:
    with pytest.raises(ValidationError):
        TokenUsage.model_validate({"output": bad_value})


def test_is_immutable() -> None:
    usage = TokenUsage(output=5)

    with pytest.raises(ValidationError):
        usage.output = 10  # type: ignore[misc]
