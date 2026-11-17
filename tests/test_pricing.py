"""Tests for pricing files and the cost calculator."""

from datetime import date
from pathlib import Path

import pytest

from burnscope.models import TokenUsage
from burnscope.pricing import FREE, ModelPrice, PricingError, cost_of, load_pricing

VALID_YAML = """\
currency: USD
providers:
  acme:
    models:
      full-model:
        input: 2.0
        output: 8.0
        cached_input: 0.5
        cache_write: 2.5
        reasoning: 10.0
        source: https://example.com/pricing
        checked: 2026-11-01
      basic-model:
        input: 1.0
        output: 4.0
        source: https://example.com/pricing
        checked: 2026-01-01
        verified: false
  ollama:
    local: true
"""


@pytest.fixture
def pricing_file(tmp_path: Path) -> Path:
    path = tmp_path / "pricing.yaml"
    path.write_text(VALID_YAML, encoding="utf-8")
    return path


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "pricing.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --- loading ---------------------------------------------------------------


def test_loads_valid_file(pricing_file: Path) -> None:
    table = load_pricing(pricing_file)
    price = table.price_for("acme", "full-model")

    assert price.input == 2.0
    assert price.checked == date(2026, 11, 1)
    assert price.verified is True


def test_missing_file_gives_clear_error(tmp_path: Path) -> None:
    with pytest.raises(PricingError, match="not found"):
        load_pricing(tmp_path / "nope.yaml")


def test_invalid_yaml_gives_clear_error(tmp_path: Path) -> None:
    with pytest.raises(PricingError, match="not valid YAML"):
        load_pricing(write(tmp_path, "currency: [unclosed"))


def test_negative_price_is_rejected(tmp_path: Path) -> None:
    text = VALID_YAML.replace("input: 1.0", "input: -1.0")

    with pytest.raises(PricingError, match="basic-model"):
        load_pricing(write(tmp_path, text))


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    text = VALID_YAML.replace("output: 4.0", "outptu: 4.0")

    with pytest.raises(PricingError, match="outptu"):
        load_pricing(write(tmp_path, text))


def test_paid_model_without_source_is_rejected(tmp_path: Path) -> None:
    text = """\
currency: USD
providers:
  acme:
    models:
      m: {input: 1.0, output: 2.0, checked: 2026-11-01}
"""
    with pytest.raises(PricingError, match="source"):
        load_pricing(write(tmp_path, text))


def test_only_usd_is_supported(tmp_path: Path) -> None:
    with pytest.raises(PricingError, match="currency"):
        load_pricing(write(tmp_path, VALID_YAML.replace("USD", "EUR")))


# --- lookup ----------------------------------------------------------------


def test_unknown_model_error_says_how_to_fix(pricing_file: Path) -> None:
    table = load_pricing(pricing_file)

    with pytest.raises(PricingError, match=r"acme/mystery.*providers\.acme\.models"):
        table.price_for("acme", "mystery")


def test_unknown_provider_error_lists_known_ones(pricing_file: Path) -> None:
    table = load_pricing(pricing_file)

    with pytest.raises(PricingError, match="acme, ollama"):
        table.price_for("nobody", "m")


def test_local_provider_is_free_for_any_model(pricing_file: Path) -> None:
    table = load_pricing(pricing_file)

    assert table.price_for("ollama", "llama3.2") == FREE


def test_lists_unverified_models(pricing_file: Path) -> None:
    assert load_pricing(pricing_file).unverified() == ["acme/basic-model"]


def test_lists_stale_models(pricing_file: Path) -> None:
    table = load_pricing(pricing_file)

    assert table.stale(today=date(2026, 11, 17)) == ["acme/basic-model"]
    assert table.stale(today=date(2026, 11, 17), max_age_days=400) == []


# --- cost calculation --------------------------------------------------------


def test_cost_uses_each_kinds_own_price() -> None:
    price = ModelPrice(input=2.0, output=8.0, cached_input=0.5, cache_write=2.5, reasoning=10.0)
    usage = TokenUsage(
        uncached_input=1_000_000,
        cached_input=1_000_000,
        cache_write=1_000_000,
        output=1_000_000,
        reasoning=1_000_000,
    )

    cost = cost_of(usage, price)

    assert cost.uncached_input == pytest.approx(2.0)
    assert cost.cached_input == pytest.approx(0.5)
    assert cost.cache_write == pytest.approx(2.5)
    assert cost.output == pytest.approx(8.0)
    assert cost.reasoning == pytest.approx(10.0)
    assert cost.total == pytest.approx(23.0)


def test_missing_prices_fall_back() -> None:
    price = ModelPrice(input=1.0, output=4.0)
    usage = TokenUsage(cached_input=1_000_000, cache_write=1_000_000, reasoning=1_000_000)

    cost = cost_of(usage, price)

    assert cost.cached_input == pytest.approx(1.0)  # no discount known: input price
    assert cost.cache_write == pytest.approx(1.0)  # input price
    assert cost.reasoning == pytest.approx(4.0)  # output price


def test_realistic_call() -> None:
    # 3,000 input and 200 output tokens at $2.50 / $10 per million.
    price = ModelPrice(input=2.5, output=10.0)
    cost = cost_of(TokenUsage(uncached_input=3_000, output=200), price)

    assert cost.total == pytest.approx(0.0075 + 0.002)


def test_zero_usage_costs_nothing() -> None:
    assert cost_of(TokenUsage(), ModelPrice(input=5.0, output=15.0)).total == 0.0


def test_local_model_costs_nothing() -> None:
    assert cost_of(TokenUsage(uncached_input=10_000, output=500), FREE).total == 0.0
