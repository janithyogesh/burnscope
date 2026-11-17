"""Model prices and the cost calculator.

Prices live in a YAML file, in US dollars per million tokens:

    currency: USD
    providers:
      openai:
        models:
          gpt-4o-2024-08-06:
            input: 2.50
            output: 10.00
            cached_input: 1.25          # optional, see fallback rules below
            source: https://...         # where the price was published
            checked: 2026-11-17         # when someone last checked it
            verified: true              # false if the price could not be confirmed
      ollama:
        local: true                     # runs on your machine: every model costs $0

Fallback rules for optional prices: reasoning tokens use the output price, and
cached input and cache writes use the normal input price (no discount known).
"""

from datetime import date
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, ValidationError, model_validator

from burnscope.models import TokenUsage

TOKENS_PER_PRICE_UNIT = 1_000_000


class PricingError(Exception):
    """A pricing file is invalid, or a model has no price."""


class ModelPrice(BaseModel):
    """Published prices for one model, in dollars per million tokens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input: NonNegativeFloat
    output: NonNegativeFloat
    cached_input: NonNegativeFloat | None = None
    cache_write: NonNegativeFloat | None = None
    reasoning: NonNegativeFloat | None = None
    source: str | None = None
    checked: date | None = None
    verified: bool = True

    @property
    def effective_cached_input(self) -> float:
        """Price for cached input; the normal input price if none is published."""
        return self.input if self.cached_input is None else self.cached_input

    @property
    def effective_cache_write(self) -> float:
        """Price for cache writes; the normal input price if none is published."""
        return self.input if self.cache_write is None else self.cache_write

    @property
    def effective_reasoning(self) -> float:
        """Price for reasoning tokens; the output price if none is published."""
        return self.output if self.reasoning is None else self.reasoning


FREE = ModelPrice(input=0.0, output=0.0, source="local model: no per-token price")


class ProviderPricing(BaseModel):
    """Prices for every model of one provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    local: bool = False
    models: dict[str, ModelPrice] = {}

    @model_validator(mode="after")
    def _paid_models_need_a_source(self) -> Self:
        if self.local:
            return self
        for name, price in self.models.items():
            if price.source is None or price.checked is None:
                raise ValueError(f"model {name!r} needs both 'source' and 'checked'")
        return self


class PricingTable(BaseModel):
    """All prices from one pricing file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    currency: Literal["USD"]
    providers: dict[str, ProviderPricing]

    def price_for(self, provider: str, model: str) -> ModelPrice:
        """Return the price of `model` from `provider`.

        Every model of a local provider is free. Raises PricingError if the
        provider or model is not in the table.
        """
        if provider not in self.providers:
            known = ", ".join(sorted(self.providers)) or "none"
            raise PricingError(
                f"No prices for provider {provider!r} (known providers: {known}). "
                "Add it to your pricing file."
            )
        entry = self.providers[provider]
        if entry.local:
            return FREE
        if model not in entry.models:
            raise PricingError(
                f"No price for {provider}/{model}. Add it under providers.{provider}.models "
                "in your pricing file."
            )
        return entry.models[model]

    def unverified(self) -> list[str]:
        """Models whose price could not be confirmed, as 'provider/model'."""
        return [
            f"{provider}/{name}"
            for provider, entry in self.providers.items()
            for name, price in entry.models.items()
            if not price.verified
        ]

    def stale(self, today: date, max_age_days: int = 90) -> list[str]:
        """Models whose price was last checked more than `max_age_days` ago."""
        return [
            f"{provider}/{name}"
            for provider, entry in self.providers.items()
            for name, price in entry.models.items()
            if price.checked is not None and (today - price.checked).days > max_age_days
        ]


class CostBreakdown(BaseModel):
    """Cost of one or more model calls in dollars, split by token kind."""

    model_config = ConfigDict(frozen=True)

    uncached_input: float = 0.0
    cached_input: float = 0.0
    cache_write: float = 0.0
    output: float = 0.0
    reasoning: float = 0.0

    @property
    def total(self) -> float:
        """Cost of every token kind together."""
        return (
            self.uncached_input
            + self.cached_input
            + self.cache_write
            + self.output
            + self.reasoning
        )


def cost_of(usage: TokenUsage, price: ModelPrice) -> CostBreakdown:
    """Calculate what `usage` costs at `price`."""

    def dollars(tokens: int, price_per_million: float) -> float:
        return tokens * price_per_million / TOKENS_PER_PRICE_UNIT

    return CostBreakdown(
        uncached_input=dollars(usage.uncached_input, price.input),
        cached_input=dollars(usage.cached_input, price.effective_cached_input),
        cache_write=dollars(usage.cache_write, price.effective_cache_write),
        output=dollars(usage.output, price.output),
        reasoning=dollars(usage.reasoning, price.effective_reasoning),
    )


def load_pricing(path: Path) -> PricingTable:
    """Read and validate a pricing YAML file.

    Raises PricingError with the file name and the problem if the file is
    missing, is not valid YAML, or does not match the expected format.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PricingError(f"Pricing file not found: {path}") from None
    except yaml.YAMLError as error:
        raise PricingError(f"Pricing file {path} is not valid YAML:\n{error}") from None
    try:
        return PricingTable.model_validate(raw)
    except ValidationError as error:
        raise PricingError(f"Pricing file {path} has problems:\n{error}") from None
