"""Shared data types used across burnscope."""

from pydantic import BaseModel, ConfigDict, NonNegativeInt


class TokenUsage(BaseModel):
    """Tokens used by one model call, split into non-overlapping kinds.

    Each token is counted in exactly one field. Providers often include reasoning
    tokens inside their output count; here `output` is visible output only, so
    adding the fields never counts a token twice.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    uncached_input: NonNegativeInt = 0
    """Prompt tokens the model read fresh (no cache hit)."""
    cached_input: NonNegativeInt = 0
    """Prompt tokens served from the provider's prompt cache."""
    cache_write: NonNegativeInt = 0
    """Prompt tokens written into the cache for reuse by later calls."""
    output: NonNegativeInt = 0
    """Visible output tokens (the answer), excluding reasoning."""
    reasoning: NonNegativeInt = 0
    """Hidden thinking tokens produced before the answer."""

    @property
    def total(self) -> int:
        """All tokens of every kind."""
        return (
            self.uncached_input
            + self.cached_input
            + self.cache_write
            + self.output
            + self.reasoning
        )
