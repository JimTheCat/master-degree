"""Cost estimation and budget enforcement for LLM API calls."""

from dataclasses import dataclass, field

# Approximate pricing per 1M tokens (USD) as of early 2026
MODEL_PRICING = {
    # OpenAI
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    # Anthropic
    "claude-opus-5": {"input": 5.0, "output": 25.0},
    # claude-sonnet-5: $2/$10 intro pricing through 2026-08-31, then $3/$15.
    "claude-sonnet-5": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},
    # Local models (free)
    "ollama": {"input": 0.0, "output": 0.0},
}


@dataclass
class CostEstimate:
    model_name: str
    total_speeches: int
    avg_input_tokens: float
    avg_output_tokens: float
    estimated_total_input_tokens: int
    estimated_total_output_tokens: int
    estimated_cost_usd: float
    batch_api_cost_usd: float  # 50% discount via OpenAI Batch API


@dataclass
class CostTracker:
    """Track cumulative cost during a run."""
    budget_usd: float = 100.0
    warn_at_usd: float = 50.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    _model_name: str = ""
    _warned: bool = False

    def record(self, input_tokens: int, output_tokens: int):
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd = self._calculate_cost()

    def _calculate_cost(self) -> float:
        pricing = _get_pricing(self._model_name)
        input_cost = (self.total_input_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.total_output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost

    def check_budget(self) -> tuple[bool, str]:
        """Check if we're within budget. Returns (ok, message)."""
        if self.total_cost_usd >= self.budget_usd:
            return False, f"Budget exceeded: ${self.total_cost_usd:.2f} >= ${self.budget_usd:.2f}"
        if not self._warned and self.total_cost_usd >= self.warn_at_usd:
            self._warned = True
            return True, f"Warning: cost at ${self.total_cost_usd:.2f} (budget: ${self.budget_usd:.2f})"
        return True, ""


def _get_pricing(model_name: str) -> dict[str, float]:
    """Get pricing for a model, with fallback."""
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]
    # Check prefix matches
    for key, pricing in MODEL_PRICING.items():
        if model_name.startswith(key):
            return pricing
    # Local/unknown models are free
    return {"input": 0.0, "output": 0.0}


def estimate_cost(
    model_name: str,
    total_speeches: int,
    avg_input_tokens: float,
    avg_output_tokens: float,
) -> CostEstimate:
    """Estimate the total cost for classifying a corpus."""
    pricing = _get_pricing(model_name)

    total_input = int(total_speeches * avg_input_tokens)
    total_output = int(total_speeches * avg_output_tokens)

    cost = (total_input / 1_000_000) * pricing["input"] + \
           (total_output / 1_000_000) * pricing["output"]

    return CostEstimate(
        model_name=model_name,
        total_speeches=total_speeches,
        avg_input_tokens=avg_input_tokens,
        avg_output_tokens=avg_output_tokens,
        estimated_total_input_tokens=total_input,
        estimated_total_output_tokens=total_output,
        estimated_cost_usd=cost,
        batch_api_cost_usd=cost * 0.5,
    )
