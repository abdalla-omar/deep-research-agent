from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .models import BudgetLedger
from .token_utils import estimate_tokens

MODEL_PRICING_PER_1M = {
    # Conservative placeholders; tune for your real billing profile.
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "tavily-search": {"input": 0.50, "output": 0.0},
}


@dataclass
class CallPlan:
    allowed: bool
    reason: str
    projected_input_tokens: int
    projected_output_tokens: int
    projected_cost_usd: float
    use_degraded_profile: bool = False


class BudgetGovernor:
    def __init__(self, ledger: BudgetLedger):
        self.ledger = ledger

    @staticmethod
    def estimate_call_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = MODEL_PRICING_PER_1M.get(model, MODEL_PRICING_PER_1M["gpt-4.1-mini"])
        return (input_tokens / 1_000_000) * pricing["input"] + (
            output_tokens / 1_000_000
        ) * pricing["output"]

    def preflight(
        self,
        *,
        model: str,
        input_text: str,
        projected_output_tokens: int,
        operation_name: str,
    ) -> CallPlan:
        input_tokens = estimate_tokens(input_text, model=model)
        projected_cost = self.estimate_call_cost(model, input_tokens, projected_output_tokens)

        if input_tokens > self.ledger.max_context_tokens_per_call:
            return CallPlan(
                allowed=False,
                reason=(
                    f"{operation_name}: projected context {input_tokens} exceeds "
                    f"limit {self.ledger.max_context_tokens_per_call}"
                ),
                projected_input_tokens=input_tokens,
                projected_output_tokens=projected_output_tokens,
                projected_cost_usd=projected_cost,
            )

        if self.ledger.total_cost_usd + projected_cost > self.ledger.max_session_cost_usd:
            # first ask caller to degrade
            return CallPlan(
                allowed=False,
                reason=(
                    f"{operation_name}: projected session cost "
                    f"{self.ledger.total_cost_usd + projected_cost:.5f} exceeds "
                    f"limit {self.ledger.max_session_cost_usd:.5f}"
                ),
                projected_input_tokens=input_tokens,
                projected_output_tokens=projected_output_tokens,
                projected_cost_usd=projected_cost,
                use_degraded_profile=not self.ledger.degrade_mode,
            )

        return CallPlan(
            allowed=True,
            reason="ok",
            projected_input_tokens=input_tokens,
            projected_output_tokens=projected_output_tokens,
            projected_cost_usd=projected_cost,
        )

    def record_call(
        self,
        *,
        model: str,
        actual_input_tokens: int,
        actual_output_tokens: int,
        operation_name: str,
    ) -> None:
        call_cost = self.estimate_call_cost(model, actual_input_tokens, actual_output_tokens)
        self.ledger.total_tokens += actual_input_tokens + actual_output_tokens
        self.ledger.total_cost_usd += call_cost
        self.ledger.max_context_tokens_seen = max(
            self.ledger.max_context_tokens_seen, actual_input_tokens
        )

        if self.ledger.total_cost_usd > self.ledger.max_session_cost_usd:
            self.ledger.constraint_violations.append(
                f"{operation_name}: session cost exceeded hard limit"
            )

    def check_runtime(self, operation_name: str) -> bool:
        elapsed = (datetime.now(tz=timezone.utc) - self.ledger.started_at).total_seconds()
        if elapsed > self.ledger.max_runtime_seconds:
            self.ledger.constraint_violations.append(
                f"{operation_name}: runtime exceeded {self.ledger.max_runtime_seconds}s"
            )
            return False
        return True

    def activate_degrade_mode(self, reason: str) -> None:
        self.ledger.degrade_mode = True
        self.ledger.constraint_violations.append(f"degrade-mode: {reason}")
