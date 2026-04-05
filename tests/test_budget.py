from research_agent.budget import BudgetGovernor
from research_agent.models import BudgetLedger


def build_governor() -> BudgetGovernor:
    ledger = BudgetLedger(
        session_id="test-session",
        max_context_tokens_per_call=2000,
        max_session_cost_usd=0.05,
        max_runtime_seconds=480,
    )
    return BudgetGovernor(ledger)


def test_preflight_blocks_large_context() -> None:
    governor = build_governor()
    huge_text = "x" * 22000
    plan = governor.preflight(
        model="gpt-4.1",
        input_text=huge_text,
        projected_output_tokens=200,
        operation_name="unit-test",
    )
    assert not plan.allowed
    assert "exceeds" in plan.reason


def test_cost_limit_triggers_degrade_request() -> None:
    governor = build_governor()
    governor.ledger.total_cost_usd = 0.0499
    plan = governor.preflight(
        model="gpt-4.1",
        input_text="short text",
        projected_output_tokens=400,
        operation_name="unit-test-cost",
    )
    assert not plan.allowed
    assert plan.use_degraded_profile
