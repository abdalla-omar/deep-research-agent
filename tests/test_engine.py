from research_agent.models import ResearchRequest
from research_agent.orchestrator import DeepResearchEngine


def test_engine_offline_path_returns_structured_response() -> None:
    engine = DeepResearchEngine()
    request = ResearchRequest(
        query="Compare memory strategies for deep research agents under strict token limits.",
    )
    response = engine.run(request)

    assert response.query_breakdown
    assert response.trace.subquestions_processed >= 1
    assert response.trace.evidence_cards_created >= 1
    assert "## Evidence-backed Findings" in response.answer_markdown
    assert response.budget_report.max_context_tokens_seen <= 2000


def test_engine_respects_custom_constraints() -> None:
    engine = DeepResearchEngine()
    request = ResearchRequest(
        query="How should an enterprise design budget-aware retrieval for AI research workflows?",
        constraints={
            "max_context_tokens_per_call": 1200,
            "max_session_cost_usd": 0.02,
            "max_runtime_seconds": 120,
        },
    )
    response = engine.run(request)
    assert response.budget_report.total_cost_usd <= 0.02 + 1e-6
