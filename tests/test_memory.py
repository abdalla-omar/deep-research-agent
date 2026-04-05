from datetime import datetime, timezone

from research_agent.memory import HybridMemoryManager
from research_agent.models import EvidenceCard


def test_context_pack_stays_within_budget() -> None:
    memory = HybridMemoryManager(max_context_tokens_per_call=2000)

    for i in range(12):
        card = EvidenceCard(
            id=f"c{i}",
            session_id="s1",
            subquestion_id="sq1",
            claim_candidate=f"Claim {i}",
            supporting_excerpt="Evidence excerpt " * 30,
            source_url=f"https://example.com/{i}",
            source_title=f"Source {i}",
            retrieval_score=0.9 - i * 0.02,
            source_quality=0.8,
            recency_score=0.5,
            novelty_score=0.7,
            estimated_tokens=120,
            created_at=datetime.now(tz=timezone.utc),
        )
        memory.add_evidence_card(card)

    packed = memory.build_context_pack()
    assert packed["estimated_context_tokens"] <= 2000
    assert len(packed["evidence_cards"]) > 0


def test_compaction_happens_every_two_cards() -> None:
    memory = HybridMemoryManager(max_context_tokens_per_call=2000)
    card_template = dict(
        session_id="s1",
        subquestion_id="sq1",
        supporting_excerpt="abc",
        source_url="https://example.com",
        source_title="Example",
        retrieval_score=0.8,
        source_quality=0.8,
        recency_score=0.5,
        novelty_score=0.7,
        estimated_tokens=32,
        created_at=datetime.now(tz=timezone.utc),
    )

    memory.add_evidence_card(EvidenceCard(id="a1", claim_candidate="A", **card_template))
    memory.compact_every_n_cards()
    assert memory.summary_memory == ""

    memory.add_evidence_card(EvidenceCard(id="a2", claim_candidate="B", **card_template))
    memory.compact_every_n_cards()
    assert "[a1]" in memory.summary_memory or "[a2]" in memory.summary_memory
