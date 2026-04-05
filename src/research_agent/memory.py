from __future__ import annotations

from collections import deque
from typing import Any

from .models import EvidenceCard, MemorySnapshot
from .token_utils import estimate_tokens


class HybridMemoryManager:
    def __init__(
        self,
        *,
        max_context_tokens_per_call: int,
        model: str = "gpt-4.1",
        max_working_turns: int = 8,
        memory_mode: str = "hybrid",
    ):
        self.max_context_tokens_per_call = max_context_tokens_per_call
        self.model = model
        self.memory_mode = memory_mode
        self.working_memory = deque(maxlen=max_working_turns)
        self.summary_memory = ""
        self.evidence_cards: list[EvidenceCard] = []

    def add_working_turn(self, turn: dict[str, Any]) -> None:
        self.working_memory.append(turn)

    def add_evidence_card(self, card: EvidenceCard) -> None:
        if card.estimated_tokens == 0:
            card.estimated_tokens = estimate_tokens(
                card.claim_candidate + "\n" + card.supporting_excerpt,
                model=self.model,
            )
        self.evidence_cards.append(card)

    def compact_every_n_cards(self, n: int = 2) -> None:
        if len(self.evidence_cards) == 0 or len(self.evidence_cards) % n != 0:
            return

        latest = self.evidence_cards[-n:]
        fragments = [
            f"- [{card.id}] {card.claim_candidate} (src: {card.source_title})"
            for card in latest
        ]
        delta_summary = "\n".join(fragments)
        if not self.summary_memory:
            self.summary_memory = delta_summary
        else:
            self.summary_memory = f"{self.summary_memory}\n{delta_summary}"[-6000:]

    def utility(self, card: EvidenceCard) -> float:
        return (
            0.45 * card.retrieval_score
            + 0.25 * card.novelty_score
            + 0.20 * card.source_quality
            + 0.10 * card.recency_score
        )

    def build_context_pack(
        self,
        *,
        planner_tokens: int = 250,
        summary_tokens: int = 450,
        evidence_tokens: int = 1050,
        instructions_tokens: int = 250,
    ) -> dict[str, Any]:
        budget_total = planner_tokens + summary_tokens + evidence_tokens + instructions_tokens
        hard_budget = min(self.max_context_tokens_per_call, budget_total)

        # working memory section
        working_text = "\n".join(str(x) for x in self.working_memory)
        working_text = self._truncate_to_tokens(working_text, planner_tokens)

        # summary section
        summary_text = self._truncate_to_tokens(self.summary_memory, summary_tokens)
        if self.memory_mode in {"none", "vector-only"}:
            summary_text = ""

        # evidence section via utility-per-token packing
        evidence_candidates = sorted(
            self.evidence_cards,
            key=lambda card: self.utility(card) / max(card.estimated_tokens, 1),
            reverse=True,
        )
        if self.memory_mode in {"none", "summary-only"}:
            evidence_candidates = []

        selected: list[EvidenceCard] = []
        used_tokens = 0
        for card in evidence_candidates:
            card_tokens = max(12, card.estimated_tokens)
            if used_tokens + card_tokens > evidence_tokens:
                continue
            selected.append(card)
            used_tokens += card_tokens

        # If over hard budget, prune lowest utility first.
        total_tokens = (
            estimate_tokens(working_text, model=self.model)
            + estimate_tokens(summary_text, model=self.model)
            + used_tokens
            + instructions_tokens
        )
        while total_tokens > hard_budget and selected:
            selected.sort(key=self.utility)
            removed = selected.pop(0)
            used_tokens -= max(12, removed.estimated_tokens)
            total_tokens = (
                estimate_tokens(working_text, model=self.model)
                + estimate_tokens(summary_text, model=self.model)
                + used_tokens
                + instructions_tokens
            )

        return {
            "working_memory": working_text,
            "summary_memory": summary_text,
            "evidence_cards": [card.model_dump() for card in selected],
            "estimated_context_tokens": total_tokens,
        }

    def _truncate_to_tokens(self, text: str, token_budget: int) -> str:
        if estimate_tokens(text, model=self.model) <= token_budget:
            return text
        out = text
        while out and estimate_tokens(out, model=self.model) > token_budget:
            out = out[int(len(out) * 0.9) :]
        return out

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(
            working_memory=list(self.working_memory),
            summary_memory=self.summary_memory,
            evidence_cards=self.evidence_cards,
        )
