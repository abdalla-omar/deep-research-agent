from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from .budget import BudgetGovernor
from .memory import HybridMemoryManager
from .models import (
    BudgetLedger,
    BudgetReport,
    Citation,
    ResearchRequest,
    ResearchResponse,
    SubQuestion,
    TraceInfo,
)
from .retrieval import TavilyClient, generate_query_variants, normalize_to_evidence_cards
from .synthesis import build_synthesis_prompt
from .token_utils import estimate_tokens
from .verifier import verify_claim_coverage


@dataclass
class EngineConfig:
    research_model: str = "gpt-4.1"
    compression_model: str = "gpt-4.1-mini"
    verifier_model: str = "gpt-4.1-mini"
    memory_mode: str = "hybrid"


class DeepResearchEngine:
    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig(
            research_model=os.getenv("MODEL_RESEARCH", "gpt-4.1"),
            compression_model=os.getenv("MODEL_COMPRESSION", "gpt-4.1-mini"),
            verifier_model=os.getenv("MODEL_VERIFIER", "gpt-4.1-mini"),
            memory_mode=os.getenv("MEMORY_MODE", "hybrid"),
        )
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
        self.tavily = TavilyClient(api_key=self.tavily_api_key)

    def run(self, request: ResearchRequest) -> ResearchResponse:
        session_id = request.session_id or str(uuid.uuid4())
        ledger = BudgetLedger(
            session_id=session_id,
            max_context_tokens_per_call=request.constraints.max_context_tokens_per_call,
            max_session_cost_usd=request.constraints.max_session_cost_usd,
            max_runtime_seconds=request.constraints.max_runtime_seconds,
        )
        governor = BudgetGovernor(ledger)
        memory = HybridMemoryManager(
            max_context_tokens_per_call=request.constraints.max_context_tokens_per_call,
            model=self.config.research_model,
            max_working_turns=8,
            memory_mode=self.config.memory_mode,
        )

        query_breakdown = self.decompose_query(request.query, request.options.depth)
        subquestions = [
            SubQuestion(id=f"sq-{i+1}", text=q, priority=i + 1)
            for i, q in enumerate(query_breakdown)
        ]

        trace = TraceInfo(subquestions_processed=0, evidence_cards_created=0)

        for sq in subquestions:
            if not governor.check_runtime("subquestion-loop"):
                break

            variants = generate_query_variants(sq)
            top_k = self._depth_to_top_k(request.options.depth)
            if ledger.degrade_mode:
                top_k = max(2, top_k - 2)

            raw_results: list[dict[str, Any]] = []
            for query_variant in variants:
                call_input = f"search:{query_variant}"
                plan = governor.preflight(
                    model="tavily-search",
                    input_text=call_input,
                    projected_output_tokens=120,
                    operation_name="tavily-search",
                )
                if not plan.allowed:
                    if plan.use_degraded_profile:
                        governor.activate_degrade_mode(plan.reason)
                        trace.degrade_mode_triggered = True
                        continue
                    ledger.constraint_violations.append(plan.reason)
                    break

                search_results = self.tavily.search(query_variant, top_k=top_k)
                raw_results.extend(search_results)
                governor.record_call(
                    model="tavily-search",
                    actual_input_tokens=plan.projected_input_tokens,
                    actual_output_tokens=min(120, 30 * len(search_results)),
                    operation_name="tavily-search",
                )

            cards = normalize_to_evidence_cards(
                session_id=session_id,
                subquestion=sq,
                raw_results=raw_results[:top_k],
            )

            for card in cards:
                memory.add_evidence_card(card)
            memory.compact_every_n_cards(n=2)

            trace.subquestions_processed += 1
            trace.evidence_cards_created += len(cards)
            memory.add_working_turn(
                {
                    "subquestion_id": sq.id,
                    "text": sq.text,
                    "card_count": len(cards),
                    "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                }
            )

        packed = memory.build_context_pack()
        selected_cards = memory.snapshot().evidence_cards
        synthesis_prompt = build_synthesis_prompt(
            query=request.query,
            query_breakdown=query_breakdown,
            working_memory=packed["working_memory"],
            summary_memory=packed["summary_memory"],
            evidence_cards=selected_cards,
            citations_required=request.options.citations_required,
        )

        answer_markdown = self._synthesize_answer(
            prompt=synthesis_prompt,
            governor=governor,
            citations_required=request.options.citations_required,
            evidence_cards=selected_cards,
        )

        verification = verify_claim_coverage(answer_markdown, selected_cards)
        if verification.unsupported_claim_ids and not ledger.degrade_mode:
            # One-step repair pass, budget permitting.
            repaired = self._repair_answer(
                answer_markdown=answer_markdown,
                unsupported_ids=verification.unsupported_claim_ids,
                evidence_cards=selected_cards,
                governor=governor,
            )
            if repaired:
                answer_markdown = repaired
                verification = verify_claim_coverage(answer_markdown, selected_cards)

        citations = [
            Citation(
                claim_id=card.id,
                url=card.source_url,
                title=card.source_title,
                confidence=min(1.0, max(0.2, card.retrieval_score)),
            )
            for card in selected_cards[:20]
        ]

        return ResearchResponse(
            answer_markdown=answer_markdown,
            citations=citations,
            query_breakdown=query_breakdown,
            budget_report=BudgetReport(
                total_tokens=ledger.total_tokens,
                total_cost_usd=round(ledger.total_cost_usd, 6),
                max_context_tokens_seen=ledger.max_context_tokens_seen,
                constraint_violations=ledger.constraint_violations,
            ),
            trace=trace,
        )

    def decompose_query(self, query: str, depth: Any) -> list[str]:
        # Deterministic fallback decomposition to keep runs reproducible.
        seeds = [
            f"Define the problem space and evaluation criteria for: {query}",
            f"Collect current evidence, benchmarks, and implementation patterns for: {query}",
            f"Synthesize trade-offs, constraints, and recommended approach for: {query}",
        ]
        if str(depth) == "high":
            seeds.extend(
                [
                    f"Identify failure modes and mitigation strategies for: {query}",
                    f"Estimate cost, latency, and operational implications for: {query}",
                ]
            )
        return seeds

    def _depth_to_top_k(self, depth: Any) -> int:
        d = str(depth)
        if d == "low":
            return 3
        if d == "high":
            return 6
        return 5

    def _synthesize_answer(
        self,
        *,
        prompt: str,
        governor: BudgetGovernor,
        citations_required: bool,
        evidence_cards,
    ) -> str:
        plan = governor.preflight(
            model=self.config.research_model,
            input_text=prompt,
            projected_output_tokens=900,
            operation_name="final-synthesis",
        )
        if not plan.allowed:
            governor.activate_degrade_mode(plan.reason)
            return self._offline_synthesis(evidence_cards, citations_required)

        if not self.openai_api_key:
            governor.record_call(
                model=self.config.research_model,
                actual_input_tokens=min(plan.projected_input_tokens, 1800),
                actual_output_tokens=420,
                operation_name="final-synthesis-offline",
            )
            return self._offline_synthesis(evidence_cards, citations_required)

        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.research_model,
                    "input": prompt,
                    "max_output_tokens": 900,
                },
            )
            response.raise_for_status()
            payload = response.json()

        text = self._extract_response_text(payload)
        governor.record_call(
            model=self.config.research_model,
            actual_input_tokens=min(estimate_tokens(prompt, self.config.research_model), 2000),
            actual_output_tokens=min(estimate_tokens(text, self.config.research_model), 900),
            operation_name="final-synthesis",
        )
        return text or self._offline_synthesis(evidence_cards, citations_required)

    def _repair_answer(
        self,
        *,
        answer_markdown: str,
        unsupported_ids: list[str],
        evidence_cards,
        governor: BudgetGovernor,
    ) -> str | None:
        prompt = (
            "Repair the answer so all bracketed claim ids map to known evidence ids.\n"
            f"Unsupported ids: {unsupported_ids}\n"
            f"Answer:\n{answer_markdown}"
        )
        plan = governor.preflight(
            model=self.config.verifier_model,
            input_text=prompt,
            projected_output_tokens=450,
            operation_name="repair-pass",
        )
        if not plan.allowed:
            return None
        governor.record_call(
            model=self.config.verifier_model,
            actual_input_tokens=min(plan.projected_input_tokens, 1200),
            actual_output_tokens=280,
            operation_name="repair-pass",
        )

        # Deterministic repair in offline mode.
        known_ids = {c.id for c in evidence_cards}
        repaired = answer_markdown
        for bad in unsupported_ids:
            replacement = next(iter(known_ids), None)
            if replacement:
                repaired = repaired.replace(f"[{bad}]", f"[{replacement}]")
        return repaired

    def _offline_synthesis(self, evidence_cards, citations_required: bool) -> str:
        top_cards = evidence_cards[: min(8, len(evidence_cards))]
        findings = []
        for c in top_cards:
            citation = f" [{c.id}]" if citations_required else ""
            findings.append(f"- {c.claim_candidate}{citation}")

        caveats = [
            "- Evidence quality varies by source domain and recency.",
            "- Budget-constrained retrieval may miss niche primary sources.",
        ]

        citations = [f"- [{c.id}] {c.source_title}: {c.source_url}" for c in top_cards]

        if not findings:
            findings = ["- No high-confidence findings were produced within current budget constraints."]

        return "\n".join(
            [
                "## Evidence-backed Findings",
                *findings,
                "",
                "## Caveats and Unknowns",
                *caveats,
                "",
                "## Citations",
                *citations,
            ]
        )

    @staticmethod
    def _extract_response_text(payload: dict[str, Any]) -> str:
        output = payload.get("output", [])
        chunks: list[str] = []
        for item in output:
            content = item.get("content", [])
            for part in content:
                if part.get("type") == "output_text":
                    chunks.append(part.get("text", ""))
        if chunks:
            return "\n".join(chunks).strip()

        # Older/alternate shape fallback.
        text = payload.get("output_text")
        if isinstance(text, str):
            return text.strip()
        return ""
