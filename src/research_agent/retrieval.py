from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

import httpx

from .models import EvidenceCard, SubQuestion
from .token_utils import estimate_tokens


class TavilyClient:
    def __init__(self, api_key: str | None = None, timeout: float = 30.0):
        self.api_key = api_key
        self.timeout = timeout

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.api_key:
            return self._offline_stub(query, top_k)

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "query": query,
            "search_depth": "advanced",
            "max_results": top_k,
            "include_answer": False,
            "include_raw_content": True,
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post("https://api.tavily.com/search", headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        return data.get("results", [])

    def _offline_stub(self, query: str, top_k: int) -> list[dict[str, Any]]:
        seeds = [
            {
                "title": "Open source deep research patterns",
                "url": "https://example.com/deep-research-patterns",
                "content": "Multi-query retrieval and compression improve answer quality while controlling context size.",
            },
            {
                "title": "RAG memory architecture",
                "url": "https://example.com/rag-memory",
                "content": "Hybrid memory combines short-term working memory with episodic vector storage and summary memory.",
            },
            {
                "title": "Cost-aware LLM orchestration",
                "url": "https://example.com/cost-aware-orchestration",
                "content": "A budget governor enforces token and dollar limits, using degrade modes before hard stop.",
            },
        ]
        return seeds[:top_k]


def generate_query_variants(subquestion: SubQuestion) -> list[str]:
    q = subquestion.text.strip()
    variants = [
        q,
        f"{q} best practices",
        f"{q} trade-offs and benchmarks",
    ]
    return variants


def score_source_quality(url: str) -> float:
    domain = re.sub(r"^https?://", "", url).split("/")[0]
    high_quality = ["arxiv.org", "github.com", "docs.", ".gov", ".edu"]
    if any(token in domain for token in high_quality):
        return 0.9
    return 0.6


def score_recency(_: str) -> float:
    # Placeholder: keep deterministic. Can be replaced by metadata date parsing.
    return 0.5


def normalize_to_evidence_cards(
    *,
    session_id: str,
    subquestion: SubQuestion,
    raw_results: list[dict[str, Any]],
) -> list[EvidenceCard]:
    cards: list[EvidenceCard] = []
    for idx, r in enumerate(raw_results):
        content = (r.get("content") or "").strip()
        title = (r.get("title") or "Untitled source").strip()
        url = (r.get("url") or "https://example.com/unknown").strip()
        claim_candidate = content.split(".")[0][:240].strip() or f"Finding {idx + 1} for {subquestion.text}"

        card_id = hashlib.sha1(f"{session_id}|{subquestion.id}|{idx}|{url}".encode()).hexdigest()[:12]
        cards.append(
            EvidenceCard(
                id=card_id,
                session_id=session_id,
                subquestion_id=subquestion.id,
                claim_candidate=claim_candidate,
                supporting_excerpt=content[:900],
                source_url=url,
                source_title=title,
                retrieval_score=max(0.2, 1.0 - idx * 0.12),
                source_quality=score_source_quality(url),
                recency_score=score_recency(url),
                novelty_score=max(0.3, 0.9 - idx * 0.08),
                estimated_tokens=estimate_tokens(claim_candidate + "\n" + content),
                created_at=datetime.now(tz=timezone.utc),
            )
        )
    return cards
