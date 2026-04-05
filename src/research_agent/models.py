from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator


class Depth(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Constraints(BaseModel):
    max_context_tokens_per_call: int = Field(default=2000, ge=256, le=16000)
    max_session_cost_usd: float = Field(default=0.05, gt=0, le=5.0)
    max_runtime_seconds: int = Field(default=480, ge=30, le=3600)


class Options(BaseModel):
    depth: Depth = Depth.MEDIUM
    citations_required: bool = True


class ResearchRequest(BaseModel):
    query: str = Field(min_length=8, max_length=4000)
    session_id: str | None = Field(default=None, min_length=4, max_length=128)
    constraints: Constraints = Field(default_factory=Constraints)
    options: Options = Field(default_factory=Options)

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()


class Citation(BaseModel):
    claim_id: str
    url: HttpUrl
    title: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class BudgetReport(BaseModel):
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    max_context_tokens_seen: int = 0
    constraint_violations: list[str] = Field(default_factory=list)


class TraceInfo(BaseModel):
    subquestions_processed: int = 0
    evidence_cards_created: int = 0
    degrade_mode_triggered: bool = False


class ResearchResponse(BaseModel):
    answer_markdown: str
    citations: list[Citation]
    query_breakdown: list[str]
    budget_report: BudgetReport
    trace: TraceInfo


class BudgetLedger(BaseModel):
    session_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    max_context_tokens_per_call: int
    max_session_cost_usd: float
    max_runtime_seconds: int
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    max_context_tokens_seen: int = 0
    degrade_mode: bool = False
    constraint_violations: list[str] = Field(default_factory=list)


class SubQuestion(BaseModel):
    id: str
    text: str
    priority: int = 1


class EvidenceCard(BaseModel):
    id: str
    session_id: str
    subquestion_id: str
    claim_candidate: str
    supporting_excerpt: str
    source_url: str
    source_title: str
    retrieval_score: float = Field(ge=0.0, le=1.0)
    source_quality: float = Field(default=0.5, ge=0.0, le=1.0)
    recency_score: float = Field(default=0.5, ge=0.0, le=1.0)
    novelty_score: float = Field(default=0.5, ge=0.0, le=1.0)
    estimated_tokens: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class MemorySnapshot(BaseModel):
    working_memory: list[dict[str, Any]] = Field(default_factory=list)
    summary_memory: str = ""
    evidence_cards: list[EvidenceCard] = Field(default_factory=list)


class VerificationResult(BaseModel):
    supported_claim_ids: list[str] = Field(default_factory=list)
    unsupported_claim_ids: list[str] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str = ""
