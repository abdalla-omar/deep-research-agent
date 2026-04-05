from __future__ import annotations

from .models import EvidenceCard


def format_evidence_for_prompt(cards: list[EvidenceCard]) -> str:
    lines = []
    for c in cards:
        lines.append(
            f"[{c.id}] claim={c.claim_candidate}\n"
            f"source={c.source_title} | {c.source_url}\n"
            f"excerpt={c.supporting_excerpt[:350]}"
        )
    return "\n\n".join(lines)


def build_synthesis_prompt(
    *,
    query: str,
    query_breakdown: list[str],
    working_memory: str,
    summary_memory: str,
    evidence_cards: list[EvidenceCard],
    citations_required: bool,
) -> str:
    citation_instruction = (
        "Every non-trivial factual statement must include claim IDs in bracket format like [abc123]."
        if citations_required
        else "Add citations where possible."
    )
    breakdown = "\n- ".join(query_breakdown)

    return f"""
You are a deep research synthesis assistant.

User Query:
{query}

Sub-questions:
- {breakdown}

Working Memory:
{working_memory}

Summary Memory:
{summary_memory}

Evidence Cards:
{format_evidence_for_prompt(evidence_cards)}

Output Requirements:
1) Produce a concise but complete answer in Markdown.
2) Include a section "## Evidence-backed Findings".
3) Include a section "## Caveats and Unknowns".
4) {citation_instruction}
5) At end include "## Citations" list with claim_id -> URL mapping.
""".strip()
