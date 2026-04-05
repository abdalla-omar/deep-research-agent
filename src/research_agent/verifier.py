from __future__ import annotations

import re

from .models import EvidenceCard, VerificationResult


CLAIM_ID_PATTERN = re.compile(r"\[([a-f0-9]{6,12})\]")


def verify_claim_coverage(answer_markdown: str, evidence_cards: list[EvidenceCard]) -> VerificationResult:
    found_claims = set(CLAIM_ID_PATTERN.findall(answer_markdown))
    known_claims = {c.id for c in evidence_cards}

    supported = sorted(found_claims & known_claims)
    unsupported = sorted(found_claims - known_claims)

    if not found_claims:
        return VerificationResult(
            supported_claim_ids=[],
            unsupported_claim_ids=[],
            score=0.0,
            notes="No claim IDs were found in answer.",
        )

    score = len(supported) / max(1, len(found_claims))
    notes = "ok" if not unsupported else "Some claim IDs are not supported by selected evidence cards."

    return VerificationResult(
        supported_claim_ids=supported,
        unsupported_claim_ids=unsupported,
        score=score,
        notes=notes,
    )
