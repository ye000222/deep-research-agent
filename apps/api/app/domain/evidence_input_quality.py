"""Explain and improve evidence inputs without changing canonical acceptance.

This module deliberately has no persistence or decision-loop dependencies.  It
provides conservative diagnostics plus a claim-aware quote locator that may be
used before the existing acceptance gate.  The acceptance gate remains the
canonical source of truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from app.domain.adaptive_scheduler import claim_quote_entails, source_role_fits_claim


class SourceRoleAudit(StrEnum):
    ROLE_CLASSIFICATION_TOO_COARSE = "ROLE_CLASSIFICATION_TOO_COARSE"
    ROLE_METADATA_MISSING = "ROLE_METADATA_MISSING"
    TRUE_SOURCE_ROLE_MISMATCH = "TRUE_SOURCE_ROLE_MISMATCH"
    MULTI_CAUSE = "MULTI_CAUSE"
    UNKNOWN = "UNKNOWN"


class QuoteSupportAudit(StrEnum):
    SUPPORT_EXISTS_WRONG_QUOTE = "SUPPORT_EXISTS_WRONG_QUOTE"
    SOURCE_RELEVANT_BUT_NO_DIRECT_SUPPORT = "SOURCE_RELEVANT_BUT_NO_DIRECT_SUPPORT"
    SOURCE_DOES_NOT_SUPPORT_CLAIM = "SOURCE_DOES_NOT_SUPPORT_CLAIM"
    QUOTE_TRUNCATED_OR_MALFORMED = "QUOTE_TRUNCATED_OR_MALFORMED"
    CLAIM_TOO_BROAD_FOR_PASSAGE = "CLAIM_TOO_BROAD_FOR_PASSAGE"
    UNKNOWN = "UNKNOWN"


class Suitability(StrEnum):
    SUITABLE = "SUITABLE"
    PARTIAL = "PARTIAL"
    UNSUITABLE = "UNSUITABLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class EvidenceInputQualityAssessment:
    source_suitability: Suitability
    source_role_audit: SourceRoleAudit
    quote_support: QuoteSupportAudit
    final_candidate_disposition: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source_suitability": self.source_suitability.value,
            "source_role_audit": self.source_role_audit.value,
            "quote_support": self.quote_support.value,
            "final_candidate_disposition": self.final_candidate_disposition,
        }


_MEDIA_ROLES = frozenset({"webpage", "html", "pdf", "document", "unknown", ""})
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9._%-]+", re.IGNORECASE)


def assess_source_suitability(
    *,
    claim_type: str,
    observed_role: str | None,
    expected_role: str | None = None,
    metadata_complete: bool = True,
) -> tuple[Suitability, SourceRoleAudit]:
    """Assess requirement/source compatibility, never generic page quality."""

    if not metadata_complete or not observed_role:
        return Suitability.UNKNOWN, SourceRoleAudit.ROLE_METADATA_MISSING
    role = observed_role.casefold()
    if role in _MEDIA_ROLES:
        return Suitability.UNKNOWN, SourceRoleAudit.ROLE_CLASSIFICATION_TOO_COARSE
    if expected_role and role != expected_role.casefold():
        return Suitability.UNSUITABLE, SourceRoleAudit.TRUE_SOURCE_ROLE_MISMATCH
    if source_role_fits_claim(claim_type=claim_type, source_role=role):
        return Suitability.SUITABLE, SourceRoleAudit.UNKNOWN
    return Suitability.UNSUITABLE, SourceRoleAudit.TRUE_SOURCE_ROLE_MISMATCH


def classify_quote_support(
    *, claim: str, quote: str, source_text: str
) -> QuoteSupportAudit:
    """Classify quote failure conservatively; absence of proof stays unknown."""

    if not quote.strip() or len(quote.strip()) < 10:
        return QuoteSupportAudit.QUOTE_TRUNCATED_OR_MALFORMED
    if quote.casefold() not in source_text.casefold():
        if select_supporting_quote(claim=claim, source_text=source_text) is not None:
            return QuoteSupportAudit.SUPPORT_EXISTS_WRONG_QUOTE
        return QuoteSupportAudit.SOURCE_RELEVANT_BUT_NO_DIRECT_SUPPORT
    if claim_quote_entails(claim, quote):
        return QuoteSupportAudit.UNKNOWN
    claim_words = set(_WORD_RE.findall(claim.casefold()))
    quote_words = set(_WORD_RE.findall(quote.casefold()))
    if claim_words and len(claim_words & quote_words) >= max(1, int(len(claim_words) * 0.45)):
        return QuoteSupportAudit.CLAIM_TOO_BROAD_FOR_PASSAGE
    return QuoteSupportAudit.SOURCE_DOES_NOT_SUPPORT_CLAIM


def select_supporting_quote(*, claim: str, source_text: str) -> str | None:
    """Return one contiguous sentence likely to support a claim, if any."""

    sentences = [
        piece.strip()
        for piece in re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+", source_text)
        if piece.strip()
    ]
    claim_words = set(_WORD_RE.findall(claim.casefold()))
    best: tuple[float, str] | None = None
    for sentence in sentences:
        if len(sentence) < 10:
            continue
        words = set(_WORD_RE.findall(sentence.casefold()))
        overlap = len(claim_words & words) / max(1, len(claim_words))
        if overlap < 0.45 or not claim_quote_entails(claim, sentence):
            continue
        score = overlap + min(len(sentence), 500) / 10_000
        if best is None or score > best[0]:
            best = (score, sentence[:2000])
    return best[1] if best else None


def assess_candidate(
    *,
    claim: str,
    quote: str,
    source_text: str,
    claim_type: str,
    observed_role: str | None,
    expected_role: str | None = None,
    metadata_complete: bool = True,
    accepted: bool = False,
) -> EvidenceInputQualityAssessment:
    suitability, role_audit = assess_source_suitability(
        claim_type=claim_type,
        observed_role=observed_role,
        expected_role=expected_role,
        metadata_complete=metadata_complete,
    )
    quote_support = classify_quote_support(claim=claim, quote=quote, source_text=source_text)
    return EvidenceInputQualityAssessment(
        source_suitability=suitability,
        source_role_audit=role_audit,
        quote_support=quote_support,
        final_candidate_disposition="accepted" if accepted else "rejected",
    )


__all__ = [
    "EvidenceInputQualityAssessment",
    "QuoteSupportAudit",
    "SourceRoleAudit",
    "Suitability",
    "assess_candidate",
    "assess_source_suitability",
    "classify_quote_support",
    "select_supporting_quote",
]
