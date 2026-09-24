"""V2.0 passive Claim contracts.

These contracts are deliberately not wired into research decisions. They make
the information that a future claim-verification pipeline must preserve
explicit without changing V1.1 extraction, acceptance, closure, or reporting.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CanonicalClaimType(StrEnum):
    FACTUAL = "factual"
    QUANTITATIVE = "quantitative"
    COMPARATIVE = "comparative"
    CAUSAL = "causal"
    DESCRIPTIVE = "descriptive"
    UNKNOWN = "unknown"


class ClaimEvidenceRelation(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    MENTIONS = "mentions"
    INSUFFICIENT = "insufficient"


class ClaimRelationType(StrEnum):
    SAME = "same"
    RELATED = "related"
    CONTRADICTS = "contradicts"
    UNKNOWN = "unknown"


class ClaimVerificationState(StrEnum):
    UNSUPPORTED = "unsupported"
    SUPPORTED = "supported"
    VERIFIED = "verified"
    CONFLICTED = "conflicted"
    UNKNOWN = "unknown"


class ClaimDiagnosticCode(StrEnum):
    NORMALIZATION_FAILED = "CLAIM_NORMALIZATION_FAILED"
    MISSING_SUBJECT = "CLAIM_MISSING_SUBJECT"
    MISSING_VALUE = "CLAIM_MISSING_VALUE"
    QUALIFIER_AMBIGUOUS = "QUALIFIER_AMBIGUOUS"
    SOURCE_IDENTITY_UNKNOWN = "SOURCE_IDENTITY_UNKNOWN"
    EVIDENCE_LINK_MISSING = "EVIDENCE_LINK_MISSING"
    VERIFICATION_NOT_COMPUTABLE = "VERIFICATION_NOT_COMPUTABLE"
    RELATION_UNKNOWN = "RELATION_UNKNOWN"


class ClaimProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: UUID | None = None
    source_id: UUID | None = None
    source_owner_key: str | None = Field(default=None, min_length=1, max_length=255)
    quote_reference: str | None = None
    method: str = Field(default="unspecified", min_length=1, max_length=100)


class CanonicalClaim(BaseModel):
    """A proposition record; entity identity is not semantic identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: UUID
    run_id: UUID
    question_id: str = Field(min_length=1, max_length=50)
    dimension_key: str = Field(min_length=1, max_length=100)
    normalized_text: str = Field(min_length=1, max_length=4000)
    original_text: str = Field(min_length=1, max_length=4000)
    claim_type: CanonicalClaimType = CanonicalClaimType.UNKNOWN
    subject: str | None = Field(default=None, max_length=1000)
    predicate: str | None = Field(default=None, max_length=1000)
    object_or_value: str | None = Field(default=None, max_length=2000)
    qualifiers: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    polarity: bool | None = None
    provenance: tuple[ClaimProvenance, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def normalized_text_must_preserve_original_content(self) -> CanonicalClaim:
        if not self.original_text.strip() or not self.normalized_text.strip():
            raise ValueError("claim text must not be blank")
        return self

    @classmethod
    def from_text(
        cls,
        *,
        claim_id: UUID,
        original_text: str,
        **kwargs: Any,
    ) -> CanonicalClaim:
        # V2.0 normalization is intentionally lexical and qualifier-preserving.
        normalized = re.sub(r"\s+", " ", original_text).strip()
        return cls(
            claim_id=claim_id,
            original_text=original_text,
            normalized_text=normalized,
            **kwargs,
        )


class ClaimEvidenceLink(BaseModel):
    """Evidence-to-Claim edge with explicit source and quote provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: UUID
    evidence_id: UUID
    relation: ClaimEvidenceRelation
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    source_id: UUID
    source_owner_key: str | None = Field(default=None, min_length=1, max_length=255)
    quote_reference: str | None = None
    created_by: str = Field(default="unspecified", min_length=1, max_length=100)

    def validate_scope(
        self,
        *,
        claim: CanonicalClaim,
        run_id: UUID,
        question_id: str,
        dimension_key: str,
        evidence_source_id: UUID,
        evidence_owner_key: str | None,
    ) -> None:
        """Raise on cross-scope/mis-provenanced links; never infer owner identity."""

        if self.claim_id != claim.claim_id:
            raise ValueError("claim/evidence link references a different Claim")
        if run_id != claim.run_id or question_id != claim.question_id:
            raise ValueError("claim/evidence link crosses run or question scope")
        if dimension_key != claim.dimension_key:
            raise ValueError("claim/evidence link crosses dimension scope")
        if self.source_id != evidence_source_id:
            raise ValueError("claim/evidence link source differs from evidence source")
        if self.source_owner_key != evidence_owner_key:
            raise ValueError("source_owner_key must be propagated unchanged from Source")


class ClaimRelation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    left_claim_id: UUID
    right_claim_id: UUID
    relation: ClaimRelationType
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str | None = None

    @model_validator(mode="after")
    def distinct_claims(self) -> ClaimRelation:
        if self.left_claim_id == self.right_claim_id:
            raise ValueError("a Claim relation must connect distinct claim records")
        return self


class ClaimVerificationResult(BaseModel):
    """A state plus explanation; UNKNOWN is not silently converted to false."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: UUID
    state: ClaimVerificationState
    reason: str | None = None
    supporting_evidence_ids: tuple[UUID, ...] = ()
    contradicting_evidence_ids: tuple[UUID, ...] = ()
    independent_owner_keys: tuple[str, ...] = ()


def exact_proposition_key(claim: CanonicalClaim) -> str:
    """Return a provisional lexical key, never a semantic-equivalence verdict."""

    return re.sub(r"\s+", " ", claim.normalized_text).strip().casefold()


def deterministic_relation_candidate(
    left: CanonicalClaim,
    right: CanonicalClaim,
) -> ClaimRelationType:
    """Recognize only explicit structured matches; defer semantic language work.

    This is a candidate primitive, not a verification decision. Missing or
    ambiguous structure returns UNKNOWN rather than guessing.
    """

    if exact_proposition_key(left) == exact_proposition_key(right):
        return ClaimRelationType.SAME
    if not left.subject or not right.subject or not left.predicate or not right.predicate:
        return ClaimRelationType.UNKNOWN
    if left.subject.casefold() != right.subject.casefold():
        return ClaimRelationType.UNKNOWN
    if left.predicate.casefold() != right.predicate.casefold():
        return ClaimRelationType.UNKNOWN

    qualifier_keys = set(left.qualifiers) | set(right.qualifiers)
    differing = {
        key for key in qualifier_keys if left.qualifiers.get(key) != right.qualifiers.get(key)
    }
    if differing:
        return ClaimRelationType.RELATED
    if left.polarity is not None and right.polarity is not None and left.polarity != right.polarity:
        return ClaimRelationType.CONTRADICTS

    left_value = _canonical_value(left.object_or_value, left.qualifiers.get("unit"))
    right_value = _canonical_value(right.object_or_value, right.qualifiers.get("unit"))
    if left_value is not None and right_value is not None:
        return (
            ClaimRelationType.SAME if left_value == right_value else ClaimRelationType.CONTRADICTS
        )
    if (
        left.object_or_value
        and right.object_or_value
        and left.object_or_value.strip().casefold() == right.object_or_value.strip().casefold()
    ):
        return ClaimRelationType.SAME
    return ClaimRelationType.UNKNOWN


def _canonical_value(value: str | None, qualifier_unit: object) -> tuple[Decimal, str] | None:
    if value is None:
        return None
    match = re.fullmatch(
        r"\s*([+-]?[\d,]+(?:\.\d+)?)\s*(billion|bn|b|million|mn|m|thousand|k)?\s*([A-Za-z%$]*)\s*",
        value,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    try:
        amount = Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None
    scale = {
        "billion": Decimal(1_000_000_000),
        "bn": Decimal(1_000_000_000),
        "b": Decimal(1_000_000_000),
        "million": Decimal(1_000_000),
        "mn": Decimal(1_000_000),
        "m": Decimal(1_000_000),
        "thousand": Decimal(1_000),
        "k": Decimal(1_000),
    }.get((match.group(2) or "").casefold(), Decimal(1))
    unit = (match.group(3) or str(qualifier_unit or "")).casefold()
    unit = {"$": "usd", "us$": "usd", "usd": "usd"}.get(unit, unit)
    return amount * scale, unit
