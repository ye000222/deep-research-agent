"""Phase 14.0-A Evidence failure classification model.

Explains *why* collected evidence is not enough to close a ``GapRequirement``.
The Phase 13.x feedback loop already guarantees every non-CLOSED gap carries a
``ClosureFeedback``; this module upgrades the explanation from "needs more
evidence" to "needs a primary source / independent validation / quantitative
support / method description / ...".

Design boundaries:

* Pure domain logic -- no database, no search, no scheduler imports.
* Every rule reads only generic state: requirement counters, verification
  status, evidence alignment status and optional per-item evidence/source
  metadata.  Optional signals are tri-state (``None`` means "not observed")
  so a rule never fires on absent data and no rule is tied to a question id,
  gap id, dimension key or benchmark case.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final
from uuid import UUID

from app.domain.evidence_alignment import EvidenceAlignmentStatus
from app.domain.gap_closure import (
    GapClosureStatus,
    GapRequirement,
    GapRequirementType,
    VerificationStatus,
)


class EvidenceFailureReason(StrEnum):
    NO_PRIMARY_SOURCE = "no_primary_source"
    NO_INDEPENDENT_VALIDATION = "no_independent_validation"
    CLAIM_NOT_SUPPORTED = "claim_not_supported"
    INSUFFICIENT_DETAIL = "insufficient_detail"
    NO_QUANTITATIVE_SUPPORT = "no_quantitative_support"
    NO_METHOD_DESCRIPTION = "no_method_description"
    SOURCE_QUALITY_LOW = "source_quality_low"
    OUTDATED_SOURCE = "outdated_source"
    DIMENSION_MISMATCH = "dimension_mismatch"
    UNKNOWN = "unknown"


class MissingEvidenceType(StrEnum):
    """The kind of evidence a still-open requirement is missing."""

    PRIMARY_SOURCE = "primary_source"
    INDEPENDENT_VALIDATION = "independent_validation"
    VERIFICATION_SUPPORT = "verification_support"
    SUPPORTING_EVIDENCE = "supporting_evidence"
    EVIDENCE_DETAIL = "evidence_detail"
    QUANTITATIVE_DATA = "quantitative_data"
    METHOD_DESCRIPTION = "method_description"
    HIGH_QUALITY_SOURCE = "high_quality_source"
    UP_TO_DATE_SOURCE = "up_to_date_source"
    DIMENSION_ALIGNED_EVIDENCE = "dimension_aligned_evidence"
    UNKNOWN = "unknown"


#: Conservative, domain-agnostic defaults.  They classify metadata quality,
#: never a specific research topic or benchmark.
MIN_SOURCE_RELIABILITY: Final[float] = 0.4
MIN_EVIDENCE_CONFIDENCE: Final[float] = 0.4
MIN_EVIDENCE_RELEVANCE: Final[float] = 0.4
OUTDATED_SOURCE_DAYS: Final[float] = 3650.0

_REASON_TO_MISSING_TYPE: Final[dict[EvidenceFailureReason, MissingEvidenceType]] = {
    EvidenceFailureReason.NO_PRIMARY_SOURCE: MissingEvidenceType.PRIMARY_SOURCE,
    EvidenceFailureReason.NO_INDEPENDENT_VALIDATION: (
        MissingEvidenceType.INDEPENDENT_VALIDATION
    ),
    EvidenceFailureReason.CLAIM_NOT_SUPPORTED: MissingEvidenceType.VERIFICATION_SUPPORT,
    EvidenceFailureReason.INSUFFICIENT_DETAIL: MissingEvidenceType.EVIDENCE_DETAIL,
    EvidenceFailureReason.NO_QUANTITATIVE_SUPPORT: (
        MissingEvidenceType.QUANTITATIVE_DATA
    ),
    EvidenceFailureReason.NO_METHOD_DESCRIPTION: (
        MissingEvidenceType.METHOD_DESCRIPTION
    ),
    EvidenceFailureReason.SOURCE_QUALITY_LOW: MissingEvidenceType.HIGH_QUALITY_SOURCE,
    EvidenceFailureReason.OUTDATED_SOURCE: MissingEvidenceType.UP_TO_DATE_SOURCE,
    EvidenceFailureReason.DIMENSION_MISMATCH: (
        MissingEvidenceType.DIMENSION_ALIGNED_EVIDENCE
    ),
    EvidenceFailureReason.UNKNOWN: MissingEvidenceType.UNKNOWN,
}


def missing_type_for_reason(reason: EvidenceFailureReason) -> MissingEvidenceType:
    """Return the generic missing-evidence category one reason points at."""

    return _REASON_TO_MISSING_TYPE[reason]


@dataclass(frozen=True, slots=True)
class EvidenceItemFacts:
    """Generic per-item evidence/source metadata (``None`` = not observed)."""

    evidence_id: UUID
    alignment_status: EvidenceAlignmentStatus = EvidenceAlignmentStatus.UNKNOWN
    accepted: bool = False
    source_type: str | None = None
    source_reliability: float | None = None
    relevance: float | None = None
    confidence: float | None = None
    is_primary_source: bool | None = None
    has_quantitative_data: bool | None = None
    has_method_description: bool | None = None
    age_days: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "evidence_id": str(self.evidence_id),
            "alignment_status": self.alignment_status.value,
            "accepted": self.accepted,
            "source_type": self.source_type,
            "source_reliability": self.source_reliability,
            "relevance": self.relevance,
            "confidence": self.confidence,
            "is_primary_source": self.is_primary_source,
            "has_quantitative_data": self.has_quantitative_data,
            "has_method_description": self.has_method_description,
            "age_days": self.age_days,
        }


@dataclass(frozen=True, slots=True)
class EvidenceFailureClassification:
    """Deterministic explanation of which evidence qualities are missing."""

    gap_id: UUID
    claim_id: UUID | None
    reasons: tuple[EvidenceFailureReason, ...]
    primary_reason: EvidenceFailureReason

    def as_dict(self) -> dict[str, object]:
        return {
            "gap_id": str(self.gap_id),
            "claim_id": str(self.claim_id) if self.claim_id is not None else None,
            "failure_reasons": [reason.value for reason in self.reasons],
            "primary_failure_reason": self.primary_reason.value,
        }


def classify_evidence_failure(
    requirement: GapRequirement,
    facts: Sequence[EvidenceItemFacts] = (),
    *,
    claim_id: UUID | None = None,
) -> EvidenceFailureClassification:
    """Classify why one requirement is not closed yet.

    Rules are evaluated in a fixed order so the result is deterministic and
    identical for any question, dimension, gap id or domain.  A rule only
    fires on *observed* signals; absent metadata yields ``UNKNOWN`` rather
    than a fabricated defect.
    """

    items = tuple(facts)
    reasons: list[EvidenceFailureReason] = []

    # Rule 1: evidence exists but nothing aligns with the required dimension.
    if items and all(
        item.alignment_status is EvidenceAlignmentStatus.NOT_ALIGNED for item in items
    ):
        reasons.append(EvidenceFailureReason.DIMENSION_MISMATCH)

    # Rule 2: the claim still demands verification that was never achieved.
    if (
        requirement.requirement_type is GapRequirementType.CLAIM_VERIFICATION
        or requirement.verification_status is VerificationStatus.REQUIRED
    ) and requirement.verification_status is not VerificationStatus.VERIFIED:
        reasons.append(EvidenceFailureReason.CLAIM_NOT_SUPPORTED)

    # Rule 3: not enough independent sources to validate what was found.
    if (
        requirement.current_independent_sources
        < requirement.required_independent_sources
    ):
        reasons.append(EvidenceFailureReason.NO_INDEPENDENT_VALIDATION)

    # Rule 4: nothing was collected at all -> no primary source observed.
    if not items and requirement.current_evidence_count == 0:
        reasons.append(EvidenceFailureReason.NO_PRIMARY_SOURCE)

    accepted_items = tuple(item for item in items if item.accepted) or items

    # Rule 5: primary-source flag observed on items, yet no item is primary.
    primary_flags = [
        item.is_primary_source for item in accepted_items if item.is_primary_source
        is not None
    ]
    if primary_flags and not any(primary_flags):
        reasons.append(EvidenceFailureReason.NO_PRIMARY_SOURCE)

    # Rule 6: every observed reliability value sits below the floor.
    reliabilities = [
        item.source_reliability
        for item in accepted_items
        if item.source_reliability is not None
    ]
    if reliabilities and all(
        value < MIN_SOURCE_RELIABILITY for value in reliabilities
    ):
        reasons.append(EvidenceFailureReason.SOURCE_QUALITY_LOW)

    # Rule 7: observed confidence/relevance signal is uniformly weak.
    confidences = [
        item.confidence for item in accepted_items if item.confidence is not None
    ]
    relevances = [
        item.relevance for item in accepted_items if item.relevance is not None
    ]
    weak_confidence = bool(confidences) and all(
        value < MIN_EVIDENCE_CONFIDENCE for value in confidences
    )
    weak_relevance = bool(relevances) and all(
        value < MIN_EVIDENCE_RELEVANCE for value in relevances
    )
    if weak_confidence or weak_relevance:
        reasons.append(EvidenceFailureReason.INSUFFICIENT_DETAIL)

    # Rule 8: quantitative/methodology flags observed, never satisfied.
    quantitative_flags = [
        item.has_quantitative_data
        for item in accepted_items
        if item.has_quantitative_data is not None
    ]
    if quantitative_flags and not any(quantitative_flags):
        reasons.append(EvidenceFailureReason.NO_QUANTITATIVE_SUPPORT)
    method_flags = [
        item.has_method_description
        for item in accepted_items
        if item.has_method_description is not None
    ]
    if method_flags and not any(method_flags):
        reasons.append(EvidenceFailureReason.NO_METHOD_DESCRIPTION)

    # Rule 9: every observed publication age exceeds the staleness bound.
    ages = [item.age_days for item in accepted_items if item.age_days is not None]
    if ages and all(value > OUTDATED_SOURCE_DAYS for value in ages):
        reasons.append(EvidenceFailureReason.OUTDATED_SOURCE)

    if not reasons:
        # Rule 10: the gap is not closed but no generic signal explains it.
        reasons.append(EvidenceFailureReason.UNKNOWN)

    return EvidenceFailureClassification(
        gap_id=requirement.gap_id,
        claim_id=claim_id,
        reasons=tuple(reasons),
        primary_reason=reasons[0],
    )


def is_closed_requirement(requirement: GapRequirement) -> bool:
    """Convenience predicate so callers never classify CLOSED gaps."""

    return requirement.closure_status is GapClosureStatus.CLOSED
