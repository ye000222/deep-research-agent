"""Phase 14.0-B Evidence quality analyzer.

Turns generic requirement state plus (optional) per-item evidence metadata
into one explainable quality assessment: which evidence failure reasons apply
and which concrete research direction they recommend.  A dedicated claim
analyzer adds the verification dimension, and ``build_feedback_analysis``
composes both into the flat, JSON-safe metadata payload that Phase 14.0
attaches to ``ClosureFeedback``.

Pure domain logic: no database access, no search trigger, no scheduler.  All
rules read only generic counters, statuses, and tri-state evidence signals.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from app.domain.claim_verification_analyzer import (
    ClaimVerificationAnalyzer,
    ClaimVerificationResult,
)
from app.domain.evidence_alignment import EvidenceAlignment, EvidenceAlignmentStatus
from app.domain.evidence_failure_classification import (
    EvidenceFailureClassification,
    EvidenceFailureReason,
    EvidenceItemFacts,
    MissingEvidenceType,
    classify_evidence_failure,
    missing_type_for_reason,
)
from app.domain.gap_closure import GapClosureStatus, GapRequirement
from app.domain.research_need import ResearchNeedType
from app.domain.research_need_refinement import refine_research_need


class EvidenceQualityStatus(StrEnum):
    SUFFICIENT = "sufficient"
    DEGRADED = "degraded"
    INSUFFICIENT = "insufficient"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class EvidenceQualityAssessment:
    """Why evidence does not yet satisfy one requirement, and what to seek."""

    gap_id: UUID
    claim_id: UUID | None
    quality_status: EvidenceQualityStatus
    failure_reasons: tuple[EvidenceFailureReason, ...]
    missing_evidence_type: MissingEvidenceType
    recommended_need_type: ResearchNeedType
    query_hints: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "gap_id": str(self.gap_id),
            "claim_id": str(self.claim_id) if self.claim_id is not None else None,
            "quality_status": self.quality_status.value,
            "failure_reasons": [reason.value for reason in self.failure_reasons],
            "missing_evidence_type": self.missing_evidence_type.value,
            "recommended_need_type": self.recommended_need_type.value,
            "query_hints": list(self.query_hints),
        }


class EvidenceQualityAnalyzer:
    """Assess evidence quality for one requirement from generic state only."""

    def assess(
        self,
        requirement: GapRequirement,
        facts: Sequence[EvidenceItemFacts] = (),
        *,
        claim_id: UUID | None = None,
        classification: EvidenceFailureClassification | None = None,
    ) -> EvidenceQualityAssessment:
        items = tuple(facts)
        verdict = classification or classify_evidence_failure(
            requirement, items, claim_id=claim_id
        )
        refinement = refine_research_need(verdict.primary_reason)
        return EvidenceQualityAssessment(
            gap_id=requirement.gap_id,
            claim_id=claim_id,
            quality_status=self._status(requirement, items, verdict),
            failure_reasons=verdict.reasons,
            missing_evidence_type=missing_type_for_reason(verdict.primary_reason),
            recommended_need_type=refinement.persisted_need_type,
            query_hints=refinement.query_hints,
        )

    def _status(
        self,
        requirement: GapRequirement,
        items: tuple[EvidenceItemFacts, ...],
        verdict: EvidenceFailureClassification,
    ) -> EvidenceQualityStatus:
        if requirement.closure_status is GapClosureStatus.CLOSED:
            return EvidenceQualityStatus.SUFFICIENT
        if not items and requirement.current_evidence_count == 0:
            return EvidenceQualityStatus.INSUFFICIENT
        usable = any(
            item.alignment_status
            in (
                EvidenceAlignmentStatus.ALIGNED,
                EvidenceAlignmentStatus.PARTIAL,
            )
            for item in items
        )
        if items and not usable:
            return EvidenceQualityStatus.INSUFFICIENT
        if verdict.reasons == (EvidenceFailureReason.UNKNOWN,):
            return EvidenceQualityStatus.UNKNOWN
        return EvidenceQualityStatus.DEGRADED


def facts_from_alignment(alignment: EvidenceAlignment) -> EvidenceItemFacts:
    """Project one persisted alignment onto generic per-item facts.

    Per-item quality metadata (reliability, quantitative/method flags) is
    *not* observed here, so classification rules stay tri-state safe.
    """

    return EvidenceItemFacts(
        evidence_id=alignment.evidence_id,
        alignment_status=alignment.alignment_status,
    )


def build_feedback_analysis(
    requirement: GapRequirement,
    alignments: Iterable[EvidenceAlignment] = (),
    *,
    claim_id: UUID | None = None,
) -> dict[str, object]:
    """Compose quality + claim analysis into flat JSON-safe feedback metadata.

    The payload only *adds* explanation fields; the persisted failure reason
    and the event chain (``closure.feedback.generated`` and successors) keep
    their existing shape.
    """

    items = tuple(
        facts_from_alignment(alignment)
        for alignment in alignments
        if alignment.gap_id == requirement.gap_id
    )
    assessment = EvidenceQualityAnalyzer().assess(
        requirement, items, claim_id=claim_id
    )
    verification = ClaimVerificationAnalyzer().analyze(
        requirement, items, claim_id=claim_id
    )
    return feedback_analysis_metadata(assessment, verification)


def feedback_analysis_metadata(
    assessment: EvidenceQualityAssessment,
    verification: ClaimVerificationResult,
) -> dict[str, object]:
    """Flatten one assessment + verification pair into metadata fields."""

    refinement = refine_research_need(
        assessment.failure_reasons[0]
        if assessment.failure_reasons
        else EvidenceFailureReason.UNKNOWN
    )
    return {
        "quality_status": assessment.quality_status.value,
        "failure_reasons": [reason.value for reason in assessment.failure_reasons],
        "primary_failure_reason": (
            assessment.failure_reasons[0].value
            if assessment.failure_reasons
            else EvidenceFailureReason.UNKNOWN.value
        ),
        "missing_evidence_type": assessment.missing_evidence_type.value,
        "support_level": verification.support_level.value,
        "missing_support_type": verification.missing_support_type.value,
        "recommended_need_type": assessment.recommended_need_type.value,
        "refined_need_type": refinement.refined_need_type.value,
        "query_hints": list(refinement.query_hints),
    }


__all__ = [
    "EvidenceQualityAnalyzer",
    "EvidenceQualityAssessment",
    "EvidenceQualityStatus",
    "build_feedback_analysis",
    "facts_from_alignment",
    "feedback_analysis_metadata",
]
