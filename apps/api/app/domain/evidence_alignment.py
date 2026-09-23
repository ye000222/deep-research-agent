"""Explanatory alignment between evidence and canonical GapRequirements.

EvidenceAlignment is deliberately observation-only.  It never changes a
GapRequirement or a ClosureEvaluation; it explains why evidence did or did
not satisfy an existing requirement.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from app.domain.gap_closure import GapRequirement, GapRequirementType


class EvidenceAlignmentStatus(StrEnum):
    ALIGNED = "aligned"
    PARTIAL = "partial"
    NOT_ALIGNED = "not_aligned"
    UNKNOWN = "unknown"


class EvidenceAlignmentReason(StrEnum):
    MISSING_INDEPENDENT_SOURCE = "missing_independent_source"
    INSUFFICIENT_QUALITY = "insufficient_quality"
    CLAIM_NOT_VERIFIED = "claim_not_verified"
    DIMENSION_MISMATCH = "dimension_mismatch"
    REQUIREMENT_NOT_SATISFIED = "requirement_not_satisfied"


@dataclass(frozen=True, slots=True)
class EvidenceAlignment:
    """One immutable explanation linking an evidence item to a Gap."""

    alignment_id: UUID
    evidence_id: UUID
    gap_id: UUID
    question_id: str
    dimension_key: str
    requirement_type: GapRequirementType
    satisfies_requirement: bool
    alignment_status: EvidenceAlignmentStatus
    rejection_reason: EvidenceAlignmentReason | None
    created_at: datetime
    state_version: int
    extraction_id: UUID | None = None
    reader_execution_id: UUID | None = None
    alignment_request_id: UUID | None = None

    def __post_init__(self) -> None:
        if self.state_version < 1:
            raise ValueError("state_version must be positive")
        if (
            self.satisfies_requirement
            and self.alignment_status is not EvidenceAlignmentStatus.ALIGNED
        ):
            raise ValueError("satisfied evidence must be aligned")
        if (
            self.alignment_status is EvidenceAlignmentStatus.ALIGNED
            and self.rejection_reason is not None
        ):
            raise ValueError("aligned evidence cannot have a rejection reason")

    @classmethod
    def evaluate(
        cls,
        requirement: GapRequirement,
        *,
        evidence_id: UUID,
        evidence_dimension_key: str | None,
        evidence_quality_passed: bool,
        independent_source_count: int,
        claim_verified: bool,
        state_version: int | None = None,
        created_at: datetime | None = None,
    ) -> EvidenceAlignment:
        """Classify one evidence item without mutating the requirement."""

        reason: EvidenceAlignmentReason | None = None
        status = EvidenceAlignmentStatus.ALIGNED
        satisfies = True
        if evidence_dimension_key != requirement.dimension_key:
            status = EvidenceAlignmentStatus.NOT_ALIGNED
            reason = EvidenceAlignmentReason.DIMENSION_MISMATCH
            satisfies = False
        elif not evidence_quality_passed:
            status = EvidenceAlignmentStatus.NOT_ALIGNED
            reason = EvidenceAlignmentReason.INSUFFICIENT_QUALITY
            satisfies = False
        elif (
            requirement.requirement_type
            in (
                GapRequirementType.CLAIM_VERIFICATION,
                GapRequirementType.INDEPENDENT_SOURCE,
            )
            and independent_source_count < requirement.required_independent_sources
        ):
            # Claim verification and independent-source requirements share one
            # numeric rule: enough *distinct* source owners must back the
            # Requirement.  Independence is judged from the canonical source
            # identity (per-source-owner) count, never from a hard-coded
            # boolean or the coarse source role, so two different owners that
            # both expose ``role == webpage`` stay independent.  The rejection
            # reason keeps distinguishing the two requirement kinds so the
            # existing closure-feedback chain is unchanged.
            status = EvidenceAlignmentStatus.PARTIAL
            reason = (
                EvidenceAlignmentReason.CLAIM_NOT_VERIFIED
                if requirement.requirement_type is GapRequirementType.CLAIM_VERIFICATION
                else EvidenceAlignmentReason.MISSING_INDEPENDENT_SOURCE
            )
            satisfies = False
        elif (
            requirement.requirement_type is GapRequirementType.EVIDENCE_QUALITY
            and not claim_verified
        ):
            status = EvidenceAlignmentStatus.PARTIAL
            reason = EvidenceAlignmentReason.REQUIREMENT_NOT_SATISFIED
            satisfies = False
        return cls(
            alignment_id=uuid4(),
            evidence_id=evidence_id,
            gap_id=requirement.gap_id,
            question_id=requirement.question_id,
            dimension_key=requirement.dimension_key,
            requirement_type=requirement.requirement_type,
            satisfies_requirement=satisfies,
            alignment_status=status,
            rejection_reason=reason,
            created_at=created_at or datetime.now(UTC),
            state_version=state_version or requirement.state_version,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "alignment_id": str(self.alignment_id),
            "evidence_id": str(self.evidence_id),
            "gap_id": str(self.gap_id),
            "question_id": self.question_id,
            "dimension_key": self.dimension_key,
            "requirement_type": self.requirement_type.value,
            "satisfies_requirement": self.satisfies_requirement,
            "alignment_status": self.alignment_status.value,
            "rejection_reason": (
                self.rejection_reason.value if self.rejection_reason is not None else None
            ),
            "created_at": self.created_at.isoformat(),
            "state_version": self.state_version,
            "extraction_id": (
                str(self.extraction_id) if self.extraction_id is not None else None
            ),
            "reader_execution_id": (
                str(self.reader_execution_id)
                if self.reader_execution_id is not None
                else None
            ),
            "alignment_request_id": (
                str(self.alignment_request_id)
                if self.alignment_request_id is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class EvidenceAlignmentSummary:
    aligned_count: int
    partial_count: int
    not_aligned_count: int
    unknown_count: int
    remaining_requirement_reason: EvidenceAlignmentReason | None


def summarize_alignments(
    requirement: GapRequirement,
    alignments: Iterable[EvidenceAlignment],
) -> EvidenceAlignmentSummary:
    """Summarize explanatory evidence without changing Gap state."""

    relevant = [item for item in alignments if item.gap_id == requirement.gap_id]
    counts = {
        status: sum(item.alignment_status is status for item in relevant)
        for status in EvidenceAlignmentStatus
    }
    reason = next(
        (
            item.rejection_reason
            for item in relevant
            if item.rejection_reason is not None
        ),
        None,
    )
    if not relevant and requirement.closure_status.value != "closed":
        reason = EvidenceAlignmentReason.REQUIREMENT_NOT_SATISFIED
    return EvidenceAlignmentSummary(
        aligned_count=counts[EvidenceAlignmentStatus.ALIGNED],
        partial_count=counts[EvidenceAlignmentStatus.PARTIAL],
        not_aligned_count=counts[EvidenceAlignmentStatus.NOT_ALIGNED],
        unknown_count=counts[EvidenceAlignmentStatus.UNKNOWN],
        remaining_requirement_reason=reason,
    )
