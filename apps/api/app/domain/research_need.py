"""Explain what unresolved GapRequirement facts still need research."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, UUID, uuid5

from app.domain.evidence_alignment import (
    EvidenceAlignment,
    EvidenceAlignmentReason,
    EvidenceAlignmentStatus,
)
from app.domain.gap_closure import (
    ClosureEvaluation,
    GapClosureStatus,
    GapRequirement,
    GapRequirementType,
    VerificationStatus,
)
from app.domain.research_context import ResearchContext

if TYPE_CHECKING:
    from app.domain.closure_feedback import ClosureFeedback


class ResearchNeedType(StrEnum):
    ADDITIONAL_SOURCE = "additional_source"
    INDEPENDENT_SOURCE = "independent_source"
    CLAIM_VERIFICATION = "claim_verification"
    EVIDENCE_QUALITY = "evidence_quality"
    MISSING_DATA = "missing_data"
    DIMENSION_COVERAGE = "dimension_coverage"
    UNKNOWN = "unknown"


class ResearchNeedStatus(StrEnum):
    OPEN = "open"
    SATISFIED = "satisfied"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ResearchNeed:
    """An explanatory bridge from one Gap to a future research action."""

    need_id: UUID
    gap_id: UUID
    question_id: str
    dimension_key: str
    requirement_type: GapRequirementType
    need_type: ResearchNeedType
    description: str
    priority: int
    status: ResearchNeedStatus
    created_at: datetime
    feedback_id: UUID | None = None
    #: Phase 14.1 evidence-aware fields; ``None``/empty means the need carries
    #: no ClosureFeedback analysis metadata and behaves exactly like 13.x.
    evidence_failure_reason: str | None = None
    missing_evidence_type: str | None = None
    refined_need_type: str | None = None
    query_hints: tuple[str, ...] = ()
    #: Phase 15.1 Branch B: the owner/domain identities already counted for this
    #: Requirement plus the numeric independence bar, so the follow-up query and
    #: post-search eligibility target a genuinely *new* independent source.
    existing_source_owners: tuple[str, ...] = ()
    existing_domains: tuple[str, ...] = ()
    required_source_count: int | None = None
    missing_source_count: int | None = None
    target_claim_id: str | None = None

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("research need description must not be empty")
        if self.priority < 1:
            raise ValueError("research need priority must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "need_id": str(self.need_id),
            "gap_id": str(self.gap_id),
            "question_id": self.question_id,
            "dimension_key": self.dimension_key,
            "requirement_type": self.requirement_type.value,
            "need_type": self.need_type.value,
            "description": self.description,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "feedback_id": str(self.feedback_id) if self.feedback_id else None,
            "evidence_failure_reason": self.evidence_failure_reason,
            "missing_evidence_type": self.missing_evidence_type,
            "refined_need_type": self.refined_need_type,
            "query_hints": list(self.query_hints),
            "existing_source_owners": list(self.existing_source_owners),
            "existing_domains": list(self.existing_domains),
            "required_source_count": self.required_source_count,
            "missing_source_count": self.missing_source_count,
            "target_claim_id": self.target_claim_id,
        }

    @property
    def research_context(self) -> ResearchContext:
        """The evidence-aware context this need propagates into the query pipeline."""

        return ResearchContext(
            evidence_failure_reason=self.evidence_failure_reason,
            missing_evidence_type=self.missing_evidence_type,
            refined_need_type=self.refined_need_type,
            query_hints=self.query_hints,
            existing_source_owners=self.existing_source_owners,
            existing_domains=self.existing_domains,
            required_source_count=self.required_source_count,
            missing_source_count=self.missing_source_count,
            target_claim_id=self.target_claim_id,
        )


class ResearchNeedGenerator:
    """Generate explanatory needs without scheduling or executing work."""

    @staticmethod
    def generate(
        requirement: GapRequirement,
        *,
        alignments: Iterable[EvidenceAlignment] = (),
        closure_evaluation: ClosureEvaluation | None = None,
        feedback: ClosureFeedback | None = None,
        question_priority: int | None = None,
        now: datetime | None = None,
    ) -> tuple[ResearchNeed, ...]:
        if requirement.closure_status is GapClosureStatus.CLOSED:
            return ()
        if (
            closure_evaluation is not None
            and closure_evaluation.closure_status is GapClosureStatus.CLOSED
        ):
            return ()

        relevant = [item for item in alignments if item.gap_id == requirement.gap_id]
        need_type, description = _classify_need(requirement, relevant, feedback=feedback)
        context = _research_context_from(feedback)
        timestamp = now or datetime.now(UTC)
        priority = max(1, question_priority or 3)
        need = ResearchNeed(
            need_id=uuid5(
                NAMESPACE_URL,
                f"research-need:{requirement.gap_id}:{need_type.value}",
            ),
            gap_id=requirement.gap_id,
            question_id=requirement.question_id,
            dimension_key=requirement.dimension_key,
            requirement_type=requirement.requirement_type,
            need_type=need_type,
            description=description,
            priority=priority,
            status=ResearchNeedStatus.OPEN,
            created_at=timestamp,
            feedback_id=feedback.feedback_id if feedback is not None else None,
            evidence_failure_reason=context.evidence_failure_reason,
            missing_evidence_type=context.missing_evidence_type,
            refined_need_type=context.refined_need_type,
            query_hints=context.query_hints,
            existing_source_owners=context.existing_source_owners,
            existing_domains=context.existing_domains,
            required_source_count=context.required_source_count,
            missing_source_count=context.missing_source_count,
            target_claim_id=context.target_claim_id,
        )
        return (need,)


def _classify_need(
    requirement: GapRequirement,
    alignments: list[EvidenceAlignment],
    *,
    feedback: ClosureFeedback | None = None,
) -> tuple[ResearchNeedType, str]:
    if feedback is not None:
        return (
            feedback.recommended_need_type,
            _feedback_need_description(feedback),
        )
    reasons = {
        item.rejection_reason
        for item in alignments
        if item.alignment_status
        in {EvidenceAlignmentStatus.PARTIAL, EvidenceAlignmentStatus.NOT_ALIGNED}
    }
    if EvidenceAlignmentReason.MISSING_INDEPENDENT_SOURCE in reasons:
        return (
            ResearchNeedType.INDEPENDENT_SOURCE,
            "需要额外独立来源验证该 Requirement。",
        )
    if EvidenceAlignmentReason.CLAIM_NOT_VERIFIED in reasons:
        return (
            ResearchNeedType.CLAIM_VERIFICATION,
            "需要进一步验证该 Claim 与现有证据之间的关系。",
        )
    if EvidenceAlignmentReason.INSUFFICIENT_QUALITY in reasons:
        return (
            ResearchNeedType.EVIDENCE_QUALITY,
            "需要质量更高、可定位且可验证的证据。",
        )
    if EvidenceAlignmentReason.DIMENSION_MISMATCH in reasons:
        return (
            ResearchNeedType.DIMENSION_COVERAGE,
            "需要针对该 Evidence Dimension 的研究材料。",
        )
    if requirement.requirement_type is GapRequirementType.INDEPENDENT_SOURCE:
        return (
            ResearchNeedType.INDEPENDENT_SOURCE,
            "需要额外独立来源满足来源多样性要求。",
        )
    if requirement.requirement_type is GapRequirementType.CLAIM_VERIFICATION:
        return (
            ResearchNeedType.CLAIM_VERIFICATION,
            "需要完成该 Claim 的验证。",
        )
    if requirement.requirement_type is GapRequirementType.DIMENSION_COVERAGE:
        return (
            ResearchNeedType.DIMENSION_COVERAGE,
            "需要补足该研究维度的 Coverage。",
        )
    if requirement.requirement_type is GapRequirementType.EVIDENCE_QUALITY:
        return (
            ResearchNeedType.EVIDENCE_QUALITY,
            "需要补充满足质量要求的证据。",
        )
    if requirement.verification_status is not VerificationStatus.VERIFIED:
        return (
            ResearchNeedType.MISSING_DATA,
            "需要补充当前 Requirement 缺失的研究数据。",
        )
    return ResearchNeedType.UNKNOWN, "需要进一步研究以满足当前 Requirement。"


def _research_context_from(feedback: ClosureFeedback | None) -> ResearchContext:
    """Read the Phase 14.0 analysis payload, if any, off one feedback.

    Phase 13.x feedback carries no metadata, so this yields an empty context
    and the generated need stays byte-for-byte identical to the old behavior.
    """

    metadata: Mapping[str, object] = {} if feedback is None else feedback.metadata
    return ResearchContext.from_mapping(metadata)


def _feedback_need_description(feedback: ClosureFeedback) -> str:
    return f"{feedback.missing_requirement} 未满足: {feedback.failure_reason.value}。"
