"""Explain candidate research actions without executing them."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, UUID, uuid5

from app.domain.evidence_alignment import EvidenceAlignment
from app.domain.gap_closure import GapClosureStatus, GapRequirement
from app.domain.research_need import ResearchNeed, ResearchNeedStatus, ResearchNeedType

if TYPE_CHECKING:
    from app.domain.closure_feedback import ClosureFeedback


class SuggestedResearchActionType(StrEnum):
    FIND_ADDITIONAL_SOURCE = "find_additional_source"
    FIND_INDEPENDENT_SOURCE = "find_independent_source"
    VERIFY_CLAIM = "verify_claim"
    IMPROVE_EVIDENCE_QUALITY = "improve_evidence_quality"
    COLLECT_MISSING_DATA = "collect_missing_data"
    EXPAND_DIMENSION_COVERAGE = "expand_dimension_coverage"
    UNKNOWN = "unknown"


class SuggestedResearchActionStatus(StrEnum):
    SUGGESTED = "suggested"
    CONSUMED = "consumed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class SuggestedResearchAction:
    """A candidate action that is not yet authorized for execution."""

    action_id: UUID
    need_id: UUID
    gap_id: UUID
    question_id: str
    dimension_key: str
    action_type: SuggestedResearchActionType
    objective: str
    description: str
    priority: int
    status: SuggestedResearchActionStatus
    created_at: datetime
    source_feedback_id: UUID | None = None

    def __post_init__(self) -> None:
        if not self.objective.strip() or not self.description.strip():
            raise ValueError("research action text must not be empty")
        if self.priority < 1:
            raise ValueError("research action priority must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "action_id": str(self.action_id),
            "need_id": str(self.need_id),
            "gap_id": str(self.gap_id),
            "question_id": self.question_id,
            "dimension_key": self.dimension_key,
            "action_type": self.action_type.value,
            "objective": self.objective,
            "description": self.description,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "source_feedback_id": (
                str(self.source_feedback_id) if self.source_feedback_id else None
            ),
        }


class ResearchActionGenerator:
    """Map an open ResearchNeed to one explanatory candidate action."""

    @staticmethod
    def generate(
        need: ResearchNeed,
        *,
        requirement: GapRequirement,
        alignments: Iterable[EvidenceAlignment] = (),
        feedback: ClosureFeedback | None = None,
        now: datetime | None = None,
    ) -> tuple[SuggestedResearchAction, ...]:
        del alignments  # Alignment explains the Need; it does not alter mapping.
        if need.status is not ResearchNeedStatus.OPEN:
            return ()
        if requirement.gap_id != need.gap_id:
            raise ValueError("ResearchNeed and GapRequirement do not match")
        if requirement.closure_status is GapClosureStatus.CLOSED:
            return ()
        action_type = _action_type_for_need(need.need_type)
        objective, description = _action_text(
            action_type,
            question_id=need.question_id,
            dimension_key=need.dimension_key,
        )
        timestamp = now or datetime.now(UTC)
        return (
            SuggestedResearchAction(
                action_id=uuid5(
                    NAMESPACE_URL,
                    f"research-action:{need.need_id}:{action_type.value}",
                ),
                need_id=need.need_id,
                gap_id=need.gap_id,
                question_id=need.question_id,
                dimension_key=need.dimension_key,
                action_type=action_type,
                objective=objective,
                description=description,
                priority=need.priority,
                status=SuggestedResearchActionStatus.SUGGESTED,
                created_at=timestamp,
                source_feedback_id=(
                    feedback.feedback_id
                    if feedback is not None
                    else need.feedback_id
                ),
            ),
        )


def _action_type_for_need(need_type: ResearchNeedType) -> SuggestedResearchActionType:
    return {
        ResearchNeedType.ADDITIONAL_SOURCE: SuggestedResearchActionType.FIND_ADDITIONAL_SOURCE,
        ResearchNeedType.INDEPENDENT_SOURCE: SuggestedResearchActionType.FIND_INDEPENDENT_SOURCE,
        ResearchNeedType.CLAIM_VERIFICATION: SuggestedResearchActionType.VERIFY_CLAIM,
        ResearchNeedType.EVIDENCE_QUALITY: SuggestedResearchActionType.IMPROVE_EVIDENCE_QUALITY,
        ResearchNeedType.MISSING_DATA: SuggestedResearchActionType.COLLECT_MISSING_DATA,
        ResearchNeedType.DIMENSION_COVERAGE: SuggestedResearchActionType.EXPAND_DIMENSION_COVERAGE,
        ResearchNeedType.UNKNOWN: SuggestedResearchActionType.UNKNOWN,
    }[need_type]


def _action_text(
    action_type: SuggestedResearchActionType,
    *,
    question_id: str,
    dimension_key: str,
) -> tuple[str, str]:
    if action_type is SuggestedResearchActionType.FIND_INDEPENDENT_SOURCE:
        return (
            "寻找额外独立来源满足 Requirement",
            f"寻找满足 {dimension_key} Requirement 的额外独立来源。",
        )
    if action_type is SuggestedResearchActionType.FIND_ADDITIONAL_SOURCE:
        return (
            "寻找额外来源补充 Requirement",
            f"寻找额外来源补充 {dimension_key} Requirement。",
        )
    if action_type is SuggestedResearchActionType.VERIFY_CLAIM:
        return (
            "验证目标 Claim",
            f"验证 {dimension_key} 对应 Claim 与现有证据的关系。",
        )
    if action_type is SuggestedResearchActionType.IMPROVE_EVIDENCE_QUALITY:
        return (
            "改善 Evidence Quality",
            f"寻找更高质量证据满足 {dimension_key} Requirement。",
        )
    if action_type is SuggestedResearchActionType.COLLECT_MISSING_DATA:
        return (
            "收集缺失研究数据",
            f"收集 {dimension_key} 缺失的研究数据。",
        )
    if action_type is SuggestedResearchActionType.EXPAND_DIMENSION_COVERAGE:
        return (
            "扩展 Dimension Coverage",
            f"补足 {dimension_key} 的研究 Coverage。",
        )
    return (
        "进一步研究目标 Requirement",
        f"为 {dimension_key} Requirement 生成后续研究候选。",
    )
