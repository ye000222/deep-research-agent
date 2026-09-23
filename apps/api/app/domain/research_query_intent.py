"""Describe future query directions without creating or executing queries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import NAMESPACE_URL, UUID, uuid5

from app.domain.gap_closure import GapClosureStatus, GapRequirement
from app.domain.research_action import (
    SuggestedResearchAction,
    SuggestedResearchActionStatus,
    SuggestedResearchActionType,
)
from app.domain.research_context import ResearchContext, meaningful_context


class ResearchQueryIntentType(StrEnum):
    FIND_ADDITIONAL_SOURCE = "find_additional_source"
    FIND_INDEPENDENT_SOURCE = "find_independent_source"
    VERIFY_CLAIM = "verify_claim"
    COLLECT_MISSING_DATA = "collect_missing_data"
    IMPROVE_EVIDENCE_QUALITY = "improve_evidence_quality"
    COVER_DIMENSION = "cover_dimension"
    UNKNOWN = "unknown"


class ResearchQueryIntentStatus(StrEnum):
    SUGGESTED = "suggested"
    CONSUMED = "consumed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ResearchQueryIntent:
    """A non-executable description of a future research query direction."""

    intent_id: UUID
    run_id: UUID
    question_id: str
    research_need_id: UUID
    suggested_action_id: UUID
    gap_id: UUID
    dimension_key: str
    requirement_type: str
    intent_type: ResearchQueryIntentType
    objective: str
    target_information: tuple[str, ...]
    preferred_source_types: tuple[str, ...]
    query_constraints: tuple[str, ...]
    status: ResearchQueryIntentStatus
    created_at: datetime
    #: Phase 14.1 additive evidence-aware context; ``None`` keeps the intent
    #: identical to the pre-14.1 shape.
    research_context: ResearchContext | None = None

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("query intent objective must not be empty")
        if not self.target_information:
            raise ValueError("query intent must define target information")

    def as_dict(self) -> dict[str, object]:
        return {
            "intent_id": str(self.intent_id),
            "run_id": str(self.run_id),
            "question_id": self.question_id,
            "research_need_id": str(self.research_need_id),
            "suggested_action_id": str(self.suggested_action_id),
            "gap_id": str(self.gap_id),
            "dimension_key": self.dimension_key,
            "requirement_type": self.requirement_type,
            "intent_type": self.intent_type.value,
            "objective": self.objective,
            "target_information": list(self.target_information),
            "preferred_source_types": list(self.preferred_source_types),
            "query_constraints": list(self.query_constraints),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "research_context": (
                self.research_context.as_dict()
                if self.research_context is not None
                else None
            ),
        }


class ResearchQueryIntentGenerator:
    """Map a suggested action to one explanatory query intent."""

    @staticmethod
    def generate(
        action: SuggestedResearchAction,
        *,
        requirement: GapRequirement,
        research_context: ResearchContext | None = None,
        now: datetime | None = None,
    ) -> tuple[ResearchQueryIntent, ...]:
        if action.status is not SuggestedResearchActionStatus.SUGGESTED:
            return ()
        if requirement.gap_id != action.gap_id:
            raise ValueError("SuggestedResearchAction and GapRequirement do not match")
        if requirement.question_id != action.question_id:
            raise ValueError("action and requirement question do not match")
        if requirement.dimension_key != action.dimension_key:
            raise ValueError("action and requirement dimension do not match")
        if requirement.closure_status is GapClosureStatus.CLOSED:
            return ()

        intent_type = _intent_type_for_action(action.action_type)
        objective, targets, source_types, constraints = _intent_text(
            intent_type,
            dimension_key=action.dimension_key,
        )
        context = meaningful_context(research_context)
        timestamp = now or datetime.now(UTC)
        return (
            ResearchQueryIntent(
                intent_id=uuid5(
                    NAMESPACE_URL,
                    f"research-query-intent:{action.action_id}:{intent_type.value}",
                ),
                run_id=requirement.run_id,
                question_id=action.question_id,
                research_need_id=action.need_id,
                suggested_action_id=action.action_id,
                gap_id=action.gap_id,
                dimension_key=action.dimension_key,
                requirement_type=requirement.requirement_type.value,
                intent_type=intent_type,
                objective=objective,
                target_information=targets,
                preferred_source_types=source_types,
                query_constraints=constraints,
                status=ResearchQueryIntentStatus.SUGGESTED,
                created_at=timestamp,
                research_context=context,
            ),
        )


def _intent_type_for_action(
    action_type: SuggestedResearchActionType,
) -> ResearchQueryIntentType:
    return {
        SuggestedResearchActionType.FIND_ADDITIONAL_SOURCE: (
            ResearchQueryIntentType.FIND_ADDITIONAL_SOURCE
        ),
        SuggestedResearchActionType.FIND_INDEPENDENT_SOURCE: (
            ResearchQueryIntentType.FIND_INDEPENDENT_SOURCE
        ),
        SuggestedResearchActionType.VERIFY_CLAIM: ResearchQueryIntentType.VERIFY_CLAIM,
        SuggestedResearchActionType.IMPROVE_EVIDENCE_QUALITY: (
            ResearchQueryIntentType.IMPROVE_EVIDENCE_QUALITY
        ),
        SuggestedResearchActionType.COLLECT_MISSING_DATA: (
            ResearchQueryIntentType.COLLECT_MISSING_DATA
        ),
        SuggestedResearchActionType.EXPAND_DIMENSION_COVERAGE: (
            ResearchQueryIntentType.COVER_DIMENSION
        ),
        SuggestedResearchActionType.UNKNOWN: ResearchQueryIntentType.UNKNOWN,
    }[action_type]


def _intent_text(
    intent_type: ResearchQueryIntentType,
    *,
    dimension_key: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if intent_type is ResearchQueryIntentType.FIND_INDEPENDENT_SOURCE:
        return (
            "寻找满足独立来源要求的新证据",
            ("第三方数据", "行业报告", "官方统计", "独立研究"),
            ("government", "industry_report", "research_institution", "independent_analysis"),
            ("排除已用于该 Requirement 的来源", "保持独立来源约束"),
        )
    if intent_type is ResearchQueryIntentType.FIND_ADDITIONAL_SOURCE:
        return (
            "寻找额外来源补充当前 Requirement",
            ("行业数据", "研究报告", "官方资料"),
            ("government", "industry_report", "research_institution"),
            ("优先选择尚未使用的来源",),
        )
    if intent_type is ResearchQueryIntentType.VERIFY_CLAIM:
        return (
            "验证指定 Claim 的真实性和可靠性",
            ("官方声明", "技术文档", "实验数据", "第三方验证"),
            ("official", "technical_document", "research_institution", "independent_analysis"),
            (f"验证 {dimension_key} 对应 Claim",),
        )
    if intent_type is ResearchQueryIntentType.IMPROVE_EVIDENCE_QUALITY:
        return (
            "寻找质量更高、更权威的证据",
            ("primary source", "authoritative report", "official documentation"),
            ("primary_source", "authoritative_report", "official_documentation"),
            ("优先使用可定位且可验证的原始材料",),
        )
    if intent_type is ResearchQueryIntentType.COLLECT_MISSING_DATA:
        return (
            "收集当前 Requirement 缺失的研究数据",
            ("官方数据集", "行业统计", "研究机构数据"),
            ("government", "dataset", "research_institution"),
            (f"补足 {dimension_key} 所需数据",),
        )
    if intent_type is ResearchQueryIntentType.COVER_DIMENSION:
        return (
            "补足目标研究维度的 Coverage",
            ("维度相关的官方资料", "行业报告", "独立研究"),
            ("government", "industry_report", "research_institution"),
            (f"查询必须直接覆盖 {dimension_key}",),
        )
    return (
        "进一步研究目标 Requirement",
        ("与目标 Requirement 直接相关的信息",),
        ("unknown",),
        (f"围绕 {dimension_key} 形成可验证信息",),
    )
