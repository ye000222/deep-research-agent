"""Describe future query planning without creating or executing searches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import NAMESPACE_URL, UUID, uuid5

from app.domain.gap_closure import GapClosureStatus, GapRequirement
from app.domain.research_context import ResearchContext, meaningful_context
from app.domain.research_query_intent import (
    ResearchQueryIntent,
    ResearchQueryIntentStatus,
    ResearchQueryIntentType,
)


class ResearchQueryPlanStrategy(StrEnum):
    ADDITIONAL_SOURCE_DISCOVERY = "additional_source_discovery"
    INDEPENDENT_SOURCE_DISCOVERY = "independent_source_discovery"
    CLAIM_VALIDATION = "claim_validation"
    DATA_COLLECTION = "data_collection"
    EVIDENCE_UPGRADE = "evidence_upgrade"
    DIMENSION_COMPLETION = "dimension_completion"
    UNKNOWN = "unknown"


class ResearchQueryPlanStatus(StrEnum):
    SUGGESTED = "suggested"
    CONSUMED = "consumed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ResearchQueryPlan:
    """A non-executable plan describing how a future search may be shaped."""

    query_plan_id: UUID
    run_id: UUID
    question_id: str
    gap_id: UUID
    dimension_key: str
    requirement_type: str
    research_query_intent_id: UUID
    objective: str
    search_strategy_type: ResearchQueryPlanStrategy
    target_topics: tuple[str, ...]
    preferred_source_types: tuple[str, ...]
    exclusion_constraints: tuple[str, ...]
    priority: int
    status: ResearchQueryPlanStatus
    created_at: datetime
    #: Phase 14.1 additive evidence-aware context inherited from the intent.
    research_context: ResearchContext | None = None

    def __post_init__(self) -> None:
        if not self.objective.strip():
            raise ValueError("query plan objective must not be empty")
        if not self.target_topics:
            raise ValueError("query plan must define target topics")
        if self.priority < 1:
            raise ValueError("query plan priority must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "query_plan_id": str(self.query_plan_id),
            "run_id": str(self.run_id),
            "question_id": self.question_id,
            "gap_id": str(self.gap_id),
            "dimension_key": self.dimension_key,
            "requirement_type": self.requirement_type,
            "research_query_intent_id": str(self.research_query_intent_id),
            "objective": self.objective,
            "search_strategy_type": self.search_strategy_type.value,
            "target_topics": list(self.target_topics),
            "preferred_source_types": list(self.preferred_source_types),
            "exclusion_constraints": list(self.exclusion_constraints),
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "research_context": (
                self.research_context.as_dict()
                if self.research_context is not None
                else None
            ),
        }


class ResearchQueryPlanGenerator:
    """Map a suggested query intent to one explanatory query plan."""

    @staticmethod
    def generate(
        intent: ResearchQueryIntent,
        *,
        requirement: GapRequirement,
        priority: int = 3,
        now: datetime | None = None,
    ) -> tuple[ResearchQueryPlan, ...]:
        if intent.status is not ResearchQueryIntentStatus.SUGGESTED:
            return ()
        if requirement.gap_id != intent.gap_id:
            raise ValueError("ResearchQueryIntent and GapRequirement do not match")
        if requirement.question_id != intent.question_id:
            raise ValueError("intent and requirement question do not match")
        if requirement.dimension_key != intent.dimension_key:
            raise ValueError("intent and requirement dimension do not match")
        if requirement.closure_status is GapClosureStatus.CLOSED:
            return ()

        strategy = _strategy_for_intent(intent.intent_type)
        objective, topics, source_types, constraints = _plan_text(
            strategy,
            dimension_key=intent.dimension_key,
        )
        timestamp = now or datetime.now(UTC)
        return (
            ResearchQueryPlan(
                query_plan_id=uuid5(
                    NAMESPACE_URL,
                    f"research-query-plan:{intent.intent_id}:{strategy.value}",
                ),
                run_id=intent.run_id,
                question_id=intent.question_id,
                gap_id=intent.gap_id,
                dimension_key=intent.dimension_key,
                requirement_type=intent.requirement_type,
                research_query_intent_id=intent.intent_id,
                objective=objective,
                search_strategy_type=strategy,
                target_topics=topics,
                preferred_source_types=source_types,
                exclusion_constraints=constraints,
                priority=priority,
                status=ResearchQueryPlanStatus.SUGGESTED,
                created_at=timestamp,
                research_context=meaningful_context(intent.research_context),
            ),
        )


def _strategy_for_intent(
    intent_type: ResearchQueryIntentType,
) -> ResearchQueryPlanStrategy:
    return {
        ResearchQueryIntentType.FIND_ADDITIONAL_SOURCE: (
            ResearchQueryPlanStrategy.ADDITIONAL_SOURCE_DISCOVERY
        ),
        ResearchQueryIntentType.FIND_INDEPENDENT_SOURCE: (
            ResearchQueryPlanStrategy.INDEPENDENT_SOURCE_DISCOVERY
        ),
        ResearchQueryIntentType.VERIFY_CLAIM: ResearchQueryPlanStrategy.CLAIM_VALIDATION,
        ResearchQueryIntentType.COLLECT_MISSING_DATA: (
            ResearchQueryPlanStrategy.DATA_COLLECTION
        ),
        ResearchQueryIntentType.IMPROVE_EVIDENCE_QUALITY: (
            ResearchQueryPlanStrategy.EVIDENCE_UPGRADE
        ),
        ResearchQueryIntentType.COVER_DIMENSION: (
            ResearchQueryPlanStrategy.DIMENSION_COMPLETION
        ),
        ResearchQueryIntentType.UNKNOWN: ResearchQueryPlanStrategy.UNKNOWN,
    }[intent_type]


def _plan_text(
    strategy: ResearchQueryPlanStrategy,
    *,
    dimension_key: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if strategy is ResearchQueryPlanStrategy.INDEPENDENT_SOURCE_DISCOVERY:
        return (
            "寻找满足独立来源要求的新证据",
            ("industry report", "official statistics", "third party analysis"),
            ("government", "research_institution", "industry_report"),
            ("exclude_existing_sources=true", "require_independent_source=true"),
        )
    if strategy is ResearchQueryPlanStrategy.ADDITIONAL_SOURCE_DISCOVERY:
        return (
            "寻找额外来源补充当前 Requirement",
            ("industry data", "research report", "official material"),
            ("government", "research_institution", "industry_report"),
            ("exclude_existing_sources=true",),
        )
    if strategy is ResearchQueryPlanStrategy.CLAIM_VALIDATION:
        return (
            "寻找能够验证或反驳 Claim 的信息",
            ("official documentation", "technical report", "experimental evidence"),
            ("official", "technical_document", "research_institution"),
            (f"validate_claim_dimension={dimension_key}",),
        )
    if strategy is ResearchQueryPlanStrategy.EVIDENCE_UPGRADE:
        return (
            "寻找更高可信度证据替代当前证据",
            ("primary source", "official document", "authoritative report"),
            ("primary_source", "official_documentation", "authoritative_report"),
            ("prefer_primary_or_authoritative_sources=true",),
        )
    if strategy is ResearchQueryPlanStrategy.DATA_COLLECTION:
        return (
            "收集当前 Requirement 缺失的研究数据",
            ("official dataset", "industry statistics", "research institution data"),
            ("government", "dataset", "research_institution"),
            (f"collect_data_for_dimension={dimension_key}",),
        )
    if strategy is ResearchQueryPlanStrategy.DIMENSION_COMPLETION:
        return (
            "补足目标研究维度的 Coverage",
            ("dimension-specific official material", "industry report", "independent research"),
            ("government", "industry_report", "research_institution"),
            (f"must_cover_dimension={dimension_key}",),
        )
    return (
        "进一步研究目标 Requirement",
        ("information directly related to the target Requirement",),
        ("unknown",),
        (f"form_verifiable_information_for={dimension_key}",),
    )
