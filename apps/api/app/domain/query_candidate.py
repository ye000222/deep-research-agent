"""Generate non-executable query expressions from validated query plans."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, UUID, uuid5

from app.domain.gap_closure import GapClosureStatus, GapRequirement
from app.domain.query_plan_validation import (
    QueryPlanValidation,
    QueryPlanValidationStatus,
)
from app.domain.research_context import (
    ResearchContext,
    apply_query_hints,
    meaningful_context,
)
from app.domain.research_query_plan import (
    ResearchQueryPlan,
    ResearchQueryPlanStatus,
    ResearchQueryPlanStrategy,
)

if TYPE_CHECKING:
    from app.domain.closure_feedback import ClosureFeedback


class QueryCandidateStatus(StrEnum):
    GENERATED = "generated"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class QueryCandidate:
    """A candidate query expression that has not been submitted for search."""

    query_candidate_id: UUID
    run_id: UUID
    question_id: str
    gap_id: UUID
    dimension_key: str
    requirement_type: str
    research_query_plan_id: UUID
    query_text: str
    query_purpose: str
    target_topics: tuple[str, ...]
    preferred_source_types: tuple[str, ...]
    constraints: tuple[str, ...]
    generation_reason: str
    status: QueryCandidateStatus
    feedback_id: UUID | None = None
    avoid_previous_failure_reason: str | None = None
    #: Phase 14.1 additive evidence-aware context inherited from the plan.
    research_context: ResearchContext | None = None

    def __post_init__(self) -> None:
        if not self.query_text.strip():
            raise ValueError("query candidate text must not be empty")
        if not self.query_purpose.strip():
            raise ValueError("query candidate purpose must not be empty")
        if not self.generation_reason.strip():
            raise ValueError("query candidate generation reason must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "query_candidate_id": str(self.query_candidate_id),
            "run_id": str(self.run_id),
            "question_id": self.question_id,
            "gap_id": str(self.gap_id),
            "dimension_key": self.dimension_key,
            "requirement_type": self.requirement_type,
            "research_query_plan_id": str(self.research_query_plan_id),
            "query_text": self.query_text,
            "query_purpose": self.query_purpose,
            "target_topics": list(self.target_topics),
            "preferred_source_types": list(self.preferred_source_types),
            "constraints": list(self.constraints),
            "generation_reason": self.generation_reason,
            "status": self.status.value,
            "feedback_id": str(self.feedback_id) if self.feedback_id else None,
            "avoid_previous_failure_reason": self.avoid_previous_failure_reason,
            "research_context": (
                self.research_context.as_dict()
                if self.research_context is not None
                else None
            ),
        }


class QueryCandidateGenerator:
    """Map a validated query plan to one explanatory query candidate."""

    @staticmethod
    def generate(
        plan: ResearchQueryPlan,
        *,
        requirement: GapRequirement,
        validation: QueryPlanValidation,
        feedback: ClosureFeedback | None = None,
    ) -> tuple[QueryCandidate, ...]:
        if plan.status is not ResearchQueryPlanStatus.SUGGESTED:
            return ()
        if requirement.closure_status is GapClosureStatus.CLOSED:
            return ()
        if validation.validation_status is not QueryPlanValidationStatus.VALID:
            return ()
        if validation.query_plan_id != plan.query_plan_id:
            raise ValueError("validation and query plan do not match")
        if validation.gap_id != plan.gap_id:
            raise ValueError("validation and query plan gap do not match")
        if requirement.gap_id != plan.gap_id:
            raise ValueError("query plan and requirement do not match")
        if requirement.question_id != plan.question_id:
            raise ValueError("query plan and requirement question do not match")
        if requirement.dimension_key != plan.dimension_key:
            raise ValueError("query plan and requirement dimension do not match")

        query_text, purpose = _candidate_text(
            plan.search_strategy_type,
            dimension_key=plan.dimension_key,
        )
        context = meaningful_context(plan.research_context)
        if context is not None:
            # Phase 14.1: only append the missing-evidence hints; the base
            # expression, strategy mapping, and ranking inputs stay untouched.
            query_text = apply_query_hints(query_text, context.query_hints)
        return (
            QueryCandidate(
                query_candidate_id=uuid5(
                    NAMESPACE_URL,
                    f"query-candidate:{plan.query_plan_id}:{query_text}",
                ),
                run_id=plan.run_id,
                question_id=plan.question_id,
                gap_id=plan.gap_id,
                dimension_key=plan.dimension_key,
                requirement_type=plan.requirement_type,
                research_query_plan_id=plan.query_plan_id,
                query_text=query_text,
                query_purpose=purpose,
                target_topics=plan.target_topics,
                preferred_source_types=plan.preferred_source_types,
                constraints=plan.exclusion_constraints,
                generation_reason=(
                    "generated_from_valid_query_plan:" + str(validation.validation_id)
                    + (
                        ";refined_from_feedback:" + str(feedback.feedback_id)
                        if feedback is not None
                        else ""
                    )
                ),
                status=QueryCandidateStatus.GENERATED,
                feedback_id=feedback.feedback_id if feedback is not None else None,
                avoid_previous_failure_reason=(
                    feedback.failure_reason.value if feedback is not None else None
                ),
                research_context=context,
            ),
        )


def _candidate_text(
    strategy: ResearchQueryPlanStrategy,
    *,
    dimension_key: str,
) -> tuple[str, str]:
    if strategy is ResearchQueryPlanStrategy.INDEPENDENT_SOURCE_DISCOVERY:
        return (
            f"{dimension_key} market report official statistics "
            "third party analysis",
            "寻找满足独立来源要求的行业数据",
        )
    if strategy is ResearchQueryPlanStrategy.ADDITIONAL_SOURCE_DISCOVERY:
        return (
            f"{dimension_key} industry data research report official material",
            "寻找额外来源补充当前 Requirement",
        )
    if strategy is ResearchQueryPlanStrategy.CLAIM_VALIDATION:
        return (
            f"{dimension_key} claim official documentation technical report "
            "verification",
            "寻找能够验证或反驳 Claim 的官方或第三方依据",
        )
    if strategy is ResearchQueryPlanStrategy.EVIDENCE_UPGRADE:
        return (
            f"{dimension_key} primary source official report authoritative evidence",
            "寻找更高可信度证据",
        )
    if strategy is ResearchQueryPlanStrategy.DATA_COLLECTION:
        return (
            f"{dimension_key} official dataset industry statistics research data",
            "收集当前 Requirement 缺失的研究数据",
        )
    if strategy is ResearchQueryPlanStrategy.DIMENSION_COMPLETION:
        return (
            f"{dimension_key} official material industry report independent research",
            "补足目标研究维度的 Coverage",
        )
    return (
        f"{dimension_key} research evidence Requirement",
        "生成与目标 Requirement 直接相关的查询候选",
    )
