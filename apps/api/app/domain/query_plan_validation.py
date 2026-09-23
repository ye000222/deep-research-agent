"""Validate query-plan alignment with a canonical GapRequirement."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import NAMESPACE_URL, UUID, uuid5

from app.domain.gap_closure import GapClosureStatus, GapRequirement, GapRequirementType
from app.domain.research_query_plan import (
    ResearchQueryPlan,
    ResearchQueryPlanStrategy,
)


class QueryPlanValidationStatus(StrEnum):
    VALID = "valid"
    PARTIAL = "partial"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class QueryPlanAlignmentType(StrEnum):
    DIRECT_ALIGNMENT = "direct_alignment"
    PARTIAL_ALIGNMENT = "partial_alignment"
    MISSING_REQUIREMENT_MATCH = "missing_requirement_match"
    INSUFFICIENT_SOURCE_STRATEGY = "insufficient_source_strategy"
    CLAIM_MISMATCH = "claim_mismatch"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class QueryPlanValidation:
    """An observation explaining whether a QueryPlan fits a GapRequirement."""

    validation_id: UUID
    query_plan_id: UUID
    gap_id: UUID
    question_id: str
    validation_status: QueryPlanValidationStatus
    alignment_type: QueryPlanAlignmentType
    issues: tuple[str, ...]
    explanation: str

    def __post_init__(self) -> None:
        if not self.explanation.strip():
            raise ValueError("query plan validation explanation must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "validation_id": str(self.validation_id),
            "query_plan_id": str(self.query_plan_id),
            "gap_id": str(self.gap_id),
            "question_id": self.question_id,
            "validation_status": self.validation_status.value,
            "alignment_type": self.alignment_type.value,
            "issues": list(self.issues),
            "explanation": self.explanation,
        }


class QueryPlanValidator:
    """Check plan alignment without changing either input model."""

    @staticmethod
    def validate(
        plan: ResearchQueryPlan,
        *,
        requirement: GapRequirement,
    ) -> tuple[QueryPlanValidation, ...]:
        if requirement.closure_status is GapClosureStatus.CLOSED:
            return ()
        if plan.gap_id != requirement.gap_id:
            raise ValueError("ResearchQueryPlan and GapRequirement do not match")
        if plan.question_id != requirement.question_id:
            raise ValueError("query plan and requirement question do not match")
        if plan.dimension_key != requirement.dimension_key:
            raise ValueError("query plan and requirement dimension do not match")

        status, alignment, issues, explanation = _evaluate_alignment(
            requirement.requirement_type,
            plan.search_strategy_type,
        )
        return (
            QueryPlanValidation(
                validation_id=uuid5(
                    NAMESPACE_URL,
                    f"query-plan-validation:{plan.query_plan_id}:{requirement.state_version}",
                ),
                query_plan_id=plan.query_plan_id,
                gap_id=requirement.gap_id,
                question_id=requirement.question_id,
                validation_status=status,
                alignment_type=alignment,
                issues=issues,
                explanation=explanation,
            ),
        )


def _evaluate_alignment(
    requirement_type: GapRequirementType,
    strategy: ResearchQueryPlanStrategy,
) -> tuple[
    QueryPlanValidationStatus,
    QueryPlanAlignmentType,
    tuple[str, ...],
    str,
]:
    expected = {
        GapRequirementType.INDEPENDENT_SOURCE: (
            ResearchQueryPlanStrategy.INDEPENDENT_SOURCE_DISCOVERY,
            "独立来源 Requirement 与独立来源发现策略直接匹配。",
        ),
        GapRequirementType.CLAIM_VERIFICATION: (
            ResearchQueryPlanStrategy.CLAIM_VALIDATION,
            "Claim Verification Requirement 与 Claim Validation 策略直接匹配。",
        ),
        GapRequirementType.EVIDENCE_QUALITY: (
            ResearchQueryPlanStrategy.EVIDENCE_UPGRADE,
            "Evidence Quality Requirement 与 Evidence Upgrade 策略直接匹配。",
        ),
        GapRequirementType.DIMENSION_COVERAGE: (
            ResearchQueryPlanStrategy.DIMENSION_COMPLETION,
            "Dimension Coverage Requirement 与 Dimension Completion 策略直接匹配。",
        ),
    }.get(requirement_type)
    if expected is None:
        if (
            requirement_type.value == "missing_data"
            and strategy is ResearchQueryPlanStrategy.DATA_COLLECTION
        ):
            return (
                QueryPlanValidationStatus.VALID,
                QueryPlanAlignmentType.DIRECT_ALIGNMENT,
                (),
                "Missing Data Requirement 与 Data Collection 策略直接匹配。",
            )
        return (
            QueryPlanValidationStatus.UNKNOWN,
            QueryPlanAlignmentType.UNKNOWN,
            ("no_validation_rule_for_requirement",),
            "当前 Requirement 类型没有可用的 Query Plan 验证规则。",
        )

    expected_strategy, explanation = expected
    if strategy is expected_strategy:
        return (
            QueryPlanValidationStatus.VALID,
            QueryPlanAlignmentType.DIRECT_ALIGNMENT,
            (),
            explanation,
        )
    if requirement_type is GapRequirementType.CLAIM_VERIFICATION:
        return (
            QueryPlanValidationStatus.INVALID,
            QueryPlanAlignmentType.CLAIM_MISMATCH,
            ("query_strategy_cannot_directly_validate_claim",),
            "当前查询策略不能直接验证 Claim。",
        )
    if strategy is ResearchQueryPlanStrategy.UNKNOWN:
        return (
            QueryPlanValidationStatus.PARTIAL,
            QueryPlanAlignmentType.MISSING_REQUIREMENT_MATCH,
            ("query_strategy_is_unknown",),
            "查询规划未提供与当前 Requirement 明确匹配的策略。",
        )
    return (
        QueryPlanValidationStatus.PARTIAL,
        QueryPlanAlignmentType.INSUFFICIENT_SOURCE_STRATEGY,
        ("query_strategy_does_not_target_requirement",),
        "当前查询策略不能充分覆盖该 Requirement。",
    )
