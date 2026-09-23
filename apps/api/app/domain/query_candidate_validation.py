"""Validate query candidates without executing or ranking them."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import NAMESPACE_URL, UUID, uuid5

from app.domain.gap_closure import GapClosureStatus, GapRequirement, GapRequirementType
from app.domain.query_candidate import QueryCandidate, QueryCandidateStatus
from app.domain.research_query_plan import ResearchQueryPlan


class QueryCandidateValidationStatus(StrEnum):
    VALID = "valid"
    PARTIAL = "partial"
    INVALID = "invalid"
    UNKNOWN = "unknown"


class QueryCandidateAlignmentType(StrEnum):
    PLAN_ALIGNED = "plan_aligned"
    GAP_ALIGNED = "gap_aligned"
    SOURCE_MISMATCH = "source_mismatch"
    TOPIC_MISMATCH = "topic_mismatch"
    INSUFFICIENT_QUERY_INTENT = "insufficient_query_intent"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class QueryCandidateValidation:
    """An observation explaining whether a candidate fits a Gap and Plan."""

    validation_id: UUID
    query_candidate_id: UUID
    query_plan_id: UUID
    gap_id: UUID
    question_id: str
    validation_status: QueryCandidateValidationStatus
    alignment_type: QueryCandidateAlignmentType
    issues: tuple[str, ...]
    explanation: str

    def __post_init__(self) -> None:
        if not self.explanation.strip():
            raise ValueError("query candidate validation explanation must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "validation_id": str(self.validation_id),
            "query_candidate_id": str(self.query_candidate_id),
            "query_plan_id": str(self.query_plan_id),
            "gap_id": str(self.gap_id),
            "question_id": self.question_id,
            "validation_status": self.validation_status.value,
            "alignment_type": self.alignment_type.value,
            "issues": list(self.issues),
            "explanation": self.explanation,
        }


class QueryCandidateValidator:
    """Validate a candidate against its plan and canonical GapRequirement."""

    @staticmethod
    def validate(
        candidate: QueryCandidate,
        *,
        plan: ResearchQueryPlan,
        requirement: GapRequirement,
    ) -> tuple[QueryCandidateValidation, ...]:
        if requirement.closure_status is GapClosureStatus.CLOSED:
            return ()
        if candidate.status is not QueryCandidateStatus.GENERATED:
            return ()
        if candidate.research_query_plan_id != plan.query_plan_id:
            raise ValueError("query candidate and query plan do not match")
        if candidate.gap_id != plan.gap_id or plan.gap_id != requirement.gap_id:
            raise ValueError("query candidate, plan, and requirement gaps do not match")
        if candidate.question_id != plan.question_id:
            raise ValueError("query candidate and query plan questions do not match")
        if plan.question_id != requirement.question_id:
            raise ValueError("query plan and requirement questions do not match")
        if candidate.dimension_key != plan.dimension_key:
            raise ValueError("query candidate and query plan dimensions do not match")
        if plan.dimension_key != requirement.dimension_key:
            raise ValueError("query plan and requirement dimensions do not match")

        status, alignment, issues, explanation = _evaluate_candidate(
            requirement.requirement_type,
            candidate,
        )
        return (
            QueryCandidateValidation(
                validation_id=uuid5(
                    NAMESPACE_URL,
                    f"query-candidate-validation:{candidate.query_candidate_id}",
                ),
                query_candidate_id=candidate.query_candidate_id,
                query_plan_id=plan.query_plan_id,
                gap_id=requirement.gap_id,
                question_id=requirement.question_id,
                validation_status=status,
                alignment_type=alignment,
                issues=issues,
                explanation=explanation,
            ),
        )


def _evaluate_candidate(
    requirement_type: GapRequirementType,
    candidate: QueryCandidate,
) -> tuple[
    QueryCandidateValidationStatus,
    QueryCandidateAlignmentType,
    tuple[str, ...],
    str,
]:
    text = candidate.query_text.casefold()
    source_types = set(candidate.preferred_source_types)

    if requirement_type is GapRequirementType.INDEPENDENT_SOURCE:
        if not source_types.intersection(
            {"government", "research_institution", "industry_report", "independent_analysis"}
        ):
            return (
                QueryCandidateValidationStatus.INVALID,
                QueryCandidateAlignmentType.SOURCE_MISMATCH,
                ("candidate_sources_do_not_support_independent_source",),
                "候选查询的来源约束不支持独立来源要求。",
            )
        if not any(
            term in text
            for term in ("industry", "report", "statistics", "third party", "analysis")
        ):
            return (
                QueryCandidateValidationStatus.INVALID,
                QueryCandidateAlignmentType.TOPIC_MISMATCH,
                ("candidate_topics_do_not_target_independent_source",),
                "候选查询主题没有指向第三方或独立来源。",
            )
        return (
            QueryCandidateValidationStatus.VALID,
            QueryCandidateAlignmentType.GAP_ALIGNED,
            (),
            "候选查询同时满足独立来源的主题和来源约束。",
        )

    if requirement_type is GapRequirementType.CLAIM_VERIFICATION:
        if not source_types.intersection(
            {"official", "technical_document", "research_institution"}
        ):
            return (
                QueryCandidateValidationStatus.INVALID,
                QueryCandidateAlignmentType.SOURCE_MISMATCH,
                ("candidate_sources_do_not_support_claim_validation",),
                "候选查询的来源约束不支持 Claim 验证。",
            )
        if not any(
            term in text
            for term in ("claim", "verification", "official", "documentation", "technical")
        ):
            return (
                QueryCandidateValidationStatus.INVALID,
                QueryCandidateAlignmentType.TOPIC_MISMATCH,
                ("candidate_topics_do_not_target_claim",),
                "候选查询主题无法直接验证目标 Claim。",
            )
        return (
            QueryCandidateValidationStatus.VALID,
            QueryCandidateAlignmentType.GAP_ALIGNED,
            (),
            "候选查询指向官方或技术材料,可用于验证目标 Claim。",
        )

    expected_topics = {
        GapRequirementType.EVIDENCE_QUALITY: ("primary", "official", "authoritative"),
        GapRequirementType.DIMENSION_COVERAGE: (candidate.dimension_key.casefold(),),
    }.get(requirement_type)
    if expected_topics is not None and any(term in text for term in expected_topics):
        return (
            QueryCandidateValidationStatus.VALID,
            QueryCandidateAlignmentType.PLAN_ALIGNED,
            (),
            "候选查询与 Query Plan 的目标方向一致。",
        )
    if requirement_type.value == "missing_data" and any(
        term in text for term in ("data", "dataset", "statistics")
    ):
        return (
            QueryCandidateValidationStatus.VALID,
            QueryCandidateAlignmentType.PLAN_ALIGNED,
            (),
            "候选查询指向 Requirement 所需的缺失数据。",
        )
    return (
        QueryCandidateValidationStatus.UNKNOWN,
        QueryCandidateAlignmentType.INSUFFICIENT_QUERY_INTENT,
        ("candidate_intent_is_not_supported_by_validation_rule",),
        "当前 Requirement 类型没有足够的候选查询验证规则。",
    )
