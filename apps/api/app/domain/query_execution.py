"""Adapt an eligible query candidate into a future execution request."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, UUID, uuid5

from app.domain.gap_closure import GapClosureStatus, GapRequirement
from app.domain.query_candidate import QueryCandidate, QueryCandidateStatus
from app.domain.query_candidate_ranking import QueryCandidateRanking
from app.domain.query_candidate_validation import (
    QueryCandidateValidation,
    QueryCandidateValidationStatus,
)

if TYPE_CHECKING:
    from app.domain.feedback_execution import FeedbackExecutionContext


class QueryExecutionEligibility(StrEnum):
    READY = "ready"
    BLOCKED = "blocked"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class QueryExecutionDecision:
    """Explain whether a candidate may become an execution request."""

    candidate_id: UUID
    eligibility: QueryExecutionEligibility
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("query execution decision reason must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": str(self.candidate_id),
            "eligibility": self.eligibility.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class QueryExecutionRequest:
    """A request payload ready for a future search adapter."""

    execution_id: UUID
    run_id: UUID
    question_id: str
    gap_id: UUID
    dimension_key: str
    candidate_id: UUID
    ranking_score: float
    query_text: str
    source_constraints: tuple[str, ...]
    requirement_type: str
    execution_reason: str
    feedback_id: UUID | None = None
    feedback_execution_id: UUID | None = None
    trigger_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.query_text.strip():
            raise ValueError("query execution request text must not be empty")
        if not 0 <= self.ranking_score <= 1:
            raise ValueError("ranking score must be between 0 and 1")
        if not self.execution_reason.strip():
            raise ValueError("query execution reason must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "execution_id": str(self.execution_id),
            "run_id": str(self.run_id),
            "question_id": self.question_id,
            "gap_id": str(self.gap_id),
            "dimension_key": self.dimension_key,
            "candidate_id": str(self.candidate_id),
            "ranking_score": self.ranking_score,
            "query_text": self.query_text,
            "source_constraints": list(self.source_constraints),
            "requirement_type": self.requirement_type,
            "execution_reason": self.execution_reason,
            "feedback_id": str(self.feedback_id) if self.feedback_id else None,
            "feedback_execution_id": (
                str(self.feedback_execution_id)
                if self.feedback_execution_id
                else None
            ),
            "trigger_reason": self.trigger_reason,
        }


class QueryExecutionAdapter:
    """Create request data without calling a provider or scheduling work."""

    @staticmethod
    def evaluate(
        candidate: QueryCandidate,
        *,
        validation: QueryCandidateValidation | None,
        ranking: QueryCandidateRanking | None,
        requirement: GapRequirement,
    ) -> QueryExecutionDecision:
        if requirement.closure_status is GapClosureStatus.CLOSED:
            return QueryExecutionDecision(
                candidate_id=candidate.query_candidate_id,
                eligibility=QueryExecutionEligibility.BLOCKED,
                reason="gap_already_closed",
            )
        if candidate.status is QueryCandidateStatus.REJECTED:
            return QueryExecutionDecision(
                candidate_id=candidate.query_candidate_id,
                eligibility=QueryExecutionEligibility.REJECTED,
                reason="invalid_query_candidate",
            )
        if candidate.status is QueryCandidateStatus.UNKNOWN:
            return QueryExecutionDecision(
                candidate_id=candidate.query_candidate_id,
                eligibility=QueryExecutionEligibility.UNKNOWN,
                reason="candidate_status_unknown",
            )
        if validation is None:
            return QueryExecutionDecision(
                candidate_id=candidate.query_candidate_id,
                eligibility=QueryExecutionEligibility.UNKNOWN,
                reason="candidate_validation_missing",
            )
        if validation.query_candidate_id != candidate.query_candidate_id:
            raise ValueError("validation and query candidate do not match")
        if validation.validation_status is QueryCandidateValidationStatus.INVALID:
            return QueryExecutionDecision(
                candidate_id=candidate.query_candidate_id,
                eligibility=QueryExecutionEligibility.REJECTED,
                reason="invalid_query_candidate",
            )
        if validation.validation_status is not QueryCandidateValidationStatus.VALID:
            return QueryExecutionDecision(
                candidate_id=candidate.query_candidate_id,
                eligibility=QueryExecutionEligibility.BLOCKED,
                reason="candidate_validation_not_ready",
            )
        if ranking is None:
            return QueryExecutionDecision(
                candidate_id=candidate.query_candidate_id,
                eligibility=QueryExecutionEligibility.BLOCKED,
                reason="candidate_ranking_missing",
            )
        if ranking.candidate_id != candidate.query_candidate_id:
            raise ValueError("ranking and query candidate do not match")
        return QueryExecutionDecision(
            candidate_id=candidate.query_candidate_id,
            eligibility=QueryExecutionEligibility.READY,
            reason="valid_candidate_with_ranking_and_open_gap",
        )

    @staticmethod
    def create_request(
        candidate: QueryCandidate,
        *,
        validation: QueryCandidateValidation | None,
        ranking: QueryCandidateRanking | None,
        requirement: GapRequirement,
        feedback_context: FeedbackExecutionContext | None = None,
    ) -> tuple[QueryExecutionRequest, ...]:
        decision = QueryExecutionAdapter.evaluate(
            candidate,
            validation=validation,
            ranking=ranking,
            requirement=requirement,
        )
        if decision.eligibility is not QueryExecutionEligibility.READY:
            return ()
        assert ranking is not None
        return (
            QueryExecutionRequest(
                execution_id=uuid5(
                    NAMESPACE_URL,
                    f"query-execution:{candidate.query_candidate_id}",
                ),
                run_id=candidate.run_id,
                question_id=candidate.question_id,
                gap_id=candidate.gap_id,
                dimension_key=candidate.dimension_key,
                candidate_id=candidate.query_candidate_id,
                ranking_score=ranking.score,
                query_text=candidate.query_text,
                source_constraints=candidate.constraints,
                requirement_type=candidate.requirement_type,
                execution_reason="; ".join(ranking.reasons),
                feedback_id=(
                    feedback_context.feedback_id
                    if feedback_context is not None
                    else candidate.feedback_id
                ),
                feedback_execution_id=(
                    feedback_context.feedback_execution_id
                    if feedback_context is not None
                    else None
                ),
                trigger_reason=(
                    candidate.avoid_previous_failure_reason
                    if feedback_context is not None
                    else None
                ),
            ),
        )
