"""Adapter that marks a feedback query for the existing Research Loop."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.domain.feedback_execution import FeedbackExecutionContext
from app.domain.query_execution import QueryExecutionRequest


@dataclass(frozen=True, slots=True)
class FeedbackQueryExecutionContext:
    feedback_execution_id: UUID
    feedback_id: UUID
    run_id: UUID
    question_id: str
    gap_id: UUID
    dimension_key: str
    requirement_type: str
    query_candidate_id: UUID
    query_execution_id: UUID
    trigger_reason: str

    @classmethod
    def from_request(
        cls,
        context: FeedbackExecutionContext,
        request: QueryExecutionRequest,
    ) -> FeedbackQueryExecutionContext:
        if request.feedback_id != context.feedback_id:
            raise ValueError("feedback request does not match its context")
        if request.feedback_execution_id != context.feedback_execution_id:
            raise ValueError("feedback execution request does not match its context")
        return cls(
            feedback_execution_id=context.feedback_execution_id,
            feedback_id=context.feedback_id,
            run_id=context.run_id,
            question_id=context.question_id,
            gap_id=context.gap_id,
            dimension_key=context.dimension_key,
            requirement_type=context.requirement_type,
            query_candidate_id=context.query_candidate_id,
            query_execution_id=request.execution_id,
            trigger_reason=request.trigger_reason or "feedback_recovery",
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "feedback_execution_id": str(self.feedback_execution_id),
            "feedback_id": str(self.feedback_id),
            "run_id": str(self.run_id),
            "question_id": self.question_id,
            "gap_id": str(self.gap_id),
            "dimension_key": self.dimension_key,
            "requirement_type": self.requirement_type,
            "query_candidate_id": str(self.query_candidate_id),
            "query_execution_id": str(self.query_execution_id),
            "trigger_reason": self.trigger_reason,
        }
