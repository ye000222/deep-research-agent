"""Context and guardrails for executing closure-feedback queries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, UUID, uuid5


@dataclass(frozen=True, slots=True)
class FeedbackExecutionContext:
    """Trace context carried by one feedback-driven execution handoff."""

    feedback_execution_id: UUID
    feedback_id: UUID
    run_id: UUID
    question_id: str
    gap_id: UUID
    dimension_key: str
    requirement_type: str
    research_need_id: UUID
    query_candidate_id: UUID
    created_at: datetime

    @classmethod
    def create(
        cls,
        *,
        feedback_id: UUID,
        run_id: UUID,
        question_id: str,
        gap_id: UUID,
        dimension_key: str,
        requirement_type: str,
        research_need_id: UUID,
        query_candidate_id: UUID,
        created_at: datetime | None = None,
    ) -> FeedbackExecutionContext:
        execution_id = uuid5(
            NAMESPACE_URL,
            f"feedback-execution:{feedback_id}:{query_candidate_id}",
        )
        return cls(
            feedback_execution_id=execution_id,
            feedback_id=feedback_id,
            run_id=run_id,
            question_id=question_id,
            gap_id=gap_id,
            dimension_key=dimension_key,
            requirement_type=requirement_type,
            research_need_id=research_need_id,
            query_candidate_id=query_candidate_id,
            created_at=created_at or datetime.now(UTC),
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
            "research_need_id": str(self.research_need_id),
            "query_candidate_id": str(self.query_candidate_id),
            "created_at": self.created_at.isoformat(),
        }


def feedback_execution_allowed(*, execution_count: int, limit: int) -> bool:
    """Bound feedback-driven handoffs without changing Recovery policy."""

    if execution_count < 0 or limit < 1:
        raise ValueError("execution count and limit must be non-negative/positive")
    return execution_count < limit
