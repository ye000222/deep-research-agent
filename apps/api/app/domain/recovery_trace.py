"""Immutable trace model for one Recovery-to-Closure research attempt."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class RecoveryResearchTrace:
    """Correlate a Recovery attempt with every downstream research boundary."""

    trace_id: UUID
    run_id: UUID
    question_id: str
    recovery_attempt_id: UUID
    gap_id: UUID
    query_execution_ids: tuple[UUID, ...]
    reader_execution_ids: tuple[UUID, ...]
    extraction_ids: tuple[UUID, ...]
    alignment_ids: tuple[UUID, ...]
    closure_evaluation_ids: tuple[UUID, ...]
    before_gap_state: tuple[str, ...]
    after_gap_state: tuple[str, ...]
    before_coverage: float
    after_coverage: float
    outcome_type: str
    closure_transition_reason: str | None = None
    event_sequence: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly trace representation."""

        return {
            "trace_id": str(self.trace_id),
            "run_id": str(self.run_id),
            "question_id": self.question_id,
            "recovery_attempt_id": str(self.recovery_attempt_id),
            "gap_id": str(self.gap_id),
            "query_execution_ids": [str(value) for value in self.query_execution_ids],
            "reader_execution_ids": [
                str(value) for value in self.reader_execution_ids
            ],
            "extraction_ids": [str(value) for value in self.extraction_ids],
            "alignment_ids": [str(value) for value in self.alignment_ids],
            "closure_evaluation_ids": [
                str(value) for value in self.closure_evaluation_ids
            ],
            "before_gap_state": list(self.before_gap_state),
            "after_gap_state": list(self.after_gap_state),
            "before_coverage": self.before_coverage,
            "after_coverage": self.after_coverage,
            "outcome_type": self.outcome_type,
            "closure_transition_reason": self.closure_transition_reason,
            "event_sequence": list(self.event_sequence),
        }
