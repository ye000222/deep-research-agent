"""State and event contracts for an authorized recovery attempt."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID


class RecoveryExecutionState(StrEnum):
    NOT_STARTED = "not_started"
    AUTHORIZED = "authorized"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RecoveryOutcomeState(StrEnum):
    NOT_EVALUATED = "not_evaluated"
    EVALUATED = "evaluated"


class RecoveryOutcomeType(StrEnum):
    SUCCESSFUL = "successful"
    PARTIAL = "partial"
    LOW_GAIN = "low_gain"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RecoveryOutcome:
    """Observation-only before/after measurement for one Recovery attempt."""

    run_id: UUID
    question_id: str
    plan_version: int
    recovery_attempt_id: UUID
    coverage_before: float
    coverage_after: float
    gap_count_before: int
    gap_count_after: int
    accepted_evidence_before: int
    accepted_evidence_after: int
    candidate_evidence_before: int
    candidate_evidence_after: int
    independent_sources_before: int
    independent_sources_after: int
    tokens_reserved: int
    tokens_used: int
    new_sources_count: int
    new_independent_sources_count: int
    outcome_type: RecoveryOutcomeType
    utility_gain: float
    state: RecoveryOutcomeState = RecoveryOutcomeState.EVALUATED
    closure_evaluation_id: UUID | None = None
    closure_evaluation_ids: tuple[UUID, ...] = ()
    closed_gap_ids: tuple[str, ...] = ()
    partial_gap_ids: tuple[str, ...] = ()
    remaining_gap_ids: tuple[str, ...] = ()

    @classmethod
    def evaluate(
        cls,
        *,
        run_id: UUID,
        question_id: str,
        plan_version: int,
        recovery_attempt_id: UUID,
        coverage_before: float,
        coverage_after: float,
        gap_count_before: int,
        gap_count_after: int,
        accepted_evidence_before: int,
        accepted_evidence_after: int,
        candidate_evidence_before: int,
        candidate_evidence_after: int,
        independent_sources_before: int,
        independent_sources_after: int,
        tokens_reserved: int,
        tokens_used: int,
        new_sources_count: int,
        new_independent_sources_count: int,
        execution_failed: bool = False,
        closure_evaluation_id: UUID | None = None,
        closure_evaluation_ids: tuple[UUID, ...] = (),
        closed_gap_ids: tuple[str, ...] = (),
        partial_gap_ids: tuple[str, ...] = (),
        remaining_gap_ids: tuple[str, ...] = (),
    ) -> RecoveryOutcome:
        coverage_delta = max(0.0, coverage_after - coverage_before)
        gap_reduction = max(0, gap_count_before - gap_count_after)
        accepted_delta = max(0, accepted_evidence_after - accepted_evidence_before)
        candidate_delta = max(0, candidate_evidence_after - candidate_evidence_before)
        independent_delta = max(
            0, independent_sources_after - independent_sources_before
        )
        useful_result = any(
            (
                coverage_delta > 0,
                gap_reduction > 0,
                accepted_delta > 0,
                candidate_delta > 0,
                independent_delta > 0,
                new_sources_count > 0,
            )
        )
        if execution_failed:
            outcome_type = RecoveryOutcomeType.FAILED
        elif not useful_result:
            outcome_type = (
                RecoveryOutcomeType.LOW_GAIN
                if tokens_used > 0 or tokens_reserved > 0
                else RecoveryOutcomeType.FAILED
            )
        elif (
            coverage_delta >= 0.2
            or gap_reduction > 0
            or accepted_delta >= 5
            or independent_delta > 0
        ):
            outcome_type = RecoveryOutcomeType.SUCCESSFUL
        else:
            outcome_type = RecoveryOutcomeType.PARTIAL
        utility_gain = round(
            coverage_delta
            + gap_reduction * 0.1
            + accepted_delta * 0.02
            + independent_delta * 0.05
            + new_sources_count * 0.01,
            6,
        )
        return cls(
            run_id=run_id,
            question_id=question_id,
            plan_version=plan_version,
            recovery_attempt_id=recovery_attempt_id,
            coverage_before=coverage_before,
            coverage_after=coverage_after,
            gap_count_before=max(0, gap_count_before),
            gap_count_after=max(0, gap_count_after),
            accepted_evidence_before=max(0, accepted_evidence_before),
            accepted_evidence_after=max(0, accepted_evidence_after),
            candidate_evidence_before=max(0, candidate_evidence_before),
            candidate_evidence_after=max(0, candidate_evidence_after),
            independent_sources_before=max(0, independent_sources_before),
            independent_sources_after=max(0, independent_sources_after),
            tokens_reserved=max(0, tokens_reserved),
            tokens_used=max(0, tokens_used),
            new_sources_count=max(0, new_sources_count),
            new_independent_sources_count=max(0, new_independent_sources_count),
            outcome_type=outcome_type,
            utility_gain=utility_gain,
            closure_evaluation_id=closure_evaluation_id,
            closure_evaluation_ids=closure_evaluation_ids,
            closed_gap_ids=closed_gap_ids,
            partial_gap_ids=partial_gap_ids,
            remaining_gap_ids=remaining_gap_ids,
        )

    def as_event_fields(self) -> dict[str, object]:
        return {
            "outcome_state": self.state.value,
            "outcome_type": self.outcome_type.value,
            "coverage_before": self.coverage_before,
            "coverage_after": self.coverage_after,
            "gap_count_before": self.gap_count_before,
            "gap_count_after": self.gap_count_after,
            "accepted_evidence_before": self.accepted_evidence_before,
            "accepted_evidence_after": self.accepted_evidence_after,
            "candidate_evidence_before": self.candidate_evidence_before,
            "candidate_evidence_after": self.candidate_evidence_after,
            "independent_sources_before": self.independent_sources_before,
            "independent_sources_after": self.independent_sources_after,
            "tokens_reserved": self.tokens_reserved,
            "tokens_used": self.tokens_used,
            "new_sources_count": self.new_sources_count,
            "new_independent_sources_count": self.new_independent_sources_count,
            "utility_gain": self.utility_gain,
            "closure_evaluation_id": (
                str(self.closure_evaluation_id)
                if self.closure_evaluation_id is not None
                else None
            ),
            "closure_evaluation_ids": [
                str(value) for value in self.closure_evaluation_ids
            ],
            "closed_gap_ids": list(self.closed_gap_ids),
            "partial_gap_ids": list(self.partial_gap_ids),
            "remaining_gap_ids": list(self.remaining_gap_ids),
        }


@dataclass(frozen=True, slots=True)
class RecoveryContext:
    """Stable context shared by every event in one recovery attempt."""

    run_id: UUID
    question_id: str
    plan_version: int
    recovery_attempt_id: UUID
    trigger_reason: str
    coverage_before: float
    gap_before: tuple[str, ...]
    created_at: datetime

    @classmethod
    def create(
        cls,
        *,
        run_id: UUID,
        question_id: str,
        plan_version: int,
        recovery_attempt_id: UUID,
        trigger_reason: str,
        coverage_before: float,
        gap_before: tuple[str, ...],
    ) -> RecoveryContext:
        return cls(
            run_id=run_id,
            question_id=question_id,
            plan_version=plan_version,
            recovery_attempt_id=recovery_attempt_id,
            trigger_reason=trigger_reason,
            coverage_before=coverage_before,
            gap_before=gap_before,
            created_at=datetime.now(UTC),
        )

    def as_refs(self) -> dict[str, object]:
        return {
            "recovery_attempt_id": str(self.recovery_attempt_id),
            "recovery_question_id": self.question_id,
            "recovery_plan_version": self.plan_version,
            "recovery_trigger_reason": self.trigger_reason,
            "recovery_coverage_before": self.coverage_before,
            "recovery_gap_before": list(self.gap_before),
            "recovery_context_created_at": self.created_at.isoformat(),
        }


def merge_recovery_context(
    refs: Mapping[str, object],
    context: RecoveryContext | None,
) -> dict[str, object]:
    """Attach Recovery metadata without affecting ordinary events."""

    if context is None:
        return dict(refs)
    return {**context.as_refs(), **refs}


def clear_recovery_context(
    usage: Mapping[str, object],
    attempt_id: UUID,
) -> dict[str, object]:
    """Close only the active attempt that owns the stored context."""

    cleaned = dict(usage)
    if cleaned.get("recovery_attempt_id") != str(attempt_id):
        return cleaned
    cleaned.pop("recovery_attempt_id", None)
    cleaned.pop("recovery_question_id", None)
    cleaned.pop("recovery_context", None)
    return cleaned


@dataclass(slots=True)
class RecoveryExecution:
    run_id: UUID
    question_id: str
    plan_version: int
    attempt_id: UUID
    coverage_before: float
    gap_before: tuple[str, ...]
    tokens_reserved: int = 0
    state: RecoveryExecutionState = RecoveryExecutionState.NOT_STARTED
    coverage_after: float | None = None
    gap_after: tuple[str, ...] = ()
    reason: str | None = None
    accepted_evidence_before: int = 0
    accepted_evidence_after: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def authorize(
        cls,
        *,
        run_id: UUID,
        question_id: str,
        plan_version: int,
        attempt_id: UUID,
        coverage_before: float,
        gap_before: tuple[str, ...] = (),
        tokens_reserved: int = 0,
        accepted_evidence_before: int = 0,
    ) -> RecoveryExecution:
        execution = cls(
            run_id=run_id,
            question_id=question_id,
            plan_version=plan_version,
            attempt_id=attempt_id,
            coverage_before=coverage_before,
            gap_before=gap_before,
            tokens_reserved=tokens_reserved,
            accepted_evidence_before=accepted_evidence_before,
        )
        execution.state = RecoveryExecutionState.AUTHORIZED
        return execution

    def start(self) -> None:
        if self.state is not RecoveryExecutionState.AUTHORIZED:
            raise ValueError(f"cannot start recovery from {self.state}")
        self.state = RecoveryExecutionState.RUNNING

    def complete(
        self,
        *,
        coverage_after: float,
        gap_after: tuple[str, ...] = (),
        accepted_evidence_after: int = 0,
        reason: str | None = None,
    ) -> None:
        if self.state is not RecoveryExecutionState.RUNNING:
            raise ValueError(f"cannot complete recovery from {self.state}")
        self.coverage_after = coverage_after
        self.gap_after = gap_after
        self.accepted_evidence_after = accepted_evidence_after
        self.reason = reason
        self.state = RecoveryExecutionState.COMPLETED

    def fail(self, *, reason: str) -> None:
        if self.state is not RecoveryExecutionState.RUNNING:
            raise ValueError(f"cannot fail recovery from {self.state}")
        self.reason = reason
        self.state = RecoveryExecutionState.FAILED

    def cancel(self, *, reason: str = "cancelled") -> None:
        if self.state not in {
            RecoveryExecutionState.AUTHORIZED,
            RecoveryExecutionState.RUNNING,
        }:
            raise ValueError(f"cannot cancel recovery from {self.state}")
        self.reason = reason
        self.state = RecoveryExecutionState.CANCELLED


def build_recovery_event(
    *,
    event: str,
    run_id: UUID,
    question_id: str,
    plan_version: int,
    attempt_id: UUID,
    coverage_before: float,
    gap_before: tuple[str, ...] = (),
    coverage_after: float | None = None,
    gap_after: tuple[str, ...] = (),
    accepted_evidence_before: int = 0,
    accepted_evidence_after: int = 0,
    tokens_reserved: int = 0,
    reason: str | None = None,
    outcome: RecoveryOutcome | None = None,
) -> dict[str, object]:
    """Build the stable, queryable payload shared by recovery events."""

    payload: dict[str, object] = {
        "event": event,
        "run_id": str(run_id),
        "question_id": question_id,
        "plan_version": plan_version,
        "attempt_id": str(attempt_id),
        "recovery_attempt_id": str(attempt_id),
        "coverage_before": coverage_before,
        "coverage_after": coverage_after,
        "gap_before": list(gap_before),
        "accepted_evidence_before": accepted_evidence_before,
        "accepted_evidence_after": accepted_evidence_after,
        "tokens_reserved": tokens_reserved,
    }
    if outcome is None:
        # Kept for non-terminal/legacy events only. Terminal outcomes use the
        # ClosureEvaluation reference as the authoritative Gap projection.
        payload["gap_after"] = list(gap_after)
    if reason is not None:
        payload["reason"] = reason
    if outcome is not None:
        payload.update(outcome.as_event_fields())
    return payload
