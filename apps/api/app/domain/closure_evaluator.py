"""Integrate alignment observations into monotonic Gap closure evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from app.domain.closure_evaluation_context import ClosureEvaluationContext
from app.domain.evidence_alignment import EvidenceAlignmentReason
from app.domain.gap_closure import (
    ClosureEvaluation,
    ClosureSnapshot,
    GapClosureStatus,
    GapRequirement,
    VerificationStatus,
)


class ClosureEvaluationEventType(StrEnum):
    STARTED = "gap.closure.started"
    COMPLETED = "gap.closure.completed"
    TRANSITION = "gap.closure.transition"


@dataclass(frozen=True, slots=True)
class ClosureEvaluationResult:
    evaluation_id: UUID
    gap_id: UUID
    question_id: str
    before_status: GapClosureStatus
    after_status: GapClosureStatus
    closure_status: GapClosureStatus
    transition_reason: str
    aligned_evidence_count: int
    partial_evidence_count: int
    remaining_requirements: tuple[str, ...]
    closed_requirements: tuple[str, ...]
    state_version: int


@dataclass(frozen=True, slots=True)
class ClosureEvaluationEvent:
    event_type: ClosureEvaluationEventType
    run_id: UUID
    question_id: str
    gap_id: UUID
    evaluation_id: UUID
    before_status: GapClosureStatus
    after_status: GapClosureStatus
    state_version: int
    transition_reason: str
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class ClosureEvaluationExecution:
    requirement: GapRequirement
    result: ClosureEvaluationResult
    events: tuple[ClosureEvaluationEvent, ...]


class ClosureEvaluator:
    """Evaluate a ClosureEvaluationContext and apply only monotonic state changes."""

    @staticmethod
    def evaluate(
        context: ClosureEvaluationContext,
        *,
        requirement: GapRequirement,
        now: datetime | None = None,
    ) -> ClosureEvaluationExecution:
        if context.gap_id != requirement.gap_id:
            raise ValueError("closure context and requirement gaps do not match")
        timestamp = now or datetime.now(UTC)
        evaluation_id = uuid4()
        started = _event(
            ClosureEvaluationEventType.STARTED,
            requirement=requirement,
            evaluation_id=evaluation_id,
            before_status=requirement.closure_status,
            after_status=requirement.closure_status,
            state_version=requirement.state_version,
            transition_reason="closure_evaluation_started",
            timestamp=timestamp,
        )

        if requirement.closure_status is GapClosureStatus.CLOSED:
            after_requirement = requirement
            reason = "closed_state_immutable"
            raw_status = GapClosureStatus.CLOSED
        else:
            after_snapshot = _after_snapshot(context, requirement)
            raw_evaluation = ClosureEvaluation.evaluate(
                requirement,
                before_snapshot=context.before_snapshot,
                after_snapshot=after_snapshot,
            )
            raw_status = raw_evaluation.closure_status
            reason = _transition_reason(context, raw_evaluation.reason)
            if _status_rank(raw_status) < _status_rank(requirement.closure_status):
                raw_status = requirement.closure_status
                reason = "existing_closure_state_preserved"
            after_requirement = requirement.apply_closure_transition(
                new_status=raw_status,
                transition_reason=reason,
                updated_at=timestamp,
            )

        result = ClosureEvaluationResult(
            evaluation_id=evaluation_id,
            gap_id=requirement.gap_id,
            question_id=requirement.question_id,
            before_status=requirement.closure_status,
            after_status=after_requirement.closure_status,
            closure_status=raw_status,
            transition_reason=reason,
            aligned_evidence_count=context.aligned_evidence_count,
            partial_evidence_count=context.partial_evidence_count,
            remaining_requirements=_remaining_requirements(
                requirement,
                context,
                after_requirement.closure_status,
            ),
            closed_requirements=(
                (requirement.dimension_key,)
                if after_requirement.closure_status is GapClosureStatus.CLOSED
                else ()
            ),
            state_version=after_requirement.state_version,
        )
        completed = _event(
            ClosureEvaluationEventType.COMPLETED,
            requirement=after_requirement,
            evaluation_id=evaluation_id,
            before_status=requirement.closure_status,
            after_status=after_requirement.closure_status,
            state_version=after_requirement.state_version,
            transition_reason=reason,
            timestamp=timestamp,
        )
        events = [started, completed]
        if after_requirement.closure_status is not requirement.closure_status:
            events.insert(
                1,
                _event(
                    ClosureEvaluationEventType.TRANSITION,
                    requirement=after_requirement,
                    evaluation_id=evaluation_id,
                    before_status=requirement.closure_status,
                    after_status=after_requirement.closure_status,
                    state_version=after_requirement.state_version,
                    transition_reason=reason,
                    timestamp=timestamp,
                ),
            )
        return ClosureEvaluationExecution(
            requirement=after_requirement,
            result=result,
            events=tuple(events),
        )


def _after_snapshot(
    context: ClosureEvaluationContext,
    requirement: GapRequirement,
) -> ClosureSnapshot:
    verification_status = context.before_snapshot.verification_status
    if requirement.requirement_type.value == "claim_verification":
        # Observability projection only: mirror the numeric independence verdict
        # so the snapshot never reports a stale verification flag that conflicts
        # with the closure decision.  The decision itself is made from the
        # numeric ``independent_sources`` in ``_requirement_is_satisfied``.
        verification_status = (
            VerificationStatus.VERIFIED
            if context.independent_source_count
            >= requirement.required_independent_sources
            else VerificationStatus.REQUIRED
        )
    return ClosureSnapshot(
        coverage=context.before_snapshot.coverage,
        evidence_count=(
            context.before_snapshot.evidence_count
            + context.aligned_evidence_count
            + context.partial_evidence_count
        ),
        independent_sources=context.independent_source_count,
        verification_status=verification_status,
    )


def _transition_reason(context: ClosureEvaluationContext, fallback: str) -> str:
    reasons = {
        item.rejection_reason
        for item in context.evidence_alignments
        if item.rejection_reason is not None
    }
    if EvidenceAlignmentReason.MISSING_INDEPENDENT_SOURCE in reasons:
        return "missing_independent_source"
    if EvidenceAlignmentReason.CLAIM_NOT_VERIFIED in reasons:
        return "claim_verification_required"
    if EvidenceAlignmentReason.DIMENSION_MISMATCH in reasons:
        return "dimension_mismatch"
    return fallback


def _remaining_requirements(
    requirement: GapRequirement,
    context: ClosureEvaluationContext,
    status: GapClosureStatus,
) -> tuple[str, ...]:
    if status is GapClosureStatus.CLOSED:
        return ()
    if (
        requirement.requirement_type.value == "independent_source"
        and context.independent_source_count < requirement.required_independent_sources
    ):
        return ("independent_source",)
    if (
        requirement.requirement_type.value == "claim_verification"
        and context.independent_source_count
        < requirement.required_independent_sources
    ):
        return ("claim_verification",)
    return (requirement.dimension_key,)


def _status_rank(status: GapClosureStatus) -> int:
    return {
        GapClosureStatus.OPEN: 0,
        GapClosureStatus.PARTIAL: 1,
        GapClosureStatus.CLOSED: 2,
    }[status]


def _event(
    event_type: ClosureEvaluationEventType,
    *,
    requirement: GapRequirement,
    evaluation_id: UUID,
    before_status: GapClosureStatus,
    after_status: GapClosureStatus,
    state_version: int,
    transition_reason: str,
    timestamp: datetime,
) -> ClosureEvaluationEvent:
    return ClosureEvaluationEvent(
        event_type=event_type,
        run_id=requirement.run_id,
        question_id=requirement.question_id,
        gap_id=requirement.gap_id,
        evaluation_id=evaluation_id,
        before_status=before_status,
        after_status=after_status,
        state_version=state_version,
        transition_reason=transition_reason,
        timestamp=timestamp,
    )
