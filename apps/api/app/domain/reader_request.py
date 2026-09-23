"""Adapt Evidence Routing decisions into non-executing Reader requests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from app.domain.evidence_routing import (
    EvidenceRoutingDecision,
    EvidenceRoutingStatus,
)
from app.domain.gap_closure import GapClosureStatus, GapRequirement


class ReaderDispatchState(StrEnum):
    READY = "ready"
    DISPATCHED = "dispatched"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class ReaderDispatchEventType(StrEnum):
    READY = "reader.dispatch.ready"
    BLOCKED = "reader.dispatch.blocked"
    SKIPPED = "reader.dispatch.skipped"


@dataclass(frozen=True, slots=True)
class ReaderDispatchEvent:
    """One observation of Reader dispatch eligibility."""

    event_type: ReaderDispatchEventType
    run_id: UUID
    question_id: str
    gap_id: UUID
    routing_id: UUID
    alignment_id: UUID
    source_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class ReaderExecutionRequest:
    """A standard Reader entry payload that does not execute Reader."""

    routing_id: UUID
    alignment_id: UUID
    source_id: str
    run_id: UUID
    question_id: str
    gap_id: UUID
    dimension_key: str
    requirement_type: str
    execution_reason: str
    state: ReaderDispatchState
    events: tuple[ReaderDispatchEvent, ...] = ()
    query_execution_id: UUID | None = None

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("Reader request source must not be empty")
        if not self.execution_reason.strip():
            raise ValueError("Reader request reason must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "routing_id": str(self.routing_id),
            "alignment_id": str(self.alignment_id),
            "source_id": self.source_id,
            "run_id": str(self.run_id),
            "question_id": self.question_id,
            "gap_id": str(self.gap_id),
            "dimension_key": self.dimension_key,
            "requirement_type": self.requirement_type,
            "execution_reason": self.execution_reason,
            "state": self.state.value,
            "query_execution_id": (
                str(self.query_execution_id) if self.query_execution_id else None
            ),
            "events": [
                {
                    "event_type": event.event_type.value,
                    "run_id": str(event.run_id),
                    "question_id": event.question_id,
                    "gap_id": str(event.gap_id),
                    "routing_id": str(event.routing_id),
                    "alignment_id": str(event.alignment_id),
                    "source_id": event.source_id,
                    "reason": event.reason,
                }
                for event in self.events
            ],
        }


class ReaderRequestAdapter:
    """Create Reader dispatch data without calling Reader or Evidence code."""

    @staticmethod
    def create(
        decision: EvidenceRoutingDecision,
        *,
        requirement: GapRequirement,
    ) -> tuple[ReaderExecutionRequest, ...]:
        if requirement.closure_status is GapClosureStatus.CLOSED:
            return ()
        if decision.gap_id != requirement.gap_id:
            raise ValueError("routing decision and requirement gaps do not match")
        if decision.question_id != requirement.question_id:
            raise ValueError("routing decision and requirement questions do not match")

        state, event_type = _dispatch_state(decision.decision)
        event = ReaderDispatchEvent(
            event_type=event_type,
            run_id=requirement.run_id,
            question_id=requirement.question_id,
            gap_id=requirement.gap_id,
            routing_id=decision.routing_id,
            alignment_id=decision.alignment_id,
            source_id=decision.source_id,
            reason=decision.reason,
        )
        return (
            ReaderExecutionRequest(
                routing_id=decision.routing_id,
                alignment_id=decision.alignment_id,
                source_id=decision.source_id,
                run_id=requirement.run_id,
                question_id=requirement.question_id,
                gap_id=requirement.gap_id,
                dimension_key=requirement.dimension_key,
                requirement_type=requirement.requirement_type.value,
                execution_reason=decision.reason,
                state=state,
                events=(event,),
            ),
        )


def _dispatch_state(
    decision: EvidenceRoutingStatus,
) -> tuple[ReaderDispatchState, ReaderDispatchEventType]:
    if decision is EvidenceRoutingStatus.ROUTE:
        return ReaderDispatchState.READY, ReaderDispatchEventType.READY
    if decision is EvidenceRoutingStatus.SKIP:
        return ReaderDispatchState.SKIPPED, ReaderDispatchEventType.SKIPPED
    return ReaderDispatchState.BLOCKED, ReaderDispatchEventType.BLOCKED
