from __future__ import annotations

from uuid import UUID

from app.domain.recovery_execution import (
    RecoveryExecution,
    RecoveryExecutionState,
    build_recovery_event,
)
from app.infrastructure.db.research_tools import ResearchToolRepository


def test_recovery_execution_transitions_from_authorized_to_completed() -> None:
    execution = RecoveryExecution.authorize(
        run_id=UUID("00000000-0000-0000-0000-000000000001"),
        question_id="q7",
        plan_version=13,
        attempt_id=UUID("00000000-0000-0000-0000-000000000002"),
        coverage_before=0.0,
        gap_before=("q7",),
        tokens_reserved=12000,
    )

    assert execution.state is RecoveryExecutionState.AUTHORIZED
    execution.start()
    assert execution.state is RecoveryExecutionState.RUNNING
    execution.complete(coverage_after=0.25, gap_after=())
    assert execution.state is RecoveryExecutionState.COMPLETED


def test_recovery_event_contains_question_and_result_context() -> None:
    event = build_recovery_event(
        event="recovery.completed",
        run_id=UUID("00000000-0000-0000-0000-000000000001"),
        question_id="q7",
        plan_version=13,
        attempt_id=UUID("00000000-0000-0000-0000-000000000002"),
        coverage_before=0.0,
        coverage_after=0.25,
        gap_before=("q7",),
        gap_after=(),
        accepted_evidence_before=0,
        accepted_evidence_after=2,
        tokens_reserved=12000,
        reason="coverage_improved",
    )

    assert event["event"] == "recovery.completed"
    assert event["question_id"] == "q7"
    assert event["attempt_id"]
    assert event["coverage_before"] == 0.0
    assert event["coverage_after"] == 0.25
    assert event["gap_before"] == ["q7"]
    assert event["gap_after"] == []
    assert event["accepted_evidence_before"] == 0
    assert event["accepted_evidence_after"] == 2


def test_repository_exposes_recovery_event_persistence() -> None:
    assert callable(getattr(ResearchToolRepository, "record_recovery_event", None))
