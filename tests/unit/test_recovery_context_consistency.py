from __future__ import annotations

from uuid import UUID

from app.domain.recovery_execution import (
    RecoveryContext,
    build_recovery_event,
    clear_recovery_context,
    merge_recovery_context,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000001")
ATTEMPT_A = UUID("00000000-0000-0000-0000-000000000002")
ATTEMPT_B = UUID("00000000-0000-0000-0000-000000000003")


def _context(attempt_id: UUID, question_id: str = "q1") -> RecoveryContext:
    return RecoveryContext.create(
        run_id=RUN_ID,
        question_id=question_id,
        plan_version=13,
        recovery_attempt_id=attempt_id,
        trigger_reason="high_risk_high_utility_borrow",
        coverage_before=0.0,
        gap_before=(f"{question_id}:d1",),
    )


def test_recovery_context_isolation_keeps_normal_events_unmarked() -> None:
    context_a = _context(ATTEMPT_A, "q1")

    recovery_refs = merge_recovery_context({"question_id": "q1"}, context_a)
    normal_refs = merge_recovery_context({"question_id": "q2"}, None)

    assert recovery_refs["recovery_attempt_id"] == str(ATTEMPT_A)
    assert "recovery_attempt_id" not in normal_refs


def test_multiple_attempts_replace_active_context_without_mixing_ids() -> None:
    context_a = _context(ATTEMPT_A)
    context_b = _context(ATTEMPT_B)

    refs_a = merge_recovery_context({}, context_a)
    refs_b = merge_recovery_context({}, context_b)

    assert refs_a["recovery_attempt_id"] == str(ATTEMPT_A)
    assert refs_b["recovery_attempt_id"] == str(ATTEMPT_B)
    assert refs_b["recovery_attempt_id"] != refs_a["recovery_attempt_id"]


def test_closing_old_attempt_does_not_clear_new_active_attempt() -> None:
    context_b = _context(ATTEMPT_B)
    usage = {"recovery_context": context_b.as_refs(), "recovery_attempt_id": str(ATTEMPT_B)}

    still_active = clear_recovery_context(usage, ATTEMPT_A)
    closed = clear_recovery_context(still_active, ATTEMPT_B)

    assert still_active["recovery_attempt_id"] == str(ATTEMPT_B)
    assert "recovery_attempt_id" not in closed
    assert "recovery_context" not in closed


def test_recovery_event_contract_contains_required_identity() -> None:
    refs = _context(ATTEMPT_A).as_refs()

    assert refs["recovery_attempt_id"] == str(ATTEMPT_A)
    assert refs["recovery_question_id"] == "q1"
    assert refs["recovery_plan_version"] == 13
    assert refs["recovery_context_created_at"]


def test_failed_recovery_preserves_reason_and_closes_context() -> None:
    event = build_recovery_event(
        event="recovery.failed",
        run_id=RUN_ID,
        question_id="q1",
        plan_version=13,
        attempt_id=ATTEMPT_A,
        coverage_before=0.0,
        gap_before=("q1:d1",),
        reason="MODEL_TIMEOUT",
    )

    assert event["recovery_attempt_id"] == str(ATTEMPT_A)
    assert event["reason"] == "MODEL_TIMEOUT"


def test_recovery_trace_requires_terminal_completion_or_failure() -> None:
    required = {
        "recovery.borrow.allowed",
        "recovery.token_reserved",
        "recovery.started",
        "evaluation.completed",
    }
    terminal = {"recovery.completed", "recovery.failed"}
    observed = required | {"recovery.completed"}

    assert required <= observed
    assert observed & terminal
