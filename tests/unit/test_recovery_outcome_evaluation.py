from __future__ import annotations

from uuid import UUID

from app.domain.recovery_execution import (
    RecoveryOutcome,
    RecoveryOutcomeState,
    RecoveryOutcomeType,
    build_recovery_event,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000001")
ATTEMPT_A = UUID("00000000-0000-0000-0000-000000000002")
ATTEMPT_B = UUID("00000000-0000-0000-0000-000000000003")


def _outcome(
    attempt_id: UUID,
    *,
    coverage_before: float = 0.2,
    coverage_after: float = 0.2,
    gaps_before: int = 2,
    gaps_after: int = 2,
    accepted_before: int = 5,
    accepted_after: int = 5,
    candidate_before: int = 5,
    candidate_after: int = 5,
    independent_before: int = 1,
    independent_after: int = 1,
    tokens_used: int = 12_000,
    new_sources: int = 0,
    failed: bool = False,
) -> RecoveryOutcome:
    return RecoveryOutcome.evaluate(
        run_id=RUN_ID,
        question_id="q1",
        plan_version=13,
        recovery_attempt_id=attempt_id,
        coverage_before=coverage_before,
        coverage_after=coverage_after,
        gap_count_before=gaps_before,
        gap_count_after=gaps_after,
        accepted_evidence_before=accepted_before,
        accepted_evidence_after=accepted_after,
        candidate_evidence_before=candidate_before,
        candidate_evidence_after=candidate_after,
        independent_sources_before=independent_before,
        independent_sources_after=independent_after,
        tokens_reserved=12_000,
        tokens_used=tokens_used,
        new_sources_count=new_sources,
        new_independent_sources_count=0,
        execution_failed=failed,
    )


def test_successful_recovery_is_evaluated() -> None:
    outcome = _outcome(
        ATTEMPT_A,
        coverage_after=0.6,
        gaps_after=0,
        accepted_after=12,
        candidate_after=14,
        independent_after=3,
        new_sources=3,
    )

    assert outcome.state is RecoveryOutcomeState.EVALUATED
    assert outcome.outcome_type is RecoveryOutcomeType.SUCCESSFUL
    assert outcome.utility_gain > 0


def test_partial_recovery_has_new_evidence_but_limited_gain() -> None:
    outcome = _outcome(
        ATTEMPT_A,
        coverage_after=0.3,
        accepted_after=7,
        candidate_after=8,
        new_sources=1,
    )

    assert outcome.outcome_type is RecoveryOutcomeType.PARTIAL


def test_low_gain_recovery_records_resource_use_without_progress() -> None:
    outcome = _outcome(ATTEMPT_A)

    assert outcome.outcome_type is RecoveryOutcomeType.LOW_GAIN
    assert outcome.tokens_used == 12_000
    assert outcome.utility_gain == 0


def test_failed_recovery_is_distinct_from_low_gain() -> None:
    outcome = _outcome(ATTEMPT_A, tokens_used=0, failed=True)

    assert outcome.outcome_type is RecoveryOutcomeType.FAILED
    assert outcome.state is RecoveryOutcomeState.EVALUATED


def test_outcomes_are_isolated_by_attempt_id() -> None:
    first = _outcome(ATTEMPT_A, coverage_after=0.6, gaps_after=0)
    second = _outcome(ATTEMPT_B, coverage_after=0.2, tokens_used=12_000)

    assert first.recovery_attempt_id == ATTEMPT_A
    assert second.recovery_attempt_id == ATTEMPT_B
    assert first.outcome_type is RecoveryOutcomeType.SUCCESSFUL
    assert second.outcome_type is RecoveryOutcomeType.LOW_GAIN


def test_completed_event_can_carry_outcome_measurement() -> None:
    outcome = _outcome(
        ATTEMPT_A,
        coverage_after=0.6,
        gaps_after=0,
        accepted_after=12,
        candidate_after=14,
        new_sources=3,
    )
    event = build_recovery_event(
        event="recovery.completed",
        run_id=RUN_ID,
        question_id="q1",
        plan_version=13,
        attempt_id=ATTEMPT_A,
        coverage_before=outcome.coverage_before,
        coverage_after=outcome.coverage_after,
        gap_before=("q1:d1", "q1:d2"),
        gap_after=(),
        accepted_evidence_before=outcome.accepted_evidence_before,
        accepted_evidence_after=outcome.accepted_evidence_after,
        tokens_reserved=outcome.tokens_reserved,
        outcome=outcome,
    )

    assert event["outcome_state"] == "evaluated"
    assert event["outcome_type"] == "successful"
    assert event["utility_gain"] > 0
