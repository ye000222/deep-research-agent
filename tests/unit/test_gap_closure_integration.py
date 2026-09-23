from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from app.domain.gap_closure import (
    ClosureEvaluation,
    ClosureSnapshot,
    GapClosureStatus,
    GapRequirement,
    GapRequirementType,
    VerificationStatus,
    project_gap_requirements,
)
from app.domain.recovery_execution import RecoveryOutcome, build_recovery_event

RUN_ID = UUID("00000000-0000-0000-0000-000000000001")
ATTEMPT_ID = UUID("00000000-0000-0000-0000-000000000002")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _requirements(
    *,
    coverage: float,
    evidence: int,
    sources: int,
    dimension_key: str,
    required_sources: int,
) -> tuple[GapRequirement, ...]:
    return project_gap_requirements(
        run_id=RUN_ID,
        plan_version=1,
        state_version=3,
        coverage_map=[
            {
                "dimension_key": dimension_key.split(":", 1)[0],
                "requirement_statuses": [
                    {
                        "dimension_key": dimension_key,
                        "criterion": "test criterion",
                        "coverage": coverage,
                        "accepted_evidence": evidence,
                        "independent_sources": sources,
                        "required_sources": required_sources,
                    }
                ],
            }
        ],
        now=NOW,
    )


def test_q5_recovery_can_reference_a_closed_closure_evaluation() -> None:
    requirement = _requirements(
        coverage=1.0,
        evidence=9,
        sources=2,
        dimension_key="q5:d1",
        required_sources=2,
    )[0]
    evaluation = ClosureEvaluation.evaluate(
        requirement,
        before_snapshot=ClosureSnapshot(coverage=0.5, evidence_count=1),
        after_snapshot=ClosureSnapshot(
            coverage=requirement.current_coverage,
            evidence_count=requirement.current_evidence_count,
            independent_sources=requirement.current_independent_sources,
            verification_status=requirement.verification_status,
        ),
    )
    outcome = RecoveryOutcome.evaluate(
        run_id=RUN_ID,
        question_id="q5",
        plan_version=1,
        recovery_attempt_id=ATTEMPT_ID,
        coverage_before=0.5,
        coverage_after=1.0,
        gap_count_before=1,
        gap_count_after=0,
        accepted_evidence_before=1,
        accepted_evidence_after=9,
        candidate_evidence_before=1,
        candidate_evidence_after=9,
        independent_sources_before=1,
        independent_sources_after=2,
        tokens_reserved=100,
        tokens_used=90,
        new_sources_count=1,
        new_independent_sources_count=1,
        closure_evaluation_id=evaluation.evaluation_id,
        closure_evaluation_ids=(evaluation.evaluation_id,),
        closed_gap_ids=(str(requirement.gap_id),),
        remaining_gap_ids=(),
    )

    event = build_recovery_event(
        event="recovery.completed",
        run_id=RUN_ID,
        question_id="q5",
        plan_version=1,
        attempt_id=ATTEMPT_ID,
        coverage_before=0.5,
        coverage_after=1.0,
        gap_before=("q5:d1",),
        gap_after=("stale:old-state",),
        outcome=outcome,
    )

    assert evaluation.closure_status is GapClosureStatus.CLOSED
    assert event["closure_evaluation_id"] == str(evaluation.evaluation_id)
    assert event["closed_gap_ids"] == [str(requirement.gap_id)]
    assert event["remaining_gap_ids"] == []
    assert "gap_after" not in event


def test_q7_remains_partial_when_independent_source_requirement_is_missing() -> None:
    requirement = _requirements(
        coverage=0.5,
        evidence=1,
        sources=1,
        dimension_key="q7:d2",
        required_sources=2,
    )[0]
    evaluation = ClosureEvaluation.evaluate(
        requirement,
        before_snapshot=ClosureSnapshot(coverage=0.25),
        after_snapshot=ClosureSnapshot(
            coverage=requirement.current_coverage,
            evidence_count=requirement.current_evidence_count,
            independent_sources=requirement.current_independent_sources,
            verification_status=requirement.verification_status,
        ),
    )

    assert requirement.requirement_type is GapRequirementType.INDEPENDENT_SOURCE
    assert evaluation.closure_status is GapClosureStatus.PARTIAL
    assert evaluation.requirement_satisfied is False


def test_q1_evidence_gain_keeps_remaining_requirement_reference() -> None:
    requirement = _requirements(
        coverage=0.5,
        evidence=1,
        sources=1,
        dimension_key="q1:d2",
        required_sources=2,
    )[0]
    evaluation = ClosureEvaluation.evaluate(
        requirement,
        before_snapshot=ClosureSnapshot(coverage=0.5, evidence_count=1),
        after_snapshot=ClosureSnapshot(
            coverage=0.5,
            evidence_count=2,
            independent_sources=1,
            verification_status=VerificationStatus.REQUIRED,
        ),
    )

    assert evaluation.closure_status is GapClosureStatus.PARTIAL
    assert evaluation.after_snapshot.independent_sources == 1


def test_q4_unresolved_claim_projects_to_claim_verification() -> None:
    requirements = project_gap_requirements(
        run_id=RUN_ID,
        plan_version=1,
        state_version=3,
        coverage_map=[
            {
                "dimension_key": "q4",
                "requirement_statuses": [
                    {
                        "dimension_key": "q4:d1",
                        "criterion": "claim criterion",
                        "coverage": 1.0,
                        "accepted_evidence": 4,
                        "independent_sources": 2,
                        "required_sources": 2,
                    }
                ],
            }
        ],
        claim_states={
            "claim-1": {
                "dimension_key": "q4:d1",
                "unresolved": True,
            }
        },
        now=NOW,
    )

    assert requirements[0].requirement_type is GapRequirementType.CLAIM_VERIFICATION
    # verification_status is a legacy, non-canonical projection; it can stay
    # REQUIRED while the requirement still closes on the numeric bar.
    assert requirements[0].verification_status is VerificationStatus.REQUIRED
    # Branch A: the dimension satisfies independence (2 owners >= bar of 2), so
    # the claim-verification requirement now closes rather than being pinned
    # PARTIAL by the dead verification_status flag.
    assert requirements[0].closure_status is GapClosureStatus.CLOSED

    below_bar = project_gap_requirements(
        run_id=RUN_ID,
        plan_version=1,
        state_version=3,
        coverage_map=[
            {
                "dimension_key": "q4",
                "requirement_statuses": [
                    {
                        "dimension_key": "q4:d2",
                        "criterion": "claim criterion",
                        "coverage": 1.0,
                        "accepted_evidence": 4,
                        "independent_sources": 1,
                        "required_sources": 2,
                    }
                ],
            }
        ],
        claim_states={
            "claim-1": {
                "dimension_key": "q4:d2",
                "unresolved": True,
            }
        },
        now=NOW,
    )
    # Task R invariant: a required bar above the observed independence can never
    # close, independent of the legacy verification flag.
    assert below_bar[0].requirement_type is GapRequirementType.CLAIM_VERIFICATION
    assert below_bar[0].closure_status is not GapClosureStatus.CLOSED


def test_recovery_outcome_does_not_override_closure_state_with_legacy_gap_after() -> None:
    outcome = RecoveryOutcome.evaluate(
        run_id=RUN_ID,
        question_id="q1",
        plan_version=1,
        recovery_attempt_id=ATTEMPT_ID,
        coverage_before=0.2,
        coverage_after=0.3,
        gap_count_before=1,
        gap_count_after=1,
        accepted_evidence_before=1,
        accepted_evidence_after=2,
        candidate_evidence_before=1,
        candidate_evidence_after=2,
        independent_sources_before=1,
        independent_sources_after=1,
        tokens_reserved=100,
        tokens_used=90,
        new_sources_count=0,
        new_independent_sources_count=0,
        closure_evaluation_id=UUID("00000000-0000-0000-0000-000000000003"),
        closed_gap_ids=(),
        remaining_gap_ids=("q1:d2",),
    )
    event = build_recovery_event(
        event="recovery.completed",
        run_id=RUN_ID,
        question_id="q1",
        plan_version=1,
        attempt_id=ATTEMPT_ID,
        coverage_before=0.2,
        coverage_after=0.3,
        gap_before=("q1:d2",),
        gap_after=(),
        outcome=outcome,
    )

    assert event["remaining_gap_ids"] == ["q1:d2"]
    assert "gap_after" not in event
