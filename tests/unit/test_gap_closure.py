from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from app.domain.gap_closure import (
    GAP_STATE_SOURCE_PRIORITY,
    ClosureEvaluation,
    ClosureSnapshot,
    GapClosureStatus,
    GapRequirement,
    GapRequirementType,
    VerificationStatus,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000001")
GAP_ID = UUID("00000000-0000-0000-0000-000000000002")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _requirement(
    requirement_type: GapRequirementType,
    *,
    required_sources: int = 0,
    required_evidence: int = 0,
    required_coverage: float | None = None,
) -> GapRequirement:
    return GapRequirement(
        gap_id=GAP_ID,
        run_id=RUN_ID,
        question_id="q7",
        dimension_key="q7:d2",
        requirement_type=requirement_type,
        criterion="test criterion",
        required_evidence_count=required_evidence,
        required_independent_sources=required_sources,
        current_evidence_count=0,
        current_independent_sources=0,
        verification_status=VerificationStatus.REQUIRED,
        closure_status=GapClosureStatus.OPEN,
        created_at=NOW,
        updated_at=NOW,
        state_version=1,
        required_coverage=required_coverage,
    )


def test_q7_independent_source_requirement_is_partial_until_second_source() -> None:
    requirement = _requirement(
        GapRequirementType.INDEPENDENT_SOURCE,
        required_sources=2,
    )

    evaluation = ClosureEvaluation.evaluate(
        requirement,
        before_snapshot=ClosureSnapshot(independent_sources=0),
        after_snapshot=ClosureSnapshot(independent_sources=1),
    )

    assert evaluation.requirement_satisfied is False
    assert evaluation.closure_status is GapClosureStatus.PARTIAL
    assert evaluation.independent_source_delta == 1
    assert evaluation.reason == "evidence_or_verification_progress_without_closure"


def test_q5_dimension_requirement_can_close_independently_of_recovery_event() -> None:
    requirement = _requirement(
        GapRequirementType.DIMENSION_COVERAGE,
        required_coverage=1.0,
    )

    evaluation = ClosureEvaluation.evaluate(
        requirement,
        before_snapshot=ClosureSnapshot(coverage=0.5, evidence_count=10),
        after_snapshot=ClosureSnapshot(coverage=1.0, evidence_count=19),
    )

    assert evaluation.requirement_satisfied is True
    assert evaluation.closure_status is GapClosureStatus.CLOSED
    assert evaluation.evidence_delta == 9


def test_q1_evidence_gain_without_required_independent_source_is_partial() -> None:
    requirement = _requirement(
        GapRequirementType.INDEPENDENT_SOURCE,
        required_sources=2,
    )

    evaluation = ClosureEvaluation.evaluate(
        requirement,
        before_snapshot=ClosureSnapshot(evidence_count=2, independent_sources=1),
        after_snapshot=ClosureSnapshot(evidence_count=4, independent_sources=1),
    )

    assert evaluation.evidence_delta == 2
    assert evaluation.independent_source_delta == 0
    assert evaluation.closure_status is GapClosureStatus.PARTIAL


def test_q4_claim_verification_remains_open_without_independent_sources() -> None:
    # Branch A: claim verification satisfaction is the Requirement/dimension
    # numeric independence fact (distinct accepted owners >= required).  With a
    # bar of two but only one owner observed, the Requirement stays unsatisfied
    # regardless of the legacy verification_status flag.
    requirement = _requirement(
        GapRequirementType.CLAIM_VERIFICATION,
        required_sources=2,
    )

    evaluation = ClosureEvaluation.evaluate(
        requirement,
        before_snapshot=ClosureSnapshot(
            evidence_count=7,
            independent_sources=1,
            verification_status=VerificationStatus.REQUIRED,
        ),
        after_snapshot=ClosureSnapshot(
            evidence_count=7,
            independent_sources=1,
            verification_status=VerificationStatus.REQUIRED,
        ),
    )

    assert evaluation.requirement_satisfied is False
    assert evaluation.closure_status is GapClosureStatus.OPEN
    assert evaluation.reason == "requirement_not_satisfied"


def test_q4_claim_verification_closes_when_independent_sources_met() -> None:
    # Same bar of two, but two distinct owners observed -> the numeric
    # independence fact now closes the Requirement even though verification_status
    # was never persisted as VERIFIED (the legacy dead field is not canonical).
    requirement = _requirement(
        GapRequirementType.CLAIM_VERIFICATION,
        required_sources=2,
    )

    evaluation = ClosureEvaluation.evaluate(
        requirement,
        before_snapshot=ClosureSnapshot(
            evidence_count=7,
            independent_sources=1,
            verification_status=VerificationStatus.REQUIRED,
        ),
        after_snapshot=ClosureSnapshot(
            evidence_count=7,
            independent_sources=2,
            verification_status=VerificationStatus.REQUIRED,
        ),
    )

    assert evaluation.requirement_satisfied is True
    assert evaluation.closure_status is GapClosureStatus.CLOSED


def test_closure_evaluation_does_not_mutate_gap_requirement() -> None:
    requirement = _requirement(
        GapRequirementType.DIMENSION_COVERAGE,
        required_coverage=1.0,
    )

    evaluation = ClosureEvaluation.evaluate(
        requirement,
        before_snapshot=ClosureSnapshot(coverage=0.5),
        after_snapshot=ClosureSnapshot(coverage=1.0),
    )

    assert evaluation.closure_status is GapClosureStatus.CLOSED
    assert requirement.closure_status is GapClosureStatus.OPEN
    assert requirement.state_version == 1


def test_gap_state_source_priority_is_explicit() -> None:
    assert GAP_STATE_SOURCE_PRIORITY == (
        "closure_evaluation",
        "gap_requirement",
        "evaluator_snapshot",
        "recovery_outcome",
    )
