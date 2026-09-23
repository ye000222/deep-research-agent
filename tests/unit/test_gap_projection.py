from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest
from app.domain.gap_closure import (
    ClosureEvaluation,
    ClosureSnapshot,
    GapClosureStatus,
    project_gap_requirements,
)
from app.domain.gap_projection import (
    ResearchGapProjection,
    apply_closure_evaluation,
    project_research_gaps,
)
from app.infrastructure.db.research_models import GapRequirementRow

RUN_ID = UUID("00000000-0000-0000-0000-000000000001")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _project(
    question_id: str,
    dimension_key: str,
    *,
    coverage: float,
    evidence: int = 1,
    sources: int = 1,
    required_sources: int = 1,
    claim_unresolved: bool = False,
) -> ResearchGapProjection:
    requirements = project_gap_requirements(
        run_id=RUN_ID,
        plan_version=1,
        state_version=7,
        coverage_map=[
            {
                "dimension_key": question_id,
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
        claim_states=(
            {"claim": {"dimension_key": dimension_key, "unresolved": True}}
            if claim_unresolved
            else None
        ),
        now=NOW,
    )
    return project_research_gaps(requirements, plan_version=1)[0]


def test_q5_closed_projection_is_canonical_and_legacy_compatible() -> None:
    projection = _project("q5", "q5:d1", coverage=1.0, sources=2, required_sources=2)

    assert projection.status is GapClosureStatus.CLOSED
    assert projection.legacy_status == "resolved"
    assert projection.source == "gap_requirement"
    assert projection.state_version == 7


def test_q7_partial_projection_preserves_independent_source_gap() -> None:
    projection = _project(
        "q7",
        "q7:d2",
        coverage=0.7895,
        sources=1,
        required_sources=2,
    )

    assert projection.status is GapClosureStatus.PARTIAL
    assert projection.dimension_key == "q7:d2"
    assert projection.unresolved


def test_q1_partial_projection_identifies_remaining_dimension() -> None:
    projection = _project(
        "q1",
        "q1:d2",
        coverage=0.5,
        evidence=2,
        sources=1,
        required_sources=2,
    )

    assert projection.status is GapClosureStatus.PARTIAL
    assert projection.dimension_key == "q1:d2"
    assert projection.legacy_status == "open"


def test_q4_claim_verification_closes_when_dimension_independence_met() -> None:
    # Branch A: a claim-verification Requirement projects to CLOSED once its
    # dimension satisfies the numeric independence bar (2 distinct owners for a
    # bar of 2), instead of being pinned to PARTIAL by the dead verification_status
    # field.  Claim-level *support* status is a separate concern and unchanged.
    projection = _project(
        "q4",
        "q4:d1",
        coverage=1.0,
        sources=2,
        required_sources=2,
        claim_unresolved=True,
    )

    assert projection.status is GapClosureStatus.CLOSED
    assert not projection.unresolved


def test_q4_claim_verification_stays_partial_below_independence_bar() -> None:
    projection = _project(
        "q4",
        "q4:d1",
        coverage=1.0,
        sources=1,
        required_sources=2,
        claim_unresolved=True,
    )

    assert projection.status is GapClosureStatus.PARTIAL
    assert projection.unresolved


def test_stale_legacy_status_cannot_override_canonical_projection() -> None:
    projection = _project("q5", "q5:d1", coverage=1.0, sources=2, required_sources=2)
    stale = ResearchGapProjection.from_legacy_record(
        gap_id=projection.gap_id,
        run_id=RUN_ID,
        plan_version=1,
        question_id="q5",
        dimension_key="q5:d1",
        status="open",
        updated_at=NOW,
    )

    assert projection.status is GapClosureStatus.CLOSED
    assert stale.source == "legacy_compatibility"
    assert stale.status is GapClosureStatus.OPEN


def test_q1_partial_transition_records_version_and_reason() -> None:
    projection = _project(
        "q1",
        "q1:d2",
        coverage=0.5,
        evidence=2,
        sources=1,
        required_sources=2,
    )
    requirement = project_gap_requirements(
        run_id=RUN_ID,
        plan_version=1,
        state_version=7,
        coverage_map=[
            {
                "dimension_key": "q1",
                "requirement_statuses": [
                    {
                        "dimension_key": "q1:d2",
                        "coverage": 0.5,
                        "accepted_evidence": 2,
                        "independent_sources": 1,
                        "required_sources": 2,
                    }
                ],
            }
        ],
        now=NOW,
    )[0]
    evaluation = ClosureEvaluation.evaluate(
        requirement,
        before_snapshot=ClosureSnapshot(coverage=0.25),
        after_snapshot=ClosureSnapshot(
            coverage=0.5,
            evidence_count=2,
            independent_sources=1,
            verification_status=requirement.verification_status,
        ),
    )
    updated = apply_closure_evaluation(
        requirement,
        evaluation,
        now=datetime(2026, 1, 2, tzinfo=UTC),
        transition_reason="q1_dimension_progress",
    )

    assert projection.status is GapClosureStatus.PARTIAL
    assert updated.closure_status is GapClosureStatus.PARTIAL
    assert updated.state_version == requirement.state_version + 1
    assert updated.transition_reason == "q1_dimension_progress"


def test_q5_closed_transition_is_monotonic() -> None:
    requirement = project_gap_requirements(
        run_id=RUN_ID,
        plan_version=1,
        state_version=7,
        coverage_map=[
            {
                "dimension_key": "q5",
                "requirement_statuses": [
                    {
                        "dimension_key": "q5:d1",
                        "coverage": 1.0,
                        "accepted_evidence": 3,
                        "independent_sources": 2,
                        "required_sources": 2,
                    }
                ],
            }
        ],
        now=NOW,
    )[0]
    evaluation = ClosureEvaluation.evaluate(
        requirement,
        before_snapshot=ClosureSnapshot(coverage=0.5, evidence_count=1),
        after_snapshot=ClosureSnapshot(
            coverage=1.0,
            evidence_count=3,
            independent_sources=2,
            verification_status=requirement.verification_status,
        ),
    )

    updated = apply_closure_evaluation(requirement, evaluation, now=NOW)

    assert updated.closure_status is GapClosureStatus.CLOSED
    assert updated.state_version == 8


def test_gap_transition_cannot_reopen_closed_requirement() -> None:
    requirement = project_gap_requirements(
        run_id=RUN_ID,
        plan_version=1,
        state_version=7,
        coverage_map=[
            {
                "dimension_key": "q7",
                "requirement_statuses": [
                    {
                        "dimension_key": "q7:d2",
                        "coverage": 1.0,
                        "accepted_evidence": 2,
                        "independent_sources": 2,
                        "required_sources": 2,
                    }
                ],
            }
        ],
        now=NOW,
    )[0]
    closed = apply_closure_evaluation(
        requirement,
        ClosureEvaluation.evaluate(
            requirement,
            before_snapshot=ClosureSnapshot(),
            after_snapshot=ClosureSnapshot(
                coverage=1.0,
                evidence_count=2,
                independent_sources=2,
                verification_status=requirement.verification_status,
            ),
        ),
        now=NOW,
    )
    with pytest.raises(ValueError, match="cannot move backwards"):
        apply_closure_evaluation(
            closed,
            ClosureEvaluation(
                evaluation_id=closed.gap_id,
                gap_id=closed.gap_id,
                before_snapshot=ClosureSnapshot(coverage=1.0),
                after_snapshot=ClosureSnapshot(coverage=0.0),
                evidence_delta=-2,
                independent_source_delta=-2,
                requirement_satisfied=False,
                closure_status=GapClosureStatus.OPEN,
                reason="stale_legacy_event",
            ),
            now=NOW,
        )


def test_gap_requirement_persistence_model_contains_canonical_state() -> None:
    columns = set(GapRequirementRow.__table__.columns.keys())

    assert {
        "gap_id",
        "run_id",
        "plan_version",
        "question_id",
        "dimension_key",
        "requirement_type",
        "criterion",
        "required_evidence_count",
        "required_independent_sources",
        "verification_status",
        "closure_status",
        "state_version",
        "transition_reason",
        "created_at",
        "updated_at",
    } <= columns
