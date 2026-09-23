from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.domain.evidence_alignment import EvidenceAlignment
from app.domain.gap_closure import (
    ClosureEvaluation,
    ClosureSnapshot,
    GapClosureStatus,
    GapRequirementType,
    VerificationStatus,
    project_gap_requirements,
)
from app.domain.research_need import (
    ResearchNeedGenerator,
    ResearchNeedStatus,
    ResearchNeedType,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000001")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _requirement(
    question_id: str,
    dimension_key: str,
    *,
    coverage: float,
    sources: int,
    required_sources: int,
    unresolved_claim: bool = False,
):
    return project_gap_requirements(
        run_id=RUN_ID,
        plan_version=1,
        state_version=4,
        coverage_map=[
            {
                "dimension_key": question_id,
                "requirement_statuses": [
                    {
                        "dimension_key": dimension_key,
                        "coverage": coverage,
                        "accepted_evidence": 4,
                        "independent_sources": sources,
                        "required_sources": required_sources,
                    }
                ],
            }
        ],
        claim_states=(
            {"claim": {"dimension_key": dimension_key, "unresolved": True}}
            if unresolved_claim
            else None
        ),
        now=NOW,
    )[0]


def _alignment(requirement, *, reason: bool = True) -> EvidenceAlignment:
    return EvidenceAlignment.evaluate(
        requirement,
        evidence_id=uuid4(),
        evidence_dimension_key=requirement.dimension_key,
        evidence_quality_passed=True,
        independent_source_count=(
            requirement.current_independent_sources if reason else 2
        ),
        claim_verified=not reason,
        created_at=NOW,
    )


def test_q7_generates_independent_source_need() -> None:
    requirement = _requirement("q7", "q7:d2", coverage=0.7, sources=1, required_sources=2)

    needs = ResearchNeedGenerator.generate(
        requirement,
        alignments=[_alignment(requirement)],
        question_priority=2,
        now=NOW,
    )

    assert needs[0].need_type is ResearchNeedType.INDEPENDENT_SOURCE
    assert needs[0].status is ResearchNeedStatus.OPEN
    assert needs[0].priority == 2


def test_q1_generates_independent_source_need_for_second_source() -> None:
    requirement = _requirement("q1", "q1:d2", coverage=0.5, sources=1, required_sources=2)

    needs = ResearchNeedGenerator.generate(requirement, alignments=[_alignment(requirement)])

    assert needs[0].need_type is ResearchNeedType.INDEPENDENT_SOURCE


def test_q4_generates_claim_verification_need() -> None:
    requirement = _requirement(
        "q4",
        "q4:d1",
        coverage=1.0,
        sources=1,
        required_sources=2,
        unresolved_claim=True,
    )

    needs = ResearchNeedGenerator.generate(requirement, alignments=[_alignment(requirement)])

    assert requirement.requirement_type is GapRequirementType.CLAIM_VERIFICATION
    assert needs[0].need_type is ResearchNeedType.CLAIM_VERIFICATION


def test_q5_closed_gap_generates_no_open_need() -> None:
    requirement = _requirement("q5", "q5:d1", coverage=1.0, sources=2, required_sources=2)

    assert requirement.closure_status is GapClosureStatus.CLOSED
    assert ResearchNeedGenerator.generate(requirement) == ()


def test_insufficient_quality_generates_evidence_quality_need() -> None:
    requirement = _requirement("q7", "q7:d2", coverage=0.5, sources=1, required_sources=1)
    alignment = EvidenceAlignment.evaluate(
        requirement,
        evidence_id=uuid4(),
        evidence_dimension_key=requirement.dimension_key,
        evidence_quality_passed=False,
        independent_source_count=1,
        claim_verified=True,
        created_at=NOW,
    )

    needs = ResearchNeedGenerator.generate(requirement, alignments=[alignment])

    assert needs[0].need_type is ResearchNeedType.EVIDENCE_QUALITY


def test_closure_evaluation_closed_suppresses_need_without_mutating_requirement() -> None:
    requirement = _requirement("q7", "q7:d2", coverage=0.5, sources=1, required_sources=1)
    evaluation = ClosureEvaluation.evaluate(
        requirement,
        before_snapshot=ClosureSnapshot(coverage=0.0),
        after_snapshot=ClosureSnapshot(
            coverage=1.0,
            evidence_count=4,
            independent_sources=1,
            verification_status=VerificationStatus.VERIFIED,
        ),
    )

    assert evaluation.closure_status is GapClosureStatus.CLOSED
    assert ResearchNeedGenerator.generate(
        requirement,
        closure_evaluation=evaluation,
    ) == ()
    assert requirement.closure_status is GapClosureStatus.PARTIAL


def test_need_generation_does_not_execute_or_change_research() -> None:
    requirement = _requirement("q7", "q7:d2", coverage=0.5, sources=1, required_sources=2)

    needs = ResearchNeedGenerator.generate(requirement)

    assert len(needs) == 1
    assert needs[0].need_type is ResearchNeedType.INDEPENDENT_SOURCE
    assert needs[0].status is ResearchNeedStatus.OPEN
