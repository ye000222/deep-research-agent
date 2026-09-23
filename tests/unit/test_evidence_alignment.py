from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.domain.evidence_alignment import (
    EvidenceAlignment,
    EvidenceAlignmentReason,
    EvidenceAlignmentStatus,
    summarize_alignments,
)
from app.domain.gap_closure import (
    ClosureEvaluation,
    ClosureSnapshot,
    GapClosureStatus,
    GapRequirement,
    GapRequirementType,
    project_gap_requirements,
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
) -> GapRequirement:
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


def test_q5_evidence_alignment_is_aligned_without_changing_closure() -> None:
    requirement = _requirement("q5", "q5:d1", coverage=1.0, sources=2, required_sources=2)
    alignment = EvidenceAlignment.evaluate(
        requirement,
        evidence_id=uuid4(),
        evidence_dimension_key="q5:d1",
        evidence_quality_passed=True,
        independent_source_count=2,
        claim_verified=True,
        created_at=NOW,
    )
    evaluation = ClosureEvaluation.evaluate(
        requirement,
        before_snapshot=ClosureSnapshot(coverage=0.5),
        after_snapshot=ClosureSnapshot(
            coverage=1.0,
            evidence_count=4,
            independent_sources=2,
            verification_status=requirement.verification_status,
        ),
    )

    assert alignment.alignment_status is EvidenceAlignmentStatus.ALIGNED
    assert alignment.satisfies_requirement
    assert evaluation.closure_status is GapClosureStatus.CLOSED
    assert requirement.closure_status is GapClosureStatus.CLOSED


def test_q7_partial_alignment_explains_missing_independent_source() -> None:
    requirement = _requirement("q7", "q7:d2", coverage=0.7, sources=1, required_sources=2)
    alignment = EvidenceAlignment.evaluate(
        requirement,
        evidence_id=uuid4(),
        evidence_dimension_key="q7:d2",
        evidence_quality_passed=True,
        independent_source_count=1,
        claim_verified=True,
    )
    summary = summarize_alignments(requirement, [alignment])

    assert alignment.alignment_status is EvidenceAlignmentStatus.PARTIAL
    assert alignment.rejection_reason is EvidenceAlignmentReason.MISSING_INDEPENDENT_SOURCE
    assert (
        summary.remaining_requirement_reason
        is EvidenceAlignmentReason.MISSING_INDEPENDENT_SOURCE
    )


def test_q1_second_source_gap_remains_partial() -> None:
    requirement = _requirement("q1", "q1:d2", coverage=0.5, sources=1, required_sources=2)
    alignment = EvidenceAlignment.evaluate(
        requirement,
        evidence_id=uuid4(),
        evidence_dimension_key="q1:d2",
        evidence_quality_passed=True,
        independent_source_count=1,
        claim_verified=True,
    )

    assert alignment.alignment_status is EvidenceAlignmentStatus.PARTIAL
    assert requirement.closure_status is GapClosureStatus.PARTIAL


def test_q4_alignment_reports_unverified_claim() -> None:
    # Branch A: "unverified" is now the numeric independence fact, not a boolean.
    # One distinct owner against a bar of two stays PARTIAL with the claim reason.
    requirement = _requirement(
        "q4",
        "q4:d1",
        coverage=1.0,
        sources=1,
        required_sources=2,
        unresolved_claim=True,
    )
    alignment = EvidenceAlignment.evaluate(
        requirement,
        evidence_id=uuid4(),
        evidence_dimension_key="q4:d1",
        evidence_quality_passed=True,
        independent_source_count=1,
        claim_verified=False,
    )

    assert requirement.requirement_type is GapRequirementType.CLAIM_VERIFICATION
    assert alignment.alignment_status is EvidenceAlignmentStatus.PARTIAL
    assert alignment.rejection_reason is EvidenceAlignmentReason.CLAIM_NOT_VERIFIED


def test_q4_alignment_is_independent_of_hardcoded_claim_verified_flag() -> None:
    # Branch A core: two distinct owners satisfy the bar, so the evidence aligns
    # even though claim_verified is False -- independence is judged from the
    # canonical per-source-owner count, never from a hard-coded boolean or the
    # coarse source role.
    requirement = _requirement(
        "q4",
        "q4:d1",
        coverage=1.0,
        sources=2,
        required_sources=2,
        unresolved_claim=True,
    )
    alignment = EvidenceAlignment.evaluate(
        requirement,
        evidence_id=uuid4(),
        evidence_dimension_key="q4:d1",
        evidence_quality_passed=True,
        independent_source_count=2,
        claim_verified=False,
    )

    assert alignment.alignment_status is EvidenceAlignmentStatus.ALIGNED
    assert alignment.satisfies_requirement is True
    assert alignment.rejection_reason is None


def test_alignment_does_not_change_gap_state_or_normal_research_behavior() -> None:
    requirement = _requirement("q7", "q7:d2", coverage=0.7, sources=1, required_sources=2)
    EvidenceAlignment.evaluate(
        requirement,
        evidence_id=uuid4(),
        evidence_dimension_key="other:d1",
        evidence_quality_passed=True,
        independent_source_count=3,
        claim_verified=True,
    )

    assert requirement.closure_status is GapClosureStatus.PARTIAL
