"""Unit tests for the Phase 13.1 Closure Feedback Completeness framework.

These cover the generic feedback decision table, the CLOSED-gap behaviour,
the completeness check/invariant, and downstream ResearchNeed consumption.
Nothing here depends on a specific question id, benchmark, or gap id.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from app.domain.closure_feedback import ClosureFeedbackReason
from app.domain.closure_feedback_completeness import (
    MISSING_FEEDBACK_REASON,
    check_feedback_completeness,
    complete_feedback_gaps,
    derive_feedback_reason,
    feedback_completeness_violations,
    generate_completeness_feedback,
    missing_feedback_requirements,
)
from app.domain.evidence_alignment import EvidenceAlignmentStatus
from app.domain.gap_closure import (
    GapClosureStatus,
    GapRequirement,
    GapRequirementType,
    VerificationStatus,
)
from app.domain.research_need import ResearchNeedGenerator, ResearchNeedType

RUN_ID = UUID("00000000-0000-0000-0000-000000000abc")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _requirement(**overrides: object) -> GapRequirement:
    """Build a satisfiable-but-open requirement; individual rules are toggled."""

    defaults: dict[str, object] = {
        "gap_id": uuid4(),
        "run_id": RUN_ID,
        "question_id": "qA",
        "dimension_key": "qA:d1",
        "requirement_type": GapRequirementType.DIMENSION_COVERAGE,
        "criterion": "generic criterion",
        "required_evidence_count": 1,
        "required_independent_sources": 1,
        "current_evidence_count": 1,
        "current_independent_sources": 1,
        "verification_status": VerificationStatus.VERIFIED,
        "closure_status": GapClosureStatus.OPEN,
        "created_at": NOW,
        "updated_at": NOW,
        "state_version": 1,
    }
    defaults.update(overrides)
    return GapRequirement(**defaults)  # type: ignore[arg-type]


# --- Rule / Case 1: zero evidence -----------------------------------------

def test_case_one_zero_evidence_gap_is_insufficient() -> None:
    requirement = _requirement(current_evidence_count=0)
    assert derive_feedback_reason(requirement) is ClosureFeedbackReason.INSUFFICIENT_EVIDENCE


# --- Rule / Case 2: independent source shortfall --------------------------

def test_case_two_independent_source_gap() -> None:
    requirement = _requirement(required_independent_sources=2, current_independent_sources=1)
    reason = derive_feedback_reason(requirement)
    assert reason is ClosureFeedbackReason.MISSING_INDEPENDENT_SOURCE


# --- Rule / Case 3: claim verification outstanding ------------------------

def test_case_three_claim_verification_gap() -> None:
    requirement = _requirement(
        requirement_type=GapRequirementType.CLAIM_VERIFICATION,
        verification_status=VerificationStatus.REQUIRED,
    )
    assert derive_feedback_reason(requirement) is ClosureFeedbackReason.CLAIM_NOT_VERIFIED


# --- Rule / Case 4: evidence quality insufficient -------------------------

@pytest.mark.parametrize(
    "alignment_status",
    [EvidenceAlignmentStatus.PARTIAL, EvidenceAlignmentStatus.ALIGNED],
)
def test_case_four_evidence_quality_gap(alignment_status: EvidenceAlignmentStatus) -> None:
    requirement = _requirement(requirement_type=GapRequirementType.EVIDENCE_QUALITY)
    reason = derive_feedback_reason(requirement, alignment_status)
    assert reason is ClosureFeedbackReason.EVIDENCE_QUALITY_LOW


# --- Rule 5 / 6 reachability ----------------------------------------------

def test_dimension_mismatch_when_evidence_not_aligned() -> None:
    reason = derive_feedback_reason(
        _requirement(), EvidenceAlignmentStatus.NOT_ALIGNED
    )
    assert reason is ClosureFeedbackReason.DIMENSION_MISMATCH


def test_no_valid_path_is_the_generic_residual() -> None:
    assert derive_feedback_reason(_requirement()) is ClosureFeedbackReason.NO_VALID_PATH


# --- Case 5 ( CLOSED ) : no feedback anywhere ------------------------------

def test_case_five_closed_gap_produces_no_feedback() -> None:
    requirement = _requirement(closure_status=GapClosureStatus.CLOSED)
    assert generate_completeness_feedback(requirement, now=NOW) is None
    assert ResearchNeedGenerator.generate(requirement, now=NOW) == ()


def test_all_six_buckets_are_reachable() -> None:
    cases: list[tuple[GapRequirement, EvidenceAlignmentStatus]] = [
        (_requirement(current_evidence_count=0), EvidenceAlignmentStatus.UNKNOWN),
        (_requirement(required_independent_sources=2, current_independent_sources=1),
         EvidenceAlignmentStatus.UNKNOWN),
        (_requirement(
            requirement_type=GapRequirementType.CLAIM_VERIFICATION,
            verification_status=VerificationStatus.REQUIRED,
        ), EvidenceAlignmentStatus.UNKNOWN),
        (_requirement(requirement_type=GapRequirementType.EVIDENCE_QUALITY),
         EvidenceAlignmentStatus.PARTIAL),
        (_requirement(), EvidenceAlignmentStatus.NOT_ALIGNED),
        (_requirement(), EvidenceAlignmentStatus.UNKNOWN),
    ]
    reasons = {derive_feedback_reason(req, status) for req, status in cases}
    assert reasons == {
        ClosureFeedbackReason.INSUFFICIENT_EVIDENCE,
        ClosureFeedbackReason.MISSING_INDEPENDENT_SOURCE,
        ClosureFeedbackReason.CLAIM_NOT_VERIFIED,
        ClosureFeedbackReason.EVIDENCE_QUALITY_LOW,
        ClosureFeedbackReason.DIMENSION_MISMATCH,
        ClosureFeedbackReason.NO_VALID_PATH,
    }


# --- Case 6: genericity — identity must not change the outcome -------------

@pytest.mark.parametrize(
    ("question_id", "dimension_key"),
    [
        ("q1", "q1:d1"),
        ("q4", "q4:d2"),
        ("q7", "q7:d3"),
        ("anything", "unrelated:zz"),
    ],
)
def test_case_six_reason_is_identity_independent(
    question_id: str, dimension_key: str
) -> None:
    requirement = _requirement(
        question_id=question_id,
        dimension_key=dimension_key,
        current_evidence_count=0,
    )
    feedback = generate_completeness_feedback(requirement, now=NOW)
    assert feedback is not None
    assert feedback.failure_reason is ClosureFeedbackReason.INSUFFICIENT_EVIDENCE


def test_same_state_different_ids_yield_same_reason() -> None:
    a = _requirement(question_id="q1", dimension_key="q1:d1", gap_id=uuid4())
    b = _requirement(question_id="q9", dimension_key="q9:d5", gap_id=uuid4())
    assert derive_feedback_reason(a) == derive_feedback_reason(b)


# --- Generator output integrity -------------------------------------------

def test_generated_feedback_carries_generic_fields() -> None:
    requirement = _requirement(required_independent_sources=2, current_independent_sources=1)
    feedback = generate_completeness_feedback(requirement, now=NOW)
    assert feedback is not None
    assert feedback.run_id == requirement.run_id
    assert feedback.gap_id == requirement.gap_id
    assert feedback.dimension_key == requirement.dimension_key
    assert feedback.missing_requirement == "independent_source"
    assert feedback.recommended_need_type is ResearchNeedType.INDEPENDENT_SOURCE
    assert feedback.failure_reason is ClosureFeedbackReason.MISSING_INDEPENDENT_SOURCE


# --- Completeness check + invariant ---------------------------------------

def test_check_reports_present_and_missing_feedback() -> None:
    covered = _requirement()
    uncovered = _requirement(current_evidence_count=0)
    closed = _requirement(closure_status=GapClosureStatus.CLOSED)
    existing = generate_completeness_feedback(covered, now=NOW)
    assert existing is not None

    records = check_feedback_completeness(
        [covered, uncovered, closed], [existing]
    )
    by_gap = {record.gap_id: record for record in records}
    assert str(closed.gap_id) not in by_gap  # CLOSED gaps are excluded
    assert by_gap[str(covered.gap_id)].has_feedback is True
    assert by_gap[str(covered.gap_id)].feedback_id == str(existing.feedback_id)
    assert by_gap[str(uncovered.gap_id)].has_feedback is False
    assert by_gap[str(uncovered.gap_id)].feedback_id is None
    assert by_gap[str(uncovered.gap_id)].missing_reason == MISSING_FEEDBACK_REASON


def test_completeness_invariant_after_backfill() -> None:
    gaps = [
        _requirement(current_evidence_count=0),
        _requirement(required_independent_sources=2, current_independent_sources=1),
        _requirement(
            requirement_type=GapRequirementType.CLAIM_VERIFICATION,
            verification_status=VerificationStatus.REQUIRED,
        ),
        _requirement(closure_status=GapClosureStatus.CLOSED),
    ]
    # With no feedback yet, every non-CLOSED gap violates the invariant.
    assert set(feedback_completeness_violations(gaps, [])) == {
        str(gaps[0].gap_id),
        str(gaps[1].gap_id),
        str(gaps[2].gap_id),
    }
    backfill = complete_feedback_gaps(gaps, [], now=NOW)
    # One feedback per non-CLOSED gap, and the CLOSED gap receives none.
    assert len(backfill) == 3
    assert str(gaps[3].gap_id) not in {str(fb.gap_id) for fb in backfill}
    # After backfill the invariant holds: no violations remain.
    assert feedback_completeness_violations(gaps, backfill) == ()


def test_missing_feedback_requirements_skips_closed_and_covered() -> None:
    open_gap = _requirement(current_evidence_count=0)
    closed_gap = _requirement(closure_status=GapClosureStatus.CLOSED)
    missing = missing_feedback_requirements([open_gap, closed_gap], [])
    assert [str(g.gap_id) for g in missing] == [str(open_gap.gap_id)]


# --- Case 4 wiring: ResearchNeed consumes the completeness feedback -------

def test_research_need_consumes_completeness_feedback() -> None:
    requirement = _requirement(required_independent_sources=2, current_independent_sources=1)
    feedback = generate_completeness_feedback(requirement, now=NOW)
    assert feedback is not None
    need = ResearchNeedGenerator.generate(
        requirement, feedback=feedback, now=NOW
    )[0]
    assert need.feedback_id == feedback.feedback_id
    assert need.need_type is feedback.recommended_need_type
