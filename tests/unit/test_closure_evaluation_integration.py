from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

import pytest
from app.domain.closure_evaluation_context import ClosureEvaluationContext
from app.domain.closure_evaluator import (
    ClosureEvaluationEventType,
    ClosureEvaluator,
)
from app.domain.evidence_alignment_executor import EvidenceAlignmentExecutor
from app.domain.evidence_alignment_request import EvidenceAlignmentRequestFactory
from app.domain.evidence_extractor import (
    EvidenceExtractionResult,
    EvidenceExtractionStatus,
)
from app.domain.gap_closure import (
    ClosureSnapshot,
    GapClosureStatus,
    GapRequirementType,
    project_gap_requirements,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000045")
READER_ID = UUID("00000000-0000-0000-0000-000000000046")
ROUTING_ID = UUID("00000000-0000-0000-0000-000000000047")
ALIGNMENT_ID = UUID("00000000-0000-0000-0000-000000000048")
EXTRACTION_ID = UUID("00000000-0000-0000-0000-000000000049")


def _requirement(
    question_id: str,
    dimension_key: str,
    requirement_type: GapRequirementType,
    *,
    required_sources: int = 2,
    independent_sources: int = 1,
    coverage: float = 0.5,
):
    requirement = project_gap_requirements(
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
                        "accepted_evidence": 1,
                        "independent_sources": independent_sources,
                        "required_sources": required_sources,
                    }
                ],
            }
        ],
        claim_states=(
            {"claim": {"dimension_key": dimension_key, "unresolved": True}}
            if requirement_type is GapRequirementType.CLAIM_VERIFICATION
            else None
        ),
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )[0]
    return replace(requirement, closure_status=GapClosureStatus.OPEN)


def _aligned(requirement, *, source_count: int):
    extraction = EvidenceExtractionResult(
        extraction_id=EXTRACTION_ID,
        reader_execution_id=READER_ID,
        source_id="https://industry-report.example/report",
        question_id=requirement.question_id,
        gap_id=requirement.gap_id,
        dimension_key=requirement.dimension_key,
        status=EvidenceExtractionStatus.SUCCESS,
        candidate_evidence_count=1,
        extracted_claims=("claim",),
        evidence_items=("evidence",),
        failure_reason=None,
        routing_id=ROUTING_ID,
        alignment_id=ALIGNMENT_ID,
    )
    request = EvidenceAlignmentRequestFactory.create(
        extraction,
        requirement=requirement,
    )[0]
    return EvidenceAlignmentExecutor.execute(
        request,
        requirement=requirement,
        independent_source_count=source_count,
        claim_verified=False,
    ).alignment


def _context(requirement, alignment, *, source_count: int):
    assert alignment is not None
    return ClosureEvaluationContext.from_alignments(
        gap_id=requirement.gap_id,
        before_snapshot=ClosureSnapshot(
            coverage=requirement.current_coverage,
            evidence_count=requirement.current_evidence_count,
            independent_sources=requirement.current_independent_sources,
            verification_status=requirement.verification_status,
        ),
        evidence_alignments=(alignment,),
        independent_source_count=source_count,
    )


def test_open_to_partial_for_missing_independent_source() -> None:
    requirement = _requirement(
        "q7",
        "q7:d2",
        GapRequirementType.INDEPENDENT_SOURCE,
    )
    alignment = _aligned(requirement, source_count=1)
    execution = ClosureEvaluator.evaluate(
        _context(requirement, alignment, source_count=1),
        requirement=requirement,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert execution.result.after_status is GapClosureStatus.PARTIAL
    assert execution.result.transition_reason == "missing_independent_source"
    assert execution.requirement.state_version == requirement.state_version + 1
    assert execution.events[1].event_type is ClosureEvaluationEventType.TRANSITION


def test_open_to_closed_when_requirements_are_satisfied() -> None:
    requirement = _requirement(
        "q5",
        "q5:d1",
        GapRequirementType.INDEPENDENT_SOURCE,
    )
    alignment = _aligned(requirement, source_count=2)
    execution = ClosureEvaluator.evaluate(
        _context(requirement, alignment, source_count=2),
        requirement=requirement,
    )

    assert execution.result.closure_status is GapClosureStatus.CLOSED
    assert execution.requirement.closure_status is GapClosureStatus.CLOSED
    assert execution.result.remaining_requirements == ()


def test_partial_to_closed_after_missing_source_is_added() -> None:
    requirement = _requirement(
        "q5",
        "q5:d1",
        GapRequirementType.INDEPENDENT_SOURCE,
    )
    partial = replace(
        requirement,
        closure_status=GapClosureStatus.PARTIAL,
        state_version=requirement.state_version + 1,
    )
    alignment = _aligned(partial, source_count=2)
    execution = ClosureEvaluator.evaluate(
        _context(partial, alignment, source_count=2),
        requirement=partial,
    )

    assert execution.result.before_status is GapClosureStatus.PARTIAL
    assert execution.result.after_status is GapClosureStatus.CLOSED
    assert execution.requirement.state_version == partial.state_version + 1


def test_claim_verification_below_independence_bar_remains_partial() -> None:
    # Branch A: claim verification is judged from dimension-scoped, accepted-only
    # distinct source owners.  One owner against a bar of two stays PARTIAL and
    # reports the genuine missing-independent-source reason (not a stale boolean).
    requirement = _requirement(
        "q4",
        "q4:d1",
        GapRequirementType.CLAIM_VERIFICATION,
        required_sources=2,
        independent_sources=1,
    )
    alignment = _aligned(requirement, source_count=1)
    execution = ClosureEvaluator.evaluate(
        _context(requirement, alignment, source_count=1),
        requirement=requirement,
    )

    assert execution.result.after_status is GapClosureStatus.PARTIAL
    assert execution.result.transition_reason == "claim_verification_required"
    assert "claim_verification" in execution.result.remaining_requirements


def test_claim_verification_closes_when_dimension_independence_satisfied() -> None:
    # Branch A core: two distinct owners satisfy the bar, so the Requirement
    # closes even though the legacy verification_status column was never VERIFIED.
    requirement = _requirement(
        "q4",
        "q4:d1",
        GapRequirementType.CLAIM_VERIFICATION,
        required_sources=2,
        independent_sources=1,
    )
    alignment = _aligned(requirement, source_count=2)
    execution = ClosureEvaluator.evaluate(
        _context(requirement, alignment, source_count=2),
        requirement=requirement,
    )

    assert execution.result.after_status is GapClosureStatus.CLOSED
    assert execution.result.remaining_requirements == ()
    assert execution.result.closed_requirements == ("q4:d1",)


def test_closed_gap_cannot_reopen_and_versions_are_monotonic() -> None:
    requirement = _requirement(
        "q7",
        "q7:d2",
        GapRequirementType.INDEPENDENT_SOURCE,
    )
    alignment = _aligned(requirement, source_count=1)
    closed = replace(requirement, closure_status=GapClosureStatus.CLOSED, state_version=8)
    execution = ClosureEvaluator.evaluate(
        _context(closed, alignment, source_count=1),
        requirement=closed,
    )

    assert execution.requirement.closure_status is GapClosureStatus.CLOSED
    assert execution.requirement.state_version == 8
    assert execution.result.transition_reason == "closed_state_immutable"
    with pytest.raises(ValueError):
        closed.apply_closure_transition(
            new_status=GapClosureStatus.OPEN,
            transition_reason="stale_event",
        )


def test_context_and_requirement_gap_must_match() -> None:
    requirement = _requirement(
        "q1",
        "q1:d2",
        GapRequirementType.INDEPENDENT_SOURCE,
    )
    alignment = _aligned(requirement, source_count=1)
    context = _context(requirement, alignment, source_count=1)
    mismatched = replace(
        context,
        gap_id=UUID("00000000-0000-0000-0000-000000000050"),
    )

    with pytest.raises(ValueError, match="gaps do not match"):
        ClosureEvaluator.evaluate(mismatched, requirement=requirement)


def test_closure_events_preserve_question_and_gap_identity() -> None:
    requirement = _requirement(
        "q1",
        "q1:d2",
        GapRequirementType.INDEPENDENT_SOURCE,
    )
    alignment = _aligned(requirement, source_count=2)
    execution = ClosureEvaluator.evaluate(
        _context(requirement, alignment, source_count=2),
        requirement=requirement,
    )

    assert all(event.run_id == RUN_ID for event in execution.events)
    assert all(event.question_id == "q1" for event in execution.events)
    assert all(event.gap_id == requirement.gap_id for event in execution.events)
