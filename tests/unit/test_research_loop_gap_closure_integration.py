from datetime import UTC, datetime
from uuid import uuid4

from app.domain.closure_evaluation_context import ClosureEvaluationContext
from app.domain.closure_evaluator import ClosureEvaluationEventType, ClosureEvaluator
from app.domain.evidence_alignment_executor import EvidenceAlignmentExecutor
from app.domain.evidence_alignment_request import EvidenceAlignmentRequest
from app.domain.gap_closure import (
    ClosureSnapshot,
    GapClosureStatus,
    GapRequirement,
    GapRequirementType,
    VerificationStatus,
)
from app.domain.recovery_execution import RecoveryOutcome, RecoveryOutcomeType


def _requirement(*, required_sources: int = 1) -> GapRequirement:
    now = datetime.now(UTC)
    return GapRequirement(
        gap_id=uuid4(),
        run_id=uuid4(),
        question_id="q7",
        dimension_key="q7:d2",
        requirement_type=GapRequirementType.INDEPENDENT_SOURCE,
        criterion="需要独立来源",
        required_evidence_count=1,
        required_independent_sources=required_sources,
        current_evidence_count=0,
        current_independent_sources=0,
        verification_status=VerificationStatus.REQUIRED,
        closure_status=GapClosureStatus.OPEN,
        created_at=now,
        updated_at=now,
        state_version=1,
    )


def _alignment(requirement: GapRequirement, *, owner_count: int):
    request = EvidenceAlignmentRequest(
        alignment_request_id=uuid4(),
        extraction_id=uuid4(),
        reader_execution_id=uuid4(),
        source_id=str(uuid4()),
        evidence_id=uuid4(),
        run_id=requirement.run_id,
        question_id=requirement.question_id,
        gap_id=requirement.gap_id,
        dimension_key=requirement.dimension_key,
        requirement_type=requirement.requirement_type.value,
        claim_id=None,
        candidate_evidence="industrial market evidence",
    )
    return EvidenceAlignmentExecutor.execute(
        request,
        requirement=requirement,
        evidence_dimension_key=requirement.dimension_key,
        evidence_quality_passed=True,
        independent_source_count=owner_count,
    )


def test_live_alignment_and_closure_produce_canonical_events() -> None:
    requirement = _requirement()
    alignment_execution = _alignment(requirement, owner_count=1)
    assert alignment_execution.alignment is not None
    assert [event.event_type.value for event in alignment_execution.events] == [
        "evidence.alignment.started",
        "evidence.alignment.completed",
    ]

    context = ClosureEvaluationContext.from_alignments(
        gap_id=requirement.gap_id,
        before_snapshot=ClosureSnapshot(
            evidence_count=0,
            independent_sources=0,
            verification_status=VerificationStatus.REQUIRED,
        ),
        evidence_alignments=(alignment_execution.alignment,),
        independent_source_count=1,
    )
    closure = ClosureEvaluator.evaluate(context, requirement=requirement)
    assert closure.requirement.closure_status is GapClosureStatus.CLOSED
    assert [event.event_type for event in closure.events] == [
        ClosureEvaluationEventType.STARTED,
        ClosureEvaluationEventType.TRANSITION,
        ClosureEvaluationEventType.COMPLETED,
    ]
    assert [event.event_type.value for event in closure.events] == [
        "gap.closure.started",
        "gap.closure.transition",
        "gap.closure.completed",
    ]


def test_live_alignment_keeps_unresolved_requirement_partial() -> None:
    requirement = _requirement(required_sources=2)
    alignment_execution = _alignment(requirement, owner_count=1)
    assert alignment_execution.alignment is not None
    assert alignment_execution.alignment.alignment_status.value == "partial"

    context = ClosureEvaluationContext.from_alignments(
        gap_id=requirement.gap_id,
        before_snapshot=ClosureSnapshot(),
        evidence_alignments=(alignment_execution.alignment,),
        independent_source_count=1,
    )
    closure = ClosureEvaluator.evaluate(context, requirement=requirement)
    assert closure.requirement.closure_status is GapClosureStatus.PARTIAL
    assert closure.result.transition_reason == "missing_independent_source"


def test_closed_requirement_is_not_reopened_by_late_evidence() -> None:
    requirement = _requirement().apply_closure_transition(
        new_status=GapClosureStatus.CLOSED,
        transition_reason="requirement_satisfied",
    )
    assert _alignment(requirement, owner_count=2).alignment is None


def test_recovery_outcome_references_closure_result() -> None:
    run_id = uuid4()
    attempt_id = uuid4()
    outcome = RecoveryOutcome.evaluate(
        run_id=run_id,
        question_id="q7",
        plan_version=13,
        recovery_attempt_id=attempt_id,
        coverage_before=0.25,
        coverage_after=0.4,
        gap_count_before=1,
        gap_count_after=0,
        accepted_evidence_before=2,
        accepted_evidence_after=3,
        candidate_evidence_before=2,
        candidate_evidence_after=3,
        independent_sources_before=1,
        independent_sources_after=2,
        tokens_reserved=100,
        tokens_used=100,
        new_sources_count=1,
        new_independent_sources_count=1,
        closure_evaluation_ids=(attempt_id,),
        closure_evaluation_id=attempt_id,
        closed_gap_ids=("gap-1",),
        partial_gap_ids=(),
        remaining_gap_ids=(),
    )
    assert outcome.outcome_type is RecoveryOutcomeType.SUCCESSFUL
    fields = outcome.as_event_fields()
    assert fields["closure_evaluation_id"] == str(attempt_id)
    assert fields["closed_gap_ids"] == ["gap-1"]
    assert fields["partial_gap_ids"] == []
