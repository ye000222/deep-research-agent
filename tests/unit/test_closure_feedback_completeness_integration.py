"""Phase 13.3 integration tests: Closure Feedback Completeness Pass.

These tests prove the generic loop activation promised by Phase 13.3:

    GapRequirement -> Completeness Pass -> ClosureFeedback
    -> ClosureFeedbackDispatcher -> ResearchNeed -> SuggestedResearchAction
    -> QueryIntent

No case depends on a specific question id, gap id, dimension key or benchmark
fixture; every expectation is derived from generic requirement state only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from app.domain.closure_feedback import ClosureFeedbackReason
from app.domain.closure_feedback_completeness import feedback_completeness_violations
from app.domain.closure_feedback_completeness_pass import (
    FeedbackCompletenessPassResult,
    FeedbackCompletenessPassStatus,
    run_closure_feedback_completeness_pass,
)
from app.domain.closure_feedback_dispatcher import (
    ClosureFeedbackDispatcher,
    FeedbackDispatchStatus,
)
from app.domain.gap_closure import (
    GapClosureStatus,
    GapRequirement,
    GapRequirementType,
    VerificationStatus,
)
from app.domain.research_action import SuggestedResearchActionType
from app.domain.research_need import ResearchNeedType
from app.domain.research_query_intent import ResearchQueryIntentGenerator

RUN_ID = UUID("00000000-0000-0000-0000-000000000abc")
NOW = datetime(2026, 3, 1, tzinfo=UTC)


def _requirement(
    *,
    status: GapClosureStatus = GapClosureStatus.OPEN,
    question_id: str = "qA",
    dimension_key: str = "qA:d1",
    gap_id: UUID | None = None,
) -> GapRequirement:
    return GapRequirement(
        gap_id=gap_id or uuid4(),
        run_id=RUN_ID,
        question_id=question_id,
        dimension_key=dimension_key,
        requirement_type=GapRequirementType.INDEPENDENT_SOURCE,
        criterion="需要独立来源",
        required_evidence_count=1,
        required_independent_sources=2,
        current_evidence_count=1,
        current_independent_sources=1,
        verification_status=VerificationStatus.VERIFIED,
        closure_status=status,
        created_at=NOW,
        updated_at=NOW,
        state_version=1,
    )


# --- Case 1: OPEN gap without feedback -> pass -> feedback -> need ------------

def test_open_gap_without_feedback_reaches_research_need_via_pass() -> None:
    requirement = _requirement(status=GapClosureStatus.OPEN)
    pass_result = run_closure_feedback_completeness_pass(
        (requirement,), {}, now=NOW
    )
    assert pass_result.status is FeedbackCompletenessPassStatus.GENERATED
    assert len(pass_result.generated_feedbacks) == 1
    assert pass_result.violations == (requirement.gap_id,)

    feedback = pass_result.generated_feedbacks[0]
    dispatch = ClosureFeedbackDispatcher().dispatch(feedback, requirement, now=NOW)
    assert dispatch.dispatch_status is FeedbackDispatchStatus.CREATED
    assert len(dispatch.research_needs) == 1
    assert dispatch.research_needs[0].need_type is ResearchNeedType.INDEPENDENT_SOURCE


# --- Case 2: PARTIAL gap without feedback -> feedback generated ---------------

def test_partial_gap_without_feedback_generates_completeness_feedback() -> None:
    requirement = _requirement(status=GapClosureStatus.PARTIAL)
    pass_result = run_closure_feedback_completeness_pass(
        (requirement,), {}, now=NOW
    )
    assert pass_result.status is FeedbackCompletenessPassStatus.GENERATED
    feedback = pass_result.generated_feedbacks[0]
    assert feedback.failure_reason is ClosureFeedbackReason.MISSING_INDEPENDENT_SOURCE
    assert feedback.previous_status is GapClosureStatus.PARTIAL


# --- Case 3: CLOSED gap -> no feedback -----------------------------------------

def test_closed_gap_is_skipped_and_generates_no_feedback() -> None:
    requirement = _requirement(status=GapClosureStatus.CLOSED)
    pass_result = run_closure_feedback_completeness_pass(
        (requirement,), {}, now=NOW
    )
    assert pass_result.generated_feedbacks == ()
    assert pass_result.skipped_closed == (requirement.gap_id,)
    assert pass_result.violations == ()
    verdict = pass_result.verdicts[0]
    assert verdict.status is FeedbackCompletenessPassStatus.SKIPPED
    assert verdict.reason == "closed_gap"
    assert verdict.feedback_id is None


# --- Case 4: gap with existing feedback is never fed twice ---------------------

def test_existing_feedback_is_skipped_and_never_duplicated() -> None:
    requirement = _requirement()
    first = run_closure_feedback_completeness_pass(
        (requirement,), {}, now=NOW
    )
    assert first.generated_feedbacks

    feedback = first.generated_feedbacks[0]
    existing = {str(requirement.gap_id): str(feedback.feedback_id)}
    second = run_closure_feedback_completeness_pass(
        (requirement,), existing, now=NOW
    )
    assert second.generated_feedbacks == ()
    assert second.skipped_existing == (requirement.gap_id,)
    assert second.violations == ()
    assert second.status is FeedbackCompletenessPassStatus.COMPLETE
    assert second.verdicts[0].feedback_id == feedback.feedback_id

    # The dispatcher keeps its own idempotency guard for the same feedback id.
    dedup = ClosureFeedbackDispatcher(
        already_dispatched_feedback_ids=frozenset({str(feedback.feedback_id)})
    )
    duplicate = dedup.dispatch(feedback, requirement, now=NOW)
    assert duplicate.dispatch_status is FeedbackDispatchStatus.DUPLICATED


# --- Case 5: identity never changes the outcome --------------------------------

@pytest.mark.parametrize(
    ("question_id", "dimension_key"),
    [("q1", "q1:d1"), ("q4", "q4:d2"), ("alpha", "other:dim")],
)
def test_pass_reason_is_identity_independent(
    question_id: str, dimension_key: str
) -> None:
    requirement = _requirement(
        question_id=question_id, dimension_key=dimension_key, gap_id=uuid4()
    )
    pass_result = run_closure_feedback_completeness_pass(
        (requirement,), {}, now=NOW
    )
    feedback = pass_result.generated_feedbacks[0]
    # Same generic state always yields the same reason, regardless of identity.
    assert feedback.failure_reason is ClosureFeedbackReason.MISSING_INDEPENDENT_SOURCE
    assert pass_result.verdicts[0].reason == "completeness_feedback_generated"


# --- Case 6: full closed loop through the shared dispatcher --------------------

def test_full_loop_pass_feedback_dispatcher_need_action_intent() -> None:
    requirement = _requirement()
    pass_result: FeedbackCompletenessPassResult = (
        run_closure_feedback_completeness_pass((requirement,), {}, now=NOW)
    )
    feedback = pass_result.generated_feedbacks[0]
    dispatch = ClosureFeedbackDispatcher().dispatch(feedback, requirement, now=NOW)
    assert dispatch.created

    need = dispatch.research_needs[0]
    action = dispatch.suggested_actions[0]
    assert need.feedback_id == feedback.feedback_id
    assert action.source_feedback_id == feedback.feedback_id
    assert action.action_type is SuggestedResearchActionType.FIND_INDEPENDENT_SOURCE

    intent = ResearchQueryIntentGenerator.generate(
        action, requirement=requirement, now=NOW
    )[0]
    assert intent.gap_id == requirement.gap_id

    # Invariant: after the pass, the question no longer violates completeness.
    assert (
        feedback_completeness_violations(
            (requirement,), pass_result.generated_feedbacks
        )
        == ()
    )


# --- Event payload shapes -------------------------------------------------------

def test_pass_result_event_metrics_expose_required_counts() -> None:
    open_gap = _requirement()
    covered_gap = _requirement()
    closed_gap = _requirement(status=GapClosureStatus.CLOSED)
    pass_result = run_closure_feedback_completeness_pass(
        (open_gap, covered_gap, closed_gap),
        {str(covered_gap.gap_id): str(uuid4())},
        now=NOW,
    )
    metrics = pass_result.as_event_metrics()
    assert metrics == {
        "pass_status": "generated",
        "generated_count": 1,
        "skipped_count": 2,
        "violation_count": 1,
    }


def test_verdict_event_refs_expose_required_fields() -> None:
    requirement = _requirement()
    pass_result = run_closure_feedback_completeness_pass(
        (requirement,), {}, now=NOW
    )
    refs = pass_result.verdicts[0].as_event_refs(
        run_id=RUN_ID, question_id=requirement.question_id
    )
    assert refs["run_id"] == str(RUN_ID)
    assert refs["question_id"] == requirement.question_id
    assert refs["gap_id"] == str(requirement.gap_id)
    assert refs["feedback_id"] == str(
        pass_result.generated_feedbacks[0].feedback_id
    )
