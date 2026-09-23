"""Unit tests for the Phase 13.2 Closure Feedback Dispatcher.

The dispatcher is a pure, generic bridge from ClosureFeedback to the research
pipeline.  These tests cover the four dispatch states, the four required
invariants, and an end-to-end hand-off into the existing query pipeline.  No
case depends on a specific question id, gap id, dimension key or benchmark.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import pytest
from app.domain.closure_feedback import ClosureFeedback, ClosureFeedbackReason
from app.domain.closure_feedback_dispatcher import (
    ClosureFeedbackDispatcher,
    FeedbackDispatchResult,
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

RUN_ID = UUID("00000000-0000-0000-0000-000000000def")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


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


def _feedback(requirement: GapRequirement) -> ClosureFeedback:
    return ClosureFeedback(
        feedback_id=uuid5(NAMESPACE_URL, f"t-feedback:{requirement.gap_id}"),
        run_id=requirement.run_id,
        question_id=requirement.question_id,
        gap_id=requirement.gap_id,
        dimension_key=requirement.dimension_key,
        requirement_type=requirement.requirement_type.value,
        previous_status=requirement.closure_status,
        current_status=requirement.closure_status,
        closure_result=requirement.closure_status,
        failure_reason=ClosureFeedbackReason.MISSING_INDEPENDENT_SOURCE,
        missing_requirement="independent_source",
        recommended_need_type=ResearchNeedType.INDEPENDENT_SOURCE,
        created_at=NOW,
    )


# --- Case 1 / Invariant 1: OPEN gap + feedback -> ResearchNeed created ------

def test_open_gap_with_feedback_creates_research_need() -> None:
    requirement = _requirement(status=GapClosureStatus.OPEN)
    result = ClosureFeedbackDispatcher().dispatch(_feedback(requirement), requirement)
    assert result.dispatch_status is FeedbackDispatchStatus.CREATED
    assert result.research_need_id is not None
    assert len(result.research_needs) == 1
    assert result.research_needs[0].need_type is ResearchNeedType.INDEPENDENT_SOURCE


# --- Case 2 / Invariant 1: PARTIAL gap + feedback -> ResearchNeed created ----

def test_partial_gap_with_feedback_creates_research_need() -> None:
    requirement = _requirement(status=GapClosureStatus.PARTIAL)
    result = ClosureFeedbackDispatcher().dispatch(_feedback(requirement), requirement)
    assert result.dispatch_status is FeedbackDispatchStatus.CREATED
    assert result.created is True


# --- Case 3 / Invariant 2: CLOSED gap -> skipped, no research need -----------

def test_closed_gap_is_skipped_and_produces_no_need() -> None:
    requirement = _requirement(status=GapClosureStatus.CLOSED)
    result = ClosureFeedbackDispatcher().dispatch(_feedback(requirement), requirement)
    assert result.dispatch_status is FeedbackDispatchStatus.SKIPPED
    assert result.reason == "closed_gap"
    assert result.research_need_id is None
    assert result.research_needs == ()


# --- Case 4 / Invariant 3: same feedback_id dispatches exactly one need -------

def test_duplicate_feedback_id_is_not_redispatched() -> None:
    requirement = _requirement()
    feedback = _feedback(requirement)
    first = ClosureFeedbackDispatcher().dispatch(feedback, requirement)
    assert first.dispatch_status is FeedbackDispatchStatus.CREATED

    dedup = ClosureFeedbackDispatcher(
        already_dispatched_feedback_ids=frozenset({str(feedback.feedback_id)})
    )
    second = dedup.dispatch(feedback, requirement)
    assert second.dispatch_status is FeedbackDispatchStatus.DUPLICATED
    assert second.reason == "feedback_already_dispatched"
    assert second.research_needs == ()


# --- Case 5 / Invariant 4: identity never changes the outcome -----------------

@pytest.mark.parametrize(
    ("question_id", "dimension_key"),
    [("q1", "q1:d1"), ("q4", "q4:d2"), ("q7", "q7:d9"), ("zzz", "unrelated:key")],
)
def test_dispatch_is_identity_independent(
    question_id: str, dimension_key: str
) -> None:
    requirement = _requirement(
        question_id=question_id, dimension_key=dimension_key, gap_id=uuid4()
    )
    result = ClosureFeedbackDispatcher().dispatch(_feedback(requirement), requirement)
    assert result.dispatch_status is FeedbackDispatchStatus.CREATED
    assert len(result.research_needs) == 1
    assert result.research_needs[0].need_type is ResearchNeedType.INDEPENDENT_SOURCE


# --- Guard: feedback not matching the requirement is blocked ------------------

def test_feedback_for_a_different_gap_is_blocked() -> None:
    requirement = _requirement()
    foreign = _feedback(_requirement(gap_id=uuid4()))
    result = ClosureFeedbackDispatcher().dispatch(foreign, requirement)
    assert result.dispatch_status is FeedbackDispatchStatus.BLOCKED
    assert result.reason == "feedback_requirement_mismatch"


# --- Case 6: full pipeline hand-off (need -> action -> intent) ----------------

def test_dispatch_flows_through_need_action_and_intent() -> None:
    requirement = _requirement()
    feedback = _feedback(requirement)
    result: FeedbackDispatchResult = ClosureFeedbackDispatcher().dispatch(
        feedback, requirement
    )
    assert result.dispatch_status is FeedbackDispatchStatus.CREATED

    need = result.research_needs[0]
    action = result.suggested_actions[0]
    assert need.feedback_id == feedback.feedback_id
    assert action.source_feedback_id == feedback.feedback_id
    assert action.action_type is SuggestedResearchActionType.FIND_INDEPENDENT_SOURCE

    intent = ResearchQueryIntentGenerator.generate(
        action, requirement=requirement, now=NOW
    )[0]
    assert intent.gap_id == requirement.gap_id


def test_event_refs_expose_the_required_dispatch_fields() -> None:
    requirement = _requirement()
    feedback = _feedback(requirement)
    result = ClosureFeedbackDispatcher().dispatch(feedback, requirement)
    refs = result.as_event_refs()
    assert refs["feedback_id"] == str(feedback.feedback_id)
    assert refs["gap_id"] == str(requirement.gap_id)
    assert refs["research_need_id"] == str(result.research_need_id)
    assert refs["dispatch_status"] == FeedbackDispatchStatus.CREATED.value
    assert refs["reason"] == "research_need_created"
