"""Phase 13.4 tests: Run-Level Closure Feedback Finalizer.

These tests prove the run-end explainability invariant: no Research Run may
finish while an OPEN/PARTIAL ``GapRequirement`` stays unexplained -- every such
gap either already carries ClosureFeedback, has an explicit terminal reason, or
receives a generated completeness feedback that flows through the shared
Phase 13.2 dispatcher.  No case depends on question id, gap id, dimension key
or benchmark identity.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from app.domain.closure_feedback import ClosureFeedbackReason
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
from app.domain.research_query_intent import ResearchQueryIntentGenerator
from app.domain.run_closure_feedback_finalizer import (
    GapFinalizationAction,
    RunClosureFeedbackFinalizationResult,
    RunClosureFeedbackFinalizationStatus,
    finalize_run_closure_feedback,
    is_terminal_run_stop_reason,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000404")
NOW = datetime(2026, 4, 1, tzinfo=UTC)


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


def _finalize(
    requirements: tuple[GapRequirement, ...],
    *,
    existing: dict[str, str] | None = None,
    terminal: dict[str, str] | None = None,
) -> RunClosureFeedbackFinalizationResult:
    return finalize_run_closure_feedback(
        run_id=RUN_ID,
        gap_requirements=requirements,
        existing_feedback_ids_by_gap=existing or {},
        terminal_reasons=terminal or {},
        now=NOW,
    )


# --- Case 1: OPEN gap without feedback -> feedback generated -------------------

def test_open_gap_without_feedback_gets_generated_feedback() -> None:
    gap = _requirement(status=GapClosureStatus.OPEN)
    result = _finalize((gap,))
    assert result.status is RunClosureFeedbackFinalizationStatus.GENERATED
    assert result.violations == (gap.gap_id,)
    verdict = result.verdicts[0]
    assert verdict.action is GapFinalizationAction.GENERATED_FEEDBACK
    assert verdict.feedback_exists is False
    feedback = result.generated_feedbacks[0]
    assert verdict.feedback_id == feedback.feedback_id
    assert feedback.failure_reason is ClosureFeedbackReason.MISSING_INDEPENDENT_SOURCE


# --- Case 2: PARTIAL gap without feedback -> feedback generated ----------------

def test_partial_gap_without_feedback_gets_generated_feedback() -> None:
    gap = _requirement(status=GapClosureStatus.PARTIAL)
    result = _finalize((gap,))
    assert len(result.generated_feedbacks) == 1
    feedback = result.generated_feedbacks[0]
    assert feedback.previous_status is GapClosureStatus.PARTIAL
    assert result.verdicts[0].action is GapFinalizationAction.GENERATED_FEEDBACK


# --- Case 3: CLOSED gap -> no feedback ------------------------------------------

def test_closed_gap_is_complete_and_generates_nothing() -> None:
    gap = _requirement(status=GapClosureStatus.CLOSED)
    result = _finalize((gap,))
    assert result.generated_feedbacks == ()
    assert result.violations == ()
    assert result.status is RunClosureFeedbackFinalizationStatus.COMPLETED
    verdict = result.verdicts[0]
    assert verdict.action is GapFinalizationAction.EXISTING_COMPLETE
    assert verdict.feedback_id is None


# --- Case 4: existing feedback is never duplicated ------------------------------

def test_existing_feedback_is_skipped_and_never_duplicated() -> None:
    gap = _requirement()
    first = _finalize((gap,))
    feedback = first.generated_feedbacks[0]

    existing = {str(gap.gap_id): str(feedback.feedback_id)}
    second = _finalize((gap,), existing=existing)
    assert second.generated_feedbacks == ()
    assert second.existing_feedbacks == (gap.gap_id,)
    assert second.violations == ()
    assert second.status is RunClosureFeedbackFinalizationStatus.COMPLETED
    verdict = second.verdicts[0]
    assert verdict.action is GapFinalizationAction.EXISTING_FEEDBACK
    assert verdict.feedback_exists is True
    assert verdict.feedback_id == feedback.feedback_id


# --- Case 5: explicit terminal reason explains the gap --------------------------

def test_terminal_reason_allows_gap_without_feedback() -> None:
    gap = _requirement()
    result = _finalize((gap,), terminal={str(gap.gap_id): "source_exhausted"})
    assert result.generated_feedbacks == ()
    assert result.terminal_gaps == (gap.gap_id,)
    assert result.violations == ()
    verdict = result.verdicts[0]
    assert verdict.action is GapFinalizationAction.TERMINAL
    assert verdict.terminal_reason == "source_exhausted"


# --- Case 6: full closed loop finalizer -> feedback -> dispatcher -> pipeline ---

def test_finalizer_flows_through_dispatcher_into_research_pipeline() -> None:
    gap = _requirement()
    result = _finalize((gap,))
    feedback = result.generated_feedbacks[0]
    dispatch = ClosureFeedbackDispatcher().dispatch(feedback, gap, now=NOW)
    assert dispatch.dispatch_status is FeedbackDispatchStatus.CREATED

    need = dispatch.research_needs[0]
    action = dispatch.suggested_actions[0]
    assert need.feedback_id == feedback.feedback_id
    assert action.source_feedback_id == feedback.feedback_id
    assert action.action_type is SuggestedResearchActionType.FIND_INDEPENDENT_SOURCE

    intent = ResearchQueryIntentGenerator.generate(
        action, requirement=gap, now=NOW
    )[0]
    assert intent.gap_id == gap.gap_id


# --- Case 7: identity never changes the action ----------------------------------

@pytest.mark.parametrize(
    ("question_id", "dimension_key"),
    [("q1", "q1:d1"), ("q7", "other:dim"), ("zzz", "x:y")],
)
def test_finalization_action_is_identity_independent(
    question_id: str, dimension_key: str
) -> None:
    gap = _requirement(question_id=question_id, dimension_key=dimension_key)
    result = _finalize((gap,))
    verdict = result.verdicts[0]
    # Identical generic state always yields the identical action and reason.
    assert verdict.action is GapFinalizationAction.GENERATED_FEEDBACK
    assert (
        result.generated_feedbacks[0].failure_reason
        is ClosureFeedbackReason.MISSING_INDEPENDENT_SOURCE
    )


# --- Run stop reason classification (generic string state, no identity) ---------

@pytest.mark.parametrize(
    ("reason", "terminal"),
    [
        ("token_budget_exhausted", True),
        ("source_space_exhausted", True),
        ("deadline_exhausted", True),
        ("cancelled", True),
        ("user_stopped", True),
        ("quality_met", False),
        ("stagnation", False),
        (None, False),
        ("", False),
    ],
)
def test_terminal_stop_reason_classification(reason: str | None, terminal: bool) -> None:
    assert is_terminal_run_stop_reason(reason) is terminal


# --- Aggregate invariants --------------------------------------------------------

def test_mixed_run_explains_every_open_gap() -> None:
    closed = _requirement(status=GapClosureStatus.CLOSED)
    covered = _requirement()
    terminal = _requirement()
    leftover = _requirement()
    existing = {str(covered.gap_id): str(uuid4())}
    terminal_reasons = {str(terminal.gap_id): "cancelled"}
    result = finalize_run_closure_feedback(
        run_id=RUN_ID,
        gap_requirements=(closed, covered, terminal, leftover),
        existing_feedback_ids_by_gap=existing,
        terminal_reasons=terminal_reasons,
        now=NOW,
    )
    assert result.total_gaps == 4
    assert result.violations == (leftover.gap_id,)
    assert len(result.generated_feedbacks) == 1
    actions = {verdict.gap_id: verdict.action for verdict in result.verdicts}
    assert actions == {
        closed.gap_id: GapFinalizationAction.EXISTING_COMPLETE,
        covered.gap_id: GapFinalizationAction.EXISTING_FEEDBACK,
        terminal.gap_id: GapFinalizationAction.TERMINAL,
        leftover.gap_id: GapFinalizationAction.GENERATED_FEEDBACK,
    }
    # Invariant: every non-CLOSED gap ends explained by feedback or terminal reason.
    for verdict in result.verdicts:
        if verdict.closure_status is GapClosureStatus.CLOSED:
            continue
        assert (
            verdict.feedback_id is not None
            or verdict.terminal_reason is not None
            or verdict.action is GapFinalizationAction.GENERATED_FEEDBACK
        )


def test_as_dict_exposes_run_level_summary() -> None:
    gap = _requirement()
    result = _finalize((gap,))
    payload = result.as_dict()
    assert payload["run_id"] == str(RUN_ID)
    assert payload["status"] == "generated"
    assert payload["total_gaps"] == 1
    assert payload["generated_feedback_count"] == 1
    assert payload["existing_feedback_count"] == 0
    assert payload["terminal_gap_count"] == 0
    assert payload["blocked_gap_count"] == 0
