"""Fail-first validation for the Recovery execution lifecycle.

The current implementation proves the pure Borrow decision, but does not yet
expose a durable Recovery lifecycle recorder.  These tests intentionally keep
that gap visible without changing the Research Loop or Budget behavior.
"""

from app.domain.question_research_state import (
    QuestionResearchFacts,
    project_question_research_state,
)
from app.domain.research_budget import classify_question_risk, decide_question_borrow
from app.infrastructure.db.research_tools import ResearchToolRepository

RECOVERY_EVENT_TYPES = (
    "recovery.evaluation_started",
    "recovery.eligible",
    "recovery.borrow.allowed",
    "recovery.token_reserved",
    "recovery.started",
    "recovery.completed",
)


def _q7_state():
    return project_question_research_state(
        QuestionResearchFacts(
            question_id="q7",
            search_queries_completed=15,
            readable_sources=7,
            evidence_extraction_started=2,
            candidate_evidence=2,
            accepted_evidence=0,
            evaluation_completed=True,
            coverage=0.0,
            gap_open=True,
            gap_resolution_attempts=15,
            query_family_marker_present=False,
            has_recovery_path=True,
        )
    )


def _q7_risk():
    return classify_question_risk(
        question_id="q7",
        priority=2,
        coverage=0.0,
        requirements=["至少两个独立来源"],
        gap_open=True,
        open_dimension_keys=["q7:d1", "q7:d2"],
    )


def test_recovery_eligible_reaches_borrow_allowed() -> None:
    state = _q7_state()
    decision = decide_question_borrow(
        state=_q7_risk(),
        all_first_passes_complete=False,
        research_state=state,
        projected_spend=22_028,
        target_tokens=18_015,
        expected_utility=0.03,
        low_gain_streak=1,
        has_untried_query_family=True,
    )

    assert state.recovery_eligibility.value == "eligible"
    assert decision.allowed is True
    assert decision.reason == "high_risk_high_utility_borrow"


def test_recovery_execution_event_sink_is_available() -> None:
    """Fail first: the current repository has no Recovery event sink."""

    assert callable(getattr(ResearchToolRepository, "record_recovery_event", None))
    assert RECOVERY_EVENT_TYPES


def test_recovery_safety_guards_remain_active() -> None:
    state = _q7_state()
    risk = _q7_risk()
    low_utility = decide_question_borrow(
        state=risk,
        all_first_passes_complete=False,
        research_state=state,
        projected_spend=22_028,
        target_tokens=18_015,
        expected_utility=0.001,
        low_gain_streak=1,
        has_untried_query_family=True,
    )
    over_cap = decide_question_borrow(
        state=risk,
        all_first_passes_complete=False,
        research_state=state,
        projected_spend=40_000,
        target_tokens=18_015,
        expected_utility=0.03,
        low_gain_streak=1,
        has_untried_query_family=True,
    )

    assert low_utility.reason == "expected_utility_below_threshold"
    assert over_cap.reason == "bounded_recovery_limit"


def test_completed_quality_cannot_start_recovery() -> None:
    state = project_question_research_state(
        QuestionResearchFacts(
            question_id="q7",
            search_queries_completed=1,
            evaluation_completed=True,
            coverage=1.0,
            gap_open=False,
            has_recovery_path=False,
        )
    )

    decision = decide_question_borrow(
        state=_q7_risk(),
        all_first_passes_complete=False,
        research_state=state,
        projected_spend=22_028,
        target_tokens=18_015,
        expected_utility=0.03,
        low_gain_streak=0,
    )

    assert state.recovery_eligibility.value == "denied"
    assert decision.reason == "recovery_ineligible"
