from app.domain.question_research_state import (
    QuestionResearchFacts,
    project_question_research_state,
)
from app.domain.research_budget import classify_question_risk, decide_question_borrow


def _q7_risk_state():
    return classify_question_risk(
        question_id="q7",
        priority=2,
        coverage=0.0,
        requirements=["至少两个独立来源"],
        gap_open=True,
        open_dimension_keys=["q7:d1", "q7:d2"],
    )


def test_recovery_eligible_question_can_reach_coverage_aware_borrow() -> None:
    research_state = project_question_research_state(
        QuestionResearchFacts(
            question_id="q7",
            search_queries_completed=15,
            readable_sources=7,
            evidence_extraction_started=2,
            coverage=0.0,
            gap_open=True,
            evaluation_completed=True,
        )
    )

    decision = decide_question_borrow(
        state=_q7_risk_state(),
        all_first_passes_complete=False,
        research_state=research_state,
        projected_spend=22_028,
        target_tokens=18_015,
        expected_utility=0.03,
        low_gain_streak=1,
        has_untried_query_family=True,
    )

    assert research_state.recovery_eligibility.value == "eligible"
    assert decision.allowed is True


def test_first_pass_protection_remains_for_questions_without_research_opportunity() -> None:
    research_state = project_question_research_state(
        QuestionResearchFacts(question_id="q7")
    )

    decision = decide_question_borrow(
        state=_q7_risk_state(),
        all_first_passes_complete=False,
        research_state=research_state,
        projected_spend=22_028,
        target_tokens=18_015,
        expected_utility=0.03,
        low_gain_streak=0,
    )

    assert decision.allowed is False
    assert decision.reason == "first_pass_floor_protected"


def test_completed_quality_does_not_enter_recovery_borrow() -> None:
    research_state = project_question_research_state(
        QuestionResearchFacts(
            question_id="q7",
            search_queries_completed=1,
            evaluation_completed=True,
            coverage=1.0,
            gap_open=False,
        )
    )

    decision = decide_question_borrow(
        state=_q7_risk_state(),
        all_first_passes_complete=False,
        research_state=research_state,
        projected_spend=22_028,
        target_tokens=18_015,
        expected_utility=0.03,
        low_gain_streak=0,
    )

    assert decision.allowed is False
    assert decision.reason == "recovery_ineligible"
