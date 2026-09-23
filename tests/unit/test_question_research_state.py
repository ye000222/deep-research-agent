from app.domain.question_research_state import (
    FirstPassExecutionState,
    QuestionResearchFacts,
    RecoveryEligibilityState,
    ResearchOpportunityState,
    project_question_research_state,
)


def test_missing_query_family_marker_does_not_hide_a_real_research_opportunity() -> None:
    state = project_question_research_state(
        QuestionResearchFacts(
            question_id="q7",
            search_queries_started=15,
            search_queries_completed=15,
            readable_sources=7,
            evidence_extraction_started=2,
            accepted_evidence=0,
            coverage=0.0,
            gap_open=True,
            evaluation_completed=True,
            replan_count=2,
            gap_resolution_attempts=15,
            query_family_marker_present=False,
        )
    )

    assert state.first_pass_execution == FirstPassExecutionState.COMPLETED
    assert state.research_opportunity == ResearchOpportunityState.COMPLETED_WITH_GAP
    assert state.recovery_eligibility == RecoveryEligibilityState.ELIGIBLE


def test_budget_block_is_deferred_without_erasing_research_history() -> None:
    state = project_question_research_state(
        QuestionResearchFacts(
            question_id="q7",
            search_queries_completed=15,
            readable_sources=7,
            evidence_extraction_started=2,
            coverage=0.0,
            gap_open=True,
            evaluation_completed=True,
            budget_blocked=True,
            budget_blocking_reason="first_pass_floor_protected",
        )
    )

    assert state.research_opportunity == ResearchOpportunityState.COMPLETED_WITH_GAP
    assert state.recovery_eligibility == RecoveryEligibilityState.DEFERRED
    assert state.blocking_reason == "first_pass_floor_protected"


def test_completed_coverage_denies_recovery() -> None:
    state = project_question_research_state(
        QuestionResearchFacts(
            question_id="q1",
            search_queries_completed=1,
            evaluation_completed=True,
            coverage=1.0,
            gap_open=False,
        )
    )

    assert state.research_opportunity == ResearchOpportunityState.COMPLETED_QUALITY
    assert state.recovery_eligibility == RecoveryEligibilityState.DENIED
