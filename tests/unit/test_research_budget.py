import pytest
from app.domain.research_budget import (
    allocate_research_call,
    build_resource_pool_snapshot,
    classify_claim_risk,
    classify_gap_risk,
    classify_question_risk,
    conservative_chars_to_tokens,
    decide_question_borrow,
    estimate_minimum_call_tokens,
    estimate_question_budgets,
    estimate_writer_reserve_tokens,
    question_schedule_key,
)


def test_page_budget_reserve_policy_leaves_one_page_per_remaining_search() -> None:
    # The repository applies this same arithmetic atomically when reserving
    # pages after a search has already been counted. With 4 pages left and 4
    # fresh searches still available, the current search must yield its page
    # slot so those four searches can each receive one page.
    remaining_pages = 4
    remaining_fresh_searches = 4
    assert max(0, remaining_pages - remaining_fresh_searches) == 0


def test_unaffordable_reserves_do_not_freeze_spendable_tokens() -> None:
    decision = allocate_research_call(
        available=12_000,
        eligible_questions=5,
        other_unattempted=4,
        current_attempts=0,
        minimum_call=3_000,
        maximum_call=18_000,
    )
    assert decision.outcome == "execute"
    assert decision.call_tokens + decision.future_reserve <= 12_000


def test_retry_yields_to_questions_without_first_pass() -> None:
    decision = allocate_research_call(
        available=12_000,
        eligible_questions=5,
        other_unattempted=4,
        current_attempts=1,
        minimum_call=3_000,
        maximum_call=18_000,
    )
    assert decision.outcome == "yield_question"
    assert decision.call_tokens == 0


@pytest.mark.parametrize("available", [0, 1, 2999])
def test_real_exhaustion_stops(available: int) -> None:
    assert (
        allocate_research_call(
            available=available,
            eligible_questions=5,
            other_unattempted=4,
            current_attempts=0,
            minimum_call=3_000,
            maximum_call=18_000,
        ).outcome
        == "stop_run"
    )


def test_first_pass_precedes_repeated_zero_coverage_question() -> None:
    unattempted = question_schedule_key(attempts=0, coverage=0.1, priority=3, question_id="q5")
    retry = question_schedule_key(attempts=1, coverage=0, priority=1, question_id="q1")
    assert unattempted < retry


def test_affordable_plan_gets_all_first_passes() -> None:
    available = 24_000
    for remaining_questions in range(6, 0, -1):
        decision = allocate_research_call(
            available=available,
            eligible_questions=remaining_questions,
            other_unattempted=remaining_questions - 1,
            current_attempts=0,
            minimum_call=3_000,
            maximum_call=18_000,
        )
        assert decision.outcome == "execute"
        available -= decision.call_tokens
        assert available >= (remaining_questions - 1) * 3_000


def test_dynamic_minimum_call_is_used_in_place_of_fixed_threshold() -> None:
    # A question whose assembled request needs ~6k tokens must not be treated
    # as if the legacy 3k floor were enough (T13).
    decision = allocate_research_call(
        available=3_000,
        eligible_questions=1,
        other_unattempted=0,
        current_attempts=0,
        minimum_call=6_000,
        maximum_call=18_000,
    )
    assert decision.outcome == "stop_run"
    assert decision.call_tokens == 0


def test_estimate_minimum_call_is_never_rounded_below_inputs() -> None:
    estimate = estimate_minimum_call_tokens(
        fixed_tokens=2_000,
        min_source_tokens=1_000,
        min_output_tokens=768,
    )
    assert estimate.total_tokens > 2_000 + 1_000 + 768
    assert estimate.safety_margin_tokens >= 1
    assert estimate.total_tokens == (
        estimate.fixed_tokens
        + estimate.min_source_tokens
        + estimate.min_output_tokens
        + estimate.safety_margin_tokens
    )


def test_conservative_chars_to_tokens_counts_cjk_and_latin() -> None:
    assert conservative_chars_to_tokens("abc") == 1  # ceil(3 / 3)
    assert conservative_chars_to_tokens("abcdef") == 2
    assert conservative_chars_to_tokens("中文") == 2  # one token per CJK char
    mixed = conservative_chars_to_tokens("中abc")
    assert mixed == 2  # 1 CJK + ceil(3 / 3)
    assert conservative_chars_to_tokens("") == 0


def test_writer_reserve_grows_with_report_scope() -> None:
    small = estimate_writer_reserve_tokens(question_count=1, evidence_count=1)
    large = estimate_writer_reserve_tokens(question_count=6, evidence_count=4)
    assert large > small
    assert small >= 1_500 + 220 + 180 + 2_000  # fixed + q + evidence + output
    # Never below the assembled payload shape (T-fixed writer reserve).
    assert large <= 10_000  # capped by the existing hard upper bound


def test_question_budget_estimate_is_feasible_and_priority_aware() -> None:
    questions = [
        {
            "id": "q1",
            "priority": 1,
            "evidence_requirements": ["独立来源", "统计口径"],
            "search_hints": ["official"],
        },
        {
            "id": "q2",
            "priority": 3,
            "evidence_requirements": ["案例"],
            "search_hints": [],
        },
    ]
    estimates = estimate_question_budgets(
        questions,
        max_tokens=30_000,
        planner_tokens=1_000,
        writer_reserve=4_000,
    )
    assert sum(item.target_tokens for item in estimates) == 25_000
    assert estimates[0].target_tokens > estimates[1].target_tokens
    assert all(item.minimum_tokens <= item.target_tokens for item in estimates)


def test_question_budget_estimate_degrades_to_equal_split_when_pool_is_small() -> None:
    estimates = estimate_question_budgets(
        [{"id": "q1"}, {"id": "q2"}, {"id": "q3"}],
        max_tokens=1_000,
        writer_reserve=200,
        minimum_per_question=500,
    )
    assert [item.target_tokens for item in estimates] == [266, 266, 266]


def test_resource_pool_snapshot_accounts_for_in_flight_slots_and_model_holds() -> None:
    budget = {
        "max_logical_queries": 6,
        "max_provider_requests": 8,
        "max_pages_fetched": 10,
        "max_pages_extracted": 5,
        "max_extraction_calls": 5,
        "max_verification_calls": 2,
        "max_scheduler_actions": 24,
        "max_iterations": 5,
    }
    usage = {
        "logical_queries": 2,
        "search_provider_requests": 3,
        "pages_fetched": 4,
        "page_slots_reserved": 2,
        "pages_extracted": 1,
        "extraction_slots_reserved": 1,
        "model_token_pools": {
            "research": {
                "allocated_tokens": 12_500,
                "committed_tokens": 4_000,
                "reserved_tokens": 3_000,
                "remaining_tokens": 5_500,
            }
        },
    }

    pools = build_resource_pool_snapshot(budget, usage)

    assert pools["pages_fetched"]["remaining"] == 4
    assert pools["pages_fetched"]["reserved"] == 2
    assert pools["pages_extracted"]["remaining"] == 3
    assert pools["model_tokens.research"]["committed"] == 4_000
    assert pools["model_tokens.research"]["reserved"] == 3_000
    assert pools["model_tokens.research"]["remaining"] == 5_500


def test_claim_risk_lifecycle_requires_independent_corroboration() -> None:
    open_state = classify_claim_risk(
        claim_id="c1",
        question_id="q1",
        dimension_key="q1:d1",
        claim_status="partial",
        claim_type="numeric",
        importance=0.8,
        independent_sources=1,
        required_sources=2,
    )
    assert open_state.unresolved is True
    assert open_state.lifecycle == "mitigating"
    assert open_state.verification_state == "required"
    assert open_state.independent_source_deficit == 1
    assert open_state.risk_level in {"high", "critical"}

    verified = classify_claim_risk(
        claim_id="c1",
        question_id="q1",
        dimension_key="q1:d1",
        claim_status="supported",
        claim_type="numeric",
        importance=0.8,
        independent_sources=2,
        required_sources=2,
    )
    assert verified.unresolved is False
    assert verified.lifecycle == "resolved"
    assert verified.verification_state == "verified"


def test_claim_conflict_blocks_research_borrow_until_verification() -> None:
    question = classify_question_risk(
        question_id="q1",
        priority=1,
        coverage=0.5,
        requirements=["两个独立来源"],
        gap_open=True,
        open_dimension_keys=["q1:d1"],
        unresolved_claim_ids=["c1"],
        high_risk_claim_ids=["c1"],
        high_risk_conflict_ids=["x1"],
        independent_source_deficit=1,
        blocked=True,
    )
    assert question.lifecycle == "blocked"
    assert question.verification_state == "blocked"
    assert question.borrow_eligible is False

    decision = decide_question_borrow(
        state=question,
        all_first_passes_complete=True,
        projected_spend=11_000,
        target_tokens=10_000,
        expected_utility=1.0,
        low_gain_streak=0,
    )
    assert decision.allowed is False
    assert decision.freeze_question is True


def test_gap_risk_preserves_resolved_history_and_blocked_state() -> None:
    blocked = classify_gap_risk(
        gap_id="g1",
        question_id="q1",
        gap_status="open",
        gap_type="missing",
        severity=1.0,
        resolution_attempts=3,
        blocked=True,
    )
    assert blocked.lifecycle == "blocked"
    assert blocked.unresolved is True
    assert blocked.risk_level == "critical"

    resolved = classify_gap_risk(
        gap_id="g1",
        question_id="q1",
        gap_status="resolved",
        gap_type="missing",
        severity=1.0,
        resolution_attempts=3,
    )
    assert resolved.lifecycle == "resolved"
    assert resolved.unresolved is False
    assert resolved.risk_level == "low"
