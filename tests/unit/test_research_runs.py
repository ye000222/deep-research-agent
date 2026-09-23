from datetime import UTC, datetime, timedelta

from app.domain.research_management import allocate_budget_shares
from app.infrastructure.db.research_runs import (
    _bounded_retry_delay,
    _record_search_transport_retry,
    _reset_plan_scoped_question_budget,
    _reset_replanned_quality_state,
)
from app.services.research_runs import _BUDGETS


def test_provider_attempt_budget_leaves_room_for_follow_up_queries() -> None:
    for tier in ("quick", "standard", "deep"):
        budget = _BUDGETS[tier]
        assert int(budget["max_provider_requests"]) >= (
            int(budget["max_logical_queries"]) * 3
        )


def test_research_tier_budgets_are_monotonic_and_executable() -> None:
    ordered = [_BUDGETS[tier] for tier in ("quick", "standard", "deep")]
    for key in (
        "max_iterations",
        "max_searches",
        "max_logical_queries",
        "max_provider_requests",
        "max_pages",
        "max_pages_fetched",
        "max_pages_extracted",
        "max_extraction_calls",
        "max_scheduler_actions",
        "max_tokens",
        "max_wall_clock_seconds",
    ):
        values = [int(budget[key]) for budget in ordered]
        assert values == sorted(values), f"{key} must not shrink in a deeper tier"
    for budget in ordered:
        assert int(budget["max_iterations"]) >= int(budget["max_logical_queries"])


def test_standard_budget_preserves_v1_repair_extraction_headroom() -> None:
    budget = _BUDGETS["standard"]
    allocation = allocate_budget_shares(
        max_iterations=int(budget["max_iterations"]),
        max_searches=int(budget["max_searches"]),
        max_pages=int(budget["max_pages"]),
        max_tokens=int(budget["max_tokens"]),
        max_logical_queries=int(budget["max_logical_queries"]),
        max_provider_requests=int(budget["max_provider_requests"]),
        max_pages_fetched=int(budget["max_pages_fetched"]),
        max_pages_extracted=int(budget["max_pages_extracted"]),
        max_extraction_calls=int(budget["max_extraction_calls"]),
        max_verification_calls=int(budget["max_verification_calls"]),
        max_scheduler_actions=int(budget["max_scheduler_actions"]),
    )

    assert budget["max_pages_extracted"] == 56
    assert budget["max_extraction_calls"] == 56
    # The repair tail needs room for the untouched market question and the
    # independent-source passes after the first 40 extracts. Retain the same
    # conservative 1.5k-per-extract envelope for all 56.
    assert allocation["research_tokens"] >= 56 * 1_500


def test_replan_clears_question_budget_state_from_previous_plan() -> None:
    previous = {
        "replans": 2,
        "question_budget_exhausted_by_question": {"q1": True, "q2": True},
    }

    current = _reset_plan_scoped_question_budget(previous, reset_question_ids=("q1",))

    assert current == {
        "replans": 2,
        "question_budget_exhausted_by_question": {"q2": True},
    }
    assert previous["question_budget_exhausted_by_question"] == {"q1": True, "q2": True}


def test_replan_reopens_query_families_for_targeted_questions() -> None:
    previous = {
        "executed_query_families_by_question": {
            "q1": ["scope", "authoritative", "alternate", "contradiction"],
            "q2": ["scope"],
        }
    }
    current = _reset_plan_scoped_question_budget(
        previous,
        reset_question_ids=("q1",),
    )
    assert current["executed_query_families_by_question"] == {"q2": ["scope"]}


def test_replan_resets_only_targeted_question_state() -> None:
    usage = {
        "query_strategy_exhausted_by_question": {"q1": True, "q2": True},
    }
    quality = {
        "low_information_gain_streak": 2,
        "low_information_gain_streak_by_question": {"q1": 2, "q2": 1},
        "risk_state_by_question": {
            "q1": {
                "lifecycle": "blocked",
                "gap_open": True,
                "unresolved_high_risk": True,
                "borrow_eligible": False,
                "reasons": ["no_eligible_action"],
            },
            "q2": {"lifecycle": "mitigating", "borrow_eligible": True},
        },
        "risk_state": {"version": "claim_gap.v2", "summary": {}},
    }

    current_usage = _reset_plan_scoped_question_budget(
        usage, reset_question_ids=("q1",)
    )
    current_quality = _reset_replanned_quality_state(
        quality, reset_question_ids=("q1",)
    )

    assert current_usage["query_strategy_exhausted_by_question"] == {"q2": True}
    assert current_quality["low_information_gain_streak_by_question"] == {"q2": 1}
    q1_risk = current_quality["risk_state_by_question"]["q1"]
    assert q1_risk["lifecycle"] == "open"
    assert q1_risk["borrow_eligible"] is True
    assert "replan_reopened" in q1_risk["reasons"]
    assert current_quality["risk_state_by_question"]["q2"]["lifecycle"] == "mitigating"
    assert quality["low_information_gain_streak_by_question"] == {"q1": 2, "q2": 1}


def test_replan_preserves_proven_hard_question_budget_exhaustion() -> None:
    usage = {
        "question_budget_exhausted_by_question": {
            "q1": {"reason": "hard_question_token_limit", "recorded_plan_version": 2}
        }
    }

    current = _reset_plan_scoped_question_budget(usage, reset_question_ids=("q1",))

    assert current == usage


def test_durable_search_requeue_counts_as_technical_retry() -> None:
    original: dict[str, object] = {"technical_retries": 2, "search_transport_requeues": 1}

    updated = _record_search_transport_retry(original)

    assert updated["technical_retries"] == 3
    assert updated["search_transport_requeues"] == 1
    assert original["technical_retries"] == 2


def test_durable_retry_is_shortened_to_leave_completion_time() -> None:
    now = datetime(2026, 9, 14, tzinfo=UTC)
    budget = {"deadline_at": (now + timedelta(seconds=80)).isoformat()}

    assert _bounded_retry_delay(budget, 120, now=now) == 35


def test_durable_retry_is_rejected_when_completion_window_is_gone() -> None:
    now = datetime(2026, 9, 14, tzinfo=UTC)
    budget = {"deadline_at": (now + timedelta(seconds=49)).isoformat()}

    assert _bounded_retry_delay(budget, 30, now=now) is None


def test_durable_retry_keeps_requested_delay_with_ample_time() -> None:
    now = datetime(2026, 9, 14, tzinfo=UTC)
    budget = {"deadline_at": (now + timedelta(minutes=10)).isoformat()}

    assert _bounded_retry_delay(budget, 60, now=now) == 60
