from app.infrastructure.db.research_runs import (
    _record_search_transport_retry,
    _reset_plan_scoped_question_budget,
    _reset_replanned_quality_state,
)


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
