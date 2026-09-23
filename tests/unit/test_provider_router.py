"""Phase 12.1 provider routing decision unit tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from app.domain.provider_failure_classification import ProviderFailureType
from app.domain.provider_health import ProviderHealthState, ProviderHealthTracker
from app.domain.provider_router import ProviderRouter
from app.domain.query_execution import QueryExecutionRequest

_T0 = datetime(2026, 9, 19, 12, 0, 0, tzinfo=UTC)
_PROVIDERS = ("SearXNG", "Bing")


def _request(
    *,
    feedback_id: UUID | None = None,
    feedback_execution_id: UUID | None = None,
) -> QueryExecutionRequest:
    return QueryExecutionRequest(
        execution_id=uuid4(),
        run_id=uuid4(),
        question_id="q6",
        gap_id=uuid4(),
        dimension_key="d1",
        candidate_id=uuid4(),
        ranking_score=0.8,
        query_text="industrial vision defect detection vendor",
        source_constraints=(),
        requirement_type="claim_verification",
        execution_reason="unit_test",
        feedback_id=feedback_id,
        feedback_execution_id=feedback_execution_id,
    )


def _healthy(tracker: ProviderHealthTracker, provider: str) -> None:
    tracker.record_success(provider, when=_T0)


def _unavailable(tracker: ProviderHealthTracker, provider: str) -> None:
    tracker.record_failure(
        provider, failure_type=ProviderFailureType.CIRCUIT_OPEN, when=_T0
    )


def test_case1_unavailable_primary_selects_healthy_fallback() -> None:
    tracker = ProviderHealthTracker()
    _unavailable(tracker, "SearXNG")
    _healthy(tracker, "Bing")
    router = ProviderRouter(tracker, candidate_providers=_PROVIDERS)

    decision = router.select(_request())

    assert decision.selected_provider == "Bing"
    assert "SearXNG" in decision.excluded_providers
    assert decision.health_state is ProviderHealthState.HEALTHY
    assert decision.fallback_used is True


def test_case2_all_unavailable_stops() -> None:
    tracker = ProviderHealthTracker()
    _unavailable(tracker, "SearXNG")
    _unavailable(tracker, "Bing")
    router = ProviderRouter(tracker, candidate_providers=_PROVIDERS)

    decision = router.select(_request())

    assert decision.selected_provider is None
    assert decision.reason == "all_providers_unavailable"
    assert set(decision.excluded_providers) == set(_PROVIDERS)


def test_case3_feedback_query_prefers_healthy_provider() -> None:
    tracker = ProviderHealthTracker()
    # Primary is degraded; a healthy secondary exists.
    tracker.record_success("SearXNG", when=_T0)
    tracker.record_failure(
        "SearXNG", failure_type=ProviderFailureType.TIMEOUT, when=_T0
    )
    _healthy(tracker, "Bing")
    router = ProviderRouter(tracker, candidate_providers=_PROVIDERS)

    feedback_id = uuid4()
    feedback_execution_id = uuid4()
    decision = router.select(
        _request(feedback_id=feedback_id, feedback_execution_id=feedback_execution_id)
    )

    assert decision.selected_provider == "Bing"
    assert decision.reason == "feedback_query_healthy_provider_selected"
    assert decision.feedback_execution_id == feedback_execution_id


def test_feedback_execution_id_is_never_dropped() -> None:
    tracker = ProviderHealthTracker()
    _healthy(tracker, "SearXNG")
    router = ProviderRouter(tracker, candidate_providers=_PROVIDERS)

    feedback_execution_id = uuid4()
    decision = router.select(
        _request(feedback_id=uuid4(), feedback_execution_id=feedback_execution_id)
    )

    assert decision.feedback_execution_id == feedback_execution_id
    # as_dict keeps it as a stable string so downstream events can carry it.
    assert decision.as_dict()["feedback_execution_id"] == str(feedback_execution_id)


def test_failure_recovery_excludes_failed_provider_then_succeeds() -> None:
    tracker = ProviderHealthTracker()
    _healthy(tracker, "SearXNG")
    _healthy(tracker, "Bing")
    router = ProviderRouter(tracker, candidate_providers=_PROVIDERS)
    request = _request()

    # Provider A fails hard and its health is updated.
    tracker.record_failure(
        "SearXNG", failure_type=ProviderFailureType.CIRCUIT_OPEN, when=_T0
    )
    # Router excludes A and selects B for the next attempt.
    decision = router.select(request, already_excluded=("SearXNG",))
    assert decision.selected_provider == "Bing"
    assert decision.fallback_used is True

    # Provider B succeeds and its health reflects recovery.
    tracker.record_success("Bing", when=_T0 + timedelta(seconds=1))
    assert tracker.state_of("Bing") is ProviderHealthState.HEALTHY


def test_switch_budget_is_bounded() -> None:
    tracker = ProviderHealthTracker()
    _unavailable(tracker, "SearXNG")
    _healthy(tracker, "Bing")
    router = ProviderRouter(
        tracker, candidate_providers=_PROVIDERS, max_provider_switches=1
    )

    decision = router.select(_request(), provider_switches_used=1)

    assert decision.selected_provider is None
    assert decision.reason == "provider_switch_budget_exhausted"


def test_normal_research_does_not_use_feedback_routing() -> None:
    tracker = ProviderHealthTracker()
    _healthy(tracker, "SearXNG")
    _healthy(tracker, "Bing")
    router = ProviderRouter(tracker, candidate_providers=_PROVIDERS)

    decision = router.select(_request(feedback_id=None))

    # No feedback id -> primary healthy provider kept, standard reason.
    assert decision.selected_provider == "SearXNG"
    assert decision.fallback_used is False
    assert decision.reason == "healthy_provider_selected"
    assert decision.feedback_execution_id is None
