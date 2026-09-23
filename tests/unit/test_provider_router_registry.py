"""Phase 12.4 D/G: router reads the registry, honors DISABLED_FOR_RUN, and
emits pool observability fields."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.domain.provider_failure_classification import ProviderFailureType
from app.domain.provider_health import ProviderHealthState, ProviderHealthTracker
from app.domain.provider_registry import SearchProviderRegistry
from app.domain.provider_router import ProviderRouter

_T0 = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)


def _mark(tracker: ProviderHealthTracker, provider: str, state: ProviderHealthState) -> None:
    """Drive a provider into a target health state via public tracker APIs."""

    if state is ProviderHealthState.HEALTHY:
        tracker.record_success(provider, when=_T0)
        return
    if state is ProviderHealthState.UNAVAILABLE:
        tracker.record_failure(
            provider, failure_type=ProviderFailureType.CIRCUIT_OPEN, when=_T0
        )
        return
    if state is ProviderHealthState.DISABLED_FOR_RUN:
        # cooldown_seconds + max_probe_failures must be tuned on the tracker;
        # a single re-probe failure then disables it for the run.
        tracker.record_failure(
            provider, failure_type=ProviderFailureType.CIRCUIT_OPEN, when=_T0
        )
        tracker.expire_cooldowns(_T0 + timedelta(seconds=16))
        tracker.record_failure(
            provider, failure_type=ProviderFailureType.TIMEOUT, when=_T0 + timedelta(seconds=16)
        )
        return
    raise ValueError(f"unsupported fixture state: {state}")


def _disabled_tracker() -> ProviderHealthTracker:
    return ProviderHealthTracker(cooldown_seconds=15.0, max_probe_failures=1)


def test_three_provider_reroute_selects_third_and_reports_rank() -> None:
    tracker = _disabled_tracker()
    # Disable SearXNG first: its expire_cooldowns() call would otherwise also
    # rearm a Bing UNAVAILABLE record to COOLDOWN and skew the eligible count.
    _mark(tracker, "SearXNG", ProviderHealthState.DISABLED_FOR_RUN)
    _mark(tracker, "Bing", ProviderHealthState.UNAVAILABLE)
    _mark(tracker, "DuckDuckGo", ProviderHealthState.HEALTHY)
    router = ProviderRouter(
        tracker, candidate_providers=("Bing", "DuckDuckGo", "SearXNG")
    )

    decision = router.select_for(is_feedback=False)

    assert decision.selected_provider == "DuckDuckGo"
    assert decision.provider_pool_size == 3
    assert decision.available_provider_count == 1
    # DuckDuckGo sits at pool index 1 -> selection rank 2.
    assert decision.selection_rank == 2
    assert set(decision.excluded_providers) == {"Bing", "SearXNG"}
    assert decision.fallback_used is True


def test_disabled_for_run_is_excluded_even_when_only_candidate() -> None:
    tracker = _disabled_tracker()
    _mark(tracker, "Bing", ProviderHealthState.HEALTHY)
    _mark(tracker, "SearXNG", ProviderHealthState.DISABLED_FOR_RUN)
    router = ProviderRouter(tracker, candidate_providers=("Bing", "SearXNG"))

    decision = router.select_for(is_feedback=False, already_excluded=("Bing",))

    assert decision.selected_provider is None
    assert decision.reason == "all_providers_unavailable"
    assert "SearXNG" in decision.excluded_providers
    assert decision.available_provider_count == 0


def test_all_blocked_reports_pool_size_and_zero_available() -> None:
    tracker = ProviderHealthTracker()
    _mark(tracker, "SearXNG", ProviderHealthState.UNAVAILABLE)
    _mark(tracker, "Bing", ProviderHealthState.UNAVAILABLE)
    router = ProviderRouter(tracker, candidate_providers=("SearXNG", "Bing"))

    decision = router.select_for(is_feedback=False)

    assert decision.selected_provider is None
    assert decision.provider_pool_size == 2
    assert decision.available_provider_count == 0
    assert decision.selection_rank == 0


def test_registry_supplies_candidate_pool_order() -> None:
    registry = SearchProviderRegistry.default(
        brave_api_key_present=False, duckduckgo_enabled=True
    )
    tracker = ProviderHealthTracker()
    router = ProviderRouter(tracker, registry=registry)

    decision = router.select_for(is_feedback=False)

    # Cold start (all UNKNOWN): the registry pool order decides, and SearXNG is
    # now the first enabled provider in the production default registry.
    assert decision.selected_provider == "SearXNG"
    assert decision.provider_pool_size == 3
    assert decision.selection_rank == 1


def test_as_dict_exposes_pool_fields() -> None:
    tracker = ProviderHealthTracker()
    _mark(tracker, "SearXNG", ProviderHealthState.HEALTHY)
    router = ProviderRouter(tracker, candidate_providers=("SearXNG", "Bing"))

    payload = router.select_for(is_feedback=False).as_dict()

    assert payload["provider_pool_size"] == 2
    # SearXNG HEALTHY and Bing UNKNOWN are both eligible (only UNAVAILABLE /
    # DISABLED_FOR_RUN are blocked), so the pool reports 2 available providers.
    assert payload["available_provider_count"] == 2
    assert payload["selection_rank"] == 1
