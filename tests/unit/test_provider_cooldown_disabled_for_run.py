"""Phase 12.4 F: cooldown re-probe loop protection (DISABLED_FOR_RUN)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.domain.provider_failure_classification import ProviderFailureType
from app.domain.provider_health import ProviderHealthState, ProviderHealthTracker

_T0 = datetime(2026, 9, 20, 12, 0, 0, tzinfo=UTC)


def _to_unavailable(tracker: ProviderHealthTracker, provider: str) -> None:
    tracker.record_success(provider, when=_T0)
    tracker.record_failure(
        provider, failure_type=ProviderFailureType.CIRCUIT_OPEN, when=_T0
    )
    assert tracker.state_of(provider) is ProviderHealthState.UNAVAILABLE


def _probe_fail(
    tracker: ProviderHealthTracker, provider: str, offset_seconds: int
) -> ProviderHealthState:
    """Expire the cooldown (rearm to COOLDOWN) then fail the probe."""

    tracker.expire_cooldowns(_T0 + timedelta(seconds=offset_seconds))
    assert tracker.state_of(provider) is ProviderHealthState.COOLDOWN
    update = tracker.record_failure(
        provider,
        failure_type=ProviderFailureType.TIMEOUT,
        when=_T0 + timedelta(seconds=offset_seconds),
    )
    return update.state


def test_repeated_probe_failures_disable_provider_for_run() -> None:
    tracker = ProviderHealthTracker(cooldown_seconds=15.0, max_probe_failures=3)
    _to_unavailable(tracker, "SearXNG")

    assert _probe_fail(tracker, "SearXNG", 16) is ProviderHealthState.UNAVAILABLE
    assert _probe_fail(tracker, "SearXNG", 32) is ProviderHealthState.UNAVAILABLE
    final = _probe_fail(tracker, "SearXNG", 48)
    assert final is ProviderHealthState.DISABLED_FOR_RUN


def test_disabled_for_run_does_not_rearm_on_expiry() -> None:
    tracker = ProviderHealthTracker(cooldown_seconds=15.0, max_probe_failures=2)
    _to_unavailable(tracker, "SearXNG")
    _probe_fail(tracker, "SearXNG", 16)
    _probe_fail(tracker, "SearXNG", 32)  # reaches the (2) threshold
    assert tracker.state_of("SearXNG") is ProviderHealthState.DISABLED_FOR_RUN

    # Unlike UNAVAILABLE, a disabled provider must never flip back to COOLDOWN.
    updates = tracker.expire_cooldowns(_T0 + timedelta(seconds=600))
    assert updates == ()
    assert tracker.state_of("SearXNG") is ProviderHealthState.DISABLED_FOR_RUN


def test_success_resets_probe_streak_and_recovers() -> None:
    tracker = ProviderHealthTracker(cooldown_seconds=15.0, max_probe_failures=3)
    _to_unavailable(tracker, "SearXNG")
    _probe_fail(tracker, "SearXNG", 16)  # probe_streak 1
    _probe_fail(tracker, "SearXNG", 32)  # probe_streak 2

    # A successful re-probe recovers to HEALTHY and clears the probe streak.
    tracker.expire_cooldowns(_T0 + timedelta(seconds=48))
    update = tracker.record_success("SearXNG", when=_T0 + timedelta(seconds=48))
    assert update.state is ProviderHealthState.HEALTHY

    # Recovery resets the probe count, so the provider must again fail the
    # full max_probe_failures (3) re-probe cycle before it is disabled: 1 and
    # 2 stay UNAVAILABLE, only the 3rd disables it.
    tracker.record_failure(
        "SearXNG",
        failure_type=ProviderFailureType.CIRCUIT_OPEN,
        when=_T0 + timedelta(seconds=49),
    )
    assert _probe_fail(tracker, "SearXNG", 64) is ProviderHealthState.UNAVAILABLE
    assert _probe_fail(tracker, "SearXNG", 80) is ProviderHealthState.UNAVAILABLE
    assert _probe_fail(tracker, "SearXNG", 96) is ProviderHealthState.DISABLED_FOR_RUN


def test_disabled_for_run_round_trips_through_payload() -> None:
    tracker = ProviderHealthTracker(cooldown_seconds=15.0, max_probe_failures=2)
    _to_unavailable(tracker, "SearXNG")
    _probe_fail(tracker, "SearXNG", 16)
    _probe_fail(tracker, "SearXNG", 32)
    assert tracker.state_of("SearXNG") is ProviderHealthState.DISABLED_FOR_RUN

    restored = ProviderHealthTracker.from_payload(
        tracker.to_payload(), cooldown_seconds=15.0, max_probe_failures=2
    )
    assert restored.state_of("SearXNG") is ProviderHealthState.DISABLED_FOR_RUN
