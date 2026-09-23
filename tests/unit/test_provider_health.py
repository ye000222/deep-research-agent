"""Phase 12.1 provider health model unit tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.domain.provider_failure_classification import ProviderFailureType
from app.domain.provider_health import (
    FAILURE_STREAK_TO_UNAVAILABLE,
    ProviderHealthState,
    ProviderHealthTracker,
)

_T0 = datetime(2026, 9, 19, 12, 0, 0, tzinfo=UTC)


def _fail_to_unavailable(tracker: ProviderHealthTracker, provider: str) -> None:
    for index in range(FAILURE_STREAK_TO_UNAVAILABLE):
        tracker.record_failure(
            provider,
            failure_type=ProviderFailureType.TIMEOUT,
            when=_T0 + timedelta(seconds=index),
        )


def test_unknown_promotes_to_healthy_on_success() -> None:
    tracker = ProviderHealthTracker()
    assert tracker.state_of("SearXNG") is ProviderHealthState.UNKNOWN
    update = tracker.record_success("SearXNG", when=_T0)
    assert update.previous_state is ProviderHealthState.UNKNOWN
    assert update.state is ProviderHealthState.HEALTHY
    assert update.changed is True


def test_healthy_degrades_on_failure() -> None:
    tracker = ProviderHealthTracker()
    tracker.record_success("SearXNG", when=_T0)
    update = tracker.record_failure(
        "SearXNG", failure_type=ProviderFailureType.TIMEOUT, when=_T0
    )
    assert update.previous_state is ProviderHealthState.HEALTHY
    assert update.state is ProviderHealthState.DEGRADED


def test_degraded_becomes_unavailable_after_failure_streak() -> None:
    tracker = ProviderHealthTracker()
    tracker.record_success("SearXNG", when=_T0)
    _fail_to_unavailable(tracker, "SearXNG")
    assert tracker.state_of("SearXNG") is ProviderHealthState.UNAVAILABLE


def test_circuit_open_marks_unavailable_immediately() -> None:
    tracker = ProviderHealthTracker()
    tracker.record_success("SearXNG", when=_T0)
    update = tracker.record_failure(
        "SearXNG", failure_type=ProviderFailureType.CIRCUIT_OPEN, when=_T0
    )
    assert update.state is ProviderHealthState.UNAVAILABLE
    assert update.reason == "circuit_open_unavailable"


def test_unavailable_transitions_to_cooldown_after_cooling() -> None:
    tracker = ProviderHealthTracker(cooldown_seconds=15.0)
    tracker.record_success("SearXNG", when=_T0)
    tracker.record_failure(
        "SearXNG", failure_type=ProviderFailureType.CIRCUIT_OPEN, when=_T0
    )
    assert tracker.state_of("SearXNG") is ProviderHealthState.UNAVAILABLE
    # Before the cooldown window elapses nothing changes.
    assert tracker.expire_cooldowns(_T0 + timedelta(seconds=5)) == ()
    updates = tracker.expire_cooldowns(_T0 + timedelta(seconds=16))
    assert len(updates) == 1
    assert updates[0].previous_state is ProviderHealthState.UNAVAILABLE
    assert updates[0].state is ProviderHealthState.COOLDOWN


def test_counters_and_rates_are_tracked() -> None:
    tracker = ProviderHealthTracker()
    tracker.record_success("SearXNG", latency_ms=100.0, when=_T0)
    tracker.record_failure(
        "SearXNG", failure_type=ProviderFailureType.TIMEOUT, latency_ms=200.0, when=_T0
    )
    snapshot = tracker.snapshot("SearXNG")
    assert snapshot.success_count == 1
    assert snapshot.failure_count == 1
    assert snapshot.timeout_count == 1
    assert snapshot.recent_failure_rate == 0.5
    assert snapshot.average_latency == 150.0
    assert snapshot.last_success_at is not None
    assert snapshot.last_failure_at is not None


def test_tracker_payload_round_trip_preserves_state() -> None:
    tracker = ProviderHealthTracker()
    tracker.record_success("SearXNG", when=_T0)
    tracker.record_failure(
        "SearXNG", failure_type=ProviderFailureType.SERVER_ERROR, when=_T0
    )
    restored = ProviderHealthTracker.from_payload(tracker.to_payload())
    assert restored.state_of("SearXNG") is tracker.state_of("SearXNG")
    assert restored.snapshot("SearXNG").as_dict() == tracker.snapshot("SearXNG").as_dict()
