"""Provider health state model and event-driven health tracking.

This module is a pure domain layer. It consumes normalized provider attempt
outcomes (the ``provider.attempt.completed`` / ``provider.attempt.failed``
events produced by the research tools) and maintains an per-provider health
state so that a routing layer can decide which provider to try next. It never
performs network calls, never changes search strategy, and never mutates budget,
coverage, evidence, or recovery policy.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from app.domain.provider_failure_classification import ProviderFailureType


class ProviderHealthState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    COOLDOWN = "cooldown"
    UNKNOWN = "unknown"
    # Phase 12.4 F: after repeated failed cooldown re-probes a provider is
    # disabled for the remainder of the run so routing stops burning budget on
    # a doomed probe loop. Unlike UNAVAILABLE it does not auto-rearm to COOLDOWN.
    DISABLED_FOR_RUN = "disabled_for_run"


# Bounded, tunable thresholds shared with the routing layer. A single success
# promotes an unknown provider to healthy; a single failure degrades a healthy
# provider; a bounded failure streak (or a circuit-open signal) marks a provider
# unavailable and starts a cooldown window during which only a re-probe is
# allowed.
CONSECUTIVE_SUCCESS_TO_HEALTHY = 1
FAILURE_STREAK_TO_DEGRADED = 1
FAILURE_STREAK_TO_UNAVAILABLE = 3
PROVIDER_COOLDOWN_SECONDS = 15.0
# Phase 12.4 F: consecutive failed cooldown re-probes before a provider is
# disabled for the whole run. Bounds the probe -> fail -> probe loop.
MAX_PROBE_FAILURES = 3
_RECENT_WINDOW = 10
_UNRESPONSIVE_TYPES = frozenset(
    {
        ProviderFailureType.UNRESPONSIVE,
        ProviderFailureType.EMPTY_RESPONSE,
    }
)


@dataclass(frozen=True, slots=True)
class ProviderHealthSnapshot:
    """Immutable projection of one provider's health at a point in time."""

    provider_name: str
    state: ProviderHealthState
    success_count: int
    failure_count: int
    timeout_count: int
    unresponsive_count: int
    recent_failure_rate: float
    average_latency: float
    last_success_at: datetime | None
    last_failure_at: datetime | None
    updated_at: datetime

    def as_dict(self) -> dict[str, object]:
        return {
            "provider_name": self.provider_name,
            "state": self.state.value,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "timeout_count": self.timeout_count,
            "unresponsive_count": self.unresponsive_count,
            "recent_failure_rate": round(self.recent_failure_rate, 4),
            "average_latency": round(self.average_latency, 2),
            "last_success_at": (
                self.last_success_at.isoformat() if self.last_success_at else None
            ),
            "last_failure_at": (
                self.last_failure_at.isoformat() if self.last_failure_at else None
            ),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ProviderHealthUpdate:
    """Outcome of one tracked attempt, including the state transition."""

    provider_name: str
    previous_state: ProviderHealthState
    state: ProviderHealthState
    changed: bool
    reason: str
    snapshot: ProviderHealthSnapshot

    def as_dict(self) -> dict[str, object]:
        return {
            "provider_name": self.provider_name,
            "previous_state": self.previous_state.value,
            "new_state": self.state.value,
            "changed": self.changed,
            "reason": self.reason,
            "snapshot": self.snapshot.as_dict(),
        }


@dataclass
class _ProviderRecord:
    """Mutable internal accounting; only exposed as immutable snapshots."""

    provider_name: str
    state: ProviderHealthState = ProviderHealthState.UNKNOWN
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    unresponsive_count: int = 0
    success_streak: int = 0
    failure_streak: int = 0
    probe_failure_streak: int = 0
    recent: deque[bool] = field(default_factory=lambda: deque(maxlen=_RECENT_WINDOW))
    latency_sum: float = 0.0
    latency_count: int = 0
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    cooldown_until: datetime | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def snapshot(self) -> ProviderHealthSnapshot:
        window = len(self.recent)
        recent_failures = sum(1 for value in self.recent if value)
        failure_rate = (recent_failures / window) if window else 0.0
        average_latency = (
            (self.latency_sum / self.latency_count) if self.latency_count else 0.0
        )
        return ProviderHealthSnapshot(
            provider_name=self.provider_name,
            state=self.state,
            success_count=self.success_count,
            failure_count=self.failure_count,
            timeout_count=self.timeout_count,
            unresponsive_count=self.unresponsive_count,
            recent_failure_rate=failure_rate,
            average_latency=average_latency,
            last_success_at=self.last_success_at,
            last_failure_at=self.last_failure_at,
            updated_at=self.updated_at,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "timeout_count": self.timeout_count,
            "unresponsive_count": self.unresponsive_count,
            "success_streak": self.success_streak,
            "failure_streak": self.failure_streak,
            "probe_failure_streak": self.probe_failure_streak,
            "recent": [bool(value) for value in self.recent],
            "latency_sum": self.latency_sum,
            "latency_count": self.latency_count,
            "last_success_at": (
                self.last_success_at.isoformat() if self.last_success_at else None
            ),
            "last_failure_at": (
                self.last_failure_at.isoformat() if self.last_failure_at else None
            ),
            "cooldown_until": (
                self.cooldown_until.isoformat() if self.cooldown_until else None
            ),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_payload(
        cls, provider_name: str, payload: Mapping[str, object]
    ) -> _ProviderRecord:
        record = cls(provider_name=provider_name)
        record.state = _state_from_value(payload.get("state"))
        record.success_count = _int(payload.get("success_count"))
        record.failure_count = _int(payload.get("failure_count"))
        record.timeout_count = _int(payload.get("timeout_count"))
        record.unresponsive_count = _int(payload.get("unresponsive_count"))
        record.success_streak = _int(payload.get("success_streak"))
        record.failure_streak = _int(payload.get("failure_streak"))
        record.probe_failure_streak = _int(payload.get("probe_failure_streak"))
        recent = payload.get("recent")
        if isinstance(recent, list):
            record.recent = deque(
                (bool(value) for value in recent[-_RECENT_WINDOW:]),
                maxlen=_RECENT_WINDOW,
            )
        record.latency_sum = _float(payload.get("latency_sum"))
        record.latency_count = _int(payload.get("latency_count"))
        record.last_success_at = _datetime(payload.get("last_success_at"))
        record.last_failure_at = _datetime(payload.get("last_failure_at"))
        record.cooldown_until = _datetime(payload.get("cooldown_until"))
        record.updated_at = _datetime(payload.get("updated_at")) or datetime.now(UTC)
        return record


class ProviderHealthTracker:
    """Maintain per-provider health from attempt outcomes."""

    def __init__(
        self,
        *,
        cooldown_seconds: float = PROVIDER_COOLDOWN_SECONDS,
        failure_streak_to_unavailable: int = FAILURE_STREAK_TO_UNAVAILABLE,
        max_probe_failures: int = MAX_PROBE_FAILURES,
    ) -> None:
        self._records: dict[str, _ProviderRecord] = {}
        self._cooldown_seconds = cooldown_seconds
        self._unavailable_streak = max(1, failure_streak_to_unavailable)
        self._max_probe_failures = max(1, max_probe_failures)

    def state_of(self, provider_name: str) -> ProviderHealthState:
        record = self._records.get(provider_name)
        return record.state if record is not None else ProviderHealthState.UNKNOWN

    def snapshot(self, provider_name: str) -> ProviderHealthSnapshot:
        record = self._records.get(provider_name)
        if record is None:
            now = datetime.now(UTC)
            return ProviderHealthSnapshot(
                provider_name=provider_name,
                state=ProviderHealthState.UNKNOWN,
                success_count=0,
                failure_count=0,
                timeout_count=0,
                unresponsive_count=0,
                recent_failure_rate=0.0,
                average_latency=0.0,
                last_success_at=None,
                last_failure_at=None,
                updated_at=now,
            )
        return record.snapshot()

    def snapshots(self) -> tuple[ProviderHealthSnapshot, ...]:
        return tuple(record.snapshot() for record in self._records.values())

    def record_success(
        self,
        provider_name: str,
        *,
        latency_ms: float = 0.0,
        when: datetime | None = None,
    ) -> ProviderHealthUpdate:
        now = when or datetime.now(UTC)
        record = self._records.setdefault(
            provider_name, _ProviderRecord(provider_name=provider_name)
        )
        previous = record.state
        record.success_count += 1
        record.success_streak += 1
        record.failure_streak = 0
        record.probe_failure_streak = 0
        record.recent.append(False)
        record.last_success_at = now
        record.cooldown_until = None
        record.updated_at = now
        if latency_ms > 0:
            record.latency_sum += latency_ms
            record.latency_count += 1
        reason = (
            "recovered_healthy"
            if previous is not ProviderHealthState.HEALTHY
            else "healthy"
        )
        if record.success_streak >= CONSECUTIVE_SUCCESS_TO_HEALTHY:
            record.state = ProviderHealthState.HEALTHY
        return self._update(record, previous, reason)

    def record_failure(
        self,
        provider_name: str,
        *,
        failure_type: ProviderFailureType = ProviderFailureType.UNKNOWN,
        latency_ms: float = 0.0,
        when: datetime | None = None,
    ) -> ProviderHealthUpdate:
        now = when or datetime.now(UTC)
        record = self._records.setdefault(
            provider_name, _ProviderRecord(provider_name=provider_name)
        )
        previous = record.state
        record.failure_count += 1
        record.failure_streak += 1
        record.success_streak = 0
        record.recent.append(True)
        record.last_failure_at = now
        record.updated_at = now
        if failure_type is ProviderFailureType.TIMEOUT:
            record.timeout_count += 1
        if failure_type in _UNRESPONSIVE_TYPES:
            record.unresponsive_count += 1
        if latency_ms > 0:
            record.latency_sum += latency_ms
            record.latency_count += 1

        circuit_open = failure_type is ProviderFailureType.CIRCUIT_OPEN
        if previous is ProviderHealthState.COOLDOWN:
            # The provider was re-probed and failed again. Count the probe
            # failure so a chronically broken provider is eventually disabled
            # for the whole run instead of looping UNAVAILABLE -> COOLDOWN ->
            # probe -> fail forever and burning budget on doomed probes.
            record.probe_failure_streak += 1
            if record.probe_failure_streak >= self._max_probe_failures:
                record.state = ProviderHealthState.DISABLED_FOR_RUN
                record.cooldown_until = None
                reason = "probe_failures_exceeded_disabled_for_run"
            else:
                record.state = ProviderHealthState.UNAVAILABLE
                record.cooldown_until = now + timedelta(seconds=self._cooldown_seconds)
                reason = "cooldown_probe_failed"
            return self._update(record, previous, reason)

        unavailable = circuit_open or record.failure_streak >= self._unavailable_streak
        if unavailable:
            record.state = ProviderHealthState.UNAVAILABLE
            record.cooldown_until = now + timedelta(seconds=self._cooldown_seconds)
            reason = (
                "circuit_open_unavailable"
                if circuit_open
                else "failure_streak_unavailable"
            )
        elif record.failure_streak >= FAILURE_STREAK_TO_DEGRADED:
            record.state = ProviderHealthState.DEGRADED
            reason = "failure_streak_degraded"
        else:
            reason = "failure_recorded"
        return self._update(record, previous, reason)

    def expire_cooldowns(
        self, now: datetime | None = None
    ) -> tuple[ProviderHealthUpdate, ...]:
        """Move cooled-down providers from UNAVAILABLE to COOLDOWN (re-probe)."""

        moment = now or datetime.now(UTC)
        updates: list[ProviderHealthUpdate] = []
        for record in self._records.values():
            if (
                record.state is ProviderHealthState.UNAVAILABLE
                and record.cooldown_until is not None
                and moment >= record.cooldown_until
            ):
                previous = record.state
                record.state = ProviderHealthState.COOLDOWN
                record.updated_at = moment
                updates.append(
                    self._update(record, previous, "cooldown_expired_ready_to_probe")
                )
        return tuple(updates)

    def to_payload(self) -> dict[str, dict[str, object]]:
        return {
            name: record.to_payload() for name, record in self._records.items()
        }

    @classmethod
    def from_payload(
        cls,
        payload: Iterable[tuple[str, Mapping[str, object]]] | Mapping[str, Mapping[str, object]],
        *,
        cooldown_seconds: float = PROVIDER_COOLDOWN_SECONDS,
        failure_streak_to_unavailable: int = FAILURE_STREAK_TO_UNAVAILABLE,
        max_probe_failures: int = MAX_PROBE_FAILURES,
    ) -> ProviderHealthTracker:
        tracker = cls(
            cooldown_seconds=cooldown_seconds,
            failure_streak_to_unavailable=failure_streak_to_unavailable,
            max_probe_failures=max_probe_failures,
        )
        items = payload.items() if isinstance(payload, Mapping) else payload
        for name, record_payload in items:
            if isinstance(record_payload, Mapping):
                tracker._records[name] = _ProviderRecord.from_payload(
                    name, record_payload
                )
        return tracker

    def _update(
        self, record: _ProviderRecord, previous: ProviderHealthState, reason: str
    ) -> ProviderHealthUpdate:
        return ProviderHealthUpdate(
            provider_name=record.provider_name,
            previous_state=previous,
            state=record.state,
            changed=previous is not record.state,
            reason=reason,
            snapshot=record.snapshot(),
        )


def _state_from_value(value: object) -> ProviderHealthState:
    if isinstance(value, str):
        try:
            return ProviderHealthState(value)
        except ValueError:
            return ProviderHealthState.UNKNOWN
    return ProviderHealthState.UNKNOWN


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _float(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _datetime(value: object) -> datetime | None:
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None
