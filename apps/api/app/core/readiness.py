from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from enum import StrEnum
from time import perf_counter

from pydantic import BaseModel, Field

Probe = Callable[[], Awaitable[None] | None]


class ProbeStatus(StrEnum):
    READY = "ready"
    NOT_READY = "not_ready"


class ProbeResult(BaseModel):
    name: str
    status: ProbeStatus
    latency_ms: int = Field(ge=0)
    detail: str | None = None


class ReadinessReport(BaseModel):
    status: ProbeStatus
    checks: list[ProbeResult]


class ReadinessRegistry:
    def __init__(self) -> None:
        self._probes: dict[str, Probe] = {}

    def register(self, name: str, probe: Probe) -> None:
        if name in self._probes:
            raise ValueError(f"readiness probe already registered: {name}")
        self._probes[name] = probe

    async def evaluate(self) -> ReadinessReport:
        results: list[ProbeResult] = []
        for name, probe in self._probes.items():
            started = perf_counter()
            try:
                outcome = probe()
                if inspect.isawaitable(outcome):
                    await outcome
                status = ProbeStatus.READY
                detail = None
            except Exception as exc:  # readiness must aggregate all failed dependencies
                status = ProbeStatus.NOT_READY
                detail = _safe_error(exc)
            latency_ms = max(0, round((perf_counter() - started) * 1000))
            results.append(
                ProbeResult(name=name, status=status, latency_ms=latency_ms, detail=detail)
            )

        overall = (
            ProbeStatus.READY
            if all(result.status is ProbeStatus.READY for result in results)
            else ProbeStatus.NOT_READY
        )
        return ReadinessReport(status=overall, checks=results)


def _safe_error(exc: Exception) -> str:
    """Expose an error category without leaking connection strings or credentials."""

    return type(exc).__name__
