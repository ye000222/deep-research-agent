"""Health-aware provider execution adapter.

Phase 12.2 promotes the :class:`ProviderSelectionDecision` produced by the
routing layer from an *observability annotation* into a real execution input:
the :class:`ProviderExecutor` dispatches one query attempt to exactly the
provider named by the router. The internal ``web_search`` fallback chain is
bypassed whenever the router owns the selection, so a run cannot silently
switch provider twice for the same query.

The module is a pure domain layer: it never performs network calls itself. The
caller supplies an :class:`ProviderCall` implementation per candidate provider
(for example one wrapping :class:`SearXNGSearchProvider.search` and another
wrapping ``bing_only_search``) and the executor resolves, invokes, times, and
normalizes the outcome into a :class:`ProviderExecutionResult`.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from app.domain.provider_failure_classification import (
    ProviderFailureType,
    classify_provider_failure,
)
from app.domain.provider_health import ProviderHealthState
from app.domain.provider_router import ProviderSelectionDecision

# Signature every provider adapter exposes to the executor. Implementations
# must raise (any exception) on transport failure and return a Mapping or an
# iterable of URLs on success; empty iterable is treated as ``EMPTY``.
ProviderCall = Callable[[str, int, Mapping[str, Any]], Awaitable[Any]]


class ProviderExecutionStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    EMPTY = "empty"


@dataclass(frozen=True, slots=True)
class ProviderExecutionResult:
    """Normalized, provider-neutral outcome of one router-selected attempt."""

    provider_name: str
    status: ProviderExecutionStatus
    results: tuple[str, ...]
    error_type: ProviderFailureType | None
    latency_ms: float
    execution_id: UUID
    # Preserved for downstream event wiring so a gap-closure feedback attempt
    # is never orphaned when the executor is the one that ultimately fails.
    feedback_execution_id: UUID | None = None
    reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is ProviderExecutionStatus.SUCCESS

    def as_dict(self) -> dict[str, object]:
        return {
            "provider_name": self.provider_name,
            "status": self.status.value,
            "result_count": len(self.results),
            "results": list(self.results),
            "error_type": self.error_type.value if self.error_type else None,
            "latency_ms": round(self.latency_ms, 2),
            "execution_id": str(self.execution_id),
            "feedback_execution_id": (
                str(self.feedback_execution_id)
                if self.feedback_execution_id
                else None
            ),
            "reason": self.reason,
        }


class ProviderExecutor:
    """Dispatch a query attempt to the provider named by a routing decision.

    The executor is intentionally small: it validates that the chosen provider
    has a registered call, invokes it under a wall-clock timer, classifies any
    raised exception through the Phase 12.0 failure taxonomy, and returns a
    neutral result. It never retries, never switches provider, and never
    consults health — those concerns stay in the tracker / router / outer
    research loop so the switch budget remains bounded.
    """

    def __init__(
        self,
        *,
        calls: Mapping[str, ProviderCall],
        candidate_providers: tuple[str, ...] | list[str],
    ) -> None:
        if not candidate_providers:
            raise ValueError("provider executor requires at least one candidate")
        self._calls: dict[str, ProviderCall] = dict(calls)
        self._candidates = tuple(dict.fromkeys(candidate_providers))

    @property
    def candidate_providers(self) -> tuple[str, ...]:
        return self._candidates

    def is_registered(self, provider_name: str) -> bool:
        return provider_name in self._calls

    async def execute(
        self,
        decision: ProviderSelectionDecision,
        *,
        query: str,
        limit: int = 8,
        options: Mapping[str, Any] | None = None,
    ) -> ProviderExecutionResult:
        execution_id = uuid4()
        provider = decision.selected_provider
        if provider is None:
            return ProviderExecutionResult(
                provider_name="",
                status=ProviderExecutionStatus.FAILED,
                results=(),
                error_type=None,
                latency_ms=0.0,
                execution_id=execution_id,
                feedback_execution_id=decision.feedback_execution_id,
                reason=decision.reason,
            )
        if provider not in self._candidates:
            return ProviderExecutionResult(
                provider_name=provider,
                status=ProviderExecutionStatus.FAILED,
                results=(),
                error_type=ProviderFailureType.UNKNOWN,
                latency_ms=0.0,
                execution_id=execution_id,
                feedback_execution_id=decision.feedback_execution_id,
                reason="provider_not_registered",
            )
        call = self._calls.get(provider)
        if call is None:
            return ProviderExecutionResult(
                provider_name=provider,
                status=ProviderExecutionStatus.FAILED,
                results=(),
                error_type=ProviderFailureType.UNKNOWN,
                latency_ms=0.0,
                execution_id=execution_id,
                feedback_execution_id=decision.feedback_execution_id,
                reason="provider_call_missing",
            )

        started = time.monotonic()
        try:
            raw = await call(query, limit, dict(options or {}))
        except Exception as exc:  # normalized below via taxonomy
            latency = (time.monotonic() - started) * 1000.0
            failure_type = _classify_exception(exc)
            return ProviderExecutionResult(
                provider_name=provider,
                status=ProviderExecutionStatus.FAILED,
                results=(),
                error_type=failure_type,
                latency_ms=latency,
                execution_id=execution_id,
                feedback_execution_id=decision.feedback_execution_id,
                reason=f"provider_failure:{failure_type.value}",
            )

        latency = (time.monotonic() - started) * 1000.0
        urls = _extract_urls(raw)
        if not urls:
            return ProviderExecutionResult(
                provider_name=provider,
                status=ProviderExecutionStatus.EMPTY,
                results=(),
                error_type=ProviderFailureType.EMPTY_RESPONSE,
                latency_ms=latency,
                execution_id=execution_id,
                feedback_execution_id=decision.feedback_execution_id,
                reason="empty_result",
            )
        return ProviderExecutionResult(
            provider_name=provider,
            status=ProviderExecutionStatus.SUCCESS,
            results=urls,
            error_type=None,
            latency_ms=latency,
            execution_id=execution_id,
            feedback_execution_id=decision.feedback_execution_id,
            reason="provider_success",
        )


def _classify_exception(exc: Exception) -> ProviderFailureType:
    """Bridge raw provider exceptions to the Phase 12.0 taxonomy."""

    details: Mapping[str, object] | None = None
    raw_details = getattr(exc, "details", None)
    if isinstance(raw_details, Mapping):
        details = raw_details
    code = str(getattr(exc, "code", "") or "") or type(exc).__name__
    return classify_provider_failure(error_code=code, details=details)


def _extract_urls(raw: object) -> tuple[str, ...]:
    """Accept either an iterable of strings or an iterable of URL-bearing rows."""

    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,) if raw else ()
    if not isinstance(raw, Iterable):
        return ()
    urls: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            url = item
        else:
            url_value = getattr(item, "url", None)
            if url_value is None and isinstance(item, Mapping):
                url_value = item.get("url")
            url = str(url_value) if url_value else ""
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return tuple(urls)


# Convenience alias kept for tests and callers that want the "healthy" state
# re-exported without importing the health module directly.
HEALTHY = ProviderHealthState.HEALTHY

__all__ = [
    "HEALTHY",
    "ProviderCall",
    "ProviderExecutionResult",
    "ProviderExecutionStatus",
    "ProviderExecutor",
]
