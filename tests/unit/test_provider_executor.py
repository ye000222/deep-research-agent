"""Phase 12.2 provider executor unit tests.

Covers the J section scenarios:

1. Executor dispatch: router selection drives the concrete adapter call.
2. No double fallback: ``ProviderExecutor`` never re-invokes another
   provider inside a single execution, and the executor returns exactly
   one result per decision.
3. Health consistency: the tracker state used by the router matches the
   tracker after the executor's outcome has been folded back in.
4. Feedback isolation: normal research (no ``feedback_id``) does not
   carry feedback context through the executor result.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest
from app.domain.provider_executor import (
    ProviderExecutionResult,
    ProviderExecutionStatus,
    ProviderExecutor,
)
from app.domain.provider_failure_classification import ProviderFailureType
from app.domain.provider_health import ProviderHealthState, ProviderHealthTracker
from app.domain.provider_router import ProviderRouter, ProviderSelectionDecision

_T0 = datetime(2026, 9, 19, 12, 0, 0, tzinfo=UTC)
_PROVIDERS = ("SearXNG", "Bing")


class _RecordingAdapter:
    """A scripted provider adapter used by every dispatch test below."""

    def __init__(
        self,
        *,
        response: object = None,
        error: Exception | None = None,
    ) -> None:
        self.calls: list[tuple[str, int, Mapping[str, Any]]] = []
        self._response = response
        self._error = error

    async def __call__(
        self, query: str, limit: int, options: Mapping[str, Any]
    ) -> object:
        self.calls.append((query, limit, dict(options)))
        if self._error is not None:
            raise self._error
        return self._response


def _decision(provider: str | None, *, fallback: bool = False) -> ProviderSelectionDecision:
    return ProviderSelectionDecision(
        selected_provider=provider,
        excluded_providers=(),
        reason="unit_test_selection",
        health_state=ProviderHealthState.HEALTHY if provider else None,
        fallback_used=fallback,
        feedback_execution_id=None,
    )


def test_executor_dispatches_to_router_selected_provider() -> None:
    bing = _RecordingAdapter(response=["https://example.com/a", "https://example.com/b"])
    searxng = _RecordingAdapter(response=["https://unexpected"])
    executor = ProviderExecutor(
        calls={"SearXNG": searxng, "Bing": bing},
        candidate_providers=_PROVIDERS,
    )

    result = asyncio.run(
        executor.execute(
            _decision("Bing", fallback=True),
            query="industrial vision defect",
            limit=5,
        )
    )

    assert isinstance(result, ProviderExecutionResult)
    assert result.status is ProviderExecutionStatus.SUCCESS
    assert result.provider_name == "Bing"
    assert result.results == ("https://example.com/a", "https://example.com/b")
    assert len(bing.calls) == 1
    # The SearXNG adapter is untouched: the router selection is authoritative
    # and the executor must not silently run a second, hidden fallback.
    assert searxng.calls == []


def test_executor_marks_failed_and_propagates_error_type() -> None:
    class _TimeoutError(Exception):
        code = "SEARCH_TIMEOUT"

    failing = _RecordingAdapter(error=_TimeoutError("provider timeout"))
    executor = ProviderExecutor(
        calls={"SearXNG": failing, "Bing": _RecordingAdapter(response=[])},
        candidate_providers=_PROVIDERS,
    )

    result = asyncio.run(
        executor.execute(_decision("SearXNG"), query="q", limit=5)
    )

    assert result.status is ProviderExecutionStatus.FAILED
    assert result.error_type is ProviderFailureType.TIMEOUT
    assert result.provider_name == "SearXNG"
    assert result.results == ()


def test_executor_returns_empty_status_for_empty_success() -> None:
    empty = _RecordingAdapter(response=[])
    executor = ProviderExecutor(
        calls={"SearXNG": empty, "Bing": _RecordingAdapter(response=[])},
        candidate_providers=_PROVIDERS,
    )

    result = asyncio.run(
        executor.execute(_decision("SearXNG"), query="q", limit=5)
    )

    assert result.status is ProviderExecutionStatus.EMPTY
    assert result.error_type is ProviderFailureType.EMPTY_RESPONSE


def test_executor_returns_failed_for_stop_decision_without_calling_any_adapter() -> None:
    a = _RecordingAdapter(response=["https://x"])
    b = _RecordingAdapter(response=["https://y"])
    executor = ProviderExecutor(
        calls={"SearXNG": a, "Bing": b},
        candidate_providers=_PROVIDERS,
    )

    result = asyncio.run(
        executor.execute(
            _decision(None),
            query="q",
            limit=5,
        )
    )

    assert result.status is ProviderExecutionStatus.FAILED
    assert result.provider_name == ""
    assert a.calls == [] and b.calls == []


def test_health_state_used_by_router_matches_tracker_after_execution() -> None:
    # A provider is UNAVAILABLE in the tracker; the router excludes it. Once
    # the router-selected provider succeeds, the tracker reflects the recovery
    # and the same tracker (not a copy) is what the next router read uses.
    tracker = ProviderHealthTracker()
    tracker.record_success("SearXNG", when=_T0)
    tracker.record_failure(
        "SearXNG", failure_type=ProviderFailureType.CIRCUIT_OPEN, when=_T0
    )
    tracker.record_success("Bing", when=_T0)
    assert tracker.state_of("SearXNG") is ProviderHealthState.UNAVAILABLE
    assert tracker.state_of("Bing") is ProviderHealthState.HEALTHY

    router = ProviderRouter(tracker, candidate_providers=_PROVIDERS)
    decision = router.select_for(is_feedback=False)
    assert decision.selected_provider == "Bing"

    success = _RecordingAdapter(response=["https://b"])
    searxng_spy = _RecordingAdapter(response=["https://a"])
    executor = ProviderExecutor(
        calls={"SearXNG": searxng_spy, "Bing": success},
        candidate_providers=_PROVIDERS,
    )
    result = asyncio.run(executor.execute(decision, query="q", limit=5))
    assert result.status is ProviderExecutionStatus.SUCCESS
    # Health consistency: the executor never touches tracker; the caller
    # (research loop) records the outcome once. The next router read uses the
    # same tracker and still reflects health correctly.
    tracker.record_success(result.provider_name, when=_T0)
    next_decision = router.select_for(is_feedback=False)
    assert next_decision.selected_provider == "Bing"
    assert tracker.state_of("Bing") is ProviderHealthState.HEALTHY


def test_feedback_execution_id_is_preserved_end_to_end() -> None:
    tracker = ProviderHealthTracker()
    tracker.record_success("SearXNG", when=_T0)
    tracker.record_success("Bing", when=_T0)
    tracker.record_failure(
        "SearXNG", failure_type=ProviderFailureType.CIRCUIT_OPEN, when=_T0
    )
    router = ProviderRouter(tracker, candidate_providers=_PROVIDERS)
    feedback_execution_id = uuid4()
    decision = router.select_for(
        is_feedback=True, feedback_execution_id=feedback_execution_id
    )
    assert decision.selected_provider == "Bing"
    assert decision.feedback_execution_id == feedback_execution_id

    bing = _RecordingAdapter(response=["https://gap.example/evidence"])
    executor = ProviderExecutor(
        calls={"SearXNG": _RecordingAdapter(response=[]), "Bing": bing},
        candidate_providers=_PROVIDERS,
    )
    result = asyncio.run(executor.execute(decision, query="gap query", limit=5))
    assert result.feedback_execution_id == feedback_execution_id
    assert result.as_dict()["feedback_execution_id"] == str(feedback_execution_id)


def test_normal_query_carries_no_feedback_context() -> None:
    tracker = ProviderHealthTracker()
    tracker.record_success("SearXNG", when=_T0)
    tracker.record_success("Bing", when=_T0)
    router = ProviderRouter(tracker, candidate_providers=_PROVIDERS)

    decision = router.select_for(is_feedback=False, feedback_execution_id=None)
    assert decision.feedback_execution_id is None

    adapter = _RecordingAdapter(response=["https://x"])
    executor = ProviderExecutor(
        calls={"SearXNG": adapter, "Bing": _RecordingAdapter(response=[])},
        candidate_providers=_PROVIDERS,
    )
    result = asyncio.run(executor.execute(decision, query="q", limit=5))
    assert result.feedback_execution_id is None
    assert result.as_dict()["feedback_execution_id"] is None


@pytest.mark.parametrize("status", list(ProviderExecutionStatus))
def test_status_enum_matches_spec(status: ProviderExecutionStatus) -> None:
    assert status.value in {"success", "failed", "empty"}
