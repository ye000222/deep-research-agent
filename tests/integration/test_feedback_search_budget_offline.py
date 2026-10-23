"""Real repository reservations and Search execution on SQLite/MockTransport only."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest
from app.domain.state import BudgetLimits, BudgetUsage, ResearchState
from app.infrastructure.db.research_tools import ResearchToolRepository
from app.infrastructure.db.run_models import AgentEventRow, ResearchRunRow
from app.services.research_loop import ResearchLoopService
from app.tools.duckduckgo_search import DuckDuckGoAdapter
from app.tools.provider_adapters import BingAdapter, SearXNGAdapter
from app.tools.web_search import SearXNGSearchProvider
from sqlalchemy import select

from tests.integration.test_v3_deterministic_e2e import (
    _sqlite_utc_datetime_compatibility,  # noqa: F401
)
from tests.unit.test_first_evidence_budget_regression import store as first_store  # noqa: F401
from tests.unit.test_research_loop import FakeArtifacts, FakeExtractor, FakeReader


@pytest.fixture
def store(first_store):  # noqa: F811
    return first_store


async def target_and_budget(store, used=5):
    factory, run_id = store
    tools = ResearchToolRepository(factory)
    target = await tools.prepare_target(run_id, worker_task_id="offline-worker")
    assert target is not None
    target = replace(target, provider_request_allowance=3)
    with factory._sync_factory() as session, session.begin():
        run = session.get(ResearchRunRow, run_id)
        run.budget_snapshot = {
            **run.budget_snapshot,
            "max_searches": 6,
            "max_logical_queries": 6,
            "max_provider_requests": 18,
            "deadline_at": (datetime.now(UTC) + timedelta(seconds=180)).isoformat(),
        }
        run.usage_snapshot = {**run.usage_snapshot, "searches": used, "logical_queries": used}
    return tools, target


def usage(store):
    factory, run_id = store
    with factory._sync_factory() as session:
        return dict(session.get(ResearchRunRow, run_id).usage_snapshot)


class EmptySearch:
    provider_name = "offline"
    last_request_count = 1
    last_timeout_count = last_fallback_count = last_healthy_response_count = 0
    last_unresponsive_response_count = last_productive_response_count = 0

    def __init__(self):
        self.requests = []

    async def search(self, query, **kwargs):
        self.requests.append(query)
        return []


@pytest.mark.parametrize("kind", ["initial", "recovery", "feedback", "v3"])
@pytest.mark.parametrize("used", [5, 6])
async def test_all_executor_paths_use_one_hard_dispatch_guard(store, kind, used):
    tools, target = await target_and_budget(store, used)
    if kind == "feedback":
        target = replace(
            target,
            feedback_id=uuid4(),
            query_execution_id=uuid4(),
            query="q1:d2 primary source official report authoritative evidence",
        )
    elif kind == "recovery":
        target = replace(target, attempt_index=2, gap_attempt_index=2)
    elif kind == "v3":
        target = replace(target, execution_work_ref=uuid4(), execution_attempt_ref=uuid4())
    search = EmptySearch()
    loop = ResearchLoopService(tools, search, FakeReader(), FakeExtractor(), FakeArtifacts())
    result = await loop.execute_target(store[1], worker_task_id="offline-worker", target=target)
    assert len(search.requests) == (1 if used == 5 else 0)
    assert usage(store)["logical_queries"] == 6
    assert usage(store)["searches"] == 6
    if used == 6:
        assert result.outcome == "search_budget_exhausted"
        assert usage(store).get("search_provider_requests", 0) == 0
    else:
        assert usage(store)["search_provider_requests"] == 1
        assert usage(store)["search_provider_requests_reserved"] == 0
    # The state validator remains strict and now receives a representable state.
    ResearchState(
        run_id=store[1],
        budget_limits=BudgetLimits(max_searches=6, max_logical_queries=6),
        budget_usage=BudgetUsage(searches=usage(store)["searches"], logical_queries=6),
    )


@pytest.mark.parametrize("outcome", ["success", "known_failure", "timeout", "crash"])
async def test_settlement_no_refund_or_replay_across_repository_restart(store, outcome):
    tools, target = await target_and_budget(store)
    reservation = await tools.reserve_search_dispatch(
        store[1], worker_task_id="offline-worker", target=target
    )
    assert reservation.allowed and usage(store)["logical_queries"] == 6
    reserved = replace(target, search_dispatch_ref=reservation.dispatch_ref)
    if outcome == "success":
        await tools.record_search_results(
            store[1],
            worker_task_id="offline-worker",
            target=reserved,
            results=[],
            provider_requests=1,
        )
    elif outcome != "crash":
        for _ in range(2):
            await tools.record_tool_failure(
                store[1],
                worker_task_id="offline-worker",
                target=reserved,
                error_code="SEARCH_DEADLINE_EXHAUSTED"
                if outcome == "timeout"
                else "SEARCH_PROVIDER_DEGRADED",
                retryable=True,
                provider_requests=1,
            )
    restarted = ResearchToolRepository(store[0])
    duplicate = await restarted.reserve_search_dispatch(
        store[1], worker_task_id="offline-worker", target=target
    )
    assert not duplicate.allowed and duplicate.reason == "search_dispatch_no_replay"
    next_target = replace(target, attempt_index=target.attempt_index + 1)
    again = await restarted.reserve_search_dispatch(
        store[1], worker_task_id="offline-worker", target=next_target
    )
    assert not again.allowed
    assert usage(store)["logical_queries"] == 6
    if outcome in {"crash", "timeout"}:
        assert usage(store)["search_provider_requests_reserved"] > 0
    assert usage(store).get("search_provider_requests", 0) == (0 if outcome == "crash" else 1)


async def test_competing_targets_cannot_both_take_the_last_slot(store):
    tools, target = await target_and_budget(store)
    results = await asyncio.gather(
        *[
            tools.reserve_search_dispatch(
                store[1],
                worker_task_id="offline-worker",
                target=replace(target, tool_call_id=uuid4()),
            )
            for _ in range(16)
        ]
    )
    assert sum(item.allowed for item in results) == 1
    assert usage(store)["logical_queries"] == 6
    assert len(usage(store)["search_dispatches"]) == 1


@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
async def test_interrupted_search_retains_persistent_no_replay_intent(store, interruption):
    tools, target = await target_and_budget(store)

    class Interrupted(EmptySearch):
        async def search(self, query, **kwargs):
            self.requests.append(query)
            if interruption == "cancel":
                raise asyncio.CancelledError
            raise TimeoutError

    search = Interrupted()
    loop = ResearchLoopService(tools, search, FakeReader(), FakeExtractor(), FakeArtifacts())
    if interruption == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await loop.execute_target(store[1], worker_task_id="offline-worker", target=target)
    else:
        result = await loop.execute_target(store[1], worker_task_id="offline-worker", target=target)
        assert result.outcome == "deadline_exhausted"
    assert len(search.requests) == 1 and usage(store)["logical_queries"] == 6
    assert usage(store)["search_provider_requests_reserved"] > 0
    restarted = ResearchToolRepository(store[0])
    blocked = await restarted.reserve_search_dispatch(
        store[1],
        worker_task_id="offline-worker",
        target=replace(target, tool_call_id=uuid4()),
    )
    assert not blocked.allowed and "unknown" in blocked.reason


async def test_actual_run_lock_compiles_to_postgres_for_update(store):
    from sqlalchemy.dialects import postgresql

    class CapturedSession:
        async def scalar(self, statement):
            assert str(statement.compile(dialect=postgresql.dialect())).endswith("FOR UPDATE")
            with store[0]._sync_factory() as session:
                return session.get(ResearchRunRow, store[1])

    await ResearchToolRepository._locked_run(CapturedSession(), store[1], "offline-worker")


async def test_intent_persistence_failure_cannot_reach_search(store, monkeypatch):
    tools, target = await target_and_budget(store)
    original = tools._append_event

    async def fail(*args, **kwargs):
        if kwargs.get("event_type") == "search.dispatch.reserved":
            raise OSError("synthetic persistence failure")
        return await original(*args, **kwargs)

    monkeypatch.setattr(tools, "_append_event", fail)
    search = EmptySearch()
    loop = ResearchLoopService(tools, search, FakeReader(), FakeExtractor(), FakeArtifacts())
    with pytest.raises(OSError, match="persistence failure"):
        await loop.execute_target(store[1], worker_task_id="offline-worker", target=target)
    assert not search.requests
    assert usage(store)["logical_queries"] == 5
    assert not usage(store).get("search_dispatches")


async def test_known_failed_retry_is_new_bounded_logical_attempt(store):
    tools, target = await target_and_budget(store, used=4)
    reserved = await tools.reserve_search_dispatch(
        store[1], worker_task_id="offline-worker", target=target
    )
    await tools.record_tool_failure(
        store[1],
        worker_task_id="offline-worker",
        target=replace(target, search_dispatch_ref=reserved.dispatch_ref),
        error_code="SEARCH_PROVIDER_DEGRADED",
        retryable=True,
        provider_requests=1,
    )
    retry = await tools.reserve_search_dispatch(
        store[1],
        worker_task_id="offline-worker",
        target=replace(target, attempt_index=target.attempt_index + 1),
    )
    assert retry.allowed and usage(store)["logical_queries"] == 6


async def test_feedback_execution_cannot_replay_with_changed_target_indices(store):
    tools, target = await target_and_budget(store, used=4)
    target = replace(target, feedback_id=uuid4(), query_execution_id=uuid4())
    reservation = await tools.reserve_search_dispatch(
        store[1], worker_task_id="offline-worker", target=target
    )
    await tools.record_tool_failure(
        store[1],
        worker_task_id="offline-worker",
        target=replace(target, search_dispatch_ref=reservation.dispatch_ref),
        error_code="SEARCH_PROVIDER_DEGRADED",
        retryable=True,
        provider_requests=1,
    )
    replay = await tools.reserve_search_dispatch(
        store[1],
        worker_task_id="offline-worker",
        target=replace(target, tool_call_id=uuid4(), attempt_index=9, gap_attempt_index=9),
    )
    assert not replay.allowed and replay.reason == "search_dispatch_no_replay"
    assert usage(store)["logical_queries"] == 5


@pytest.mark.parametrize("allowance", [0, 1, 3])
async def test_fallback_http_shares_the_sixth_logical_reservation(store, monkeypatch, allowance):
    tools, target = await target_and_budget(store)
    target = replace(target, provider_request_allowance=allowance)
    monkeypatch.setattr(tools, "_candidate_providers", lambda: ("SearXNG", "Bing", "DuckDuckGo"))
    requests = []

    def handler(request):
        requests.append(request.url.host)
        # Charge has already committed before any MockTransport callback.
        assert usage(store)["logical_queries"] == 6
        if request.url.host == "searxng":
            return httpx.Response(
                200, json={"results": [], "unresponsive_engines": [["engine", "unresponsive"]]}
            )
        if request.url.host == "www.bing.com":
            return httpx.Response(200, text="<html>no results</html>")
        return httpx.Response(
            200,
            text=(
                '<a class="result__a" href="https://example.gov/statement">Official source</a>'
                '<a class="result__snippet">An official statement.</a>'
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        search = SearXNGSearchProvider(client, "http://searxng")
        loop = ResearchLoopService(
            tools,
            search,
            FakeReader(),
            FakeExtractor(),
            FakeArtifacts(),
            search_adapters={
                "SearXNG": SearXNGAdapter(search),
                "Bing": BingAdapter(search),
                "DuckDuckGo": DuckDuckGoAdapter(client),
            },
        )
        await loop.execute_target(store[1], worker_task_id="offline-worker", target=target)
        before = len(requests)
        await loop.execute_target(
            store[1],
            worker_task_id="offline-worker",
            target=replace(target, tool_call_id=uuid4()),
        )
        assert len(requests) == before
    assert len(requests) <= allowance
    if allowance == 3:
        assert requests == ["searxng", "www.bing.com", "html.duckduckgo.com"]
    assert usage(store)["logical_queries"] == (5 if allowance == 0 else 6)
    assert usage(store).get("search_provider_requests", 0) == len(requests)
    with store[0]._sync_factory() as session:
        events = list(session.scalars(select(AgentEventRow)))
        assert sum(e.event_type == "search.dispatch.reserved" for e in events) == (allowance > 0)
