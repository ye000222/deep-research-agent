"""No services: transactional SQLite source grants and MockTransport fallback."""

import asyncio
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from app.domain.planning import ResearchPlanDraft
from app.domain.providers import TokenUsage, UsageAccuracy
from app.infrastructure.db.model_budget_models import ModelBudgetReservationRow
from app.infrastructure.db.reports import ReportRepository
from app.infrastructure.db.research_models import ResearchEvidenceRow
from app.infrastructure.db.research_tools import ResearchToolRepository
from app.infrastructure.db.run_models import AgentEventRow, ResearchPlanItemRow, ResearchRunRow
from app.services.planner import _scoped_plan
from app.services.research_loop import ResearchLoopService
from app.tools.errors import ToolExecutionError
from app.tools.provider_adapters import BingAdapter, SearXNGAdapter
from app.tools.web_search import SearXNGSearchProvider
from sqlalchemy import delete, func, select

from tests.integration.test_v3_deterministic_e2e import (
    _sqlite_utc_datetime_compatibility,  # noqa: F401 -- SQLite lease compatibility
)
from tests.unit.test_first_evidence_budget_regression import store as first_store  # noqa: F401
from tests.unit.test_research_loop import FakeArtifacts, FakeExtractor, FakeReader, FakeRepository

URL = "https://example.gov/second"
PAGE = {"url": URL, "content_hash": "b" * 64}


@pytest.fixture
def store(first_store):  # noqa: F811 -- shared fixture registered by import
    factory, run_id = first_store
    with factory._sync_factory() as session, session.begin():
        session.execute(
            delete(ResearchPlanItemRow).where(
                ResearchPlanItemRow.question_id.in_(["q5", "q6", "q7"]),
            )
        )
        row = session.get(ResearchRunRow, run_id)
        allocation = {
            **row.budget_snapshot["allocation"],
            "question_token_budgets": {f"q{i}": 3125 for i in range(1, 5)},
            "question_token_floors": {f"q{i}": 2500 for i in range(1, 5)},
        }
        row.budget_snapshot = {
            **row.budget_snapshot,
            "allocation": allocation,
            "deadline_at": (datetime.now(UTC) + timedelta(seconds=180)).isoformat(),
        }
    return factory, run_id


def event(store, kind, **refs):
    factory, run_id = store
    with factory._sync_factory() as session, session.begin():
        run = session.get(ResearchRunRow, run_id)
        session.add(
            AgentEventRow(
                run_id=run_id,
                run_seq=run.next_event_seq,
                schema_version=1,
                phase=run.phase,
                event_type=kind,
                public_summary="offline fixture",
                refs={"question_id": "q1", **refs},
                created_at=datetime.now(UTC),
            )
        )
        run.next_event_seq += 1


def evidence(store, *, accepted=False, reason="evidence_relevance_below_threshold"):
    factory, run_id = store
    with factory._sync_factory() as session, session.begin():
        session.add(
            ResearchEvidenceRow(
                id=uuid4(),
                run_id=run_id,
                plan_version=1,
                question_id="q1",
                source_id=uuid4(),
                claim="A background statement",
                exact_quote="A background statement",
                relation="context",
                relevance=0.35,
                confidence=0.9,
                source_reliability=0.9,
                evidence_score=0.7,
                accepted=accepted,
                rejection_reason=None if accepted else reason,
                evidence_hash=uuid4().hex * 2,
                created_at=datetime.now(UTC),
            )
        )


async def reserve(store, *, amount=2875, attempt=None, page=PAGE):
    factory, run_id = store
    return await ResearchToolRepository(factory).reserve_model_tokens(
        run_id,
        worker_task_id="offline-worker",
        question_id="q1",
        node="evidence_extractor",
        attempt_id=attempt or uuid4(),
        estimated_input=amount - 768,
        max_output=768,
        minimum_call_tokens=2700,
        continuation_page=page,
    )


async def completed_first(store, *, accepted=False, empty=False):
    factory, run_id = store
    event(store, "source.readable", final_url="https://example.gov/first")
    attempt = uuid4()
    assert (await reserve(store, amount=3125, attempt=attempt, page=None)).granted
    await ResearchToolRepository(factory).settle_model_reservation(
        run_id,
        worker_task_id="offline-worker",
        attempt_id=attempt,
        actual_total=1000,
        usage_estimated=False,
    )
    event(
        store,
        "evidence.extraction_started",
        final_url="https://example.gov/first",
        content_hash="a" * 64,
    )
    event(store, "evidence.extracted")
    event(store, "source.readable", final_url=URL)
    if not empty:
        evidence(store, accepted=accepted)


@pytest.mark.parametrize("empty", [False, True])
async def test_settled_unaccepted_first_extraction_can_borrow_for_read_second_source(store, empty):
    await completed_first(store, empty=empty)
    # Reproduce the original state-machine denial before using the verified
    # continuation path. A denied reservation consumes no continuation grant.
    old_path = await reserve(store, page=None)
    assert not old_path.granted and old_path.reason == "first_pass_floor_protected"
    attempt = uuid4()
    assert (await reserve(store, attempt=attempt)).granted
    assert (await reserve(store, attempt=attempt)).granted
    factory, run_id = store
    with factory._sync_factory() as session:
        assert session.scalar(select(func.count(ModelBudgetReservationRow.id))) == 2
        grant = session.scalar(
            select(AgentEventRow).where(
                AgentEventRow.event_type == "budget.source_continuation_reserved",
            )
        )
        assert grant.metrics["future_question_reserve_tokens"] == 8100
        assert grant.metrics["requested_tokens"] == 2875
        assert session.get(ResearchRunRow, run_id).budget_snapshot["max_tokens"] == 30000


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "same_url",
        "same_hash",
        "accepted",
        "deadline",
        "global",
        "future_floor",
        "uncertain",
        "released",
        "reserved",
    ],
)
async def test_source_continuation_rejects_unsafe_or_unaffordable_grants(store, case):
    await completed_first(store, accepted=case == "accepted")
    factory, run_id = store
    page = PAGE
    if case == "missing":
        page = {**PAGE, "url": "https://example.gov/not-read"}
    elif case == "same_url":
        page = {**PAGE, "url": "https://example.gov/first"}
    elif case == "same_hash":
        page = {**PAGE, "content_hash": "a" * 64}
    with factory._sync_factory() as session, session.begin():
        run = session.get(ResearchRunRow, run_id)
        if case == "deadline":
            run.budget_snapshot = {
                **run.budget_snapshot,
                "deadline_at": (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
            }
        if case == "global":
            run.usage_snapshot = {"model_tokens": 29500}
        if case == "future_floor":
            allocation = {
                **run.budget_snapshot["allocation"],
                "question_token_floors": {f"q{i}": 4000 for i in range(2, 5)},
            }
            run.budget_snapshot = {**run.budget_snapshot, "allocation": allocation}
        if case in {"uncertain", "released", "reserved"}:
            row = session.scalar(select(ModelBudgetReservationRow))
            row.status, row.actual_total = case, None
    assert not (await reserve(store, page=page)).granted


async def test_one_extra_source_grant_is_not_refunded_by_release(store):
    await completed_first(store)
    attempt = uuid4()
    assert (await reserve(store, attempt=attempt)).granted
    factory, run_id = store
    await ResearchToolRepository(factory).release_model_reservation(
        run_id,
        worker_task_id="offline-worker",
        attempt_id=attempt,
    )
    assert not (await reserve(store)).granted
    assert not (await reserve(store, attempt=attempt)).granted


async def test_settled_continuation_cannot_grant_a_third_source(store):
    await completed_first(store)
    attempt = uuid4()
    assert (await reserve(store, attempt=attempt)).granted
    factory, run_id = store
    await ResearchToolRepository(factory).settle_model_reservation(
        run_id,
        worker_task_id="offline-worker",
        attempt_id=attempt,
        actual_total=1000,
        usage_estimated=False,
    )
    third = {"url": "https://example.gov/third", "content_hash": "c" * 64}
    event(store, "source.readable", final_url=third["url"])
    assert not (await reserve(store, page=third)).granted


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("北斗二号在中国境内2019年的开通日期是什么?", "subject_identity_changed"),
        ("北斗三号在中国境内2024年的开通日期是什么?", "time_scope_changed"),
        ("北斗三号在欧盟境内2019年的开通日期是什么?", "region_changed"),
        ("北斗三号在中国境内2019年开始建设的日期是什么?", "requested_event_replaced"),
        ("北斗三号未来升级计划是什么?", "unrequested_fact_type"),
    ],
)
def test_planner_rejection_logs_index_and_deterministic_reason_only(caplog, text, reason):
    goal = "北斗三号在中国境内2018至2020年何时正式开通服务?"
    question = {
        "question": "北斗三号在中国境内2019年的开通日期是什么?",
        "priority": 1,
        "rationale": "核验原始来源日期。",
        "evidence_requirements": ["原始引用"],
        "search_hints": ["北斗三号"],
    }
    draft = ResearchPlanDraft(
        scope_summary=goal,
        questions=[question, {**question, "question": text}],
        completion_criteria=["日期可核验", "原文可追溯"],
    )
    caplog.set_level("INFO", logger="app.services.planner")
    scoped = _scoped_plan(
        goal,
        draft,
        TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2, accuracy=UsageAccuracy.EXACT),
    )
    assert len(scoped.questions) == 1
    assert f"question_index=2 reason_code={reason}" in caplog.text
    assert text not in caplog.text


@pytest.mark.parametrize(
    "reason", ["evidence_relevance_below_threshold", "quote_not_found_in_source", "mixed"]
)
async def test_no_evidence_failure_cause_uses_actual_validation_rejections(store, reason):
    await completed_first(store, empty=True)
    evidence(store, reason="evidence_relevance_below_threshold" if reason == "mixed" else reason)
    if reason == "mixed":
        evidence(store, reason="quote_not_found_in_source")
    factory, run_id = store
    with factory._sync_factory() as session, session.begin():
        row = session.get(ResearchRunRow, run_id)
        row.phase, row.termination_reason = "writing", "deadline_exhausted"
    await ReportRepository(factory).fail_no_evidence(run_id, worker_task_id="offline-worker")
    with factory._sync_factory() as session:
        failure = session.scalar(
            select(AgentEventRow).where(AgentEventRow.event_type == "run.failed")
        )
        assert failure.refs["failure_cause"] == (
            "evidence_validation_rejected" if reason == "mixed" else reason
        )
        assert failure.metrics["validation_rejections"]


@pytest.mark.parametrize(
    "case",
    [
        "success",
        "degraded",
        "primary_open",
        "both_open",
        "bing_open",
        "unconfigured",
        "fallback_failed",
        "budget",
        "deadline",
    ],
)
async def test_routed_fallback_is_real_bounded_and_accurately_counted(store, case):
    requests = []

    async def handler(request):
        requests.append(request.url.host)
        if request.url.host == "searxng":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Industrial inspection",
                            "url": "https://example.gov/inspection",
                            "content": "Industrial inspection original source",
                        }
                    ]
                    if case == "success"
                    else [],
                    "unresponsive_engines": []
                    if case == "success"
                    else [["engine", "unresponsive"]],
                },
            )
        return httpx.Response(
            502 if case == "fallback_failed" else 200,
            text=(
                '<li class="b_algo"><h2><a href="https://example.gov/inspection">'
                "Industrial inspection"
                "</a></h2><p>Industrial inspection original source</p></li>"
            ),
        )

    factory, run_id = store
    repository = ResearchToolRepository(factory)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SearXNGSearchProvider(client, "http://searxng")
        if case in {"primary_open", "both_open"}:
            provider.cooldown_until = time.monotonic() + 300
        if case in {"both_open", "bing_open"}:
            provider._provider_states["Bing"].update(
                state="cooldown", cooldown_until=time.monotonic() + 300
            )
        adapters = {"SearXNG": SearXNGAdapter(provider)}
        if case != "unconfigured":
            adapters["Bing"] = BingAdapter(provider)
        service = ResearchLoopService(
            repository,
            provider,
            FakeReader(),
            FakeExtractor(),
            FakeArtifacts(),
            search_adapters=adapters,
        )
        target = replace(
            FakeRepository().target,
            provider_request_allowance=0 if case == "budget" else 3,
            deadline_at=datetime.now(UTC) + timedelta(seconds=-1 if case == "deadline" else 180),
        )
        telemetry = SimpleNamespace(last_request_count=0)
        if case in {"success", "degraded", "primary_open"}:
            results, used = await service._routed_search(
                run_id, worker_task_id="offline-worker", target=target, telemetry=telemetry
            )
            assert results[0].url == "https://example.gov/inspection"
            assert used == ("SearXNG" if case == "success" else "Bing")
        else:
            with pytest.raises((ToolExecutionError, TimeoutError)) as failure:
                await service._routed_search(
                    run_id, worker_task_id="offline-worker", target=target, telemetry=telemetry
                )
            if case == "unconfigured":
                assert failure.value.details["provider"] == "SearXNG"
                assert telemetry.selected_provider == "SearXNG"
                assert not getattr(telemetry, "fallback_requests", 0)
        assert telemetry.last_request_count == len(requests) <= target.provider_request_allowance
        assert requests.count("www.bing.com") == (
            0
            if case in {"success", "both_open", "bing_open", "unconfigured", "budget", "deadline"}
            else 1
        )
        with factory._sync_factory() as session:
            events = list(
                session.scalars(
                    select(AgentEventRow).where(
                        AgentEventRow.event_type == "provider.execution.completed"
                    )
                )
            )
            assert sum(e.metrics["provider_requests"] for e in events) == len(requests)
            assert sum(bool(e.refs["fallback_attempted"]) for e in events) == requests.count(
                "www.bing.com"
            )


async def test_deadline_cancellation_keeps_started_http_count_without_fallback(store):
    requests = []
    entered_http = asyncio.Event()

    async def handler(request):
        requests.append(request.url.host)
        entered_http.set()
        await asyncio.Event().wait()
        return httpx.Response(200, json={"results": []})

    factory, run_id = store
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SearXNGSearchProvider(client, "http://searxng")
        service = ResearchLoopService(
            ResearchToolRepository(factory),
            provider,
            FakeReader(),
            FakeExtractor(),
            FakeArtifacts(),
        )
        telemetry = SimpleNamespace(last_request_count=0)
        task = asyncio.create_task(
            service._routed_search(
                run_id,
                worker_task_id="offline-worker",
                target=replace(FakeRepository().target, provider_request_allowance=3),
                telemetry=telemetry,
            )
        )
        await asyncio.wait_for(entered_http.wait(), timeout=1)
        # Same cancellation delivered by the production outer wait_for deadline.
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert telemetry.last_request_count == len(requests) == 1
        assert requests == ["searxng"]
    with factory._sync_factory() as session:
        event = session.scalar(
            select(AgentEventRow).where(AgentEventRow.event_type == "provider.execution.completed")
        )
        assert event.metrics["provider_requests"] == 1
        assert event.refs["error_code"] == "SEARCH_DEADLINE_EXHAUSTED"
        assert event.refs["fallback_attempted"] is False
