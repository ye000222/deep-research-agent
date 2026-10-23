"""Actual Loop/Repository/Reader/Extractor; SQLite memory + MockTransport only."""
# ruff: noqa: RUF001 -- preserve Chinese factual fixtures

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from app.domain.controlled_tools import EvidenceSearchResult
from app.infrastructure.db.research_models import ResearchEvidenceRow
from app.infrastructure.db.run_models import AgentEventRow, ResearchPlanItemRow, ResearchRunRow
from app.services.research_loop import ResearchLoopService
from app.tools import web_reader
from app.tools.analyze_data import AnalyzeDataTool
from app.tools.duckduckgo_search import DuckDuckGoAdapter
from app.tools.gateway import ControlledToolGateway
from app.tools.provider_adapters import BingAdapter, SearXNGAdapter
from app.tools.web_search import SearXNGSearchProvider
from sqlalchemy import select

from tests.integration.test_evidence_input_chain_offline import InputGateway
from tests.integration.test_feedback_closure_evidence_offline import setup
from tests.integration.test_research_reliability_offline import TASK
from tests.integration.test_v3_deterministic_e2e import (
    _sqlite_utc_datetime_compatibility,  # noqa: F401
)

QUESTION = "北斗三号全球卫星导航系统何时正式开通服务？"
QUOTE = "2020年7月31日，北斗三号全球卫星导航系统正式开通服务。"
BAD_BODY = "卫星导航信号处理教材，介绍载波频率与嵌入式系统、时间膨胀理论。" * 35


@pytest.mark.parametrize("good_provider", ["Bing", "DuckDuckGo", None, "SearXNG"])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("validation_mode", ["valid", "low_relevance", "wrong_claim", "empty"])
async def test_nonempty_downstream_continues_one_logical_opportunity(
    tmp_path,
    monkeypatch,
    good_provider,
    parallel,
    validation_mode,
):
    factory, artifacts, run_id, tools, extractor = await setup(
        tmp_path, InputGateway(validation_mode, QUESTION, QUOTE)
    )
    monkeypatch.setattr(web_reader, "_require_public_destination", AsyncMock())
    monkeypatch.setattr(tools, "_candidate_providers", lambda: ("SearXNG", "Bing", "DuckDuckGo"))
    requests = []
    # PostgreSQL FTS (@@) is outside this SQLite fixture. Only the evidence
    # lookup boundary is doubled; ControlledToolGateway/ToolPolicyGuard run
    # unchanged on every provider continuation.
    evidence_lookup = SimpleNamespace(
        execute=AsyncMock(
            return_value=EvidenceSearchResult(
                call_id=uuid4(),
                status="succeeded",
                items=(),
                result_refs=(),
            )
        )
    )
    with factory._sync_factory() as session, session.begin():
        run = session.get(ResearchRunRow, run_id)
        run.budget_snapshot = {
            **run.budget_snapshot,
            "max_pages_fetched": 10,
            "max_page_fetch_attempts": 16,
        }

    def result_url(provider, index):
        return f"https://agency.gov/{provider}-{index}"

    def handler(request):
        host = request.url.host
        requests.append((host, request.url.path))
        provider = {
            "searxng": "SearXNG",
            "www.bing.com": "Bing",
            "html.duckduckgo.com": "DuckDuckGo",
        }.get(host)
        if provider:
            # Every real provider HTTP is inside the SAME committed reservation.
            with factory._sync_factory() as session:
                run = session.get(ResearchRunRow, run_id)
                assert run.usage_snapshot["logical_queries"] == 1
                entries = list(run.usage_snapshot["search_dispatches"].values())
                assert len(entries) == 1 and entries[0]["state"] == "RESERVED_NO_REPLAY"
            count = 1 if provider == good_provider else 3
            if provider == "SearXNG":
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "title": "北斗三号正式开通服务日期 官方公告",
                                "url": result_url(provider, i),
                                "content": "BeiDou official opening service date",
                            }
                            for i in range(count)
                        ],
                        "unresponsive_engines": [],
                    },
                )
            if provider == "Bing":
                return httpx.Response(
                    200,
                    text="<ol>"
                    + "".join(
                        f'<li class="b_algo"><h2><a href="{result_url(provider, i)}">'
                        "北斗三号正式开通服务日期</a></h2><p>BeiDou opening service date</p></li>"
                        for i in range(count)
                    )
                    + "</ol>",
                )
            return httpx.Response(
                200,
                text="".join(
                    f'<a class="result__a" href="{result_url(provider, i)}">北斗三号开通</a>'
                    '<a class="result__snippet">BeiDou official opening service date</a>'
                    for i in range(count)
                ),
            )
        body = (
            QUOTE + "官方原始公告介绍卫星导航系统的服务范围与建设背景。" * 35
            if good_provider and request.url.path.startswith(f"/{good_provider}-")
            else BAD_BODY
        )
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text=f"<html><body><article><p>{body}</p></article></body></html>",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        search = SearXNGSearchProvider(client, "http://searxng")
        loop = ResearchLoopService(
            tools,
            search,
            web_reader.PublicWebReader(client),
            extractor,
            artifacts,
            controlled_tools=ControlledToolGateway(evidence_lookup, AnalyzeDataTool(factory)),
            parallel_reads_enabled=parallel,
            search_adapters={
                "SearXNG": SearXNGAdapter(search),
                "Bing": BingAdapter(search),
                "DuckDuckGo": DuckDuckGoAdapter(client),
            },
        )
        target = await tools.prepare_target(run_id, worker_task_id=TASK)
        result = await loop.execute_target(run_id, worker_task_id=TASK, target=target)
    expected_order = ["SearXNG", "Bing", "DuckDuckGo"]
    if good_provider:
        expected_order = expected_order[: expected_order.index(good_provider) + 1]
    actual = [host for host, _ in requests if host != "agency.gov"]
    assert actual == [
        {"SearXNG": "searxng", "Bing": "www.bing.com", "DuckDuckGo": "html.duckduckgo.com"}[p]
        for p in expected_order
    ]
    with factory._sync_factory() as session:
        run = session.get(ResearchRunRow, run_id)
        usage = run.usage_snapshot
        assert usage["logical_queries"] == usage["searches"] == 1
        assert usage["search_provider_requests"] == len(expected_order)
        assert usage["search_provider_requests_reserved"] == 0
        assert usage["query_family_stats"]["scope"]["logical_queries"] == 1
        assert usage.get("page_slots_reserved", 0) == 0
        entry = next(iter(usage["search_dispatches"].values()))
        assert entry["state"] == "SUCCEEDED" and entry["provider_requests"] == len(expected_order)
        events = list(session.scalars(select(AgentEventRow)))
        assessments = [e.refs for e in events if e.event_type == "search.downstream.assessed"]
        assert len(assessments) == len(expected_order)
        assert all(a["logical_budget_after"] == 1 for a in assessments)
        assert assessments[0]["logical_budget_before"] == 0
        assert all(a["logical_budget_before"] == 1 for a in assessments[1:])
        for assessment in assessments[:-1]:
            assert assessment["usable_count"] == 0
            assert assessment["continuation_reason"] == "no_downstream_usable_source"
        assert assessments[-1]["continuation_decision"] == "stop"
        evidence = list(session.scalars(select(ResearchEvidenceRow)))
        if good_provider:
            assert assessments[-1]["usable_count"] > 0
            assert assessments[-1]["stop_reason"] == "usable_source_found"
            if validation_mode == "valid":
                assert result.accepted_evidence > 0
                assert evidence and evidence[0].accepted and evidence[0].exact_quote == QUOTE
                assert evidence[0].snapshot_id and evidence[0].chunk_id
            else:
                assert result.accepted_evidence == 0
                assert not any(item.accepted for item in evidence)
        else:
            assert result.accepted_evidence == 0 and not evidence


@pytest.mark.parametrize("stop", ["requests", "deadline", "logical", "unknown", "cancel"])
async def test_stop_boundaries_never_create_an_extra_provider_request(tmp_path, monkeypatch, stop):
    factory, artifacts, run_id, tools, extractor = await setup(
        tmp_path, InputGateway("valid", QUESTION, QUOTE)
    )
    monkeypatch.setattr(web_reader, "_require_public_destination", AsyncMock())
    monkeypatch.setattr(tools, "_candidate_providers", lambda: ("SearXNG", "Bing", "DuckDuckGo"))
    target = await tools.prepare_target(run_id, worker_task_id=TASK)
    target = replace(target, provider_request_allowance=1 if stop == "requests" else 3)
    with factory._sync_factory() as session, session.begin():
        run = session.get(ResearchRunRow, run_id)
        if stop == "logical":
            run.usage_snapshot = {**run.usage_snapshot, "searches": 6, "logical_queries": 6}
    if stop == "deadline":
        target = replace(target, deadline_at=datetime.now(UTC) - timedelta(seconds=1))
    requests = []

    def handler(request):
        requests.append(request.url.host)
        if request.url.host == "searxng":
            if stop == "unknown":
                raise RuntimeError("offline interrupted transport")
            if stop == "cancel":
                raise asyncio.CancelledError
            return httpx.Response(200, json={"results": [], "unresponsive_engines": []})
        raise AssertionError("no fallback HTTP allowed")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        search = SearXNGSearchProvider(client, "http://searxng")
        loop = ResearchLoopService(
            tools,
            search,
            web_reader.PublicWebReader(client),
            extractor,
            artifacts,
            search_adapters={
                "SearXNG": SearXNGAdapter(search),
                "Bing": BingAdapter(search),
                "DuckDuckGo": DuckDuckGoAdapter(client),
            },
        )
        if stop == "unknown":
            with pytest.raises(RuntimeError, match="interrupted transport"):
                await loop.execute_target(run_id, worker_task_id=TASK, target=target)
        elif stop == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await loop.execute_target(run_id, worker_task_id=TASK, target=target)
        else:
            await loop.execute_target(run_id, worker_task_id=TASK, target=target)
    assert requests == ([] if stop in {"deadline", "logical"} else ["searxng"])
    with factory._sync_factory() as session:
        usage = session.get(ResearchRunRow, run_id).usage_snapshot
        assert usage.get("logical_queries", 0) <= 6
        if stop in {"unknown", "cancel"}:
            entry = next(iter(usage["search_dispatches"].values()))
            assert entry["state"] in {"RESERVED_NO_REPLAY", "UNKNOWN_NO_REPLAY"}


@pytest.mark.parametrize("first_eligible", [True, False])
async def test_another_question_can_extract_cached_body_after_fetch_and_search_limits(
    tmp_path,
    monkeypatch,
    first_eligible,
):
    first_question = QUESTION if first_eligible else "北斗三号最后一颗组网卫星何时成功发射？"
    factory, artifacts, run_id, tools, extractor = await setup(
        tmp_path, InputGateway("valid", first_question, QUOTE)
    )
    monkeypatch.setattr(web_reader, "_require_public_destination", AsyncMock())
    requests = []

    def handler(request):
        requests.append(request.url.host)
        if request.url.host == "searxng":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": first_question + " 北斗三号正式开通服务日期",
                            "url": "https://agency.gov/opening",
                            "content": "Official BeiDou opening service date",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text=f"<html><body><article><p>{QUOTE}</p><p>"
            + "官方公告介绍北斗系统服务覆盖与建设背景。" * 40
            + "</p></article></body></html>",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        search = SearXNGSearchProvider(client, "http://searxng")
        loop = ResearchLoopService(
            tools,
            search,
            web_reader.PublicWebReader(client),
            extractor,
            artifacts,
            search_adapters={"SearXNG": SearXNGAdapter(search)},
        )
        first = await tools.prepare_target(run_id, worker_task_id=TASK)
        initial = await loop.execute_target(run_id, worker_task_id=TASK, target=first)
        assert bool(initial.accepted_evidence) == first_eligible
        with factory._sync_factory() as session, session.begin():
            item = session.scalar(
                select(ResearchPlanItemRow).where(
                    ResearchPlanItemRow.run_id == run_id, ResearchPlanItemRow.question_id == "q1"
                )
            )
            item.status = "researched"
            session.add(
                ResearchPlanItemRow(
                    id=uuid4(),
                    run_id=run_id,
                    plan_version=1,
                    question_id="q3",
                    question=QUESTION,
                    priority=1,
                    rationale="Synthetic independent dimension reuse",
                    evidence_requirements=["正式开通的官方日期原文"],
                    search_hints=[QUESTION],
                    status="pending",
                )
            )
            run = session.get(ResearchRunRow, run_id)
            run.usage_snapshot = {**run.usage_snapshot, "searches": 6, "logical_queries": 6}
            run.budget_snapshot = {**run.budget_snapshot, "max_pages_fetched": 1}
        count_before = len(requests)
        # The work scheduler may prefer q1's independent-source gap. Select
        # q3 explicitly through the same target-preparation implementation.
        second = (
            await tools._prepare_target(run_id, worker_task_id=TASK, selected_question_id="q3")
            if first_eligible
            else await tools.prepare_target(run_id, worker_task_id=TASK)
        )
        assert second is not None and second.question_id == "q3"
        result = await loop.execute_target(run_id, worker_task_id=TASK, target=second)
        assert result.accepted_evidence > 0
        assert len(requests) == count_before
    with factory._sync_factory() as session:
        usage = session.get(ResearchRunRow, run_id).usage_snapshot
        assert usage["pages_fetched"] == usage["page_fetch_attempts"] == 1
        assert usage["logical_queries"] == 6
        assert (
            session.scalar(
                select(ResearchEvidenceRow).where(
                    ResearchEvidenceRow.question_id == "q3", ResearchEvidenceRow.accepted.is_(True)
                )
            )
            is not None
        )
