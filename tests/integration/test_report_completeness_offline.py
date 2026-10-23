"""Accepted-row authority and event/report closure over the production repositories."""

import asyncio
import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from app.api.dependencies import get_client_session, get_research_run_service
from app.context.manager import ContextBudgetManager
from app.core.config import Settings
from app.domain.providers import CanonicalModelResult, TokenUsage, UsageAccuracy
from app.domain.research_tools import ReadPage
from app.infrastructure.db.evidence_counts import sync_accepted_evidence_count
from app.infrastructure.db.model_budget_models import ModelBudgetReservationRow
from app.infrastructure.db.reports import ReportRepository
from app.infrastructure.db.research_models import ResearchEvidenceRow
from app.infrastructure.db.research_runs import ResearchRunRepository
from app.infrastructure.db.research_tools import ResearchTarget, ResearchToolRepository
from app.infrastructure.db.run_models import AgentEventRow, ResearchPlanItemRow, ResearchRunRow
from app.main import create_app
from app.security.client_sessions import ClientSession
from app.services.evidence_extractor import EvidenceExtractorService
from app.services.report_writer import ReportWriterService
from app.services.research_loop import ResearchLoopService
from app.tools import web_reader
from app.tools.web_search import SearXNGSearchProvider
from sqlalchemy import select

from tests.integration.test_evidence_input_chain_offline import InputGateway
from tests.integration.test_feedback_closure_evidence_offline import setup as source_setup
from tests.integration.test_research_reliability_offline import TASK as SOURCE_TASK
from tests.integration.test_v3_deterministic_e2e import (
    _Bindings,
    _Cipher,
    _new_store,
    _seed_run,
    _sqlite_utc_datetime_compatibility,  # noqa: F401
)
from tests.integration.test_writer_context_offline import draft_for
from tests.unit.test_event_coverage import GOAL, QUOTES

TASK = "offline-writer-worker"
OWNER = "a" * 64


async def materialize(tmp_path, *, count=5, rejected=False, full=False):
    factory, artifacts = _new_store(tmp_path)
    run_id = uuid4()
    _seed_run(factory, run_id, GOAL)
    runs, tools = ResearchRunRepository(factory), ResearchToolRepository(factory)
    assert await runs.acquire_for_execution(run_id, worker_task_id=TASK)
    # One broad question with four independent projected event dimensions.
    # This exercises compatibility with existing persisted plans as well.
    with factory._sync_factory() as session:
        row = session.get(ResearchRunRow, run_id)
        row.plan_version = 1
        row.phase = "researching"
        row.orchestrator_version = "V2_LEGACY"
        row.budget_snapshot = {
            **row.budget_snapshot,
            "max_tokens": 30000,
            "max_extraction_calls": 5,
            "max_wall_clock_seconds": 180,
            "allocation": {"writer_tokens_initial": 4500, "research_tokens": 12500},
        }
        session.add(
            ResearchPlanItemRow(
                id=uuid4(),
                run_id=run_id,
                plan_version=1,
                question_id="q1",
                question=GOAL,
                priority=1,
                rationale="核实原始事件时间。",
                evidence_requirements=["原始报道中的事件日期"],
                search_hints=[GOAL],
                status="active",
            )
        )
        session.commit()
    target = ResearchTarget(
        plan_version=1,
        question_id="q1",
        question=GOAL,
        query=GOAL,
        gap_id=uuid4(),
        tool_call_id=uuid4(),
        source_id_seed=uuid4(),
        original_goal=GOAL,
        acceptance_dimensions=(("q1:d1", "Atlas系统事件时间"),),
    )
    quotes = (
        QUOTES
        if full
        else [QUOTES[2]]
        if count == 1
        else [*QUOTES, "Atlas系统发布了原始运行资料。"][:count]
    )
    bindings = _Bindings(run_id)

    async def generate_structured(**kwargs):
        return CanonicalModelResult(
            parsed_object={
                "items": [
                    {
                        "claim": quote,
                        "exact_quote": quote,
                        "relation": "supports",
                        "dimension_key": "q1:d1",
                        "relevance": 0.01 if rejected else 0.99,
                        "confidence": 0.99,
                    }
                    for quote in quotes
                ]
            },
            usage=TokenUsage(
                input_tokens=100, output_tokens=100, total_tokens=200, accuracy=UsageAccuracy.EXACT
            ),
        )

    extractor = EvidenceExtractorService(
        bindings,
        _Cipher(),
        SimpleNamespace(generate_structured=AsyncMock(side_effect=generate_structured)),
        None,
        artifacts,
    )
    persistence_args = []
    for owner in range(2 if full else 1):
        text = "\n".join(quotes) + (f"Atlas系统独立来源{owner}的调查背景和原始公告。" * 12)
        page = ReadPage(
            final_url=f"https://publisher{owner}.gov/original",
            title="Atlas原始事件资料",
            clean_text=text,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            fetched_at=datetime.now(UTC),
        )
        source_id = uuid4()
        scored, usage, manifest = await extractor.extract(
            run_id,
            question=GOAL,
            page=page,
            source_id=source_id,
            acceptance_dimensions=target.acceptance_dimensions,
        )
        assert len(scored) == len(quotes)
        assert sum(e.accepted for e in scored) == (0 if rejected else len(quotes))
        uri = await artifacts.save_page(run_id, source_id, page.clean_text)
        args = dict(
            worker_task_id=TASK,
            target=target,
            source_id=source_id,
            page=page,
            artifact_uri=uri,
            evidence=scored,
            usage=usage,
            context_manifest=manifest,
        )
        await tools.record_page(run_id, **args)
        persistence_args.append(args)
    return factory, runs, tools, run_id, target, persistence_args


@pytest.mark.asyncio
@pytest.mark.parametrize("count,rejected", [(0, False), (1, False), (5, False), (5, True)])
async def test_authoritative_counter_endpoint_snapshot_duplicate_and_rollback(
    tmp_path, count, rejected
):
    factory, runs, tools, run_id, _target, args = await materialize(
        tmp_path, count=count, rejected=rejected
    )
    expected = 0 if rejected else count
    with factory._sync_factory() as session:
        row = session.get(ResearchRunRow, run_id)
        assert (
            row.usage_snapshot["accepted_evidence"]
            == row.quality_snapshot["accepted_evidence"]
            == expected
        )
    # Replaying a persistence result does not create another accepted record.
    await tools.record_page(run_id, **args[0])
    assert sum(e.accepted for e in await tools.list_evidence(OWNER, run_id)) == expected
    with pytest.raises(RuntimeError, match="rollback"):
        async with factory() as session, session.begin():
            row = await session.get(ResearchRunRow, run_id)
            for evidence in (await session.scalars(select(ResearchEvidenceRow))).all():
                evidence.accepted = False
            assert await sync_accepted_evidence_count(session, row) == 0
            raise RuntimeError("rollback")
    with factory._sync_factory() as session:
        assert session.get(ResearchRunRow, run_id).usage_snapshot["accepted_evidence"] == expected
    service = SimpleNamespace(
        get_run=runs.get,
        list_llm_calls=AsyncMock(return_value=[]),
        list_evidence=tools.list_evidence,
    )
    app = create_app(
        Settings(app_env="test", external_probes_enabled=False, langgraph_strict_msgpack=False)
    )
    app.dependency_overrides[get_client_session] = lambda: ClientSession(
        client_id=uuid4(), owner_hash=OWNER, is_new=False
    )
    app.dependency_overrides[get_research_run_service] = lambda: service
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://offline.test"
    ) as client:
        response = await client.get(f"/api/v1/research-runs/{run_id}/performance")
        assert response.status_code == 200
        assert response.json()["accepted_evidence"] == expected
        evidence = await client.get(f"/api/v1/research-runs/{run_id}/evidence")
        assert sum(e["accepted"] for e in evidence.json()) == expected
    # Even deliberately stale historical JSON must not drive the final event.
    async with factory() as session, session.begin():
        row = await session.get(ResearchRunRow, run_id)
        row.usage_snapshot = {**row.usage_snapshot, "accepted_evidence": 0}
        await tools._enter_writing(
            session, row, reason="logical_query_budget_exhausted", summary="offline"
        )
    with factory._sync_factory() as session:
        event = session.scalars(
            select(AgentEventRow).where(AgentEventRow.event_type == "report.writing_started")
        ).one()
        assert event.refs["accepted_evidence"] == expected
        assert ("尚无有效证据" in event.public_summary) is (expected == 0)
    if expected == 5:

        async def writer_response(**kwargs):
            return CanonicalModelResult(
                parsed_object=draft_for(kwargs["request"]),
                usage=TokenUsage(
                    input_tokens=400,
                    output_tokens=100,
                    total_tokens=500,
                    accuracy=UsageAccuracy.EXACT,
                ),
            )

        reports = ReportRepository(factory)
        writer = ReportWriterService(
            reports,
            _Bindings(run_id),
            _Cipher(),
            SimpleNamespace(generate_structured=AsyncMock(side_effect=writer_response)),
            ContextBudgetManager(factory),
            budget_repository=tools,
        )
        await writer.write(run_id, worker_task_id=TASK)
        with factory._sync_factory() as session:
            row = session.get(ResearchRunRow, run_id)
            assert (
                row.usage_snapshot["accepted_evidence"]
                == row.quality_snapshot["accepted_evidence"]
                == 5
            )
            final = session.scalars(
                select(AgentEventRow).where(AgentEventRow.event_type == "run.completed")
            ).one()
            assert final.refs["accepted_evidence"] == 5
        assert (await reports.get_for_run_by_id(run_id)).verification_result["verified"]


@pytest.mark.asyncio
@pytest.mark.parametrize("full", [False, True])
async def test_real_evidence_to_report_preserves_event_completeness(tmp_path, full):
    factory, _runs, tools, run_id, target, _ = await materialize(tmp_path, count=1, full=full)
    evaluation = await tools.finish_iteration(run_id, worker_task_id=TASK, target=target)
    with factory._sync_factory() as session:
        row = session.get(ResearchRunRow, run_id)
        assert row.quality_snapshot["event_coverage"]["complete"] is full
        assert (row.quality_snapshot["critical_gaps"] == 0) is full, row.quality_snapshot[
            "coverage_map"
        ]
        # Search/Reader are external deterministic boundaries in this fixture;
        # record their known successful outcomes without issuing any request.
        row.usage_snapshot = {
            **row.usage_snapshot,
            "searches": 1,
            "pages_fetched": 2 if full else 1,
        }
        session.commit()
    if not full:
        async with factory() as session, session.begin():
            row = await session.get(ResearchRunRow, run_id)
            await tools._enter_writing(session, row, reason="sources_exhausted", summary="offline")
    else:
        assert evaluation.stop_reason == "quality_met"
    requests = []

    async def generate_structured(**kwargs):
        requests.append(kwargs["request"])
        return CanonicalModelResult(
            parsed_object=draft_for(kwargs["request"]),
            usage=TokenUsage(
                input_tokens=500, output_tokens=200, total_tokens=700, accuracy=UsageAccuracy.EXACT
            ),
        )

    gateway = SimpleNamespace(generate_structured=AsyncMock(side_effect=generate_structured))
    reports = ReportRepository(factory)
    writer = ReportWriterService(
        reports,
        _Bindings(run_id),
        _Cipher(),
        gateway,
        ContextBudgetManager(factory),
        budget_repository=tools,
    )
    await writer.write(run_id, worker_task_id=TASK)
    report = await reports.get_for_run_by_id(run_id)
    assert report.verification_result["verified"] is True
    assert report.verification_result["research_complete"] is full
    assert report.verification_result["technical_e2e_success"] is True
    assert gateway.generate_structured.await_count == 1
    with factory._sync_factory() as session:
        row = session.get(ResearchRunRow, run_id)
        assert row.status == ("completed" if full else "completed_with_limitations")
        assert row.quality_snapshot["technical_e2e_success"] is True
        assert row.quality_snapshot["research_complete"] is full
        assert row.usage_snapshot["accepted_evidence"] == (8 if full else 1)
    if not full:
        sections = {s.section_key: s for s in report.sections}
        assert "正式开通" in sections["OFFICIAL_SERVICE_OPENING"].draft_markdown
        for key in ["START_OF_CONSTRUCTION", "COMPLETION", "OPERATIONAL_ENTRY"]:
            assert "当前检索未获得" in sections[key].draft_markdown


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["timeout", "crash"])
async def test_unknown_writer_reservation_is_not_refunded_by_template_report(tmp_path, mode):
    factory, _runs, tools, run_id, _target, _args = await materialize(tmp_path, count=1)
    async with factory() as session, session.begin():
        row = await session.get(ResearchRunRow, run_id)
        await tools._enter_writing(session, row, reason="sources_exhausted", summary="offline")
    gateway = SimpleNamespace(
        generate_structured=AsyncMock(
            side_effect=TimeoutError() if mode == "timeout" else asyncio.CancelledError()
        )
    )
    writer = ReportWriterService(
        ReportRepository(factory),
        _Bindings(run_id),
        _Cipher(),
        gateway,
        ContextBudgetManager(factory),
        budget_repository=tools,
    )
    if mode == "crash":
        with pytest.raises(asyncio.CancelledError):
            await writer.write(run_id, worker_task_id=TASK)
        # The held writer pool prevents a second gateway request on recovery.
        await writer.write(run_id, worker_task_id=TASK)
    else:
        await writer.write(run_id, worker_task_id=TASK)
    gateway.generate_structured.assert_awaited_once()
    with factory._sync_factory() as session:
        row = session.get(ResearchRunRow, run_id)
        held = list(
            session.scalars(
                select(ModelBudgetReservationRow).where(
                    ModelBudgetReservationRow.question_id == "__writer__"
                )
            )
        )
        assert len(held) == 1 and held[0].status == (
            "uncertain" if mode == "timeout" else "reserved"
        )
        pool = row.usage_snapshot["model_token_pools"]["writer"]
        assert pool["reserved_tokens"] + pool["uncertain_tokens"] == 4500
        assert pool["remaining_tokens"] == 0
        assert row.usage_snapshot["writer_pending_tokens"] == 4500
        assert row.usage_snapshot["writer_token_reserve_released"] == 0
        assert row.status == "completed_with_limitations"


@pytest.mark.asyncio
async def test_v2_legacy_search_reader_evidence_writer_joint_path(tmp_path, monkeypatch):
    question = "北斗三号全球卫星导航系统何时正式开通服务?"
    quote = "2020年7月31日，北斗三号全球卫星导航系统正式开通服务。"  # noqa: RUF001
    factory, artifacts, run_id, tools, extractor = await source_setup(
        tmp_path, InputGateway("valid", question, quote)
    )
    monkeypatch.setattr(web_reader, "_require_public_destination", AsyncMock())
    requests = []

    def transport(request):
        requests.append(request.url.host)
        if request.url.host == "searxng":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://agency.gov/opening",
                            "title": "北斗三号官方开通公告",
                            "content": "北斗三号正式开通服务日期和原文",
                        }
                    ],
                    "unresponsive_engines": [],
                },
            )
        assert request.url.host == "agency.gov"
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text=f"<html><body><article><p>{quote}</p><p>"
            + "北斗三号官方公告提供卫星导航服务背景和原始记录。" * 30
            + "</p></article></body></html>",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        loop = ResearchLoopService(
            tools,
            SearXNGSearchProvider(client, "http://searxng"),
            web_reader.PublicWebReader(client),
            extractor,
            artifacts,
        )
        outcome = await loop.run_one_iteration(run_id, worker_task_id=SOURCE_TASK)
    assert outcome.accepted_evidence == 1
    assert requests.count("agency.gov") == 1 and set(requests) == {"searxng", "agency.gov"}
    async with factory() as session, session.begin():
        row = await session.get(ResearchRunRow, run_id)
        await tools._enter_writing(session, row, reason="sources_exhausted", summary="offline")

    async def response(**kwargs):
        return CanonicalModelResult(
            parsed_object=draft_for(kwargs["request"]),
            usage=TokenUsage(
                input_tokens=500, output_tokens=100, total_tokens=600, accuracy=UsageAccuracy.EXACT
            ),
        )

    gateway = SimpleNamespace(generate_structured=AsyncMock(side_effect=response))
    reports = ReportRepository(factory)
    await ReportWriterService(
        reports,
        _Bindings(run_id),
        _Cipher(),
        gateway,
        ContextBudgetManager(factory),
        budget_repository=tools,
    ).write(run_id, worker_task_id=SOURCE_TASK)
    gateway.generate_structured.assert_awaited_once()
    report = await reports.get_for_run_by_id(run_id)
    assert report.verification_result["verified"] and report.citations
    assert report.verification_result["technical_e2e_success"] is True
    with factory._sync_factory() as session:
        row = session.get(ResearchRunRow, run_id)
        assert row.quality_snapshot["technical_e2e_success"]
        assert not row.quality_snapshot["research_complete"]
        assert row.status == "completed_with_limitations"
        assert row.usage_snapshot["searches"] == 1
        assert row.usage_snapshot["search_provider_requests"] == requests.count("searxng")
        assert row.usage_snapshot["pages_fetched"] == 1
        assert row.usage_snapshot["extraction_calls"] == 1
