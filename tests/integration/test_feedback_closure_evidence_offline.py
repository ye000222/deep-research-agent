"""Planner -> evidence rejection -> real closure feedback -> bounded recovery."""
# ruff: noqa: RUF001 -- preserve natural Chinese fixtures

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from app.context.manager import ContextBudgetManager
from app.domain.providers import CanonicalModelResult, TokenUsage, UsageAccuracy
from app.domain.research_management import allocate_budget_shares
from app.infrastructure.db.evidence_graph_models import (
    ResearchSourceChunkRow,
    ResearchSourceSnapshotRow,
)
from app.infrastructure.db.research_models import ResearchEvidenceRow
from app.infrastructure.db.research_runs import ResearchRunRepository
from app.infrastructure.db.research_tools import ResearchToolRepository
from app.infrastructure.db.run_models import AgentEventRow, ResearchRunRow
from app.services.evidence_extractor import EvidenceExtractorService
from app.services.planner import PlannerService
from app.services.research_loop import ResearchLoopService
from app.tools import web_reader
from app.tools.duckduckgo_search import DuckDuckGoAdapter
from app.tools.provider_adapters import BingAdapter, SearXNGAdapter
from app.tools.web_search import SearXNGSearchProvider
from sqlalchemy import select

from tests.integration.test_evidence_input_chain_offline import GOAL, InputGateway
from tests.integration.test_research_reliability_offline import TASK
from tests.integration.test_v3_deterministic_e2e import (
    _Bindings,
    _Cipher,
    _new_store,
    _seed_run,
    _sqlite_utc_datetime_compatibility,  # noqa: F401
)

QUESTION = "最后一颗北斗三号组网卫星何时成功发射？"
QUOTE = "On June 23, 2020, the 55th BeiDou satellite was successfully launched at 9:43 a.m."
CLAIM = "2020年6月23日9时43分，第55颗北斗卫星成功发射。"


class Gateway(InputGateway):
    def __init__(self, quote=QUOTE, claim=CLAIM, weak_first=False):
        super().__init__("valid", QUESTION, quote)
        self.claim, self.weak_first, self.extract_calls = claim, weak_first, 0

    async def generate_structured(self, *, request, **kwargs):
        result = await super().generate_structured(request=request, **kwargs)
        if request.task_kind == "research_planning":
            payload = dict(result.parsed_object)
            payload["questions"][0]["search_hints"] = [
                "BeiDou final satellite successful launch date"
            ]
            return CanonicalModelResult(parsed_object=payload, usage=result.usage)
        self.extract_calls += 1
        item = {**result.parsed_object["items"][0], "claim": self.claim}
        if self.weak_first and self.extract_calls == 1:
            item["relevance"] = 0.55
        return CanonicalModelResult(
            parsed_object={"items": [item]},
            usage=TokenUsage(
                input_tokens=50, output_tokens=10, total_tokens=60, accuracy=UsageAccuracy.EXACT
            ),
        )


async def setup(tmp_path, gateway):
    factory, artifacts = _new_store(tmp_path)
    run_id = uuid4()
    _seed_run(factory, run_id, GOAL)
    with factory._sync_factory() as session, session.begin():
        run = session.get(ResearchRunRow, run_id)
        run.orchestrator_version = "V2_LEGACY"
        run.budget_snapshot = {
            **run.budget_snapshot,
            "max_searches": 6,
            "max_logical_queries": 6,
            "max_provider_requests": 18,
            "max_tokens": 30000,
            "max_pages_extracted": 5,
            "max_extraction_calls": 5,
            "cheap_triage_enabled": True,
            "deadline_at": (datetime.now(UTC) + timedelta(seconds=180)).isoformat(),
            "allocation": allocate_budget_shares(
                max_tokens=30000, max_iterations=8, max_searches=6, max_pages=5
            ),
        }
    runs, tools = ResearchRunRepository(factory), ResearchToolRepository(factory)
    assert await runs.acquire_for_execution(run_id, worker_task_id=TASK)
    bindings = _Bindings(run_id)
    bindings.binding.goal = GOAL
    bindings.binding.relevant_chunks_enabled = True
    contexts = ContextBudgetManager(factory)
    planner = PlannerService(bindings, _Cipher(), gateway, contexts, budget_repository=tools)
    plan, model_usage = await planner.generate(run_id, worker_task_id=TASK)
    assert await runs.save_generated_plan(run_id, worker_task_id=TASK, plan=plan, usage=model_usage)
    extractor = EvidenceExtractorService(bindings, _Cipher(), gateway, contexts, artifacts)
    return factory, artifacts, run_id, tools, extractor


@pytest.mark.parametrize(
    ("include_year", "claim_year"), [(False, True), (True, True), (False, False)]
)
async def test_translated_candidate_uses_only_quote_facts_and_real_provenance(
    tmp_path, monkeypatch, include_year, claim_year
):
    quote = QUOTE if include_year else QUOTE.replace(", 2020", "")
    claim = CLAIM if claim_year else CLAIM.replace("2020年", "")
    expected_acceptance = include_year or not claim_year
    gateway = Gateway(quote=quote, claim=claim)
    factory, artifacts, run_id, tools, extractor = await setup(tmp_path, gateway)
    monkeypatch.setattr(web_reader, "_require_public_destination", AsyncMock())

    def handler(request):
        if request.url.host == "searxng":
            return httpx.Response(
                200,
                json={
                    "unresponsive_engines": [],
                    "results": [
                        {
                            "title": "BeiDou final satellite successful launch",
                            "url": "https://agency.gov/2020/launch",
                            "content": "Official satellite successful launch date",
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text='<html><head><meta name="date" content="2020-06-23"></head><body><article><p>'
            + quote
            + "</p><p>Archive publication year 2020. </p><p>"
            + "BeiDou navigation mission background. " * 20
            + "</p></article></body></html>",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SearXNGSearchProvider(client, "http://searxng")
        loop = ResearchLoopService(
            tools, provider, web_reader.PublicWebReader(client), extractor, artifacts
        )
        await loop.run_one_iteration(run_id, worker_task_id=TASK)
    with factory._sync_factory() as session:
        evidence = list(session.scalars(select(ResearchEvidenceRow)))
        assert len(evidence) == 1, [
            (e.event_type, e.refs, e.metrics)
            for e in session.scalars(select(AgentEventRow))
            if e.event_type
            in {
                "source.triage_rejected",
                "source.fetch_failed",
                "search.dispatch.blocked",
                "tool.failed",
                "budget.question_borrow_denied",
                "evidence.selection_skipped",
            }
        ]
        item = evidence[0]
        assert item.accepted == expected_acceptance
        if expected_acceptance:
            snapshot = session.get(ResearchSourceSnapshotRow, item.snapshot_id)
            chunk = session.get(ResearchSourceChunkRow, item.chunk_id)
            canonical = await artifacts.read_text(snapshot.artifact_uri)
            assert canonical[chunk.char_start : chunk.char_end] == chunk.text
            assert item.exact_quote in chunk.text
            assert item.claim == claim
        else:
            assert item.rejection_reason == "claim_quote_entailment_failed"
        events = list(session.scalars(select(AgentEventRow)))
        facts = [
            e.metrics["facts"]
            for e in events
            if e.event_type == "evidence.formation.diagnostic" and e.refs["stage"] == "ACCEPTANCE"
        ]
        assert len(facts) == 1 and "normalized_claim_signals" in facts[0]
        if not expected_acceptance:
            assert "year:2020" in facts[0]["unmatched_required_facts"]


async def test_actual_closure_feedback_stays_on_topic_and_sixth_search_can_fallback(
    tmp_path, monkeypatch
):
    gateway = Gateway(weak_first=True)
    factory, artifacts, run_id, tools, extractor = await setup(tmp_path, gateway)
    monkeypatch.setattr(web_reader, "_require_public_destination", AsyncMock())
    monkeypatch.setattr(tools, "_candidate_providers", lambda: ("SearXNG", "Bing", "DuckDuckGo"))
    requests = []
    recovery = False

    def handler(request):
        requests.append((request.url.host, str(request.url)))
        if request.url.host == "searxng":
            return httpx.Response(
                200,
                json={
                    "results": []
                    if recovery
                    else [
                        {
                            "title": "BeiDou final satellite successful launch",
                            "url": "https://agency.gov/first",
                            "content": "BeiDou final satellite launch date",
                        }
                    ],
                    "unresponsive_engines": [["engine", "unresponsive"]] if recovery else [],
                },
            )
        if request.url.host == "www.bing.com":
            return httpx.Response(200, text="<html>empty</html>")
        if request.url.host == "html.duckduckgo.com":
            return httpx.Response(
                200,
                text=(
                    '<a class="result__a" href="https://agency.gov/recovery">BeiDou launch</a>'
                    '<a class="result__snippet">Official BeiDou final satellite launch date.</a>'
                ),
            )
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            text="<html><body><article><p>Official statement "
            + request.url.path
            + "</p><p>"
            + QUOTE
            + "</p><p>"
            + "BeiDou navigation mission background. " * 20
            + "</p></article></body></html>",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = SearXNGSearchProvider(client, "http://searxng")
        loop = ResearchLoopService(
            tools,
            provider,
            web_reader.PublicWebReader(client),
            extractor,
            artifacts,
            search_adapters={
                "SearXNG": SearXNGAdapter(provider),
                "Bing": BingAdapter(provider),
                "DuckDuckGo": DuckDuckGoAdapter(client),
            },
        )
        await loop.run_one_iteration(run_id, worker_task_id=TASK)
        with factory._sync_factory() as session, session.begin():
            events = list(session.scalars(select(AgentEventRow)))
            assert any(e.event_type == "feedback.query.execution.started" for e in events), [
                (e.event_type, e.refs, e.metrics)
                for e in events
                if e.event_type
                in {
                    "source.triage_rejected",
                    "source.fetch_failed",
                    "tool.failed",
                    "research.query_candidate.generated",
                    "evidence.selection_skipped",
                }
            ]
            run = session.get(ResearchRunRow, run_id)
            # Four previous opportunities are synthetic preconsumed fixture state.
            run.usage_snapshot = {**run.usage_snapshot, "searches": 5, "logical_queries": 5}
        recovery = True
        target = await tools.prepare_target(run_id, worker_task_id=TASK)
        assert target.feedback_id is not None
        assert "北斗三号" in target.query and "发射" in target.query and "q1:d" not in target.query
        # The real feedback target uses the existing fair HTTP allocator.
        assert target.provider_request_allowance == 3
        await loop.execute_target(run_id, worker_task_id=TASK, target=target)
        before = len(requests)
        blocked = await loop.execute_target(
            run_id,
            worker_task_id=TASK,
            target=replace(target, tool_call_id=uuid4(), query_execution_id=uuid4()),
        )
        assert blocked.outcome == "search_budget_exhausted" and len(requests) == before
    with factory._sync_factory() as session:
        run = session.get(ResearchRunRow, run_id)
        assert run.usage_snapshot["logical_queries"] == 6
        events = list(session.scalars(select(AgentEventRow)))
        assert any(e.event_type == "feedback.query.anchored" for e in events)
        accepted = list(
            session.scalars(
                select(ResearchEvidenceRow).where(ResearchEvidenceRow.accepted.is_(True))
            )
        )
        assert accepted and accepted[0].claim == CLAIM, [
            (e.event_type, e.refs, e.metrics)
            for e in events
            if e.event_type
            in {
                "source.triage_rejected",
                "evidence.selection_skipped",
                "tool.failed",
                "search.dispatch.blocked",
                "evidence.extracted",
            }
        ]
    hosts = [host for host, _ in requests]
    assert "www.bing.com" in hosts and "html.duckduckgo.com" in hosts
