"""Actual planner/HTML reader/selector/validator/SQLite persistence, all offline."""
# ruff: noqa: RUF001 -- Chinese fixture text is deliberately not normalized

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from app.context.manager import ContextBudgetManager
from app.domain.providers import CanonicalModelResult, TokenUsage, UsageAccuracy
from app.domain.research_management import allocate_budget_shares
from app.domain.research_tools import SearchResult
from app.infrastructure.db.evidence_graph_models import (
    ResearchSourceChunkRow,
    ResearchSourceSnapshotRow,
)
from app.infrastructure.db.research_models import ResearchEvidenceRow, ResearchSourceRow
from app.infrastructure.db.research_runs import ResearchRunRepository
from app.infrastructure.db.research_tools import ResearchToolRepository
from app.infrastructure.db.run_models import AgentEventRow, ResearchRunRow
from app.services.evidence_extractor import EvidenceExtractorService
from app.services.planner import PlannerService
from app.services.research_loop import ResearchLoopService
from app.tools import web_reader
from sqlalchemy import select

from tests.integration.test_research_reliability_offline import TASK, Gateway
from tests.integration.test_v3_deterministic_e2e import (
    _Bindings,
    _Cipher,
    _new_store,
    _seed_run,
    _sqlite_utc_datetime_compatibility,  # noqa: F401
)
from tests.unit.test_evidence_input_chain_repair import ANSWER, BACKGROUND, LAUNCH, QUESTION

GOAL = (
    "北斗三号全球卫星导航系统何时正式开通服务？区分开始建设、建成、正式开通和投入运营，"
    "并提供原始来源证据。"
)
PAGES = {
    "A": "最后一颗北斗三号组网卫星已运抵发射场，计划五月发射。" * 20,
    "A2": "北斗三号最后一颗组网卫星计划于五月发射，现正进行测试和总装。" * 20,
    "B": ("The agency published background about navigation signals and its mission. " * 6)
    + LAUNCH,
    "C": BACKGROUND * 150 + "\n\n" + ANSWER,
    "D": ("官方公告介绍北斗三号系统建设和服务覆盖情况。" * 20) + ANSWER,
    "E": "�" * 600,
    "F": "北斗三号原子钟性能研究，描述授时精度、频率和参数，不涉及服务日期。" * 20,
}


class Search:
    provider_name = "offline-only"
    last_request_count = last_healthy_response_count = last_productive_response_count = 1
    last_timeout_count = last_fallback_count = last_unresponsive_response_count = 0

    def __init__(self, names):
        self.names = names

    async def search(self, query, *, limit, **kwargs):
        return [
            SearchResult(
                title="北斗三号官方资料",
                url=f"https://agency.gov/{name}",
                snippet="原始公告及背景资料",
                rank=i,
            )
            for i, name in enumerate(self.names, 1)
        ][:limit]


class InputGateway(Gateway):
    def __init__(self, mode, question, quote):
        super().__init__(mode)
        self.question, self.quote = question, quote

    async def generate_structured(self, *, request, **kwargs):
        result = await super().generate_structured(request=request, **kwargs)
        if request.task_kind == "research_planning":
            payload = dict(result.parsed_object)
            payload["scope_summary"] = GOAL
            payload["questions"] = [{**payload["questions"][0], "question": self.question}]
        else:
            content = str(request.content_parts[0].value)
            assert self.quote in content, "the model must actually receive the answer passage"
            payload = {
                "items": [
                    {
                        "claim": self.quote,
                        "exact_quote": self.quote,
                        "dimension_key": None,
                        "relation": "supports",
                        "relevance": 0.99,
                        "confidence": 0.99,
                    }
                ]
            }
            if self.mode == "empty":
                payload = {"items": []}
            elif self.mode == "wrong_claim":
                payload["items"][0]["claim"] = "北斗三号于2039年1月1日正式开通服务。"
            elif self.mode == "low_relevance":
                payload["items"][0]["relevance"] = 0.6
            elif self.mode == "malformed":
                payload = {"items": [{"claim": "missing quote and scores"}]}
        return CanonicalModelResult(
            parsed_object=payload,
            usage=TokenUsage(
                input_tokens=50, output_tokens=10, total_tokens=60, accuracy=UsageAccuracy.EXACT
            ),
        )


@pytest.mark.parametrize(
    ("bad", "good", "mode"),
    [
        ("A", "C", "valid"),
        ("E", "D", "valid"),
        ("F", "C", "valid"),
        ("A", "B", "valid"),
        ("F", "D", "empty"),
        ("E", "D", "wrong_claim"),
        ("F", "D", "low_relevance"),
        ("E", "D", "malformed"),
        ("A", "A2", "only_plans"),
    ],
)
async def test_joined_reader_ranked_input_validation_persistence(
    tmp_path,
    monkeypatch,
    bad,
    good,
    mode,
):
    factory, artifacts = _new_store(tmp_path)
    run_id = uuid4()
    _seed_run(factory, run_id, GOAL)
    with factory._sync_factory() as session:
        run = session.get(ResearchRunRow, run_id)
        run.orchestrator_version = "V2_LEGACY"
        run.budget_snapshot = {
            **run.budget_snapshot,
            "max_tokens": 30000,
            "cheap_triage_enabled": True,
            "max_wall_clock_seconds": 180,
            "max_pages_extracted": 5,
            "max_extraction_calls": 5,
            "deadline_at": (datetime.now(UTC) + timedelta(seconds=180)).isoformat(),
            "allocation": allocate_budget_shares(
                max_tokens=30000,
                max_iterations=24,
                max_searches=6,
                max_pages=5,
            ),
        }
        session.commit()
    runs, tools = ResearchRunRepository(factory), ResearchToolRepository(factory)
    assert await runs.acquire_for_execution(run_id, worker_task_id=TASK)
    bindings = _Bindings(run_id)
    bindings.binding.goal = GOAL
    bindings.binding.relevant_chunks_enabled = True
    question = "最后一颗北斗三号组网卫星何时成功发射？" if good == "B" else QUESTION
    gateway = InputGateway(mode, question, LAUNCH if good == "B" else ANSWER)
    contexts = ContextBudgetManager(factory)
    planner = PlannerService(bindings, _Cipher(), gateway, contexts, budget_repository=tools)
    plan, usage = await planner.generate(run_id, worker_task_id=TASK)
    assert len(plan.questions) == 1 and plan.questions[0].question == question
    assert await runs.save_generated_plan(run_id, worker_task_id=TASK, plan=plan, usage=usage)
    monkeypatch.setattr(web_reader, "_require_public_destination", AsyncMock())
    requests = []

    def handler(request):
        name = request.url.path.rsplit("/", 1)[-1]
        requests.append(name)
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=(
                "<html><head><title>北斗三号官方资料</title></head><body><article><p>"
                + PAGES[name]
                + "</p></article></body></html>"
            ).encode(),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        reader = web_reader.PublicWebReader(client)
        extractor = EvidenceExtractorService(bindings, _Cipher(), gateway, contexts, artifacts)
        loop = ResearchLoopService(
            tools, Search((bad, good)), reader, extractor, artifacts, parallel_reads_enabled=True
        )
        await loop.run_one_iteration(run_id, worker_task_id=TASK)
    with factory._sync_factory() as session:
        evidence = list(session.scalars(select(ResearchEvidenceRow)))
        accepted = [item for item in evidence if item.accepted]
        events = list(session.scalars(select(AgentEventRow)))
        starts = [item for item in events if item.event_type == "evidence.extraction_started"]
        run = session.get(ResearchRunRow, run_id)
        expected_calls = 0 if mode == "only_plans" else 1
        assert run.usage_snapshot.get("extraction_calls", 0) == expected_calls
        assert len(starts) == expected_calls  # bad/irrelevant bodies never consume an extraction
        assert sorted(requests) == sorted((bad, good))  # no reread or live HTTP
        if mode == "valid":
            assert len(accepted) == 1, [(e.rejection_reason, e.claim) for e in evidence]
            source = session.get(ResearchSourceRow, accepted[0].source_id)
            assert source.canonical_url == f"https://agency.gov/{good}"
            assert accepted[0].exact_quote == (LAUNCH if good == "B" else ANSWER)
            assert source.artifact_uri
            snapshot = session.get(ResearchSourceSnapshotRow, accepted[0].snapshot_id)
            chunk = session.get(ResearchSourceChunkRow, accepted[0].chunk_id)
            assert snapshot.source_id == source.id and chunk.snapshot_id == snapshot.id
            original = await artifacts.read_text(snapshot.artifact_uri)
            assert original[chunk.char_start : chunk.char_end] == chunk.text
            assert accepted[0].exact_quote in chunk.text
            assert starts[0].refs["source_id"] == str(source.id)
            if good == "C":
                assert source.char_count > 80000
                assert chunk.char_start > 0.9 * source.char_count
        else:
            assert not accepted
        diagnostics = [e for e in events if e.event_type == "evidence.formation.diagnostic"]
        assert diagnostics
        if mode == "malformed":
            assert "candidate_schema_invalid" in str([e.metrics for e in diagnostics])
