"""Joined production scope/search/budget/validation/persistence/memory path.

Every external boundary is a deterministic double; SQLite lives in memory.
No deployed DB, credentials, public requests or real Research Run are used.
"""

import hashlib
import logging
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from app.domain.providers import CanonicalModelResult, TokenUsage, UsageAccuracy
from app.domain.research_management import allocate_budget_shares
from app.infrastructure.artifacts import LocalArtifactStore
from app.infrastructure.db.reports import ReportRepository
from app.infrastructure.db.research_models import ResearchEvidenceRow, ResearchSourceRow
from app.infrastructure.db.research_runs import ResearchRunRepository
from app.infrastructure.db.research_tools import ResearchToolRepository
from app.infrastructure.db.run_models import AgentEventRow, ResearchRunRow
from app.infrastructure.db.state_runtime import ResearchStateRuntimeRepository
from app.memory.manager import ResearchMemoryManager
from app.retrieval.models import MemorySearchDocumentRow
from app.services.evidence_extractor import EvidenceExtractorService
from app.services.planner import PlannerService
from app.services.research_loop import ResearchLoopService
from app.tools.web_reader import PublicWebReader
from app.tools.web_search import SearXNGSearchProvider
from sqlalchemy import select

from tests.integration.test_research_reliability_offline import (
    GOAL,
    QUOTE,
    TASK,
    URLS,
    Gateway,
    Reader,
)
from tests.integration.test_v3_deterministic_e2e import (
    _Bindings,
    _Cipher,
    _sqlite_utc_datetime_compatibility,  # noqa: F401 -- SQLite lease compatibility
)
from tests.unit.test_memory_projection_incremental import store as memory_store  # noqa: F401

BACKGROUND = "本来源只介绍该系统的历史背景, 没有给出正式开通日期。"


class JoinedGateway(Gateway):
    async def generate_structured(self, *, request, **kwargs):
        result = await super().generate_structured(request=request, **kwargs)
        payload = dict(result.parsed_object)
        if request.task_kind == "research_planning":
            good = payload["questions"][0]
            payload["questions"] = [
                {**good, "question": text}
                for text in (
                    "北斗三号正式开通服务的日期是哪一天?",
                    "北斗三号建成与开通的官方表述有何区别?",
                    "哪些原始来源能够交叉核验该日期?",
                    "北斗三号投入运营与开通的日期是否一致?",
                )
            ] + payload["questions"][1:]
        else:
            first = URLS[0] in "\n".join(str(part.value) for part in request.content_parts)
            quote = BACKGROUND if first else QUOTE
            payload = {
                "items": [
                    {
                        "claim": quote,
                        "exact_quote": quote,
                        "dimension_key": None,
                        "relation": "context" if first else "supports",
                        "relevance": 0.35 if first or self.mode == "all_rejected" else 0.99,
                        "confidence": 0.99,
                    }
                ]
            }
            if first and self.mode == "empty_then_accepted":
                payload = {"items": []}
        return CanonicalModelResult(
            parsed_object=payload,
            usage=TokenUsage(
                input_tokens=800, output_tokens=200, total_tokens=1000, accuracy=UsageAccuracy.EXACT
            ),
            provider_request_id="offline-only",
        )


class JoinedReader(Reader):
    def consume_events(self, url):
        return [
            PublicWebReader._event(
                "source.readable", requested_url=url, current_url=url, stage="reader"
            )
        ]

    async def read(self, url):
        page = await super().read(url)
        if url == URLS[0]:
            text = page.clean_text + BACKGROUND
            page = page.model_copy(
                update={
                    "clean_text": text,
                    "content_hash": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
        return page


@pytest.mark.parametrize(
    "mode",
    [
        "accepted",
        "empty_then_accepted",
        "only_first",
        "fallback_failed",
        "budget",
        "deadline",
        "all_rejected",
    ],
)
async def test_joined_fallback_source_loan_real_validation_and_incremental_memory(
    memory_store,  # noqa: F811 -- shared fixture registered by import
    tmp_path,
    monkeypatch,
    caplog,
    mode,
):
    caplog.set_level(logging.INFO)
    store = memory_store
    factory, run_id = store.sessions, store.run_id
    with store.sync() as session, session.begin():
        run = session.get(ResearchRunRow, run_id)
        run.original_query = run.normalized_goal = GOAL
        run.orchestrator_version = "V2_LEGACY"
        run.budget_snapshot = {
            **run.budget_snapshot,
            "max_tokens": 30000,
            "deadline_at": (datetime.now(UTC) + timedelta(seconds=180)).isoformat(),
            "max_wall_clock_seconds": 180,
            "allocation": allocate_budget_shares(
                max_tokens=30000, max_iterations=8, max_searches=6, max_pages=5
            ),
        }
    runs, tools = ResearchRunRepository(factory), ResearchToolRepository(factory)
    assert await runs.acquire_for_execution(run_id, worker_task_id=TASK)
    binding, gateway, reader = _Bindings(run_id), JoinedGateway(mode), JoinedReader()
    binding.binding.goal = GOAL
    planner = PlannerService(binding, _Cipher(), gateway, budget_repository=tools)
    plan, usage = await planner.generate(run_id, worker_task_id=TASK)
    assert len(plan.questions) == 4
    assert "reason_code=unrequested_fact_type" in caplog.text
    assert "未来升级计划" not in caplog.text
    assert await runs.save_generated_plan(run_id, worker_task_id=TASK, plan=plan, usage=usage)
    if mode in {"budget", "deadline"}:
        with store.sync() as session, session.begin():
            run = session.get(ResearchRunRow, run_id)
            if mode == "deadline":
                run.budget_snapshot = {
                    **run.budget_snapshot,
                    "deadline_at": (datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
                }
            else:
                run.budget_snapshot = {
                    **run.budget_snapshot,
                    "allocation": {
                        **run.budget_snapshot["allocation"],
                        "question_token_floors": {f"q{i}": 4000 for i in range(2, 5)},
                    },
                }
    runtime = ResearchStateRuntimeRepository(factory)
    await runtime.ensure_initialized(run_id)
    requests = []

    async def handler(request):
        requests.append(request.url.host)
        if request.url.host == "searxng":
            return httpx.Response(
                200, json={"results": [], "unresponsive_engines": [["engine", "unresponsive"]]}
            )
        body = "".join(
            f'<li class="b_algo"><h2><a href="{url}">北斗三号正式开通服务'
            "</a></h2><p>北斗三号原始公告日期</p></li>"
            for url in (URLS[:1] if mode == "only_first" else URLS)
        )
        return httpx.Response(502 if mode == "fallback_failed" else 200, text=body)

    artifacts = LocalArtifactStore(tmp_path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        search = SearXNGSearchProvider(client, "http://searxng")
        extractor = EvidenceExtractorService(binding, _Cipher(), gateway, None, artifacts)
        loop = ResearchLoopService(
            tools, search, reader, extractor, artifacts, parallel_reads_enabled=True
        )
        result = await loop.run_one_iteration(run_id, worker_task_id=TASK)
    state = await runtime.synchronize(run_id, node_name="research_iteration", worker_task_id=TASK)
    from app.retrieval import projections

    monkeypatch.setattr(
        projections, "rebuild_memory", lambda *_a, **_kw: pytest.fail("full rebuild")
    )
    await ResearchMemoryManager(factory).capture_state(state, node_name="research_iteration")
    with store.sync() as session:
        facts = list(session.scalars(select(ResearchEvidenceRow)))
        accepted = [row for row in facts if row.accepted]
        events = list(session.scalars(select(AgentEventRow).order_by(AgentEventRow.run_seq)))
        starts = [row for row in events if row.event_type == "evidence.extraction_started"]
        grants = [row for row in events if row.event_type == "budget.source_continuation_reserved"]
        if mode in {"accepted", "empty_then_accepted"}:
            assert len(starts) == 2 and len(grants) == 1, "\n".join(
                f"{row.event_type} {row.refs} {row.metrics}"
                for row in events
                if "denied" in row.event_type or "skipped" in row.event_type
            )
            assert len(accepted) == result.accepted_evidence == 1
            if mode == "accepted":
                assert facts[0].rejection_reason == "evidence_relevance_below_threshold"
            assert accepted[0].exact_quote == QUOTE
            source = session.get(ResearchSourceRow, accepted[0].source_id)
            assert source.canonical_url == URLS[1]
            assert QUOTE in await artifacts.read_text(source.artifact_uri)
            assert session.get(ResearchRunRow, run_id).usage_snapshot["searches"] == 1
        else:
            assert not accepted
            expected_starts = (
                1 if mode in {"budget", "only_first"} else 2 if mode == "all_rejected" else 0
            )
            assert len(starts) == expected_starts
            assert len(grants) == (1 if mode == "all_rejected" else 0)
        assert len(list(session.scalars(select(MemorySearchDocumentRow)))) <= 3
        usage = session.get(ResearchRunRow, run_id).usage_snapshot
        assert usage.get("search_provider_requests", 0) == len(requests)
        assert len(requests) <= 3
        assert session.get(ResearchRunRow, run_id).budget_snapshot["max_wall_clock_seconds"] == 180
    assert "memory_projection_changes" in caplog.text
    assert "stage=memory.rebuild.total" not in caplog.text
    if not accepted:
        with store.sync() as session, session.begin():
            session.get(ResearchRunRow, run_id).phase = "writing"
        await ReportRepository(factory).fail_no_evidence(run_id, worker_task_id=TASK)
        with store.sync() as session:
            failure = session.scalar(
                select(AgentEventRow).where(AgentEventRow.event_type == "run.failed")
            )
            assert failure.refs["failure_cause"] != "no_candidate_passed_provenance_validation"
            assert session.get(ResearchRunRow, run_id).status == "failed"
