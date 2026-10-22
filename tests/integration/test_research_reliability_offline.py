"""Joined production planner/loop/validation regression, with no live boundaries.

Only an in-memory SQLite fixture and deterministic model/search/reader doubles
are used. No API Run creation, credentials, service or external request occurs.
"""

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from uuid import uuid4

import pytest
from app.domain.planning_scope import PlanScopeError
from app.domain.providers import CanonicalModelResult, TokenUsage, UsageAccuracy
from app.domain.research_management import allocate_budget_shares
from app.domain.research_tools import ReadPage, SearchResult
from app.infrastructure.db.evidence_graph_models import ResearchSourceSnapshotRow
from app.infrastructure.db.model_budget_models import ModelBudgetReservationRow
from app.infrastructure.db.reports import ReportRepository
from app.infrastructure.db.research_models import ResearchEvidenceRow, ResearchSourceRow
from app.infrastructure.db.research_runs import ResearchRunRepository
from app.infrastructure.db.research_tools import ResearchToolRepository
from app.infrastructure.db.run_models import AgentEventRow, ResearchPlanItemRow, ResearchRunRow
from app.services.evidence_extractor import EvidenceExtractorService
from app.services.planner import PlannerService
from app.services.research_loop import ResearchLoopService
from sqlalchemy import select

from tests.integration.test_v3_deterministic_e2e import (
    _Bindings,
    _Cipher,
    _new_store,
    _seed_run,
    _sqlite_utc_datetime_compatibility,  # noqa: F401 (module-local fixture)
)

GOAL = "北斗三号何时正式开通服务? 请区分建成、开通与投入运营。"
QUOTE = "北斗三号于2020年7月31日正式开通服务。"
# Synthetic government-role URLs; never fetched. A date/numeric claim must
# still satisfy the production source-role gate, not an unknown blog fixture.
URLS = ("https://example.gov/first", "https://example.gov/second")
TASK = "offline-reliability-worker"


class Search:
    provider_name = "offline-only"
    last_request_count = 1
    last_healthy_response_count = 1
    last_productive_response_count = 1
    last_timeout_count = last_fallback_count = last_unresponsive_response_count = 0

    async def search(self, query, *, limit, **kwargs):
        return [
            SearchResult(
                title="北斗三号正式开通服务原始公告",
                url=url,
                snippet="核验建成、开通和投入运营的日期",
                rank=i,
            )
            for i, url in enumerate(URLS, 1)
        ][:limit]


class Reader:
    def __init__(self, *, second_short=False, redirect=False):
        self.calls = []
        self.second_short = second_short
        self.redirect = redirect

    async def read(self, url):
        self.calls.append(url)
        content = (
            "北斗三号正式开通服务资料。正文区分系统建成、服务开通和投入运营。"
            "应使用原始公告核验日期, 避免将开始建设误作正式开通。"
        ) * 8
        content += "第一份来源的历史背景。" if url == URLS[0] else QUOTE
        if self.second_short and url == URLS[1]:
            content = "北斗三号的过短正文。" * 12
        return ReadPage(
            final_url=url + "/canonical" if self.redirect else url,
            title="北斗三号正式开通服务公告",
            clean_text=content,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            fetched_at=datetime.now(UTC),
        )

    def consume_events(self, url):
        return []  # persistence still records fetch/read/extraction lifecycle


class Gateway:
    def __init__(self, mode):
        self.mode = mode
        self.calls = []

    async def generate_structured(self, *, request, **kwargs):
        self.calls.append(request)
        if request.task_kind == "research_planning":

            def q(text):
                return {
                    "question": text,
                    "priority": 1,
                    "rationale": "为用户要求的原始日期提供对应证据。",
                    "evidence_requirements": ["原文引用可追溯"],
                    "search_hints": ["北斗三号 正式开通 原始公告"],
                }

            payload = {
                "scope_summary": GOAL,
                "questions": [
                    q("北斗三号正式开通服务的日期是哪一天?"),
                    q("北斗三号的未来升级计划是什么?"),
                ],
                "completion_criteria": ["引用可追溯原文", "区分建成与开通及运营"],
            }
        else:
            assert request.task_kind in {"evidence_extraction", "evidence_extraction_compact"}
            content = json.dumps([p.value for p in request.content_parts], ensure_ascii=False)
            # First page is always empty. The second is genuine parsed model
            # input and must pass production localization/relevance/provenance.
            payload = {"items": []}
            if QUOTE in content and self.mode != "empty":
                payload = {
                    "items": [
                        {
                            "claim": QUOTE,
                            "exact_quote": QUOTE,
                            "dimension_key": None,
                            "relation": "supports",
                            "relevance": 0.99 if self.mode == "valid" else 0.01,
                            "confidence": 0.99,
                        }
                    ]
                }
        return CanonicalModelResult(
            parsed_object=payload,
            usage=TokenUsage(
                input_tokens=60, output_tokens=20, total_tokens=80, accuracy=UsageAccuracy.EXACT
            ),
            provider_request_id="offline-only",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["valid", "empty", "rejected", "short", "redirect", "existing_source"]
)
async def test_joined_scope_prefetched_continuation_and_real_validation(tmp_path, caplog, mode):
    caplog.set_level(logging.INFO)
    factory, artifacts = _new_store(tmp_path)
    run_id = uuid4()
    _seed_run(factory, run_id, GOAL)
    with factory._sync_factory() as session:
        row = session.get(ResearchRunRow, run_id)
        budget = dict(row.budget_snapshot)
        budget.update(
            {
                "max_tokens": 30000,
                "deadline_seconds": 180,
                "deadline_at": (datetime.now(UTC) + timedelta(seconds=180)).isoformat(),
                "allocation": allocate_budget_shares(
                    max_tokens=30000, max_iterations=24, max_searches=6, max_pages=10
                ),
            }
        )
        row.budget_snapshot = budget
        row.orchestrator_version = "V2_LEGACY"
        session.commit()
    runs, tools = ResearchRunRepository(factory), ResearchToolRepository(factory)
    assert await runs.acquire_for_execution(run_id, worker_task_id=TASK)
    bindings, gateway, reader = (
        _Bindings(run_id),
        Gateway("valid" if mode in {"redirect", "existing_source"} else mode),
        Reader(second_short=mode == "short", redirect=mode == "redirect"),
    )
    bindings.binding.goal = GOAL
    planner = PlannerService(bindings, _Cipher(), gateway, budget_repository=tools)
    plan, usage = await planner.generate(run_id, worker_task_id=TASK)
    assert len(plan.questions) == 1 and "未来" not in plan.questions[0].question
    bad = plan.model_copy(
        update={
            "questions": [
                plan.questions[0].model_copy(update={"question": "北斗三号的未来升级计划是什么?"})
            ]
        }
    )
    with pytest.raises(PlanScopeError, match="no_in_scope_question"):
        await runs.save_generated_plan(run_id, worker_task_id=TASK, plan=bad, usage=usage)
    with factory._sync_factory() as session:
        assert not list(session.scalars(select(ResearchPlanItemRow)))
    assert await runs.save_generated_plan(run_id, worker_task_id=TASK, plan=plan, usage=usage)
    existing_source_id = None
    if mode == "existing_source":
        prior_page = await Reader().read(URLS[1])
        existing_source_id = uuid4()
        prior_artifact = await artifacts.save_page(
            run_id, existing_source_id, prior_page.clean_text
        )
        with factory._sync_factory() as session:
            session.add(
                ResearchSourceRow(
                    id=existing_source_id,
                    run_id=run_id,
                    canonical_url=URLS[1],
                    url_hash=hashlib.sha256(URLS[1].encode()).hexdigest(),
                    domain="example.gov",
                    source_owner_key="example.gov",
                    title=prior_page.title,
                    source_type="webpage",
                    reliability=0.93,
                    artifact_uri=prior_artifact,
                    content_hash=prior_page.content_hash,
                    char_count=len(prior_page.clean_text),
                    fetched_at=prior_page.fetched_at,
                )
            )
            session.commit()
    extractor = EvidenceExtractorService(bindings, _Cipher(), gateway, None, artifacts)
    loop = ResearchLoopService(
        tools, Search(), reader, extractor, artifacts, parallel_reads_enabled=True
    )
    outcome = await loop.run_one_iteration(run_id, worker_task_id=TASK)
    with factory._sync_factory() as session:
        evidence = list(session.scalars(select(ResearchEvidenceRow)))
        accepted = [e for e in evidence if e.accepted]
        events = list(session.scalars(select(AgentEventRow).order_by(AgentEventRow.run_seq)))
        starts = [e for e in events if e.event_type == "evidence.extraction_started"]
        finishes = [e for e in events if e.event_type == "evidence.extracted"]
        assert len(starts) == len(finishes) == (1 if mode == "short" else 2)
        assert len(reader.calls) == 2 and set(reader.calls) == set(URLS)
        assert (
            len(accepted)
            == outcome.accepted_evidence
            == (1 if mode in {"valid", "redirect", "existing_source"} else 0)
        ), [(e.rejection_reason, e.source_reliability, e.evidence_score) for e in evidence]
        assert len(list(session.scalars(select(ResearchPlanItemRow)))) == 1
        assert all(a.created_at <= b.created_at for a, b in pairwise(events))
        for i, (start, finish) in enumerate(zip(starts, finishes, strict=True)):
            assert start.refs["source_id"] == finish.refs["source_id"]
            assert start.refs["candidate_id"] == finish.refs["candidate_id"]
            assert start.refs["requested_url"] == URLS[i]
        if accepted:
            snapshot = session.get(ResearchSourceSnapshotRow, accepted[0].snapshot_id)
            assert accepted[0].exact_quote == QUOTE
            assert snapshot.final_url == URLS[1] + ("/canonical" if mode == "redirect" else "")
            assert QUOTE in await artifacts.read_text(snapshot.artifact_uri)
            if existing_source_id is not None:
                assert accepted[0].source_id == existing_source_id
                assert starts[1].refs["source_id"] == str(existing_source_id)
        ledger = list(session.scalars(select(ModelBudgetReservationRow)))
        assert all(r.status == "settled" for r in ledger)
        assert sum(r.reserved_total for r in ledger) <= 30000
        assert session.get(ResearchRunRow, run_id).budget_snapshot["deadline_seconds"] == 180
    stages = {r.stage_timing["stage"] for r in caplog.records if hasattr(r, "stage_timing")}
    assert {
        "llm.planner_request",
        "search.request",
        "reader.batch",
        "llm.evidence_request",
        "evidence.validation_persistence",
    } <= stages
    if mode == "empty":
        # Exercise real zero-evidence failure semantics, without a report or a
        # fabricated pass. Only the in-memory fixture's phase is transitioned.
        with factory._sync_factory() as session:
            row = session.get(ResearchRunRow, run_id)
            row.phase = "writing"
            row.termination_reason = "deadline_exhausted"
            session.commit()
        await ReportRepository(factory).fail_no_evidence(run_id, worker_task_id=TASK)
        with factory._sync_factory() as session:
            failure = session.scalars(
                select(AgentEventRow).where(AgentEventRow.event_type == "run.failed")
            ).one()
            assert failure.refs["detail_code"] == "EVIDENCE_VALIDATION_EMPTY"
            assert failure.refs["failure_cause"] == "extraction_returned_no_candidates"
            assert failure.refs["research_stop_reason"] == "deadline_exhausted"
