"""Transactional store for one bounded Web research iteration."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TypedDict, cast
from urllib.parse import urlsplit
from uuid import UUID

from sqlalchemy import and_, distinct, exists, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from app.domain.adaptive_scheduler import (
    QUERY_FAMILIES,
    ActionCost,
    QueryFamily,
    build_family_query,
    classify_source_role,
    expected_utility,
    infer_claim_type,
    normalized_gain,
    numeric_scope_consistent,
    query_family_for_attempt,
    source_role_fits_claim,
)
from app.domain.evaluation import EvaluationScope, EvaluationSnapshot, EvaluationVerdict
from app.domain.evidence_graph import (
    EvidenceGraphClaimEdgeView,
    EvidenceGraphClaimNode,
    EvidenceGraphConflictView,
    EvidenceGraphEvidenceRef,
    EvidenceGraphView,
    build_evidence_chunk,
    claim_fingerprint,
    derive_claim_status,
)
from app.domain.identifiers import uuid7
from app.domain.providers import TokenUsage, UsageAccuracy
from app.domain.research_budget import (
    allocate_research_call,
    build_resource_pool_snapshot,
    classify_claim_risk,
    classify_gap_risk,
    classify_question_risk,
    decide_question_borrow,
    estimate_writer_reserve_tokens,
    model_token_pool_for_node,
    model_token_pool_limits,
)
from app.domain.research_management import ResearchFactCounts, calculate_information_gain
from app.domain.research_runs import EXECUTION_LEASE_SECONDS, RunPhase, RunStatus
from app.domain.research_tools import (
    EvidenceView,
    ReadPage,
    ReusablePageRef,
    ScoredEvidence,
    SearchResult,
)
from app.domain.source_policy import normalize_source_url, source_owner_key
from app.infrastructure.db.evaluation_models import EvaluationSnapshotRow
from app.infrastructure.db.evidence_graph_models import (
    ResearchClaimEdgeRow,
    ResearchClaimRow,
    ResearchConflictRow,
    ResearchSourceChunkRow,
    ResearchSourceSnapshotRow,
)
from app.infrastructure.db.evidence_graph_relations import (
    RelationRefreshStats,
    refresh_question_relations,
)
from app.infrastructure.db.model_budget_models import ModelBudgetReservationRow
from app.infrastructure.db.research_models import (
    ResearchEvidenceRow,
    ResearchGapRow,
    ResearchSourceRow,
    ResearchToolCallRow,
    SearchQueryRow,
    SearchResultRow,
)
from app.infrastructure.db.research_runs import ResearchRunNotFoundError
from app.infrastructure.db.run_models import AgentEventRow, ResearchPlanItemRow, ResearchRunRow

_SEARCH_QUERY_SEPARATOR_RE = re.compile(r"[^\w\u3400-\u9fff]+", re.UNICODE)


def normalize_search_query(value: str) -> str:
    """Canonicalize query spelling before duplicate and idempotency checks."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = _SEARCH_QUERY_SEPARATOR_RE.sub(" ", normalized)
    return " ".join(normalized.split())


def search_query_duplicate_key(question_id: str, query: str) -> str:
    """Build a run-local tool idempotency key independent of plan version."""

    normalized_query = normalize_search_query(query)
    return hashlib.sha256(f"{question_id}:{normalized_query}".encode()).hexdigest()


class ResearchLeaseLostError(RuntimeError):
    pass


# A gap retry is deliberately cheaper than a productive research round when
# the candidate page is unreadable or yields no evidence.  Three attempts were
# too aggressive: a run could exhaust every question's gap counter while most
# search/page/token budget was still available.  This value now controls the
# deterministic query-variant cycle only; real resource budgets (not a gap
# counter) decide when research is terminal.
# Four explicit QueryFamily actions replace the old implicit suffix cycle.
_MAX_GAP_ATTEMPTS = 4
# Allow several genuinely different gap-resolution passes.  Query hashes remain
# run-scoped, so this is bounded retry headroom rather than permission to replay
# the same search.  Eight passes was too small once a polluted hint consumed a
# family; after hint compaction, twelve gives both P1 corroboration and vendor
# product gaps a fair recovery window.
_MAX_REPLANS = 12
_LOW_INFORMATION_GAIN_THRESHOLD = 0.10
_LOW_INFORMATION_GAIN_STREAK_TO_STOP = 2
_MAX_EVIDENCE_MODEL_CALL_TOKENS = 12_000
_MODEL_RESERVATION_LEASE = timedelta(minutes=15)
# A final evidence pass must remain possible even when the run is close to its
# model budget.  This is a *guard threshold*, not the extractor output size:
# the extractor still applies its own schema/output limits and will use its
# compact contract when the available call budget is small.
_MIN_EVIDENCE_MODEL_CALL_TOKENS = 3_000
_MAX_REPORT_WRITER_RESERVE_TOKENS = 15_000
# SearXNG may fan one logical query out to several engine groups. A per-target
# cap prevents one empty/unstable query from consuming the entire run-wide
# provider pool before untouched questions receive their first search.
_MAX_PROVIDER_REQUESTS_PER_QUERY = 3
_GLOBAL_TRIAGE_FAILURE_REASONS = frozenset(
    {
        "body_too_short",
        "login_or_navigation_page",
        "prompt_injection_detected",
    }
)
_NON_DETERMINISTIC_PAGE_FAILURE_SUFFIXES = (
    "_DNS_FAILED",
    "_NETWORK_ERROR",
    "_PROVIDER_UNAVAILABLE",
    "_TIMEOUT",
)
_FAILED_OWNER_EXCLUSION_THRESHOLD = 2
_NON_CONSUMING_ITERATION_OUTCOMES = frozenset(
    {
        "yield_question",
        "question_budget_exhausted",
        "stop_run",
        "insufficient_question_budget",
        "zero_results",
        "unreadable",
        "no_evidence",
        "provider_error",
        "budget_exhausted",
    }
)
_SEARCH_ACQUISITION_BUDGET_REASONS = frozenset(
    {
        "search_budget_exhausted",
        "logical_query_budget_exhausted",
        "provider_request_budget_exhausted",
        "page_fetch_attempt_budget_exhausted",
    }
)


def _search_acquisition_budget_exhausted(reason: str | None) -> bool:
    """Return whether new searches are blocked while cached work may continue."""

    return reason in _SEARCH_ACQUISITION_BUDGET_REASONS


def _fair_provider_request_allowance(
    remaining_requests: int,
    unattempted_questions: int,
    *,
    per_query_cap: int = _MAX_PROVIDER_REQUESTS_PER_QUERY,
) -> int:
    """Share remaining upstream capacity across untouched questions first."""

    remaining = max(0, int(remaining_requests))
    if remaining == 0:
        return 0
    untouched = max(0, int(unattempted_questions))
    fair_share = (
        max(1, remaining // untouched)
        if untouched
        else remaining
    )
    return min(remaining, max(1, int(per_query_cap)), fair_share)


def _source_failure_feedback(
    events: Iterable[tuple[str, Mapping[str, object]]],
) -> tuple[frozenset[str], tuple[str, ...]]:
    """Return run-wide failed URLs and repeatedly unusable source owners.

    A URL that already failed reading should not consume another page slot in
    the same run.  Domain-level exclusion is deliberately more conservative:
    only deterministic failures count and an owner must fail twice before it
    is removed from later search results.  Topic mismatch remains
    question-local because the same page may answer another plan item.
    """

    failed_urls: set[str] = set()
    deterministic_owner_failures: dict[str, int] = {}
    for event_type, refs in events:
        primary_url = refs.get("url")
        if not isinstance(primary_url, str) or not primary_url.strip():
            continue
        if event_type == "source.triage_rejected":
            reason = str(refs.get("reason") or "")
            if reason not in _GLOBAL_TRIAGE_FAILURE_REASONS:
                continue
            deterministic = True
        elif event_type == "source.rejected":
            error_code = str(refs.get("error_code") or "")
            deterministic = not error_code.endswith(
                _NON_DETERMINISTIC_PAGE_FAILURE_SUFFIXES
            )
        else:
            continue
        failed_urls.add(normalize_source_url(primary_url))
        requested_url = refs.get("requested_url")
        if isinstance(requested_url, str) and requested_url.strip():
            failed_urls.add(normalize_source_url(requested_url))
        if not deterministic:
            continue
        owner = source_owner_key(primary_url)
        if owner != "unknown":
            deterministic_owner_failures[owner] = (
                deterministic_owner_failures.get(owner, 0) + 1
            )
    excluded_owners = tuple(
        sorted(
            owner
            for owner, count in deterministic_owner_failures.items()
            if count >= _FAILED_OWNER_EXCLUSION_THRESHOLD
        )
    )
    return frozenset(failed_urls), excluded_owners


def _research_attempt_consumes_iteration(
    attempt_outcome: str,
    *,
    technical_outcome: bool = False,
) -> bool:
    """Return whether an attempt represents a billable research round.

    Search outages, unreadable/empty candidates and budget yields are retained
    for diagnosis, but they are not useful research rounds. Their own bounded
    retry counters and search/page budgets remain the termination controls.
    """

    return not technical_outcome and attempt_outcome not in _NON_CONSUMING_ITERATION_OUTCOMES


@dataclass(frozen=True, slots=True)
class ResearchTarget:
    plan_version: int
    question_id: str
    question: str
    query: str
    gap_id: UUID
    tool_call_id: UUID
    source_id_seed: UUID
    alternate_query: str = ""
    priority: int = 2
    attempt_index: int = 0
    gap_attempt_index: int = 0
    acceptance_dimensions: tuple[tuple[str, str], ...] = ()
    used_source_owner_keys: tuple[str, ...] = ()
    search_excluded_owner_keys: tuple[str, ...] = ()
    search_excluded_urls: tuple[str, ...] = ()
    reusable_results: tuple[SearchResult, ...] = ()
    reusable_pages: tuple[ReusablePageRef, ...] = ()
    first_pass: bool = False
    query_already_executed: bool = False
    search_budget_exhausted: bool = False
    query_family: str = QueryFamily.SCOPE.value
    provider_request_allowance: int = 0
    baseline_model_tokens: int = 0
    baseline_pages_fetched: int = 0
    baseline_pages_extracted: int = 0
    baseline_provider_requests: int = 0
    action_started_at: datetime | None = None
    deadline_at: datetime | None = None
    cheap_triage_enabled: bool = False
    owner_acceptance_rates: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceModelBudget:
    allowed: bool
    max_call_tokens: int
    remaining_tokens: int
    writer_reserve_tokens: int
    outcome: str = "execute"


@dataclass(frozen=True, slots=True)
class PageBudgetReservation:
    """The number of page-read slots granted by the locked run budget."""

    granted: int
    remaining: int


@dataclass(frozen=True, slots=True)
class ModelTokenReservation:
    """Outcome of an atomic per-attempt model token reservation."""

    granted: bool
    reserved_total: int
    status: str
    reason: str = ""


@dataclass(frozen=True, slots=True)
class IterationEvaluation:
    continue_research: bool
    decision: str
    stop_reason: str | None
    question_status: str
    coverage: float = 0.0
    information_gain: float = 0.0
    low_information_gain_streak: int = 0


class _CoverageMapEntry(TypedDict):
    dimension_key: str
    question: str
    priority: int
    coverage: float
    accepted_evidence: int
    independent_sources: int
    acceptance_criteria: list[str]
    requirement_statuses: list[dict[str, object]]
    missing_reasons: list[str]


class ResearchToolRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def evidence_model_budget(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        question_id: str | None = None,
        minimum_call: int | None = None,
    ) -> EvidenceModelBudget:
        """Apply the run-level pre-call guard while preserving report-writing capacity.

        ``minimum_call`` is the cheapest provably-assemblable single-call cost
        (see EvidenceExtractorService.estimate_minimum_request_tokens). When a
        caller does not supply one, the legacy 3,000-token guard threshold is
        used for process-unaware callers.
        """

        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            maximum = int(run.budget_snapshot.get("max_tokens", 0) or 0)
            used = _model_tokens(run)
            outstanding = await self._outstanding_reservation_total(session, run_id)
            remaining = max(0, maximum - used - outstanding)
            # Keep a small, explicit floor for every *other* unfinished plan
            # item.  Previously the guard only reserved the writer budget, so
            # a large evidence call could consume the last useful tokens and
            # terminate around 80--85% with one or two questions untouched.
            search_attempts = (
                select(
                    SearchQueryRow.run_id,
                    SearchQueryRow.question_id,
                    func.count(SearchQueryRow.id).label("attempts"),
                )
                .where(SearchQueryRow.run_id == run_id)
                .group_by(SearchQueryRow.run_id, SearchQueryRow.question_id)
                .subquery()
            )
            eligible = (
                await session.execute(
                    select(
                        ResearchPlanItemRow.question_id,
                        func.coalesce(search_attempts.c.attempts, 0),
                    )
                    .outerjoin(
                        search_attempts,
                        (search_attempts.c.question_id == ResearchPlanItemRow.question_id)
                        & (search_attempts.c.run_id == ResearchPlanItemRow.run_id),
                    )
                    .where(
                        ResearchPlanItemRow.run_id == run_id,
                        ResearchPlanItemRow.plan_version == run.plan_version,
                        ResearchPlanItemRow.status.in_(
                            ("pending", "active", "blocked", "partial", "technical_retry")
                        ),
                    )
                )
            ).all()
            pending_questions = len(eligible)
            estimated_writer_reserve = (
                estimate_writer_reserve_tokens(
                    question_count=pending_questions,
                    # Writer selects up to 20 evidence cards.  Reserving one
                    # card here underestimates the real payload and causes the
                    # final Writer call to be skipped after research stops.
                    evidence_count=20
                    if int(run.usage_snapshot.get("accepted_evidence", 0) or 0) > 0
                    else 0,
                    maximum=_MAX_REPORT_WRITER_RESERVE_TOKENS,
                )
                if maximum > 0
                else 0
            )
            allocation = run.budget_snapshot.get("allocation", {})
            allocation = allocation if isinstance(allocation, dict) else {}
            pool_limits = model_token_pool_limits(allocation)
            writer_reserve = (
                pool_limits["writer"]
                or estimated_writer_reserve
            )
            verification_reserve = _as_int(allocation.get("verification_tokens", 0))
            safety_reserve = _as_int(allocation.get("safety_tokens", 0))
            global_research_available = (
                max(
                    0,
                    remaining - writer_reserve - verification_reserve - safety_reserve,
                )
                if maximum > 0
                else _MAX_EVIDENCE_MODEL_CALL_TOKENS * max(1, pending_questions)
            )
            pool_committed = await self._model_pool_committed(session, run_id)
            research_pool_limit = pool_limits["research"]
            research_available = (
                min(
                    global_research_available,
                    max(0, research_pool_limit - pool_committed["research"]),
                )
                if research_pool_limit > 0
                else global_research_available
            )
            maximum_call = min(
                _MAX_EVIDENCE_MODEL_CALL_TOKENS,
                max(
                    _MIN_EVIDENCE_MODEL_CALL_TOKENS,
                    _as_int(
                        run.budget_snapshot.get(
                            "max_evidence_call_tokens", _MAX_EVIDENCE_MODEL_CALL_TOKENS
                        )
                    ),
                ),
            )
            effective_minimum = (
                _MIN_EVIDENCE_MODEL_CALL_TOKENS
                if minimum_call is None
                else max(1, int(minimum_call))
            )
            if effective_minimum > maximum_call:
                return EvidenceModelBudget(
                    allowed=False,
                    max_call_tokens=0,
                    remaining_tokens=remaining,
                    writer_reserve_tokens=writer_reserve,
                    outcome="stop_run",
                )
            decision = allocate_research_call(
                available=research_available,
                eligible_questions=pending_questions,
                current_attempts=next((int(n) for q, n in eligible if q == question_id), 0),
                # Fairness is decided by prepare_target()'s stable question
                # schedule. The model-budget guard must never turn a local
                # yield into a persistent question exhaustion state.
                other_unattempted=0,
                minimum_call=effective_minimum,
                maximum_call=maximum_call,
            )
            usage = dict(run.usage_snapshot)
            usage["model_tokens"] = used
            usage["model_tokens_remaining"] = remaining
            usage["model_tokens_reserved"] = outstanding
            usage["writer_token_reserve"] = writer_reserve
            usage["verification_token_reserve"] = verification_reserve
            usage["safety_token_reserve"] = safety_reserve
            usage["pending_questions"] = pending_questions
            run.usage_snapshot = _usage_with_resource_pools(run, usage)
            allowed = decision.outcome == "execute"
            max_call_tokens = decision.call_tokens
            future_question_reserve = decision.future_reserve
            if decision.outcome == "stop_run" and not bool(
                run.usage_snapshot.get("model_budget_guarded")
            ):
                usage = dict(run.usage_snapshot)
                usage["model_tokens"] = used
                usage["model_budget_guarded"] = True
                usage["model_tokens_remaining"] = remaining
                usage["writer_token_reserve"] = writer_reserve
                run.usage_snapshot = _usage_with_resource_pools(run, usage)
                await self._append_event(
                    session,
                    run,
                    event_type="budget.updated",
                    public_summary=(
                        "模型 Token 预算已触发调用前保护; 停止新增证据抽取并保留报告生成额度。"
                    ),
                    refs={"reason": "model_token_pre_call_guard"},
                    metrics={
                        "model_tokens": used,
                        "max_model_tokens": maximum,
                        "remaining_tokens": remaining,
                        "writer_reserve_tokens": writer_reserve,
                        "verification_reserve_tokens": verification_reserve,
                        "safety_reserve_tokens": safety_reserve,
                        "pending_questions": pending_questions,
                        "future_question_reserve_tokens": future_question_reserve,
                    },
                )
            return EvidenceModelBudget(
                allowed=allowed,
                max_call_tokens=max_call_tokens,
                remaining_tokens=remaining,
                writer_reserve_tokens=writer_reserve,
                outcome=decision.outcome,
            )

    async def deadline_remaining_seconds(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
    ) -> float | None:
        """Read the run deadline for control-plane model calls."""

        async with self._sessions() as session:
            run = await self._locked_run(session, run_id, worker_task_id)
            deadline = _deadline_at(run)
            if deadline is None:
                return None
            return max(0.0, (deadline - datetime.now(UTC)).total_seconds())

    async def reserve_page_slots(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        requested: int,
        fresh_search: bool = True,
    ) -> PageBudgetReservation:
        """Atomically reserve page slots before any external read is started.

        A research iteration may inspect several candidates, so checking the page
        budget only once at iteration start can overshoot the limit. The reservation
        is kept in the run usage snapshot and consumed by record_page,
        record_extraction_failure, or record_page_failure.
        """

        requested = max(0, int(requested))
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            maximum = int(
                run.budget_snapshot.get(
                    "max_pages_fetched", run.budget_snapshot.get("max_pages", 0)
                )
                or 0
            )
            usage = dict(run.usage_snapshot)
            committed = max(0, _as_int(usage.get("pages_fetched", usage.get("pages", 0))))
            committed_attempts = max(
                0, _as_int(usage.get("page_fetch_attempts", committed))
            )
            reserved = max(0, _as_int(usage.get("page_slots_reserved", 0)))
            remaining = max(0, maximum - committed - reserved)
            attempt_maximum = int(
                run.budget_snapshot.get("max_page_fetch_attempts", maximum) or maximum
            )
            attempt_remaining = max(0, attempt_maximum - committed_attempts - reserved)
            # A fresh search has already been counted by record_search_results
            # before this reservation is requested. Keep one page slot for
            # every still-available fresh search, so the page budget cannot
            # terminate the run while the search budget visibly remains.
            maximum_searches = int(run.budget_snapshot.get("max_searches", 0) or 0)
            used_searches = max(0, _as_int(usage.get("searches", 0)))
            remaining_fresh_searches = (
                max(0, maximum_searches - used_searches) if fresh_search else 0
            )
            current_search_capacity = max(0, remaining - remaining_fresh_searches)
            granted = min(requested, current_search_capacity, attempt_remaining)
            if granted:
                usage["page_slots_reserved"] = reserved + granted
                owners_raw = usage.get("page_slots_reserved_by_worker", {})
                owners = dict(owners_raw) if isinstance(owners_raw, dict) else {}
                owners[worker_task_id] = _as_int(owners.get(worker_task_id, 0)) + granted
                usage["page_slots_reserved_by_worker"] = owners
                run.usage_snapshot = _usage_with_resource_pools(run, usage)
            return PageBudgetReservation(granted=granted, remaining=remaining - granted)

    async def record_page_fetched(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
        url: str,
        reused: bool,
        latency_ms: int,
    ) -> None:
        """Settle one fetch slot before cheap triage or model extraction."""

        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            usage = dict(run.usage_snapshot)
            owners_raw = usage.get("page_slots_reserved_by_worker", {})
            owners = dict(owners_raw) if isinstance(owners_raw, dict) else {}
            owned = _as_int(owners.get(worker_task_id, 0))
            settled = 1 if not owners else min(1, max(0, owned))
            if owners:
                owners[worker_task_id] = max(0, owned - settled)
                if owners[worker_task_id] == 0:
                    owners.pop(worker_task_id, None)
                usage["page_slots_reserved_by_worker"] = owners
            usage["page_slots_reserved"] = max(
                0, _as_int(usage.get("page_slots_reserved", 0)) - settled
            )
            usage["page_fetch_attempts"] = _as_int(usage.get("page_fetch_attempts", 0)) + int(
                not reused
            )
            if not reused:
                usage["pages_fetched"] = (
                    _as_int(usage.get("pages_fetched", usage.get("pages", 0))) + 1
                )
            usage["page_fetch_latency_ms"] = _as_int(usage.get("page_fetch_latency_ms", 0)) + max(
                0, latency_ms
            )
            run.usage_snapshot = _usage_with_resource_pools(run, usage)
            await self._append_event(
                session,
                run,
                event_type="source.fetched" if not reused else "source.fetch_reused",
                public_summary=(
                    "页面抓取完成并进入廉价预筛。"
                    if not reused
                    else "复用已抓取页面并进入廉价预筛。"
                ),
                refs={"question_id": target.question_id, "url": url[:1000]},
                metrics={"reused": reused, "latency_ms": max(0, latency_ms)},
            )

    async def reserve_extraction_slot(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
    ) -> bool:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            budget = run.budget_snapshot
            usage = dict(run.usage_snapshot)
            maximum = _as_int(budget.get("max_pages_extracted", budget.get("max_pages", 0)))
            max_calls = _as_int(budget.get("max_extraction_calls", maximum))
            extracted = _as_int(usage.get("pages_extracted", usage.get("pages", 0)))
            calls = _as_int(usage.get("extraction_calls", extracted))
            reserved = _as_int(usage.get("extraction_slots_reserved", 0))
            if (
                (
                    _as_int(budget.get("max_wall_clock_seconds", 0)) > 0
                    and _deadline_remaining_seconds(run) <= 0
                )
                or (maximum > 0 and extracted + reserved >= maximum)
                or (max_calls > 0 and calls + reserved >= max_calls)
            ):
                return False
            usage["extraction_slots_reserved"] = reserved + 1
            run.usage_snapshot = _usage_with_resource_pools(run, usage)
            return True

    async def release_extraction_slot(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
    ) -> None:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            usage = dict(run.usage_snapshot)
            usage["extraction_slots_reserved"] = max(
                0, _as_int(usage.get("extraction_slots_reserved", 0)) - 1
            )
            run.usage_snapshot = _usage_with_resource_pools(run, usage)

    async def record_triage_rejection(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
        url: str,
        score: float,
        reason: str,
        source_role: str,
        requested_url: str | None = None,
    ) -> None:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            usage = dict(run.usage_snapshot)
            usage["cheap_triage_rejections"] = _as_int(usage.get("cheap_triage_rejections", 0)) + 1
            source_stats_raw = usage.get("source_type_stats", {})
            source_stats = dict(source_stats_raw) if isinstance(source_stats_raw, dict) else {}
            role_raw = source_stats.get(source_role, {})
            role = dict(role_raw) if isinstance(role_raw, dict) else {}
            role["triaged"] = _as_int(role.get("triaged", 0)) + 1
            role["rejected"] = _as_int(role.get("rejected", 0)) + 1
            source_stats[source_role] = role
            usage["source_type_stats"] = source_stats
            owner = source_owner_key(url)
            owner_stats_raw = usage.get("owner_stats", {})
            owner_stats = dict(owner_stats_raw) if isinstance(owner_stats_raw, dict) else {}
            owner_raw = owner_stats.get(owner, {})
            owner_metrics = dict(owner_raw) if isinstance(owner_raw, dict) else {}
            owner_metrics["triaged"] = _as_int(owner_metrics.get("triaged", 0)) + 1
            owner_metrics["rejected"] = _as_int(owner_metrics.get("rejected", 0)) + 1
            owner_stats[owner] = owner_metrics
            usage["owner_stats"] = owner_stats
            run.usage_snapshot = _usage_with_resource_pools(run, usage)
            await self._append_event(
                session,
                run,
                event_type="source.triage_rejected",
                public_summary="页面未通过廉价证据预筛, 未调用 Evidence Extractor。",
                refs={
                    "question_id": target.question_id,
                    "url": url[:1000],
                    **(
                        {"requested_url": requested_url[:1000]}
                        if requested_url and requested_url != url
                        else {}
                    ),
                    "reason": reason[:100],
                    "source_role": source_role[:50],
                },
                metrics={"triage_score": round(max(0.0, min(1.0, score)), 4)},
            )

    async def release_page_slots(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        count: int,
    ) -> None:
        """Release unused reservations when an iteration stops early."""

        count = max(0, int(count))
        if count == 0:
            return
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            usage = dict(run.usage_snapshot)
            reserved = max(0, _as_int(usage.get("page_slots_reserved", 0)))
            owners_raw = usage.get("page_slots_reserved_by_worker", {})
            owners = dict(owners_raw) if isinstance(owners_raw, dict) else {}
            if owners:
                owned = max(0, _as_int(owners.get(worker_task_id, 0)))
                released = min(count, owned)
                owners[worker_task_id] = owned - released
                if owners[worker_task_id] == 0:
                    owners.pop(worker_task_id, None)
                usage["page_slots_reserved_by_worker"] = owners
            else:
                released = min(count, reserved)
            usage["page_slots_reserved"] = max(0, reserved - released)
            run.usage_snapshot = _usage_with_resource_pools(run, usage)

    async def reserve_model_tokens(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        question_id: str,
        node: str,
        attempt_id: UUID,
        estimated_input: int,
        max_output: int,
        lease_until: datetime | None = None,
    ) -> ModelTokenReservation:
        """Atomically reserve tokens for one provider attempt before it starts.

        The reservation is idempotent per (run, attempt): replaying the same
        attempt never doubles the reservation. A reservation whose billing
        outcome is unknown must be settled or marked uncertain later; it is
        never silently refunded here.
        """

        estimated_input = max(0, int(estimated_input))
        max_output = max(0, int(max_output))
        requested = estimated_input + max_output
        if requested <= 0:
            raise ValueError("reservation must reserve a positive token amount")
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            existing = await self._reservation_row(session, run_id, attempt_id)
            if existing is not None:
                if existing.status == "reserved":
                    return ModelTokenReservation(True, existing.reserved_total, "reserved")
                return ModelTokenReservation(False, 0, existing.status, "attempt_already_terminal")
            maximum = int(run.budget_snapshot.get("max_tokens", 0) or 0)
            borrowed_question_tokens = 0
            if maximum > 0:
                available = (
                    maximum
                    - _model_tokens(run)
                    - await self._outstanding_reservation_total(session, run_id)
                )
                if available < requested:
                    return ModelTokenReservation(
                        False, 0, "insufficient_budget", "outstanding_reservations"
                    )
                allocation = run.budget_snapshot.get("allocation", {})
                allocation = allocation if isinstance(allocation, dict) else {}
                pool_name = model_token_pool_for_node(node)
                pool_limits = model_token_pool_limits(allocation)
                pool_committed = await self._model_pool_committed(session, run_id)
                pool_limit = pool_limits[pool_name]
                if pool_limit > 0 and pool_committed[pool_name] + requested > pool_limit:
                    usage = dict(run.usage_snapshot)
                    usage["last_model_pool_denial"] = {
                        "pool": pool_name,
                        "allocated_tokens": pool_limit,
                        "committed_tokens": pool_committed[pool_name],
                        "requested_tokens": requested,
                    }
                    run.usage_snapshot = _usage_with_resource_pools(run, usage)
                    await self._append_event(
                        session,
                        run,
                        event_type="budget.pool_exhausted",
                        public_summary=f"{pool_name} 模型 Token 池不足; 调用未发起。",
                        refs={"pool": pool_name, "node": node[:100]},
                        metrics={
                            "allocated_tokens": pool_limit,
                            "committed_tokens": pool_committed[pool_name],
                            "requested_tokens": requested,
                        },
                    )
                    return ModelTokenReservation(
                        False,
                        0,
                        "pool_budget_exhausted",
                        f"{pool_name}_token_pool",
                    )
                protected_reserve = sum(
                    _as_int(allocation.get(key, 0))
                    for key in ("verification_tokens", "safety_tokens")
                )
                if node != "report_writer":
                    protected_reserve += max(
                        _as_int(allocation.get("writer_tokens_initial", 0)),
                        _as_int(allocation.get("writer_tokens", 0)),
                    )
                if available - requested < protected_reserve:
                    return ModelTokenReservation(
                        False, 0, "insufficient_budget", "protected_token_pools"
                    )
                question_limits = (
                    allocation.get("question_token_budgets", {})
                    if isinstance(allocation, dict)
                    else {}
                )
                question_limit = (
                    int(question_limits.get(question_id, 0) or 0)
                    if isinstance(question_limits, dict)
                    else 0
                )
                if question_limit > 0:
                    question_spend = int(
                        await session.scalar(
                            select(
                                func.coalesce(
                                    func.sum(
                                        func.coalesce(
                                            ModelBudgetReservationRow.actual_total,
                                            ModelBudgetReservationRow.reserved_total,
                                        )
                                    ),
                                    0,
                                )
                            ).where(
                                ModelBudgetReservationRow.run_id == run_id,
                                ModelBudgetReservationRow.question_id == question_id[:50],
                                ModelBudgetReservationRow.status.in_(
                                    ("reserved", "uncertain", "settled")
                                ),
                            )
                        )
                        or 0
                    )
                    if question_spend + requested > question_limit:
                        projected_spend = question_spend + requested
                        borrowed_question_tokens = min(
                            requested,
                            max(0, projected_spend - max(question_spend, question_limit)),
                        )
                        plan_items = (
                            await session.scalars(
                                select(ResearchPlanItemRow).where(
                                    ResearchPlanItemRow.run_id == run_id,
                                    ResearchPlanItemRow.plan_version == run.plan_version,
                                )
                            )
                        ).all()
                        all_first_attempted = all(
                            bool(_executed_query_families(run.usage_snapshot, item.question_id))
                            for item in plan_items
                        )
                        current_item = next(
                            (item for item in plan_items if item.question_id == question_id),
                            None,
                        )
                        priority = current_item.priority if current_item is not None else 3
                        requirements = (
                            [str(value) for value in current_item.evidence_requirements]
                            if current_item is not None
                            else []
                        )
                        coverage_map = run.quality_snapshot.get("coverage_map", [])
                        coverage = (
                            next(
                                (
                                    _as_float(item.get("coverage", 0.0))
                                    for item in coverage_map
                                    if isinstance(item, dict)
                                    and str(item.get("dimension_key")) == question_id
                                ),
                                0.0,
                            )
                            if isinstance(coverage_map, list)
                            else 0.0
                        )
                        risk_states = run.quality_snapshot.get("risk_state_by_question", {})
                        raw_risk = (
                            risk_states.get(question_id, {})
                            if isinstance(risk_states, dict)
                            else {}
                        )
                        raw_risk = raw_risk if isinstance(raw_risk, dict) else {}
                        risk_state = classify_question_risk(
                            question_id=question_id,
                            priority=priority,
                            coverage=_as_float(raw_risk.get("coverage", coverage)),
                            requirements=requirements,
                            gap_open=bool(raw_risk.get("gap_open", coverage < 1.0)),
                            open_dimension_keys=_string_sequence(
                                raw_risk.get("open_dimension_keys", ())
                            ),
                            unresolved_claim_ids=_string_sequence(
                                raw_risk.get("unresolved_claim_ids", ())
                            ),
                            high_risk_claim_ids=_string_sequence(
                                raw_risk.get("high_risk_claim_ids", ())
                            ),
                            high_risk_conflict_ids=_string_sequence(
                                raw_risk.get("high_risk_conflict_ids", ())
                            ),
                            independent_source_deficit=_as_int(
                                raw_risk.get("independent_source_deficit", 0)
                            ),
                            blocked=str(raw_risk.get("lifecycle", "")) == "blocked",
                        )
                        streaks = run.quality_snapshot.get(
                            "low_information_gain_streak_by_question", {}
                        )
                        low_streak = (
                            _as_int(streaks.get(question_id, 0)) if isinstance(streaks, dict) else 0
                        )
                        utility = expected_utility(
                            priority=priority,
                            gap_risk=risk_state.risk_score,
                            probability_of_accepted_evidence=max(0.15, 0.7 - low_streak * 0.25),
                            expected_coverage_delta=max(0.1, 1.0 - coverage),
                            source_novelty=0.7,
                            cost=ActionCost(estimated_tokens=requested, pages=1),
                        )
                        borrow_decision = decide_question_borrow(
                            state=risk_state,
                            all_first_passes_complete=all_first_attempted,
                            projected_spend=projected_spend,
                            target_tokens=question_limit,
                            expected_utility=utility,
                            low_gain_streak=low_streak,
                            has_untried_query_family=(
                                len(
                                    _executed_query_families(
                                        run.usage_snapshot,
                                        question_id,
                                    )
                                )
                                < _query_strategy_limit(
                                    current_item.question if current_item is not None else ""
                                )
                            ),
                        )
                        if not borrow_decision.allowed:
                            usage = dict(run.usage_snapshot)
                            decisions_raw = usage.get("question_borrow_decisions", {})
                            decisions = (
                                dict(decisions_raw) if isinstance(decisions_raw, dict) else {}
                            )
                            decisions[question_id] = {
                                "allowed": False,
                                "reason": borrow_decision.reason,
                                "hard_limit": borrow_decision.hard_limit,
                                "projected_spend": projected_spend,
                                "target_tokens": question_limit,
                                "expected_utility": utility,
                                "risk_level": risk_state.risk_level,
                            }
                            usage["question_borrow_decisions"] = decisions
                            if borrow_decision.freeze_question:
                                frozen_raw = usage.get("frozen_questions", [])
                                frozen = list(frozen_raw) if isinstance(frozen_raw, list) else []
                                if question_id not in frozen:
                                    frozen.append(question_id)
                                usage["frozen_questions"] = frozen
                            run.usage_snapshot = _usage_with_resource_pools(run, usage)
                            run.quality_snapshot = _mark_question_risk_blocked(
                                run.quality_snapshot,
                                question_id=question_id,
                                reason=borrow_decision.reason,
                            )
                            await self._append_event(
                                session,
                                run,
                                event_type="budget.question_borrow_denied",
                                public_summary="问题追加 Token 借用未通过风险与收益门槛。",
                                refs={"question_id": question_id[:50]},
                                metrics=decisions[question_id],
                            )
                            return ModelTokenReservation(
                                False,
                                0,
                                (
                                    "question_budget_exhausted"
                                    if borrow_decision.hard_limit
                                    else "question_budget_yielded"
                                ),
                                borrow_decision.reason,
                            )
            now = datetime.now(UTC)
            session.add(
                ModelBudgetReservationRow(
                    id=uuid7(),
                    run_id=run_id,
                    question_id=question_id[:50],
                    node=node[:100],
                    attempt_id=attempt_id,
                    estimated_input=estimated_input,
                    max_output=max_output,
                    reserved_total=requested,
                    status="reserved",
                    lease_until=lease_until or (now + _MODEL_RESERVATION_LEASE),
                    created_at=now,
                )
            )
            await session.flush()
            await self._refresh_model_pool_usage(session, run)
            await self._append_event(
                session,
                run,
                event_type="budget.reserved",
                public_summary="已为一次模型调用原子预留 Token 额度。",
                refs={"question_id": question_id[:50], "attempt_id": str(attempt_id)},
                metrics={
                    "estimated_input_tokens": estimated_input,
                    "max_output_tokens": max_output,
                    "reserved_total_tokens": requested,
                    "borrowed_question_tokens": borrowed_question_tokens,
                },
            )
            if borrowed_question_tokens > 0:
                await self._append_event(
                    session,
                    run,
                    event_type="budget.question_allocation_borrowed",
                    public_summary=(
                        "当前问题已从全局剩余 Token 池借用额度; 全局预算上限保持不变。"
                    ),
                    refs={"question_id": question_id[:50]},
                    metrics={"borrowed_tokens": borrowed_question_tokens},
                )
            return ModelTokenReservation(True, requested, "reserved")

    async def settle_model_reservation(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        attempt_id: UUID,
        actual_total: int,
        usage_estimated: bool = False,
    ) -> bool:
        """Idempotently settle one attempt with its audited token total."""

        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            return await self._settle_reservation_row(
                session,
                run,
                attempt_id=attempt_id,
                actual_total=actual_total,
                usage_estimated=usage_estimated,
            )

    async def mark_model_reservation_uncertain(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        attempt_id: UUID,
    ) -> bool:
        """Keep occupying the conservative upper bound when usage is unknown."""

        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            row = await self._reservation_row(session, run_id, attempt_id)
            if row is None or row.status != "reserved":
                return False
            row.status = "uncertain"
            await self._refresh_model_pool_usage(session, run)
            await self._append_event(
                session,
                run,
                event_type="budget.reservation_uncertain",
                public_summary="一次模型调用的计费结果未知; 保守占用预留额度直至对账。",
                refs={"question_id": row.question_id, "attempt_id": str(attempt_id)},
                metrics={"reserved_total_tokens": row.reserved_total},
            )
            return True

    async def release_model_reservation(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        attempt_id: UUID,
    ) -> bool:
        """Refund a reservation that never produced a billable provider call."""

        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            row = await self._reservation_row(session, run_id, attempt_id)
            if row is None or row.status != "reserved":
                return False
            row.status = "released"
            row.settled_at = datetime.now(UTC)
            await self._refresh_model_pool_usage(session, run)
            await self._append_event(
                session,
                run,
                event_type="budget.reservation_released",
                public_summary="预留 Token 额度已释放; 该尝试未产生计费调用。",
                refs={"question_id": row.question_id, "attempt_id": str(attempt_id)},
                metrics={"released_tokens": row.reserved_total},
            )
            return True

    @staticmethod
    async def _reservation_row(
        session: AsyncSession, run_id: UUID, attempt_id: UUID
    ) -> ModelBudgetReservationRow | None:
        return cast(
            ModelBudgetReservationRow | None,
            await session.scalar(
                select(ModelBudgetReservationRow).where(
                    ModelBudgetReservationRow.run_id == run_id,
                    ModelBudgetReservationRow.attempt_id == attempt_id,
                )
            ),
        )

    @staticmethod
    async def _settle_reservation_row(
        session: AsyncSession,
        run: ResearchRunRow,
        *,
        attempt_id: UUID,
        actual_total: int,
        usage_estimated: bool,
    ) -> bool:
        row = await ResearchToolRepository._reservation_row(session, run.id, attempt_id)
        if row is None or row.status in ("settled", "released"):
            return False
        row.status = "settled"
        row.actual_total = max(0, int(actual_total))
        row.usage_estimated = usage_estimated
        row.settled_at = datetime.now(UTC)
        usage = dict(run.usage_snapshot)
        usage["settled_reservations"] = _as_int(usage.get("settled_reservations", 0)) + 1
        run.usage_snapshot = _usage_with_resource_pools(run, usage)
        await ResearchToolRepository._refresh_model_pool_usage(session, run)
        await ResearchToolRepository._append_event(
            session,
            run,
            event_type="budget.settled",
            public_summary="模型调用预留已按实际用量结算。",
            refs={"question_id": row.question_id, "attempt_id": str(attempt_id)},
            metrics={
                "reserved_total_tokens": row.reserved_total,
                "actual_total_tokens": row.actual_total,
                "usage_estimated": usage_estimated,
            },
        )
        return True

    @staticmethod
    async def _outstanding_reservation_total(session: AsyncSession, run_id: UUID) -> int:
        """Tokens currently held by in-flight or unreconciled attempts.

        ``reserved`` rows only hold funds while their lease is alive; a worker
        crash lets the lease expire and the funds return until reconciliation.
        ``uncertain`` rows keep their conservative upper bound indefinitely.
        """

        rows = (
            await session.execute(
                select(
                    ModelBudgetReservationRow.reserved_total,
                    ModelBudgetReservationRow.status,
                    ModelBudgetReservationRow.lease_until,
                ).where(
                    ModelBudgetReservationRow.run_id == run_id,
                    ModelBudgetReservationRow.status.in_(("reserved", "uncertain")),
                )
            )
        ).all()
        now = datetime.now(UTC)
        total = 0
        for reserved_total, status, lease_until in rows:
            if status == "reserved" and lease_until is not None and lease_until < now:
                continue
            total += int(reserved_total)
        return total

    @staticmethod
    async def _model_pool_committed(
        session: AsyncSession,
        run_id: UUID,
    ) -> dict[str, int]:
        """Return conservative committed tokens for every executable node pool."""

        rows = (
            await session.execute(
                select(
                    ModelBudgetReservationRow.node,
                    ModelBudgetReservationRow.reserved_total,
                    ModelBudgetReservationRow.actual_total,
                    ModelBudgetReservationRow.status,
                    ModelBudgetReservationRow.lease_until,
                ).where(
                    ModelBudgetReservationRow.run_id == run_id,
                    ModelBudgetReservationRow.status.in_(
                        ("reserved", "uncertain", "settled")
                    ),
                )
            )
        ).all()
        committed = {
            "planner": 0,
            "research": 0,
            "verification": 0,
            "writer": 0,
            "safety": 0,
        }
        now = datetime.now(UTC)
        for node, reserved_total, actual_total, status, lease_until in rows:
            if status == "reserved" and lease_until is not None and lease_until < now:
                continue
            charge = (
                int(actual_total or 0)
                if status == "settled"
                else int(reserved_total or 0)
            )
            pool = model_token_pool_for_node(str(node))
            committed[pool] += max(0, charge)
        return committed

    @staticmethod
    async def _model_pool_accounting(
        session: AsyncSession,
        run_id: UUID,
    ) -> dict[str, dict[str, int]]:
        """Split settled, in-flight, and uncertain charges by model pool."""

        rows = (
            await session.execute(
                select(
                    ModelBudgetReservationRow.node,
                    ModelBudgetReservationRow.reserved_total,
                    ModelBudgetReservationRow.actual_total,
                    ModelBudgetReservationRow.status,
                    ModelBudgetReservationRow.lease_until,
                ).where(
                    ModelBudgetReservationRow.run_id == run_id,
                    ModelBudgetReservationRow.status.in_(("reserved", "uncertain", "settled")),
                )
            )
        ).all()
        accounting = {
            pool: {"settled": 0, "reserved": 0, "uncertain": 0}
            for pool in ("planner", "research", "verification", "writer", "safety")
        }
        now = datetime.now(UTC)
        for node, reserved_total, actual_total, status, lease_until in rows:
            if status == "reserved" and lease_until is not None and lease_until < now:
                continue
            pool = model_token_pool_for_node(str(node))
            if status == "settled":
                accounting[pool]["settled"] += max(0, int(actual_total or 0))
            elif status == "uncertain":
                accounting[pool]["uncertain"] += max(0, int(reserved_total or 0))
            else:
                accounting[pool]["reserved"] += max(0, int(reserved_total or 0))
        return accounting

    @staticmethod
    async def _question_token_pool_usage(
        session: AsyncSession,
        run: ResearchRunRow,
    ) -> dict[str, dict[str, object]]:
        allocation = run.budget_snapshot.get("allocation", {})
        if not isinstance(allocation, dict):
            return {}
        targets = allocation.get("question_token_budgets", {})
        floors = allocation.get("question_token_floors", {})
        if not isinstance(targets, dict):
            return {}
        floors = floors if isinstance(floors, dict) else {}
        rows = (
            await session.execute(
                select(
                    ModelBudgetReservationRow.question_id,
                    ModelBudgetReservationRow.node,
                    ModelBudgetReservationRow.reserved_total,
                    ModelBudgetReservationRow.actual_total,
                    ModelBudgetReservationRow.status,
                    ModelBudgetReservationRow.lease_until,
                ).where(
                    ModelBudgetReservationRow.run_id == run.id,
                    ModelBudgetReservationRow.status.in_(("reserved", "uncertain", "settled")),
                )
            )
        ).all()
        charges: dict[str, dict[str, int]] = {}
        now = datetime.now(UTC)
        for question_id, node, reserved_total, actual_total, status, lease_until in rows:
            if model_token_pool_for_node(str(node)) != "research":
                continue
            if status == "reserved" and lease_until is not None and lease_until < now:
                continue
            bucket = charges.setdefault(
                str(question_id), {"settled": 0, "reserved": 0, "uncertain": 0}
            )
            if status == "settled":
                bucket["settled"] += max(0, int(actual_total or 0))
            elif status == "uncertain":
                bucket["uncertain"] += max(0, int(reserved_total or 0))
            else:
                bucket["reserved"] += max(0, int(reserved_total or 0))
        risk_by_question = run.quality_snapshot.get("risk_state_by_question", {})
        risk_by_question = risk_by_question if isinstance(risk_by_question, dict) else {}
        result: dict[str, dict[str, object]] = {}
        for question_id, raw_target in targets.items():
            key = str(question_id)
            target = max(0, _as_int(raw_target))
            floor = max(0, _as_int(floors.get(key, 0)))
            charge = charges.get(key, {"settled": 0, "reserved": 0, "uncertain": 0})
            committed = charge["settled"]
            held = charge["reserved"]
            projected = committed + held + charge["uncertain"]
            risk = risk_by_question.get(key, {})
            p1_exception = bool(
                isinstance(risk, dict)
                and _as_int(risk.get("priority", 3)) == 1
                and risk.get("unresolved_high_risk")
            )
            hard_limit = int(target * 1.5)
            # High-risk P1 recovery may cross the normal 150% envelope only
            # within the same bounded two-target ceiling enforced by
            # decide_question_borrow().  Never render an exception as an
            # unlimited borrowing state in the dashboard.
            recovery_limit = int(target * 2.0)
            result[key] = {
                "floor_tokens": floor,
                "target_tokens": target,
                "hard_limit_tokens": hard_limit,
                "settled_tokens": charge["settled"],
                "reserved_tokens": held,
                "uncertain_tokens": charge["uncertain"],
                "committed_tokens": committed,
                "projected_tokens": projected,
                "borrowed_tokens": max(0, projected - target),
                "remaining_to_target_tokens": max(0, target - projected),
                "remaining_to_hard_limit_tokens": max(0, hard_limit - projected),
                "recovery_limit_tokens": recovery_limit,
                "remaining_to_recovery_limit_tokens": max(0, recovery_limit - projected),
                "p1_high_risk_exception": p1_exception,
                "status": (
                    "hard_exhausted"
                    if (
                        recovery_limit > 0 and projected >= recovery_limit
                    ) or (
                        hard_limit > 0 and projected >= hard_limit and not p1_exception
                    )
                    else "borrowing"
                    if projected > target
                    else "target_exhausted"
                    if target > 0 and projected >= target
                    else "available"
                ),
            }
        return result

    @staticmethod
    async def _refresh_model_pool_usage(
        session: AsyncSession,
        run: ResearchRunRow,
    ) -> None:
        allocation = run.budget_snapshot.get("allocation", {})
        if not isinstance(allocation, dict):
            return
        limits = model_token_pool_limits(allocation)
        accounting = await ResearchToolRepository._model_pool_accounting(session, run.id)
        usage = dict(run.usage_snapshot)
        usage["model_token_pools"] = {
            pool: {
                "allocated_tokens": limit,
                "committed_tokens": accounting[pool]["settled"],
                "settled_tokens": accounting[pool]["settled"],
                "reserved_tokens": accounting[pool]["reserved"],
                "uncertain_tokens": accounting[pool]["uncertain"],
                "remaining_tokens": max(
                    0,
                    limit
                    - accounting[pool]["settled"]
                    - accounting[pool]["reserved"]
                    - accounting[pool]["uncertain"],
                ),
                "status": (
                    "exhausted"
                    if limit > 0
                    and accounting[pool]["settled"]
                    + accounting[pool]["reserved"]
                    + accounting[pool]["uncertain"]
                    >= limit
                    else "available"
                    if limit > 0
                    else "disabled"
                ),
            }
            for pool, limit in limits.items()
        }
        usage["question_token_pools"] = (
            await ResearchToolRepository._question_token_pool_usage(session, run)
        )
        run.usage_snapshot = _usage_with_resource_pools(run, usage)

    async def prepare_target(self, run_id: UUID, *, worker_task_id: str) -> ResearchTarget | None:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            budget_stop_reason = _budget_exhaustion_reason(run)
            search_budget_exhausted = _search_acquisition_budget_exhausted(
                budget_stop_reason
            )
            if budget_stop_reason is not None and not search_budget_exhausted:
                await self._enter_writing(
                    session,
                    run,
                    reason=budget_stop_reason,
                    summary=_budget_stop_summary(budget_stop_reason),
                )
                return None
            candidates = (
                await session.scalars(
                    select(ResearchPlanItemRow)
                    .where(
                        ResearchPlanItemRow.run_id == run_id,
                        ResearchPlanItemRow.plan_version == run.plan_version,
                        ResearchPlanItemRow.status.in_(
                            ("pending", "active", "blocked", "partial", "technical_retry")
                        ),
                    )
                    .with_for_update()
                )
            ).all()
            quality_snapshot = cast(dict[str, object], run.quality_snapshot or {})
            coverage_entries = quality_snapshot.get("coverage_map", [])
            if not isinstance(coverage_entries, list):
                coverage_entries = []
            coverage_by_question = {
                str(entry.get("dimension_key")): float(entry.get("coverage", 0.0) or 0.0)
                for entry in coverage_entries
                if isinstance(entry, dict) and entry.get("dimension_key") is not None
            }
            raw_quality_repair_targets = quality_snapshot.get("quality_repair_targets", {})
            quality_repair_targets = (
                {
                    str(question_id): [str(reason) for reason in reasons]
                    for question_id, reasons in raw_quality_repair_targets.items()
                    if isinstance(reasons, list)
                }
                if isinstance(raw_quality_repair_targets, dict)
                else {}
            )
            # Recompute corroboration targets from the live dimension map as
            # well.  The persisted quality-repair projection can lag one
            # extraction behind (especially across a replan), which otherwise
            # lets zero-yield P1 retries outrank claims that already need a
            # second independent source.
            corroboration_targets: set[str] = set(quality_repair_targets)
            accepted_by_question: dict[str, int] = {}
            for entry in coverage_entries:
                if not isinstance(entry, dict):
                    continue
                question_id = str(entry.get("dimension_key", ""))
                if not question_id:
                    continue
                accepted_by_question[question_id] = _as_int(
                    entry.get("accepted_evidence", 0)
                )
                statuses = entry.get("requirement_statuses", [])
                if not isinstance(statuses, list):
                    continue
                if any(
                    isinstance(status, dict)
                    and _as_int(status.get("accepted_evidence", 0)) > 0
                    and _as_int(status.get("independent_sources", 0))
                    < max(1, _as_int(status.get("required_sources", 1)))
                    for status in statuses
                ):
                    corroboration_targets.add(question_id)
            # Keep a small tail of the page budget available for unresolved
            # priority-one dimensions and quality-gate repair. Without this,
            # low-priority questions can consume every page before source
            # quality/cross-validation enrichment gets a chance to run.
            max_pages = int(run.budget_snapshot.get("max_pages", 0) or 0)
            committed_pages = max(0, _as_int(run.usage_snapshot.get("pages", 0)))
            reserved_pages = max(0, _as_int(run.usage_snapshot.get("page_slots_reserved", 0)))
            remaining_pages = max(0, max_pages - committed_pages - reserved_pages)
            critical_gap_count = _as_int(quality_snapshot.get("critical_gaps", 0))
            quality_reserve = max(critical_gap_count * 2, len(quality_repair_targets) * 2)
            # Reserve pages only for concrete dimensions capable of changing
            # the open gate; a global low score alone cannot nominate whichever
            # question happened to run most recently.
            page_reserve = min(max_pages, max(0, quality_reserve))
            protected_page_mode = page_reserve > 0 and remaining_pages <= page_reserve
            unfinished_p1_variants: list[str] = []
            eligible: list[tuple[int, int, float, int, str, ResearchPlanItemRow]] = []
            exhausted_by_question = run.usage_snapshot.get(
                "question_budget_exhausted_by_question", {}
            )
            exhausted_questions = {
                str(question_id)
                for question_id, exhausted in (
                    exhausted_by_question if isinstance(exhausted_by_question, dict) else {}
                ).items()
                if exhausted
            }
            strategy_exhausted = run.usage_snapshot.get("query_strategy_exhausted_by_question", {})
            strategy_exhausted_questions = {
                str(question_id)
                for question_id, exhausted in (
                    strategy_exhausted if isinstance(strategy_exhausted, dict) else {}
                ).items()
                if exhausted
            }
            frozen_raw = run.usage_snapshot.get("frozen_questions", [])
            frozen_questions = {
                str(question_id)
                for question_id in (frozen_raw if isinstance(frozen_raw, list) else [])
            }
            for candidate in candidates:
                if (
                    candidate.status == "partial"
                    and coverage_by_question.get(candidate.question_id, 0.0) >= 1.0
                    and candidate.question_id not in quality_repair_targets
                    and candidate.question_id not in corroboration_targets
                ):
                    # Repair stale state produced by the former global-anchor
                    # policy. A complete question without an actionable gate
                    # dimension is terminal for this plan version.  A question
                    # that still needs an independent source is actionable even
                    # when its coverage score has reached 1.0.
                    candidate.status = "researched"
            historical_attempts = {
                question_id: int(attempts)
                for question_id, attempts in (
                    await session.execute(
                        select(
                            SearchQueryRow.question_id,
                            func.count(SearchQueryRow.id),
                        )
                        .where(SearchQueryRow.run_id == run_id)
                        .group_by(SearchQueryRow.question_id)
                    )
                ).all()
            }
            cached_question_ids: set[str] = set()
            if search_budget_exhausted:
                cached_question_ids = {
                    str(question_id)
                    for question_id in (
                        await session.scalars(
                            select(distinct(SearchQueryRow.question_id))
                            .join(
                                SearchResultRow,
                                SearchResultRow.search_query_id == SearchQueryRow.id,
                            )
                            .where(
                                SearchQueryRow.run_id == run_id,
                                SearchQueryRow.status == "succeeded",
                            )
                        )
                    ).all()
                }
            executed_family_attempts = {
                candidate.question_id: len(
                    _executed_query_families(run.usage_snapshot, candidate.question_id)
                )
                for candidate in candidates
            }
            zero_yield_deprioritized_questions = {
                candidate.question_id
                for candidate in candidates
                if _zero_yield_retry_deprioritized(
                    search_budget_exhausted=search_budget_exhausted,
                    attempts=executed_family_attempts.get(
                        candidate.question_id,
                        historical_attempts.get(candidate.question_id, 0),
                    ),
                    coverage=coverage_by_question.get(candidate.question_id, 0.0),
                    accepted_evidence=accepted_by_question.get(candidate.question_id, 0),
                    is_corroboration_target=(
                        candidate.question_id in corroboration_targets
                    ),
                )
            }
            unattempted_questions = sum(
                1
                for candidate in candidates
                if candidate.question_id not in exhausted_questions
                and candidate.question_id not in strategy_exhausted_questions
                and candidate.question_id not in frozen_questions
                and candidate.status != "researched"
                and executed_family_attempts.get(candidate.question_id, 0) == 0
            )
            for candidate in candidates:
                if (
                    candidate.question_id in exhausted_questions
                    or candidate.question_id in strategy_exhausted_questions
                    or candidate.question_id in frozen_questions
                    or (
                        candidate.question_id in zero_yield_deprioritized_questions
                        and candidate.priority != 1
                        and candidate.question_id not in corroboration_targets
                    )
                    or candidate.status == "researched"
                    or (
                        search_budget_exhausted and candidate.question_id not in cached_question_ids
                    )
                ):
                    continue
                attempts = executed_family_attempts.get(
                    candidate.question_id,
                    historical_attempts.get(candidate.question_id, 0),
                )
                candidate_coverage = coverage_by_question.get(candidate.question_id, 0.0)
                if (
                    candidate.priority == 1 or candidate.question_id in corroboration_targets
                ) and candidate_coverage < 1.0 and _query_family_capacity_remaining(
                    question=candidate.question,
                    attempted_families=attempts,
                ):
                    unfinished_p1_variants.append(candidate.question_id)
            p1_variant_mode = bool(unfinished_p1_variants)
            # A P1 variant must not starve the first pass of other questions.
            # The allocator deliberately yields a retried question when an
            # untouched question still exists; keep that fairness contract at
            # the scheduler boundary as well. Once every question has had a
            # first attempt, unresolved P1 items regain the protected variant
            # lane and can receive their bounded follow-up searches.
            # Reuse the eligibility-aware count above. A frozen or exhausted
            # question with zero attempts must not keep the P1 recovery lane
            # disabled for the rest of the run.
            has_unattempted_question = unattempted_questions > 0
            p1_variant_mode = p1_variant_mode and not has_unattempted_question
            for candidate in candidates:
                if (
                    candidate.question_id in exhausted_questions
                    or candidate.question_id in strategy_exhausted_questions
                    or candidate.question_id in frozen_questions
                    or candidate.status == "researched"
                    or (
                        search_budget_exhausted and candidate.question_id not in cached_question_ids
                    )
                ):
                    continue
                attempts = executed_family_attempts.get(
                    candidate.question_id,
                    historical_attempts.get(candidate.question_id, 0),
                )
                candidate_coverage = coverage_by_question.get(candidate.question_id, 0.0)
                if p1_variant_mode and candidate.question_id not in unfinished_p1_variants:
                # Give every unfinished P1 every bounded query family
                # before ordinary P2/P3 work can compete for pages.
                    continue
                effective_priority = candidate.priority
                if protected_page_mode and candidate.priority > 1 and candidate_coverage >= 1.0:
                    # P1/gap repair gets the remaining tail first.
                    effective_priority += 10
                family_stats_raw = run.usage_snapshot.get("query_family_stats", {})
                next_family = query_family_for_attempt(attempts) or QueryFamily.CONTRADICTION
                family_stats = (
                    family_stats_raw.get(next_family.value, {})
                    if isinstance(family_stats_raw, dict)
                    else {}
                )
                family_stats = family_stats if isinstance(family_stats, dict) else {}
                family_queries = _as_int(family_stats.get("logical_queries", 0))
                family_accepted = _as_int(family_stats.get("accepted_evidence", 0))
                acceptance_probability = (
                    min(0.9, max(0.15, family_accepted / family_queries))
                    if family_queries
                    else 0.55
                )
                utility = expected_utility(
                    priority=effective_priority,
                    gap_risk=max(0.1, 1.0 - candidate_coverage),
                    probability_of_accepted_evidence=acceptance_probability,
                    expected_coverage_delta=max(0.1, 1.0 - candidate_coverage),
                    source_novelty=0.9 if attempts == 0 else 0.65,
                    cost=ActionCost(
                        estimated_tokens=6_000,
                        latency_ms=2_000,
                        pages=1,
                        provider_requests=1,
                    ),
                )
                utility_rank = -int(utility * 1_000_000)
                if attempts > 0 and candidate.question_id in corroboration_targets:
                    # Corroboration is a quality-gate action, not an ordinary
                    # retry. Give it a dedicated lane and rotate by attempt
                    # count so one already-covered P1 cannot monopolize it.
                    utility_rank = -1_000_000_000 + attempts * 1_000_000 - int(
                        utility * 1_000
                    )
                elif attempts > 0 and candidate_coverage > 0.0:
                    utility_rank -= 500_000_000
                elif candidate.question_id in zero_yield_deprioritized_questions:
                    # Two empty families should yield to untouched and
                    # productive work, but must remain recoverable: the third
                    # alternate-language family may be the first useful one.
                    utility_rank += 500_000_000
                schedule_key = (
                    0 if attempts == 0 else 1,
                    0 if attempts == 0 else utility_rank,
                    candidate_coverage,
                    effective_priority,
                    candidate.question_id,
                )
                if protected_page_mode:
                    protected_candidate = (
                        candidate.priority == 1
                        or candidate_coverage < 1.0
                        or candidate.question_id in quality_repair_targets
                        or candidate.status in {"partial", "blocked", "technical_retry"}
                    )
                    if not protected_candidate:
                        # Once the reserve zone is reached, do not spend its
                        # slots on already-covered ordinary P2/P3 questions.
                        continue
                if p1_variant_mode:
                    # The P1 gate is a minimum across all P1 questions.  An
                    # already productive 50% item must therefore not outrank
                    # a 0% P1 merely because its next extraction is likelier
                    # to succeed.  Lowest coverage wins; attempts only rotate
                    # candidates tied at the same coverage.
                    schedule_key = _p1_variant_schedule_key(
                        attempts=attempts,
                        coverage=candidate_coverage,
                        priority=effective_priority,
                        question_id=candidate.question_id,
                    )
                elif protected_page_mode:
                    # Keep the first-pass fairness axis even in the protected
                    # tail. Without it, an unresolved P1 could repeatedly win
                    # by priority and prevent untouched questions from ever
                    # completing their first pass; the borrow gate would then
                    # remain closed forever for every follow-up attempt.
                    schedule_key = _protected_page_schedule_key(
                        attempts=attempts,
                        coverage=candidate_coverage,
                        priority=effective_priority,
                        question_id=candidate.question_id,
                    )
                eligible.append(
                    (
                        *schedule_key,
                        candidate,
                    )
                )
            selected_entry = min(eligible, default=None, key=lambda item: item[:5])
            if selected_entry is None:
                quality_snapshot = cast(dict[str, object], run.quality_snapshot or {})
                quality_met = _quality_gate_met_from_snapshot(quality_snapshot)
                unresolved_gap_count = _as_int(quality_snapshot.get("unresolved_gap_count", 0))
                replans_used = _as_int(run.usage_snapshot.get("replans", 0))
                if (
                    not search_budget_exhausted
                    and not quality_met
                    and unresolved_gap_count > 0
                    and replans_used < _MAX_REPLANS
                ):
                    # No currently eligible candidate is not equivalent to
                    # source exhaustion: frozen/strategy-exhausted questions
                    # may still be recoverable through a targeted REPLAN.
                    usage = dict(run.usage_snapshot)
                    usage["replan_requested"] = True
                    run.usage_snapshot = _usage_with_resource_pools(run, usage)
                    run.phase = RunPhase.RESEARCHING.value
                    await self._append_event(
                        session,
                        run,
                        event_type="research.replan_requested",
                        public_summary="当前候选均不可执行但仍有开放验收缺口; 转入定向 REPLAN。",
                        refs={"unresolved_gap_count": unresolved_gap_count},
                    )
                    return None
                stop_reason = (
                    budget_stop_reason or "logical_query_budget_exhausted"
                    if search_budget_exhausted
                    else "quality_met"
                    if quality_met
                    else "sources_exhausted"
                )
                await self._enter_writing(
                    session,
                    run,
                    reason=stop_reason,
                    summary=(
                        _budget_stop_summary(stop_reason)
                        if search_budget_exhausted
                        else "研究质量门已满足; 自动进入证据驱动报告写作。"
                        if quality_met
                        else "没有可继续执行的研究候选且质量门未满足; 使用现有证据生成带限制报告。"
                    ),
                )
                return None
            question: ResearchPlanItemRow = selected_entry[5]
            now = datetime.now(UTC)
            gap = await session.scalar(
                select(ResearchGapRow).where(
                    ResearchGapRow.run_id == run_id,
                    ResearchGapRow.plan_version == run.plan_version,
                    ResearchGapRow.question_id == question.question_id,
                )
            )
            created_gap = gap is None
            if gap is None:
                # A replan creates a new Gap row, but it must not reset the
                # search strategy to attempt zero.  Count prior searches for
                # this stable question id so later plan versions use a new
                # query variant instead of rereading the same failed result.
                prior_attempts = int(
                    await session.scalar(
                        select(func.count(SearchQueryRow.id)).where(
                            SearchQueryRow.run_id == run_id,
                            SearchQueryRow.question_id == question.question_id,
                        )
                    )
                    or 0
                )
                gap = ResearchGapRow(
                    id=uuid7(),
                    run_id=run_id,
                    plan_version=run.plan_version,
                    question_id=question.question_id,
                    gap_type="missing",
                    description=f"当前缺少对研究问题 {question.question_id} 的可验证证据。",
                    acceptance_criteria="至少获得一条可定位到原网页逐字引文的有效证据。",
                    severity=1.0,
                    status="open",
                    resolution_attempts=prior_attempts,
                    created_at=now,
                    updated_at=now,
                )
                session.add(gap)
                await session.flush()
            gap.updated_at = now
            question.status = "active"
            # Query strategy is keyed to actual searches for the stable
            # question id across all plan versions. Gap/evaluator passes may
            # include cache reuse or temporary yields and must not advance the
            # search-variant sequence.
            executed_families = _executed_query_families(
                run.usage_snapshot, question.question_id
            )
            attempt_index = len(executed_families)
            current_coverage = next(
                (
                    entry
                    for entry in coverage_entries
                    if isinstance(entry, dict)
                    and str(entry.get("dimension_key")) == question.question_id
                ),
                None,
            )
            unmet_criterion: str | None = None
            unmet_dimension_key: str | None = None
            repair_reasons = quality_repair_targets.get(question.question_id, [])
            if current_coverage is not None:
                requirement_statuses = current_coverage.get("requirement_statuses", [])
                if isinstance(requirement_statuses, list):
                    unmet = _select_unmet_requirement(requirement_statuses)
                    if unmet is not None:
                        unmet_dimension_key, unmet_criterion = unmet
                    if unmet_criterion is None and repair_reasons:
                        repair_dimensions = {
                            reason.split(":", 1)[1] for reason in repair_reasons if ":" in reason
                        }
                        repair_target = next(
                            (
                                (
                                    str(status.get("dimension_key")),
                                    str(status.get("criterion")),
                                )
                                for status in requirement_statuses
                                if isinstance(status, dict)
                                and str(status.get("dimension_key")) in repair_dimensions
                                and status.get("criterion")
                            ),
                            None,
                        )
                        if repair_target is not None:
                            unmet_dimension_key, unmet_criterion = repair_target
            executed_query_hashes = set(
                (
                    await session.scalars(
                        select(SearchQueryRow.normalized_hash).where(
                            SearchQueryRow.run_id == run_id,
                            SearchQueryRow.question_id == question.question_id,
                            SearchQueryRow.status == "succeeded",
                        )
                    )
                ).all()
            )
            query = ""
            query_family: QueryFamily | None = None
            prefer_authoritative = (
                any(reason.startswith("source_quality:") for reason in repair_reasons)
                or question.question_id in corroboration_targets
            )
            family_order = _query_family_order(prefer_authoritative=prefer_authoritative)
            for family in family_order:
                # A replan may deliberately keep the same query family while
                # changing the gap-specific search hint.  Treating the family
                # itself as exhausted made every replan a no-op: all four
                # families had already been used by the parent plan, so the
                # new hints were never materialized into a query.  Duplicate
                # protection is query-hash scoped below, which still prevents
                # replaying an identical successful query while allowing a
                # genuinely new angle in the same family.
                candidate_query = build_family_query(
                    question=question.question,
                    criterion=(unmet_criterion or "").strip(),
                    hints=tuple(str(value) for value in question.search_hints),
                    family=family,
                )
                if unmet_criterion and family is not QueryFamily.SCOPE:
                    candidate_query = " ".join(
                        (
                            candidate_query,
                            _source_hint_for_requirement(unmet_criterion),
                        )
                    )[:400]
                candidate_hash = hashlib.sha256(
                    normalize_search_query(candidate_query).encode()
                ).hexdigest()
                if candidate_query and candidate_hash not in executed_query_hashes:
                    query = candidate_query
                    query_family = family
                    attempt_index = QUERY_FAMILIES.index(family)
                    break
            if not query:
                strategy_usage = dict(run.usage_snapshot)
                exhausted_strategies = dict(
                    strategy_usage.get("query_strategy_exhausted_by_question", {})
                )
                exhausted_strategies[question.question_id] = True
                strategy_usage["query_strategy_exhausted_by_question"] = exhausted_strategies
                source_space = dict(
                    strategy_usage.get("source_space_exhausted_by_question", {})
                )
                source_space[question.question_id] = True
                strategy_usage["source_space_exhausted_by_question"] = source_space
                run.usage_snapshot = strategy_usage
                question.status = (
                    "partial"
                    if coverage_by_question.get(question.question_id, 0.0) > 0.0
                    else "blocked"
                )
                await self._append_event(
                    session,
                    run,
                    event_type="search.source_space_exhausted",
                    public_summary=(
                        f"问题 {question.question_id} 没有未执行的标准化检索角度; "
                        "禁止跨计划重复搜索。"
                    ),
                    refs={"question_id": question.question_id},
                )
                # This is an exceptional legacy-state path. The next graph
                # step will schedule another question or enter limited writing.
                return None
            normalized_query = normalize_search_query(query)
            if query_family is None:  # pragma: no cover - guarded by ``query`` above
                query_family = query_family_for_attempt(attempt_index) or QueryFamily.SCOPE
            alternate_query = build_family_query(
                question=question.question,
                criterion=(unmet_criterion or "").strip(),
                hints=tuple(str(value) for value in question.search_hints),
                family=query_family,
                prefer_alternate_hint=True,
            )
            if normalize_search_query(alternate_query) == normalized_query:
                alternate_query = ""
            # Query idempotency is run-scoped, not plan-version-scoped. A
            # replan may change the search angle, but it must never authorize
            # replaying a query that already succeeded in an older plan.
            duplicate_key = search_query_duplicate_key(question.question_id, normalized_query)
            tool_call = await session.scalar(
                select(ResearchToolCallRow).where(
                    ResearchToolCallRow.run_id == run_id,
                    ResearchToolCallRow.tool_name == "web_search",
                    ResearchToolCallRow.duplicate_key == duplicate_key,
                )
            )
            query_already_executed = tool_call is not None and tool_call.status == "succeeded"
            if tool_call is None:
                tool_call = ResearchToolCallRow(
                    id=uuid7(),
                    run_id=run_id,
                    question_id=question.question_id,
                    gap_id=gap.id,
                    action_id=uuid7(),
                    tool_name="web_search",
                    duplicate_key=duplicate_key,
                    status="running",
                    arguments={
                        "query": query,
                        "alternate_query": alternate_query,
                        "limit": 10,
                        "query_family": query_family.value,
                    },
                    result_refs={},
                    started_at=now,
                )
                session.add(tool_call)
                await session.flush()
            else:
                tool_call.status = "running"
                tool_call.error_code = None
                tool_call.retryable = False
                tool_call.started_at = now
                tool_call.finished_at = None
                tool_call.arguments = {
                    **(tool_call.arguments or {}),
                    "query": query,
                    "alternate_query": alternate_query,
                    "query_family": query_family.value,
                }
            if created_gap:
                await self._append_event(
                    session,
                    run,
                    event_type="gap.opened",
                    public_summary=f"识别到问题 {question.question_id} 的证据缺口。",
                    refs={"gap_id": str(gap.id), "question_id": question.question_id},
                )
            await self._append_event(
                session,
                run,
                event_type="action.selected",
                public_summary="Agent 根据当前缺口选择 Web Search。",
                refs={
                    "action_id": str(tool_call.action_id),
                    "gap_id": str(gap.id),
                    "question_id": question.question_id,
                    "query_family": query_family.value,
                },
            )
            await self._append_event(
                session,
                run,
                event_type="tool.called",
                public_summary=f"正在搜索: {query[:300]}",
                refs={
                    "tool_call_id": str(tool_call.id),
                    "tool_name": "web_search",
                    "question_id": question.question_id,
                },
            )
            used_owner_keys = tuple(
                str(owner)
                for owner in (
                    await session.scalars(
                        select(distinct(ResearchSourceRow.source_owner_key))
                        .join(
                            ResearchEvidenceRow,
                            ResearchEvidenceRow.source_id == ResearchSourceRow.id,
                        )
                        .where(
                            ResearchEvidenceRow.run_id == run_id,
                            ResearchEvidenceRow.question_id == question.question_id,
                            ResearchEvidenceRow.accepted.is_(True),
                        )
                    )
                ).all()
                if owner
            )
            read_urls = {
                normalize_source_url(url)
                for url in (
                    await session.scalars(
                        select(ResearchSourceRow.canonical_url).where(
                            ResearchSourceRow.run_id == run_id,
                            ResearchSourceRow.canonical_url.is_not(None),
                        )
                    )
                ).all()
                if url
            }
            reusable_rows = (
                (
                    await session.execute(
                        select(SearchResultRow)
                        .join(SearchQueryRow, SearchResultRow.search_query_id == SearchQueryRow.id)
                        .where(
                            SearchQueryRow.run_id == run_id,
                            SearchQueryRow.question_id == question.question_id,
                        )
                        .order_by(SearchQueryRow.created_at.desc(), SearchResultRow.rank)
                        .limit(20)
                    )
                )
                .scalars()
                .all()
            )
            cross_question_rows: list[SearchResultRow] = []
            if bool(run.budget_snapshot.get("cross_question_search_cache_enabled", True)):
                query_hash = hashlib.sha256(normalized_query.encode()).hexdigest()
                cross_question_rows = list(
                    (
                        await session.execute(
                            select(SearchResultRow)
                            .join(
                                SearchQueryRow,
                                SearchResultRow.search_query_id == SearchQueryRow.id,
                            )
                            .where(
                                SearchQueryRow.run_id == run_id,
                                SearchQueryRow.question_id != question.question_id,
                                SearchQueryRow.normalized_hash == query_hash,
                                SearchQueryRow.status == "succeeded",
                            )
                            .order_by(SearchResultRow.rank)
                            .limit(20)
                        )
                    )
                    .scalars()
                    .all()
                )
                if cross_question_rows:
                    query_already_executed = True
                    reusable_rows = [*reusable_rows, *cross_question_rows]
            failure_event_rows = (
                await session.execute(
                    select(AgentEventRow.event_type, AgentEventRow.refs).where(
                        AgentEventRow.run_id == run_id,
                        AgentEventRow.event_type.in_(
                            ("source.rejected", "source.triage_rejected")
                        ),
                    )
                )
            ).all()
            rejected_urls, failed_owner_keys = _source_failure_feedback(
                (event_type, refs)
                for event_type, refs in failure_event_rows
                if isinstance(refs, dict)
            )
            reusable_results = tuple(
                SearchResult(
                    title=row.title,
                    url=row.url,
                    snippet=row.snippet,
                    published_at=row.published_at,
                    rank=row.rank,
                )
                for row in reusable_rows
                if (
                    normalize_source_url(row.url) not in read_urls
                    and normalize_source_url(row.url) not in rejected_urls
                )
            )
            source_searched_for_question = exists(
                select(SearchResultRow.id)
                .join(SearchQueryRow, SearchResultRow.search_query_id == SearchQueryRow.id)
                .where(
                    SearchQueryRow.run_id == run_id,
                    SearchQueryRow.question_id == question.question_id,
                    or_(
                        SearchResultRow.url == ResearchSourceRow.canonical_url,
                        SearchResultRow.url == ResearchSourceSnapshotRow.final_url,
                    ),
                )
            )
            processed_source_ids = set(
                (
                    await session.scalars(
                        select(distinct(ResearchEvidenceRow.source_id)).where(
                            ResearchEvidenceRow.run_id == run_id,
                            ResearchEvidenceRow.question_id == question.question_id,
                        )
                    )
                ).all()
            )
            processed_event_refs = (
                await session.scalars(
                    select(AgentEventRow.refs).where(
                        AgentEventRow.run_id == run_id,
                        AgentEventRow.event_type.in_(("source.read", "evidence.failed")),
                        AgentEventRow.refs["question_id"].as_string() == question.question_id,
                    )
                )
            ).all()
            for refs in processed_event_refs:
                if not isinstance(refs, dict):
                    continue
                try:
                    processed_source_ids.add(UUID(str(refs.get("source_id"))))
                except (TypeError, ValueError):
                    continue
            reusable_page_conditions: list[ColumnElement[bool]] = [source_searched_for_question]
            if processed_source_ids:
                reusable_page_conditions.append(ResearchSourceRow.id.not_in(processed_source_ids))
            reusable_snapshot_rows = (
                await session.execute(
                    select(ResearchSourceSnapshotRow, ResearchSourceRow)
                    .join(
                        ResearchSourceRow,
                        ResearchSourceRow.id == ResearchSourceSnapshotRow.source_id,
                    )
                    .where(
                        ResearchSourceSnapshotRow.run_id == run_id,
                        ResearchSourceRow.run_id == run_id,
                        *reusable_page_conditions,
                    )
                    .order_by(ResearchSourceSnapshotRow.fetched_at.desc())
                    .limit(8)
                )
            ).all()
            reusable_pages = tuple(
                ReusablePageRef(
                    source_id=source.id,
                    final_url=snapshot.final_url,
                    title=source.title,
                    artifact_uri=snapshot.artifact_uri,
                    content_hash=snapshot.content_hash,
                    fetched_at=snapshot.fetched_at,
                    published_at=snapshot.published_at,
                )
                for snapshot, source in reusable_snapshot_rows
                if snapshot.artifact_uri
            )
            if search_budget_exhausted and not (reusable_results or reusable_pages):
                # Search cannot buy any new candidates. If this question has
                # no cached candidate or fetched artifact either, stop with a
                # precise cause instead of creating a zero-yield loop.
                tool_call.status = "skipped"
                terminal_search_reason = budget_stop_reason or "logical_query_budget_exhausted"
                tool_call.error_code = terminal_search_reason.upper()
                tool_call.retryable = False
                tool_call.finished_at = now
                await self._enter_writing(
                    session,
                    run,
                    reason=terminal_search_reason,
                    summary=_budget_stop_summary(terminal_search_reason),
                )
                return None
            action_usage = dict(run.usage_snapshot)
            action_usage["scheduler_actions"] = (
                _as_int(action_usage.get("scheduler_actions", 0)) + 1
            )
            action_usage["deadline_remaining"] = _deadline_remaining_seconds(run)
            run.usage_snapshot = action_usage
            remaining_provider_requests = max(
                0,
                _as_int(run.budget_snapshot.get("max_provider_requests", 0))
                - _as_int(run.usage_snapshot.get("search_provider_requests", 0)),
            )
            provider_request_allowance = _fair_provider_request_allowance(
                remaining_provider_requests,
                unattempted_questions,
            )
            all_dimensions = tuple(
                (f"{question.question_id}:d{index}", str(criterion))
                for index, criterion in enumerate(question.evidence_requirements, start=1)
            )
            dimensions = tuple(
                sorted(
                    all_dimensions,
                    key=lambda item: (item[0] != unmet_dimension_key, item[0]),
                )
            )
            return ResearchTarget(
                plan_version=run.plan_version,
                question_id=question.question_id,
                question=question.question,
                query=query,
                gap_id=gap.id,
                tool_call_id=tool_call.id,
                source_id_seed=uuid7(),
                alternate_query=alternate_query,
                priority=question.priority,
                attempt_index=attempt_index,
                gap_attempt_index=gap.resolution_attempts,
                first_pass=not executed_families
                and len(eligible) > 1,
                acceptance_dimensions=dimensions,
                used_source_owner_keys=used_owner_keys,
                search_excluded_owner_keys=tuple(
                    sorted(set(used_owner_keys) | set(failed_owner_keys))
                ),
                search_excluded_urls=tuple(sorted(rejected_urls)),
                reusable_results=reusable_results,
                reusable_pages=reusable_pages,
                query_already_executed=query_already_executed,
                search_budget_exhausted=search_budget_exhausted,
                query_family=query_family.value,
                provider_request_allowance=provider_request_allowance,
                baseline_model_tokens=_model_tokens(run),
                baseline_pages_fetched=_as_int(
                    run.usage_snapshot.get("pages_fetched", run.usage_snapshot.get("pages", 0))
                ),
                baseline_pages_extracted=_as_int(
                    run.usage_snapshot.get("pages_extracted", run.usage_snapshot.get("pages", 0))
                ),
                baseline_provider_requests=_as_int(
                    run.usage_snapshot.get("search_provider_requests", 0)
                ),
                action_started_at=now,
                deadline_at=_deadline_at(run),
                cheap_triage_enabled=bool(run.budget_snapshot.get("cheap_triage_enabled", False)),
                owner_acceptance_rates=tuple(
                    (
                        str(owner),
                        round(
                            _as_int(values.get("accepted_evidence", 0))
                            / max(1, _as_int(values.get("extracted", 0))),
                            4,
                        ),
                    )
                    for owner, values in (
                        run.usage_snapshot.get("owner_stats", {})
                        if isinstance(run.usage_snapshot.get("owner_stats", {}), dict)
                        else {}
                    ).items()
                    if isinstance(values, dict)
                ),
            )

    async def research_phase_active(
        self, run_id: UUID, *, worker_task_id: str
    ) -> bool:
        """Return whether an empty scheduler pass should advance, not write.

        ``prepare_target`` can retire one query-exhausted question without
        selecting another target in the same transaction. In that case the
        durable run is still researching and the graph must take another
        scheduler step. Budget/source exhaustion paths enter writing before
        returning ``None``.
        """

        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            return RunPhase(run.phase) == RunPhase.RESEARCHING

    async def replan_requested(
        self, run_id: UUID, *, worker_task_id: str
    ) -> bool:
        """Return whether the scheduler requested a gap-repair replan."""

        async with self._sessions() as session:
            run = await self._locked_run(session, run_id, worker_task_id)
            return bool(run.usage_snapshot.get("replan_requested"))

    async def record_search_results(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
        results: list[SearchResult],
        reused: bool = False,
        provider_requests: int = 0,
        provider_timeouts: int = 0,
        provider_fallbacks: int = 0,
        provider_healthy: int = 0,
        provider_unresponsive: int = 0,
        provider_productive: int = 0,
        latency_ms: int = 0,
    ) -> None:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            tool_call = await session.get(ResearchToolCallRow, target.tool_call_id)
            if tool_call is None:
                raise ResearchLeaseLostError("tool call disappeared")
            if reused:
                query_hash = hashlib.sha256(
                    normalize_search_query(target.query).encode()
                ).hexdigest()
                query_row = await session.scalar(
                    select(SearchQueryRow).where(
                        SearchQueryRow.run_id == run_id,
                        SearchQueryRow.question_id == target.question_id,
                        SearchQueryRow.normalized_hash == query_hash,
                    )
                )
                if query_row is None:
                    query_row = SearchQueryRow(
                        id=uuid7(),
                        run_id=run_id,
                        question_id=target.question_id,
                        tool_call_id=tool_call.id,
                        query=target.query,
                        normalized_hash=query_hash,
                        provider="run_cache",
                        status="succeeded",
                        result_count=len(results),
                        created_at=datetime.now(UTC),
                    )
                    session.add(query_row)
                    await session.flush()
                    for rank, result in enumerate(results, start=1):
                        session.add(
                            SearchResultRow(
                                id=uuid7(),
                                search_query_id=query_row.id,
                                rank=rank,
                                title=result.title,
                                url=result.url,
                                snippet=result.snippet,
                                published_at=result.published_at,
                            )
                        )
                tool_call.status = "succeeded"
                tool_call.result_refs = {"reused": True, "count": len(results)}
                tool_call.finished_at = datetime.now(UTC)
                usage = dict(run.usage_snapshot)
                usage["search_reuses"] = _as_int(usage.get("search_reuses", 0)) + 1
                usage["search_provider_requests"] = _as_int(
                    usage.get("search_provider_requests", 0)
                ) + max(0, provider_requests)
                _update_provider_health_usage(
                    usage,
                    healthy=provider_healthy,
                    unresponsive=provider_unresponsive,
                    productive=provider_productive,
                )
                _update_query_family_usage(
                    usage,
                    family=target.query_family,
                    reused=True,
                    provider_requests=provider_requests,
                    timeouts=provider_timeouts,
                    fallbacks=provider_fallbacks,
                    usable_results=len(results),
                    latency_ms=latency_ms,
                )
                usage["candidate_urls"] = _as_int(usage.get("candidate_urls", 0)) + len(results)
                run.usage_snapshot = _usage_with_resource_pools(run, usage)
                await self._append_event(
                    session,
                    run,
                    event_type="search.reused",
                    public_summary=f"复用本任务已有候选, 获得 {len(results)} 个未读取结果。",
                    refs={"question_id": target.question_id},
                    metrics={"result_count": len(results), "search_budget_consumed": False},
                )
                return
            query_hash = hashlib.sha256(normalize_search_query(target.query).encode()).hexdigest()
            query_row = await session.scalar(
                select(SearchQueryRow).where(
                    SearchQueryRow.run_id == run_id,
                    SearchQueryRow.question_id == target.question_id,
                    SearchQueryRow.normalized_hash == query_hash,
                )
            )
            if query_row is None:
                now = datetime.now(UTC)
                query_row = SearchQueryRow(
                    id=uuid7(),
                    run_id=run_id,
                    question_id=target.question_id,
                    tool_call_id=tool_call.id,
                    query=target.query,
                    normalized_hash=query_hash,
                    provider="searxng",
                    status="succeeded",
                    result_count=len(results),
                    created_at=now,
                )
                session.add(query_row)
                await session.flush()
            else:
                query_row.status = "succeeded"
                query_row.result_count = max(query_row.result_count, len(results))
            existing_urls = set(
                normalize_source_url(str(url))
                for url in (
                    await session.scalars(
                        select(SearchResultRow.url).where(
                            SearchResultRow.search_query_id == query_row.id
                        )
                    )
                ).all()
            )
            next_rank = max(
                [
                    int(rank)
                    for rank in (
                        await session.scalars(
                            select(SearchResultRow.rank).where(
                                SearchResultRow.search_query_id == query_row.id
                            )
                        )
                    ).all()
                ],
                default=0,
            )
            for result in results:
                normalized_result_url = normalize_source_url(result.url)
                if normalized_result_url in existing_urls:
                    continue
                next_rank += 1
                session.add(
                    SearchResultRow(
                        id=uuid7(),
                        search_query_id=query_row.id,
                        rank=next_rank,
                        title=result.title,
                        url=result.url,
                        snippet=result.snippet,
                        published_at=result.published_at,
                    )
                )
                existing_urls.add(normalized_result_url)
            query_row.result_count = len(existing_urls)
            tool_call.status = "succeeded"
            tool_call.result_refs = {"search_query_id": str(query_row.id), "count": len(results)}
            tool_call.finished_at = datetime.now(UTC)
            usage = dict(run.usage_snapshot)
            usage["searches"] = int(usage.get("searches", 0)) + 1
            usage["logical_queries"] = (
                _as_int(usage.get("logical_queries", usage.get("searches", 0) - 1)) + 1
            )
            usage["search_provider_requests"] = _as_int(
                usage.get("search_provider_requests", 0)
            ) + max(0, provider_requests)
            _update_provider_health_usage(
                usage,
                healthy=provider_healthy,
                unresponsive=provider_unresponsive,
                productive=provider_productive,
            )
            if provider_requests > 0 and (
                provider_healthy > 0 or provider_productive > 0 or bool(results)
            ):
                _mark_query_family_executed(
                    usage,
                    question_id=target.question_id,
                    family=target.query_family,
                )
            _update_query_family_usage(
                usage,
                family=target.query_family,
                reused=False,
                provider_requests=provider_requests,
                timeouts=provider_timeouts,
                fallbacks=provider_fallbacks,
                usable_results=len(results),
                latency_ms=latency_ms,
            )
            _update_provider_observability_usage(
                usage,
                {
                    "provider_requests": provider_requests,
                    "healthy_responses": provider_healthy,
                    "unresponsive_responses": provider_unresponsive,
                    "timeout_count": provider_timeouts,
                    "network_error_count": 0,
                    "http_error_count": 0,
                    "empty_response_count": 0,
                    "invalid_response_count": 0,
                    "fallback_attempts": provider_fallbacks,
                    "fallback_successes": max(provider_productive - provider_healthy, 0),
                    "circuit_open_count": 0,
                },
            )
            usage["candidate_urls"] = _as_int(usage.get("candidate_urls", 0)) + len(results)
            run.usage_snapshot = _usage_with_resource_pools(run, usage)
            await self._append_event(
                session,
                run,
                event_type="search.completed",
                public_summary=f"搜索完成, 获得 {len(results)} 个候选结果。",
                refs={
                    "search_query_id": str(query_row.id),
                    "question_id": target.question_id,
                },
                metrics={
                    "provider": "SearXNG",
                    "result_count": len(results),
                    "query_family": target.query_family,
                    "provider_requests": provider_requests,
                    "timeouts": provider_timeouts,
                    "fallbacks": provider_fallbacks,
                    "healthy_responses": provider_healthy,
                    "unresponsive_responses": provider_unresponsive,
                    "productive_responses": provider_productive,
                    "fallback_attempts": provider_fallbacks,
                    "fallback_successes": max(provider_productive - provider_healthy, 0),
                    "latency_ms": latency_ms,
                },
            )

    async def record_tool_failure(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
        error_code: str,
        retryable: bool,
        details: Mapping[str, object] | None = None,
        provider_requests: int = 0,
        provider_timeouts: int = 0,
        provider_fallbacks: int = 0,
        provider_healthy: int = 0,
        provider_unresponsive: int = 0,
        provider_productive: int = 0,
        latency_ms: int = 0,
    ) -> None:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            tool_call = await session.get(ResearchToolCallRow, target.tool_call_id)
            if tool_call is not None:
                tool_call.status = "failed"
                tool_call.error_code = error_code[:100]
                tool_call.retryable = retryable
                tool_call.finished_at = datetime.now(UTC)
            usage_snapshot = dict(run.usage_snapshot)
            usage_snapshot["search_provider_failures"] = (
                int(usage_snapshot.get("search_provider_failures", 0)) + 1
            )
            usage_snapshot["searches"] = _as_int(usage_snapshot.get("searches", 0)) + 1
            usage_snapshot["logical_queries"] = (
                _as_int(
                    usage_snapshot.get("logical_queries", usage_snapshot.get("searches", 0) - 1)
                )
                + 1
            )
            usage_snapshot["search_provider_requests"] = _as_int(
                usage_snapshot.get("search_provider_requests", 0)
            ) + max(0, provider_requests)
            _update_provider_health_usage(
                usage_snapshot,
                healthy=provider_healthy,
                unresponsive=provider_unresponsive,
                productive=provider_productive,
            )
            raw_failure_metrics = details.get("metrics") if details else None
            failure_metrics = (
                raw_failure_metrics
                if isinstance(raw_failure_metrics, Mapping)
                else {
                    "provider_requests": provider_requests,
                    "healthy_responses": provider_healthy,
                    "unresponsive_responses": provider_unresponsive,
                    "timeout_count": provider_timeouts,
                    "network_error_count": 0,
                    "http_error_count": 0,
                    "empty_response_count": 0,
                    "invalid_response_count": 0,
                    "fallback_attempts": provider_fallbacks,
                    "fallback_successes": max(provider_productive - provider_healthy, 0),
                    "circuit_open_count": 0,
                }
            )
            _update_provider_observability_usage(usage_snapshot, failure_metrics)
            # A transport-only failure must remain retryable.  Counting its
            # family as executed would make the scheduler report source-space
            # exhaustion even though no healthy provider response was seen.
            if provider_requests > 0 and (
                provider_healthy > 0 or provider_productive > 0
            ):
                _mark_query_family_executed(
                    usage_snapshot,
                    question_id=target.question_id,
                    family=target.query_family,
                )
            _update_query_family_usage(
                usage_snapshot,
                family=target.query_family,
                reused=False,
                provider_requests=provider_requests,
                timeouts=provider_timeouts,
                fallbacks=provider_fallbacks,
                usable_results=0,
                latency_ms=latency_ms,
            )
            if details:
                # Keep only bounded, provider-safe diagnostics; never persist raw
                # response bodies or request headers.
                usage_snapshot["last_search_failure"] = {
                    "code": error_code[:100],
                    "details": dict(details),
                }
            run.usage_snapshot = _usage_with_resource_pools(run, usage_snapshot)
            await self._append_event(
                session,
                run,
                event_type="tool.failed",
                public_summary="Web Search 执行失败, 公开轨迹仅记录安全错误码。",
                refs={"question_id": target.question_id, "error_code": error_code[:100]},
                metrics=dict(details) if details else None,
            )

    async def record_extraction_started(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
        source_id: UUID,
    ) -> None:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            await self._append_event(
                session,
                run,
                event_type="evidence.extraction_started",
                public_summary="正在调用模型从网页正文提取可验证证据。",
                refs={
                    "question_id": target.question_id,
                    "source_id": str(source_id),
                    "plan_version": target.plan_version,
                },
            )

    async def page_already_processed(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
        requested_url: str,
        page: ReadPage,
    ) -> bool:
        """Check final URL and content identity before spending extraction budget.

        Search-result URLs are only hints: redirects may reveal that an unseen
        HTTP URL is the HTTPS form of a page already processed for this
        question. Content hashes catch mirrors and alternate canonical URLs.
        """

        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            processed_source_ids = set(
                (
                    await session.scalars(
                        select(distinct(ResearchEvidenceRow.source_id)).where(
                            ResearchEvidenceRow.run_id == run_id,
                            ResearchEvidenceRow.question_id == target.question_id,
                        )
                    )
                ).all()
            )
            event_refs = (
                await session.scalars(
                    select(AgentEventRow.refs).where(
                        AgentEventRow.run_id == run_id,
                        AgentEventRow.event_type.in_(("source.read", "evidence.failed")),
                        AgentEventRow.refs["question_id"].as_string() == target.question_id,
                    )
                )
            ).all()
            for refs in event_refs:
                if not isinstance(refs, dict):
                    continue
                try:
                    processed_source_ids.add(UUID(str(refs.get("source_id"))))
                except (TypeError, ValueError):
                    continue
            if not processed_source_ids:
                return False
            identities = {
                normalize_source_url(requested_url),
                normalize_source_url(page.final_url),
            }
            known_sources = (
                await session.scalars(
                    select(ResearchSourceRow).where(
                        ResearchSourceRow.run_id == run_id,
                        ResearchSourceRow.id.in_(processed_source_ids),
                    )
                )
            ).all()
            duplicate = next(
                (
                    source
                    for source in known_sources
                    if normalize_source_url(source.canonical_url) in identities
                    or source.content_hash == page.content_hash
                ),
                None,
            )
            if duplicate is None:
                return False
            usage = dict(run.usage_snapshot)
            usage["duplicate_pages_skipped"] = _as_int(usage.get("duplicate_pages_skipped", 0)) + 1
            run.usage_snapshot = _usage_with_resource_pools(run, usage)
            await self._append_event(
                session,
                run,
                event_type="source.duplicate_skipped",
                public_summary=(
                    "候选页面已在当前问题中处理; 已记录本次抓取成本并跳过重复抽取。"
                ),
                refs={
                    "question_id": target.question_id,
                    "source_id": str(duplicate.id),
                    "requested_url": requested_url[:1000],
                    "final_url": page.final_url[:1000],
                    "dedupe_layers": ["source_id", "final_url", "content_hash"],
                },
                metrics={"fetch_budget_consumed": True, "extraction_budget_consumed": False},
            )
            return True

    async def record_page(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
        source_id: UUID,
        page: ReadPage,
        artifact_uri: str,
        evidence: list[ScoredEvidence],
        usage: TokenUsage,
        context_manifest: dict[str, int | bool],
        attempt_id: UUID | None = None,
    ) -> tuple[int, int]:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            canonical_url = normalize_source_url(page.final_url)
            url_hash = hashlib.sha256(canonical_url.encode()).hexdigest()
            source = await session.scalar(
                select(ResearchSourceRow).where(
                    ResearchSourceRow.run_id == run_id,
                    or_(
                        ResearchSourceRow.url_hash == url_hash,
                        ResearchSourceRow.content_hash == page.content_hash,
                    ),
                )
            )
            reliability = evidence[0].source_reliability if evidence else 0.72
            if source is None:
                source = ResearchSourceRow(
                    id=source_id,
                    run_id=run_id,
                    canonical_url=canonical_url,
                    url_hash=url_hash,
                    domain=(urlsplit(canonical_url).hostname or "unknown")[:255],
                    source_owner_key=source_owner_key(canonical_url),
                    title=page.title,
                    source_type="webpage",
                    reliability=reliability,
                    artifact_uri=artifact_uri,
                    content_hash=page.content_hash,
                    char_count=len(page.clean_text),
                    fetched_at=page.fetched_at,
                )
                session.add(source)
                await session.flush()
            snapshot = await session.scalar(
                select(ResearchSourceSnapshotRow).where(
                    ResearchSourceSnapshotRow.source_id == source.id,
                    ResearchSourceSnapshotRow.content_hash == page.content_hash,
                )
            )
            if snapshot is None:
                snapshot = ResearchSourceSnapshotRow(
                    id=uuid7(),
                    run_id=run_id,
                    source_id=source.id,
                    final_url=canonical_url,
                    fetched_at=page.fetched_at,
                    published_at=page.published_at,
                    content_hash=page.content_hash,
                    parser_version="web-reader-v1",
                    artifact_uri=artifact_uri,
                    char_count=len(page.clean_text),
                )
                session.add(snapshot)
                await session.flush()
            inserted = 0
            accepted = 0
            graph_claim_ids: set[UUID] = set()
            graph_chunk_ids: set[UUID] = set()
            now = datetime.now(UTC)
            for item in evidence:
                candidate = item.candidate
                evidence_hash = hashlib.sha256(
                    (
                        f"{target.question_id}\n{source.id}\n{candidate.claim}\n"
                        f"{candidate.exact_quote}\n{candidate.relation.value}"
                    ).encode()
                ).hexdigest()
                exists = await session.scalar(
                    select(ResearchEvidenceRow.id).where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.evidence_hash == evidence_hash,
                    )
                )
                if exists is not None:
                    continue

                claim_hash = claim_fingerprint(candidate.claim)
                claim = await session.scalar(
                    select(ResearchClaimRow).where(
                        ResearchClaimRow.run_id == run_id,
                        ResearchClaimRow.question_id == target.question_id,
                        ResearchClaimRow.claim_hash == claim_hash,
                    )
                )
                chunk_window = build_evidence_chunk(page.clean_text, candidate.exact_quote)
                accepted_by_graph = item.accepted and chunk_window is not None
                rejection_reason = item.rejection_reason
                if item.accepted and chunk_window is None:
                    rejection_reason = "quote_not_located_for_graph"
                if claim is None:
                    dimension_key = _evidence_dimension_key(
                        target,
                        candidate.dimension_key,
                        f"{candidate.claim} {candidate.exact_quote}",
                    )
                    criterion_by_dimension = dict(target.acceptance_dimensions)
                    claim = ResearchClaimRow(
                        id=uuid7(),
                        run_id=run_id,
                        plan_version=target.plan_version,
                        question_id=target.question_id,
                        dimension_key=dimension_key,
                        atomic_claim=candidate.claim,
                        claim_hash=claim_hash,
                        claim_type=infer_claim_type(
                            criterion_by_dimension.get(dimension_key, candidate.claim)
                        ),
                        importance=0.8 if accepted_by_graph else 0.5,
                        status=derive_claim_status(
                            has_accepted_evidence=accepted_by_graph,
                            has_refuting_evidence=(
                                accepted_by_graph and candidate.relation.value == "refutes"
                            ),
                            independent_source_count=1 if accepted_by_graph else 0,
                        ),
                        confidence=item.evidence_score if accepted_by_graph else 0.0,
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(claim)
                    await session.flush()
                elif accepted_by_graph:
                    claim.importance = max(claim.importance, 0.8)
                    claim.confidence = max(claim.confidence, item.evidence_score)
                    claim.updated_at = now

                chunk: ResearchSourceChunkRow | None = None
                if chunk_window is not None:
                    chunk = await session.scalar(
                        select(ResearchSourceChunkRow).where(
                            ResearchSourceChunkRow.snapshot_id == snapshot.id,
                            ResearchSourceChunkRow.chunk_hash == chunk_window.chunk_hash,
                        )
                    )
                    if chunk is None:
                        chunk = ResearchSourceChunkRow(
                            id=uuid7(),
                            run_id=run_id,
                            snapshot_id=snapshot.id,
                            heading_path=None,
                            char_start=chunk_window.char_start,
                            char_end=chunk_window.char_end,
                            text=chunk_window.text,
                            token_count=chunk_window.token_count,
                            chunk_hash=chunk_window.chunk_hash,
                        )
                        session.add(chunk)
                        await session.flush()
                evidence_row = ResearchEvidenceRow(
                    id=uuid7(),
                    run_id=run_id,
                    plan_version=target.plan_version,
                    question_id=target.question_id,
                    source_id=source.id,
                    claim_id=claim.id,
                    snapshot_id=snapshot.id,
                    chunk_id=chunk.id if chunk is not None else None,
                    claim=candidate.claim,
                    exact_quote=candidate.exact_quote,
                    relation=candidate.relation.value,
                    relevance=candidate.relevance,
                    confidence=candidate.confidence,
                    source_reliability=item.source_reliability,
                    evidence_score=item.evidence_score,
                    accepted=accepted_by_graph,
                    rejection_reason=rejection_reason,
                    evidence_hash=evidence_hash,
                    created_at=now,
                )
                session.add(evidence_row)
                await session.flush()
                independent_source_count = await session.scalar(
                    select(func.count(distinct(ResearchSourceRow.source_owner_key)))
                    .select_from(ResearchEvidenceRow)
                    .join(
                        ResearchSourceRow,
                        ResearchEvidenceRow.source_id == ResearchSourceRow.id,
                    )
                    .where(
                        ResearchEvidenceRow.claim_id == claim.id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                )
                refuting_evidence_count = await session.scalar(
                    select(func.count(ResearchEvidenceRow.id)).where(
                        ResearchEvidenceRow.claim_id == claim.id,
                        ResearchEvidenceRow.accepted.is_(True),
                        ResearchEvidenceRow.relation == "refutes",
                    )
                )
                claim.status = derive_claim_status(
                    has_accepted_evidence=int(independent_source_count or 0) > 0,
                    has_refuting_evidence=int(refuting_evidence_count or 0) > 0,
                    independent_source_count=int(independent_source_count or 0),
                )
                graph_claim_ids.add(claim.id)
                if chunk is not None:
                    graph_chunk_ids.add(chunk.id)
                inserted += 1
                accepted += int(accepted_by_graph)
            relation_stats = RelationRefreshStats()
            if inserted:
                relation_stats = await refresh_question_relations(
                    session,
                    run_id=run_id,
                    question_id=target.question_id,
                    touched_claim_ids=graph_claim_ids,
                )
            usage_snapshot = dict(run.usage_snapshot)
            usage_snapshot["pages_extracted"] = (
                _as_int(usage_snapshot.get("pages_extracted", usage_snapshot.get("pages", 0))) + 1
            )
            usage_snapshot["pages"] = usage_snapshot["pages_extracted"]
            usage_snapshot["extraction_calls"] = (
                _as_int(usage_snapshot.get("extraction_calls", 0))
                + int(not bool(context_manifest.get("cache_hit", False)))
            )
            if bool(context_manifest.get("cache_hit", False)):
                usage_snapshot["extraction_cache_hits"] = (
                    _as_int(usage_snapshot.get("extraction_cache_hits", 0)) + 1
                )
            if accepted == 0:
                usage_snapshot["zero_yield_pages"] = (
                    _as_int(usage_snapshot.get("zero_yield_pages", 0)) + 1
                )
            usage_snapshot["extraction_slots_reserved"] = max(
                0, _as_int(usage_snapshot.get("extraction_slots_reserved", 0)) - 1
            )
            source_role = classify_source_role(page.final_url, text=page.clean_text[:2_000])
            source_stats_raw = usage_snapshot.get("source_type_stats", {})
            source_stats = dict(source_stats_raw) if isinstance(source_stats_raw, dict) else {}
            role_raw = source_stats.get(source_role, {})
            role_stats = dict(role_raw) if isinstance(role_raw, dict) else {}
            role_stats["extracted"] = _as_int(role_stats.get("extracted", 0)) + 1
            role_stats["accepted_evidence"] = (
                _as_int(role_stats.get("accepted_evidence", 0)) + accepted
            )
            source_stats[source_role] = role_stats
            usage_snapshot["source_type_stats"] = source_stats
            owner = source_owner_key(source.canonical_url)
            owner_stats_raw = usage_snapshot.get("owner_stats", {})
            owner_stats = dict(owner_stats_raw) if isinstance(owner_stats_raw, dict) else {}
            owner_raw = owner_stats.get(owner, {})
            owner_metrics = dict(owner_raw) if isinstance(owner_raw, dict) else {}
            owner_metrics["extracted"] = _as_int(owner_metrics.get("extracted", 0)) + 1
            owner_metrics["accepted_evidence"] = (
                _as_int(owner_metrics.get("accepted_evidence", 0)) + accepted
            )
            owner_stats[owner] = owner_metrics
            usage_snapshot["owner_stats"] = owner_stats
            usage_snapshot["evidence_input_tokens"] = (
                int(usage_snapshot.get("evidence_input_tokens", 0)) + usage.input_tokens
            )
            usage_snapshot["evidence_output_tokens"] = (
                int(usage_snapshot.get("evidence_output_tokens", 0)) + usage.output_tokens
            )
            usage_snapshot["evidence_total_tokens"] = (
                int(usage_snapshot.get("evidence_total_tokens", 0)) + usage.total_tokens
            )
            usage_snapshot["model_tokens"] = _model_tokens_from_usage(usage_snapshot)
            run.usage_snapshot = _usage_with_resource_pools(run, usage_snapshot)
            if attempt_id is not None:
                await self._settle_reservation_row(
                    session,
                    run,
                    attempt_id=attempt_id,
                    actual_total=usage.total_tokens,
                    usage_estimated=usage.accuracy != UsageAccuracy.EXACT,
                )
            await self._append_event(
                session,
                run,
                event_type="source.read",
                public_summary=f"已读取来源: {page.title[:300]}",
                refs={
                    "source_id": str(source.id),
                    "question_id": target.question_id,
                    "plan_version": target.plan_version,
                },
                metrics={"clean_chars": len(page.clean_text), "truncated": page.truncated},
            )
            await self._append_event(
                session,
                run,
                event_type="context.assembled",
                public_summary="Context Manager 已选择当前问题与必要网页片段。",
                refs={"source_id": str(source.id), "question_id": target.question_id},
                metrics=cast(dict[str, object], context_manifest),
            )
            await self._append_event(
                session,
                run,
                event_type="evidence.graph_updated",
                public_summary=(
                    "Evidence Graph synchronized atomic claims, source snapshot, and quote chunks."
                ),
                refs={
                    "source_id": str(source.id),
                    "snapshot_id": str(snapshot.id),
                    "question_id": target.question_id,
                },
                metrics={
                    "claim_count": len(graph_claim_ids),
                    "chunk_count": len(graph_chunk_ids),
                    "bound_evidence_count": inserted,
                },
            )
            if any(
                (
                    relation_stats.detected_edges,
                    relation_stats.deleted_edges,
                    relation_stats.detected_conflicts,
                    relation_stats.dismissed_conflicts,
                    relation_stats.reopened_conflicts,
                )
            ):
                await self._append_event(
                    session,
                    run,
                    event_type="evidence.relationships_updated",
                    public_summary=(
                        "Evidence Graph updated Claim relations and disclosed "
                        "independent-source conflicts."
                    ),
                    refs={"question_id": target.question_id},
                    metrics={
                        "examined_pairs": relation_stats.examined_pairs,
                        "detected_edges": relation_stats.detected_edges,
                        "created_edges": relation_stats.created_edges,
                        "deleted_edges": relation_stats.deleted_edges,
                        "detected_conflicts": relation_stats.detected_conflicts,
                        "created_conflicts": relation_stats.created_conflicts,
                        "dismissed_conflicts": relation_stats.dismissed_conflicts,
                        "reopened_conflicts": relation_stats.reopened_conflicts,
                    },
                )
            await self._append_event(
                session,
                run,
                event_type="evidence.extracted",
                public_summary=f"提取 {inserted} 条候选证据, 其中 {accepted} 条通过验证。",
                refs={"source_id": str(source.id), "question_id": target.question_id},
                metrics={"candidate_count": inserted, "accepted_count": accepted},
            )
            return inserted, accepted

    async def record_extraction_failure(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
        source_id: UUID,
        page: ReadPage,
        artifact_uri: str,
        error_code: str,
        detail_code: str | None = None,
        usage: TokenUsage | None = None,
        attempt_id: UUID | None = None,
    ) -> None:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            canonical_url = normalize_source_url(page.final_url)
            url_hash = hashlib.sha256(canonical_url.encode()).hexdigest()
            source = await session.scalar(
                select(ResearchSourceRow).where(
                    ResearchSourceRow.run_id == run_id,
                    or_(
                        ResearchSourceRow.url_hash == url_hash,
                        ResearchSourceRow.content_hash == page.content_hash,
                    ),
                )
            )
            if source is None:
                source = ResearchSourceRow(
                    id=source_id,
                    run_id=run_id,
                    canonical_url=canonical_url,
                    url_hash=url_hash,
                    domain=(urlsplit(canonical_url).hostname or "unknown")[:255],
                    source_owner_key=source_owner_key(canonical_url),
                    title=page.title,
                    source_type="webpage",
                    reliability=0.72,
                    artifact_uri=artifact_uri,
                    content_hash=page.content_hash,
                    char_count=len(page.clean_text),
                    fetched_at=page.fetched_at,
                )
                session.add(source)
                await session.flush()
            snapshot = await session.scalar(
                select(ResearchSourceSnapshotRow).where(
                    ResearchSourceSnapshotRow.source_id == source.id,
                    ResearchSourceSnapshotRow.content_hash == page.content_hash,
                )
            )
            if snapshot is None:
                snapshot = ResearchSourceSnapshotRow(
                    id=uuid7(),
                    run_id=run_id,
                    source_id=source.id,
                    final_url=canonical_url,
                    fetched_at=page.fetched_at,
                    published_at=page.published_at,
                    content_hash=page.content_hash,
                    parser_version="web-reader-v1",
                    artifact_uri=artifact_uri,
                    char_count=len(page.clean_text),
                )
                session.add(snapshot)
                await session.flush()
            usage_snapshot = dict(run.usage_snapshot)
            usage_snapshot["pages_extracted"] = (
                _as_int(usage_snapshot.get("pages_extracted", usage_snapshot.get("pages", 0))) + 1
            )
            usage_snapshot["pages"] = usage_snapshot["pages_extracted"]
            usage_snapshot["extraction_calls"] = (
                _as_int(usage_snapshot.get("extraction_calls", 0)) + 1
            )
            usage_snapshot["zero_yield_pages"] = (
                _as_int(usage_snapshot.get("zero_yield_pages", 0)) + 1
            )
            usage_snapshot["extraction_slots_reserved"] = max(
                0, _as_int(usage_snapshot.get("extraction_slots_reserved", 0)) - 1
            )
            usage_snapshot["evidence_extraction_failures"] = (
                int(usage_snapshot.get("evidence_extraction_failures", 0)) + 1
            )
            if usage is not None:
                usage_snapshot["evidence_input_tokens"] = (
                    int(usage_snapshot.get("evidence_input_tokens", 0)) + usage.input_tokens
                )
                usage_snapshot["evidence_output_tokens"] = (
                    int(usage_snapshot.get("evidence_output_tokens", 0)) + usage.output_tokens
                )
                usage_snapshot["evidence_total_tokens"] = (
                    int(usage_snapshot.get("evidence_total_tokens", 0)) + usage.total_tokens
                )
            usage_snapshot["model_tokens"] = _model_tokens_from_usage(usage_snapshot)
            run.usage_snapshot = _usage_with_resource_pools(run, usage_snapshot)
            if attempt_id is not None:
                if usage is not None:
                    await self._settle_reservation_row(
                        session,
                        run,
                        attempt_id=attempt_id,
                        actual_total=usage.total_tokens,
                        usage_estimated=usage.accuracy != UsageAccuracy.EXACT,
                    )
                else:
                    # The provider may still have billed the attempt; never
                    # refund an unknown outcome to zero.
                    row = await self._reservation_row(session, run_id, attempt_id)
                    if row is not None and row.status == "reserved":
                        row.status = "uncertain"
                        await self._append_event(
                            session,
                            run,
                            event_type="budget.reservation_uncertain",
                            public_summary=("失败调用未返回用量; 保留保守占用直至对账。"),
                            refs={
                                "question_id": row.question_id,
                                "attempt_id": str(attempt_id),
                            },
                            metrics={"reserved_total_tokens": row.reserved_total},
                        )
            await self._append_event(
                session,
                run,
                event_type="source.read",
                public_summary=f"已读取来源: {page.title[:300]}",
                refs={"source_id": str(source.id), "question_id": target.question_id},
                metrics={"clean_chars": len(page.clean_text), "truncated": page.truncated},
            )
            await self._append_event(
                session,
                run,
                event_type="context.assembled",
                public_summary="Context Manager 已选择当前问题与必要网页片段。",
                refs={"source_id": str(source.id), "question_id": target.question_id},
                metrics={
                    "source_chars": len(page.clean_text),
                    "selected_chars": min(len(page.clean_text), 14_000),
                    "truncated": len(page.clean_text) > 14_000,
                },
            )
            await self._append_event(
                session,
                run,
                event_type="evidence.failed",
                public_summary="该来源的结构化证据抽取失败; 本轮保留其他已验证证据。",
                refs={
                    "source_id": str(source.id),
                    "question_id": target.question_id,
                    "error_code": error_code[:100],
                    "plan_version": target.plan_version,
                    **({"detail_code": detail_code[:100]} if detail_code is not None else {}),
                },
            )

    async def record_page_failure(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
        url: str,
        error_code: str,
        latency_ms: int = 0,
    ) -> None:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            usage_snapshot = dict(run.usage_snapshot)
            owners_raw = usage_snapshot.get("page_slots_reserved_by_worker", {})
            owners = dict(owners_raw) if isinstance(owners_raw, dict) else {}
            owned = _as_int(owners.get(worker_task_id, 0))
            settled = 1 if not owners else min(1, max(0, owned))
            if owners:
                owners[worker_task_id] = max(0, owned - settled)
                if owners[worker_task_id] == 0:
                    owners.pop(worker_task_id, None)
                usage_snapshot["page_slots_reserved_by_worker"] = owners
            usage_snapshot["page_slots_reserved"] = max(
                0, _as_int(usage_snapshot.get("page_slots_reserved", 0)) - settled
            )
            # A failed HTTP attempt consumes the separate attempt pool, but
            # does not consume the successful-page evidence budget. This keeps
            # transient 403/timeout failures from starving readable sources.
            usage_snapshot["page_fetch_attempts"] = (
                _as_int(usage_snapshot.get("page_fetch_attempts", 0)) + 1
            )
            usage_snapshot["page_fetch_latency_ms"] = _as_int(
                usage_snapshot.get("page_fetch_latency_ms", 0)
            ) + max(0, latency_ms)
            usage_snapshot["page_read_failures"] = (
                int(usage_snapshot.get("page_read_failures", 0)) + 1
            )
            run.usage_snapshot = _usage_with_resource_pools(run, usage_snapshot)
            await self._append_event(
                session,
                run,
                event_type="source.rejected",
                public_summary="候选来源未通过安全读取或内容校验。",
                refs={
                    "question_id": target.question_id,
                    "domain": (urlsplit(url).hostname or "unknown")[:255],
                    "url": url[:1000],
                    "error_code": error_code[:100],
                },
                metrics={"latency_ms": max(0, latency_ms), "fetch_budget_consumed": True},
            )

    async def finish_iteration(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
        attempt_outcome: str = "evidence_gained",
    ) -> IterationEvaluation:
        async with self._sessions() as session, session.begin():
            run = await self._locked_run(session, run_id, worker_task_id)
            previous_quality = dict(run.quality_snapshot)
            if attempt_outcome == "question_budget_exhausted":
                usage = dict(run.usage_snapshot)
                exhausted = dict(usage.get("question_budget_exhausted_by_question", {}))
                exhausted[target.question_id] = {
                    "reason": "hard_question_token_limit",
                    "recorded_plan_version": target.plan_version,
                }
                usage["question_budget_exhausted_by_question"] = exhausted
                run.usage_snapshot = _usage_with_resource_pools(run, usage)
                plan_item = await session.scalar(
                    select(ResearchPlanItemRow).where(
                        ResearchPlanItemRow.run_id == run_id,
                        ResearchPlanItemRow.plan_version == target.plan_version,
                        ResearchPlanItemRow.question_id == target.question_id,
                    )
                )
                accepted_for_question = int(
                    await session.scalar(
                        select(func.count(ResearchEvidenceRow.id)).where(
                            ResearchEvidenceRow.run_id == run_id,
                            ResearchEvidenceRow.question_id == target.question_id,
                            ResearchEvidenceRow.accepted.is_(True),
                        )
                    )
                    or 0
                )
                question_status = "partial" if accepted_for_question else "blocked"
                if plan_item is not None:
                    plan_item.status = question_status
                gap = await session.get(ResearchGapRow, target.gap_id)
                if gap is not None:
                    gap.status = "open"
                    gap.description = (
                        "本题 Token 预算已达到保护阈值; 已让出后续检索, 避免在同一问题上循环消耗。"
                    )
                    gap.updated_at = datetime.now(UTC)
                await self._append_event(
                    session,
                    run,
                    event_type="budget.question_exhausted",
                    public_summary=(
                        f"问题 {target.question_id} 达到本题 Token 预算保护阈值; 转处理其他问题。"
                    ),
                    refs={"question_id": target.question_id},
                    metrics={"research_attempt_consumed": False},
                )
                return IterationEvaluation(
                    continue_research=True,
                    decision="continue_plan",
                    stop_reason=None,
                    question_status=question_status,
                    coverage=float(previous_quality.get("coverage", 0.0) or 0.0),
                )
            if attempt_outcome == "yield_question":
                usage = dict(run.usage_snapshot)
                usage["question_yields"] = _as_int(usage.get("question_yields", 0)) + 1
                run.usage_snapshot = _usage_with_resource_pools(run, usage)
                await self._append_event(
                    session,
                    run,
                    event_type="budget.question_yielded",
                    public_summary="当前问题让出追加研究机会; 先处理尚未尝试的问题。",
                    refs={"question_id": target.question_id},
                    metrics={"research_attempt_consumed": False},
                )
                return IterationEvaluation(
                    continue_research=True,
                    decision="continue_plan",
                    stop_reason=None,
                    question_status="active",
                    coverage=float(previous_quality.get("coverage", 0.0)),
                )
            accepted_for_question = int(
                await session.scalar(
                    select(func.count(ResearchEvidenceRow.id)).where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.question_id == target.question_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                )
                or 0
            )
            sources_for_question = int(
                await session.scalar(
                    select(func.count(distinct(ResearchSourceRow.source_owner_key)))
                    .join(
                        ResearchEvidenceRow,
                        ResearchEvidenceRow.source_id == ResearchSourceRow.id,
                    )
                    .where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.question_id == target.question_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                )
                or 0
            )
            plan_item = await session.scalar(
                select(ResearchPlanItemRow).where(
                    ResearchPlanItemRow.run_id == run_id,
                    ResearchPlanItemRow.plan_version == target.plan_version,
                    ResearchPlanItemRow.question_id == target.question_id,
                )
            )
            gap = await session.get(ResearchGapRow, target.gap_id)
            target_dimension_counts = (
                await session.execute(
                    select(
                        ResearchClaimRow.dimension_key,
                        func.count(ResearchEvidenceRow.id),
                        func.count(distinct(ResearchSourceRow.source_owner_key)),
                    )
                    .join(
                        ResearchEvidenceRow,
                        ResearchEvidenceRow.claim_id == ResearchClaimRow.id,
                    )
                    .join(
                        ResearchSourceRow,
                        ResearchSourceRow.id == ResearchEvidenceRow.source_id,
                    )
                    .where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.question_id == target.question_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                    .group_by(ResearchClaimRow.dimension_key)
                )
            ).all()
            target_counts_by_dimension = {
                str(dimension_key): (int(evidence_count), int(owner_count))
                for dimension_key, evidence_count, owner_count in target_dimension_counts
            }
            target_claims = (
                await session.scalars(
                    select(ResearchClaimRow).where(
                        ResearchClaimRow.run_id == run_id,
                        ResearchClaimRow.question_id == target.question_id,
                    )
                )
            ).all()
            target_high_risk_dimension_keys = {
                claim.dimension_key for claim in target_claims if _claim_is_high_risk(claim)
            }
            requirements_met = _requirements_satisfied(
                target.question_id,
                [str(value) for value in (plan_item.evidence_requirements if plan_item else [])],
                target_counts_by_dimension,
                accepted_for_question=accepted_for_question,
                high_risk_dimension_keys=target_high_risk_dimension_keys,
            )
            usage = dict(run.usage_snapshot)
            technical_failures = dict(usage.get("technical_failures_by_question", {}))
            technical_failure_count = _as_int(technical_failures.get(target.question_id, 0))
            technical_outcome = attempt_outcome == "provider_error"
            if technical_outcome:
                technical_failure_count += 1
                technical_failures[target.question_id] = technical_failure_count
                usage["technical_failures_by_question"] = technical_failures
                usage["technical_retries"] = _as_int(usage.get("technical_retries", 0)) + 1
            elif gap is not None:
                gap.resolution_attempts += 1
            attempts = gap.resolution_attempts if gap is not None else _MAX_GAP_ATTEMPTS
            executed_families = _executed_query_families(usage, target.question_id)
            query_strategy_exhausted = (
                not technical_outcome
                and not requirements_met
                and len(executed_families) >= _query_strategy_limit(target.question)
            )
            if query_strategy_exhausted:
                exhausted_sources = dict(
                    usage.get("source_space_exhausted_by_question", {})
                )
                exhausted_sources[target.question_id] = True
                usage["source_space_exhausted_by_question"] = exhausted_sources
                run.usage_snapshot = _usage_with_resource_pools(run, usage)
            retry_current = (
                technical_failure_count < 3
                if technical_outcome
                else not requirements_met and not query_strategy_exhausted
            )
            if technical_outcome:
                question_status = "technical_retry" if retry_current else "technical_degraded"
            elif retry_current:
                question_status = "retrying"
            elif requirements_met:
                question_status = "researched"
            elif accepted_for_question > 0:
                question_status = "partial"
            else:
                question_status = "blocked"
            if plan_item is not None:
                plan_item.status = (
                    question_status
                    if technical_outcome
                    else "active"
                    if retry_current
                    else question_status
                )
            if query_strategy_exhausted:
                exhausted_strategies = dict(usage.get("query_strategy_exhausted_by_question", {}))
                exhausted_strategies[target.question_id] = True
                usage["query_strategy_exhausted_by_question"] = exhausted_strategies
            if gap is not None:
                gap.status = "open" if retry_current else "resolved" if requirements_met else "open"
                gap.updated_at = datetime.now(UTC)

            plan_items = (
                await session.scalars(
                    select(ResearchPlanItemRow)
                    .where(
                        ResearchPlanItemRow.run_id == run_id,
                        ResearchPlanItemRow.plan_version == run.plan_version,
                    )
                    .order_by(ResearchPlanItemRow.priority, ResearchPlanItemRow.question_id)
                )
            ).all()
            total_questions = len(plan_items)
            question_counts = (
                await session.execute(
                    select(
                        ResearchEvidenceRow.question_id,
                        func.count(ResearchEvidenceRow.id),
                        func.count(distinct(ResearchSourceRow.source_owner_key)),
                    )
                    .join(
                        ResearchSourceRow,
                        ResearchSourceRow.id == ResearchEvidenceRow.source_id,
                    )
                    .where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                    .group_by(ResearchEvidenceRow.question_id)
                )
            ).all()
            counts_by_question = {
                question_id: (int(evidence_count), int(question_sources))
                for question_id, evidence_count, question_sources in question_counts
            }
            dimension_counts = (
                await session.execute(
                    select(
                        ResearchClaimRow.question_id,
                        ResearchClaimRow.dimension_key,
                        func.count(ResearchEvidenceRow.id),
                        func.count(distinct(ResearchSourceRow.source_owner_key)),
                        func.max(ResearchEvidenceRow.source_reliability),
                    )
                    .join(
                        ResearchEvidenceRow,
                        ResearchEvidenceRow.claim_id == ResearchClaimRow.id,
                    )
                    .join(
                        ResearchSourceRow,
                        ResearchSourceRow.id == ResearchEvidenceRow.source_id,
                    )
                    .where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                    .group_by(ResearchClaimRow.question_id, ResearchClaimRow.dimension_key)
                )
            ).all()
            counts_by_dimension = {
                (question_id, dimension_key): (
                    int(evidence_count),
                    int(owner_count),
                    float(max_reliability or 0.0),
                )
                for (
                    question_id,
                    dimension_key,
                    evidence_count,
                    owner_count,
                    max_reliability,
                ) in dimension_counts
            }
            all_claims = (
                await session.scalars(
                    select(ResearchClaimRow).where(ResearchClaimRow.run_id == run_id)
                )
            ).all()
            high_risk_dimension_keys = {
                claim.dimension_key for claim in all_claims if _claim_is_high_risk(claim)
            }
            coverage_map: list[_CoverageMapEntry] = []
            weighted_coverage = 0.0
            total_weight = 0.0
            for item in plan_items:
                evidence_count, independent_sources = counts_by_question.get(
                    item.question_id, (0, 0)
                )
                requirements = [str(value) for value in item.evidence_requirements]
                requirement_statuses: list[dict[str, object]] = []
                missing_reasons: list[str] = []
                requirement_scores: list[float] = []
                for index, criterion in enumerate(requirements, start=1):
                    dimension_key = f"{item.question_id}:d{index}"
                    dimension_evidence, dimension_sources, max_source_reliability = (
                        counts_by_dimension.get(
                            (item.question_id, dimension_key),
                            # Every planned requirement is an explicit dimension. Never
                            # copy arbitrary question-level evidence into it: the claim
                            # must be tagged to this dimension by the extractor or it
                            # remains an unresolved gap.
                            (0, 0, 0.0),
                        )
                    )
                    required_sources = (
                        2
                        if _requires_independent_sources(criterion)
                        or dimension_key in high_risk_dimension_keys
                        else 1
                    )
                    score = (
                        1.0
                        if dimension_evidence > 0 and dimension_sources >= required_sources
                        else 0.5
                        if dimension_evidence > 0
                        else 0.0
                    )
                    requirement_scores.append(score)
                    requirement_statuses.append(
                        {
                            "dimension_key": dimension_key,
                            "criterion": criterion,
                            "coverage": score,
                            "accepted_evidence": dimension_evidence,
                            "independent_sources": dimension_sources,
                            "required_sources": required_sources,
                            "max_source_reliability": round(max_source_reliability, 4),
                        }
                    )
                    if score == 0.0:
                        missing_reasons.append(f"{criterion}: 缺少可验证网页原文证据")
                    elif score < 1.0:
                        missing_reasons.append(f"{criterion}: 缺少第二个独立来源")
                dimension_coverage = (
                    sum(requirement_scores) / len(requirement_scores) if requirement_scores else 0.0
                )
                weight = float(4 - item.priority)
                total_weight += weight
                weighted_coverage += dimension_coverage * weight
                coverage_map.append(
                    {
                        "dimension_key": item.question_id,
                        "question": item.question,
                        "priority": item.priority,
                        "coverage": dimension_coverage,
                        "accepted_evidence": evidence_count,
                        "independent_sources": independent_sources,
                        "acceptance_criteria": requirements,
                        "requirement_statuses": requirement_statuses,
                        "missing_reasons": missing_reasons,
                    }
                )
            coverage = round(weighted_coverage / total_weight, 4) if total_weight else 0.0
            covered_questions = sum(
                1 for evidence_count, _ in counts_by_question.values() if evidence_count > 0
            )
            accepted_total = int(
                await session.scalar(
                    select(func.count(ResearchEvidenceRow.id)).where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                )
                or 0
            )
            unique_claims = int(
                await session.scalar(
                    select(func.count(distinct(ResearchEvidenceRow.claim))).where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                )
                or 0
            )
            candidate_total = int(
                await session.scalar(
                    select(func.count(ResearchEvidenceRow.id)).where(
                        ResearchEvidenceRow.run_id == run_id,
                    )
                )
                or 0
            )
            source_count = int(
                await session.scalar(
                    select(func.count(distinct(ResearchEvidenceRow.source_id))).where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                )
                or 0
            )
            owner_count = int(
                await session.scalar(
                    select(func.count(distinct(ResearchSourceRow.source_owner_key)))
                    .join(
                        ResearchEvidenceRow,
                        ResearchEvidenceRow.source_id == ResearchSourceRow.id,
                    )
                    .where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                )
                or 0
            )
            # Measure source quality per independent source, not per extracted
            # Evidence Card. A verbose page can yield several cards and must not
            # receive several votes in the global quality gate.
            # Source quality is a publisher/owner property, not a URL-count
            # property. Averaging by source_id allowed three pages from one
            # low-quality domain to outweigh one authoritative independent
            # owner and kept the gate parked around 74%. Collapse snapshots
            # and URLs to one reliability vote per source owner, matching the
            # independent-source semantics used by cross-validation.
            reliability_by_owner = (
                select(
                    ResearchSourceRow.source_owner_key,
                    func.max(ResearchEvidenceRow.source_reliability).label("reliability"),
                )
                .join(
                    ResearchSourceRow,
                    ResearchSourceRow.id == ResearchEvidenceRow.source_id,
                )
                .where(
                    ResearchEvidenceRow.run_id == run_id,
                    ResearchEvidenceRow.accepted.is_(True),
                )
                .group_by(ResearchSourceRow.source_owner_key)
                .subquery()
            )
            source_quality = float(
                await session.scalar(select(func.avg(reliability_by_owner.c.reliability))) or 0.0
            )
            # Cross-validation is an acceptance-dimension property, not a
            # string-equality property of model-generated Claims.  The previous
            # implementation grouped by claim_id; because every source can
            # phrase the same fact differently, almost every claim had exactly
            # one owner and the 70% gate became unreachable.  High-risk
            # dimensions explicitly require two independent owners; measure
            # those dimensions directly and leave ordinary dimensions to their
            # own one-source acceptance policy.
            required_cross_validation_dimensions = [
                f"{item.question_id}:d{index}"
                for item in plan_items
                for index, criterion in enumerate(item.evidence_requirements, start=1)
                if _requires_independent_sources(str(criterion))
                or f"{item.question_id}:d{index}" in high_risk_dimension_keys
            ]
            corroborated_dimensions = sum(
                1
                for dimension_key in required_cross_validation_dimensions
                if counts_by_dimension.get((dimension_key.split(":", 1)[0], dimension_key), (0, 0))[
                    1
                ]
                >= 2
            )
            cross_validation = (
                round(corroborated_dimensions / len(required_cross_validation_dimensions), 4)
                if required_cross_validation_dimensions
                else 1.0
            )
            previous_facts = ResearchFactCounts(
                accepted_evidence=int(previous_quality.get("accepted_evidence", 0) or 0),
                unique_claims=int(previous_quality.get("claim_count", 0) or 0),
                independent_sources=int(
                    previous_quality.get(
                        "independent_source_count",
                        previous_quality.get("source_count", 0),
                    )
                    or 0
                ),
                evidence_candidates=int(previous_quality.get("candidate_evidence", 0) or 0),
                coverage=float(previous_quality.get("coverage", 0.0) or 0.0),
            )
            current_facts = ResearchFactCounts(
                accepted_evidence=accepted_total,
                unique_claims=unique_claims,
                independent_sources=owner_count,
                evidence_candidates=candidate_total,
                coverage=coverage,
            )
            information_gain = calculate_information_gain(previous_facts, current_facts)
            # Low information gain is a property of the current question and
            # query strategy. A global counter made four unrelated questions
            # with zero gain look like one stalled question and triggered an
            # early replan before the later questions had a chance to rotate
            # their queries. Preserve a per-question ledger and expose the
            # current question's streak for existing UI/decision consumers.
            previous_streaks_raw = previous_quality.get(
                "low_information_gain_streak_by_question", {}
            )
            previous_streaks = (
                {
                    str(question_id): _as_int(streak)
                    for question_id, streak in previous_streaks_raw.items()
                }
                if isinstance(previous_streaks_raw, dict)
                else {}
            )
            previous_low_gain_streak = _question_low_gain_streak(
                previous_quality,
                target.question_id,
            )
            low_information_gain_streak = (
                previous_low_gain_streak
                if technical_outcome
                else previous_low_gain_streak + 1
                if information_gain.score < _LOW_INFORMATION_GAIN_THRESHOLD
                else 0
            )
            previous_streaks[target.question_id] = low_information_gain_streak
            action_latency_ms = max(
                0,
                int(
                    (
                        datetime.now(UTC) - (target.action_started_at or datetime.now(UTC))
                    ).total_seconds()
                    * 1_000
                ),
            )
            action_tokens = max(0, _model_tokens(run) - target.baseline_model_tokens)
            action_fetched = max(
                0,
                _as_int(usage.get("pages_fetched", usage.get("pages", 0)))
                - target.baseline_pages_fetched,
            )
            action_extracted = max(
                0,
                _as_int(usage.get("pages_extracted", usage.get("pages", 0)))
                - target.baseline_pages_extracted,
            )
            action_requests = max(
                0,
                _as_int(usage.get("search_provider_requests", 0))
                - target.baseline_provider_requests,
            )
            normalized = normalized_gain(
                gain=information_gain.score,
                tokens=action_tokens,
                pages=action_extracted,
                requests=action_requests,
                latency_ms=action_latency_ms,
            )
            action_metrics_raw = usage.get("action_metrics", [])
            action_metrics = (
                list(action_metrics_raw) if isinstance(action_metrics_raw, list) else []
            )
            action_metrics.append(
                {
                    "action_id": str(target.tool_call_id),
                    "question_id": target.question_id,
                    "query_family": target.query_family,
                    "outcome": attempt_outcome,
                    "latency_ms": action_latency_ms,
                    "tokens": action_tokens,
                    "pages_fetched": action_fetched,
                    "pages_extracted": action_extracted,
                    "provider_requests": action_requests,
                    "accepted_evidence": information_gain.new_evidence,
                    "information_gain": information_gain.score,
                    **normalized,
                }
            )
            usage["action_metrics"] = action_metrics[-100:]
            family_stats_raw = usage.get("query_family_stats", {})
            family_stats = dict(family_stats_raw) if isinstance(family_stats_raw, dict) else {}
            current_family_raw = family_stats.get(target.query_family, {})
            current_family = (
                dict(current_family_raw) if isinstance(current_family_raw, dict) else {}
            )
            current_family["accepted_evidence"] = (
                _as_int(current_family.get("accepted_evidence", 0)) + information_gain.new_evidence
            )
            current_family["information_gain"] = round(
                _as_float(current_family.get("information_gain", 0.0)) + information_gain.score,
                6,
            )
            family_stats[target.query_family] = current_family
            usage["query_family_stats"] = family_stats
            family_outcomes_raw = usage.get("question_family_outcomes", {})
            family_outcomes = (
                dict(family_outcomes_raw) if isinstance(family_outcomes_raw, dict) else {}
            )
            question_outcomes_raw = family_outcomes.get(target.question_id, [])
            question_outcomes = (
                list(question_outcomes_raw) if isinstance(question_outcomes_raw, list) else []
            )
            question_outcomes.append(
                {
                    "family": target.query_family,
                    "accepted_evidence": information_gain.new_evidence,
                    "utility": normalized["gain_per_1k_tokens"],
                }
            )
            family_outcomes[target.question_id] = question_outcomes[-4:]
            usage["question_family_outcomes"] = family_outcomes
            critical_gaps = sum(
                1
                for dimension in coverage_map
                if int(dimension["priority"]) == 1 and float(dimension["coverage"]) < 1.0
            )
            unresolved_gap_count = sum(
                1 for dimension in coverage_map if float(dimension["coverage"]) < 1.0
            )
            priority_one_coverages = [
                float(dimension["coverage"])
                for dimension in coverage_map
                if int(dimension["priority"]) == 1
            ]
            priority_one_coverage = min(priority_one_coverages) if priority_one_coverages else 0.0
            priority_one_average_coverage = (
                sum(priority_one_coverages) / len(priority_one_coverages)
                if priority_one_coverages
                else 0.0
            )
            priority_one_completed = sum(value >= 1.0 for value in priority_one_coverages)
            recent_distinct: list[dict[str, object]] = []
            for item in reversed(question_outcomes):
                if not recent_distinct or recent_distinct[-1].get("family") != item.get("family"):
                    recent_distinct.append(item)
                if len(recent_distinct) == 2:
                    break
            if (
                plan_item is not None
                and plan_item.priority > 1
                and len(recent_distinct) == 2
                and all(_as_int(item.get("accepted_evidence", 0)) == 0 for item in recent_distinct)
                and all(_as_float(item.get("utility", 0.0)) < 0.01 for item in recent_distinct)
                and _query_space_exhausted_for_freeze(
                    executed_families=executed_families,
                    question=target.question,
                )
            ):
                frozen_raw = usage.get("frozen_questions", [])
                frozen = list(frozen_raw) if isinstance(frozen_raw, list) else []
                if target.question_id not in frozen:
                    frozen.append(target.question_id)
                usage["frozen_questions"] = frozen
            all_conflicts = (
                await session.scalars(
                    select(ResearchConflictRow).where(ResearchConflictRow.run_id == run_id)
                )
            ).all()
            open_conflicts = [conflict for conflict in all_conflicts if conflict.status == "open"]
            unresolved_conflict_ids = [str(conflict.id) for conflict in open_conflicts]
            priority_by_question = {item.question_id: item.priority for item in plan_items}
            high_risk_conflict_ids = [
                str(conflict.id)
                for conflict in open_conflicts
                if int(priority_by_question.get(conflict.question_id, 2)) == 1
                or float(conflict.severity or 0.0) >= 0.8
            ]
            criteria_by_dimension = {
                f"{item.question_id}:d{index}": str(criterion)
                for item in plan_items
                for index, criterion in enumerate(item.evidence_requirements, start=1)
            }
            accepted_quality_rows = (
                await session.execute(
                    select(
                        ResearchEvidenceRow,
                        ResearchClaimRow,
                        ResearchSourceRow,
                        ResearchSourceSnapshotRow.published_at,
                    )
                    .join(ResearchClaimRow, ResearchClaimRow.id == ResearchEvidenceRow.claim_id)
                    .join(ResearchSourceRow, ResearchSourceRow.id == ResearchEvidenceRow.source_id)
                    .outerjoin(
                        ResearchSourceSnapshotRow,
                        and_(
                            ResearchSourceSnapshotRow.source_id == ResearchSourceRow.id,
                            ResearchSourceSnapshotRow.content_hash
                            == ResearchSourceRow.content_hash,
                        ),
                    )
                    .where(
                        ResearchEvidenceRow.run_id == run_id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                )
            ).all()
            numeric_checks: list[bool] = []
            role_checks: list[bool] = []
            freshness_scores: list[float] = []
            now = datetime.now(UTC)
            for evidence, claim, source, published_at in accepted_quality_rows:
                criterion = criteria_by_dimension.get(claim.dimension_key, "")
                numeric_checks.append(
                    numeric_scope_consistent(
                        criterion=criterion,
                        claim=evidence.claim,
                        quote=evidence.exact_quote,
                    )
                )
                role_checks.append(
                    source_role_fits_claim(
                        claim_type=infer_claim_type(claim.atomic_claim),
                        source_role=classify_source_role(source.canonical_url),
                    )
                )
                freshness_at = published_at or source.fetched_at
                if freshness_at.tzinfo is None:
                    freshness_at = freshness_at.replace(tzinfo=UTC)
                age_days = max(0.0, (now - freshness_at).total_seconds() / 86_400)
                freshness_scores.append(max(0.0, min(1.0, 1.0 - age_days / 365.0)))
            numeric_scope_quality = (
                round(sum(numeric_checks) / len(numeric_checks), 4) if numeric_checks else 1.0
            )
            source_role_quality = (
                round(sum(role_checks) / len(role_checks), 4) if role_checks else 1.0
            )
            freshness = (
                round(sum(freshness_scores) / len(freshness_scores), 4)
                if freshness_scores
                else 1.0
            )
            unresolved_claims = [
                claim
                for claim in all_claims
                if claim.status in {"candidate", "partial", "disputed"}
            ]
            weak_claim_ids = [str(claim.id) for claim in unresolved_claims]
            all_gaps = (
                await session.scalars(
                    select(ResearchGapRow).where(
                        ResearchGapRow.run_id == run_id,
                        ResearchGapRow.plan_version == run.plan_version,
                    )
                )
            ).all()
            open_gaps = [gap for gap in all_gaps if gap.status == "open"]
            coverage_by_question = {
                str(entry["dimension_key"]): float(entry["coverage"])
                for entry in coverage_map
            }
            dimensions_by_question = {
                str(entry["dimension_key"]): tuple(
                    str(requirement.get("dimension_key"))
                    for requirement in entry.get("requirement_statuses", [])
                    if isinstance(requirement, dict)
                    and _as_float(requirement.get("coverage", 0.0)) < 1.0
                )
                for entry in coverage_map
            }
            evidence_owners_by_claim: dict[str, set[str]] = {}
            evidence_claim_by_id: dict[str, str] = {}
            for evidence, claim, source, _published_at in accepted_quality_rows:
                claim_key = str(claim.id)
                evidence_owners_by_claim.setdefault(claim_key, set()).add(
                    source.source_owner_key
                )
                evidence_claim_by_id[str(evidence.id)] = claim_key
            open_conflicts_by_claim: dict[str, list[str]] = {}
            for conflict in open_conflicts:
                for evidence_id in (conflict.left_evidence_id, conflict.right_evidence_id):
                    conflict_claim_key = evidence_claim_by_id.get(str(evidence_id))
                    if conflict_claim_key is not None:
                        open_conflicts_by_claim.setdefault(conflict_claim_key, []).append(
                            str(conflict.id)
                        )
            claim_states = {
                str(claim.id): classify_claim_risk(
                    claim_id=str(claim.id),
                    question_id=claim.question_id,
                    dimension_key=claim.dimension_key,
                    claim_status=claim.status,
                    claim_type=claim.claim_type,
                    importance=float(claim.importance or 0.0),
                    independent_sources=len(
                        evidence_owners_by_claim.get(str(claim.id), set())
                    ),
                    required_sources=(
                        2
                        if _claim_is_high_risk(claim)
                        or _requires_independent_sources(
                            criteria_by_dimension.get(claim.dimension_key, "")
                        )
                        else 1
                    ),
                    open_conflict_ids=open_conflicts_by_claim.get(str(claim.id), ()),
                )
                for claim in all_claims
            }
            unresolved_claims_by_question = {
                item.question_id: tuple(
                    claim_id
                    for claim_id, state in claim_states.items()
                    if state.question_id == item.question_id and state.unresolved
                )
                for item in plan_items
            }
            high_risk_claims_by_question = {
                item.question_id: tuple(
                    claim_id
                    for claim_id, state in claim_states.items()
                    if state.question_id == item.question_id
                    and state.unresolved
                    and state.risk_level in {"high", "critical"}
                )
                for item in plan_items
            }
            source_deficit_by_question = {
                item.question_id: sum(
                    state.independent_source_deficit
                    for state in claim_states.values()
                    if state.question_id == item.question_id and state.unresolved
                )
                for item in plan_items
            }
            high_risk_conflicts_by_question = {
                item.question_id: tuple(
                    str(conflict.id)
                    for conflict in open_conflicts
                    if conflict.question_id == item.question_id
                    and (
                        item.priority == 1
                        or float(conflict.severity or 0.0) >= 0.8
                    )
                )
                for item in plan_items
            }
            open_gap_by_question = {gap.question_id: gap for gap in open_gaps}
            exhausted_by_question = usage.get("question_budget_exhausted_by_question", {})
            strategy_exhausted = usage.get("query_strategy_exhausted_by_question", {})
            frozen_questions = {
                str(value)
                for value in (
                    usage.get("frozen_questions", [])
                    if isinstance(usage.get("frozen_questions", []), list)
                    else []
                )
            }
            blocked_questions = frozen_questions | {
                str(question_id)
                for question_id, exhausted in (
                    exhausted_by_question.items()
                    if isinstance(exhausted_by_question, dict)
                    else ()
                )
                if exhausted
            } | {
                str(question_id)
                for question_id, exhausted in (
                    strategy_exhausted.items() if isinstance(strategy_exhausted, dict) else ()
                )
                if exhausted
            }
            risk_states = {
                item.question_id: classify_question_risk(
                    question_id=item.question_id,
                    priority=item.priority,
                    coverage=coverage_by_question.get(item.question_id, 0.0),
                    requirements=[str(value) for value in item.evidence_requirements],
                    gap_open=item.question_id in open_gap_by_question,
                    open_dimension_keys=dimensions_by_question.get(item.question_id, ()),
                    unresolved_claim_ids=unresolved_claims_by_question.get(
                        item.question_id, ()
                    ),
                    high_risk_claim_ids=high_risk_claims_by_question.get(
                        item.question_id, ()
                    ),
                    high_risk_conflict_ids=high_risk_conflicts_by_question.get(
                        item.question_id, ()
                    ),
                    independent_source_deficit=source_deficit_by_question.get(
                        item.question_id, 0
                    ),
                    blocked=item.question_id in blocked_questions,
                ).as_dict()
                for item in plan_items
            }
            claim_risk_states = {
                claim_id: state.as_dict() for claim_id, state in claim_states.items()
            }
            gap_risk_states = {
                str(gap.id): classify_gap_risk(
                    gap_id=str(gap.id),
                    question_id=gap.question_id,
                    gap_status=gap.status,
                    gap_type=gap.gap_type,
                    severity=float(gap.severity or 0.0),
                    resolution_attempts=int(gap.resolution_attempts or 0),
                    blocked=gap.question_id in blocked_questions,
                ).as_dict()
                for gap in all_gaps
            }
            conflict_risk_states = {
                str(conflict.id): {
                    "conflict_id": str(conflict.id),
                    "question_id": conflict.question_id,
                    "severity": round(float(conflict.severity or 0.0), 4),
                    "status": conflict.status,
                    "lifecycle": "resolved" if conflict.status == "resolved" else "blocked",
                    "unresolved": conflict.status != "resolved",
                    "left_evidence_id": str(conflict.left_evidence_id),
                    "right_evidence_id": str(conflict.right_evidence_id),
                }
                for conflict in all_conflicts
            }
            run.quality_snapshot = {
                "coverage": coverage,
                "information_gain": information_gain.score,
                "low_information_gain_streak": low_information_gain_streak,
                "low_information_gain_streak_by_question": previous_streaks,
                "coverage_map": coverage_map,
                "source_quality": round(source_quality, 4),
                "independent_source_count": owner_count,
                "priority_one_coverage": round(priority_one_coverage, 4),
                "priority_one_average_coverage": round(priority_one_average_coverage, 4),
                "priority_one_completed": priority_one_completed,
                "priority_one_total": len(priority_one_coverages),
                "source_independence": round(owner_count / source_count, 4)
                if source_count
                else 0.0,
                "cross_validation": cross_validation,
                "accepted_evidence": accepted_total,
                "candidate_evidence": candidate_total,
                "claim_count": unique_claims,
                "source_count": source_count,
                "conflict_count": len(unresolved_conflict_ids),
                "unresolved_high_risk_conflicts": len(high_risk_conflict_ids),
                "high_risk_claim_dual_source_rate": cross_validation,
                "numeric_scope_consistency": numeric_scope_quality,
                "source_role_fit": source_role_quality,
                "freshness": freshness,
                "citation_count": accepted_total,
                "unanswered_questions": max(total_questions - covered_questions, 0),
                "critical_gaps": critical_gaps,
                "unresolved_gap_count": unresolved_gap_count,
                "risk_state": {
                    "version": "claim_gap.v2",
                    "plan_version": run.plan_version,
                    "questions": risk_states,
                    "claims": claim_risk_states,
                    "gaps": gap_risk_states,
                    "conflicts": conflict_risk_states,
                    "summary": {
                        "unresolved_questions": sum(
                            bool(state.get("unresolved_high_risk"))
                            for state in risk_states.values()
                        ),
                        "unresolved_claims": sum(
                            bool(state.get("unresolved"))
                            for state in claim_risk_states.values()
                        ),
                        "open_gaps": sum(
                            bool(state.get("unresolved"))
                            for state in gap_risk_states.values()
                        ),
                        "open_conflicts": len(open_conflicts),
                        "borrow_eligible_questions": sum(
                            bool(state.get("borrow_eligible"))
                            for state in risk_states.values()
                        ),
                    },
                },
                "risk_state_by_question": risk_states,
            }
            quality_gate_open = (
                _quality_enrichment_needed(
                    source_quality=source_quality,
                    cross_validation=cross_validation,
                )
                or bool(unresolved_conflict_ids)
                or numeric_scope_quality < 1.0
                or source_role_quality < 1.0
            )
            quality_repair_targets = _quality_repair_targets(
                coverage_map,
                source_quality=source_quality,
                cross_validation=cross_validation,
            )
            quality_snapshot = dict(run.quality_snapshot)
            quality_snapshot["quality_repair_targets"] = quality_repair_targets
            run.quality_snapshot = quality_snapshot
            quality_enrichment_needed = quality_gate_open and bool(quality_repair_targets)
            # A global quality metric may only reopen questions whose concrete
            # acceptance dimension can improve that metric. This prevents an
            # already-complete, high-volume question from becoming the permanent
            # enrichment anchor for gaps that actually belong to other questions.
            for item in plan_items:
                if item.question_id in quality_repair_targets and item.status == "researched":
                    item.status = "partial"
            if (
                target.question_id in quality_repair_targets
                and not technical_outcome
                and not retry_current
                and plan_item is not None
            ):
                question_status = "partial"
                plan_item.status = "partial"
                if gap is not None:
                    gap.status = "open"
                    gap.description = "该问题的具体证据维度仍可改善来源质量或交叉验证质量门。"
                    gap.updated_at = datetime.now(UTC)
            iteration_consumed = _research_attempt_consumes_iteration(
                attempt_outcome,
                technical_outcome=technical_outcome,
            )
            if iteration_consumed:
                usage["iterations"] = int(usage.get("iterations", 0)) + 1
                usage["productive_iterations"] = _as_int(usage.get("productive_iterations", 0)) + 1
            run.usage_snapshot = _usage_with_resource_pools(run, usage)

            if technical_outcome:
                await self._append_event(
                    session,
                    run,
                    event_type=(
                        "question.technical_retry_scheduled"
                        if retry_current
                        else "question.technical_degraded"
                    ),
                    public_summary=(
                        f"问题 {target.question_id} 遇到搜索服务故障; "
                        "该故障不消耗研究尝试或信息增益轮次。"
                    ),
                    refs={
                        "question_id": target.question_id,
                        "status": question_status,
                        "technical_retry": technical_failure_count,
                    },
                    metrics={"research_attempt_consumed": False},
                )
            elif query_strategy_exhausted:
                await self._append_event(
                    session,
                    run,
                    event_type="search.source_space_exhausted",
                    public_summary=(
                        f"问题 {target.question_id} 的有限检索角度已耗尽; "
                        "切换其他问题, 等待定向 REPLAN 提供新的核心检索角度。"
                    ),
                    refs={"question_id": target.question_id, "attempts": attempts},
                    metrics={"research_attempt_consumed": iteration_consumed},
                )
            elif retry_current:
                await self._append_event(
                    session,
                    run,
                    event_type="question.retry_scheduled",
                    public_summary=(
                        f"问题 {target.question_id} 当前有 {accepted_for_question} 条证据、"
                        f"{sources_for_question} 个独立来源; Evaluator 安排替代检索。"
                    ),
                    refs={
                        "question_id": target.question_id,
                        "status": "retrying",
                        "attempt": attempts,
                    },
                    metrics={
                        "accepted_count": accepted_for_question,
                        "source_count": sources_for_question,
                        "research_attempt_consumed": iteration_consumed,
                    },
                )
            else:
                await self._append_event(
                    session,
                    run,
                    event_type="question.researched",
                    public_summary=(
                        f"问题 {target.question_id} 获得 {accepted_for_question} 条有效证据。"
                    ),
                    refs={
                        "question_id": target.question_id,
                        "status": question_status,
                    },
                    metrics={
                        "accepted_count": accepted_for_question,
                        "source_count": sources_for_question,
                    },
                )

            remaining = int(
                await session.scalar(
                    select(func.count(ResearchPlanItemRow.id)).where(
                        ResearchPlanItemRow.run_id == run_id,
                        ResearchPlanItemRow.plan_version == run.plan_version,
                        ResearchPlanItemRow.status.in_(("pending", "active", "technical_retry")),
                    )
                )
                or 0
            )
            quality_met = (
                coverage >= 0.85
                and priority_one_coverage >= 0.80
                and source_quality >= 0.75
                and cross_validation >= 0.70
                and critical_gaps == 0
                and not high_risk_conflict_ids
                and numeric_scope_quality >= 1.0
                and source_role_quality >= 1.0
                and freshness >= 0.70
            )
            information_stagnated = (
                coverage >= 0.85
                and critical_gaps == 0
                and not quality_gate_open
                and not unresolved_conflict_ids
                and low_information_gain_streak >= _LOW_INFORMATION_GAIN_STREAK_TO_STOP
            )
            all_first_attempted = all(item.status != "pending" for item in plan_items)
            source_space_map = run.usage_snapshot.get("source_space_exhausted_by_question", {})
            source_space_exhausted = (
                all(
                    item.status == "researched"
                    or bool(source_space_map.get(item.question_id))
                    for item in plan_items
                )
                if isinstance(source_space_map, dict)
                else False
            )
            source_space_exhausted = source_space_exhausted and all_first_attempted
            replans_used = _as_int(run.usage_snapshot.get("replans", 0))
            # Keep both counters bounded so legacy runs without a persisted
            # ``replans`` usage field cannot replan forever.
            replan_allowed = replans_used < _MAX_REPLANS and run.plan_version <= _MAX_REPLANS
            replan_needed = (
                replan_allowed
                and all_first_attempted
                and unresolved_gap_count > 0
                and (
                    remaining == 0
                    or low_information_gain_streak >= _LOW_INFORMATION_GAIN_STREAK_TO_STOP
                    or _replan_reserve_reached(run)
                )
            )
            budget_stop_reason = _budget_exhaustion_reason(run)
            stop_reason: str | None
            if attempt_outcome == "deadline_exhausted":
                decision = "stop_budget"
                stop_reason = "deadline_exhausted"
            elif (
                budget_stop_reason is not None
                and not _search_acquisition_budget_exhausted(budget_stop_reason)
            ):
                decision = "stop_budget"
                stop_reason = budget_stop_reason
            elif quality_met:
                decision = "ready_to_write"
                stop_reason = "quality_met"
            elif _search_acquisition_budget_exhausted(budget_stop_reason):
                # Search acquisition is exhausted, but already-discovered
                # candidates and fetched artifacts may still yield evidence.
                # prepare_target owns the cache-drain decision and enters
                # writing only when no reusable work remains.
                decision = "continue_cached"
                stop_reason = None
            elif replan_needed:
                decision = "replan"
                stop_reason = None
            elif information_stagnated:
                decision = "stop_information_gain"
                stop_reason = "stagnation"
            elif quality_enrichment_needed:
                decision = "continue_quality"
                stop_reason = None
            elif remaining == 0 and unresolved_gap_count == 0:
                decision = "write_with_limitations"
                stop_reason = "sources_exhausted"
            elif remaining == 0 and unresolved_gap_count > 0 and source_space_exhausted:
                decision = "write_with_limitations"
                stop_reason = "source_space_exhausted"
            elif remaining == 0 and unresolved_gap_count > 0:
                # Every plan item may be blocked/partial while the run still
                # has real search/page/token budget. Keep the loop alive so a
                # fresh query variant or a later replan can recover evidence.
                decision = "continue_plan"
                stop_reason = None
            elif retry_current:
                decision = "retry_technical" if technical_outcome else "retry_current"
                stop_reason = None
            else:
                decision = "continue_plan"
                stop_reason = None

            verdict = (
                "write"
                if decision
                in (
                    "ready_to_write",
                    "write_with_limitations",
                    "stop_budget",
                    "stop_information_gain",
                )
                else "replan"
                if decision == "replan"
                else "continue"
            )
            session.add(
                EvaluationSnapshotRow(
                    id=uuid7(),
                    run_id=run_id,
                    scope="question",
                    state_version=run.state_version,
                    plan_version=run.plan_version,
                    coverage=coverage,
                    evidence_sufficiency=min(1.0, accepted_for_question / 2.0),
                    source_quality=source_quality,
                    source_diversity=min(1.0, owner_count / 3.0),
                    source_independence=run.quality_snapshot["source_independence"],
                    cross_validation=cross_validation,
                    freshness=1.0,
                    conflict_resolution=1.0 if not unresolved_conflict_ids else 0.0,
                    citation_completeness=1.0 if accepted_total else 0.0,
                    citation_support=cross_validation,
                    weak_claim_ids=weak_claim_ids,
                    missing_dimension_keys=[
                        d["dimension_key"] for d in coverage_map if d["coverage"] < 1.0
                    ],
                    unresolved_conflict_ids=unresolved_conflict_ids,
                    verdict=verdict,
                    created_at=datetime.now(UTC),
                )
            )
            run.phase = RunPhase.EVALUATING.value
            await self._append_event(
                session,
                run,
                event_type="research.information_gain_calculated",
                public_summary=(
                    f"本轮信息增益 {information_gain.score:.2f}: "
                    f"新增 {information_gain.new_evidence} 条有效证据、"
                    f"{information_gain.new_claims} 个 Claim、"
                    f"{information_gain.new_sources} 个独立来源, "
                    f"Coverage 提升 {information_gain.coverage_delta:.0%}。"
                ),
                refs={
                    "question_id": target.question_id,
                    "decision": decision,
                    "low_gain_streak": low_information_gain_streak,
                },
                metrics=information_gain.model_dump(mode="json"),
            )
            await self._append_event(
                session,
                run,
                event_type="evaluation.completed",
                public_summary=_evaluation_summary(decision, target.question_id),
                refs={
                    "question_id": target.question_id,
                    "decision": decision,
                    "reason": stop_reason,
                    "information_gain": information_gain.score,
                    "low_gain_streak": low_information_gain_streak,
                    "critical_gaps": critical_gaps,
                },
                metrics=run.quality_snapshot,
            )
            now = datetime.now(UTC)
            run.updated_at = now
            run.state_version += 1
            if stop_reason is None:
                run.status = RunStatus.RUNNING.value
                run.phase = RunPhase.RESEARCHING.value
                run.termination_reason = None
                run.lease_until = now + timedelta(seconds=EXECUTION_LEASE_SECONDS)
                await self._append_event(
                    session,
                    run,
                    event_type="research.continued",
                    public_summary=(
                        f"继续研究: Coverage {coverage:.0%}, "
                        f"信息增益 {information_gain.score:.2f}, "
                        f"仍有 {remaining} 个待处理问题和 {critical_gaps} 个关键缺口。"
                    ),
                    refs={
                        "decision": decision,
                        "information_gain": information_gain.score,
                        "critical_gaps": critical_gaps,
                    },
                    metrics=None,
                )
                return IterationEvaluation(
                    continue_research=True,
                    decision=decision,
                    stop_reason=None,
                    question_status=question_status,
                    coverage=coverage,
                    information_gain=information_gain.score,
                    low_information_gain_streak=low_information_gain_streak,
                )

            run.status = RunStatus.RUNNING.value
            run.phase = RunPhase.WRITING.value
            run.termination_reason = stop_reason
            run.lease_until = now + timedelta(seconds=EXECUTION_LEASE_SECONDS)
            await self._append_event(
                session,
                run,
                event_type="report.writing_started",
                public_summary=(
                    _budget_stop_summary(stop_reason)
                    if stop_reason is not None and stop_reason.endswith("_budget_exhausted")
                    else (
                        "连续两轮边际信息增益过低且无关键缺口; 停止扩展并进入写作。"
                        if stop_reason == "stagnation"
                        else (
                            "计划已遍历但仍有未满足的验收条件; 使用现有证据生成带限制报告。"
                            if stop_reason in {"sources_exhausted", "source_space_exhausted"}
                            else "全部质量门已满足; 自动进入报告写作。"
                        )
                    )
                ),
                refs={"reason": run.termination_reason},
            )
            return IterationEvaluation(
                continue_research=False,
                decision=decision,
                stop_reason=stop_reason,
                question_status=question_status,
                coverage=coverage,
                information_gain=information_gain.score,
                low_information_gain_streak=low_information_gain_streak,
            )

    async def get_evidence_graph(
        self,
        owner_hash: str,
        run_id: UUID,
    ) -> EvidenceGraphView:
        async with self._sessions() as session:
            owned = await session.scalar(
                select(ResearchRunRow.id).where(
                    ResearchRunRow.id == run_id,
                    ResearchRunRow.owner_hash == owner_hash,
                )
            )
            if owned is None:
                raise ResearchRunNotFoundError(str(run_id))

            rows = (
                await session.execute(
                    select(ResearchClaimRow, ResearchEvidenceRow)
                    .outerjoin(
                        ResearchEvidenceRow,
                        ResearchEvidenceRow.claim_id == ResearchClaimRow.id,
                    )
                    .where(ResearchClaimRow.run_id == run_id)
                    .order_by(
                        ResearchClaimRow.question_id,
                        ResearchClaimRow.created_at,
                        ResearchEvidenceRow.evidence_score.desc(),
                    )
                )
            ).tuples()
            claims: dict[UUID, EvidenceGraphClaimNode] = {}
            for claim, evidence in rows:
                node = claims.get(claim.id)
                if node is None:
                    node = EvidenceGraphClaimNode(
                        claim_id=claim.id,
                        question_id=claim.question_id,
                        dimension_key=claim.dimension_key,
                        atomic_claim=claim.atomic_claim,
                        status=claim.status,
                        confidence=claim.confidence,
                    )
                    claims[claim.id] = node
                if evidence is not None:
                    node.evidence.append(
                        EvidenceGraphEvidenceRef(
                            evidence_id=evidence.id,
                            source_id=evidence.source_id,
                            snapshot_id=evidence.snapshot_id,
                            chunk_id=evidence.chunk_id,
                            relation=evidence.relation,
                            accepted=evidence.accepted,
                            evidence_score=evidence.evidence_score,
                        )
                    )

            evidence_count = await session.scalar(
                select(func.count(ResearchEvidenceRow.id)).where(
                    ResearchEvidenceRow.run_id == run_id,
                    ResearchEvidenceRow.claim_id.is_not(None),
                )
            )
            snapshot_count = await session.scalar(
                select(func.count(ResearchSourceSnapshotRow.id)).where(
                    ResearchSourceSnapshotRow.run_id == run_id
                )
            )
            chunk_count = await session.scalar(
                select(func.count(ResearchSourceChunkRow.id)).where(
                    ResearchSourceChunkRow.run_id == run_id
                )
            )
            edge_rows = (
                await session.scalars(
                    select(ResearchClaimEdgeRow)
                    .where(ResearchClaimEdgeRow.run_id == run_id)
                    .order_by(
                        ResearchClaimEdgeRow.relation,
                        ResearchClaimEdgeRow.from_claim_id,
                        ResearchClaimEdgeRow.to_claim_id,
                    )
                )
            ).all()
            conflict_rows = (
                await session.scalars(
                    select(ResearchConflictRow)
                    .where(ResearchConflictRow.run_id == run_id)
                    .order_by(
                        ResearchConflictRow.status,
                        ResearchConflictRow.severity.desc(),
                        ResearchConflictRow.created_at,
                    )
                )
            ).all()
            return EvidenceGraphView(
                run_id=run_id,
                claim_count=len(claims),
                evidence_count=int(evidence_count or 0),
                snapshot_count=int(snapshot_count or 0),
                chunk_count=int(chunk_count or 0),
                edge_count=len(edge_rows),
                conflict_count=len(conflict_rows),
                claims=list(claims.values()),
                edges=[
                    EvidenceGraphClaimEdgeView(
                        edge_id=edge.id,
                        from_claim_id=edge.from_claim_id,
                        to_claim_id=edge.to_claim_id,
                        relation=edge.relation,
                        confidence=edge.confidence,
                    )
                    for edge in edge_rows
                ],
                conflicts=[
                    EvidenceGraphConflictView(
                        conflict_id=conflict.id,
                        question_id=conflict.question_id,
                        entity=conflict.entity,
                        attribute=conflict.attribute,
                        left_evidence_id=conflict.left_evidence_id,
                        right_evidence_id=conflict.right_evidence_id,
                        severity=conflict.severity,
                        status=conflict.status,
                        resolution_summary=conflict.resolution_summary,
                    )
                    for conflict in conflict_rows
                ],
            )

    async def list_evidence(self, owner_hash: str, run_id: UUID) -> list[EvidenceView]:
        async with self._sessions() as session:
            owned = await session.scalar(
                select(ResearchRunRow.id).where(
                    ResearchRunRow.id == run_id,
                    ResearchRunRow.owner_hash == owner_hash,
                )
            )
            if owned is None:
                raise ResearchRunNotFoundError(str(run_id))
            rows = (
                await session.execute(
                    select(ResearchEvidenceRow, ResearchSourceRow)
                    .join(ResearchSourceRow, ResearchEvidenceRow.source_id == ResearchSourceRow.id)
                    .where(ResearchEvidenceRow.run_id == run_id)
                    .order_by(
                        ResearchEvidenceRow.accepted.desc(),
                        ResearchEvidenceRow.evidence_score.desc(),
                        ResearchEvidenceRow.created_at,
                    )
                )
            ).tuples()
            return [
                EvidenceView(
                    evidence_id=evidence.id,
                    claim_id=evidence.claim_id,
                    snapshot_id=evidence.snapshot_id,
                    chunk_id=evidence.chunk_id,
                    question_id=evidence.question_id,
                    claim=evidence.claim,
                    exact_quote=evidence.exact_quote,
                    relation=evidence.relation,
                    source_title=source.title,
                    source_url=source.canonical_url,
                    source_domain=source.domain,
                    source_reliability=evidence.source_reliability,
                    relevance=evidence.relevance,
                    confidence=evidence.confidence,
                    evidence_score=evidence.evidence_score,
                    accepted=evidence.accepted,
                    rejection_reason=evidence.rejection_reason,
                )
                for evidence, source in rows
            ]

    async def list_evaluations(self, owner_hash: str, run_id: UUID) -> list[EvaluationSnapshot]:
        async with self._sessions() as session:
            owned = await session.scalar(
                select(ResearchRunRow.id).where(
                    ResearchRunRow.id == run_id,
                    ResearchRunRow.owner_hash == owner_hash,
                )
            )
            if owned is None:
                raise ResearchRunNotFoundError(str(run_id))
            rows = (
                await session.scalars(
                    select(EvaluationSnapshotRow)
                    .where(EvaluationSnapshotRow.run_id == run_id)
                    .order_by(EvaluationSnapshotRow.created_at, EvaluationSnapshotRow.id)
                )
            ).all()
            return [
                EvaluationSnapshot(
                    evaluation_id=row.id,
                    run_id=row.run_id,
                    scope=EvaluationScope(row.scope),
                    state_version=row.state_version,
                    plan_version=row.plan_version,
                    coverage=row.coverage,
                    evidence_sufficiency=row.evidence_sufficiency,
                    source_quality=row.source_quality,
                    source_diversity=row.source_diversity,
                    source_independence=row.source_independence,
                    cross_validation=row.cross_validation,
                    freshness=row.freshness,
                    conflict_resolution=row.conflict_resolution,
                    citation_completeness=row.citation_completeness,
                    citation_support=row.citation_support,
                    weak_claim_ids=tuple(row.weak_claim_ids),
                    missing_dimension_keys=tuple(row.missing_dimension_keys),
                    unresolved_conflict_ids=tuple(row.unresolved_conflict_ids),
                    verdict=EvaluationVerdict(row.verdict),
                )
                for row in rows
            ]

    async def _enter_writing(
        self,
        session: AsyncSession,
        run: ResearchRunRow,
        *,
        reason: str,
        summary: str,
    ) -> None:
        now = datetime.now(UTC)
        run.status = RunStatus.RUNNING.value
        run.phase = RunPhase.WRITING.value
        run.termination_reason = reason
        run.lease_until = now + timedelta(seconds=EXECUTION_LEASE_SECONDS)
        run.updated_at = now
        run.state_version += 1
        await self._append_event(
            session,
            run,
            event_type="report.writing_started",
            public_summary=summary,
            refs={"reason": reason},
        )

    @staticmethod
    async def _locked_run(
        session: AsyncSession, run_id: UUID, worker_task_id: str
    ) -> ResearchRunRow:
        run = await session.scalar(
            select(ResearchRunRow).where(ResearchRunRow.id == run_id).with_for_update()
        )
        if (
            run is None
            or RunStatus(run.status) != RunStatus.RUNNING
            or run.worker_task_id != worker_task_id
        ):
            raise ResearchLeaseLostError(str(run_id))
        return run

    @staticmethod
    async def _append_event(
        session: AsyncSession,
        run: ResearchRunRow,
        *,
        event_type: str,
        public_summary: str,
        refs: dict[str, object],
        metrics: dict[str, object] | None = None,
    ) -> None:
        sequence = run.next_event_seq
        run.next_event_seq += 1
        session.add(
            AgentEventRow(
                run_id=run.id,
                run_seq=sequence,
                schema_version=1,
                phase=run.phase,
                event_type=event_type,
                public_summary=public_summary,
                refs=refs,
                metrics=metrics,
            )
        )
        await session.flush()


def _model_tokens(run: ResearchRunRow) -> int:
    computed = _model_tokens_from_usage(run.usage_snapshot)
    return max(computed, int(run.usage_snapshot.get("model_tokens", 0) or 0))


def _model_tokens_from_usage(usage: dict[str, object]) -> int:
    planner = usage.get("planner")
    planner_tokens = _as_int(planner.get("total_tokens", 0)) if isinstance(planner, dict) else 0
    writer = usage.get("writer")
    writer_tokens = _as_int(writer.get("total_tokens", 0)) if isinstance(writer, dict) else 0
    return planner_tokens + writer_tokens + _as_int(usage.get("evidence_total_tokens", 0))


def _usage_with_resource_pools(
    run: ResearchRunRow,
    usage: Mapping[str, object],
) -> dict[str, object]:
    """Persist a live view for every executable non-time resource pool."""

    current = dict(usage)
    current["resource_pools"] = build_resource_pool_snapshot(
        run.budget_snapshot,
        current,
    )
    return current


def _mark_question_risk_blocked(
    quality_snapshot: Mapping[str, object],
    *,
    question_id: str,
    reason: str,
) -> dict[str, object]:
    """Apply a durable risk transition when no further research borrow is eligible."""

    quality = dict(quality_snapshot)
    raw_questions = quality.get("risk_state_by_question", {})
    questions = dict(raw_questions) if isinstance(raw_questions, dict) else {}
    raw_state = questions.get(question_id, {})
    state = dict(raw_state) if isinstance(raw_state, dict) else {}
    reasons_raw = state.get("reasons", [])
    reasons = list(reasons_raw) if isinstance(reasons_raw, list) else []
    marker = f"budget_blocked:{reason}"
    if marker not in reasons:
        reasons.append(marker)
    state.update(
        {
            "lifecycle": "blocked",
            "borrow_eligible": False,
            "reasons": reasons,
        }
    )
    questions[question_id] = state
    quality["risk_state_by_question"] = questions
    raw_risk = quality.get("risk_state", {})
    risk = dict(raw_risk) if isinstance(raw_risk, dict) else {}
    risk["questions"] = questions
    summary_raw = risk.get("summary", {})
    summary = dict(summary_raw) if isinstance(summary_raw, dict) else {}
    summary["borrow_eligible_questions"] = sum(
        bool(item.get("borrow_eligible"))
        for item in questions.values()
        if isinstance(item, dict)
    )
    risk["summary"] = summary
    quality["risk_state"] = risk
    return quality


def _quality_gate_met_from_snapshot(snapshot: Mapping[str, object]) -> bool:
    """Use the persisted hard-gate metrics when no candidate can be scheduled."""

    def number(name: str) -> float:
        value = snapshot.get(name, 0.0)
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    return (
        number("coverage") >= 0.85
        and number("priority_one_coverage") >= 0.80
        and number("source_quality") >= 0.75
        and number("cross_validation") >= 0.70
        and number("critical_gaps") == 0
        and number("unresolved_high_risk_conflicts") == 0
        and (
            "numeric_scope_consistency" not in snapshot
            or number("numeric_scope_consistency") >= 1.0
        )
        and ("source_role_fit" not in snapshot or number("source_role_fit") >= 1.0)
        and ("freshness" not in snapshot or number("freshness") >= 0.70)
    )


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, bytes, bytearray)):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, bytes, bytearray)):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _string_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _update_query_family_usage(
    usage: dict[str, object],
    *,
    family: str,
    reused: bool,
    provider_requests: int,
    timeouts: int,
    fallbacks: int,
    usable_results: int,
    latency_ms: int,
) -> None:
    raw = usage.get("query_family_stats", {})
    stats = dict(raw) if isinstance(raw, dict) else {}
    current_raw = stats.get(family, {})
    current = dict(current_raw) if isinstance(current_raw, dict) else {}
    current["logical_queries"] = _as_int(current.get("logical_queries", 0)) + int(not reused)
    current["cache_reuses"] = _as_int(current.get("cache_reuses", 0)) + int(reused)
    current["provider_requests"] = _as_int(current.get("provider_requests", 0)) + max(
        0, provider_requests
    )
    current["timeouts"] = _as_int(current.get("timeouts", 0)) + max(0, timeouts)
    current["fallbacks"] = _as_int(current.get("fallbacks", 0)) + max(0, fallbacks)
    current["usable_results"] = _as_int(current.get("usable_results", 0)) + max(0, usable_results)
    current["latency_ms"] = _as_int(current.get("latency_ms", 0)) + max(0, latency_ms)
    stats[family] = current
    usage["query_family_stats"] = stats


def _update_provider_health_usage(
    usage: dict[str, object],
    *,
    healthy: int,
    unresponsive: int,
    productive: int,
) -> None:
    """Keep transport attempts separate from useful provider responses."""

    usage["search_provider_healthy_responses"] = _as_int(
        usage.get("search_provider_healthy_responses", 0)
    ) + max(0, healthy)
    usage["search_provider_unresponsive_responses"] = _as_int(
        usage.get("search_provider_unresponsive_responses", 0)
    ) + max(0, unresponsive)
    usage["search_provider_productive_responses"] = _as_int(
        usage.get("search_provider_productive_responses", 0)
    ) + max(0, productive)


_PROVIDER_OBSERVABILITY_FIELDS = (
    "provider_requests",
    "healthy_responses",
    "unresponsive_responses",
    "timeout_count",
    "network_error_count",
    "http_error_count",
    "empty_response_count",
    "invalid_response_count",
    "fallback_attempts",
    "fallback_successes",
    "circuit_open_count",
)


def _update_provider_observability_usage(
    usage: dict[str, object], metrics: Mapping[str, object]
) -> None:
    """Accumulate typed search-provider telemetry without changing control flow."""

    raw_snapshot = usage.get("provider_observability", {})
    snapshot = dict(raw_snapshot) if isinstance(raw_snapshot, Mapping) else {}
    for field in _PROVIDER_OBSERVABILITY_FIELDS:
        value = _as_int(metrics.get(field, 0))
        snapshot[field] = _as_int(snapshot.get(field, 0)) + max(0, value)
    usage["provider_observability"] = snapshot


def _query_family_order(*, prefer_authoritative: bool) -> tuple[QueryFamily, ...]:
    """Return every applicable family exactly once in the preferred order."""

    if not prefer_authoritative:
        return QUERY_FAMILIES
    return (
        QueryFamily.AUTHORITATIVE,
        *(family for family in QUERY_FAMILIES if family is not QueryFamily.AUTHORITATIVE),
    )


def _executed_query_families(usage: Mapping[str, object], question_id: str) -> set[str]:
    raw_by_question = usage.get("executed_query_families_by_question", {})
    if not isinstance(raw_by_question, dict):
        return set()
    raw = raw_by_question.get(question_id, [])
    if not isinstance(raw, list):
        return set()
    valid = {family.value for family in QUERY_FAMILIES}
    return {str(value) for value in raw if str(value) in valid}


def _mark_query_family_executed(
    usage: dict[str, object],
    *,
    question_id: str,
    family: str,
) -> None:
    """Persist a family only after at least one real provider request began."""

    if family not in {item.value for item in QUERY_FAMILIES}:
        return
    raw_by_question = usage.get("executed_query_families_by_question", {})
    by_question = dict(raw_by_question) if isinstance(raw_by_question, dict) else {}
    executed = _executed_query_families(usage, question_id)
    executed.add(family)
    by_question[question_id] = [
        item.value for item in QUERY_FAMILIES if item.value in executed
    ]
    usage["executed_query_families_by_question"] = by_question


def _budget_exhausted(run: ResearchRunRow) -> bool:
    return _budget_exhaustion_reason(run) is not None


def _budget_exhaustion_reason(run: ResearchRunRow) -> str | None:
    """Return the exact hard limit that stopped research."""

    budget = run.budget_snapshot
    usage = run.usage_snapshot
    # Older snapshots used the singular searches/pages counters. Preserve
    # their public stop reason while V2 snapshots expose split causes.
    has_split_search_budget = "max_logical_queries" in budget or "logical_queries" in usage
    has_split_page_budget = "max_pages_fetched" in budget or "pages_fetched" in usage
    checks = (
        (
            "provider_request_budget_exhausted",
            _as_int(usage.get("search_provider_requests", 0)),
            _as_int(budget.get("max_provider_requests", 0)),
        ),
        (
            (
                "logical_query_budget_exhausted"
                if has_split_search_budget
                else "search_budget_exhausted"
            ),
            _as_int(usage.get("logical_queries", usage.get("searches", 0))),
            _as_int(budget.get("max_logical_queries", budget.get("max_searches", 0))),
        ),
        (
            "fetched_page_budget_exhausted" if has_split_page_budget else "page_budget_exhausted",
            _as_int(usage.get("pages_fetched", usage.get("pages", 0))),
            _as_int(budget.get("max_pages_fetched", budget.get("max_pages", 0))),
        ),
        (
            "page_fetch_attempt_budget_exhausted",
            _as_int(usage.get("page_fetch_attempts", usage.get("pages_fetched", 0))),
            _as_int(
                budget.get(
                    "max_page_fetch_attempts",
                    budget.get("max_pages_fetched", budget.get("max_pages", 0)),
                )
            ),
        ),
        (
            "extracted_page_budget_exhausted" if has_split_page_budget else "page_budget_exhausted",
            _as_int(usage.get("pages_extracted", usage.get("pages", 0))),
            _as_int(budget.get("max_pages_extracted", budget.get("max_pages", 0))),
        ),
        (
            "extraction_call_budget_exhausted",
            _as_int(usage.get("extraction_calls", 0)),
            _as_int(budget.get("max_extraction_calls", 0)),
        ),
        (
            "verification_call_budget_exhausted",
            _as_int(usage.get("verification_calls", 0)),
            _as_int(budget.get("max_verification_calls", 0)),
        ),
        (
            "action_budget_exhausted",
            _as_int(usage.get("scheduler_actions", 0)),
            _as_int(budget.get("max_scheduler_actions", 0)),
        ),
        (
            "token_budget_exhausted",
            _model_tokens(run),
            _as_int(budget.get("max_tokens", 0)),
        ),
        (
            "iteration_budget_exhausted",
            _as_int(usage.get("iterations", 0)),
            _as_int(budget.get("max_iterations", 0)),
        ),
    )
    if bool(usage.get("model_budget_guarded")):
        return "token_budget_exhausted"
    exhausted = next(
        (reason for reason, used, maximum in checks if maximum > 0 and used >= maximum),
        None,
    )
    if exhausted is not None:
        return exhausted
    deadline_at = _deadline_at(run)
    if deadline_at is not None and datetime.now(UTC) >= deadline_at:
        return "deadline_exhausted"
    return None


def _deadline_remaining_seconds(run: ResearchRunRow) -> int:
    deadline_at = _deadline_at(run)
    if deadline_at is None:
        return 0
    return max(0, int((deadline_at - datetime.now(UTC)).total_seconds()))


def _deadline_at(run: ResearchRunRow) -> datetime | None:
    raw = run.budget_snapshot.get("deadline_at")
    if isinstance(raw, str):
        try:
            value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        except ValueError:
            pass
    maximum = _as_int(run.budget_snapshot.get("max_wall_clock_seconds", 0))
    started_at = run.started_at or run.created_at
    if maximum <= 0 or started_at is None:
        return None
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=UTC)
    return started_at + timedelta(seconds=maximum)


def _budget_stop_summary(reason: str) -> str:
    labels = {
        "page_budget_exhausted": "页面读取预算已耗尽",
        "search_budget_exhausted": "搜索预算已耗尽",
        "logical_query_budget_exhausted": "逻辑查询预算已耗尽",
        "provider_request_budget_exhausted": "上游搜索请求预算已耗尽",
        "fetched_page_budget_exhausted": "页面抓取预算已耗尽",
        "page_fetch_attempt_budget_exhausted": "页面读取尝试预算已耗尽",
        "extracted_page_budget_exhausted": "页面抽取预算已耗尽",
        "extraction_call_budget_exhausted": "证据抽取调用预算已耗尽",
        "verification_call_budget_exhausted": "验证调用预算已耗尽",
        "source_space_exhausted": "适用的检索来源空间已耗尽",
        "action_budget_exhausted": "调度动作预算已耗尽",
        "token_budget_exhausted": "模型 Token 预算已耗尽或已触发调用前保护",
        "iteration_budget_exhausted": "研究轮次预算已耗尽",
        "deadline_exhausted": "研究截止时间已到",
    }
    return f"{labels.get(reason, '研究预算已停止扩展')}; 使用现有有效证据生成带限制报告。"


def _replan_reserve_reached(run: ResearchRunRow) -> bool:
    maximum = int(run.budget_snapshot.get("max_iterations", 0) or 0)
    used = int(run.usage_snapshot.get("iterations", 0) or 0)
    reserve = max(2, int(maximum * 0.20))
    return maximum > 0 and maximum - used <= reserve


def _requirements_satisfied(
    question_id: str,
    requirements: list[str],
    counts_by_dimension: Mapping[str, tuple[int, int]],
    *,
    accepted_for_question: int,
    high_risk_dimension_keys: set[str] | frozenset[str] = frozenset(),
) -> bool:
    """Determine completion from the planned dimensions and their source policy."""

    if not requirements:
        return accepted_for_question > 0
    for index, criterion in enumerate(requirements, start=1):
        evidence_count, owner_count = counts_by_dimension.get(
            f"{question_id}:d{index}",
            (0, 0),
        )
        dimension_key = f"{question_id}:d{index}"
        required_sources = (
            2
            if _requires_independent_sources(criterion)
            or dimension_key in high_risk_dimension_keys
            else 1
        )
        if evidence_count < 1 or owner_count < required_sources:
            return False
    return True


def _protected_page_schedule_key(
    *,
    attempts: int,
    coverage: float,
    priority: int,
    question_id: str,
) -> tuple[int, int, float, int, str]:
    """Keep untouched questions ahead of retries while protecting open gaps."""

    return (
        0 if attempts == 0 else 1,
        0,
        coverage,
        priority,
        question_id,
    )


def _p1_variant_schedule_key(
    *,
    attempts: int,
    coverage: float,
    priority: int,
    question_id: str,
) -> tuple[int, int, float, int, str]:
    """Prioritize the weakest P1 before expected-yield optimizations."""

    return (
        1,
        int(max(0.0, min(1.0, coverage)) * 1_000_000),
        float(max(0, attempts)),
        priority,
        question_id,
    )


def _zero_yield_retry_deprioritized(
    *,
    search_budget_exhausted: bool,
    attempts: int,
    coverage: float,
    accepted_evidence: int,
    is_corroboration_target: bool,
) -> bool:
    """Move a fruitless retry behind useful work without making it terminal."""

    return (
        not search_budget_exhausted
        and attempts >= 2
        and coverage <= 0.0
        and accepted_evidence <= 0
        and not is_corroboration_target
    )


def _query_space_exhausted_for_freeze(
    *,
    executed_families: Iterable[str],
    question: str,
) -> bool:
    """Freeze a zero-yield question only after every bounded family ran.

    The two-low-gain threshold is useful for scheduling priority, but it must
    not turn into a terminal decision while alternate-language and
    contradiction searches remain untried.
    """

    valid_families = {family.value for family in QUERY_FAMILIES}
    attempted = {str(value) for value in executed_families} & valid_families
    return len(attempted) >= _query_strategy_limit(question)


def _query_family_capacity_remaining(*, question: str, attempted_families: int) -> bool:
    """Keep protected scheduling aligned with the configured family count."""

    return max(0, attempted_families) < _query_strategy_limit(question)


def _question_low_gain_streak(
    quality_snapshot: Mapping[str, object], question_id: str
) -> int:
    """Read a per-question streak without leaking another question's value."""

    raw_streaks = quality_snapshot.get("low_information_gain_streak_by_question")
    if isinstance(raw_streaks, Mapping):
        if question_id in raw_streaks:
            return max(0, _as_int(raw_streaks.get(question_id, 0)))
        if raw_streaks:
            return 0
    # Legacy snapshots did not have the per-question ledger. Preserve their
    # active question's streak only until the first keyed entry is written.
    return max(0, _as_int(quality_snapshot.get("low_information_gain_streak", 0)))


def _search_query_for_attempt(
    question: str,
    search_hints: list[str],
    requirements: list[str],
    *,
    attempt_index: int,
    unmet_criterion: str | None = None,
    require_authoritative_source: bool = False,
    use_query_family: bool = False,
) -> str:
    """Build one of four explicit, auditable query families.

    Production scheduling opts into ``use_query_family``. The default retains
    the private helper's historical text shape for older replay callers while
    keeping those calls bounded and non-repeating.
    """

    if not use_query_family:
        if attempt_index == 0 and search_hints and not require_authoritative_source:
            return " ".join(search_hints[0].split())[:500]
        criterion = (unmet_criterion or (requirements[0] if requirements else "")).strip()
        chinese = any("\u4e00" <= char <= "\u9fff" for char in question)
        if require_authoritative_source:
            suffix = "官方统计 行业协会" if chinese else "official statistics industry association"
        elif attempt_index == 1:
            suffix = "官方技术文档 技术论文" if chinese else "official technical paper"
        elif attempt_index == 2:
            suffix = "综述 技术论文" if chinese else "case study"
        else:
            suffix = "基准 对比 指标 数据" if chinese else "benchmark comparison metrics data"
        base = (
            search_hints[min(attempt_index, len(search_hints) - 1)].strip()
            if search_hints
            else question.strip()
        )
        parts = (
            base,
            criterion if attempt_index > 0 else "",
            suffix if attempt_index > 0 or require_authoritative_source else "",
        )
        return " ".join(part for part in parts if part)[:500]

    family = query_family_for_attempt(attempt_index)
    if family is None:
        return ""
    criterion = (unmet_criterion or "").strip()
    if not criterion and requirements:
        criterion = requirements[min(attempt_index, len(requirements) - 1)].strip()
    if require_authoritative_source:
        family = QueryFamily.AUTHORITATIVE
    return build_family_query(
        question=question,
        criterion=criterion,
        hints=tuple(search_hints),
        family=family,
    )


def _query_strategy_limit(question: str) -> int:
    del question
    return 4


def _quality_enrichment_needed(*, source_quality: float, cross_validation: float) -> bool:
    """Keep researching while a source-related hard quality gate is still open."""

    return source_quality < 0.75 or cross_validation < 0.70


def _quality_repair_targets(
    coverage_map: list[_CoverageMapEntry],
    *,
    source_quality: float,
    cross_validation: float,
) -> dict[str, list[str]]:
    """Map each open global gate to dimensions that can actually improve it."""

    targets: dict[str, list[str]] = {}
    for question in coverage_map:
        question_id = str(question.get("dimension_key", ""))
        if not question_id:
            continue
        reasons: list[str] = []
        for raw_status in question.get("requirement_statuses", []):
            if not isinstance(raw_status, dict):
                continue
            dimension_key = str(raw_status.get("dimension_key", ""))
            accepted = _as_int(raw_status.get("accepted_evidence", 0))
            independent = _as_int(raw_status.get("independent_sources", 0))
            required = max(1, _as_int(raw_status.get("required_sources", 1)))
            reliability = _as_float(raw_status.get("max_source_reliability", 0.0))
            if cross_validation < 0.70 and accepted > 0 and independent < required:
                reasons.append(f"cross_validation:{dimension_key}")
            if source_quality < 0.75 and accepted > 0 and reliability < 0.75:
                reasons.append(f"source_quality:{dimension_key}")
        if reasons:
            targets[question_id] = reasons
    return targets


def _select_unmet_requirement(
    requirement_statuses: list[object],
) -> tuple[str, str] | None:
    """Choose the least-covered atomic dimension, not merely the first one."""

    candidates: list[tuple[float, int, int, str, str]] = []
    for raw in requirement_statuses:
        if not isinstance(raw, dict) or not raw.get("criterion"):
            continue
        coverage = _as_float(raw.get("coverage", 0.0))
        if coverage >= 1.0:
            continue
        candidates.append(
            (
                coverage,
                _as_int(raw.get("accepted_evidence", 0)),
                _as_int(raw.get("independent_sources", 0)),
                str(raw.get("dimension_key", "")),
                str(raw.get("criterion")),
            )
        )
    if not candidates:
        return None
    _coverage, _accepted, _sources, dimension_key, criterion = min(candidates)
    return dimension_key, criterion


def _source_hint_for_requirement(criterion: str) -> str:
    """Map an unmet dimension to a source genre before repeating a search.

    This is deliberately a small deterministic vocabulary rather than a domain
    classifier. It gives the search provider a chance to return primary sources
    for the dimensions that were missing in the previous attempt, while keeping
    the original question and criterion visible for auditability.
    """

    normalized = criterion.casefold()
    chinese = any("\u4e00" <= char <= "\u9fff" for char in criterion)
    if any(
        token in normalized
        for token in ("市场", "market", "预测", "forecast", "趋势", "trend", "增长", "growth")
    ):
        return (
            "官方统计 行业协会 白皮书 机构预测"
            if chinese
            else ("official statistics industry association white paper outlook")
        )
    if any(
        token in normalized
        for token in (
            "性能",
            "指标",
            "结果",
            "accuracy",
            "precision",
            "recall",
            "auroc",
            "result",
        )
    ):
        return (
            "公开基准 数据集 论文 性能结果"
            if chinese
            else ("benchmark dataset paper results accuracy AUROC F1")
        )
    if any(
        token in normalized
        for token in ("原理", "principle", "算法", "algorithm", "技术", "technology")
    ):
        return "官方技术文档 技术论文" if chinese else "official technical paper application note"
    if any(
        token in normalized
        for token in ("厂商", "厂家", "产品", "vendor", "manufacturer", "product")
    ):
        return (
            "厂商官网 产品页 客户案例"
            if chinese
            else "manufacturer official product page customer case study"
        )
    if any(
        token in normalized
        for token in ("案例", "部署", "deployment", "case", "应用", "application")
    ):
        return "官方案例 应用说明" if chinese else "official application note case study"
    if any(
        token in normalized
        for token in ("比较", "compare", "优势", "advantage", "benchmark", "对比")
    ):
        return "官方规格 基准对比" if chinese else "official specification benchmark comparison"
    return "官方文档 论文" if chinese else "official documentation paper"


def _evidence_dimension_key(
    target: ResearchTarget,
    proposed_key: str | None,
    evidence_text: str,
) -> str:
    allowed = {key: criterion for key, criterion in target.acceptance_dimensions}
    if proposed_key in allowed:
        return str(proposed_key)
    if not allowed:
        return target.question_id
    evidence_tokens = _lexical_tokens(evidence_text)
    ranked = sorted(
        (
            len(evidence_tokens & _lexical_tokens(criterion)),
            key,
        )
        for key, criterion in allowed.items()
    )
    best_score, best_key = ranked[-1]
    # A zero-overlap claim is not evidence for an arbitrary requirement. Keep it
    # question-scoped so it remains auditable but cannot inflate dimension coverage.
    return best_key if best_score > 0 else target.question_id


def _lexical_tokens(value: str) -> set[str]:
    normalized = "".join(char.casefold() if char.isalnum() else " " for char in value)
    words = {word for word in normalized.split() if len(word) > 1}
    cjk = {
        normalized[index : index + 2]
        for index in range(max(0, len(normalized) - 1))
        if all("\u4e00" <= char <= "\u9fff" for char in normalized[index : index + 2])
    }
    return words | cjk


def _requires_independent_sources(criterion: str) -> bool:
    normalized = criterion.casefold()
    markers = (
        "两个",
        "两条",
        "第二",
        "独立来源",
        "two ",
        "2 ",
        "independent",
        # Comparative, market-size, forecast, and leadership claims are
        # high-risk synthesis claims and require corroboration by default.
        "市场规模",
        "市场份额",
        "市场预测",
        "增长率",
        "预测",
        "趋势",
        "第一",
        "领先",
        "比较",
        "market size",
        "market share",
        "forecast",
        "projection",
        "growth rate",
        "leading",
        "largest",
        "comparative",
    )
    return any(marker in normalized for marker in markers)


def _claim_is_high_risk(claim: ResearchClaimRow) -> bool:
    return bool(
        claim.status == "disputed"
        or float(claim.importance or 0.0) >= 0.75
        or claim.claim_type in {"numeric", "forecast", "comparative", "market"}
    )


def _evaluation_summary(decision: str, question_id: str) -> str:
    summaries = {
        "retry_current": f"问题 {question_id} 的证据质量不足; 自动使用替代检索词重试。",
        "retry_technical": (
            f"问题 {question_id} 遇到临时搜索服务故障; 不消耗研究尝试并调度技术重试。"
        ),
        "continue_plan": f"问题 {question_id} 已完成评估; 自动推进下一个研究问题。",
        "continue_cached": (
            "新搜索预算已耗尽; 继续读取和抽取已发现候选, 避免遗失可用证据。"
        ),
        "replan": "连续低信息增益或预算进入预留区; 对原问题执行定向补证。",
        "ready_to_write": "全部研究质量门已满足; 进入报告写作。",
        "write_with_limitations": (
            "计划已遍历但部分验收条件仍未满足; 使用现有证据进入限制性写作。"
        ),
        "stop_budget": "研究预算已耗尽; 已保留当前计划、来源、证据与质量快照。",
        "stop_information_gain": (
            "研究覆盖已达到可接受水平且连续两轮信息增益低于阈值; 停止低边际价值检索。"
        ),
    }
    return summaries.get(decision, "Evaluator 已完成本轮研究质量检查。")
