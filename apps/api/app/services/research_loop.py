"""Budgeted, evidence-driven autonomous research loop."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from urllib.parse import urlsplit
from uuid import UUID

from app.domain.adaptive_scheduler import (
    cheap_triage,
    classify_source_role,
    infer_claim_type,
    source_role_fits_claim,
)
from app.domain.controlled_tools import ControlledToolName, EvidenceSearchInput, ToolDecisionRequest
from app.domain.identifiers import uuid7
from app.domain.provider_adapter import SearchProviderAdapter, SearchRequestContext
from app.domain.provider_health import ProviderHealthState
from app.domain.provider_router import ProviderSelectionDecision
from app.domain.recovery_execution import RecoveryContext
from app.domain.research_context_enricher import ResearchContextEnricher
from app.domain.research_tools import ReadPage, SearchResult
from app.domain.source_policy import is_stable_read_url, normalize_source_url, source_owner_key
from app.infrastructure.artifacts import LocalArtifactStore
from app.infrastructure.db.research_tools import ResearchTarget, ResearchToolRepository
from app.llm.adapters import ModelGatewayError
from app.services.evidence_extractor import EvidenceExtractorService, source_reliability
from app.services.parallel_reads import read_page_attempts_concurrently
from app.tools.errors import ToolExecutionError
from app.tools.gateway import ControlledToolGateway
from app.tools.provider_adapters import BingAdapter, SearXNGAdapter
from app.tools.web_reader import PublicWebReader
from app.tools.web_search import SearXNGSearchProvider

_MAX_SEARCH_RESULTS = 20
_MAX_PAGE_ATTEMPTS = 12
# A single query is exploratory.  Cap it at two pages so one weak result set
# cannot consume the page budget before the next query variant gets a turn.
# The zero-yield guard below still rotates immediately after two unproductive
# pages; this hard cap also protects callers that over-grant a reservation.
_MAX_PAGES_READ = 2


def _provider_request_count(search_provider: object, *, default: int) -> int:
    try:
        value = int(getattr(search_provider, "last_request_count", default))
    except (TypeError, ValueError):
        value = default
    return max(0, value)


def _provider_metric(search_provider: object, name: str) -> int:
    try:
        return max(0, int(getattr(search_provider, name, 0)))
    except (TypeError, ValueError):
        return 0


_ISOLATED_EXTRACTION_ERRORS = {
    "EVIDENCE_OUTPUT_SCHEMA_INVALID",
    "MODEL_OUTPUT_INVALID",
    "MODEL_OUTPUT_TRUNCATED",
    "MODEL_RESPONSE_INVALID",
    "MODEL_TIMEOUT",
    "MODEL_NETWORK_ERROR",
    "MODEL_RATE_LIMITED",
    "MODEL_PROVIDER_UNAVAILABLE",
    "MODEL_TOKEN_BUDGET_EXHAUSTED",
}
_STRUCTURED_EXTRACTION_ERRORS = {
    "EVIDENCE_OUTPUT_SCHEMA_INVALID",
    "MODEL_OUTPUT_INVALID",
    "MODEL_OUTPUT_TRUNCATED",
    "MODEL_RESPONSE_INVALID",
}
_DURABLE_EXTRACTION_TRANSPORT_ERRORS = {
    "MODEL_TIMEOUT",
    "MODEL_NETWORK_ERROR",
    "MODEL_RATE_LIMITED",
    "MODEL_PROVIDER_UNAVAILABLE",
}
_STRUCTURED_EXTRACTION_FAILURE_LIMIT = 2

logger = logging.getLogger(__name__)
_PREFERRED_READABLE_DOMAINS = {
    "arxiv.org",
    "openaccess.thecvf.com",
    "pmc.ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
}
_RESTRICTED_SOURCE_DOMAINS = {
    "doi.org",
    "ieeexplore.ieee.org",
    "link.springer.com",
    "researchgate.net",
    "sciencedirect.com",
    "tandfonline.com",
    "csdn.net",
    "wenku.so.com",
    "wenku.baidu.com",
    "baike.baidu.com",
    "zhihu.com",
}
_GENERIC_QUERY_TOKENS = {
    "official",
    "documentation",
    "primary",
    "source",
    "government",
    "university",
    "paper",
    "association",
    "manufacturer",
    "report",
    "study",
    "case",
    "官方",
    "原始",
    "来源",
    "政府",
    "高校",
    "论文",
    "行业",
    "协会",
    "制造",
    "官网",
    "文档",
}
# Search engines often return pages that share method words ("深度学习",
# "阈值", "检测") but are unrelated to industrial-vision defect inspection.
# These anchors are deliberately small and domain-oriented: a candidate must
# share at least one anchor with the query when the query contains any.
_TOPIC_DOMAIN_ANCHORS = {
    "工业",
    "缺陷",
    "机器",
    "视觉",
    "质检",
    "表面",
    "industrial",
    "defect",
    "machine",
    "vision",
    "inspection",
}
_INDUSTRIAL_CONTEXT_ANCHORS = {
    "工业",
    "industrial",
    "制造",
    "manufacturing",
    "产线",
    "生产线",
    "factory",
}
_DEFECT_INSPECTION_ANCHORS = {
    "缺陷",
    "defect",
    "质检",
    "检测",
    "inspection",
    "异常",
    "anomaly",
    "表面",
    "surface",
}
_VISUAL_TOPIC_ANCHORS = {
    "视觉",
    "图像",
    "影像",
    "相机",
    "光学",
    "机器视觉",
    "vision",
    "visual",
    "image",
    "imaging",
    "camera",
    "optical",
}
_NON_VISUAL_ANOMALY_ANCHORS = {
    "控制系统",
    "工业控制",
    "网络异常",
    "入侵检测",
    "过程异常",
    "控制网络",
    "control system",
    "industrial control",
    "network anomaly",
    "intrusion detection",
    "process anomaly",
    "control network",
}


@dataclass(frozen=True, slots=True)
class ResearchIterationResult:
    outcome: str
    continue_research: bool
    decision: str
    pages_read: int
    accepted_evidence: int
    coverage: float
    information_gain: float
    low_information_gain_streak: int


@dataclass(frozen=True, slots=True)
class RecoveryAttemptContext:
    attempt_id: UUID
    coverage_before: float
    gap_before: tuple[str, ...]
    accepted_evidence_before: int
    tokens_reserved: int


@dataclass(frozen=True, slots=True)
class ResearchAttemptResult:
    pages_read: int
    accepted_evidence: int
    outcome: str
    recovery_attempt_id: UUID | None = None
    recovery_coverage_before: float = 0.0
    recovery_gap_before: tuple[str, ...] = ()
    recovery_accepted_evidence_before: int = 0
    recovery_tokens_reserved: int = 0
    recovery_attempts: tuple[RecoveryAttemptContext, ...] = ()


class ResearchLoopService:
    def __init__(
        self,
        repository: ResearchToolRepository,
        search: SearXNGSearchProvider,
        reader: PublicWebReader,
        extractor: EvidenceExtractorService,
        artifacts: LocalArtifactStore,
        controlled_tools: ControlledToolGateway | None = None,
        *,
        parallel_reads_enabled: bool = False,
        search_adapters: Mapping[str, SearchProviderAdapter] | None = None,
        evidence_aware_context_enabled: bool = False,
    ) -> None:
        self._repository = repository
        self._search = search
        self._reader = reader
        self._extractor = extractor
        self._artifacts = artifacts
        self._controlled_tools = controlled_tools
        self._parallel_reads_enabled = parallel_reads_enabled
        # Phase 14.2: the single query-transformation layer between the Planner
        # producing a SearchTarget and the Provider executing its query. It is
        # stateless and only appends ResearchContext hints; planner decisions,
        # strategies, provider selection, and budgets stay untouched.
        self._context_enricher = ResearchContextEnricher()
        # Phase 14.3 A/B switch: when disabled the loop bypasses the
        # enrichment layer in its entirety, keeping the Planner ->
        # SearchTarget -> Provider path identical to the Phase 12.4 baseline.
        self._evidence_aware_context_enabled = evidence_aware_context_enabled
        # Phase 12.4 B: a single provider-name -> adapter dispatch map. The
        # Router-selected provider is looked up here and executed directly, so
        # there is no provider-string branching and no provider-side fallback.
        # When the worker injects a full map we use it verbatim; otherwise a
        # SearXNG/Bing default map is synthesized so the router-authoritative
        # branch always has an adapter to dispatch to.
        if search_adapters is not None:
            self._search_adapters: dict[str, SearchProviderAdapter] = dict(search_adapters)
        elif isinstance(search, SearXNGSearchProvider):
            self._search_adapters = {
                "SearXNG": SearXNGAdapter(search),
                "Bing": BingAdapter(search),
            }
        else:
            self._search_adapters = {}

    async def run_one_iteration(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
    ) -> ResearchIterationResult:
        """Execute one bounded research iteration for one LangGraph super-step."""

        target = await self._repository.prepare_target(
            run_id,
            worker_task_id=worker_task_id,
        )
        if target is None:
            replan_checker = getattr(self._repository, "replan_requested", None)
            replan_requested = (
                await replan_checker(run_id, worker_task_id=worker_task_id)
                if replan_checker is not None
                else False
            )
            continue_research = await self._repository.research_phase_active(
                run_id,
                worker_task_id=worker_task_id,
            )
            decision = (
                "replan"
                if replan_requested
                else "scheduler_advanced"
                if continue_research
                else "no_pending_question"
            )
            return ResearchIterationResult(
                outcome=self._outcome(decision, 0, 0, 0),
                continue_research=continue_research,
                decision=decision,
                pages_read=0,
                accepted_evidence=0,
                coverage=0.0,
                information_gain=0.0,
                low_information_gain_streak=0,
            )

        target = await self._enrich_target_with_research_context(
            run_id,
            worker_task_id=worker_task_id,
            target=target,
        )
        attempt = await self._research_target(
            run_id,
            worker_task_id=worker_task_id,
            target=target,
        )
        evaluation = await self._repository.finish_iteration(
            run_id,
            worker_task_id=worker_task_id,
            target=target,
            attempt_outcome=attempt.outcome,
        )
        recovery_recorder = getattr(self._repository, "record_recovery_event", None)
        if callable(recovery_recorder):
            recovery_attempts = attempt.recovery_attempts
            if not recovery_attempts and attempt.recovery_attempt_id is not None:
                recovery_attempts = (
                    RecoveryAttemptContext(
                        attempt_id=attempt.recovery_attempt_id,
                        coverage_before=attempt.recovery_coverage_before,
                        gap_before=attempt.recovery_gap_before,
                        accepted_evidence_before=attempt.recovery_accepted_evidence_before,
                        tokens_reserved=attempt.recovery_tokens_reserved,
                    ),
                )
            for recovery in recovery_attempts:
                gap_after = (
                    () if evaluation.coverage >= 1.0 else recovery.gap_before
                )
                await recovery_recorder(
                    run_id,
                    worker_task_id=worker_task_id,
                    event_type="recovery.completed",
                    question_id=target.question_id,
                    plan_version=target.plan_version,
                    attempt_id=recovery.attempt_id,
                    coverage_before=recovery.coverage_before,
                    coverage_after=evaluation.coverage,
                    gap_before=recovery.gap_before,
                    gap_after=gap_after,
                    accepted_evidence_before=recovery.accepted_evidence_before,
                    accepted_evidence_after=recovery.accepted_evidence_before
                    + attempt.accepted_evidence,
                    tokens_reserved=recovery.tokens_reserved,
                    reason=attempt.outcome,
                )
        return ResearchIterationResult(
            outcome=self._outcome(
                evaluation.decision,
                0 if attempt.outcome == "provider_error" else 1,
                attempt.pages_read,
                attempt.accepted_evidence,
            ),
            continue_research=evaluation.continue_research,
            decision=evaluation.decision,
            pages_read=attempt.pages_read,
            accepted_evidence=attempt.accepted_evidence,
            coverage=evaluation.coverage,
            information_gain=evaluation.information_gain,
            low_information_gain_streak=evaluation.low_information_gain_streak,
        )

    async def run_iteration(self, run_id: UUID, *, worker_task_id: str) -> str:
        """Compatibility wrapper; the production Graph calls one iteration per node."""

        completed_iterations = 0
        total_pages = 0
        total_accepted = 0
        last_decision = "no_pending_question"
        while True:
            result = await self.run_one_iteration(
                run_id,
                worker_task_id=worker_task_id,
            )
            completed_iterations += 1 if result.decision != "no_pending_question" else 0
            total_pages += result.pages_read
            total_accepted += result.accepted_evidence
            last_decision = result.decision
            if not result.continue_research:
                return self._outcome(
                    last_decision,
                    completed_iterations,
                    total_pages,
                    total_accepted,
                )

    async def _enrich_target_with_research_context(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
    ) -> ResearchTarget:
        """Phase 14.2: apply the evidence-aware query enrichment to one target.

        The only permitted path is ResearchLoop -> ResearchContextResolver ->
        ResearchContextEnricher -> SearchTarget: the resolver is reached through
        the repository interface (never by reading ``usage_snapshot`` here), and
        when no ResearchContext exists the target is returned byte-identical
        with no event, keeping every legacy run's behavior unchanged.  The
        Phase 14.3 A/B switch bypasses this layer entirely when disabled.
        """

        if not self._evidence_aware_context_enabled:
            # Baseline arm: skip enrichment before any resolver access so the
            # query and the event stream stay identical to Phase 12.4.
            return target
        resolver = getattr(self._repository, "research_context_for_question", None)
        if not callable(resolver):
            return target
        resolved = await resolver(
            run_id,
            worker_task_id=worker_task_id,
            question_id=target.question_id,
        )
        if resolved is None:
            # No context: no enrichment, no event, byte-identical query.
            return target
        enrichment = self._context_enricher.enrich(target.query, resolved.context)
        event_recorder = getattr(self._repository, "record_query_context_enriched", None)
        if callable(event_recorder):
            await event_recorder(
                run_id,
                worker_task_id=worker_task_id,
                target=target,
                resolved=resolved,
                enrichment=enrichment,
            )
        if not enrichment.changed:
            return target
        return replace(target, query=enrichment.enriched_query)

    async def _research_target(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
    ) -> ResearchAttemptResult:
        event_cursor_reader = getattr(self._repository, "research_event_cursor", None)
        iteration_start_seq: int | None = None
        if callable(event_cursor_reader):
            iteration_start_seq = await event_cursor_reader(
                run_id,
                worker_task_id=worker_task_id,
            )
        minimum_call = self._extractor.estimate_minimum_request_tokens(
            question=target.question,
            acceptance_dimensions=target.acceptance_dimensions,
        )
        if _deadline_expired(target):
            return ResearchAttemptResult(0, 0, "deadline_exhausted")
        initial_budget = await self._repository.evidence_model_budget(
            run_id,
            worker_task_id=worker_task_id,
            question_id=target.question_id,
            minimum_call=minimum_call.total_tokens,
        )
        if (
            not initial_budget.allowed
            or initial_budget.max_call_tokens < minimum_call.total_tokens
        ):
            # A fairness yield is temporary and must never be persisted as
            # question exhaustion. The scheduler owns fairness; the evaluator
            # records a non-consuming yield if a legacy caller still returns
            # one here.
            return ResearchAttemptResult(0, 0, initial_budget.outcome)
        if _deadline_expired(target):
            return ResearchAttemptResult(0, 0, "deadline_exhausted")
        if self._controlled_tools is not None:
            await self._controlled_tools.search_evidence(
                EvidenceSearchInput(
                    run_id=run_id,
                    action_id=target.tool_call_id,
                    target_gap_ids=(target.gap_id,),
                    question_id=target.question_id,
                    query=target.query,
                )
            )
            self._controlled_tools.authorize_web_search(
                ToolDecisionRequest(
                    action_id=target.tool_call_id,
                    tool_name=ControlledToolName.WEB_SEARCH,
                    target_gap_ids=(target.gap_id,),
                    duplicate_key=f"web-search:{target.tool_call_id}",
                    evidence_checked=True,
                )
            )
        cached_candidates = bool(target.reusable_results or target.reusable_pages)
        # A cache-only pass is cheap, but repeating it indefinitely traps a
        # question on the same low-yield pages. Alternate cache-only and fresh
        # query passes so retries exploit saved work while still discovering
        # new sources and advancing the audited search budget.
        reuse_only = (
            target.search_budget_exhausted
            or target.query_already_executed
            or (cached_candidates and target.gap_attempt_index % 2 == 0)
        )
        query_started_recorder = getattr(
            self._repository,
            "record_search_query_started",
            None,
        )
        if callable(query_started_recorder):
            await query_started_recorder(
                run_id,
                worker_task_id=worker_task_id,
                target=target,
                reused=reuse_only,
            )
        provider_requests = 0
        provider_timeouts = 0
        provider_fallbacks = 0
        provider_healthy = 0
        provider_unresponsive = 0
        provider_productive = 0
        search_latency_ms = 0
        # Phase 12.4: provider the Router actually selected and executed this
        # attempt, plus the object whose ``last_*`` counters feed budget
        # accounting. ``executed_source`` defaults to the shared search provider
        # so the reuse / non-SearXNG (test-double) paths keep their exact
        # pre-12.4 telemetry; the router branch overrides both to the executed
        # adapter's underlying source.
        executed_provider: str | None = None
        executed_source: object = self._search
        if reuse_only:
            results = list(target.reusable_results)
            audited_results = list(results)
        else:
            try:
                if _deadline_expired(target):
                    return ResearchAttemptResult(0, 0, "deadline_exhausted")
                search_started = time.monotonic()
                remaining = _deadline_remaining_seconds(target)
                if remaining <= 0:
                    return ResearchAttemptResult(0, 0, "deadline_exhausted")
                if isinstance(self._search, SearXNGSearchProvider):
                    # Phase 12.2: Router is authoritative for provider selection.
                    # The repository may not expose the router in unit-test
                    # fakes, so fall back to a synthetic HEALTHY SearXNG
                    # decision that preserves the pre-12.2 execution shape
                    # (still single-provider, still no Bing side-switch here).
                    selector = getattr(
                        self._repository, "select_provider_for_target", None
                    )
                    decision: ProviderSelectionDecision | None = None
                    if callable(selector):
                        decision = await selector(
                            run_id,
                            worker_task_id=worker_task_id,
                            target=target,
                        )
                    if decision is None:
                        decision = ProviderSelectionDecision(
                            selected_provider="SearXNG",
                            excluded_providers=(),
                            reason="health_router_unavailable_default_provider",
                            health_state=ProviderHealthState.UNKNOWN,
                            fallback_used=False,
                            feedback_execution_id=target.feedback_execution_id,
                        )
                    execution_recorder = getattr(
                        self._repository,
                        "record_provider_execution_started",
                        None,
                    )
                    if callable(execution_recorder) and decision.selected_provider:
                        await execution_recorder(
                            run_id,
                            worker_task_id=worker_task_id,
                            target=target,
                            decision=decision,
                        )
                    if decision.selected_provider is None:
                        # F.2: All providers unavailable or the switch budget
                        # is exhausted. Emit a bounded tool failure so the
                        # evaluator records a provider_error outcome, and never
                        # crash the worker.
                        raise ToolExecutionError(
                            "SEARCH_PROVIDER_DEGRADED",
                            retryable=True,
                            details={
                                "provider": "",
                                "failure_type": "circuit_open",
                                "context": {
                                    "strategy": "health_routing_stop",
                                    "reason": decision.reason,
                                    "query": target.query[:500],
                                    "fallback_attempted": False,
                                    "fallback_provider": None,
                                    "fallback_success": False,
                                },
                                "failures": [],
                            },
                        )
                    selected = decision.selected_provider or ""
                    adapter = self._search_adapters.get(selected)
                    if adapter is None:
                        # The Router selected a provider that is not wired into
                        # the dispatch map. Never substitute another provider
                        # (no silent fallback): stop as a bounded provider error.
                        raise ToolExecutionError(
                            "SEARCH_PROVIDER_DEGRADED",
                            retryable=True,
                            details={
                                "provider": selected,
                                "failure_type": "circuit_open",
                                "context": {
                                    "strategy": "adapter_dispatch_missing",
                                    "reason": "selected_provider_has_no_adapter",
                                    "query": target.query[:500],
                                    "fallback_attempted": False,
                                    "fallback_provider": None,
                                    "fallback_success": False,
                                },
                                "failures": [],
                            },
                        )
                    executed_provider = selected
                    executed_source = getattr(adapter, "execution_source", adapter)
                    request_context = SearchRequestContext(
                        provider_request_allowance=target.provider_request_allowance,
                        alternate_query=target.alternate_query or None,
                    )
                    results = await asyncio.wait_for(
                        adapter.search(
                            target.query,
                            limit=_MAX_SEARCH_RESULTS,
                            exclude_domains=target.search_excluded_owner_keys,
                            exclude_urls=target.search_excluded_urls,
                            request_context=request_context,
                        ),
                        timeout=remaining,
                    )
                else:
                    results = await asyncio.wait_for(
                        self._search.search(target.query, limit=_MAX_SEARCH_RESULTS),
                        timeout=remaining,
                    )
                search_latency_ms = int((time.monotonic() - search_started) * 1_000)
                provider_requests = _provider_request_count(executed_source, default=1)
                provider_timeouts = _provider_metric(executed_source, "last_timeout_count")
                provider_fallbacks = _provider_metric(executed_source, "last_fallback_count")
                provider_healthy = _provider_metric(
                    executed_source, "last_healthy_response_count"
                )
                provider_unresponsive = _provider_metric(
                    executed_source, "last_unresponsive_response_count"
                )
                provider_productive = _provider_metric(
                    executed_source, "last_productive_response_count"
                )
            except TimeoutError:
                return ResearchAttemptResult(0, 0, "deadline_exhausted")
            except ToolExecutionError as exc:
                search_latency_ms = int((time.monotonic() - search_started) * 1_000)
                provider_requests = _provider_request_count(executed_source, default=0)
                provider_timeouts = _provider_metric(executed_source, "last_timeout_count")
                provider_fallbacks = _provider_metric(executed_source, "last_fallback_count")
                provider_healthy = _provider_metric(
                    executed_source, "last_healthy_response_count"
                )
                provider_unresponsive = _provider_metric(
                    executed_source, "last_unresponsive_response_count"
                )
                provider_productive = _provider_metric(
                    executed_source, "last_productive_response_count"
                )
                await self._repository.record_tool_failure(
                    run_id,
                    worker_task_id=worker_task_id,
                    target=target,
                    error_code=exc.code,
                    retryable=exc.retryable,
                    details=exc.details,
                    provider_requests=provider_requests,
                    provider_timeouts=provider_timeouts,
                    provider_fallbacks=provider_fallbacks,
                    provider_healthy=provider_healthy,
                    provider_unresponsive=provider_unresponsive,
                    provider_productive=provider_productive,
                    latency_ms=search_latency_ms,
                    executed_provider=executed_provider,
                )
                if exc.retryable:
                    # SearXNG's internal fallback strategies are exhausted, but
                    # this remains an evaluable research outcome. Preserve the
                    # provider telemetry and let the evaluator decide whether
                    # existing evidence is sufficient to write or must fail.
                    return ResearchAttemptResult(0, 0, "provider_error")
                return ResearchAttemptResult(0, 0, "provider_error")

            # Only results actually returned for this query belong to its audit
            # record. Unread candidates from an older query may still be useful
            # for this turn, but must not be rewritten as if the new search found
            # them.
            audited_results = list(results)
            known_fresh_urls = {normalize_source_url(result.url) for result in results}
            results.extend(
                result
                for result in target.reusable_results
                if normalize_source_url(result.url) not in known_fresh_urls
            )

        reusable_pages = {
            normalize_source_url(page.final_url): page
            for page in target.reusable_pages
            if _is_read_candidate(page.final_url)
        }
        page_results = _deduplicate_search_results(
            [result for result in results if _is_read_candidate(result.url)]
        )
        known_urls = {normalize_source_url(result.url) for result in page_results}
        for reusable in target.reusable_pages:
            if normalize_source_url(reusable.final_url) not in known_urls:
                page_results.append(
                    SearchResult(
                        title=reusable.title,
                        url=reusable.final_url,
                        snippet="Previously fetched artifact; re-extracting without network.",
                        rank=len(page_results) + 1,
                    )
                )
        recorded_candidate_urls = await self._repository.record_search_results(
            run_id,
            worker_task_id=worker_task_id,
            target=target,
            results=audited_results,
            reused=reuse_only,
            provider_requests=provider_requests,
            provider_timeouts=provider_timeouts,
            provider_fallbacks=provider_fallbacks,
            provider_healthy=provider_healthy,
            provider_unresponsive=provider_unresponsive,
            provider_productive=provider_productive,
            latency_ms=search_latency_ms,
            executed_provider=executed_provider,
        )

        dispatch_recorder = getattr(self._repository, "record_candidate_dispatch_event", None)

        async def record_dispatch_event(
            url: str,
            stage: str,
            *,
            reason: str | None = None,
        ) -> None:
            if not callable(dispatch_recorder):
                return
            await dispatch_recorder(
                run_id,
                worker_task_id=worker_task_id,
                target=target,
                url=url,
                stage=stage,
                reason=reason,
            )

        candidate_results = _deduplicate_search_results(
            [
                result
                for result in audited_results
                if recorded_candidate_urls is None
                or normalize_source_url(result.url) in recorded_candidate_urls
            ]
        )
        for result in candidate_results:
            await record_dispatch_event(result.url, "dispatch_started")

        if not page_results:
            for result in candidate_results:
                await record_dispatch_event(
                    result.url,
                    "dispatch_skipped",
                    reason="relevance_rank",
                )
            return ResearchAttemptResult(0, 0, "zero_results")
        pages_read = 0
        accepted = 0
        recovery_attempt_id: UUID | None = None
        recovery_coverage_before = 0.0
        recovery_gap_before: tuple[str, ...] = ()
        recovery_accepted_evidence_before = 0
        recovery_tokens_reserved = 0
        recovery_attempts: list[RecoveryAttemptContext] = []
        active_recovery_context: RecoveryAttemptContext | None = None

        def attempt_result(outcome: str) -> ResearchAttemptResult:
            return ResearchAttemptResult(
                pages_read,
                accepted,
                outcome,
                recovery_attempt_id=recovery_attempt_id,
                recovery_coverage_before=recovery_coverage_before,
                recovery_gap_before=recovery_gap_before,
                recovery_accepted_evidence_before=recovery_accepted_evidence_before,
                recovery_tokens_reserved=recovery_tokens_reserved,
                recovery_attempts=tuple(recovery_attempts),
            )

        async def record_recovery_failure(reason: str) -> None:
            if active_recovery_context is None:
                return
            recovery_recorder = getattr(self._repository, "record_recovery_event", None)
            if not callable(recovery_recorder):
                return
            await recovery_recorder(
                run_id,
                worker_task_id=worker_task_id,
                event_type="recovery.failed",
                question_id=target.question_id,
                plan_version=target.plan_version,
                attempt_id=active_recovery_context.attempt_id,
                coverage_before=active_recovery_context.coverage_before,
                coverage_after=None,
                gap_before=active_recovery_context.gap_before,
                gap_after=active_recovery_context.gap_before,
                accepted_evidence_before=active_recovery_context.accepted_evidence_before,
                accepted_evidence_after=active_recovery_context.accepted_evidence_before
                + accepted,
                tokens_reserved=active_recovery_context.tokens_reserved,
                reason=reason,
            )
        structured_extraction_failures = 0
        page_reservation = await self._repository.reserve_page_slots(
            run_id,
            worker_task_id=worker_task_id,
            requested=(
                _MAX_PAGES_READ
                if target.first_pass and target.priority == 1
                else 1
                if target.first_pass
                else min(_MAX_PAGE_ATTEMPTS, _MAX_PAGES_READ)
            ),
            fresh_search=not reuse_only,
        )
        if page_reservation.granted == 0:
            for result in candidate_results:
                await record_dispatch_event(
                    result.url,
                    "dispatch_skipped",
                    reason="page_budget",
                )
            return ResearchAttemptResult(0, 0, "budget_exhausted")
        desired_page_limit = (
            _MAX_PAGES_READ
            if not target.first_pass or target.priority == 1
            else 1
        )
        page_limit = min(page_reservation.granted, desired_page_limit)
        prefetched: dict[str, ReadPage] = {}
        batch_urls: list[str] = []
        settled_slots = 0
        ordered_results = _prioritize_search_results(
            page_results,
            query=target.query,
            alternate_query=target.alternate_query,
            acceptance_criteria=tuple(
                criterion for _key, criterion in target.acceptance_dimensions
            ),
            used_owner_keys=set(target.used_source_owner_keys),
            owner_acceptance_rates=dict(target.owner_acceptance_rates),
        )
        ordered_urls = {normalize_source_url(result.url) for result in ordered_results}
        dispatch_skipped_urls: set[str] = set()
        for result in candidate_results:
            normalized_url = normalize_source_url(result.url)
            if normalized_url not in ordered_urls:
                await record_dispatch_event(
                    result.url,
                    "dispatch_skipped",
                    reason="relevance_rank",
                )
                dispatch_skipped_urls.add(normalized_url)
        dispatch_selected_urls: set[str] = set()

        async def mark_dispatch_selected(result: SearchResult) -> None:
            normalized_url = normalize_source_url(result.url)
            if normalized_url in dispatch_selected_urls:
                return
            await record_dispatch_event(result.url, "dispatch_selected")
            dispatch_selected_urls.add(normalized_url)

        async def record_reader_events(requested_url: str) -> None:
            event_consumer = getattr(self._reader, "consume_events", None)
            lifecycle_recorder = getattr(self._repository, "record_reader_lifecycle", None)
            if not callable(event_consumer) or not callable(lifecycle_recorder):
                return
            events = event_consumer(requested_url)
            if not isinstance(events, list) or not events:
                return
            await lifecycle_recorder(
                run_id,
                worker_task_id=worker_task_id,
                target=target,
                requested_url=requested_url,
                events=events,
            )

        selection_recorder = getattr(
            self._repository,
            "record_evidence_selection_event",
            None,
        )

        async def record_selection_event(
            url: str,
            stage: str,
            *,
            reason: str | None = None,
            source_id: UUID | str | None = None,
        ) -> None:
            if not callable(selection_recorder):
                return
            await selection_recorder(
                run_id,
                worker_task_id=worker_task_id,
                target=target,
                url=url,
                stage=stage,
                reason=reason,
                source_id=source_id,
            )

        if self._parallel_reads_enabled and page_limit > 0:
            batch_sample = ordered_results[:page_limit]
            for result in batch_sample:
                await mark_dispatch_selected(result)
                if normalize_source_url(result.url) not in reusable_pages:
                    batch_urls.append(result.url)
            if batch_urls:
                read_map = await read_page_attempts_concurrently(
                    batch_urls,
                    self._reader,
                    deadline_at=target.deadline_at,
                )
                for url, read_outcome in read_map.items():
                    await record_reader_events(url)
                    if read_outcome.page is None:
                        await self._repository.record_page_failure(
                            run_id,
                            worker_task_id=worker_task_id,
                            target=target,
                            url=url,
                            error_code=read_outcome.error_code or "WEBPAGE_PREFETCH_FAILED",
                            latency_ms=read_outcome.latency_ms,
                        )
                    else:
                        prefetched[url] = read_outcome.page
                        await self._repository.record_page_fetched(
                            run_id,
                            worker_task_id=worker_task_id,
                            target=target,
                            url=read_outcome.page.final_url,
                            reused=False,
                            latency_ms=read_outcome.latency_ms,
                        )
                    settled_slots += 1

        zero_yield_pages = 0
        question_budget_exhausted = False
        question_budget_yielded = False

        async def cleanup_attempt(attempt_id: UUID, *, uncertain: bool) -> None:
            """Close reservations when work fails between ledger boundaries.

            Artifact/DB failures used to leave both a model reservation and
            page slots live until lease reconciliation. That made a retry look
            budget-exhausted even though no new provider call was made.
            """

            try:
                if uncertain:
                    await self._repository.mark_model_reservation_uncertain(
                        run_id,
                        worker_task_id=worker_task_id,
                        attempt_id=attempt_id,
                    )
                else:
                    await self._repository.release_model_reservation(
                        run_id,
                        worker_task_id=worker_task_id,
                        attempt_id=attempt_id,
                    )
            finally:
                await self._repository.release_extraction_slot(
                    run_id,
                    worker_task_id=worker_task_id,
                )
                await self._repository.release_page_slots(
                    run_id,
                    worker_task_id=worker_task_id,
                    count=page_reservation.granted,
                )

        for result in ordered_results[:_MAX_PAGE_ATTEMPTS]:
            if pages_read >= page_limit:
                break
            if _deadline_expired(target):
                await self._repository.release_page_slots(
                    run_id,
                    worker_task_id=worker_task_id,
                    count=max(0, page_reservation.granted - settled_slots),
                )
                return attempt_result("deadline_exhausted")
            await mark_dispatch_selected(result)
            fetch_started = time.monotonic()
            reusable_ref = reusable_pages.get(normalize_source_url(result.url))
            if reusable_ref is not None:
                try:
                    text = await self._artifacts.read_text(reusable_ref.artifact_uri)
                    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if content_hash != reusable_ref.content_hash:
                        raise ValueError("artifact content hash mismatch")
                    page = ReadPage(
                        final_url=reusable_ref.final_url,
                        title=reusable_ref.title,
                        clean_text=text,
                        content_hash=reusable_ref.content_hash,
                        fetched_at=reusable_ref.fetched_at,
                        published_at=reusable_ref.published_at,
                    )
                except (OSError, ValueError):
                    reusable_ref = None
            if reusable_ref is None:
                if self._parallel_reads_enabled and result.url in batch_urls:
                    prefetched_page = prefetched.get(result.url)
                    if prefetched_page is None:
                        continue
                    page = prefetched_page
                else:
                    try:
                        remaining = _deadline_remaining_seconds(target)
                        if remaining <= 0:
                            await self._repository.release_page_slots(
                                run_id,
                                worker_task_id=worker_task_id,
                                count=max(0, page_reservation.granted - settled_slots),
                            )
                            return attempt_result("deadline_exhausted")
                        page = await asyncio.wait_for(
                            self._reader.read(result.url),
                            timeout=remaining,
                        )
                    except TimeoutError:
                        await record_reader_events(result.url)
                        await self._repository.record_page_failure(
                            run_id,
                            worker_task_id=worker_task_id,
                            target=target,
                            url=result.url,
                            error_code="DEADLINE_EXHAUSTED",
                            latency_ms=int((time.monotonic() - fetch_started) * 1_000),
                        )
                        settled_slots += 1
                        await self._repository.release_page_slots(
                            run_id,
                            worker_task_id=worker_task_id,
                            count=max(0, page_reservation.granted - settled_slots),
                        )
                        return attempt_result("deadline_exhausted")
                    except ToolExecutionError as exc:
                        await record_reader_events(result.url)
                        await self._repository.record_page_failure(
                            run_id,
                            worker_task_id=worker_task_id,
                            target=target,
                            url=result.url,
                            error_code=exc.code,
                            latency_ms=int((time.monotonic() - fetch_started) * 1_000),
                        )
                        settled_slots += 1
                        continue
                    except Exception:
                        await record_reader_events(result.url)
                        raise

            await record_reader_events(result.url)
            fetch_latency_ms = (
                0 if reusable_ref is not None else int((time.monotonic() - fetch_started) * 1_000)
            )
            if result.url not in batch_urls:
                await self._repository.record_page_fetched(
                    run_id,
                    worker_task_id=worker_task_id,
                    target=target,
                    url=page.final_url,
                    reused=reusable_ref is not None,
                    latency_ms=fetch_latency_ms,
                )
                settled_slots += 1

            await record_selection_event(result.url, "selection_started")
            if await self._repository.page_already_processed(
                run_id,
                worker_task_id=worker_task_id,
                target=target,
                requested_url=result.url,
                page=page,
            ):
                await record_selection_event(
                    result.url,
                    "selection_skipped",
                    reason="already_processed",
                )
                continue
            pages_read += 1

            if target.cheap_triage_enabled:
                triage = cheap_triage(
                    question=target.question,
                    criteria=tuple(criterion for _key, criterion in target.acceptance_dimensions),
                    query_hints=tuple(
                        value
                        for value in (target.query, target.alternate_query)
                        if value
                    ),
                    text=page.clean_text,
                    url=page.final_url,
                )
                if not triage.accepted:
                    await record_selection_event(
                        result.url,
                        "selection_skipped",
                        reason=(
                            "question_mismatch"
                            if triage.reason == "topic_mismatch"
                            else "low_evidence_value"
                        ),
                    )
                    await self._repository.record_triage_rejection(
                        run_id,
                        worker_task_id=worker_task_id,
                        target=target,
                        url=page.final_url,
                        requested_url=result.url,
                        score=triage.score,
                        reason=triage.reason,
                        source_role=triage.source_role,
                    )
                    continue

            if not await self._repository.reserve_extraction_slot(
                run_id,
                worker_task_id=worker_task_id,
            ):
                await record_selection_event(
                    result.url,
                    "selection_skipped",
                    reason="extraction_budget",
                )
                break
            if _deadline_expired(target):
                await self._repository.release_extraction_slot(
                    run_id,
                    worker_task_id=worker_task_id,
                )
                await self._repository.release_page_slots(
                    run_id,
                    worker_task_id=worker_task_id,
                    count=max(0, page_reservation.granted - settled_slots),
                )
                return attempt_result("deadline_exhausted")
            model_budget = await self._repository.evidence_model_budget(
                run_id,
                worker_task_id=worker_task_id,
                question_id=target.question_id,
                minimum_call=minimum_call.total_tokens,
            )
            if not model_budget.allowed or model_budget.max_call_tokens < minimum_call.total_tokens:
                await record_selection_event(
                    result.url,
                    "selection_skipped",
                    reason=(
                        "question_budget"
                        if model_budget.outcome == "yield_question"
                        else "token_budget"
                    ),
                )
                await self._repository.release_extraction_slot(
                    run_id,
                    worker_task_id=worker_task_id,
                )
                break

            # Atomic per-attempt token reservation: the ledger holds the
            # conservative call cost while the provider request is in flight,
            # so concurrent attempts cannot jointly overspend the run budget.
            attempt_id = uuid7()
            reservation = await self._repository.reserve_model_tokens(
                run_id,
                worker_task_id=worker_task_id,
                question_id=target.question_id,
                node="evidence_extractor",
                attempt_id=attempt_id,
                max_output=min(model_budget.max_call_tokens // 4, 4_000),
                estimated_input=(
                    model_budget.max_call_tokens - min(model_budget.max_call_tokens // 4, 4_000)
                ),
            )
            if not reservation.granted:
                await record_selection_event(
                    result.url,
                    "selection_skipped",
                    reason=(
                        "question_budget"
                        if reservation.status
                        in {
                            "question_budget_denied",
                            "question_budget_exhausted",
                            "question_budget_yielded",
                        }
                        else "token_budget"
                    ),
                )
                await self._repository.release_extraction_slot(
                    run_id,
                    worker_task_id=worker_task_id,
                )
                question_budget_exhausted = reservation.status in {
                    "question_budget_denied",
                    "question_budget_exhausted",
                }
                question_budget_yielded = reservation.status == "question_budget_yielded"
                break

            if reservation.recovery_attempt_id is not None:
                recovery_context = RecoveryAttemptContext(
                    attempt_id=reservation.recovery_attempt_id,
                    coverage_before=reservation.recovery_coverage_before,
                    gap_before=reservation.recovery_gap_before,
                    accepted_evidence_before=(
                        reservation.recovery_accepted_evidence_before
                    ),
                    tokens_reserved=reservation.reserved_total,
                )
                recovery_attempts.append(recovery_context)
                active_recovery_context = recovery_context
                if recovery_attempt_id is None:
                    recovery_attempt_id = recovery_context.attempt_id
                    recovery_coverage_before = recovery_context.coverage_before
                    recovery_gap_before = recovery_context.gap_before
                    recovery_accepted_evidence_before = (
                        recovery_context.accepted_evidence_before
                    )
                    recovery_tokens_reserved = recovery_context.tokens_reserved
                recovery_recorder = getattr(self._repository, "record_recovery_event", None)
                if callable(recovery_recorder):
                    await recovery_recorder(
                        run_id,
                        worker_task_id=worker_task_id,
                        event_type="recovery.started",
                        question_id=target.question_id,
                        plan_version=target.plan_version,
                        attempt_id=recovery_context.attempt_id,
                        coverage_before=recovery_context.coverage_before,
                        gap_before=recovery_context.gap_before,
                        accepted_evidence_before=recovery_context.accepted_evidence_before,
                        tokens_reserved=recovery_context.tokens_reserved,
                        reason="borrow_allowed",
                    )
                context_attacher = getattr(
                    self._repository,
                    "attach_recovery_context",
                    None,
                )
                if callable(context_attacher) and iteration_start_seq is not None:
                    recovery_context_model = RecoveryContext.create(
                        run_id=run_id,
                        question_id=target.question_id,
                        plan_version=target.plan_version,
                        recovery_attempt_id=recovery_context.attempt_id,
                        trigger_reason="borrow_allowed",
                        coverage_before=recovery_context.coverage_before,
                        gap_before=recovery_context.gap_before,
                    )
                    await context_attacher(
                        run_id,
                        worker_task_id=worker_task_id,
                        context=recovery_context_model,
                        from_run_seq=iteration_start_seq,
                    )

            source_id = reusable_ref.source_id if reusable_ref is not None else uuid7()
            try:
                await record_selection_event(
                    result.url,
                    "selection_selected",
                    source_id=source_id,
                )
                if reusable_ref is not None:
                    artifact_uri = reusable_ref.artifact_uri
                else:
                    artifact_uri = await self._artifacts.save_page(
                        run_id, source_id, page.clean_text
                    )
                await self._repository.record_extraction_started(
                    run_id,
                    worker_task_id=worker_task_id,
                    target=target,
                    source_id=source_id,
                )
            except Exception:
                await cleanup_attempt(attempt_id, uncertain=False)
                await record_recovery_failure("extraction_setup_failed")
                raise
            try:
                remaining = _deadline_remaining_seconds(target)
                if remaining <= 0:
                    raise TimeoutError
                evidence, usage, manifest = await asyncio.wait_for(
                    self._extractor.extract(
                        run_id,
                        question=target.question,
                        acceptance_dimensions=target.acceptance_dimensions,
                        page=page,
                        source_id=source_id,
                        max_total_tokens=model_budget.max_call_tokens,
                    ),
                    timeout=remaining,
                )
            except TimeoutError:
                await self._repository.record_extraction_failure(
                    run_id,
                    worker_task_id=worker_task_id,
                    target=target,
                    source_id=source_id,
                    page=page,
                    artifact_uri=artifact_uri,
                    error_code="DEADLINE_EXHAUSTED",
                    attempt_id=attempt_id,
                )
                await self._repository.release_page_slots(
                    run_id,
                    worker_task_id=worker_task_id,
                    count=max(0, page_reservation.granted - settled_slots),
                )
                return attempt_result("deadline_exhausted")
            except ModelGatewayError as exc:
                if exc.code not in _ISOLATED_EXTRACTION_ERRORS:
                    await record_recovery_failure(exc.code)
                    raise
                try:
                    await self._repository.record_extraction_failure(
                        run_id,
                        worker_task_id=worker_task_id,
                        target=target,
                        source_id=source_id,
                        page=page,
                        artifact_uri=artifact_uri,
                        error_code=exc.code,
                        detail_code=exc.detail_code,
                        usage=exc.usage,
                        attempt_id=attempt_id,
                    )
                except Exception:
                    await cleanup_attempt(attempt_id, uncertain=True)
                    raise
                if exc.code in _DURABLE_EXTRACTION_TRANSPORT_ERRORS:
                    # The fetched Snapshot is already persisted by the failure
                    # recorder. Settle this page once, release unused slots, and
                    # let the Worker schedule a durable model retry. Resume can
                    # reuse the Artifact without another webpage request.
                    await self._repository.release_page_slots(
                        run_id,
                        worker_task_id=worker_task_id,
                        count=max(0, page_reservation.granted - settled_slots),
                    )
                    await record_recovery_failure(exc.code)
                    raise
                if exc.code in _STRUCTURED_EXTRACTION_ERRORS:
                    structured_extraction_failures += 1
                    if (
                        structured_extraction_failures >= _STRUCTURED_EXTRACTION_FAILURE_LIMIT
                        and accepted == 0
                    ):
                        await self._repository.release_page_slots(
                            run_id,
                            worker_task_id=worker_task_id,
                            count=max(0, page_reservation.granted - settled_slots),
                        )
                        await record_recovery_failure("MODEL_CAPABILITY_INSUFFICIENT")
                        raise ModelGatewayError(
                            "MODEL_CAPABILITY_INSUFFICIENT",
                            retryable=False,
                            detail_code=("EVIDENCE_EXTRACTOR_CIRCUIT_OPEN_AFTER_2_SOURCES"),
                            usage=exc.usage,
                        ) from exc
                else:
                    structured_extraction_failures = 0
                continue
            except Exception:
                # The provider outcome is unknown for an unexpected extractor
                # failure; retain the conservative reservation and release
                # only unused page slots before propagating the failure.
                await cleanup_attempt(attempt_id, uncertain=True)
                await record_recovery_failure("extraction_failed")
                raise

            try:
                page_candidate_count, page_accepted = await self._repository.record_page(
                    run_id,
                    worker_task_id=worker_task_id,
                    target=target,
                    source_id=source_id,
                    page=page,
                    artifact_uri=artifact_uri,
                    evidence=evidence,
                    usage=usage,
                    context_manifest=manifest,
                    attempt_id=attempt_id,
                )
            except Exception:
                await cleanup_attempt(attempt_id, uncertain=True)
                await record_recovery_failure("evidence_persistence_failed")
                raise
            accepted += page_accepted
            structured_extraction_failures = 0
            if page_candidate_count == 0:
                # A readable page that yields no candidate is a failed query
                # variant, not a reason to spend its second page slot. Return
                # to the evaluator so the next attempt can rotate the query.
                break
            if page_accepted == 0:
                zero_yield_pages += 1
            else:
                zero_yield_pages = 0
            # A page with no accepted evidence is a strong signal that the
            # query/result set is off-topic. Allow one corroborating attempt,
            # then rotate; readable or transiently failing candidates can
            # cannot consume a third page; the next query variant must get a
            # chance instead of inheriting a weak result set's budget.
            if zero_yield_pages >= 2:
                break

        used_owner_keys = set(target.used_source_owner_keys)
        for result in candidate_results:
            normalized_url = normalize_source_url(result.url)
            if normalized_url in dispatch_selected_urls or normalized_url in dispatch_skipped_urls:
                continue
            reason = (
                "source_diversity"
                if source_owner_key(result.url) in used_owner_keys
                else "relevance_rank"
            )
            await record_dispatch_event(
                result.url,
                "dispatch_skipped",
                reason=reason,
            )

        await self._repository.release_page_slots(
            run_id,
            worker_task_id=worker_task_id,
            count=max(0, page_reservation.granted - settled_slots),
        )

        if question_budget_exhausted:
            outcome = "question_budget_exhausted"
        elif question_budget_yielded:
            outcome = "yield_question"
        elif not page_results:
            outcome = "zero_results"
        elif pages_read == 0:
            outcome = "unreadable"
        elif accepted == 0:
            outcome = "no_evidence"
        else:
            outcome = "evidence_gained"
        return ResearchAttemptResult(
            pages_read,
            accepted,
            outcome,
            recovery_attempt_id=recovery_attempt_id,
            recovery_coverage_before=recovery_coverage_before,
            recovery_gap_before=recovery_gap_before,
            recovery_accepted_evidence_before=recovery_accepted_evidence_before,
            recovery_tokens_reserved=recovery_tokens_reserved,
            recovery_attempts=tuple(recovery_attempts),
        )

    @staticmethod
    def _outcome(decision: str, iterations: int, pages: int, accepted: int) -> str:
        return (
            f"research_stopped:{decision}:iterations={iterations}:pages={pages}:accepted={accepted}"
        )


def _deadline_expired(target: ResearchTarget) -> bool:
    deadline = target.deadline_at
    if deadline is None:
        return False
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    return datetime.now(UTC) >= deadline


def _deadline_remaining_seconds(target: ResearchTarget) -> float:
    deadline = target.deadline_at
    if deadline is None:
        return 3_600.0
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    return max(0.0, (deadline - datetime.now(UTC)).total_seconds())


def _prioritize_search_results(
    results: list[SearchResult],
    *,
    query: str = "",
    alternate_query: str = "",
    acceptance_criteria: tuple[str, ...] = (),
    used_owner_keys: set[str] | None = None,
    owner_acceptance_rates: dict[str, float] | None = None,
) -> list[SearchResult]:
    """Rank readable, relevant and owner-diverse public sources first."""

    used_owners = used_owner_keys or set()
    acceptance_rates = owner_acceptance_rates or {}
    query_token_sets = [
        tokens for value in (query, alternate_query) if (tokens := _search_tokens(value))
    ]
    query_tokens = set().union(*query_token_sets) if query_token_sets else set()

    def sort_key(result: SearchResult) -> tuple[float, int]:
        parsed = urlsplit(result.url)
        hostname = (parsed.hostname or "").lower()
        path = parsed.path.lower().rstrip("/")
        penalty = 0.0
        candidate_tokens = _search_tokens(f"{result.title} {result.snippet}")
        title_tokens = _search_tokens(result.title)
        overlap = max(
            (len(tokens & candidate_tokens) / max(len(tokens), 1) for tokens in query_token_sets),
            default=0.0,
        )
        title_overlap = max(
            (len(tokens & title_tokens) for tokens in query_token_sets),
            default=0,
        )
        if _matches_domain(hostname, _PREFERRED_READABLE_DOMAINS):
            # Academic/open-access status is valuable only after topical
            # relevance. A weakly related arXiv hit must not displace a directly
            # relevant standards, market or manufacturer source.
            penalty += -20 if not query_tokens or title_overlap >= 2 else 10
        if _matches_domain(hostname, _RESTRICTED_SOURCE_DOMAINS):
            # These hosts are credible but routinely return login shells,
            # bot challenges, or empty abstracts to the public reader. Keep
            # them as a last resort after directly readable sources.
            penalty += 70
        if path.endswith(".pdf") or "/pdf" in path:
            # The public reader now parses bounded PDFs directly.  Treat an
            # explicit PDF as a readability signal instead of carrying over
            # the old unsupported-format penalty; otherwise bounded page
            # slots are spent on paywalls while open papers never get read.
            penalty -= 18
        if source_owner_key(result.url) in used_owners:
            # A corroboration pass must strongly prefer a genuinely independent
            # publisher. Retain same-owner pages only as a last resort.
            penalty += 60
        # Historical owner yield is only a bounded tie-breaker. It cannot
        # override topical relevance or the owner-diversity penalty.
        penalty -= max(0.0, min(1.0, acceptance_rates.get(source_owner_key(result.url), 0.0))) * 10
        # Prefer authoritative sources when relevance is otherwise comparable.
        # The quality gate uses the same deterministic reliability policy.
        # Topical fit dominates publisher class.  Previously the authority
        # bonus was larger than the entire relevance bonus, so an unrelated
        # arXiv page displaced a directly relevant result.
        # When the source-quality gate is open, a credible host must have a
        # meaningful chance to displace an equally topical unknown host. The
        # previous factor was small enough that a low-quality aggregator at a
        # slightly better search rank routinely won every bounded page slot.
        # Relevance still dominates (the overlap term is much larger), while
        # recognised academic/government/primary hosts now get a real quality
        # preference.
        penalty -= (source_reliability(result.url) - 0.68) * 45
        if acceptance_criteria:
            claim_type = infer_claim_type(" ".join(acceptance_criteria))
            source_role = classify_source_role(
                result.url,
                text=f"{result.title}\n{result.snippet}",
            )
            if not source_role_fits_claim(claim_type=claim_type, source_role=source_role):
                penalty += 28
            elif claim_type == "vendor_product" and source_role == "manufacturer":
                # Vendor/product questions need first-party attribution. A
                # topical news roundup is useful for discovery but must not
                # consume the only first-pass P2 page ahead of the vendor's
                # own product or application page.
                penalty -= 30
        if "官网" in result.title or "official" in result.title.casefold():
            penalty -= 12
        penalty -= overlap * 80
        penalty -= title_overlap * 4
        return penalty, result.rank

    # Do not let a generic lexical overlap reintroduce pages rejected by the
    # provider's topical gate (for example remote-sensing or pedestrian
    # tracking pages for an industrial-defect question).
    filtered = [
        result
        for result in results
        if _topic_relevance_ok(query, result)
        or bool(alternate_query and _topic_relevance_ok(alternate_query, result))
    ]
    ordered = sorted(filtered, key=sort_key)
    # Keep the first extraction batch owner-diverse. Repeated pages from one
    # content host are retained as a last resort, but cannot crowd primary
    # sources from other owners out of the bounded attempt window.
    diverse: list[SearchResult] = []
    overflow: list[SearchResult] = []
    owner_counts: dict[str, int] = {}
    for result in ordered:
        owner = source_owner_key(result.url)
        owner_counts[owner] = owner_counts.get(owner, 0) + 1
        (diverse if owner_counts[owner] <= 2 else overflow).append(result)
    return [*diverse, *overflow]


def _matches_domain(hostname: str, domains: set[str]) -> bool:
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in domains)


def _topic_relevance_ok(query: str, result: SearchResult) -> bool:
    """Require a domain anchor when a query contains one.

    Method-only overlap is too permissive for Chinese technical searches: a
    paper about SAR, optical flow, or remote sensing can contain all the same
    generic algorithm words while answering a different question.
    """

    query_tokens = _search_tokens(query)
    required_anchors = query_tokens & _TOPIC_DOMAIN_ANCHORS
    if not required_anchors:
        return True
    candidate_tokens = _search_tokens(f"{result.title} {result.snippet}")
    # Reusable/test candidates can be title-only and do not carry enough text
    # for a reliable topical decision. Leave those to the extractor; the
    # provider-side filter handles full search responses with real snippets.
    if len(candidate_tokens) < 4:
        return True
    query_has_industrial_context = bool(query_tokens & _INDUSTRIAL_CONTEXT_ANCHORS)
    query_has_defect_or_inspection = bool(query_tokens & _DEFECT_INSPECTION_ANCHORS)
    query_requires_visual_topic = bool(query_tokens & _VISUAL_TOPIC_ANCHORS)
    candidate_text = f"{result.title} {result.snippet}".casefold()
    candidate_has_visual_topic = bool(candidate_tokens & _VISUAL_TOPIC_ANCHORS)
    candidate_is_non_visual_anomaly = any(
        phrase in candidate_text for phrase in _NON_VISUAL_ANOMALY_ANCHORS
    )
    # Required and excluded concepts are derived from the current query: only
    # visually scoped questions activate this guard. This rejects an industrial
    # control/network anomaly paper while preserving it for a control-system question.
    if query_requires_visual_topic and (
        not candidate_has_visual_topic or candidate_is_non_visual_anomaly
    ):
        return False
    if query_has_industrial_context and query_has_defect_or_inspection:
        # For industrial-defect questions, generic method overlap such as
        # "vision", "algorithm", or "deep learning" is not sufficient.
        # Both the industrial context and the inspection/defect task must be
        # visible in the candidate title/snippet.
        return bool(candidate_tokens & _INDUSTRIAL_CONTEXT_ANCHORS) and bool(
            candidate_tokens & _DEFECT_INSPECTION_ANCHORS
        )
    return bool(required_anchors & candidate_tokens)


def _is_read_candidate(url: str) -> bool:
    """Reject URLs that cannot yield stable, directly attributable page text."""

    return is_stable_read_url(url)


def _search_tokens(value: str) -> set[str]:
    normalized = "".join(char.casefold() if char.isalnum() else " " for char in value)
    words = {word for word in normalized.split() if len(word) > 1}
    cjk = {
        normalized[index : index + 2]
        for index in range(max(0, len(normalized) - 1))
        if all("\u4e00" <= char <= "\u9fff" for char in normalized[index : index + 2])
    }
    return (words | cjk) - _GENERIC_QUERY_TOKENS


def _deduplicate_search_results(results: list[SearchResult]) -> list[SearchResult]:
    """Keep one candidate per canonical URL while preserving provider order."""

    unique: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        normalized = normalize_source_url(result.url)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(result)
    return unique
