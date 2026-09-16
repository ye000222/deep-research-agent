"""Application service for empty Research Run lifecycle and event replay."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Protocol, cast
from uuid import UUID

from app.context.manager import ContextBudgetManager
from app.domain.context import ContextManifestView
from app.domain.controlled_tools import (
    AnalyzeDataInput,
    AnalyzeDataResult,
    EvidenceSearchInput,
    EvidenceSearchResult,
)
from app.domain.evaluation import EvaluationSnapshot
from app.domain.evidence_graph import EvidenceGraphView
from app.domain.memory import MemoryAccessView, MemoryItemView
from app.domain.planning import ResearchPlan
from app.domain.reports import ReportCitationView, ReportView, VerificationView
from app.domain.research_management import allocate_budget_shares
from app.domain.research_runs import AgentEventView, ResearchRunView
from app.domain.research_tools import EvidenceView
from app.domain.state import ResearchState
from app.infrastructure.db.llm_calls import LLMCallRepository
from app.infrastructure.db.reports import ReportRepository
from app.infrastructure.db.research_runs import ResearchRunRepository
from app.infrastructure.db.research_tools import ResearchToolRepository
from app.infrastructure.db.state_runtime import (
    ResearchStateRuntimeRepository,
    StateSnapshotNotFoundError,
)
from app.memory.manager import ResearchMemoryManager
from app.tools.gateway import ControlledToolGateway

_BUDGETS: dict[str, dict[str, object]] = {
    "quick": {
        "max_iterations": 8,
        "max_searches": 6,
        "max_logical_queries": 6,
        "max_provider_requests": 18,
        "max_pages": 5,
        "max_pages_fetched": 10,
        "max_page_fetch_attempts": 16,
        "max_pages_extracted": 5,
        "max_extraction_calls": 5,
        "max_verification_calls": 2,
        "max_scheduler_actions": 24,
        "max_tokens": 30_000,
        "max_wall_clock_seconds": 180,
    },
    "standard": {
        "max_iterations": 64,
        # Eight planned questions need one first pass plus a corroboration /
        # repair tail.  The earlier 28-query / 84-fetch envelope was reached
        # mostly by failed or blocked pages before the final P1 repair pass.
        # Keep the ceilings aligned so transport noise cannot consume the
        # entire research window before useful evidence is extracted.
        "max_searches": 56,
        "max_logical_queries": 56,
        "max_provider_requests": 168,
        "max_pages": 40,
        # Blocked publisher pages must not end the run while logical query,
        # extraction and token pools still have useful capacity.
        "max_pages_fetched": 120,
        # Failed/blocked reads are tracked separately so they remain bounded
        # without consuming the successful-page evidence budget.
        "max_page_fetch_attempts": 160,
        # A real V1 acceptance run needed 24 successful extractions to reach
        # only 58% coverage while query, fetch and provider capacity remained.
        # Keep enough extraction headroom for the uncovered-question repair
        # pass instead of forcing an early transition to report writing.
        "max_pages_extracted": 56,
        "max_extraction_calls": 56,
        "max_verification_calls": 6,
        "max_scheduler_actions": 168,
        # With the reserved planner/writer/verification/safety pools, a long
        # V1 repair tail can spend roughly 100k research tokens before the
        # final independent-source checks. Keep a generous pool so token
        # protection cannot terminate with only one or two acceptance gaps
        # remaining; downstream writer/verification/safety reserves stay
        # explicit and protected.
        "max_tokens": 220_000,
        # A standard V1 run needs more than one 15-minute worker lease when
        # page reads hit slow/blocked publishers.  The worker heartbeat keeps
        # the lease alive while this wall-clock deadline protects runaway
        # runs, leaving time for repair, verification and report generation.
        # Arbitrary market/technical questions can require the full rotated
        # query tail.  A 30-minute wall clock truncated the run while more
        # than a third of the logical-query/page/extraction budget remained.
        "max_wall_clock_seconds": 3_600,
    },
    "deep": {
        "max_iterations": 72,
        "max_searches": 64,
        "max_logical_queries": 64,
        "max_provider_requests": 192,
        "max_pages": 64,
        "max_pages_fetched": 160,
        "max_page_fetch_attempts": 224,
        "max_pages_extracted": 64,
        "max_extraction_calls": 64,
        "max_verification_calls": 10,
        "max_scheduler_actions": 192,
        "max_tokens": 260_000,
        "max_wall_clock_seconds": 4_800,
    },
}


class ResearchRunServiceProtocol(Protocol):
    async def create_run(
        self,
        owner_hash: str,
        *,
        idempotency_key: str,
        query: str,
        saved_profile_version_id: UUID,
        budget_tier: str,
        plan_template_run_id: UUID | None = None,
    ) -> tuple[ResearchRunView, bool]: ...

    async def list_runs(self, owner_hash: str, *, limit: int) -> list[ResearchRunView]: ...

    async def get_run(self, owner_hash: str, run_id: UUID) -> ResearchRunView: ...

    async def get_plan(self, owner_hash: str, run_id: UUID) -> ResearchPlan: ...

    async def get_state(self, owner_hash: str, run_id: UUID) -> ResearchState: ...

    async def list_evidence(self, owner_hash: str, run_id: UUID) -> list[EvidenceView]: ...

    async def get_evidence_graph(
        self,
        owner_hash: str,
        run_id: UUID,
    ) -> EvidenceGraphView: ...

    async def list_context_metrics(
        self, owner_hash: str, run_id: UUID
    ) -> list[ContextManifestView]: ...

    async def list_llm_calls(self, owner_hash: str, run_id: UUID) -> list[dict[str, object]]: ...

    async def list_memory(self, owner_hash: str, run_id: UUID) -> list[MemoryItemView]: ...

    async def search_evidence(
        self, owner_hash: str, run_id: UUID, request: EvidenceSearchInput
    ) -> EvidenceSearchResult: ...

    async def analyze_data(
        self, owner_hash: str, run_id: UUID, request: AnalyzeDataInput
    ) -> AnalyzeDataResult: ...

    async def list_memory_accesses(
        self, owner_hash: str, run_id: UUID
    ) -> list[MemoryAccessView]: ...

    async def list_evaluations(self, owner_hash: str, run_id: UUID) -> list[EvaluationSnapshot]: ...

    async def get_report(self, owner_hash: str, run_id: UUID) -> ReportView: ...

    async def get_verification(self, owner_hash: str, run_id: UUID) -> VerificationView: ...

    async def get_report_citation(
        self, owner_hash: str, report_id: UUID, citation_number: int
    ) -> ReportCitationView: ...

    async def cancel_run(self, owner_hash: str, run_id: UUID) -> ResearchRunView: ...

    async def pause_run(self, owner_hash: str, run_id: UUID) -> ResearchRunView: ...

    async def resume_run(self, owner_hash: str, run_id: UUID) -> ResearchRunView: ...

    async def list_events(
        self, owner_hash: str, run_id: UUID, *, after_seq: int
    ) -> list[AgentEventView]: ...


class ResearchRunService:
    def __init__(
        self,
        repository: ResearchRunRepository,
        research_repository: ResearchToolRepository,
        report_repository: ReportRepository,
        state_repository: ResearchStateRuntimeRepository,
        context_manager: ContextBudgetManager,
        memory_manager: ResearchMemoryManager,
        controlled_tools: ControlledToolGateway,
        llm_call_repository: LLMCallRepository,
        source_revision: str = "development",
    ) -> None:
        self._repository = repository
        self._research_repository = research_repository
        self._report_repository = report_repository
        self._state_repository = state_repository
        self._context_manager = context_manager
        self._memory_manager = memory_manager
        self._controlled_tools = controlled_tools
        self._llm_call_repository = llm_call_repository
        self._source_revision = source_revision.strip() or "development"

    async def create_run(
        self,
        owner_hash: str,
        *,
        idempotency_key: str,
        query: str,
        saved_profile_version_id: UUID,
        budget_tier: str,
        plan_template_run_id: UUID | None = None,
    ) -> tuple[ResearchRunView, bool]:
        normalized = " ".join(query.split())
        if not normalized:
            raise ValueError("research query may not be empty")
        if budget_tier not in _BUDGETS:
            raise ValueError("unknown research budget tier")
        key = idempotency_key.strip()
        if not key or len(key) > 200:
            raise ValueError("Idempotency-Key must contain 1 to 200 characters")
        budget = dict(_BUDGETS[budget_tier])
        budget["source_revision"] = self._source_revision
        budget["deadline_at"] = (
            datetime.now(UTC)
            + timedelta(seconds=cast(int, budget["max_wall_clock_seconds"]))
        ).isoformat()
        budget["performance_policy_version"] = "fair_first_pass.v1"
        budget["relevant_chunks_enabled"] = True
        budget["extraction_cache_enabled"] = True
        budget["report_draft_cache_enabled"] = True
        budget["adaptive_scheduler_enabled"] = True
        budget["cheap_triage_enabled"] = True
        budget["cross_question_search_cache_enabled"] = True
        if plan_template_run_id is not None:
            # Acceptance runs may pin the already-audited baseline plan.  This
            # keeps repeated ordinals comparable; ordinary API runs remain
            # model-planned when the option is omitted.
            budget["plan_template_run_id"] = str(plan_template_run_id)
            budget["plan_template_plan_version"] = 1
        budget["max_evidence_call_tokens"] = 12_000
        budget["allocation"] = allocate_budget_shares(
            max_iterations=cast(int, budget["max_iterations"]),
            max_searches=cast(int, budget["max_searches"]),
            max_pages=cast(int, budget["max_pages"]),
            max_tokens=cast(int, budget["max_tokens"]),
            max_logical_queries=cast(int, budget["max_logical_queries"]),
            max_provider_requests=cast(int, budget["max_provider_requests"]),
            max_pages_fetched=cast(int, budget["max_pages_fetched"]),
            max_pages_extracted=cast(int, budget["max_pages_extracted"]),
            max_extraction_calls=cast(int, budget["max_extraction_calls"]),
            max_verification_calls=cast(int, budget["max_verification_calls"]),
            max_scheduler_actions=cast(int, budget["max_scheduler_actions"]),
        )
        budget["budget_pool_schema"] = {
            "version": "resource_pools.v3",
            "contract": (
                "Every pool maps to an enforced hard-limit key, a live usage key, "
                "and optional in-flight reservations. Legacy shares live only under "
                "allocation.derived_display."
            ),
            "model_token_pools": {
                "planner": "allocation.planner_tokens",
                "research": "allocation.research_tokens",
                "verification": "allocation.verification_tokens",
                "writer": "allocation.writer_tokens_initial",
                "safety": "allocation.safety_tokens",
            },
            "hard_resource_limits": {
                "logical_queries": "max_logical_queries",
                "provider_requests": "max_provider_requests",
                "pages_fetched": "max_pages_fetched",
                "pages_extracted": "max_pages_extracted",
                "extraction_calls": "max_extraction_calls",
                "verification_calls": "max_verification_calls",
                "scheduler_actions": "max_scheduler_actions",
                "productive_iterations": "max_iterations",
                "wall_clock": "deadline_at",
            },
        }
        return await self._repository.create(
            owner_hash,
            idempotency_key=key,
            original_query=query,
            normalized_goal=normalized,
            credential_version_id=saved_profile_version_id,
            budget_snapshot={"tier": budget_tier, **budget},
        )

    async def list_runs(self, owner_hash: str, *, limit: int) -> list[ResearchRunView]:
        return await self._repository.list_recent(owner_hash, limit=limit)

    async def get_run(self, owner_hash: str, run_id: UUID) -> ResearchRunView:
        return await self._repository.get(owner_hash, run_id)

    async def get_plan(self, owner_hash: str, run_id: UUID) -> ResearchPlan:
        return await self._repository.get_plan(owner_hash, run_id)

    async def get_state(self, owner_hash: str, run_id: UUID) -> ResearchState:
        await self._repository.get(owner_hash, run_id)
        return await self._state_repository.get(run_id)

    async def list_evidence(self, owner_hash: str, run_id: UUID) -> list[EvidenceView]:
        return await self._research_repository.list_evidence(owner_hash, run_id)

    async def get_evidence_graph(
        self,
        owner_hash: str,
        run_id: UUID,
    ) -> EvidenceGraphView:
        return await self._research_repository.get_evidence_graph(owner_hash, run_id)

    async def list_context_metrics(
        self, owner_hash: str, run_id: UUID
    ) -> list[ContextManifestView]:
        await self._repository.get(owner_hash, run_id)
        return await self._context_manager.list_metrics(run_id)

    async def list_llm_calls(self, owner_hash: str, run_id: UUID) -> list[dict[str, object]]:
        return await self._llm_call_repository.list_for_run(owner_hash, run_id)

    async def search_evidence(
        self, owner_hash: str, run_id: UUID, request: EvidenceSearchInput
    ) -> EvidenceSearchResult:
        await self._repository.get(owner_hash, run_id)
        if request.run_id != run_id:
            raise ValueError("tool run_id does not match path")
        return await self._controlled_tools.search_evidence(request)

    async def analyze_data(
        self, owner_hash: str, run_id: UUID, request: AnalyzeDataInput
    ) -> AnalyzeDataResult:
        await self._repository.get(owner_hash, run_id)
        if request.run_id != run_id:
            raise ValueError("tool run_id does not match path")
        return await self._controlled_tools.analyze_data(request)

    async def list_memory(self, owner_hash: str, run_id: UUID) -> list[MemoryItemView]:
        await self._repository.get(owner_hash, run_id)
        return await self._memory_manager.list_items(owner_hash, run_id)

    async def list_memory_accesses(self, owner_hash: str, run_id: UUID) -> list[MemoryAccessView]:
        await self._repository.get(owner_hash, run_id)
        return await self._memory_manager.list_accesses(owner_hash, run_id)

    async def list_evaluations(self, owner_hash: str, run_id: UUID) -> list[EvaluationSnapshot]:
        return await self._research_repository.list_evaluations(owner_hash, run_id)

    async def get_report(self, owner_hash: str, run_id: UUID) -> ReportView:
        await self._repository.get(owner_hash, run_id)
        return await self._report_repository.get_for_run(owner_hash, run_id)

    async def get_verification(self, owner_hash: str, run_id: UUID) -> VerificationView:
        """Return the persisted report gate and its latest report-scoped evaluation."""

        report = await self.get_report(owner_hash, run_id)
        evaluations = await self._research_repository.list_evaluations(owner_hash, run_id)
        report_evaluations = [item for item in evaluations if item.scope.value == "report"]
        latest = report_evaluations[-1] if report_evaluations else None
        return VerificationView(
            run_id=run_id,
            report_id=report.report_id,
            report_status=report.status,
            verified=bool(report.verification_result.get("verified", False)),
            citation_count=len(report.citations),
            analysis_artifact_citation_count=sum(
                citation.analysis_artifact_id is not None for citation in report.citations
            ),
            verification_result=report.verification_result,
            latest_report_evaluation=latest,
        )

    async def get_report_citation(
        self, owner_hash: str, report_id: UUID, citation_number: int
    ) -> ReportCitationView:
        return await self._report_repository.get_citation(owner_hash, report_id, citation_number)

    async def cancel_run(self, owner_hash: str, run_id: UUID) -> ResearchRunView:
        run = await self._repository.cancel(owner_hash, run_id)
        await self._synchronize_lifecycle_state(run_id, node_name="cancel_boundary")
        return run

    async def pause_run(self, owner_hash: str, run_id: UUID) -> ResearchRunView:
        run = await self._repository.pause(owner_hash, run_id)
        await self._synchronize_lifecycle_state(run_id, node_name="pause_boundary")
        return run

    async def resume_run(self, owner_hash: str, run_id: UUID) -> ResearchRunView:
        run = await self._repository.resume(owner_hash, run_id)
        await self._synchronize_lifecycle_state(run_id, node_name="resume_boundary")
        return run

    async def _synchronize_lifecycle_state(self, run_id: UUID, *, node_name: str) -> None:
        """Keep an existing public state snapshot aligned with lifecycle writes.

        A newly queued run may not have reached the Worker yet, so the absence of
        a state snapshot is expected and must not make cancel/resume fail.
        """

        try:
            await self._state_repository.synchronize(
                run_id,
                node_name=node_name,
                worker_task_id=None,
            )
        except StateSnapshotNotFoundError:
            return

    async def list_events(
        self, owner_hash: str, run_id: UUID, *, after_seq: int
    ) -> list[AgentEventView]:
        return await self._repository.list_events(
            owner_hash,
            run_id,
            after_seq=after_seq,
        )
