"""Application service for empty Research Run lifecycle and event replay."""

from __future__ import annotations

from typing import Protocol
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
from app.domain.reports import ReportCitationView, ReportView
from app.domain.research_runs import AgentEventView, ResearchRunView
from app.domain.research_tools import EvidenceView
from app.domain.state import ResearchState
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
        "max_iterations": 5,
        "max_searches": 5,
        "max_pages": 10,
        "max_tokens": 30_000,
    },
    "standard": {
        "max_iterations": 15,
        "max_searches": 15,
        "max_pages": 30,
        "max_tokens": 100_000,
    },
    "deep": {
        "max_iterations": 30,
        "max_searches": 30,
        "max_pages": 60,
        "max_tokens": 220_000,
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

    async def list_evaluations(
        self, owner_hash: str, run_id: UUID
    ) -> list[EvaluationSnapshot]: ...

    async def get_report(self, owner_hash: str, run_id: UUID) -> ReportView: ...

    async def get_report_citation(
        self, owner_hash: str, report_id: UUID, citation_number: int
    ) -> ReportCitationView: ...

    async def cancel_run(self, owner_hash: str, run_id: UUID) -> ResearchRunView: ...

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
    ) -> None:
        self._repository = repository
        self._research_repository = research_repository
        self._report_repository = report_repository
        self._state_repository = state_repository
        self._context_manager = context_manager
        self._memory_manager = memory_manager
        self._controlled_tools = controlled_tools

    async def create_run(
        self,
        owner_hash: str,
        *,
        idempotency_key: str,
        query: str,
        saved_profile_version_id: UUID,
        budget_tier: str,
    ) -> tuple[ResearchRunView, bool]:
        normalized = " ".join(query.split())
        if not normalized:
            raise ValueError("research query may not be empty")
        if budget_tier not in _BUDGETS:
            raise ValueError("unknown research budget tier")
        key = idempotency_key.strip()
        if not key or len(key) > 200:
            raise ValueError("Idempotency-Key must contain 1 to 200 characters")
        return await self._repository.create(
            owner_hash,
            idempotency_key=key,
            original_query=query,
            normalized_goal=normalized,
            credential_version_id=saved_profile_version_id,
            budget_snapshot={"tier": budget_tier, **_BUDGETS[budget_tier]},
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

    async def list_memory_accesses(
        self, owner_hash: str, run_id: UUID
    ) -> list[MemoryAccessView]:
        await self._repository.get(owner_hash, run_id)
        return await self._memory_manager.list_accesses(owner_hash, run_id)

    async def list_evaluations(
        self, owner_hash: str, run_id: UUID
    ) -> list[EvaluationSnapshot]:
        return await self._research_repository.list_evaluations(owner_hash, run_id)

    async def get_report(self, owner_hash: str, run_id: UUID) -> ReportView:
        await self._repository.get(owner_hash, run_id)
        return await self._report_repository.get_for_run(owner_hash, run_id)

    async def get_report_citation(
        self, owner_hash: str, report_id: UUID, citation_number: int
    ) -> ReportCitationView:
        return await self._report_repository.get_citation(owner_hash, report_id, citation_number)

    async def cancel_run(self, owner_hash: str, run_id: UUID) -> ResearchRunView:
        run = await self._repository.cancel(owner_hash, run_id)
        await self._synchronize_lifecycle_state(run_id, node_name="cancel_boundary")
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
