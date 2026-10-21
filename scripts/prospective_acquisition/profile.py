"""Fixed-depth development-only Claim acquisition orchestration.

The profile intentionally does not invoke ResearchGraph, report writing,
coverage/gap closure, recovery, or replanning. A component factory must wire
the real EvidenceExtractor to the supplied guarded Provider client.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import httpx

if TYPE_CHECKING:
    from app.infrastructure.db.research_tools import ResearchTarget

from scripts.prospective_acquisition.control import (
    CallBudget,
    ClaimRecord,
    TaskResult,
    build_guarded_http_client,
    build_guarded_provider_client,
)
from scripts.prospective_acquisition.execution_identity import (
    ActiveProfileRepository,
    FrozenProviderBinding,
    ProspectiveAcquisitionExecutionContext,
    ProviderProfileLister,
    assert_provider_binding_unchanged,
    validate_frozen_context,
)


@dataclass(frozen=True)
class AcquisitionTask:
    task_id: str
    run_id: str
    task_text: str
    execution_context_id: str
    research_target: ResearchTarget | None = None
    worker_task_id: str | None = None
    artifact_uri_prefix: str | None = None


@dataclass(frozen=True)
class AcquisitionSource:
    url: str
    title: str


class SearchComponent(Protocol):
    async def search(self, query: str, *, limit: int) -> Sequence[AcquisitionSource]: ...


class ReaderComponent(Protocol):
    async def read(self, url: str) -> object: ...


class ExtractorComponent(Protocol):
    async def extract(self, query: str, page: object) -> object: ...


class ClaimFormationComponent(Protocol):
    async def record_page(
        self,
        task: AcquisitionTask,
        page: object,
        evidence: object,
    ) -> Sequence[ClaimRecord]: ...


@dataclass(frozen=True)
class AcquisitionComponents:
    search: SearchComponent
    reader: ReaderComponent
    extractor: ExtractorComponent
    claim_formation: ClaimFormationComponent


@dataclass(frozen=True)
class AcquisitionHTTPClients:
    provider: httpx.AsyncClient
    search: httpx.AsyncClient
    reader: httpx.AsyncClient


ComponentFactory = Callable[
    [
        AcquisitionTask,
        ProspectiveAcquisitionExecutionContext,
        FrozenProviderBinding,
        AcquisitionHTTPClients,
    ],
    AcquisitionComponents,
]


@dataclass(frozen=True)
class BoundedTaskExecution:
    task_id: str
    run_id: str
    status: str
    result: TaskResult
    provider_outbound_count: int
    search_outbound_count: int
    reader_outbound_count: int
    blocked_extra_outbound_count: int
    failure_stage: str | None = None


@dataclass(frozen=True)
class BoundedProfileResult:
    executions: tuple[BoundedTaskExecution, ...]
    batch_provider_outbound_count: int
    batch_search_outbound_count: int
    batch_reader_outbound_count: int
    batch_budget_remaining: int


class BoundedProspectiveAcquisitionProfile:
    """One search result, one read, one extraction entry, then stop per task."""

    profile_version = "prospective-acquisition-profile.v1"
    search_rounds_per_task = 1
    search_results_per_task = 1
    pages_read_per_task = 1
    provider_outbound_max_per_task = 1
    search_outbound_max_per_task = 1
    # Frozen A.5.2 accounting: one initial fetch plus three redirect fetches.
    reader_outbound_max_per_task = 4

    def __init__(
        self,
        component_factory: ComponentFactory,
        *,
        expected_task_count: int = 40,
        batch_provider_budget: int | None = None,
        transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None,
        search_transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None,
        reader_transport_factory: Callable[[], httpx.AsyncBaseTransport] | None = None,
    ) -> None:
        if expected_task_count <= 0:
            raise ValueError("expected_task_count must be positive")
        expected_budget = expected_task_count * self.provider_outbound_max_per_task
        if batch_provider_budget is not None and batch_provider_budget != expected_budget:
            raise ValueError("batch budget must equal frozen task count times per-task cap")
        self._component_factory = component_factory
        self._expected_task_count = expected_task_count
        self._batch_budget = CallBudget(maximum=expected_budget)
        self._batch_search_budget = CallBudget(
            maximum=expected_task_count * self.search_outbound_max_per_task
        )
        self._batch_reader_budget = CallBudget(
            maximum=expected_task_count * self.reader_outbound_max_per_task
        )
        self._transport_factory = transport_factory
        self._search_transport_factory = search_transport_factory
        self._reader_transport_factory = reader_transport_factory

    async def execute(
        self,
        tasks: Sequence[AcquisitionTask],
        *,
        execution_context: ProspectiveAcquisitionExecutionContext,
        profile_service: ProviderProfileLister,
        profile_repository: ActiveProfileRepository,
    ) -> BoundedProfileResult:
        if len(tasks) != self._expected_task_count:
            raise ValueError("task manifest count differs from frozen profile count")
        validate_frozen_context(execution_context)
        binding = execution_context.provider_binding
        assert binding is not None
        task_ids = [task.task_id for task in tasks]
        run_ids = [task.run_id for task in tasks]
        if len(set(task_ids)) != len(task_ids) or len(set(run_ids)) != len(run_ids):
            raise ValueError("task and run identities must be unique")
        if any(
            task.execution_context_id != execution_context.execution_context_id for task in tasks
        ):
            raise ValueError("task execution context does not match frozen batch context")

        executions: list[BoundedTaskExecution] = []
        for task in tasks:
            # Scope/profile drift is checked before constructing a budgeted
            # client, so identity failures cannot consume outbound capacity.
            await assert_provider_binding_unchanged(
                execution_context, profile_service, profile_repository
            )
            task_budget = CallBudget(
                maximum=self.provider_outbound_max_per_task,
                parent=self._batch_budget,
            )
            search_budget = CallBudget(
                maximum=self.search_outbound_max_per_task,
                parent=self._batch_search_budget,
            )
            reader_budget = CallBudget(
                maximum=self.reader_outbound_max_per_task,
                parent=self._batch_reader_budget,
            )
            clients = AcquisitionHTTPClients(
                provider=build_guarded_provider_client(
                    task_budget,
                    transport=self._transport_factory() if self._transport_factory else None,
                ),
                search=build_guarded_http_client(
                    search_budget,
                    transport=(
                        self._search_transport_factory()
                        if self._search_transport_factory
                        else None
                    ),
                ),
                reader=build_guarded_http_client(
                    reader_budget,
                    transport=(
                        self._reader_transport_factory()
                        if self._reader_transport_factory
                        else None
                    ),
                ),
            )
            result = TaskResult(task.task_id, task.run_id, ())
            failure_stage: str | None = None
            status = "NO_SEARCH_RESULT"
            try:
                components = self._component_factory(task, execution_context, binding, clients)
                # The generic task itself is the fixed query; no planner call or
                # semantic-dependent query rewrite is performed in this profile.
                search_results = await components.search.search(task.task_text, limit=1)
                if search_results:
                    status = "READER_FAILED"
                    page = await components.reader.read(search_results[0].url)
                    status = "EXTRACTION_FAILED"
                    evidence = await components.extractor.extract(task.task_text, page)
                    status = "CLAIM_FORMATION_FAILED"
                    claims = await components.claim_formation.record_page(task, page, evidence)
                    result = TaskResult(task.task_id, task.run_id, tuple(claims))
                    status = "CLAIM_RESULT" if claims else "ZERO_CLAIM"
            except Exception as exc:
                failure_stage = status.lower()
                status = f"FAILED:{type(exc).__name__}"
            finally:
                await clients.provider.aclose()
                await clients.search.aclose()
                await clients.reader.aclose()
            executions.append(
                BoundedTaskExecution(
                    task_id=task.task_id,
                    run_id=task.run_id,
                    status=status,
                    result=result,
                    provider_outbound_count=task_budget.dispatched,
                    search_outbound_count=search_budget.dispatched,
                    reader_outbound_count=reader_budget.dispatched,
                    blocked_extra_outbound_count=(
                        task_budget.blocked_attempts
                        + search_budget.blocked_attempts
                        + reader_budget.blocked_attempts
                    ),
                    failure_stage=failure_stage,
                )
            )
        return BoundedProfileResult(
            executions=tuple(executions),
            batch_provider_outbound_count=self._batch_budget.dispatched,
            batch_search_outbound_count=self._batch_search_budget.dispatched,
            batch_reader_outbound_count=self._batch_reader_budget.dispatched,
            batch_budget_remaining=self._batch_budget.remaining,
        )
