"""Celery entry points. Payloads contain only opaque run identifiers."""

from __future__ import annotations

import asyncio
from uuid import UUID

import httpx
from celery import Task  # type: ignore[import-untyped]

from app.context.manager import ContextBudgetManager, ContextManifestPersistenceError
from app.core.config import Settings
from app.infrastructure.artifacts import LocalArtifactStore
from app.infrastructure.checkpoints.lifecycle import CheckpointRuntime
from app.infrastructure.db.postgres import PostgresRuntime
from app.infrastructure.db.reports import (
    ReportRepository,
    ReportWritingLeaseLostError,
)
from app.infrastructure.db.research_runs import ResearchRunRepository
from app.infrastructure.db.research_tools import ResearchLeaseLostError, ResearchToolRepository
from app.infrastructure.db.run_providers import (
    RunCredentialUnavailableError,
    RunProviderBindingRepository,
)
from app.infrastructure.db.state_runtime import (
    ResearchStateRuntimeRepository,
    StateRuntimeLeaseLostError,
    StateSnapshotNotFoundError,
)
from app.llm.adapters import LLMGateway, ModelGatewayError
from app.memory.manager import ResearchMemoryManager
from app.security.secrets import SecretCipher, load_or_create_master_key
from app.services.evidence_extractor import EvidenceExtractorService
from app.services.planner import PlannerService
from app.services.report_writer import ReportWriterService
from app.services.research_graph import ResearchGraphService
from app.services.research_loop import ResearchLoopService
from app.tools.analyze_data import AnalyzeDataTool
from app.tools.gateway import ControlledToolGateway
from app.tools.search_evidence import SearchEvidenceTool
from app.tools.web_reader import PublicWebReader
from app.tools.web_search import SearXNGSearchProvider
from app.worker.celery_app import celery_app


@celery_app.task(name="deep_research.memory_lifecycle")  # type: ignore[untyped-decorator]
def run_memory_lifecycle() -> str:
    """Expire and forget Memory items; status changes are audit-preserving."""
    result = asyncio.run(_run_memory_lifecycle())
    return f"stale={result['stale']};forgotten={result['forgotten']}"


async def _run_memory_lifecycle() -> dict[str, int]:
    settings = Settings()
    database = PostgresRuntime(settings.database_url)
    try:
        return await ResearchMemoryManager(database.session_factory).apply_lifecycle()
    finally:
        await database.close()


@celery_app.task(bind=True, name="deep_research.execute_run")  # type: ignore[untyped-decorator]
def execute_research_run(task: Task, run_id: str) -> str:
    task_id = str(task.request.id)
    return asyncio.run(_execute(UUID(run_id), task_id))


async def _execute(run_id: UUID, task_id: str) -> str:
    settings = Settings()
    database = PostgresRuntime(settings.database_url)
    checkpoints = CheckpointRuntime(
        settings.checkpoint_database_uri,
        min_size=settings.checkpoint_pool_min_size,
        max_size=settings.checkpoint_pool_max_size,
    )
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(90.0, connect=10.0),
        follow_redirects=False,
        trust_env=False,
    )
    repository = ResearchRunRepository(database.session_factory)
    research_repository = ResearchToolRepository(database.session_factory)
    report_repository = ReportRepository(database.session_factory)
    state_repository = ResearchStateRuntimeRepository(database.session_factory)
    bindings = RunProviderBindingRepository(database.session_factory)
    contexts = ContextBudgetManager(database.session_factory)
    memories = ResearchMemoryManager(database.session_factory)
    controlled_tools = ControlledToolGateway(
        SearchEvidenceTool(database.session_factory),
        AnalyzeDataTool(database.session_factory),
    )
    cipher = SecretCipher(load_or_create_master_key(settings))
    gateway = LLMGateway(client)
    planner = PlannerService(bindings, cipher, gateway, contexts)
    research_loop = ResearchLoopService(
        research_repository,
        SearXNGSearchProvider(client, settings.searxng_base_url),
        PublicWebReader(client),
        EvidenceExtractorService(bindings, cipher, gateway, contexts),
        LocalArtifactStore(settings.artifact_root),
        controlled_tools,
    )
    report_writer = ReportWriterService(
        report_repository,
        bindings,
        cipher,
        gateway,
        contexts,
    )
    graph = ResearchGraphService(
        repository,
        state_repository,
        planner,
        research_loop,
        report_writer,
        memories,
    )
    try:
        acquired = await repository.acquire_for_execution(
            run_id,
            worker_task_id=task_id,
        )
        if not acquired:
            return "skipped"
        saver = await checkpoints.open()
        return await graph.execute(
            run_id,
            worker_task_id=task_id,
            checkpointer=saver,
        )
    except (ResearchLeaseLostError, ReportWritingLeaseLostError, StateRuntimeLeaseLostError):
        return "lease_lost"
    except ModelGatewayError as exc:
        await repository.fail_execution(
            run_id,
            worker_task_id=task_id,
            error_code=exc.code,
            detail_code=exc.detail_code,
            diagnostics=exc.diagnostics,
        )
        await _synchronize_failure_state(state_repository, run_id)
        return f"failed:{exc.code}"
    except RunCredentialUnavailableError:
        await repository.fail_execution(
            run_id,
            worker_task_id=task_id,
            error_code="CREDENTIAL_UNAVAILABLE",
        )
        await _synchronize_failure_state(state_repository, run_id)
        return "failed:CREDENTIAL_UNAVAILABLE"
    except ContextManifestPersistenceError as exc:
        await repository.fail_execution(
            run_id,
            worker_task_id=task_id,
            error_code=exc.code,
        )
        await _synchronize_failure_state(state_repository, run_id)
        return f"failed:{exc.code}"
    except Exception:
        await repository.fail_execution(run_id, worker_task_id=task_id)
        await _synchronize_failure_state(state_repository, run_id)
        raise
    finally:
        await client.aclose()
        await checkpoints.close()
        await database.close()


async def _synchronize_failure_state(
    repository: ResearchStateRuntimeRepository,
    run_id: UUID,
) -> None:
    """Best-effort projection after the execution repository releases its lease."""

    try:
        await repository.synchronize(
            run_id,
            node_name="failure_boundary",
            worker_task_id=None,
        )
    except StateSnapshotNotFoundError:
        return
