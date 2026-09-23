"""Celery entry points. Payloads contain only opaque run identifiers."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from uuid import UUID

import httpx
from celery import Task  # type: ignore[import-untyped]
from pydantic import ValidationError

from app.context.manager import ContextBudgetManager, ContextManifestPersistenceError
from app.core.config import Settings
from app.domain.provider_adapter import SearchProviderAdapter
from app.domain.provider_registry import SearchProviderRegistry
from app.domain.research_runs import EXECUTION_LEASE_SECONDS
from app.infrastructure.artifacts import LocalArtifactStore
from app.infrastructure.checkpoints.lifecycle import CheckpointRuntime
from app.infrastructure.db.extraction_cache import ExtractionCacheRepository
from app.infrastructure.db.llm_calls import LLMCallRepository
from app.infrastructure.db.postgres import PostgresRuntime
from app.infrastructure.db.report_draft_cache import ReportDraftCacheRepository
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
from app.security.secrets import SecretCipher, SecretDecryptionError, load_or_create_master_key
from app.services.evidence_extractor import EvidenceExtractorService
from app.services.planner import PlannerService
from app.services.report_writer import ReportWriterService
from app.services.research_graph import ResearchGraphService
from app.services.research_loop import ResearchLoopService
from app.tools.analyze_data import AnalyzeDataTool
from app.tools.brave_search import BraveSearchAdapter
from app.tools.duckduckgo_search import DuckDuckGoAdapter
from app.tools.errors import ToolExecutionError
from app.tools.gateway import ControlledToolGateway
from app.tools.provider_adapters import BingAdapter, SearXNGAdapter
from app.tools.search_evidence import SearchEvidenceTool
from app.tools.web_reader import PublicWebReader
from app.tools.web_search import SearXNGSearchProvider
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

_DURABLE_MODEL_RETRY_CODES = {
    "MODEL_NETWORK_ERROR",
    "MODEL_PROVIDER_UNAVAILABLE",
    "MODEL_RATE_LIMITED",
    "MODEL_TIMEOUT",
}


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


@celery_app.task(name="deep_research.reconcile_stale_runs")  # type: ignore[untyped-decorator]
def reconcile_stale_runs() -> str:
    """Requeue expired worker leases for dispatcher delivery."""

    return asyncio.run(_reconcile_stale_runs())


async def _reconcile_stale_runs() -> str:
    settings = Settings()
    database = PostgresRuntime(settings.database_url)
    try:
        recovered = await ResearchRunRepository(database.session_factory).reconcile_expired_leases()
        return f"recovered={len(recovered)}"
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
    model_client = httpx.AsyncClient(
        timeout=httpx.Timeout(90.0, connect=10.0),
        # Retry connection establishment locally before consuming one of the
        # wider, durable Outbox retry windows. HTTP responses and read failures
        # are not replayed by the transport.
        transport=httpx.AsyncHTTPTransport(retries=2),
        follow_redirects=False,
        proxy=settings.model_http_proxy or None,
        trust_env=False,
    )
    search_client = httpx.AsyncClient(
        timeout=httpx.Timeout(90.0, connect=10.0),
        transport=httpx.AsyncHTTPTransport(retries=2),
        follow_redirects=False,
        trust_env=False,
    )
    # SearXNG is an internal Docker service and must stay direct, while the
    # worker's public-page reader may need a separate host proxy to reach the
    # same internet destinations returned by SearXNG.  Do not reuse the model
    # route: model and web traffic are independently configurable.
    public_web_client = httpx.AsyncClient(
        timeout=httpx.Timeout(90.0, connect=10.0),
        transport=httpx.AsyncHTTPTransport(retries=2),
        follow_redirects=False,
        proxy=settings.public_web_http_proxy or None,
        trust_env=False,
    )
    # Phase 12.4: build the search provider pool once. The registry is the
    # single source of candidate providers and is injected into the repository
    # (routing) and the loop (adapter dispatch) so both agree on the pool.
    brave_secret = getattr(settings, "brave_api_key", None)
    brave_key = brave_secret.get_secret_value() if brave_secret else ""
    provider_registry = SearchProviderRegistry.default(
        brave_api_key_present=bool(brave_key),
        duckduckgo_enabled=getattr(settings, "duckduckgo_enabled", True),
    )
    searxng_provider = SearXNGSearchProvider(
        search_client,
        settings.searxng_base_url,
        fallback_client=public_web_client,
    )
    search_adapters: dict[str, SearchProviderAdapter] = {
        "SearXNG": SearXNGAdapter(searxng_provider),
        "Bing": BingAdapter(searxng_provider),
    }
    if brave_key:
        search_adapters["Brave"] = BraveSearchAdapter(public_web_client, api_key=brave_key)
    if getattr(settings, "duckduckgo_enabled", True):
        search_adapters["DuckDuckGo"] = DuckDuckGoAdapter(public_web_client)
    repository = ResearchRunRepository(database.session_factory)
    llm_calls = LLMCallRepository(database.session_factory)
    research_repository = ResearchToolRepository(
        database.session_factory,
        provider_registry=provider_registry,
    )
    report_repository = ReportRepository(database.session_factory)
    state_repository = ResearchStateRuntimeRepository(database.session_factory)
    bindings = RunProviderBindingRepository(database.session_factory)
    contexts = ContextBudgetManager(database.session_factory)
    extraction_cache = ExtractionCacheRepository(database.session_factory)
    report_draft_cache = ReportDraftCacheRepository(database.session_factory)
    memories = ResearchMemoryManager(database.session_factory)
    controlled_tools = ControlledToolGateway(
        SearchEvidenceTool(database.session_factory),
        AnalyzeDataTool(database.session_factory),
    )
    try:
        cipher = SecretCipher(load_or_create_master_key(settings))
    except Exception:
        # Key/configuration failures happen before the execution try/finally
        # boundary below; still release runtimes created for this task.
        await checkpoints.close()
        await database.close()
        await model_client.aclose()
        await search_client.aclose()
        await public_web_client.aclose()
        raise
    gateway = LLMGateway(model_client, call_recorder=llm_calls.record)
    planner = PlannerService(
        bindings,
        cipher,
        gateway,
        contexts,
        budget_repository=research_repository,
    )
    research_loop = ResearchLoopService(
        research_repository,
        searxng_provider,
        PublicWebReader(public_web_client),
        EvidenceExtractorService(bindings, cipher, gateway, contexts, extraction_cache),
        LocalArtifactStore(settings.artifact_root),
        controlled_tools,
        parallel_reads_enabled=True,
        search_adapters=search_adapters,
        evidence_aware_context_enabled=settings.evidence_aware_context_enabled,
    )
    report_writer = ReportWriterService(
        report_repository,
        bindings,
        cipher,
        gateway,
        contexts,
        report_draft_cache,
        budget_repository=research_repository,
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
            lease_seconds=EXECUTION_LEASE_SECONDS,
        )
        if not acquired:
            return "skipped"
        lease_stop = asyncio.Event()
        lease_heartbeat = asyncio.create_task(
            _renew_worker_lease(
                repository,
                run_id,
                worker_task_id=task_id,
                stop_event=lease_stop,
                lease_seconds=EXECUTION_LEASE_SECONDS,
            ),
            name=f"research-lease-heartbeat-{run_id}",
        )
        saver = await checkpoints.open()
        return await graph.execute(
            run_id,
            worker_task_id=task_id,
            checkpointer=saver,
        )
    except (ResearchLeaseLostError, ReportWritingLeaseLostError, StateRuntimeLeaseLostError) as exc:
        logger.warning(
            "Research execution lost its worker lease: %s",
            type(exc).__name__,
            extra={
                "run_id": str(run_id),
                "worker_task_id": task_id,
                "lease_error_type": type(exc).__name__,
            },
        )
        return "lease_lost"
    except ModelGatewayError as exc:
        if _should_defer_model_error(exc):
            retry_delay = await repository.defer_retryable_model_error(
                run_id,
                worker_task_id=task_id,
                error_code=exc.code,
                detail_code=exc.detail_code,
            )
            if retry_delay is not None:
                await _synchronize_retry_state(
                    state_repository,
                    run_id,
                    node_name="model_retry_boundary",
                )
                return f"deferred:{exc.code}:{retry_delay}"
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
    except SecretDecryptionError as exc:
        await repository.fail_execution(
            run_id,
            worker_task_id=task_id,
            error_code="CREDENTIAL_UNAVAILABLE",
            detail_code=exc.detail_code,
        )
        await _synchronize_failure_state(state_repository, run_id)
        return "failed:CREDENTIAL_UNAVAILABLE"
    except ToolExecutionError as exc:
        if exc.retryable:
            retry_delay = await repository.defer_retryable_search_error(
                run_id,
                worker_task_id=task_id,
                error_code=exc.code,
                detail_code="SEARCH_PROVIDER_EXHAUSTED",
            )
            if retry_delay is not None:
                await _synchronize_retry_state(
                    state_repository,
                    run_id,
                    node_name="search_retry_boundary",
                )
                return f"deferred:{exc.code}:{retry_delay}"
        await repository.fail_execution(
            run_id,
            worker_task_id=task_id,
            error_code=exc.code,
            detail_code="TOOL_EXECUTION_REJECTED",
        )
        await _synchronize_failure_state(state_repository, run_id)
        return f"failed:{exc.code}"
    except ContextManifestPersistenceError as exc:
        await repository.fail_execution(
            run_id,
            worker_task_id=task_id,
            error_code=exc.code,
        )
        await _synchronize_failure_state(state_repository, run_id)
        return f"failed:{exc.code}"
    except Exception as exc:
        error_code, detail_code = _unexpected_failure_codes(exc)
        diagnostics: dict[str, str | int] = {}
        if detail_code == "PAGE_BUDGET_OVERRUN":
            diagnostics = {
                "failure_node": "research_iteration",
                "budget_type": "pages",
                "budget_limit": 0,
                "budget_used": 0,
                "budget_overrun": 0,
            }
        logger.exception(
            "Research worker execution failed",
            extra={"run_id": str(run_id), "error_code": error_code, "detail_code": detail_code},
        )
        await repository.fail_execution(
            run_id,
            worker_task_id=task_id,
            error_code=error_code,
            detail_code=detail_code,
            diagnostics=diagnostics,
        )
        await _synchronize_failure_state(state_repository, run_id)
        raise
    finally:
        if "lease_heartbeat" in locals():
            lease_stop.set()
            lease_heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await lease_heartbeat
        # Close every runtime even if one backend's shutdown raises. A
        # partial cleanup used to leak the checkpoint pool and keep subsequent
        # retries stuck behind exhausted connections.
        try:
            await model_client.aclose()
            await search_client.aclose()
            await public_web_client.aclose()
        finally:
            try:
                await checkpoints.close()
            finally:
                await database.close()


async def _renew_worker_lease(
    repository: ResearchRunRepository,
    run_id: UUID,
    *,
    worker_task_id: str,
    stop_event: asyncio.Event,
    lease_seconds: int = EXECUTION_LEASE_SECONDS,
) -> None:
    """Keep long LangGraph/checkpoint/report boundaries from looking stale."""

    while True:
        try:
            # Renew well before the five-minute lease expires.  A 60-second
            # interval can race with a long DB/model turn and miss the exact
            # expiry boundary, forcing an unnecessary reconciler takeover.
            await asyncio.wait_for(stop_event.wait(), timeout=30.0)
            return
        except TimeoutError:
            pass
        try:
            renewed = await repository.renew_execution_lease(
                run_id,
                worker_task_id=worker_task_id,
                lease_seconds=lease_seconds,
            )
        except Exception:
            logger.exception(
                "Failed to renew research worker lease",
                extra={"run_id": str(run_id)},
            )
            continue
        if not renewed:
            logger.warning(
                "Research worker lease is no longer owned by this task",
                extra={"run_id": str(run_id)},
            )
            return


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
    except Exception:
        # Failure projection is deliberately best effort. Never replace the
        # original worker exception with a secondary State synchronization error.
        logger.exception(
            "Failed to synchronize terminal ResearchState",
            extra={"run_id": str(run_id)},
        )


async def _synchronize_retry_state(
    repository: ResearchStateRuntimeRepository,
    run_id: UUID,
    *,
    node_name: str,
) -> None:
    """Best-effort projection after scheduling a durable dependency retry."""

    try:
        await repository.synchronize(
            run_id,
            node_name=node_name,
            worker_task_id=None,
        )
    except StateSnapshotNotFoundError:
        return
    except Exception:
        logger.exception(
            "Failed to synchronize deferred ResearchState",
            extra={"run_id": str(run_id)},
        )


def _should_defer_model_error(exc: ModelGatewayError) -> bool:
    """Only transient transport/provider failures receive durable retries."""

    return exc.retryable and exc.code in _DURABLE_MODEL_RETRY_CODES


def _unexpected_failure_codes(exc: Exception) -> tuple[str, str]:
    if isinstance(exc, ValidationError) and "model token budget exceeded" in str(exc):
        return "RESEARCH_BUDGET_EXHAUSTED", "MODEL_TOKEN_BUDGET_OVERRUN"
    if isinstance(exc, ValidationError) and "page budget exceeded" in str(exc):
        return "RESEARCH_BUDGET_EXHAUSTED", "PAGE_BUDGET_OVERRUN"
    if isinstance(exc, ValidationError):
        if exc.error_count():
            first_error = exc.errors(include_url=False)[0]
            location = "_".join(str(part) for part in first_error["loc"])
            error_type = str(first_error["type"])
        else:  # pragma: no cover - Pydantic ValidationError always has an item
            location = ""
            error_type = "validation_error"
        raw_detail = f"{location}_{error_type}" if location else error_type
        safe_detail = "".join(
            character if character.isalnum() else "_" for character in raw_detail.upper()
        )
        return "STATE_VALIDATION_FAILED", (safe_detail.strip("_") or "VALIDATION_ERROR")[:100]
    name = type(exc).__name__.upper()
    safe_name = "".join(character if character.isalnum() else "_" for character in name)
    return "WORKER_EXECUTION_FAILED", (safe_name.strip("_") or "UNKNOWN")[:100]
