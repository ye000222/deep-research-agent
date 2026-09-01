"""Research Run lifecycle and PostgreSQL-backed SSE replay endpoints."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.dependencies import get_client_session, get_research_run_service
from app.domain.context import ContextManifestView
from app.domain.controlled_tools import (
    AnalysisOperation,
    AnalyzeDataInput,
    AnalyzeDataResult,
    EvidenceSearchInput,
    EvidenceSearchResult,
)
from app.domain.evaluation import EvaluationSnapshot
from app.domain.evidence_graph import EvidenceGraphView
from app.domain.memory import MemoryAccessView, MemoryItemView
from app.domain.planning import ResearchPlan
from app.domain.reports import ReportView, VerificationView
from app.domain.research_runs import TERMINAL_RUN_STATUSES, ResearchRunView
from app.domain.research_tools import EvidenceView
from app.domain.state import (
    CoverageDimensionSnapshot,
    KnownClaimRef,
    NextAction,
    QualitySnapshot,
    ResearchGap,
    ResearchState,
)
from app.infrastructure.db.reports import ReportNotFoundError
from app.infrastructure.db.research_runs import (
    CredentialVersionNotFoundError,
    InvalidRunTransitionError,
    ResearchPlanNotFoundError,
    ResearchRunNotFoundError,
)
from app.infrastructure.db.state_runtime import StateSnapshotNotFoundError
from app.security.client_sessions import ClientSession
from app.services.research_runs import ResearchRunServiceProtocol
from app.tools.errors import ToolExecutionError

router = APIRouter(prefix="/api/v1/research-runs", tags=["research-runs"])


class ResearchRunCreate(BaseModel):
    query: str = Field(min_length=1, max_length=20_000)
    saved_profile_version_id: UUID
    budget_tier: str = Field(default="quick", pattern="^(quick|standard|deep)$")



class EvidenceSearchPayload(BaseModel):
    action_id: UUID
    target_gap_ids: tuple[UUID, ...] = Field(min_length=1)
    question_id: str = Field(min_length=1, max_length=50)
    query: str = Field(min_length=1, max_length=2000)
    min_score: float = Field(default=0.55, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1, le=20)


class AnalyzeDataPayload(BaseModel):
    action_id: UUID
    target_gap_ids: tuple[UUID, ...] = Field(min_length=1)
    question_id: str = Field(min_length=1, max_length=50)
    operation: AnalysisOperation
    data: tuple[dict[str, object], ...] = Field(min_length=1, max_length=100)
    parameters: dict[str, object] = Field(default_factory=dict)


class ResearchRunResponse(BaseModel):
    run_id: UUID
    original_query: str
    normalized_goal: str
    status: str
    phase: str
    state_version: int
    plan_version: int
    next_event_seq: int
    credential_status: str
    saved_profile_id: UUID
    credential_version_id: UUID
    llm_config_snapshot: dict[str, object]
    budget_snapshot: dict[str, object]
    usage_snapshot: dict[str, object]
    quality_snapshot: dict[str, object]
    termination_reason: str | None
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None
    status_url: str
    event_url: str

    @classmethod
    def from_view(cls, view: ResearchRunView) -> ResearchRunResponse:
        payload = jsonable_encoder(asdict(view))
        payload["status_url"] = f"/api/v1/research-runs/{view.run_id}"
        payload["event_url"] = f"/api/v1/research-runs/{view.run_id}/events"
        return cls.model_validate(payload)


class KnowledgeLedgerResponse(BaseModel):
    run_id: UUID
    state_version: int
    known: tuple[KnownClaimRef, ...]
    coverage_map: tuple[CoverageDimensionSnapshot, ...]
    quality: QualitySnapshot


class GapLedgerResponse(BaseModel):
    run_id: UUID
    state_version: int
    gaps: tuple[ResearchGap, ...]
    open_count: int


class ActionLedgerResponse(BaseModel):
    run_id: UUID
    state_version: int
    next_action: NextAction | None
    events: tuple[object, ...]


class RunLLMConfigResponse(BaseModel):
    run_id: UUID
    state_version: int
    config: dict[str, object]


@router.post("", response_model=ResearchRunResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_run(
    payload: ResearchRunCreate,
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> ResearchRunResponse:
    try:
        run, _created = await service.create_run(
            client.owner_hash,
            idempotency_key=idempotency_key,
            query=payload.query,
            saved_profile_version_id=payload.saved_profile_version_id,
            budget_tier=payload.budget_tier,
        )
    except CredentialVersionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error_code": "credential_version_not_found"},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error_code": "invalid_research_run", "message": str(exc)},
        ) from exc
    return ResearchRunResponse.from_view(run)


@router.get("", response_model=list[ResearchRunResponse])
async def list_runs(
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[ResearchRunResponse]:
    runs = await service.list_runs(client.owner_hash, limit=limit)
    return [ResearchRunResponse.from_view(run) for run in runs]


@router.get("/{run_id}", response_model=ResearchRunResponse)
async def get_run(
    run_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> ResearchRunResponse:
    return ResearchRunResponse.from_view(await _get_run(service, client, run_id))


@router.get("/{run_id}/plan", response_model=ResearchPlan)
async def get_plan(
    run_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> ResearchPlan:
    try:
        return await service.get_plan(client.owner_hash, run_id)
    except ResearchRunNotFoundError as exc:
        raise _not_found() from exc
    except ResearchPlanNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error_code": "research_plan_not_ready"},
        ) from exc


@router.get("/{run_id}/state", response_model=ResearchState)
async def get_state(
    run_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> ResearchState:
    try:
        return await service.get_state(client.owner_hash, run_id)
    except ResearchRunNotFoundError as exc:
        raise _not_found() from exc
    except StateSnapshotNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error_code": "RESEARCH_STATE_NOT_READY"},
        ) from exc


@router.get("/{run_id}/knowledge", response_model=KnowledgeLedgerResponse)
async def get_knowledge(
    run_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> KnowledgeLedgerResponse:
    state = await _get_state(service, client, run_id)
    return KnowledgeLedgerResponse(
        run_id=run_id,
        state_version=state.state_version,
        known=state.known,
        coverage_map=state.coverage_map,
        quality=state.quality,
    )


@router.get("/{run_id}/gaps", response_model=GapLedgerResponse)
async def get_gaps(
    run_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> GapLedgerResponse:
    state = await _get_state(service, client, run_id)
    open_count = sum(1 for gap in state.gaps if gap.status.value in {"open", "resolving"})
    return GapLedgerResponse(
        run_id=run_id,
        state_version=state.state_version,
        gaps=state.gaps,
        open_count=open_count,
    )


@router.get("/{run_id}/actions", response_model=ActionLedgerResponse)
async def get_actions(
    run_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> ActionLedgerResponse:
    state = await _get_state(service, client, run_id)
    events = await service.list_events(client.owner_hash, run_id, after_seq=0)
    action_events = tuple(
        event
        for event in events
        if event.event_type.startswith("action.")
        or event.event_type.startswith("tool.")
        or event.event_type in {"research.continued", "research.information_gain_calculated"}
    )
    return ActionLedgerResponse(
        run_id=run_id,
        state_version=state.state_version,
        next_action=state.next_action,
        events=action_events,
    )


@router.get("/{run_id}/evaluations", response_model=list[EvaluationSnapshot])
async def list_evaluations(
    run_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> list[EvaluationSnapshot]:
    try:
        return await service.list_evaluations(client.owner_hash, run_id)
    except ResearchRunNotFoundError as exc:
        raise _not_found() from exc


@router.get("/{run_id}/llm-config", response_model=RunLLMConfigResponse)
async def get_llm_config(
    run_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> RunLLMConfigResponse:
    run = await _get_run(service, client, run_id)
    return RunLLMConfigResponse(
        run_id=run_id,
        state_version=run.state_version,
        config=run.llm_config_snapshot,
    )


@router.get("/{run_id}/evidence", response_model=list[EvidenceView])
async def list_evidence(
    run_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> list[EvidenceView]:
    try:
        return await service.list_evidence(client.owner_hash, run_id)
    except ResearchRunNotFoundError as exc:
        raise _not_found() from exc


@router.get("/{run_id}/evidence-graph", response_model=EvidenceGraphView)
async def get_evidence_graph(
    run_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> EvidenceGraphView:
    try:
        return await service.get_evidence_graph(client.owner_hash, run_id)
    except ResearchRunNotFoundError as exc:
        raise _not_found() from exc


@router.get("/{run_id}/context-metrics", response_model=list[ContextManifestView])
async def list_context_metrics(
    run_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> list[ContextManifestView]:
    try:
        return await service.list_context_metrics(client.owner_hash, run_id)
    except ResearchRunNotFoundError as exc:
        raise _not_found() from exc


@router.get("/{run_id}/memory", response_model=list[MemoryItemView])
async def list_memory(
    run_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> list[MemoryItemView]:
    try:
        return await service.list_memory(client.owner_hash, run_id)
    except ResearchRunNotFoundError as exc:
        raise _not_found() from exc


@router.get("/{run_id}/memory-accesses", response_model=list[MemoryAccessView])
async def list_memory_accesses(
    run_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> list[MemoryAccessView]:
    try:
        return await service.list_memory_accesses(client.owner_hash, run_id)
    except ResearchRunNotFoundError as exc:
        raise _not_found() from exc


@router.post("/{run_id}/search-evidence")
async def search_evidence_tool(
    run_id: UUID,
    payload: EvidenceSearchPayload,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> EvidenceSearchResult:
    request = EvidenceSearchInput(run_id=run_id, **payload.model_dump())
    try:
        return await service.search_evidence(client.owner_hash, run_id, request)
    except ResearchRunNotFoundError as exc:
        raise _not_found() from exc
    except ToolExecutionError as exc:
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT if exc.retryable else status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error_code": exc.code, "retryable": exc.retryable},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "invalid_tool_input", "message": str(exc)},
        ) from exc


@router.post("/{run_id}/analyze-data")
async def analyze_data_tool(
    run_id: UUID,
    payload: AnalyzeDataPayload,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> AnalyzeDataResult:
    request = AnalyzeDataInput(run_id=run_id, **payload.model_dump())
    try:
        return await service.analyze_data(client.owner_hash, run_id, request)
    except ResearchRunNotFoundError as exc:
        raise _not_found() from exc
    except ToolExecutionError as exc:
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT if exc.retryable else status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error_code": exc.code, "retryable": exc.retryable},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error_code": "invalid_tool_input", "message": str(exc)},
        ) from exc


@router.get("/{run_id}/report", response_model=ReportView)
async def get_report(
    run_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> ReportView:
    try:
        return await service.get_report(client.owner_hash, run_id)
    except ResearchRunNotFoundError as exc:
        raise _not_found() from exc
    except ReportNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error_code": "REPORT_NOT_READY"},
        ) from exc


@router.post("/{run_id}/cancel", response_model=ResearchRunResponse)
async def cancel_run(
    run_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> ResearchRunResponse:
    try:
        run = await service.cancel_run(client.owner_hash, run_id)
    except ResearchRunNotFoundError as exc:
        raise _not_found() from exc
    return ResearchRunResponse.from_view(run)


@router.post("/{run_id}/resume", response_model=ResearchRunResponse)
async def resume_run(
    run_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> ResearchRunResponse:
    try:
        run = await service.resume_run(client.owner_hash, run_id)
    except ResearchRunNotFoundError as exc:
        raise _not_found() from exc
    except InvalidRunTransitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error_code": "invalid_run_transition", "message": str(exc)},
        ) from exc
    return ResearchRunResponse.from_view(run)


@router.get("/{run_id}/events")
async def stream_events(
    run_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    follow: Annotated[bool, Query()] = True,
) -> StreamingResponse:
    try:
        cursor = _parse_cursor(last_event_id)
        await service.get_run(client.owner_hash, run_id)
    except ResearchRunNotFoundError as exc:
        raise _not_found() from exc

    async def generate() -> AsyncIterator[str]:
        nonlocal cursor
        heartbeat_elapsed = 0
        yield "retry: 3000\n\n"
        while True:
            events = await service.list_events(
                client.owner_hash,
                run_id,
                after_seq=cursor,
            )
            for event in events:
                cursor = event.seq
                envelope = jsonable_encoder(asdict(event))
                data = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
                yield f"id: {event.seq}\nevent: {event.event_type}\ndata: {data}\n\n"
            if not follow:
                return
            run = await service.get_run(client.owner_hash, run_id)
            if run.status in TERMINAL_RUN_STATUSES and not events:
                return
            await asyncio.sleep(1)
            heartbeat_elapsed += 1
            if heartbeat_elapsed >= 15:
                heartbeat_elapsed = 0
                yield ": heartbeat\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


async def _get_run(
    service: ResearchRunServiceProtocol,
    client: ClientSession,
    run_id: UUID,
) -> ResearchRunView:
    try:
        return await service.get_run(client.owner_hash, run_id)
    except ResearchRunNotFoundError as exc:
        raise _not_found() from exc


async def _get_state(
    service: ResearchRunServiceProtocol,
    client: ClientSession,
    run_id: UUID,
) -> ResearchState:
    try:
        return await service.get_state(client.owner_hash, run_id)
    except ResearchRunNotFoundError as exc:
        raise _not_found() from exc
    except StateSnapshotNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error_code": "RESEARCH_STATE_NOT_READY"},
        ) from exc


@router.get("/{run_id}/verification", response_model=VerificationView)
async def get_verification(
    run_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> VerificationView:
    try:
        return await service.get_verification(client.owner_hash, run_id)
    except ResearchRunNotFoundError as exc:
        raise _not_found() from exc
    except ReportNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error_code": "REPORT_NOT_READY"},
        ) from exc


def _parse_cursor(value: str | None) -> int:
    if value is None or value == "":
        return 0
    try:
        cursor = int(value)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_code": "invalid_last_event_id"},
        ) from exc
    if cursor < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error_code": "invalid_last_event_id"},
        )
    return cursor


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error_code": "research_run_not_found"},
    )
