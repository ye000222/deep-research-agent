"""Stable report citation lookup endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.api.dependencies import get_client_session, get_research_run_service
from app.domain.reports import ReportCitationView
from app.infrastructure.db.reports import ReportNotFoundError
from app.security.client_sessions import ClientSession
from app.services.research_runs import ResearchRunServiceProtocol

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])


@router.get("/{report_id}/citations/{citation_number}", response_model=ReportCitationView)
async def get_report_citation(
    report_id: UUID,
    citation_number: Annotated[int, Path(ge=1)],
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ResearchRunServiceProtocol, Depends(get_research_run_service)],
) -> ReportCitationView:
    try:
        return await service.get_report_citation(
            client.owner_hash,
            report_id,
            citation_number,
        )
    except ReportNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error_code": "REPORT_CITATION_NOT_FOUND"},
        ) from exc
