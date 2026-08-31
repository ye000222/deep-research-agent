"""Liveness and readiness endpoints."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.core.readiness import ReadinessRegistry

router = APIRouter(tags=["system"])


@router.get("/healthz", summary="Process liveness")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", summary="Dependency readiness")
async def readyz(request: Request) -> JSONResponse:
    registry: ReadinessRegistry = request.app.state.readiness
    report = await registry.evaluate()
    return JSONResponse(
        status_code=200 if report.status.value == "ready" else 503,
        content=report.model_dump(mode="json"),
    )
