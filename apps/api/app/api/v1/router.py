"""Top-level v1 API router."""

from fastapi import APIRouter, Request

from app.api.v1.health import router as health_router
from app.api.v1.llm_providers import router as llm_providers_router
from app.api.v1.provider_profiles import router as provider_profiles_router
from app.api.v1.reports import router as reports_router
from app.api.v1.research_runs import router as research_runs_router

router = APIRouter()
router.include_router(health_router)
router.include_router(llm_providers_router)
router.include_router(provider_profiles_router)
router.include_router(research_runs_router)
router.include_router(reports_router)


@router.get("/api/v1/meta", tags=["system"])
async def meta(request: Request) -> dict[str, str]:
    settings = request.app.state.settings
    return {
        "name": settings.app_name,
        "environment": settings.app_env,
        "version": settings.app_version,
    }
