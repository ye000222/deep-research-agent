"""Development-only session-bound prospective acquisition context freeze."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.dependencies import get_existing_client_session, get_profile_service
from app.infrastructure.db.provider_profiles import ProviderProfileRepository
from app.security.client_sessions import ClientSession
from app.services.prospective_acquisition_execution_identity import (
    ExecutionIdentityError,
    ProviderCardinalityError,
    ProviderCredentialVersionInvalid,
    create_execution_context,
    resolve_provider_binding,
)
from app.services.prospective_acquisition_freeze import (
    FrozenContextConflict,
    freeze_context_artifact,
)
from app.services.provider_profiles import ProviderProfileServiceProtocol

router = APIRouter(
    prefix="/api/v1/internal/prospective-acquisition",
    tags=["prospective-acquisition-internal"],
)

_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"


class ExecutionContextFreezeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    population_id: str = Field(min_length=1, max_length=128, pattern=_ID_PATTERN)
    batch_id: str = Field(min_length=1, max_length=128, pattern=_ID_PATTERN)
    protocol_version: str = Field(min_length=1, max_length=128, pattern=_ID_PATTERN)


async def get_acquisition_profile_repository(request: Request) -> ProviderProfileRepository:
    runtime = getattr(request.app.state, "runtime", None)
    business_db = getattr(runtime, "business_db", None)
    session_factory = getattr(business_db, "session_factory", None)
    if session_factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error_code": "provider_profile_persistence_disabled"},
        )
    return ProviderProfileRepository(session_factory)


@router.post("/execution-context/freeze")
async def freeze_execution_context(
    payload: ExecutionContextFreezeRequest,
    request: Request,
    client: Annotated[ClientSession, Depends(get_existing_client_session)],
    profile_service: Annotated[
        ProviderProfileServiceProtocol, Depends(get_profile_service)
    ],
    profile_repository: Annotated[
        ProviderProfileRepository, Depends(get_acquisition_profile_repository)
    ],
) -> dict[str, Any]:
    settings = request.app.state.settings
    if settings.app_env.lower() not in {"development", "test"}:
        raise HTTPException(status_code=404, detail={"error_code": "not_found"})

    origin = request.headers.get("origin")
    if origin not in settings.cors_origin_list:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"error_code": "FREEZE_ORIGIN_NOT_ALLOWED"},
        )

    context = create_execution_context(
        client,
        population_id=payload.population_id,
        batch_id=payload.batch_id,
        protocol_version=payload.protocol_version,
    )
    try:
        binding = await resolve_provider_binding(
            context,
            profile_service,
            profile_repository,
        )
        frozen = context.with_provider_binding(binding)
        result = freeze_context_artifact(frozen, Path(settings.artifact_root))
    except ProviderCardinalityError as exc:
        code = str(exc)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error_code": code},
        ) from exc
    except ProviderCredentialVersionInvalid as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error_code": "PROVIDER_CREDENTIAL_VERSION_INVALID"},
        ) from exc
    except ExecutionIdentityError as exc:
        code = (
            "EXECUTION_PROVIDER_BINDING_CHANGED"
            if "changed" in str(exc).casefold()
            else "PROVIDER_ATTESTATION_FAILURE"
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error_code": code},
        ) from exc
    except FrozenContextConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error_code": str(exc)},
        ) from exc

    return {
        "status": "FROZEN" if result.created else "ALREADY_FROZEN",
        "execution_context": result.artifact,
    }
