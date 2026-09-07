"""Encrypted, session-isolated provider profile endpoints."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field, SecretStr

from app.api.dependencies import get_client_session, get_profile_service
from app.api.v1.llm_providers import ConnectionTestRequest, ConnectionTestResponse, test_connection
from app.domain.provider_profiles import ProviderProfileView
from app.domain.providers import AdapterType
from app.infrastructure.db.provider_profiles import ProfileNotFoundError
from app.security.client_sessions import ClientSession
from app.services.provider_profiles import ProviderProfileServiceProtocol

router = APIRouter(prefix="/api/v1/llm/profiles", tags=["llm-profiles"])


class ProviderProfileCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    adapter_type: AdapterType
    base_url: str = Field(min_length=1, max_length=500)
    model: str = Field(min_length=1, max_length=200)
    api_key: SecretStr = Field(min_length=1, max_length=10_000)
    is_default: bool = True


class ProviderProfileUpdate(BaseModel):
    adapter_type: AdapterType | None = None
    name: str | None = Field(default=None, min_length=1, max_length=120)
    base_url: str | None = Field(default=None, min_length=1, max_length=500)
    model: str | None = Field(default=None, min_length=1, max_length=200)
    is_default: bool | None = None


class CredentialRotate(BaseModel):
    api_key: SecretStr = Field(min_length=1, max_length=10_000)


class ProfileCapabilityTestRequest(BaseModel):
    test_type: str = Field(default="quick", pattern="^(quick|full)$")


class ProviderProfileResponse(BaseModel):
    profile_id: UUID
    name: str
    adapter_type: AdapterType
    base_url: str
    endpoint_host: str
    model: str
    status: str
    version: int
    is_default: bool
    credential_version_id: UUID
    credential_version: int
    credential_last_four: str
    credential_fingerprint: str
    has_saved_credential: bool = True
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_view(cls, view: ProviderProfileView) -> ProviderProfileResponse:
        payload = asdict(view)
        payload["credential_fingerprint"] = view.credential_fingerprint[:12]
        return cls.model_validate(payload)


@router.get("", response_model=list[ProviderProfileResponse])
async def list_profiles(
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ProviderProfileServiceProtocol, Depends(get_profile_service)],
) -> list[ProviderProfileResponse]:
    profiles = await service.list_profiles(client.owner_hash)
    return [ProviderProfileResponse.from_view(profile) for profile in profiles]


@router.post("", response_model=ProviderProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_profile(
    payload: ProviderProfileCreate,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ProviderProfileServiceProtocol, Depends(get_profile_service)],
) -> ProviderProfileResponse:
    try:
        profile = await service.create_profile(client.owner_hash, **payload.model_dump())
    except ValueError as exc:
        raise _invalid_profile(exc) from exc
    return ProviderProfileResponse.from_view(profile)


@router.patch("/{profile_id}", response_model=ProviderProfileResponse)
async def update_profile(
    profile_id: UUID,
    payload: ProviderProfileUpdate,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ProviderProfileServiceProtocol, Depends(get_profile_service)],
) -> ProviderProfileResponse:
    try:
        profile = await service.update_profile(
            client.owner_hash,
            profile_id,
            **payload.model_dump(),
        )
    except ProfileNotFoundError as exc:
        raise _not_found() from exc
    except ValueError as exc:
        raise _invalid_profile(exc) from exc
    return ProviderProfileResponse.from_view(profile)


@router.post("/{profile_id}/credentials/rotate", response_model=ProviderProfileResponse)
async def rotate_credential(
    profile_id: UUID,
    payload: CredentialRotate,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ProviderProfileServiceProtocol, Depends(get_profile_service)],
) -> ProviderProfileResponse:
    try:
        profile = await service.rotate_credential(
            client.owner_hash,
            profile_id,
            payload.api_key,
        )
    except ProfileNotFoundError as exc:
        raise _not_found() from exc
    return ProviderProfileResponse.from_view(profile)


@router.post("/{profile_id}/test", response_model=ConnectionTestResponse)
async def test_saved_profile(
    profile_id: UUID,
    payload: ProfileCapabilityTestRequest,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ProviderProfileServiceProtocol, Depends(get_profile_service)],
) -> ConnectionTestResponse:
    try:
        material = await service.get_test_material(client.owner_hash, profile_id)
        result = await test_connection(
            ConnectionTestRequest(
                adapter_type=material.adapter_type,
                base_url=material.base_url,
                model=material.model,
                api_key=material.api_key,
                test_type=payload.test_type,
            )
        )
        from datetime import UTC, datetime, timedelta

        capability = await service.record_capability_test(
            profile_id=material.profile_id,
            credential_version_id=material.credential_version_id,
            adapter_type=material.adapter_type.value,
            model=material.model,
            test_type=payload.test_type,
            passed=result.passed,
            capability_matrix=result.capability_matrix.model_dump(mode="json"),
            selected_fallbacks=result.selected_strategy,
            usage=result.usage,
            latency_ms=result.latency_ms,
            provider_request_id=result.provider_request_id,
            error_code=result.error_code,
            detail_code=result.detail_code,
        )
        expires_at = getattr(capability, "expires_at", datetime.now(UTC) + timedelta(hours=24))
        return result.model_copy(
            update={
                "profile_id": str(material.profile_id),
                "credential_version_id": str(material.credential_version_id),
                "capability_expires_at": expires_at.isoformat(),
            }
        )
    except ProfileNotFoundError as exc:
        raise _not_found() from exc


@router.delete("/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_profile(
    profile_id: UUID,
    client: Annotated[ClientSession, Depends(get_client_session)],
    service: Annotated[ProviderProfileServiceProtocol, Depends(get_profile_service)],
) -> Response:
    try:
        await service.delete_profile(client.owner_hash, profile_id)
    except ProfileNotFoundError as exc:
        raise _not_found() from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"error_code": "provider_profile_not_found"},
    )


def _invalid_profile(exc: ValueError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={"error_code": "invalid_provider_profile", "message": str(exc)},
    )
