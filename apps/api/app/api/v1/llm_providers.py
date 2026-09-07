"""Registered LLM adapter metadata exposed without credentials."""

from __future__ import annotations

from time import perf_counter
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, SecretStr

from app.domain.providers import (
    AdapterType,
    CanonicalModelRequest,
    CapabilityMatrix,
    CapabilitySupport,
    ContentPart,
    UsageAccuracy,
)
from app.llm.adapters import LLMGateway, ModelGatewayError

router = APIRouter(prefix="/api/v1/llm/providers", tags=["llm-providers"])


class ProviderMetadata(BaseModel):
    adapter_type: AdapterType
    display_name: str
    requires_base_url: bool
    capabilities: CapabilityMatrix


class ConnectionTestRequest(BaseModel):
    adapter_type: AdapterType
    base_url: str = Field(min_length=1, max_length=2000)
    model: str = Field(min_length=1, max_length=200)
    api_key: SecretStr = Field(min_length=1)
    test_type: str = Field(default="quick", pattern="^(quick|full)$")


class ConnectionTestResponse(BaseModel):
    connection_test_id: str
    adapter_type: AdapterType
    endpoint_host: str
    model: str
    test_type: str
    passed: bool
    latency_ms: int
    provider_request_id: str | None = None
    capability_matrix: CapabilityMatrix
    selected_strategy: dict[str, str] = Field(default_factory=dict)
    usage: dict[str, object] = Field(default_factory=dict)
    error_code: str | None = None
    detail_code: str | None = None
    profile_id: str | None = None
    credential_version_id: str | None = None
    capability_expires_at: str | None = None


_PROVIDERS = (
    ProviderMetadata(
        adapter_type=AdapterType.OPENAI_RESPONSES,
        display_name="OpenAI Responses",
        requires_base_url=True,
        capabilities=CapabilityMatrix(
            basic_generation=True,
            structured_output=CapabilitySupport.NATIVE,
            tool_calling=CapabilitySupport.NATIVE,
            streaming=CapabilitySupport.NATIVE,
            usage_reporting=UsageAccuracy.EXACT,
            reasoning_controls=CapabilitySupport.NATIVE,
            cancellation=CapabilitySupport.NATIVE,
        ),
    ),
    ProviderMetadata(
        adapter_type=AdapterType.ANTHROPIC_MESSAGES,
        display_name="Anthropic Messages",
        requires_base_url=True,
        capabilities=CapabilityMatrix(
            basic_generation=True,
            structured_output=CapabilitySupport.EMULATED,
            tool_calling=CapabilitySupport.NATIVE,
            streaming=CapabilitySupport.NATIVE,
            usage_reporting=UsageAccuracy.EXACT,
        ),
    ),
    ProviderMetadata(
        adapter_type=AdapterType.GOOGLE_GEMINI,
        display_name="Google Gemini",
        requires_base_url=True,
        capabilities=CapabilityMatrix(
            basic_generation=True,
            structured_output=CapabilitySupport.NATIVE,
            tool_calling=CapabilitySupport.NATIVE,
            streaming=CapabilitySupport.NATIVE,
            usage_reporting=UsageAccuracy.EXACT,
        ),
    ),
    ProviderMetadata(
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        display_name="OpenAI-compatible Chat",
        requires_base_url=True,
        capabilities=CapabilityMatrix(
            basic_generation=True,
            structured_output=CapabilitySupport.EMULATED,
            tool_calling=CapabilitySupport.EMULATED,
            streaming=CapabilitySupport.NATIVE,
            usage_reporting=UsageAccuracy.ESTIMATED,
        ),
    ),
)


@router.get("", response_model=list[ProviderMetadata])
async def list_providers() -> list[ProviderMetadata]:
    return list(_PROVIDERS)


@router.post("/connections/test", response_model=ConnectionTestResponse)
async def test_connection(payload: ConnectionTestRequest) -> ConnectionTestResponse:
    """Perform an ephemeral Quick/Full provider capability probe.

    The key is used only in this request and is never returned or persisted.
    Full currently repeats the structured probe contract while preserving a
    stable response shape for future streaming/tool probes.
    """

    parsed = urlsplit(payload.base_url.strip())
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error_code": "INVALID_BASE_URL"},
        )
    endpoint_host = parsed.hostname.lower()
    request = CanonicalModelRequest(
        task_kind="provider_connection_test",
        role="capability_probe",
        model=payload.model,
        instructions="Return only JSON with ok=true.",
        content_parts=(ContentPart(kind="text", value="Capability probe"),),
        response_contract={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        max_output_tokens=64,
        context_manifest_id=uuid4(),
    )
    started = perf_counter()
    capabilities = CapabilityMatrix(
        basic_generation=False,
        structured_output=CapabilitySupport.UNKNOWN,
        tool_calling=CapabilitySupport.UNKNOWN,
        streaming=CapabilitySupport.UNKNOWN,
        usage_reporting=UsageAccuracy.UNAVAILABLE,
    )
    try:
        timeout = httpx.Timeout(30.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            result = await LLMGateway(client).generate_structured(
                adapter_type=payload.adapter_type,
                base_url=payload.base_url.rstrip("/"),
                api_key=payload.api_key,
                request=request,
            )
        capabilities = capabilities.model_copy(
            update={
                "basic_generation": True,
                "structured_output": CapabilitySupport.NATIVE
                if result.capability_strategy.get("structured_output", "").startswith("native")
                else CapabilitySupport.EMULATED,
                "usage_reporting": result.usage.accuracy,
            }
        )
        return ConnectionTestResponse(
            connection_test_id=str(uuid4()),
            adapter_type=payload.adapter_type,
            endpoint_host=endpoint_host,
            model=payload.model,
            test_type=payload.test_type,
            passed=True,
            latency_ms=round((perf_counter() - started) * 1000),
            provider_request_id=result.provider_request_id,
            capability_matrix=capabilities,
            selected_strategy=result.capability_strategy,
            usage=result.usage.model_dump(mode="json"),
        )
    except ModelGatewayError as exc:
        return ConnectionTestResponse(
            connection_test_id=str(uuid4()),
            adapter_type=payload.adapter_type,
            endpoint_host=endpoint_host,
            model=payload.model,
            test_type=payload.test_type,
            passed=False,
            latency_ms=round((perf_counter() - started) * 1000),
            capability_matrix=capabilities,
            error_code=exc.code,
            detail_code=exc.detail_code,
        )
