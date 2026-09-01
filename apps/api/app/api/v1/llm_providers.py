"""Registered LLM adapter metadata exposed without credentials."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.domain.providers import AdapterType, CapabilityMatrix, CapabilitySupport, UsageAccuracy

router = APIRouter(prefix="/api/v1/llm/providers", tags=["llm-providers"])


class ProviderMetadata(BaseModel):
    adapter_type: AdapterType
    display_name: str
    requires_base_url: bool
    capabilities: CapabilityMatrix


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
