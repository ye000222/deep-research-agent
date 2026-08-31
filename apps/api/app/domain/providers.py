from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class AdapterType(StrEnum):
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    GOOGLE_GEMINI = "google_gemini"
    OPENAI_COMPATIBLE_CHAT = "openai_compatible_chat"


class CapabilitySupport(StrEnum):
    NATIVE = "native"
    EMULATED = "emulated"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class UsageAccuracy(StrEnum):
    EXACT = "exact"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


class CapabilityMatrix(BaseModel):
    basic_generation: bool
    structured_output: CapabilitySupport
    tool_calling: CapabilitySupport
    streaming: CapabilitySupport
    usage_reporting: UsageAccuracy
    context_window: int | None = Field(default=None, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)
    reasoning_controls: CapabilitySupport = CapabilitySupport.UNKNOWN
    cancellation: CapabilitySupport = CapabilitySupport.UNKNOWN


class ContentPart(BaseModel):
    kind: str = Field(pattern="^(text|json)$")
    value: str | dict[str, Any]


class CanonicalModelRequest(BaseModel):
    """Provider-neutral request. Credentials are intentionally not part of this contract."""

    task_kind: str = Field(min_length=1, max_length=100)
    role: str = Field(min_length=1, max_length=50)
    model: str = Field(min_length=1, max_length=200)
    instructions: str
    content_parts: tuple[ContentPart, ...]
    response_contract: dict[str, Any] | None = None
    allowed_tools: tuple[dict[str, Any], ...] = ()
    tool_choice: str | None = None
    generation_parameters: dict[str, Any] = Field(default_factory=dict)
    max_output_tokens: int = Field(ge=1)
    stream: bool = False
    context_manifest_id: UUID
    metadata: dict[str, str] = Field(default_factory=dict)


class NormalizedToolCall(BaseModel):
    call_id: str
    name: str
    arguments: dict[str, Any]


class TokenUsage(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    accuracy: UsageAccuracy


class CanonicalModelResult(BaseModel):
    text: str | None = None
    parsed_object: dict[str, Any] | None = None
    tool_calls: tuple[NormalizedToolCall, ...] = ()
    finish_reason: str | None = None
    usage: TokenUsage
    provider_request_id: str | None = None
    capability_strategy: dict[str, str] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class RunLLMBinding(BaseModel):
    binding_id: UUID
    run_id: UUID
    role: str
    adapter_type: AdapterType
    endpoint_host: str
    exact_model: str
    credential_version_id: UUID
    capabilities: CapabilityMatrix
    config_hash: str
