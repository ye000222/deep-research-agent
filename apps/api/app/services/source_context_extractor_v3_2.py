"""One-call, non-persistent v3.2 Measurement-admissibility shadow adapter."""

from __future__ import annotations

import hashlib
import json
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

from pydantic import SecretStr, ValidationError

from app.domain.providers import AdapterType, CanonicalModelRequest, ContentPart
from app.domain.source_context_extraction_v3_1 import (
    SourceContextExtractionResultV31,
    V31ValidationStatus,
)
from app.domain.source_context_extraction_v3_2 import (
    SOURCE_CONTEXT_EXTRACTION_PROMPT_V32,
    SOURCE_CONTEXT_EXTRACTION_SCHEMA_V32,
    SourceExtractionResponseV32,
    validate_source_context_v32,
)
from app.services.source_context_extractor_v3_1 import SOURCE_CONTEXT_V31_INSTRUCTIONS

SOURCE_CONTEXT_V32_MAX_OUTPUT_TOKENS = 8_000
SOURCE_CONTEXT_V32_INSTRUCTIONS = "\n".join(
    (
        SOURCE_CONTEXT_V31_INSTRUCTIONS,
        (
            "A MEASUREMENT context requires at least one source-supported intrinsic anchor: "
            "measurement_identity, formal_definition, observed_value, or a fully typed threshold."
        ),
        (
            "Unit, comparator, basis, scope_category, target_object, and qualifiers are "
            "modifiers only and cannot alone establish a MEASUREMENT context."
        ),
        (
            "A threshold is valid only as a typed comparator plus typed value and exact raw "
            "text. Never infer an absent anchor."
        ),
    )
)


class StructuredGatewayV32(Protocol):
    async def generate_structured(
        self,
        *,
        adapter_type: AdapterType,
        base_url: str,
        api_key: SecretStr,
        request: CanonicalModelRequest,
        allow_regeneration: bool = True,
    ) -> Any: ...


class SourceContextExtractorV32:
    """Explicit v3.2 shadow-only extraction; one generation call maximum."""

    def __init__(
        self,
        gateway: StructuredGatewayV32,
        *,
        adapter_type: AdapterType,
        base_url: str,
        api_key: SecretStr,
        model: str,
        max_calls: int = 1,
    ) -> None:
        if max_calls != 1:
            raise ValueError("source extraction v3.2 call budget must be exactly one")
        self._gateway = gateway
        self._adapter_type = adapter_type
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._calls = 0

    async def extract(
        self, claim_text: str, *, claim_ref: str | None = None
    ) -> SourceContextExtractionResultV31:
        claim_hash = hashlib.sha256(claim_text.encode("utf-8")).hexdigest()
        if not claim_text.strip():
            return self._failure(claim_ref, claim_hash, "claim_missing")
        if self._calls:
            return self._failure(claim_ref, claim_hash, "call_budget_exhausted")
        request = CanonicalModelRequest(
            task_kind="source_context_shadow_extraction_v3_2",
            role="source_context_extractor_v3_2",
            model=self._model,
            instructions=SOURCE_CONTEXT_V32_INSTRUCTIONS,
            content_parts=(ContentPart(kind="json", value={"claim_text": claim_text[:12_000]}),),
            response_contract=SourceExtractionResponseV32.model_json_schema(),
            generation_parameters={"temperature": 0.0, "reasoning_enabled": False},
            max_output_tokens=SOURCE_CONTEXT_V32_MAX_OUTPUT_TOKENS,
            context_manifest_id=uuid4(),
            metadata={
                "prompt_version": SOURCE_CONTEXT_EXTRACTION_PROMPT_V32,
                "schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V32,
                "execution_mode": "SHADOW",
                "fusion_policy": "NONE",
            },
        )
        self._calls += 1
        started = perf_counter()
        try:
            raw = await self._gateway.generate_structured(
                adapter_type=self._adapter_type,
                base_url=self._base_url,
                api_key=self._api_key,
                request=request,
                allow_regeneration=False,
            )
        except Exception:
            return self._failure(
                claim_ref,
                claim_hash,
                "gateway_error",
                model_call_count=1,
                latency_ms=round((perf_counter() - started) * 1000),
            )
        latency = round((perf_counter() - started) * 1000)
        payload = getattr(raw, "parsed_object", None)
        if payload is None and isinstance(getattr(raw, "text", None), str):
            try:
                payload = json.loads(raw.text)
            except (json.JSONDecodeError, TypeError):
                payload = None
        if not isinstance(payload, dict):
            return self._failure(
                claim_ref,
                claim_hash,
                "structured_output_invalid",
                model_call_count=1,
                latency_ms=latency,
            )
        try:
            response = SourceExtractionResponseV32.model_validate(payload)
        except ValidationError as exc:
            error_types = tuple(
                sorted({str(item.get("type", "invalid"))[:64] for item in exc.errors()})
            )[:16]
            return self._failure(
                claim_ref,
                claim_hash,
                "schema_validation_failed",
                diagnostics=error_types,
                model_call_count=1,
                latency_ms=latency,
            )
        result = validate_source_context_v32(response, claim_text, claim_ref=claim_ref)
        usage = getattr(raw, "usage", None)
        return result.model_copy(
            update={
                "model_call_count": 1,
                "input_tokens": _usage(usage, "input_tokens"),
                "output_tokens": _usage(usage, "output_tokens"),
                "latency_ms": latency,
            }
        )

    @staticmethod
    def _failure(
        claim_ref: str | None,
        claim_hash: str,
        reason: str,
        *,
        diagnostics: tuple[str, ...] = (),
        model_call_count: int = 0,
        latency_ms: int = 0,
    ) -> SourceContextExtractionResultV31:
        return SourceContextExtractionResultV31(
            claim_ref=claim_ref,
            claim_hash=claim_hash,
            status=V31ValidationStatus.UNRESOLVED,
            diagnostics=(reason, *diagnostics),
            prompt_version=SOURCE_CONTEXT_EXTRACTION_PROMPT_V32,
            schema_version=SOURCE_CONTEXT_EXTRACTION_SCHEMA_V32,
            model_call_count=model_call_count,
            latency_ms=latency_ms,
        )


def _usage(usage: Any, field: str) -> int:
    value = getattr(usage, field, 0)
    return value if isinstance(value, int) and value >= 0 else 0
