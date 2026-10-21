"""Canary-only v3.1 runner; caller must inject the HTTP client per Claim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.domain.providers import AdapterType
from app.llm.adapters import LLMGateway
from app.llm.canary_strict import AsyncPostClient, SingleOutboundClient
from app.services.source_context_extractor_v3_1 import SourceContextExtractorV31
from pydantic import SecretStr


@dataclass(frozen=True)
class OneClaimV31CanaryResult:
    extraction: dict[str, Any]
    external_http_outbound_count: int
    blocked_additional_attempt_count: int


async def execute_one_claim_v31(
    *,
    http_client: AsyncPostClient,
    claim_text: str,
    claim_ref: str,
    adapter_type: AdapterType,
    base_url: str,
    api_key: SecretStr,
    model: str,
) -> OneClaimV31CanaryResult:
    """Execute one v3.1 extraction through LLMGateway and its real POST boundary."""
    strict_client = SingleOutboundClient(http_client)
    gateway = LLMGateway(strict_client)  # type: ignore[arg-type]
    extractor = SourceContextExtractorV31(
        gateway,
        adapter_type=adapter_type,
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_calls=1,
    )
    result = await extractor.extract(claim_text, claim_ref=claim_ref)
    return OneClaimV31CanaryResult(
        extraction=result.model_dump(mode="json"),
        external_http_outbound_count=strict_client.external_http_outbound_count,
        blocked_additional_attempt_count=strict_client.blocked_additional_attempt_count,
    )
