"""Isolated D4-G-A single-claim v3 canary execution wiring.

This module deliberately has no CLI and never creates a network client. Callers
must explicitly inject an HTTP client; readiness tests inject httpx.MockTransport.
It does not read claims, profiles, credentials, or persist results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.domain.providers import AdapterType
from app.llm.adapters import LLMGateway
from app.llm.canary_strict import AsyncPostClient, SingleOutboundClient
from app.services.source_context_extractor_v3 import SourceContextExtractorV3
from pydantic import SecretStr


@dataclass(frozen=True)
class OneClaimCanaryResult:
    extraction: dict[str, Any]
    external_http_outbound_count: int
    blocked_additional_attempt_count: int


async def execute_one_claim(
    *,
    http_client: AsyncPostClient,
    claim_text: str,
    claim_ref: str,
    adapter_type: AdapterType,
    base_url: str,
    api_key: SecretStr,
    model: str,
) -> OneClaimCanaryResult:
    """Run the frozen v3 extractor through the canonical Gateway with a hard cap.

    The per-claim strict wrapper counts actual POSTs and rejects a second POST
    before reaching the injected client. No DB, recorder, retry, fallback, repair,
    or result persistence is configured here.
    """
    strict_client = SingleOutboundClient(http_client)
    gateway = LLMGateway(strict_client)  # type: ignore[arg-type]
    extractor = SourceContextExtractorV3(
        gateway,
        adapter_type=adapter_type,
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_calls=1,
    )
    extraction = await extractor.extract(claim_text, claim_ref=claim_ref)
    return OneClaimCanaryResult(
        extraction=extraction.model_dump(mode="json"),
        external_http_outbound_count=strict_client.external_http_outbound_count,
        blocked_additional_attempt_count=strict_client.blocked_additional_attempt_count,
    )
