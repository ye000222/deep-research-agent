"""Per-Claim outbound guard for isolated hybrid semantic canaries only."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import httpx

from app.llm.adapters import ModelGatewayError


class SemanticCallPurpose(StrEnum):
    EXTRACTOR = "EXTRACTOR"
    SEMANTIC_VERIFIER = "SEMANTIC_VERIFIER"


class AsyncPostClient(Protocol):
    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
    ) -> httpx.Response: ...


@dataclass
class _ClaimCounts:
    by_purpose: dict[SemanticCallPurpose, int] = field(
        default_factory=lambda: {purpose: 0 for purpose in SemanticCallPurpose}
    )
    blocked: int = 0

    @property
    def total(self) -> int:
        return sum(self.by_purpose.values())


class CanarySemanticPipelineGuard:
    """Enforce extractor<=1, verifier<=1, total<=2 before HTTP dispatch.

    Instantiate one guard for the canary execution, then pass ``client_for``
    wrappers into purpose-specific LLMGateway instances. This helper is not used
    by production gateways and has no persistence or Provider-selection logic.
    """

    def __init__(self) -> None:
        self._claims: dict[str, _ClaimCounts] = {}

    def client_for(
        self,
        claim_id: str,
        purpose: SemanticCallPurpose,
        client: AsyncPostClient,
    ) -> AsyncPostClient:
        if not claim_id.strip():
            raise ValueError("claim_id must be non-empty")
        return _ClaimScopedClient(self, claim_id, purpose, client)

    def snapshot(self, claim_id: str) -> dict[str, int]:
        counts = self._claims.get(claim_id, _ClaimCounts())
        return {
            "extractor_call_count": counts.by_purpose[SemanticCallPurpose.EXTRACTOR],
            "verifier_call_count": counts.by_purpose[SemanticCallPurpose.SEMANTIC_VERIFIER],
            "total_call_count": counts.total,
            "blocked_call_count": counts.blocked,
        }

    async def _post(
        self,
        claim_id: str,
        purpose: SemanticCallPurpose,
        client: AsyncPostClient,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
    ) -> httpx.Response:
        counts = self._claims.setdefault(claim_id, _ClaimCounts())
        if counts.by_purpose[purpose] >= 1 or counts.total >= 2:
            counts.blocked += 1
            raise ModelGatewayError(
                "CANARY_OUTBOUND_LIMIT_EXCEEDED",
                retryable=False,
                detail_code=f"{purpose.value}_ONE_CALL_PER_CLAIM",
            )
        counts.by_purpose[purpose] += 1
        return await client.post(url, headers=headers, json=json)


class _ClaimScopedClient:
    def __init__(
        self,
        guard: CanarySemanticPipelineGuard,
        claim_id: str,
        purpose: SemanticCallPurpose,
        client: AsyncPostClient,
    ) -> None:
        self._guard = guard
        self._claim_id = claim_id
        self._purpose = purpose
        self._client = client

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
    ) -> httpx.Response:
        return await self._guard._post(
            self._claim_id,
            self._purpose,
            self._client,
            url,
            headers=headers,
            json=json,
        )
