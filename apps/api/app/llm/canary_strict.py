"""Canary-only outbound accounting; production HTTP clients are unaffected."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import httpx

from app.llm.adapters import ModelGatewayError


class AsyncPostClient(Protocol):
    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
    ) -> httpx.Response: ...


class SingleOutboundClient:
    """Allow at most one actual HTTP POST per wrapped Claim execution.

    The wrapper is intentionally instantiated per Claim. A Gateway-internal retry or
    endpoint fallback is rejected before it reaches the underlying HTTP client.
    Production clients do not use this class.
    """

    def __init__(self, client: AsyncPostClient) -> None:
        self._client = client
        self.external_http_outbound_count = 0
        self.blocked_additional_attempt_count = 0

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
    ) -> httpx.Response:
        if self.external_http_outbound_count >= 1:
            self.blocked_additional_attempt_count += 1
            raise ModelGatewayError(
                "CANARY_OUTBOUND_LIMIT_EXCEEDED",
                retryable=False,
                detail_code="ONE_OUTBOUND_PER_CLAIM",
            )
        self.external_http_outbound_count += 1
        return await self._client.post(url, headers=headers, json=json)
