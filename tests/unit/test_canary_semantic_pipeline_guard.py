from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import httpx
import pytest
from app.domain.providers import AdapterType, CanonicalModelRequest, ContentPart
from app.llm.adapters import LLMGateway, ModelGatewayError
from app.llm.canary_semantic_pipeline_guard import (
    CanarySemanticPipelineGuard,
    SemanticCallPurpose,
)
from pydantic import SecretStr


class FakePostClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
    ) -> httpx.Response:
        self.calls.append(url)
        return httpx.Response(200, json={"ok": True})


class RejectStructuredOutputClient(FakePostClient):
    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
    ) -> httpx.Response:
        self.calls.append(url)
        return httpx.Response(
            400,
            json={
                "error": {
                    "type": "invalid_request_error",
                    "message": "response_format unsupported",
                }
            },
        )


class ConnectTimeoutClient(FakePostClient):
    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, Any],
    ) -> httpx.Response:
        self.calls.append(url)
        raise httpx.ConnectTimeout("synthetic timeout")


def _request() -> CanonicalModelRequest:
    return CanonicalModelRequest(
        task_kind="canary_guard_test",
        role="extractor",
        model="fake-model",
        instructions="Return a JSON object.",
        content_parts=(ContentPart(kind="text", value="synthetic claim"),),
        response_contract={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        max_output_tokens=32,
        context_manifest_id=uuid4(),
    )


def test_purpose_limits_and_total_limit_block_before_dispatch() -> None:
    async def scenario() -> None:
        guard = CanarySemanticPipelineGuard()
        client = FakePostClient()
        extractor = guard.client_for("claim-A", SemanticCallPurpose.EXTRACTOR, client)
        verifier = guard.client_for("claim-A", SemanticCallPurpose.SEMANTIC_VERIFIER, client)
        await extractor.post("https://fake/extract", headers={}, json={})
        await verifier.post("https://fake/verify", headers={}, json={})

        with pytest.raises(ModelGatewayError, match="CANARY_OUTBOUND_LIMIT_EXCEEDED"):
            await extractor.post("https://fake/request-format-fallback", headers={}, json={})
        with pytest.raises(ModelGatewayError, match="CANARY_OUTBOUND_LIMIT_EXCEEDED"):
            await verifier.post("https://fallback.invalid/verify", headers={}, json={})

        assert client.calls == ["https://fake/extract", "https://fake/verify"]
        assert guard.snapshot("claim-A") == {
            "extractor_call_count": 1,
            "verifier_call_count": 1,
            "total_call_count": 2,
            "blocked_call_count": 2,
        }

    asyncio.run(scenario())


def test_counts_are_isolated_per_claim() -> None:
    async def scenario() -> None:
        guard = CanarySemanticPipelineGuard()
        client = FakePostClient()
        claim_a = guard.client_for("claim-A", SemanticCallPurpose.EXTRACTOR, client)
        claim_b = guard.client_for("claim-B", SemanticCallPurpose.EXTRACTOR, client)
        await claim_a.post("https://fake/a", headers={}, json={})
        await claim_b.post("https://fake/b", headers={}, json={})
        assert guard.snapshot("claim-A")["total_call_count"] == 1
        assert guard.snapshot("claim-B")["total_call_count"] == 1

    asyncio.run(scenario())


def test_gateway_request_format_fallback_is_blocked_before_second_dispatch() -> None:
    async def scenario() -> None:
        guard = CanarySemanticPipelineGuard()
        client = RejectStructuredOutputClient()
        wrapped = guard.client_for("claim-A", SemanticCallPurpose.EXTRACTOR, client)
        gateway = LLMGateway(wrapped)  # type: ignore[arg-type]
        with pytest.raises(ModelGatewayError):
            await gateway.generate_structured(
                adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
                base_url="https://example.invalid/v1",
                api_key=SecretStr("fake-only"),
                request=_request(),
                allow_regeneration=False,
            )
        assert len(client.calls) == 1
        assert guard.snapshot("claim-A")["blocked_call_count"] == 1

    asyncio.run(scenario())


def test_gateway_endpoint_fallback_is_blocked_before_second_dispatch() -> None:
    async def scenario() -> None:
        guard = CanarySemanticPipelineGuard()
        client = ConnectTimeoutClient()
        wrapped = guard.client_for("claim-B", SemanticCallPurpose.EXTRACTOR, client)
        gateway = LLMGateway(wrapped)  # type: ignore[arg-type]
        with pytest.raises(ModelGatewayError):
            await gateway.generate_structured(
                adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
                base_url="https://api.deepseek.com",
                api_key=SecretStr("fake-only"),
                request=_request(),
                allow_regeneration=False,
            )
        assert len(client.calls) == 1
        assert client.calls[0] == "https://api.deepseek.com/chat/completions"
        assert guard.snapshot("claim-B")["blocked_call_count"] == 1

    asyncio.run(scenario())


def test_empty_claim_id_is_rejected() -> None:
    guard = CanarySemanticPipelineGuard()
    with pytest.raises(ValueError, match="claim_id"):
        guard.client_for(" ", SemanticCallPurpose.EXTRACTOR, FakePostClient())
