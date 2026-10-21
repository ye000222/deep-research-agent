"""D4-R-A.1 fake-only v3.3 canary path wiring proof.

The harness deliberately uses the production Gateway and Canary-only outbound
guard, but replaces the HTTP transport. It does not resolve credentials, read
claims from persistence, or contact an external service.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import httpx
import pytest
from app.domain.providers import AdapterType, CanonicalModelRequest, ContentPart
from app.domain.source_context_extraction_v3_3 import (
    SOURCE_CONTEXT_EXTRACTION_PROMPT_V33,
    SOURCE_CONTEXT_EXTRACTION_SCHEMA_V33,
    MeasurementAdmissibilityV33,
    SourceExtractionResponseV33,
)
from app.llm.adapters import LLMGateway
from app.llm.canary_strict import SingleOutboundClient
from app.services.source_context_extractor_v3_3 import (
    SOURCE_CONTEXT_V33_INSTRUCTIONS,
    evaluate_measurement_admissibility_v33,
)
from pydantic import SecretStr


def _unresolved_payload() -> dict[str, Any]:
    return {
        "schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V33,
        "status": "UNRESOLVED",
        "unresolved_items": ["measurement identity ambiguous"],
    }


def _response_for_payload(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "fake-request-id",
            "choices": [
                {
                    "message": {"content": json.dumps(payload)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


async def _run_fake_canary(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    claim_text: str = "AUC is 99.6%.",
    attempts: int = 1,
) -> tuple[list[dict[str, Any]], SingleOutboundClient, list[httpx.Request]]:
    transport_requests: list[httpx.Request] = []

    def capture(request: httpx.Request) -> httpx.Response:
        transport_requests.append(request)
        return handler(request)

    transport = httpx.MockTransport(capture)
    async with httpx.AsyncClient(transport=transport) as mock_http:
        strict = SingleOutboundClient(mock_http)
        gateway = LLMGateway(strict)  # type: ignore[arg-type]
        request = CanonicalModelRequest(
            task_kind="source_context_shadow_extraction_v3_3",
            role="source_context_extractor_v3_3",
            model="fake-v33-model",
            instructions=SOURCE_CONTEXT_V33_INSTRUCTIONS,
            content_parts=(ContentPart(kind="json", value={"claim_text": claim_text}),),
            response_contract=SourceExtractionResponseV33.model_json_schema(),
            generation_parameters={"temperature": 0.0, "reasoning_enabled": False},
            max_output_tokens=8_000,
            context_manifest_id=uuid4(),
            metadata={
                "prompt_version": SOURCE_CONTEXT_EXTRACTION_PROMPT_V33,
                "schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V33,
                "execution_mode": "SHADOW",
                "fusion_policy": "NONE",
            },
        )
        results: list[dict[str, Any]] = []
        for _ in range(attempts):
            try:
                raw = await gateway.generate_structured(
                    adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
                    base_url="https://fake-provider.invalid/v1",
                    api_key=SecretStr("fake-only-not-a-real-credential"),
                    request=request,
                    allow_regeneration=False,
                )
                response = SourceExtractionResponseV33.model_validate(raw.parsed_object)
                admissibility = evaluate_measurement_admissibility_v33(
                    response, claim_text
                )
                results.append(
                    {
                        "status": response.status,
                        "admissibility": {
                            key: value.value for key, value in admissibility.items()
                        },
                    }
                )
            except Exception as exc:  # expose only safe type/code in the test trace
                results.append(
                    {
                        "error_type": type(exc).__name__,
                        "error_code": getattr(exc, "code", None),
                    }
                )
        return results, strict, transport_requests


@pytest.mark.asyncio
async def test_second_gateway_outbound_is_blocked_before_mock_transport() -> None:
    results, strict, requests = await _run_fake_canary(
        lambda _request: _request_response(), attempts=2
    )

    assert len(results) == 2
    assert results[1]["error_code"] == "CANARY_OUTBOUND_LIMIT_EXCEEDED"
    assert strict.external_http_outbound_count == 1
    assert strict.blocked_additional_attempt_count == 1
    assert len(requests) == 1


def _request_response() -> httpx.Response:
    return _response_for_payload(_unresolved_payload())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [httpx.ReadTimeout("fake timeout"), httpx.ConnectError("fake network error")],
)
async def test_timeout_and_network_error_are_not_retried(error: Exception) -> None:
    def fail(_request: httpx.Request) -> httpx.Response:
        raise error

    _results, strict, requests = await _run_fake_canary(fail)
    assert strict.external_http_outbound_count == 1
    assert strict.blocked_additional_attempt_count == 0
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_schema_failure_does_not_regenerate() -> None:
    invalid = {"not": "the v3.3 contract"}
    _results, strict, requests = await _run_fake_canary(
        lambda _request: _response_for_payload(invalid)
    )
    assert strict.external_http_outbound_count == 1
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_unknown_measurement_remains_unknown_without_repair() -> None:
    payload = _unresolved_payload()
    payload["source_contexts"] = [
        {
            "context_ref": "ctx1",
            "context_type": "MEASUREMENT",
            "semantic_value": {
                "measurement_identity": {"name": "performance", "identity_type": "UNKNOWN"}
            },
        }
    ]
    results, strict, requests = await _run_fake_canary(
        lambda _request: _response_for_payload(payload)
    )

    assert results[0]["admissibility"]["ctx1"] == MeasurementAdmissibilityV33.UNKNOWN.value
    assert strict.external_http_outbound_count == 1
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_gateway_request_is_v33_shadow_and_has_no_fallback_attempt() -> None:
    _results, strict, requests = await _run_fake_canary(_request_response)
    assert strict.external_http_outbound_count == 1
    assert len(requests) == 1
    assert str(requests[0].url) == "https://fake-provider.invalid/v1/chat/completions"
    body = json.loads(requests[0].content)
    assert body["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_request_format_fallback_is_blocked_before_second_http_request() -> None:
    results, strict, requests = await _run_fake_canary(
        lambda _request: httpx.Response(400, json={"error": {"message": "unsupported"}})
    )
    assert results[0]["error_code"] == "CANARY_OUTBOUND_LIMIT_EXCEEDED"
    assert strict.external_http_outbound_count == 1
    assert strict.blocked_additional_attempt_count == 1
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_deepseek_endpoint_fallback_is_blocked_before_second_http_request() -> None:
    requests: list[httpx.Request] = []

    def capture(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectTimeout("fake timeout")

    async with httpx.AsyncClient(transport=httpx.MockTransport(capture)) as mock_http:
        strict = SingleOutboundClient(mock_http)
        gateway = LLMGateway(strict)  # type: ignore[arg-type]
        request = CanonicalModelRequest(
            task_kind="source_context_shadow_extraction_v3_3",
            role="source_context_extractor_v3_3",
            model="deepseek-flash",
            instructions=SOURCE_CONTEXT_V33_INSTRUCTIONS,
            content_parts=(ContentPart(kind="json", value={"claim_text": "test"}),),
            response_contract=SourceExtractionResponseV33.model_json_schema(),
            generation_parameters={"temperature": 0.0, "reasoning_enabled": False},
            max_output_tokens=8_000,
            context_manifest_id=uuid4(),
            metadata={
                "prompt_version": SOURCE_CONTEXT_EXTRACTION_PROMPT_V33,
                "schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V33,
                "execution_mode": "SHADOW",
                "fusion_policy": "NONE",
            },
        )
        with pytest.raises(Exception) as raised:
            await gateway.generate_structured(
                adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
                base_url="https://api.deepseek.com",
                api_key=SecretStr("fake-only-not-a-real-credential"),
                request=request,
                allow_regeneration=False,
            )

    assert getattr(raised.value, "code", None) == "CANARY_OUTBOUND_LIMIT_EXCEEDED"
    assert strict.external_http_outbound_count == 1
    assert strict.blocked_additional_attempt_count == 1
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.deepseek.com/chat/completions"
