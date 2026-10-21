from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from app.domain.providers import AdapterType
from pydantic import SecretStr

from scripts.v2_1r_d4g_a_canary_runner import execute_one_claim


def valid_payload(claim: str) -> dict[str, Any]:
    return {
        "schema_version": "source-context-extraction-schema-v3",
        "status": "RESOLVED",
        "source_assertions": [
            {"source_assertion_ref": "A", "evidence_refs": ["eA"], "assertion_status": "ASSERTED"}
        ],
        "source_contexts": [],
        "source_attachments": [],
        "source_scopes": [],
        "evidence_spans": [{"evidence_ref": "eA", "exact_text": claim}],
        "evidence_bindings": [
            {
                "binding_ref": "bA",
                "evidence_ref": "eA",
                "semantic_role": "ASSERTION_EVIDENCE",
                "target_kind": "ASSERTION",
                "target_ref": "A",
            }
        ],
        "semantic_relations": [],
        "measurement_bases": [],
        "unresolved_items": [],
    }


def chat_response(payload: dict[str, Any], *, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "id": "fake-request-id",
            "choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 13, "total_tokens": 24},
        },
    )


def make_request(
    transport: httpx.AsyncBaseTransport,
    *,
    base_url: str = "https://api.deepseek.com/v1",
) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=transport, base_url=base_url)


async def run(client: httpx.AsyncClient, claim: str = "A claim."):
    return await execute_one_claim(
        http_client=client,
        claim_text=claim,
        claim_ref="synthetic-claim",
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://api.deepseek.com/v1",
        api_key=SecretStr("DO_NOT_LEAK_TEST_SECRET"),
        model="deepseek-flash",
    )


@pytest.mark.asyncio
async def test_v3_real_gateway_path_uses_one_fake_structured_request() -> None:
    seen: list[httpx.Request] = []
    claim = "A synthetic claim."

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert body["model"] == "deepseek-flash"
        assert "DO_NOT_LEAK_TEST_SECRET" not in request.content.decode("utf-8")
        return chat_response(valid_payload(claim))

    async with make_request(httpx.MockTransport(handler)) as client:
        result = await run(client, claim)

    assert len(seen) == 1
    assert result.external_http_outbound_count == 1
    assert result.blocked_additional_attempt_count == 0
    assert result.extraction["status"] == "RESOLVED"
    assert result.extraction["model_call_count"] == 1


@pytest.mark.asyncio
async def test_second_outbound_is_blocked_before_fake_transport() -> None:
    from uuid import uuid4

    from app.domain.providers import CanonicalModelRequest
    from app.llm.adapters import LLMGateway, ModelGatewayError
    from app.llm.canary_strict import SingleOutboundClient

    sent: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return chat_response(valid_payload("A claim."))

    async with make_request(httpx.MockTransport(handler)) as client:
        strict = SingleOutboundClient(client)
        gateway = LLMGateway(strict)  # type: ignore[arg-type]
        request = CanonicalModelRequest(
            task_kind="readiness-test",
            role="test",
            model="deepseek-flash",
            instructions="json",
            content_parts=(),
            response_contract={"type": "object"},
            max_output_tokens=10,
            context_manifest_id=uuid4(),
        )
        await gateway.generate_structured(
            adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
            base_url="https://api.deepseek.com/v1",
            api_key=SecretStr("not-a-real-key"),
            request=request,
            allow_regeneration=False,
        )
        with pytest.raises(ModelGatewayError, match="CANARY_OUTBOUND_LIMIT_EXCEEDED"):
            await gateway.generate_structured(
                adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
                base_url="https://api.deepseek.com/v1",
                api_key=SecretStr("not-a-real-key"),
                request=request,
                allow_regeneration=False,
            )

    assert len(sent) == 1
    assert strict.external_http_outbound_count == 1
    assert strict.blocked_additional_attempt_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [httpx.ConnectTimeout("fake"), httpx.ConnectError("fake")])
async def test_timeout_or_network_error_does_not_retry(error: Exception) -> None:
    sent = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent
        sent += 1
        raise error

    async with make_request(httpx.MockTransport(handler)) as client:
        result = await run(client)

    assert sent == 1
    assert result.external_http_outbound_count == 1
    assert result.blocked_additional_attempt_count == 0
    assert result.extraction["status"] == "UNRESOLVED"


@pytest.mark.asyncio
async def test_request_format_fallback_is_blocked_before_second_post() -> None:
    sent: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(400, json={"error": {"message": "not retained"}})

    async with make_request(httpx.MockTransport(handler)) as client:
        result = await run(client)

    assert len(sent) == 1
    assert result.external_http_outbound_count == 1
    assert result.blocked_additional_attempt_count == 1
    assert "DO_NOT_LEAK_TEST_SECRET" not in json.dumps(result.extraction)


@pytest.mark.asyncio
async def test_endpoint_fallback_is_blocked_before_second_post() -> None:
    sent: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(str(request.url))
        raise httpx.ConnectTimeout("synthetic timeout")

    async with make_request(
        httpx.MockTransport(handler), base_url="https://api.deepseek.com"
    ) as client:
        result = await execute_one_claim(
            http_client=client,
            claim_text="A synthetic claim.",
            claim_ref="synthetic-claim",
            adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
            base_url="https://api.deepseek.com",
            api_key=SecretStr("DO_NOT_LEAK_TEST_SECRET"),
            model="deepseek-flash",
        )

    assert sent == ["https://api.deepseek.com/chat/completions"]
    assert result.external_http_outbound_count == 1
    assert result.blocked_additional_attempt_count == 1


@pytest.mark.asyncio
async def test_invalid_schema_result_is_not_regenerated() -> None:
    sent = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent
        sent += 1
        return chat_response({"schema_version": "wrong", "status": "RESOLVED"})

    async with make_request(httpx.MockTransport(handler)) as client:
        result = await run(client, "A claim. within plants")

    assert sent == 1
    assert result.external_http_outbound_count == 1
    assert result.blocked_additional_attempt_count == 0
    assert result.extraction["status"] == "UNRESOLVED"


@pytest.mark.asyncio
async def test_grounding_failure_is_not_repaired_or_regenerated() -> None:
    sent = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent
        sent += 1
        payload = valid_payload("not the claim")
        return chat_response(payload)

    async with make_request(httpx.MockTransport(handler)) as client:
        result = await run(client, "A different claim.")

    assert sent == 1
    assert result.external_http_outbound_count == 1
    assert result.extraction["status"] == "INVALID"


@pytest.mark.asyncio
async def test_missing_attachment_binding_remains_partial_without_repair() -> None:
    sent = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent
        sent += 1
        payload = valid_payload("A claim.")
        payload["source_contexts"] = [
            {
                "context_ref": "C",
                "context_type": "SCOPE",
                "semantic_value": {"scope": "within plants"},
                "lexical_value": "within plants",
                "materiality": "MATERIAL",
            }
        ]
        payload["source_attachments"] = [
            {
                "attachment_ref": "AT1",
                "context_ref": "C",
                "source_assertion_refs": ["A"],
                "relation": "APPLIES_TO",
                "status": "RESOLVED",
                "evidence_refs": ["eC"],
            }
        ]
        payload["evidence_spans"].append({"evidence_ref": "eC", "exact_text": "within plants"})
        payload["evidence_bindings"].append(
            {
                "binding_ref": "bC",
                "evidence_ref": "eC",
                "semantic_role": "SCOPE_VALUE",
                "target_kind": "CONTEXT",
                "target_ref": "C",
            }
        )
        return chat_response(payload)

    async with make_request(httpx.MockTransport(handler)) as client:
        result = await run(client, "A claim. within plants")

    assert sent == 1
    assert result.external_http_outbound_count == 1
    assert result.extraction["status"] == "PARTIAL"
    assert "attachment_binding_missing" in result.extraction["diagnostics"]


@pytest.mark.asyncio
async def test_missing_semantic_relation_binding_is_not_repaired() -> None:
    sent = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sent
        sent += 1
        payload = valid_payload("A claim.")
        payload["source_assertions"].append(
            {"source_assertion_ref": "B", "evidence_refs": ["eB"], "assertion_status": "ASSERTED"}
        )
        payload["evidence_spans"].append({"evidence_ref": "eB", "exact_text": "another claim"})
        payload["evidence_spans"].append({"evidence_ref": "eR", "exact_text": "causes"})
        payload["evidence_bindings"].append(
            {
                "binding_ref": "bB",
                "evidence_ref": "eB",
                "semantic_role": "ASSERTION_EVIDENCE",
                "target_kind": "ASSERTION",
                "target_ref": "B",
            }
        )
        payload["semantic_relations"].append(
            {
                "relation_ref": "R1",
                "relation_type": "CAUSES",
                "source_ref": "A",
                "source_kind": "ASSERTION",
                "target_ref": "B",
                "target_kind": "ASSERTION",
                "evidence_refs": ["eR"],
                "status": "RESOLVED",
                "schema_version": "source-context-extraction-schema-v3",
            }
        )
        return chat_response(payload)

    async with make_request(httpx.MockTransport(handler)) as client:
        result = await run(client, "A claim. another claim causes.")

    assert sent == 1
    assert result.external_http_outbound_count == 1
    assert result.extraction["status"] == "PARTIAL"
    assert "relation_binding_missing" in result.extraction["diagnostics"]
