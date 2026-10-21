from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

import httpx
import pytest
from app.domain.providers import AdapterType
from app.domain.source_context_grounding_v2 import GroundingStatusV2
from app.llm.canary_strict import SingleOutboundClient
from app.services.provider_profiles import ProviderProfileTestMaterial
from pydantic import SecretStr

from scripts.run_v2_1r_source_semantic_grounding_v2_canary import (
    EXPECTED_CLAIM_IDS,
    SELECTION_MANIFEST,
    SOURCE_DATASET,
    compute_component_hashes,
    run_one_claim,
    verify_frozen_components,
    verify_selection_and_resolve_claims,
)


def _material(base_url: str = "https://api.deepseek.com/v1") -> ProviderProfileTestMaterial:
    return ProviderProfileTestMaterial(
        profile_id=UUID("49a9ae5c-b062-4617-805f-dc587b706ce1"),
        credential_version_id=UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url=base_url,
        model="deepseek-flash",
        api_key=SecretStr("fake-only-not-a-real-key"),
    )


def _chat_response(content: str, *, status: int = 200) -> httpx.Response:
    body = {
        "id": "fake-request-id",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 5, "total_tokens": 9},
    }
    return httpx.Response(status, json=body if status == 200 else {"error": "fake"})


def _unresolved_model_result() -> str:
    return json.dumps(
        {
            "schema_version": "source-context-extraction-schema-v2",
            "status": "UNRESOLVED",
            "unresolved_items": ["conservative fake fixture"],
        }
    )


def _invalid_grounding_result() -> str:
    return json.dumps(
        {
            "schema_version": "source-context-extraction-schema-v2",
            "status": "RESOLVED",
            "source_assertions": [
                {
                    "source_assertion_ref": "a0",
                    "evidence_refs": ["e0"],
                    "assertion_status": "ASSERTED",
                }
            ],
            "evidence_spans": [
                {"evidence_ref": "e0", "exact_text": "invented evidence", "occurrence_index": 0}
            ],
            "evidence_bindings": [
                {
                    "binding_ref": "b0",
                    "evidence_ref": "e0",
                    "semantic_role": "ASSERTION_EVIDENCE",
                    "target_kind": "ASSERTION",
                    "target_ref": "a0",
                }
            ],
        }
    )


async def _execute_with_handler(
    handler: Callable[[httpx.Request], Awaitable[httpx.Response]],
    *,
    base_url: str = "https://api.deepseek.com/v1",
    claim_text: str = "Claim text.",
) -> tuple[dict[str, Any], list[httpx.Request]]:
    requests: list[httpx.Request] = []

    async def recording_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return await handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(recording_handler), timeout=10.0)
    try:
        row = await run_one_claim(
            {
                "claim_id": EXPECTED_CLAIM_IDS[0],
                "claim_text": claim_text,
                "claim_sha256": "fixture",
            },
            _material(base_url),
            client,
        )
    finally:
        await client.aclose()
    return row, requests


def test_frozen_manifest_claims_and_component_hashes_match() -> None:
    manifest, claims = verify_selection_and_resolve_claims()
    check = verify_frozen_components(manifest)
    assert [claim["claim_id"] for claim in claims] == list(EXPECTED_CLAIM_IDS)
    assert check["all_match"] is True
    assert check["expected_versions_match"] is True
    assert len(compute_component_hashes()) == 9
    assert SELECTION_MANIFEST.is_file()
    assert SOURCE_DATASET.is_file()


@pytest.mark.asyncio
async def test_one_claim_success_has_exactly_one_v2_structured_outbound() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return _chat_response(_unresolved_model_result())

    row, requests = await _execute_with_handler(handler)
    assert len(requests) == 1
    assert row["provider_attempt_count"] == 1
    assert row["external_http_outbound_count"] == 1
    assert row["grounding_result"]["schema_version"] == "source-context-extraction-schema-v2"
    assert (
        row["model_result"]["parsed_object"]["schema_version"]
        == "source-context-extraction-schema-v2"
    )
    body = json.loads(requests[0].content)
    assert body["response_format"] == {"type": "json_object"}
    assert body["model"] == "deepseek-flash"
    assert '"claim_text":"Claim text."' in body["messages"][1]["content"]


@pytest.mark.asyncio
async def test_request_format_failure_blocks_second_outbound() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return _chat_response("", status=400)

    row, requests = await _execute_with_handler(handler)
    assert len(requests) == 1
    assert row["external_http_outbound_count"] == 1
    assert row["blocked_additional_attempt_count"] == 1
    assert row["safe_gateway_error"]["code"] == "CANARY_OUTBOUND_LIMIT_EXCEEDED"


@pytest.mark.asyncio
async def test_deepseek_endpoint_fallback_is_blocked_before_second_outbound() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("fake timeout", request=request)

    row, requests = await _execute_with_handler(handler, base_url="https://api.deepseek.com")
    assert len(requests) == 1
    assert requests[0].url.path == "/chat/completions"
    assert row["external_http_outbound_count"] == 1
    assert row["blocked_additional_attempt_count"] == 1


@pytest.mark.asyncio
async def test_timeout_does_not_retry() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("fake timeout", request=request)

    row, requests = await _execute_with_handler(handler)
    assert len(requests) == 1
    assert row["external_http_outbound_count"] == 1
    assert row["blocked_additional_attempt_count"] == 0


@pytest.mark.asyncio
async def test_schema_invalid_response_does_not_regenerate_or_repair() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return _chat_response(
            json.dumps(
                {
                    "schema_version": "wrong-schema",
                    "status": "UNRESOLVED",
                    "unresolved_items": ["invalid fixture"],
                }
            )
        )

    row, requests = await _execute_with_handler(handler)
    assert len(requests) == 1
    assert row["external_http_outbound_count"] == 1
    assert row["grounding_result"]["diagnostics"][0] == "schema_validation_failed"


@pytest.mark.asyncio
async def test_grounding_invalid_response_does_not_make_second_model_call() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return _chat_response(_invalid_grounding_result())

    row, requests = await _execute_with_handler(handler)
    assert len(requests) == 1
    assert row["external_http_outbound_count"] == 1
    assert row["grounding_result"]["status"] == GroundingStatusV2.INVALID.value


@pytest.mark.asyncio
async def test_real_v2_runner_path_produces_typed_spans_bindings_and_three_dimensions() -> None:
    claim = "Atlas grew."
    payload = {
        "schema_version": "source-context-extraction-schema-v2",
        "status": "RESOLVED",
        "source_assertions": [
            {"source_assertion_ref": "a0", "evidence_refs": ["e0"], "assertion_status": "ASSERTED"}
        ],
        "evidence_spans": [{"evidence_ref": "e0", "exact_text": claim, "occurrence_index": 0}],
        "evidence_bindings": [
            {
                "binding_ref": "b0",
                "evidence_ref": "e0",
                "semantic_role": "ASSERTION_EVIDENCE",
                "target_kind": "ASSERTION",
                "target_ref": "a0",
            }
        ],
    }

    async def handler(_: httpx.Request) -> httpx.Response:
        return _chat_response(json.dumps(payload))

    row, requests = await _execute_with_handler(handler, claim_text=claim)
    grounding = row["grounding_result"]
    assert len(requests) == 1
    assert grounding["evidence_spans"][0]["exact_text"] == claim
    assert grounding["evidence_spans"][0]["start_offset"] == 0
    assert grounding["evidence_bindings"][0]["semantic_role"] == "ASSERTION_EVIDENCE"
    assessment = grounding["node_assessments"][0]
    assert set(assessment["dimensions"]) == {
        "textual_grounding",
        "semantic_support",
        "structural_grounding",
        "diagnostics",
    }
    assert assessment["usability"] == "GROUNDED_USABLE"


@pytest.mark.asyncio
async def test_strict_client_accounts_and_blocks_without_leaking_secret() -> None:
    sent: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={"ok": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    wrapped = SingleOutboundClient(client)
    try:
        await wrapped.post(
            "https://example.test", headers={"Authorization": "Bearer secret"}, json={}
        )
        with pytest.raises(Exception) as error:
            await wrapped.post("https://example.test", headers={}, json={})
    finally:
        await client.aclose()
    assert len(sent) == 1
    assert wrapped.external_http_outbound_count == 1
    assert wrapped.blocked_additional_attempt_count == 1
    assert "secret" not in str(error.value)
