"""Dependency-light D4-G-A integration check for the canonical API image.

All outbound requests are intercepted by httpx.MockTransport. This script never
loads a provider profile/credential and has no real-network transport.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from app.domain.providers import AdapterType
from pydantic import SecretStr
from v2_1r_d4g_a_canary_runner import execute_one_claim


def response(payload: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "id": "fake-runtime-request",
            "choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        },
    )


def valid(claim: str) -> dict[str, Any]:
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


async def execute(
    client: httpx.AsyncClient,
    *,
    base_url: str = "https://api.deepseek.com/v1",
    claim_text: str = "Synthetic readiness claim.",
):
    return await execute_one_claim(
        http_client=client,
        claim_text=claim_text,
        claim_ref="synthetic-only",
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url=base_url,
        api_key=SecretStr("FAKE_ONLY_SECRET"),
        model="deepseek-flash",
    )


async def main() -> None:
    # Successful real Gateway + v3 extractor wiring, with a fully intercepted POST.
    seen: list[httpx.Request] = []

    async def success_handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        assert body.get("response_format") == {"type": "json_object"}
        assert "FAKE_ONLY_SECRET" not in request.content.decode("utf-8")
        return response(valid("Synthetic readiness claim."))

    async with httpx.AsyncClient(transport=httpx.MockTransport(success_handler)) as client:
        success = await execute(client)
    assert len(seen) == 1
    assert success.external_http_outbound_count == 1
    assert success.extraction["status"] == "RESOLVED"
    assert "FAKE_ONLY_SECRET" not in json.dumps(success.extraction)

    # Gateway's request-format fallback is blocked before a second transport send.
    fallback_requests: list[httpx.Request] = []

    async def bad_format_handler(request: httpx.Request) -> httpx.Response:
        fallback_requests.append(request)
        return httpx.Response(400, json={"error": {"message": "synthetic"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(bad_format_handler)) as client:
        format_failure = await execute(client)
    assert len(fallback_requests) == 1
    assert format_failure.external_http_outbound_count == 1
    assert format_failure.blocked_additional_attempt_count == 1

    # DeepSeek path fallback is also intercepted at the hard one-POST guard.
    endpoint_urls: list[str] = []

    async def timeout_handler(request: httpx.Request) -> httpx.Response:
        endpoint_urls.append(str(request.url))
        raise httpx.ConnectTimeout("synthetic")

    async with httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler)) as client:
        endpoint_failure = await execute(client, base_url="https://api.deepseek.com")
    assert endpoint_urls == ["https://api.deepseek.com/chat/completions"]
    assert endpoint_failure.external_http_outbound_count == 1
    assert endpoint_failure.blocked_additional_attempt_count == 1

    network_posts = 0

    async def network_error_handler(request: httpx.Request) -> httpx.Response:
        nonlocal network_posts
        network_posts += 1
        raise httpx.ConnectError("synthetic")

    async with httpx.AsyncClient(transport=httpx.MockTransport(network_error_handler)) as client:
        network_failure = await execute(client)
    assert network_posts == 1
    assert network_failure.external_http_outbound_count == 1
    assert network_failure.blocked_additional_attempt_count == 0

    invalid_schema_posts = 0

    async def invalid_schema_handler(request: httpx.Request) -> httpx.Response:
        nonlocal invalid_schema_posts
        invalid_schema_posts += 1
        return response({"schema_version": "wrong", "status": "RESOLVED"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(invalid_schema_handler)) as client:
        invalid_schema = await execute(client)
    assert invalid_schema_posts == 1
    assert invalid_schema.external_http_outbound_count == 1
    assert invalid_schema.extraction["status"] == "UNRESOLVED"

    bad_grounding_posts = 0

    async def bad_grounding_handler(request: httpx.Request) -> httpx.Response:
        nonlocal bad_grounding_posts
        bad_grounding_posts += 1
        return response(valid("span absent from the claim"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(bad_grounding_handler)) as client:
        bad_grounding = await execute(client)
    assert bad_grounding_posts == 1
    assert bad_grounding.external_http_outbound_count == 1
    assert bad_grounding.extraction["status"] == "INVALID"

    attachment_posts = 0

    async def missing_attachment_binding_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attachment_posts
        attachment_posts += 1
        claim = "Synthetic readiness claim. in scope"
        payload = valid(claim)
        payload["source_contexts"] = [
            {
                "context_ref": "C",
                "context_type": "SCOPE",
                "semantic_value": {"scope": "in scope"},
                "lexical_value": "in scope",
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
        payload["evidence_spans"].append({"evidence_ref": "eC", "exact_text": "in scope"})
        payload["evidence_bindings"].append(
            {
                "binding_ref": "bC",
                "evidence_ref": "eC",
                "semantic_role": "SCOPE_VALUE",
                "target_kind": "CONTEXT",
                "target_ref": "C",
            }
        )
        return response(payload)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(missing_attachment_binding_handler)
    ) as client:
        missing_attachment = await execute(client, claim_text="Synthetic readiness claim. in scope")
    assert attachment_posts == 1
    assert missing_attachment.external_http_outbound_count == 1
    assert missing_attachment.extraction["status"] == "PARTIAL"
    assert "attachment_binding_missing" in missing_attachment.extraction["diagnostics"]

    relation_posts = 0

    async def missing_relation_binding_handler(request: httpx.Request) -> httpx.Response:
        nonlocal relation_posts
        relation_posts += 1
        claim = "Synthetic readiness claim. another claim causes."
        payload = valid(claim)
        payload["source_assertions"].append(
            {"source_assertion_ref": "B", "evidence_refs": ["eB"], "assertion_status": "ASSERTED"}
        )
        payload["evidence_spans"].extend(
            [
                {"evidence_ref": "eB", "exact_text": "another claim"},
                {"evidence_ref": "eR", "exact_text": "causes"},
            ]
        )
        payload["evidence_bindings"].append(
            {
                "binding_ref": "bB",
                "evidence_ref": "eB",
                "semantic_role": "ASSERTION_EVIDENCE",
                "target_kind": "ASSERTION",
                "target_ref": "B",
            }
        )
        payload["semantic_relations"] = [
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
        ]
        return response(payload)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(missing_relation_binding_handler)
    ) as client:
        missing_relation = await execute(
            client, claim_text="Synthetic readiness claim. another claim causes."
        )
    assert relation_posts == 1
    assert missing_relation.external_http_outbound_count == 1
    assert missing_relation.extraction["status"] == "PARTIAL"
    assert "relation_binding_missing" in missing_relation.extraction["diagnostics"]

    print(
        json.dumps(
            {
                "fake_only": True,
                "provider_calls": 0,
                "external_real_http": 0,
                "checks": {
                    "v3_gateway_success": "PASS",
                    "json_mode_request": "PASS",
                    "one_outbound_hard_limit": "PASS",
                    "request_format_fallback_blocked": "PASS",
                    "endpoint_fallback_blocked": "PASS",
                    "timeout_and_network_no_retry": "PASS",
                    "schema_invalid_no_regeneration": "PASS",
                    "grounding_invalid_no_repair": "PASS",
                    "missing_attachment_binding_no_repair": "PASS",
                    "missing_relation_binding_no_repair": "PASS",
                    "secret_not_in_result_or_body": "PASS",
                },
                "simulated_transport_posts": len(seen)
                + len(fallback_requests)
                + len(endpoint_urls)
                + network_posts
                + invalid_schema_posts
                + bad_grounding_posts
                + attachment_posts
                + relation_posts,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
