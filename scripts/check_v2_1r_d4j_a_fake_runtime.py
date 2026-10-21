"""Same-runtime Fake-only verification for the D4-J-A v3.1 runner."""

# ruff: noqa: E501

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from app.domain.providers import AdapterType
from app.services.source_context_extractor_v3_1 import SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31
from pydantic import SecretStr
from v2_1r_d4j_a_canary_runner import execute_one_claim_v31


def response(payload: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={
        "id": "fake-d4j-a", "choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
    })


def valid(claim: str = "Anchor P.") -> dict[str, Any]:
    return {
        "schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31,
        "status": "RESOLVED",
        "source_assertions": [{"source_assertion_ref": "A", "evidence_refs": ["eA"], "assertion_status": "ASSERTED"}],
        "source_contexts": [], "source_attachments": [], "source_scopes": [],
        "evidence_spans": [{"evidence_ref": "eA", "exact_text": claim}],
        "evidence_bindings": [{"binding_ref": "bA", "evidence_ref": "eA", "semantic_role": "ASSERTION_EVIDENCE", "target_kind": "ASSERTION", "target_ref": "A"}],
        "semantic_relations": [], "measurement_bases": [], "unresolved_items": [],
    }


async def invoke(client: httpx.AsyncClient, *, base_url: str = "https://api.deepseek.com/v1", claim: str = "Anchor P."):
    return await execute_one_claim_v31(
        http_client=client, claim_text=claim, claim_ref="fake-only",
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT, base_url=base_url,
        api_key=SecretStr("FAKE_D4J_SECRET"), model="deepseek-flash",
    )


async def run() -> dict[str, Any]:
    checks: dict[str, str] = {}
    posts = 0

    async def success_handler(request: httpx.Request) -> httpx.Response:
        nonlocal posts
        posts += 1
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert "FAKE_D4J_SECRET" not in request.content.decode()
        return response(valid())

    async with httpx.AsyncClient(transport=httpx.MockTransport(success_handler)) as client:
        success = await invoke(client)
    assert posts == 1 and success.external_http_outbound_count == 1
    assert success.extraction["status"] == "RESOLVED"
    checks["success_exactly_one_outbound"] = "PASS"
    checks["json_mode_request"] = "PASS"
    checks["secret_excluded"] = "PASS"

    calls = 0
    async def count_handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return response(valid())
    async with httpx.AsyncClient(transport=httpx.MockTransport(count_handler)) as client:
        guarded = await invoke(client)
        from app.llm.canary_strict import SingleOutboundClient
        raw = httpx.AsyncClient(transport=httpx.MockTransport(count_handler))
        guard = SingleOutboundClient(raw)
        await guard.post("https://example.invalid", headers={}, json={})
        try:
            await guard.post("https://example.invalid", headers={}, json={})
        except Exception:
            pass
        else:
            raise AssertionError("second outbound was not blocked")
        await raw.aclose()
    assert guarded.external_http_outbound_count == 1 and calls == 2
    checks["second_attempt_blocked_before_dispatch"] = "PASS"

    for label, exception in (("timeout_no_retry", httpx.ReadTimeout("fake")), ("network_error_no_retry", httpx.ConnectError("fake"))):
        attempt_posts = 0
        async def failure_handler(request: httpx.Request, *, _exc=exception) -> httpx.Response:
            nonlocal attempt_posts
            attempt_posts += 1
            raise _exc
        async with httpx.AsyncClient(transport=httpx.MockTransport(failure_handler)) as client:
            failed = await invoke(client)
        assert attempt_posts == 1 and failed.external_http_outbound_count == 1
        checks[label] = "PASS"

    invalid_posts = 0
    async def schema_handler(request: httpx.Request) -> httpx.Response:
        nonlocal invalid_posts
        invalid_posts += 1
        return response({"schema_version": "wrong", "status": "RESOLVED"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(schema_handler)) as client:
        invalid = await invoke(client)
    assert invalid_posts == 1 and invalid.external_http_outbound_count == 1
    assert invalid.extraction["status"] == "UNRESOLVED"
    checks["schema_invalid_no_regeneration"] = "PASS"

    role_claim = "Metric 7 units."
    wrong_role = valid(role_claim)
    wrong_role["source_contexts"] = [{"context_ref": "M", "context_type": "MEASUREMENT", "semantic_value": {"measurement_identity": "Metric"}, "lexical_value": "Metric", "materiality": "MATERIAL"}]
    wrong_role["source_attachments"] = [{"attachment_ref": "AT", "context_ref": "M", "source_assertion_refs": ["A"], "relation": "APPLIES_TO", "status": "RESOLVED", "evidence_refs": ["eA"]}]
    wrong_role["evidence_bindings"].extend([
        {"binding_ref": "bM", "evidence_ref": "eA", "semantic_role": "MEASUREMENT_DEFINITION", "target_kind": "CONTEXT", "target_ref": "M"},
        {"binding_ref": "bAT", "evidence_ref": "eA", "semantic_role": "ATTACHMENT", "target_kind": "ATTACHMENT", "target_ref": "AT"},
    ])
    role_posts = 0
    async def role_handler(request: httpx.Request) -> httpx.Response:
        nonlocal role_posts
        role_posts += 1
        return response(wrong_role)
    async with httpx.AsyncClient(transport=httpx.MockTransport(role_handler)) as client:
        role_result = await invoke(client, claim=role_claim)
    assert role_posts == 1 and role_result.external_http_outbound_count == 1
    checks["wrong_measurement_role_no_repair"] = "PASS"

    missing_attachment = json.loads(json.dumps(wrong_role))
    missing_attachment["evidence_bindings"] = [
        missing_attachment["evidence_bindings"][0],
        {"binding_ref": "bM", "evidence_ref": "eA", "semantic_role": "MEASUREMENT_IDENTITY", "target_kind": "CONTEXT", "target_ref": "M"},
    ]
    missing_posts = 0
    async def missing_handler(request: httpx.Request) -> httpx.Response:
        nonlocal missing_posts
        missing_posts += 1
        return response(missing_attachment)
    async with httpx.AsyncClient(transport=httpx.MockTransport(missing_handler)) as client:
        missing_result = await invoke(client, claim=role_claim)
    assert missing_posts == 1 and missing_result.external_http_outbound_count == 1
    assert missing_result.extraction["status"] == "INVALID"
    assert "attachment_binding_missing" in missing_result.extraction["diagnostics"]
    checks["missing_attachment_binding_no_repair"] = "PASS"

    scope_claim = "Anchor P."
    bad_scope = valid(scope_claim)
    bad_scope["source_scopes"] = [{"source_scope_ref": "S", "kind": "CLAIM", "source_assertion_refs": ["MISSING"], "evidence_refs": ["eA"], "resolution_status": "RESOLVED"}]
    scope_posts = 0
    async def scope_handler(request: httpx.Request) -> httpx.Response:
        nonlocal scope_posts
        scope_posts += 1
        return response(bad_scope)
    async with httpx.AsyncClient(transport=httpx.MockTransport(scope_handler)) as client:
        scope_result = await invoke(client)
    assert scope_posts == 1 and scope_result.external_http_outbound_count == 1
    assert scope_result.extraction["status"] == "INVALID"
    checks["invalid_scope_ref_no_repair"] = "PASS"

    # DeepSeek endpoint fallback is attempted by the canonical Gateway but blocked by the wrapper.
    fallback_urls: list[str] = []
    async def fallback_handler(request: httpx.Request) -> httpx.Response:
        fallback_urls.append(str(request.url))
        return httpx.Response(400, json={"error": {"message": "fake"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(fallback_handler)) as client:
        fallback = await invoke(client, base_url="https://api.deepseek.com")
    assert len(fallback_urls) == 1 and fallback.external_http_outbound_count == 1
    assert fallback.blocked_additional_attempt_count == 1
    checks["endpoint_fallback_blocked"] = "PASS"

    format_urls: list[str] = []
    async def format_handler(request: httpx.Request) -> httpx.Response:
        format_urls.append(str(request.url))
        return httpx.Response(400, json={"error": {"message": "fake"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(format_handler)) as client:
        format_fallback = await invoke(client)
    assert len(format_urls) == 1 and format_fallback.external_http_outbound_count == 1
    assert format_fallback.blocked_additional_attempt_count == 1
    checks["request_format_fallback_blocked"] = "PASS"

    return {"runtime_fake_only": True, "provider_calls": 0, "real_external_http": 0, "db_writes": 0, "checks": checks, "fake_transport_requests": posts + calls + 2 + invalid_posts + role_posts + missing_posts + scope_posts + len(fallback_urls) + len(format_urls)}


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), sort_keys=True))
