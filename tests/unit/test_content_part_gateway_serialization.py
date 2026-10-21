from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

import httpx
import pytest
from app.domain.atomic_proposition import ATOMIC_PROPOSITION_SCHEMA_VERSION
from app.domain.providers import AdapterType, CanonicalModelRequest, ContentPart
from app.llm.adapters import LLMGateway, ModelGatewayError, _content_text
from app.services.atomic_proposition_decomposer import AtomicPropositionDecomposer
from pydantic import SecretStr, ValidationError

from scripts.run_v2_1r_shadow_canary import (
    CanaryDecomposer,
    CanaryGateway,
    CanaryRequest,
    build_canary_http_client,
    run_one_claim_canary,
)

_CLAIM = (
    "文章预测2026年多模态VLM模型将全面超越CNN和YOLO系列，"  # noqa: RUF001
    "“零样本”工业缺陷检测成为最新范式，四张参考样本准确率可达99%。"  # noqa: RUF001
)


def _request(*parts: ContentPart) -> CanonicalModelRequest:
    return CanonicalModelRequest(
        task_kind="content_part_serialization_test",
        role="test",
        model="synthetic-model",
        instructions="Return a JSON object.",
        content_parts=parts,
        response_contract={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        },
        max_output_tokens=32,
        context_manifest_id=uuid4(),
    )


def _fake_post(request: CanonicalModelRequest) -> tuple[dict[str, Any], int]:
    bodies: list[dict[str, Any]] = []

    async def responder(http_request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads((await http_request.aread()).decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "synthetic-request",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": '{"ok":true}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    async def invoke() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(responder), trust_env=False
        ) as client:
            await LLMGateway(client).generate_structured(
                adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
                base_url="https://synthetic.invalid/v1",
                api_key=SecretStr("synthetic-only"),
                request=request,
                allow_regeneration=False,
            )

    asyncio.run(invoke())
    assert len(bodies) == 1
    return bodies[0], len(bodies)


def test_json_contentpart_is_serialized_as_model_visible_unicode_json() -> None:
    sentinel = "CLAIM_PAYLOAD_SENTINEL_7F3A9C"
    payload = {
        "claim_text": sentinel,
        "context_metadata": {"sample": "多语言 Ω тест", "nested": {"values": [1, True, None]}},
    }
    body, count = _fake_post(_request(ContentPart(kind="json", value=payload)))
    user_content = body["messages"][1]["content"]

    assert count == 1
    assert user_content
    assert json.loads(user_content) == payload
    assert sentinel in user_content
    assert "多语言 Ω тест" in user_content
    assert body["response_format"] == {"type": "json_object"}


def test_real_claim_reaches_final_http_user_content_offline() -> None:
    bodies: list[dict[str, Any]] = []
    model_output = {
        "schema_version": ATOMIC_PROPOSITION_SCHEMA_VERSION,
        "decomposition_status": "unresolved_decomposition",
        "propositions": [],
        "ambiguity_reason": "synthetic offline response",
    }

    async def responder(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads((await request.aread()).decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "synthetic-request",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": json.dumps(model_output)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    async def invoke() -> None:
        client, transport = build_canary_http_client(
            timeout=httpx.Timeout(5), delegate_transport=httpx.MockTransport(responder)
        )
        try:
            gateway = CanaryGateway(LLMGateway(client), transport_diagnostics=transport)
            decomposer = AtomicPropositionDecomposer(
                gateway,
                adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
                base_url="https://synthetic.invalid/v1",
                api_key=SecretStr("synthetic-only"),
                model="synthetic-model",
                provider_profile_id="synthetic-profile",
                max_calls=1,
            )
            request = CanaryRequest(
                claim_id="synthetic-claim-id",
                claim_text=_CLAIM,
                provider_profile_id="synthetic-profile",
                profile_ready=True,
                credential_ready=True,
                schema_version=ATOMIC_PROPOSITION_SCHEMA_VERSION,
                maximum_calls=1,
                holdout_exclusion_established=True,
                excluded_claim_ids=frozenset(),
                context_metadata={},
            )
            result = await run_one_claim_canary(
                request, decomposer=CanaryDecomposer(decomposer, gateway)
            )
            assert result.status == "COMPLETED"
            assert transport.outbound_request_count == 1
        finally:
            await client.aclose()

    asyncio.run(invoke())
    assert len(bodies) == 1
    user_content = bodies[0]["messages"][1]["content"]
    parsed = json.loads(user_content)
    assert parsed["claim_text"] == _CLAIM
    assert parsed["context_metadata"] == {}
    assert _CLAIM in user_content


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"nested": {"items": ["Ω", 7, False, None]}},
        [],
        ["多语言", {"enabled": True}],
        "a JSON string Ω",
        0,
        -3,
        3.25,
        True,
        False,
        None,
    ],
)
def test_json_values_have_deterministic_valid_serialization(value: Any) -> None:
    part = ContentPart(kind="json", value=value)
    expected = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    serialized = _content_text(_request(part))

    assert serialized == expected
    assert json.loads(serialized) == value


def test_mixed_text_json_text_parts_preserve_order_in_gateway_request() -> None:
    payload = {"middle": "JSON Ω", "ordinal": 2}
    request = _request(
        ContentPart(kind="text", value="TEXT_A"),
        ContentPart(kind="json", value=payload),
        ContentPart(kind="text", value="TEXT_C"),
    )
    expected_json = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )

    serialized = _content_text(request)
    body, count = _fake_post(request)
    assert serialized == f"TEXT_A\n\n{expected_json}\n\nTEXT_C"
    assert serialized.index("TEXT_A") < serialized.index("JSON Ω") < serialized.index("TEXT_C")
    assert count == 1
    assert body["messages"][1]["content"] == serialized


def test_text_only_single_multiple_and_structured_requests_remain_unchanged() -> None:
    single = _request(ContentPart(kind="text", value="one text part"))
    multiple = _request(
        ContentPart(kind="text", value="first"),
        ContentPart(kind="text", value="second"),
    )

    assert _content_text(single) == "one text part"
    assert _content_text(multiple) == "first\n\nsecond"
    body, count = _fake_post(single)
    assert count == 1
    assert body["messages"][1]["content"] == "one text part"
    assert body["response_format"] == {"type": "json_object"}


def test_json_only_content_no_longer_becomes_empty_user_message() -> None:
    body, count = _fake_post(_request(ContentPart(kind="json", value={"claim_text": "x"})))
    assert count == 1
    assert body["messages"][1]["content"] == '{"claim_text":"x"}'


def test_empty_serialized_content_fails_before_http_transport() -> None:
    calls = 0

    async def responder(_: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    async def invoke() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as client:
            await LLMGateway(client).generate_structured(
                adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
                base_url="https://synthetic.invalid/v1",
                api_key=SecretStr("synthetic-only"),
                request=_request(ContentPart(kind="text", value=" \n\t ")),
            )

    with pytest.raises(ModelGatewayError, match="MODEL_REQUEST_INVALID"):
        asyncio.run(invoke())
    assert calls == 0


def test_unknown_content_kind_is_rejected_explicitly() -> None:
    with pytest.raises(ValidationError):
        ContentPart(kind="image", value="unsupported")

    malformed_part = ContentPart.model_construct(kind="image", value="unsupported")
    with pytest.raises(ModelGatewayError, match="MODEL_REQUEST_INVALID"):
        _content_text(_request(malformed_part))
