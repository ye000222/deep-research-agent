import json
from uuid import uuid4

import httpx
import pytest
import respx
from app.domain.planning import ResearchPlan
from app.domain.providers import AdapterType, CanonicalModelRequest, ContentPart
from app.llm.adapters import LLMGateway, ModelGatewayError
from pydantic import SecretStr

PLAN = {
    "goal": "研究工业视觉缺陷检测行业的发展情况",
    "scope_summary": "覆盖技术、厂商、产品、模型应用和未来趋势。",
    "questions": [
        {
            "id": f"q{index}",
            "question": f"研究问题 {index} 的当前事实和变化是什么?",
            "priority": 1 if index < 3 else 2,
            "rationale": "该维度直接影响研究目标的完整回答。",
            "evidence_requirements": ["至少两个独立来源"],
            "search_hints": [f"research topic {index}"],
        }
        for index in range(1, 6)
    ],
    "completion_criteria": ["核心问题均有证据", "关键结论完成交叉验证"],
}


def request() -> CanonicalModelRequest:
    return CanonicalModelRequest(
        task_kind="research_planning",
        role="planner",
        model="test-model",
        instructions="Return a rigorous research plan.",
        content_parts=(ContentPart(kind="text", value="Research this topic"),),
        response_contract=ResearchPlan.model_json_schema(),
        max_output_tokens=1800,
        context_manifest_id=uuid4(),
    )


@pytest.mark.asyncio
@respx.mock
async def test_openai_responses_adapter_uses_native_schema() -> None:
    route = respx.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "resp_test",
                "output_text": json.dumps(PLAN, ensure_ascii=False),
                "usage": {"input_tokens": 30, "output_tokens": 60},
            },
        )
    )
    async with httpx.AsyncClient() as client:
        result = await LLMGateway(client).generate_structured(
            adapter_type=AdapterType.OPENAI_RESPONSES,
            base_url="https://api.openai.com/v1",
            api_key=SecretStr("openai-secret"),
            request=request(),
        )

    sent = json.loads(route.calls[0].request.content)
    assert route.calls[0].request.headers["authorization"] == "Bearer openai-secret"
    assert sent["text"]["format"]["type"] == "json_schema"
    assert sent["store"] is False
    assert ResearchPlan.model_validate(result.parsed_object).questions[0].id == "q1"
    assert result.usage.total_tokens == 90


@pytest.mark.asyncio
@respx.mock
async def test_anthropic_adapter_uses_output_config() -> None:
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_test",
                "content": [{"type": "text", "text": json.dumps(PLAN)}],
                "usage": {"input_tokens": 20, "output_tokens": 50},
            },
        )
    )
    async with httpx.AsyncClient() as client:
        result = await LLMGateway(client).generate_structured(
            adapter_type=AdapterType.ANTHROPIC_MESSAGES,
            base_url="https://api.anthropic.com/v1",
            api_key=SecretStr("anthropic-secret"),
            request=request(),
        )

    sent = json.loads(route.calls[0].request.content)
    assert route.calls[0].request.headers["x-api-key"] == "anthropic-secret"
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert ResearchPlan.model_validate(result.parsed_object).goal == PLAN["goal"]


@pytest.mark.asyncio
@respx.mock
async def test_gemini_adapter_uses_header_key_and_response_schema() -> None:
    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/test-model:generateContent"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "responseId": "gemini_test",
                "candidates": [{"content": {"parts": [{"text": json.dumps(PLAN)}]}}],
                "usageMetadata": {
                    "promptTokenCount": 25,
                    "candidatesTokenCount": 55,
                    "totalTokenCount": 80,
                },
            },
        )
    )
    async with httpx.AsyncClient() as client:
        result = await LLMGateway(client).generate_structured(
            adapter_type=AdapterType.GOOGLE_GEMINI,
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key=SecretStr("gemini-secret"),
            request=request(),
        )

    sent = json.loads(route.calls[0].request.content)
    assert route.calls[0].request.headers["x-goog-api-key"] == "gemini-secret"
    assert "key=" not in str(route.calls[0].request.url)
    assert sent["generationConfig"]["responseMimeType"] == "application/json"
    assert ResearchPlan.model_validate(result.parsed_object).questions[-1].id == "q5"


@pytest.mark.asyncio
@respx.mock
async def test_compatible_adapter_falls_back_when_json_mode_is_rejected() -> None:
    route = respx.post("https://api.deepseek.com/chat/completions").mock(
        side_effect=[
            httpx.Response(400, json={"error": "unsupported response_format"}),
            httpx.Response(
                200,
                json={
                    "id": "chat_test",
                    "choices": [{"message": {"content": json.dumps(PLAN)}}],
                    "usage": {
                        "prompt_tokens": 22,
                        "completion_tokens": 48,
                        "total_tokens": 70,
                    },
                },
            ),
        ]
    )
    async with httpx.AsyncClient() as client:
        result = await LLMGateway(client).generate_structured(
            adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
            base_url="https://api.deepseek.com",
            api_key=SecretStr("compatible-secret"),
            request=request(),
        )

    assert route.call_count == 2
    assert route.calls[0].request.headers["connection"] == "close"
    second_body = json.loads(route.calls[1].request.content)
    assert "response_format" not in second_body
    assert "JSON Schema" in second_body["messages"][0]["content"]
    assert '"completion_criteria"' in second_body["messages"][0]["content"]
    assert result.capability_strategy["structured_output"] == "prompt_json"


@pytest.mark.asyncio
@respx.mock
async def test_compatible_adapter_extracts_json_surrounded_by_explanation() -> None:
    respx.post("https://api.deepseek.com/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chat_explained",
                "choices": [
                    {
                        "message": {
                            "content": (
                                "下面是研究计划:\n```json\n"
                                + json.dumps(PLAN, ensure_ascii=False)
                                + "\n```\n请查收。"
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 50},
            },
        )
    )
    async with httpx.AsyncClient() as client:
        result = await LLMGateway(client).generate_structured(
            adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
            base_url="https://api.deepseek.com",
            api_key=SecretStr("compatible-secret"),
            request=request(),
        )

    assert ResearchPlan.model_validate(result.parsed_object).goal == PLAN["goal"]


@pytest.mark.asyncio
@respx.mock
async def test_compatible_adapter_regenerates_invalid_json_once_and_counts_usage() -> None:
    route = respx.post("https://api.deepseek.com/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "id": "chat_invalid",
                    "choices": [{"message": {"content": "not-json"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            ),
            httpx.Response(
                200,
                json={
                    "id": "chat_repaired",
                    "choices": [{"message": {"content": json.dumps(PLAN)}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 40},
                },
            ),
        ]
    )
    async with httpx.AsyncClient() as client:
        result = await LLMGateway(client).generate_structured(
            adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
            base_url="https://api.deepseek.com",
            api_key=SecretStr("compatible-secret"),
            request=request(),
        )

    assert route.call_count == 2
    retry_body = json.loads(route.calls[1].request.content)
    assert "前一次响应无法解析" in retry_body["messages"][1]["content"]
    assert result.usage.input_tokens == 22
    assert result.usage.output_tokens == 45
    assert result.usage.total_tokens == 67
    assert result.capability_strategy["structured_output"] == "json_mode_regenerated_once"


@pytest.mark.asyncio
@respx.mock
async def test_network_error_exposes_only_safe_detail_code() -> None:
    respx.post("https://api.deepseek.com/chat/completions").mock(
        side_effect=httpx.RemoteProtocolError("diagnostic transport message")
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(ModelGatewayError) as raised:
            await LLMGateway(client).generate_structured(
                adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
                base_url="https://api.deepseek.com",
                api_key=SecretStr("compatible-secret"),
                request=request(),
            )

    assert raised.value.code == "MODEL_NETWORK_ERROR"
    assert raised.value.detail_code == "REMOTE_PROTOCOL_ERROR"
    assert raised.value.retryable is True


@pytest.mark.asyncio
@respx.mock
async def test_invalid_model_output_is_rejected_before_orchestration() -> None:
    respx.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(200, json={"output_text": "not-json"})
    )
    async with httpx.AsyncClient() as client:
        with pytest.raises(ModelGatewayError, match="MODEL_OUTPUT_INVALID"):
            await LLMGateway(client).generate_structured(
                adapter_type=AdapterType.OPENAI_RESPONSES,
                base_url="https://api.openai.com/v1",
                api_key=SecretStr("secret-never-returned"),
                request=request(),
            )
