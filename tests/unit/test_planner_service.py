from uuid import uuid4

import pytest
from app.domain.providers import (
    AdapterType,
    CanonicalModelResult,
    TokenUsage,
    UsageAccuracy,
)
from app.infrastructure.db.run_providers import RunProviderBinding
from app.llm.adapters import ModelGatewayError
from app.security.secrets import SecretCipher
from app.services.planner import PlannerService
from pydantic import SecretStr


def plan_payload(*, priority: int = 1) -> dict[str, object]:
    return {
        "goal": "研究工业视觉缺陷检测行业的发展情况",
        "scope_summary": "覆盖技术、厂商、产品、大模型应用和未来趋势。",
        "questions": [
            {
                "id": f"q{index}",
                "question": f"研究问题 {index} 的事实与变化是什么?",
                "priority": priority if index == 1 else 2,
                "rationale": "该维度直接影响研究目标的完整回答。",
                "evidence_requirements": ["至少两个独立来源"],
                "search_hints": [f"topic {index}"],
            }
            for index in range(1, 6)
        ],
        "completion_criteria": ["核心问题均有证据", "关键结论完成交叉验证"],
    }


def model_result(
    payload: dict[str, object],
    *,
    input_tokens: int,
    output_tokens: int,
    strategy: str = "json_mode",
) -> CanonicalModelResult:
    return CanonicalModelResult(
        parsed_object=payload,
        usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            accuracy=UsageAccuracy.EXACT,
        ),
        capability_strategy={"structured_output": strategy},
    )


class FakeBindingRepository:
    def __init__(self, binding: RunProviderBinding) -> None:
        self.binding = binding

    async def get(self, run_id: object) -> RunProviderBinding:
        return self.binding


class FakeGateway:
    def __init__(self, results: list[CanonicalModelResult]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    async def generate_structured(self, **kwargs: object) -> CanonicalModelResult:
        self.calls.append(kwargs)
        return self.results[len(self.calls) - 1]


def dependencies() -> tuple[RunProviderBinding, SecretCipher]:
    run_id = uuid4()
    credential_id = uuid4()
    cipher = SecretCipher(b"x" * 32)
    encrypted = cipher.encrypt(
        SecretStr("test-provider-key"),
        credential_id=credential_id,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT.value,
        credential_version=1,
    )
    return (
        RunProviderBinding(
            run_id=run_id,
            goal="研究工业视觉缺陷检测行业的发展情况",
            adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
            base_url="https://api.deepseek.com",
            model="test-model",
            credential_id=credential_id,
            credential_version=1,
            encrypted_secret=encrypted,
        ),
        cipher,
    )


@pytest.mark.asyncio
async def test_planner_regenerates_schema_invalid_plan_once() -> None:
    binding, cipher = dependencies()
    gateway = FakeGateway(
        [
            model_result(plan_payload(priority=9), input_tokens=10, output_tokens=20),
            model_result(plan_payload(), input_tokens=12, output_tokens=24),
        ]
    )
    service = PlannerService(  # type: ignore[arg-type]
        FakeBindingRepository(binding),
        cipher,
        gateway,
    )

    plan, usage = await service.generate(binding.run_id)

    assert len(gateway.calls) == 2
    assert gateway.calls[1]["allow_regeneration"] is False
    repair_request = gateway.calls[1]["request"]
    assert repair_request.metadata["schema_repair"] == "1"  # type: ignore[union-attr]
    assert "questions.0.priority" in repair_request.instructions  # type: ignore[union-attr]
    assert "less_than_equal" in repair_request.instructions  # type: ignore[union-attr]
    assert plan.questions[0].priority == 1
    assert usage.input_tokens == 22
    assert usage.output_tokens == 44
    assert usage.total_tokens == 66


@pytest.mark.asyncio
async def test_planner_never_exceeds_two_calls_after_json_regeneration() -> None:
    binding, cipher = dependencies()
    gateway = FakeGateway(
        [
            model_result(
                plan_payload(priority=9),
                input_tokens=10,
                output_tokens=20,
                strategy="json_mode_regenerated_once",
            )
        ]
    )
    service = PlannerService(  # type: ignore[arg-type]
        FakeBindingRepository(binding),
        cipher,
        gateway,
    )

    with pytest.raises(ModelGatewayError, match="MODEL_OUTPUT_SCHEMA_INVALID"):
        await service.generate(binding.run_id)

    assert len(gateway.calls) == 1
