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
        "scope_summary": "覆盖技术、厂商、产品、大模型应用和未来趋势。",
        "questions": [
            {
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
    finish_reason: str | None = "stop",
) -> CanonicalModelResult:
    return CanonicalModelResult(
        parsed_object=payload,
        usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            accuracy=UsageAccuracy.EXACT,
        ),
        finish_reason=finish_reason,
        capability_strategy={"structured_output": strategy},
    )


class FakeBindingRepository:
    def __init__(self, binding: RunProviderBinding) -> None:
        self.binding = binding

    async def get(self, run_id: object) -> RunProviderBinding:
        return self.binding


class FakeGateway:
    def __init__(self, results: list[CanonicalModelResult | ModelGatewayError]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    async def generate_structured(self, **kwargs: object) -> CanonicalModelResult:
        self.calls.append(kwargs)
        result = self.results[len(self.calls) - 1]
        if isinstance(result, ModelGatewayError):
            raise result
        return result


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
    first_request = gateway.calls[0]["request"]
    assert first_request.generation_parameters["temperature"] == 0.0  # type: ignore[union-attr]
    assert first_request.max_output_tokens == 4000  # type: ignore[union-attr]
    assert "goal" not in first_request.response_contract["properties"]  # type: ignore[union-attr,index]
    assert gateway.calls[0]["allow_regeneration"] is False
    assert gateway.calls[1]["allow_regeneration"] is False
    repair_request = gateway.calls[1]["request"]
    assert repair_request.metadata["retry_mode"] == "compact_invalid_or_schema"  # type: ignore[union-attr]
    assert repair_request.max_output_tokens == 1200  # type: ignore[union-attr]
    assert repair_request.response_contract["properties"]["questions"]["maxItems"] == 5  # type: ignore[union-attr,index]
    assert plan.questions[0].priority == 1
    assert plan.goal == binding.goal
    assert [item.id for item in plan.questions] == ["q1", "q2", "q3", "q4", "q5"]
    assert usage.input_tokens == 22
    assert usage.output_tokens == 44
    assert usage.total_tokens == 66


@pytest.mark.asyncio
async def test_planner_never_exceeds_two_calls_after_schema_failure() -> None:
    binding, cipher = dependencies()
    gateway = FakeGateway(
        [
            model_result(
                plan_payload(priority=9),
                input_tokens=10,
                output_tokens=20,
                strategy="json_mode",
            ),
            model_result(
                plan_payload(priority=9),
                input_tokens=12,
                output_tokens=24,
                strategy="json_mode",
            ),
        ]
    )
    service = PlannerService(  # type: ignore[arg-type]
        FakeBindingRepository(binding),
        cipher,
        gateway,
    )

    with pytest.raises(
        ModelGatewayError,
        match="MODEL_OUTPUT_SCHEMA_INVALID",
    ) as raised:
        await service.generate(binding.run_id)

    assert len(gateway.calls) == 2
    assert raised.value.detail_code is not None
    assert raised.value.detail_code.startswith(
        "SCHEMA_INVALID_JSON_MODE_FINISH_STOP_ISSUES_"
    )


@pytest.mark.asyncio
async def test_planner_uses_independent_compact_contract_after_length() -> None:
    binding, cipher = dependencies()
    first_usage = TokenUsage(
        input_tokens=40,
        output_tokens=4000,
        total_tokens=4040,
        accuracy=UsageAccuracy.EXACT,
    )
    gateway = FakeGateway(
        [
            ModelGatewayError(
                "MODEL_OUTPUT_TRUNCATED",
                retryable=False,
                detail_code="OUTPUT_INVALID_JSON_MODE_FINISH_LENGTH",
                usage=first_usage,
                diagnostics={
                    "finish_reason": "length",
                    "output_tokens": 4000,
                    "max_output_tokens": 4000,
                },
            ),
            model_result(plan_payload(), input_tokens=20, output_tokens=700),
        ]
    )
    service = PlannerService(  # type: ignore[arg-type]
        FakeBindingRepository(binding),
        cipher,
        gateway,
    )

    plan, usage = await service.generate(binding.run_id)

    assert len(gateway.calls) == 2
    compact_request = gateway.calls[1]["request"]
    assert compact_request.task_kind == "research_planning_compact"  # type: ignore[union-attr]
    assert compact_request.metadata["retry_mode"] == "compact_length"  # type: ignore[union-attr]
    assert compact_request.max_output_tokens == 1200  # type: ignore[union-attr]
    assert "goal" not in compact_request.response_contract["properties"]  # type: ignore[union-attr,index]
    question_schema = compact_request.response_contract["$defs"][  # type: ignore[union-attr,index]
        "CompactResearchQuestionDraft"
    ]
    assert "id" not in question_schema["properties"]
    assert plan.goal == binding.goal
    assert plan.questions[4].id == "q5"
    assert usage.output_tokens == 4700
    assert usage.total_tokens == 4760


@pytest.mark.asyncio
async def test_compact_length_failure_becomes_plan_budget_error() -> None:
    binding, cipher = dependencies()
    first = ModelGatewayError(
        "MODEL_OUTPUT_TRUNCATED",
        retryable=False,
        usage=TokenUsage(
            input_tokens=10,
            output_tokens=4000,
            total_tokens=4010,
            accuracy=UsageAccuracy.EXACT,
        ),
        diagnostics={"finish_reason": "length"},
    )
    second = ModelGatewayError(
        "MODEL_OUTPUT_TRUNCATED",
        retryable=False,
        detail_code="OUTPUT_INVALID_JSON_MODE_FINISH_LENGTH",
        usage=TokenUsage(
            input_tokens=12,
            output_tokens=1200,
            total_tokens=1212,
            accuracy=UsageAccuracy.EXACT,
        ),
        diagnostics={
            "structured_output_strategy": "json_mode",
            "finish_reason": "length",
            "output_tokens": 1200,
            "max_output_tokens": 1200,
        },
    )
    service = PlannerService(  # type: ignore[arg-type]
        FakeBindingRepository(binding),
        cipher,
        FakeGateway([first, second]),
    )

    with pytest.raises(ModelGatewayError, match="PLAN_OUTPUT_BUDGET_EXCEEDED") as raised:
        await service.generate(binding.run_id)

    assert raised.value.diagnostics["retry_mode"] == "compact_length"
    assert raised.value.diagnostics["output_tokens"] == 1200
    assert raised.value.usage is not None
    assert raised.value.usage.output_tokens == 5200


@pytest.mark.asyncio
async def test_compact_transport_timeout_remains_retryable() -> None:
    binding, cipher = dependencies()
    first = ModelGatewayError(
        "MODEL_OUTPUT_TRUNCATED",
        retryable=False,
        usage=TokenUsage(
            input_tokens=10,
            output_tokens=4000,
            total_tokens=4010,
            accuracy=UsageAccuracy.EXACT,
        ),
        diagnostics={
            "structured_output_strategy": "json_mode",
            "finish_reason": "length",
        },
    )
    timeout = ModelGatewayError(
        "MODEL_TIMEOUT",
        retryable=True,
        detail_code="CONNECT_TIMEOUT",
    )
    service = PlannerService(  # type: ignore[arg-type]
        FakeBindingRepository(binding),
        cipher,
        FakeGateway([first, timeout]),
    )

    with pytest.raises(ModelGatewayError, match="MODEL_TIMEOUT") as raised:
        await service.generate(binding.run_id)

    assert raised.value.retryable is True
    assert raised.value.detail_code == "CONNECT_TIMEOUT"
    assert raised.value.diagnostics["retry_mode"] == "compact_transport_retry"
    assert raised.value.diagnostics["attempt_stage"] == "planner_compact"
    assert raised.value.diagnostics["compact_trigger"] == "compact_length"
    assert raised.value.diagnostics["first_error_code"] == "MODEL_OUTPUT_TRUNCATED"
    assert raised.value.usage is not None
    assert raised.value.usage.output_tokens == 4000


@pytest.mark.asyncio
async def test_compact_plan_deterministically_clips_length_overflow() -> None:
    binding, cipher = dependencies()
    first = ModelGatewayError(
        "MODEL_OUTPUT_TRUNCATED",
        retryable=False,
        usage=TokenUsage(
            input_tokens=10,
            output_tokens=4000,
            total_tokens=4010,
            accuracy=UsageAccuracy.EXACT,
        ),
    )
    oversized = plan_payload()
    oversized["scope_summary"] = "范围" * 100
    oversized["completion_criteria"] = ["标准" * 80, "核验" * 80, "额外标准"]
    questions = oversized["questions"]
    assert isinstance(questions, list)
    for index, question in enumerate(questions):
        assert isinstance(question, dict)
        question["question"] = f"问题{index}" + "很长" * 80
        question["rationale"] = "理由" * 80
        question["evidence_requirements"] = ["证据要求" * 40, "多余要求"]
        question["search_hints"] = ["搜索提示" * 40, "多余提示"]
    questions.append(dict(questions[0]))
    gateway = FakeGateway(
        [
            first,
            model_result(oversized, input_tokens=20, output_tokens=651),
        ]
    )
    service = PlannerService(  # type: ignore[arg-type]
        FakeBindingRepository(binding),
        cipher,
        gateway,
    )

    plan, usage = await service.generate(binding.run_id)

    assert len(gateway.calls) == 2
    assert len(plan.questions) == 5
    assert len(plan.scope_summary) == 100
    assert len(plan.completion_criteria) == 2
    assert all(len(item) <= 60 for item in plan.completion_criteria)
    assert all(len(question.question) <= 100 for question in plan.questions)
    assert all(len(question.rationale) <= 50 for question in plan.questions)
    assert all(len(question.evidence_requirements) == 1 for question in plan.questions)
    assert all(len(question.evidence_requirements[0]) <= 50 for question in plan.questions)
    assert all(len(question.search_hints) == 1 for question in plan.questions)
    assert all(len(question.search_hints[0]) <= 50 for question in plan.questions)
    assert usage.output_tokens == 4651
