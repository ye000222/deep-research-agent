"""Planner execution with a strict context and credential release boundary."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import ValidationError

from app.context.manager import ContextBudgetManager
from app.domain.context import ContextCandidate, ContextItemType
from app.domain.memory import MemoryItemView
from app.domain.planning import ResearchPlan
from app.domain.providers import (
    CanonicalModelRequest,
    ContentPart,
    TokenUsage,
    UsageAccuracy,
)
from app.infrastructure.db.run_providers import RunProviderBindingRepository
from app.llm.adapters import LLMGateway, ModelGatewayError
from app.security.secrets import SecretCipher

PLANNER_INSTRUCTIONS = """你是 DeepResearch Agent 的 Research Planner。
只负责把研究目标拆成 5 到 8 个可验证、尽量互斥且共同完备的研究问题。
每个问题必须说明研究理由、证据要求和最多三个搜索提示。
不要声称已经搜索。
不要生成答案。
不要输出内部思维过程。
严格按给定 JSON Schema 返回。"""


class PlannerService:
    def __init__(
        self,
        bindings: RunProviderBindingRepository,
        cipher: SecretCipher,
        gateway: LLMGateway,
        contexts: ContextBudgetManager | None = None,
    ) -> None:
        self._bindings = bindings
        self._cipher = cipher
        self._gateway = gateway
        self._contexts = contexts

    async def generate(
        self,
        run_id: UUID,
        *,
        memory_leads: tuple[MemoryItemView, ...] = (),
    ) -> tuple[ResearchPlan, TokenUsage]:
        binding = await self._bindings.get(run_id)
        api_key = self._cipher.decrypt(
            binding.encrypted_secret,
            credential_id=binding.credential_id,
            adapter_type=binding.adapter_type.value,
            credential_version=binding.credential_version,
        )
        task_brief = (
            f"当前日期: {datetime.now(UTC).date().isoformat()}\n"
            f"研究目标: {binding.goal}\n"
            "请定义研究范围、子问题和完成标准。"
        )
        memory_candidates = tuple(
            ContextCandidate(
                item_type=ContextItemType.MEMORY,
                content=(
                    "历史研究线索(不能直接作为本任务证据, 必须重新检索验证):\n"
                    f"{lead.content_summary}"
                ),
                rank_score=0.65 + 0.25 * lead.importance,
                source_ref_type="memory",
                source_ref_id=str(lead.memory_id),
                selected_reason_code=(
                    "cross_run_lead_revalidation_required"
                    if lead.revalidation_required
                    else "current_run_memory"
                ),
                provenance_refs=lead.source_ref_ids,
            )
            for lead in memory_leads
        )
        output_tokens = min(3000, binding.max_output_tokens or 3000)
        manifest_id = uuid4()
        model_context = task_brief
        if self._contexts is not None:
            envelope = await self._contexts.build(
                run_id=run_id,
                node_name="planner",
                provider_adapter=binding.adapter_type.value,
                model=binding.model,
                candidates=(
                    ContextCandidate(
                        item_type=ContextItemType.INSTRUCTION,
                        content=PLANNER_INSTRUCTIONS,
                        rank_score=1.0,
                        protected=True,
                        selected_reason_code="node_policy_required",
                    ),
                    ContextCandidate(
                        item_type=ContextItemType.TASK_BRIEF,
                        content=task_brief,
                        rank_score=1.0,
                        protected=True,
                        selected_reason_code="user_goal_required",
                    ),
                    ContextCandidate(
                        item_type=ContextItemType.OUTPUT_SCHEMA,
                        content=json.dumps(ResearchPlan.model_json_schema(), ensure_ascii=False),
                        rank_score=1.0,
                        protected=True,
                        selected_reason_code="output_contract_required",
                    ),
                    *memory_candidates,
                ),
                requested_output_tokens=output_tokens,
                context_window=binding.context_window,
                provider_max_output_tokens=binding.max_output_tokens,
                prompt_template_version="planner.v1",
            )
            manifest_id = envelope.manifest_id
            model_context = "\n\n".join(
                item.content
                for item in envelope.selected
                if item.item_type in {ContextItemType.TASK_BRIEF, ContextItemType.MEMORY}
            )
        elif memory_candidates:
            model_context = "\n\n".join((task_brief, *(item.content for item in memory_candidates)))
        request = CanonicalModelRequest(
            task_kind="research_planning",
            role="planner",
            model=binding.model,
            instructions=PLANNER_INSTRUCTIONS,
            content_parts=(ContentPart(kind="text", value=model_context),),
            response_contract=ResearchPlan.model_json_schema(),
            generation_parameters={"temperature": 0.2},
            max_output_tokens=output_tokens,
            context_manifest_id=manifest_id,
            metadata={"run_id": str(run_id), "node": "planner"},
        )
        result = await self._gateway.generate_structured(
            adapter_type=binding.adapter_type,
            base_url=binding.base_url,
            api_key=api_key,
            request=request,
        )
        validation_error: ValidationError | None = None
        try:
            plan = ResearchPlan.model_validate(result.parsed_object)
            return plan, result.usage
        except ValidationError as exc:
            validation_error = exc
            strategy = result.capability_strategy.get("structured_output", "")
            if strategy.endswith("_regenerated_once"):
                raise ModelGatewayError(
                    "MODEL_OUTPUT_SCHEMA_INVALID",
                    retryable=False,
                ) from exc

        if validation_error is None:  # pragma: no cover - defensive type guard
            raise ModelGatewayError("MODEL_OUTPUT_SCHEMA_INVALID", retryable=False)
        repair_request = _schema_repair_request(request, validation_error)
        repair_result = await self._gateway.generate_structured(
            adapter_type=binding.adapter_type,
            base_url=binding.base_url,
            api_key=api_key,
            request=repair_request,
            allow_regeneration=False,
        )
        usage = _combine_usage(result.usage, repair_result.usage)
        try:
            plan = ResearchPlan.model_validate(repair_result.parsed_object)
        except ValidationError as exc:
            raise ModelGatewayError("MODEL_OUTPUT_SCHEMA_INVALID", retryable=False) from exc
        return plan, usage


def _schema_repair_request(
    request: CanonicalModelRequest,
    error: ValidationError,
) -> CanonicalModelRequest:
    issues: list[str] = []
    for issue in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )[:12]:
        location = ".".join(str(part) for part in issue.get("loc", ())) or "root"
        issues.append(f"- {location}: {issue.get('type', 'validation_error')}")
    feedback = "\n".join(issues)
    instructions = (
        f"{request.instructions}\n\n"
        "上一次 JSON 未通过 Schema 校验。重新生成整个对象, 不要复用错误结构。\n"
        f"校验失败位置和类型:\n{feedback}"
    )
    return request.model_copy(
        update={
            "instructions": instructions,
            "metadata": {**request.metadata, "schema_repair": "1"},
        }
    )


def _combine_usage(first: TokenUsage, second: TokenUsage) -> TokenUsage:
    accuracy = (
        UsageAccuracy.EXACT
        if first.accuracy == second.accuracy == UsageAccuracy.EXACT
        else UsageAccuracy.UNAVAILABLE
    )
    return TokenUsage(
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        total_tokens=first.total_tokens + second.total_tokens,
        accuracy=accuracy,
    )
