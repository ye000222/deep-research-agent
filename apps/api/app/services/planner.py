"""Planner execution with a strict context and credential release boundary."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import ValidationError

from app.context.manager import ContextBudgetManager
from app.domain.context import ContextCandidate, ContextItemType
from app.domain.memory import MemoryItemView
from app.domain.planning import (
    CompactResearchPlanDraft,
    ResearchPlan,
    ResearchPlanDraft,
    materialize_research_plan,
    normalize_research_plan_draft_payload,
)
from app.domain.providers import (
    CanonicalModelRequest,
    CanonicalModelResult,
    ContentPart,
    TokenUsage,
    UsageAccuracy,
)
from app.infrastructure.db.run_providers import (
    RunProviderBinding,
    RunProviderBindingRepository,
)
from app.llm.adapters import LLMGateway, ModelGatewayError
from app.security.secrets import SecretCipher

PLANNER_INSTRUCTIONS = """你是 DeepResearch Agent 的 Research Planner。
只负责把研究目标拆成 5 到 8 个可验证、尽量互斥且共同完备的研究问题。
严格控制长度: scope_summary 最多 160 字; question 最多 120 字; rationale 最多 80 字;
evidence_requirements 每题 1 到 2 条且每条最多 60 字;
search_hints 每题 1 到 2 条且每条最多 80 字;
completion_criteria 2 到 4 条且每条最多 60 字。
不要输出 goal 或 question id; 它们由服务端从不可变任务配置生成。
每个证据要求必须可由网页原文直接核验。
Priority 1 只分配给直接回答研究目标所必需的定义、核心技术路线和关键事实;
厂商罗列、趋势和扩展案例通常为 Priority 2 或 3。
不要把五个行业、全部厂商、市场规模和未来趋势合并为一个无法在两次检索内完成的宽泛问题。
禁止背景介绍、重复研究目标和解释性段落。
不要声称已经搜索。
不要生成答案。
不要输出内部思维过程。
严格按给定 JSON Schema 返回。"""

COMPACT_PLANNER_INSTRUCTIONS = """你是 DeepResearch Agent 的 Compact Research Planner。
上一次计划未能在结构化输出预算内完成。本次只生成最小可执行提纲:
- 恰好 5 个研究问题;
- scope_summary 最多 100 字;
- question 最多 100 字;
- rationale 最多 50 字;
- evidence_requirements 每题恰好 1 条, 最多 50 字;
- search_hints 每题最多 1 条, 最多 50 字;
- completion_criteria 恰好 2 条, 每条最多 60 字。
不要输出 goal 或 question id; 服务端会补齐。
禁止解释、背景介绍、重复目标和额外字段。只返回满足 Schema 的 JSON 对象。"""

_PLANNER_OUTPUT_TOKENS = 4_000
_COMPACT_PLANNER_OUTPUT_TOKENS = 1_200
_PLANNER_RETRYABLE_OUTPUT_ERRORS = {
    "MODEL_OUTPUT_INVALID",
    "MODEL_OUTPUT_TRUNCATED",
}


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
        # The Planner is a bounded control-plane outline, not a prose writer.
        output_tokens = min(
            _PLANNER_OUTPUT_TOKENS,
            binding.max_output_tokens or _PLANNER_OUTPUT_TOKENS,
        )
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
                        content=json.dumps(
                            ResearchPlanDraft.model_json_schema(),
                            ensure_ascii=False,
                        ),
                        rank_score=1.0,
                        protected=True,
                        selected_reason_code="output_contract_required",
                    ),
                    *memory_candidates,
                ),
                requested_output_tokens=output_tokens,
                context_window=binding.context_window,
                provider_max_output_tokens=binding.max_output_tokens,
                prompt_template_version="planner.v2.bounded",
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
            response_contract=ResearchPlanDraft.model_json_schema(),
            # Planning is a control-plane operation. Prefer deterministic output
            # over creative variation so compatible Prompt JSON paths are stable.
            generation_parameters={"temperature": 0.0},
            max_output_tokens=output_tokens,
            context_manifest_id=manifest_id,
            metadata={
                "run_id": str(run_id),
                "node": "planner",
                "retry_mode": "normal_bounded",
            },
        )
        first_error: ModelGatewayError | None = None
        try:
            result = await self._gateway.generate_structured(
                adapter_type=binding.adapter_type,
                base_url=binding.base_url,
                api_key=api_key,
                request=request,
                allow_regeneration=False,
            )
        except ModelGatewayError as exc:
            if exc.code not in _PLANNER_RETRYABLE_OUTPUT_ERRORS:
                raise
            first_error = exc
        else:
            try:
                draft = ResearchPlanDraft.model_validate(
                    normalize_research_plan_draft_payload(
                        result.parsed_object,
                        compact=False,
                    )
                )
            except ValidationError as exc:
                first_error = ModelGatewayError(
                    "MODEL_OUTPUT_SCHEMA_INVALID",
                    retryable=False,
                    detail_code=_schema_failure_detail(result, exc),
                    usage=result.usage,
                    diagnostics=_result_diagnostics(
                        result,
                        max_output_tokens=request.max_output_tokens,
                        retry_mode="normal_bounded",
                    ),
                )
            else:
                return materialize_research_plan(binding.goal, draft), result.usage

        if first_error is None:  # pragma: no cover - defensive state guard
            raise ModelGatewayError("MODEL_OUTPUT_SCHEMA_INVALID", retryable=False)
        compact_request = await self._build_compact_request(
            run_id=run_id,
            binding=binding,
            task_brief=task_brief,
            first_error=first_error,
        )
        try:
            compact_result = await self._gateway.generate_structured(
                adapter_type=binding.adapter_type,
                base_url=binding.base_url,
                api_key=api_key,
                request=compact_request,
                allow_regeneration=False,
            )
        except ModelGatewayError as exc:
            raise _planner_retry_error(exc, first_error=first_error) from exc

        usage = _combine_optional_usage(first_error.usage, compact_result.usage)
        try:
            compact_draft = CompactResearchPlanDraft.model_validate(
                normalize_research_plan_draft_payload(
                    compact_result.parsed_object,
                    compact=True,
                )
            )
        except ValidationError as exc:
            schema_error = ModelGatewayError(
                "MODEL_OUTPUT_SCHEMA_INVALID",
                retryable=False,
                detail_code=_schema_failure_detail(compact_result, exc),
                usage=usage,
                diagnostics=_result_diagnostics(
                    compact_result,
                    max_output_tokens=compact_request.max_output_tokens,
                    retry_mode="compact_schema_repair",
                ),
            )
            raise _planner_retry_error(schema_error, first_error=first_error) from exc
        return materialize_research_plan(binding.goal, compact_draft), usage

    async def _build_compact_request(
        self,
        *,
        run_id: UUID,
        binding: RunProviderBinding,
        task_brief: str,
        first_error: ModelGatewayError,
    ) -> CanonicalModelRequest:
        output_tokens = min(
            _COMPACT_PLANNER_OUTPUT_TOKENS,
            binding.max_output_tokens or _COMPACT_PLANNER_OUTPUT_TOKENS,
        )
        retry_mode = (
            "compact_length"
            if first_error.code == "MODEL_OUTPUT_TRUNCATED"
            else "compact_invalid_or_schema"
        )
        manifest_id = uuid4()
        if self._contexts is not None:
            envelope = await self._contexts.build(
                run_id=run_id,
                node_name="planner_compact",
                provider_adapter=binding.adapter_type.value,
                model=binding.model,
                candidates=(
                    ContextCandidate(
                        item_type=ContextItemType.INSTRUCTION,
                        content=COMPACT_PLANNER_INSTRUCTIONS,
                        rank_score=1.0,
                        protected=True,
                        selected_reason_code="compact_retry_policy_required",
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
                        content=json.dumps(
                            CompactResearchPlanDraft.model_json_schema(),
                            ensure_ascii=False,
                        ),
                        rank_score=1.0,
                        protected=True,
                        selected_reason_code="compact_output_contract_required",
                    ),
                ),
                requested_output_tokens=output_tokens,
                context_window=binding.context_window,
                provider_max_output_tokens=binding.max_output_tokens,
                prompt_template_version="planner.v2.compact",
            )
            manifest_id = envelope.manifest_id
        return CanonicalModelRequest(
            task_kind="research_planning_compact",
            role="planner",
            model=binding.model,
            instructions=COMPACT_PLANNER_INSTRUCTIONS,
            content_parts=(ContentPart(kind="text", value=task_brief),),
            response_contract=CompactResearchPlanDraft.model_json_schema(),
            generation_parameters={"temperature": 0.0},
            max_output_tokens=output_tokens,
            context_manifest_id=manifest_id,
            metadata={
                "run_id": str(run_id),
                "node": "planner_compact",
                "retry_mode": retry_mode,
            },
        )


def _combine_optional_usage(
    first: TokenUsage | None,
    second: TokenUsage,
) -> TokenUsage:
    if first is None:
        return second
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


def _result_diagnostics(
    result: CanonicalModelResult,
    *,
    max_output_tokens: int,
    retry_mode: str,
) -> dict[str, str | int]:
    return {
        "structured_output_strategy": result.capability_strategy.get(
            "structured_output", "unknown"
        ),
        "finish_reason": result.finish_reason or "unknown",
        "output_tokens": result.usage.output_tokens,
        "max_output_tokens": max_output_tokens,
        "response_length": len(result.text or ""),
        "provider_request_id": result.provider_request_id or "unknown",
        "retry_mode": retry_mode,
    }


def _planner_retry_error(
    error: ModelGatewayError,
    *,
    first_error: ModelGatewayError,
) -> ModelGatewayError:
    compact_trigger = (
        "compact_length"
        if first_error.code == "MODEL_OUTPUT_TRUNCATED"
        else "compact_invalid_or_schema"
    )
    # Output repair and transport retry are different axes. If the compact request
    # never receives a provider response, preserve retryable=True so the graph's
    # bounded 3-attempt network policy can actually run. Previously this helper
    # forced every compact timeout to permanent, despite the public UI promising
    # automatic retry.
    retry_mode = "compact_transport_retry" if error.retryable else compact_trigger
    diagnostics = {
        **error.diagnostics,
        "retry_mode": retry_mode,
        "first_error_code": first_error.code,
        "attempt_stage": "planner_compact",
        "compact_trigger": compact_trigger,
    }
    usage = (
        _combine_optional_usage(first_error.usage, error.usage)
        if error.usage is not None
        else first_error.usage
    )
    code = (
        "PLAN_OUTPUT_BUDGET_EXCEEDED"
        if error.code == "MODEL_OUTPUT_TRUNCATED"
        else error.code
    )
    return ModelGatewayError(
        code,
        retryable=error.retryable,
        detail_code=error.detail_code,
        usage=usage,
        diagnostics=diagnostics,
    )


def _schema_failure_detail(
    result: CanonicalModelResult,
    error: ValidationError,
) -> str:
    raw_strategy = result.capability_strategy.get("structured_output", "unknown")
    strategy = "".join(
        character if character.isalnum() else "_" for character in raw_strategy
    )
    finish_reason = "".join(
        character if character.isalnum() else "_"
        for character in (result.finish_reason or "unknown")
    )
    issue_types = sorted(
        {
            str(issue.get("type", "validation_error"))
            for issue in error.errors(
                include_url=False,
                include_context=False,
                include_input=False,
            )[:6]
        }
    )
    issues = "_".join(issue_types) or "validation_error"
    return (
        f"SCHEMA_INVALID_{strategy}_FINISH_{finish_reason}_ISSUES_{issues}"
    ).upper()[:100]
