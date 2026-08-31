"""Deterministic Tool Policy: the model proposes, code authorizes."""

from __future__ import annotations

from app.domain.controlled_tools import (
    AnalysisOperation,
    ControlledToolName,
    PolicyVerdict,
    ToolDecisionRequest,
    ToolPolicyResult,
)


class ToolPolicyGuard:
    def authorize(
        self,
        request: ToolDecisionRequest,
        *,
        seen_duplicate_keys: set[str] | None = None,
    ) -> ToolPolicyResult:
        seen = seen_duplicate_keys or set()
        if not request.target_gap_ids:
            return self._reject("TARGET_GAP_REQUIRED", "工具调用未绑定研究缺口, 已拒绝执行。")
        if request.duplicate_key in seen:
            return self._reject(
                "DUPLICATE_TOOL_CALL", "该工具动作已执行, 使用原结果避免重复副作用。"
            )
        if request.estimated_cost > request.budget_remaining:
            return self._reject("BUDGET_EXCEEDED", "剩余预算不足, 已拒绝新的工具调用。")
        failed = sorted(key for key, value in request.preconditions.items() if not value)
        if failed:
            return self._reject(
                "PRECONDITION_FAILED",
                f"工具前置条件未满足: {', '.join(failed)}。",
            )
        if request.tool_name is ControlledToolName.WEB_SEARCH:
            if not request.evidence_checked:
                return ToolPolicyResult(
                    verdict=PolicyVerdict.FALLBACK,
                    reason_code="EVIDENCE_SEARCH_REQUIRED",
                    fallback_tool=ControlledToolName.SEARCH_EVIDENCE,
                    public_decision_summary="首次外部搜索前必须先检索当前 Evidence Store。",
                )
            if request.unread_candidate_count > 0:
                return ToolPolicyResult(
                    verdict=PolicyVerdict.FALLBACK,
                    reason_code="UNREAD_SOURCE_AVAILABLE",
                    fallback_tool=ControlledToolName.READ_WEBPAGE,
                    public_decision_summary="已有高收益未读来源, 优先阅读而不是继续堆积搜索结果。",
                )
        if request.tool_name is ControlledToolName.ANALYZE_DATA:
            allowed = {item.value for item in AnalysisOperation}
            if request.operation not in allowed:
                return self._reject("ANALYSIS_OPERATION_UNSUPPORTED", "数据分析操作不在白名单中。")
            if not request.evidence_ids:
                return self._reject(
                    "ANALYSIS_EVIDENCE_REQUIRED", "数据分析必须绑定已验证 Evidence。"
                )
        return ToolPolicyResult(
            verdict=PolicyVerdict.ALLOW,
            reason_code="POLICY_ALLOWED",
            public_decision_summary=f"工具 {request.tool_name.value} 已通过确定性策略校验。",
        )

    @staticmethod
    def _reject(reason_code: str, summary: str) -> ToolPolicyResult:
        return ToolPolicyResult(
            verdict=PolicyVerdict.REJECT,
            reason_code=reason_code,
            public_decision_summary=summary,
        )
