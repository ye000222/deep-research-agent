"""Policy boundary for controlled tools."""

from __future__ import annotations

import hashlib

from app.domain.controlled_tools import (
    AnalyzeDataInput,
    AnalyzeDataResult,
    ControlledToolName,
    EvidenceSearchInput,
    EvidenceSearchResult,
    PolicyVerdict,
    ToolDecisionRequest,
)
from app.tools.analyze_data import AnalyzeDataTool
from app.tools.errors import ToolExecutionError
from app.tools.policy import ToolPolicyGuard
from app.tools.search_evidence import SearchEvidenceTool


class ControlledToolGateway:
    def __init__(
        self,
        evidence_search: SearchEvidenceTool,
        analyze_data: AnalyzeDataTool,
        policy: ToolPolicyGuard | None = None,
    ) -> None:
        self._evidence_search = evidence_search
        self._analyze_data = analyze_data
        self._policy = policy or ToolPolicyGuard()

    async def search_evidence(self, request: EvidenceSearchInput) -> EvidenceSearchResult:
        duplicate_key = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        decision = ToolDecisionRequest(
            action_id=request.action_id,
            tool_name=ControlledToolName.SEARCH_EVIDENCE,
            target_gap_ids=request.target_gap_ids,
            duplicate_key=duplicate_key,
            evidence_checked=True,
        )
        self._require_allowed(decision)
        return await self._evidence_search.execute(request)

    async def analyze_data(self, request: AnalyzeDataInput) -> AnalyzeDataResult:
        evidence_ids = tuple(dict.fromkeys(item.evidence_id for item in request.data))
        duplicate_key = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        decision = ToolDecisionRequest(
            action_id=request.action_id,
            tool_name=ControlledToolName.ANALYZE_DATA,
            target_gap_ids=request.target_gap_ids,
            duplicate_key=duplicate_key,
            evidence_ids=evidence_ids,
            operation=request.operation.value,
        )
        self._require_allowed(decision)
        return await self._analyze_data.execute(request)

    def authorize_web_search(self, decision: ToolDecisionRequest) -> None:
        if decision.tool_name is not ControlledToolName.WEB_SEARCH:
            raise ValueError("authorize_web_search requires a web_search decision")
        self._require_allowed(decision)

    def _require_allowed(self, decision: ToolDecisionRequest) -> None:
        result = self._policy.authorize(decision)
        if result.verdict is not PolicyVerdict.ALLOW:
            raise ToolExecutionError(result.reason_code, retryable=False)
