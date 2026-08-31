from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class ToolName(StrEnum):
    WEB_SEARCH = "web_search"
    READ_WEBPAGE = "read_webpage"
    SAVE_EVIDENCE = "save_evidence"
    SEARCH_EVIDENCE = "search_evidence"
    ANALYZE_DATA = "analyze_data"


class ToolPolicyResult(StrEnum):
    ALLOW = "allow"
    REJECT = "reject"
    FALLBACK = "fallback"


class ToolDecision(BaseModel):
    action_id: UUID
    tool_name: ToolName
    target_gap_ids: tuple[UUID, ...]
    objective: str = Field(min_length=1, max_length=1000)
    arguments: dict[str, Any]
    expected_information_gain: float = Field(ge=0.0, le=1.0)
    estimated_token_cost: int = Field(default=0, ge=0)
    estimated_money_cost: float = Field(default=0.0, ge=0.0)
    duplicate_key: str = Field(min_length=1, max_length=256)
    preconditions: dict[str, bool] = Field(default_factory=dict)
    policy_result: ToolPolicyResult
    fallback_tool: ToolName | None = None
    public_decision_summary: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_policy_decision(self) -> ToolDecision:
        if not self.target_gap_ids:
            raise ValueError("a tool decision must target at least one research gap")
        if self.policy_result is ToolPolicyResult.FALLBACK and self.fallback_tool is None:
            raise ValueError("fallback policy decisions must identify a fallback tool")
        return self


class ToolCallStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    ERROR = "error"


class ToolError(BaseModel):
    code: str
    message: str
    retryable: bool
    details: dict[str, Any] = Field(default_factory=dict)


class ToolResultEnvelope(BaseModel):
    call_id: UUID
    action_id: UUID
    target_gap_ids: tuple[UUID, ...]
    status: ToolCallStatus
    result_refs: tuple[UUID, ...] = ()
    budget_delta: dict[str, int | float] = Field(default_factory=dict)
    error: ToolError | None = None

    @model_validator(mode="after")
    def error_status_requires_error(self) -> ToolResultEnvelope:
        if self.status is ToolCallStatus.ERROR and self.error is None:
            raise ValueError("an error result must include a normalized error")
        if self.status is not ToolCallStatus.ERROR and self.error is not None:
            raise ValueError("successful or partial results may not include a fatal error")
        return self
