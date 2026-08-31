"""Canonical contracts for policy-gated database and analysis tools."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ControlledToolName(StrEnum):
    SEARCH_EVIDENCE = "search_evidence"
    WEB_SEARCH = "web_search"
    READ_WEBPAGE = "read_webpage"
    SAVE_EVIDENCE = "save_evidence"
    ANALYZE_DATA = "analyze_data"


class PolicyVerdict(StrEnum):
    ALLOW = "allow"
    REJECT = "reject"
    FALLBACK = "fallback"


class ToolDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: UUID
    tool_name: ControlledToolName
    target_gap_ids: tuple[UUID, ...]
    duplicate_key: str = Field(min_length=1, max_length=256)
    estimated_cost: float = Field(default=0.0, ge=0.0)
    budget_remaining: float = Field(default=1.0, ge=0.0)
    evidence_ids: tuple[UUID, ...] = ()
    operation: str | None = None
    evidence_checked: bool = False
    unread_candidate_count: int = Field(default=0, ge=0)
    preconditions: dict[str, bool] = Field(default_factory=dict)


class ToolPolicyResult(BaseModel):
    verdict: PolicyVerdict
    reason_code: str
    fallback_tool: ControlledToolName | None = None
    public_decision_summary: str


class EvidenceSearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    action_id: UUID
    target_gap_ids: tuple[UUID, ...] = Field(min_length=1)
    question_id: str = Field(min_length=1, max_length=50)
    query: str = Field(min_length=1, max_length=2000)
    min_score: float = Field(default=0.55, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1, le=20)


class EvidenceSearchCard(BaseModel):
    evidence_id: UUID
    claim_id: UUID | None
    snapshot_id: UUID | None
    question_id: str
    claim: str
    exact_quote: str
    relation: str
    source_title: str
    source_url: str
    source_owner_key: str
    evidence_score: float
    retrieval_score: float


class EvidenceSearchResult(BaseModel):
    call_id: UUID
    status: str
    items: tuple[EvidenceSearchCard, ...]
    result_refs: tuple[UUID, ...]


class AnalysisOperation(StrEnum):
    AGGREGATE = "aggregate"
    GROWTH_RATE = "growth_rate"
    CAGR = "cagr"
    RATIO = "ratio"
    RANK = "rank"
    COMPARE = "compare"
    DESCRIPTIVE_STATS = "descriptive_stats"


class AnalysisDataPoint(BaseModel):
    evidence_id: UUID
    label: str = Field(min_length=1, max_length=200)
    value: float
    unit: str = Field(min_length=1, max_length=50)
    period: str | None = Field(default=None, max_length=50)


class AnalyzeDataInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    action_id: UUID
    target_gap_ids: tuple[UUID, ...] = Field(min_length=1)
    question_id: str = Field(min_length=1, max_length=50)
    operation: AnalysisOperation
    data: tuple[AnalysisDataPoint, ...] = Field(min_length=1, max_length=100)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def units_must_match(self) -> AnalyzeDataInput:
        units = {item.unit for item in self.data}
        if len(units) > 1 and not bool(self.parameters.get("allow_mixed_units")):
            raise ValueError("analysis requires a single unit unless explicitly converted")
        return self


class AnalyzeDataResult(BaseModel):
    artifact_id: UUID
    call_id: UUID
    status: str
    operation: AnalysisOperation
    result: dict[str, Any]
    formula: str
    formula_version: str
    input_evidence_ids: tuple[UUID, ...]
    warnings: tuple[str, ...] = ()
