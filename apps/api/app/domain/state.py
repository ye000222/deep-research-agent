from __future__ import annotations

from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

Score = Annotated[float, Field(ge=0.0, le=1.0)]


class RunStatus(StrEnum):
    CREATED = "created"
    QUEUED = "queued"
    INTERRUPTED = "interrupted"
    RESEARCHING = "researching"
    WRITING = "writing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    COMPLETED_WITH_LIMITATIONS = "completed_with_limitations"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CREDENTIALS_REQUIRED = "credentials_required"


class ResearchPhase(StrEnum):
    INIT = "init"
    ANALYZE_QUERY = "analyze_query"
    PLAN = "plan"
    RESEARCH = "research"
    EVALUATE = "evaluate"
    WRITE = "write"
    VERIFY = "verify"
    FINALIZE = "finalize"


class KnowledgeStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    DISPUTED = "disputed"
    STALE = "stale"


class GapType(StrEnum):
    MISSING = "missing"
    WEAK = "weak"
    STALE = "stale"
    CONFLICT = "conflict"
    INSUFFICIENT_DIVERSITY = "insufficient_diversity"


class GapStatus(StrEnum):
    OPEN = "open"
    RESOLVING = "resolving"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


class ActionType(StrEnum):
    RETRIEVE_MEMORY = "retrieve_memory"
    SEARCH_DATABASE = "search_database"
    SEARCH_WEB = "search_web"
    READ_SOURCE = "read_source"
    EXTRACT = "extract"
    SAVE_EVIDENCE = "save_evidence"
    ANALYZE_DATA = "analyze_data"
    REPLAN = "replan"
    EVALUATE = "evaluate"
    WRITE = "write"
    STOP = "stop"


EXTERNAL_ACTIONS = {
    ActionType.SEARCH_WEB,
    ActionType.READ_SOURCE,
    ActionType.SAVE_EVIDENCE,
    ActionType.ANALYZE_DATA,
}


class ActionStatus(StrEnum):
    PROPOSED = "proposed"
    SELECTED = "selected"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"


class StopReason(StrEnum):
    QUALITY_MET = "quality_met"
    COMPLETED_WITH_LIMITATIONS = "completed_with_limitations"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STAGNATION = "stagnation"
    SOURCES_EXHAUSTED = "sources_exhausted"
    CANCELLED = "cancelled"
    FATAL_ERROR = "fatal_error"
    CREDENTIALS_REQUIRED = "credentials_required"
    PROVIDER_CAPABILITY_INSUFFICIENT = "provider_capability_insufficient"


class KnownClaimRef(BaseModel):
    claim_id: UUID
    question_id: UUID
    dimension_key: str = Field(min_length=1, max_length=100)
    status: KnowledgeStatus
    confidence: Score
    evidence_ids: tuple[UUID, ...] = ()
    independent_source_owner_keys: tuple[str, ...] = ()

    @model_validator(mode="after")
    def supported_claim_requires_evidence(self) -> KnownClaimRef:
        if self.status is KnowledgeStatus.SUPPORTED and not self.evidence_ids:
            raise ValueError("a supported claim must reference accepted evidence")
        return self


class ResearchGap(BaseModel):
    gap_id: UUID
    question_id: UUID
    dimension_key: str = Field(min_length=1, max_length=100)
    gap_type: GapType
    description: str = Field(min_length=1, max_length=1000)
    acceptance_criteria: str = Field(min_length=1, max_length=1000)
    severity: Score
    expected_information_gain: Score = 0.5
    resolution_attempts: int = Field(default=0, ge=0)
    status: GapStatus = GapStatus.OPEN
    resolved_by_claim_ids: tuple[UUID, ...] = ()

    @model_validator(mode="after")
    def resolved_gap_requires_claim(self) -> ResearchGap:
        if self.status is GapStatus.RESOLVED and not self.resolved_by_claim_ids:
            raise ValueError("a resolved gap must identify the claims that resolved it")
        return self


class NextAction(BaseModel):
    action_id: UUID
    action_type: ActionType
    target_gap_ids: tuple[UUID, ...] = ()
    tool_name: str | None = Field(default=None, max_length=100)
    expected_output: str = Field(min_length=1, max_length=1000)
    expected_information_gain: Score = 0.5
    estimated_cost: float = Field(default=0.0, ge=0.0)
    duplicate_key: str | None = Field(default=None, max_length=256)
    public_decision_summary: str = Field(min_length=1, max_length=1000)
    status: ActionStatus = ActionStatus.SELECTED

    @model_validator(mode="after")
    def external_action_requires_gap_and_tool(self) -> NextAction:
        if self.action_type in EXTERNAL_ACTIONS:
            if not self.target_gap_ids:
                raise ValueError("an external action must target at least one research gap")
            if not self.tool_name:
                raise ValueError("an external action must name its tool")
        return self


class BudgetLimits(BaseModel):
    max_iterations: int = Field(default=8, ge=1)
    max_searches: int = Field(default=15, ge=0)
    max_pages: int = Field(default=30, ge=0)
    max_model_tokens: int = Field(default=100_000, ge=1)
    max_wall_clock_seconds: int = Field(default=720, ge=1)


class BudgetUsage(BaseModel):
    iterations: int = Field(default=0, ge=0)
    searches: int = Field(default=0, ge=0)
    pages: int = Field(default=0, ge=0)
    model_tokens: int = Field(default=0, ge=0)


class QualitySnapshot(BaseModel):
    coverage: Score = 0.0
    information_gain: Score = 0.0
    low_information_gain_streak: int = Field(default=0, ge=0)
    source_quality: Score = 0.0
    source_independence: Score = 0.0
    cross_validation: Score = 0.0
    freshness: Score = 0.0
    citation_support: Score = 0.0


class CoverageDimensionSnapshot(BaseModel):
    dimension_key: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=500)
    priority: int = Field(ge=1, le=3)
    coverage: Score
    accepted_evidence: int = Field(ge=0)
    independent_sources: int = Field(ge=0)
    acceptance_criteria: tuple[str, ...] = ()
    missing_reasons: tuple[str, ...] = ()


class ResearchState(BaseModel):
    schema_version: str = "1"
    graph_schema_revision: str = "2026-08-v1"
    run_id: UUID
    state_version: int = Field(default=0, ge=0)
    status: RunStatus = RunStatus.CREATED
    phase: ResearchPhase = ResearchPhase.INIT
    iteration: int = Field(default=0, ge=0)
    current_question_id: UUID | None = None
    known: tuple[KnownClaimRef, ...] = ()
    gaps: tuple[ResearchGap, ...] = ()
    next_action: NextAction | None = None
    budget_limits: BudgetLimits = Field(default_factory=BudgetLimits)
    budget_usage: BudgetUsage = Field(default_factory=BudgetUsage)
    quality: QualitySnapshot = Field(default_factory=QualitySnapshot)
    coverage_map: tuple[CoverageDimensionSnapshot, ...] = ()
    no_progress_iterations: int = Field(default=0, ge=0)
    cancel_requested: bool = False
    stop_reason: StopReason | None = None

    @model_validator(mode="after")
    def validate_state_invariants(self) -> ResearchState:
        claim_ids = [claim.claim_id for claim in self.known]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("known claims must be unique by claim_id")

        gap_ids = [gap.gap_id for gap in self.gaps]
        if len(gap_ids) != len(set(gap_ids)):
            raise ValueError("research gaps must be unique by gap_id")

        if self.next_action and self.next_action.action_type in EXTERNAL_ACTIONS:
            open_gap_ids = {
                gap.gap_id
                for gap in self.gaps
                if gap.status in {GapStatus.OPEN, GapStatus.RESOLVING}
            }
            unknown_targets = set(self.next_action.target_gap_ids) - open_gap_ids
            if unknown_targets:
                raise ValueError("external actions may only target open or resolving gaps")

        if self.budget_usage.iterations > self.budget_limits.max_iterations:
            raise ValueError("iteration budget exceeded")
        if self.budget_usage.searches > self.budget_limits.max_searches:
            raise ValueError("search budget exceeded")
        if self.budget_usage.pages > self.budget_limits.max_pages:
            raise ValueError("page budget exceeded")
        if self.budget_usage.model_tokens > self.budget_limits.max_model_tokens:
            raise ValueError("model token budget exceeded")
        return self


class StatePatch(BaseModel):
    patch_id: UUID
    base_version: int = Field(ge=0)
    known_upserts: tuple[KnownClaimRef, ...] = ()
    gap_upserts: tuple[ResearchGap, ...] = ()
    next_action: NextAction | None = None
    budget_usage: BudgetUsage | None = None
    quality: QualitySnapshot | None = None
    coverage_map: tuple[CoverageDimensionSnapshot, ...] | None = None
    phase: ResearchPhase | None = None
    status: RunStatus | None = None
    stop_reason: StopReason | None = None
    clear_stop_reason: bool = False
