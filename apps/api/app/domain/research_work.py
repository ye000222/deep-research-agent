"""Versioned durable work and attempt contracts for the V3 orchestrator."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.proposition_research_gap import (
    GapSearchability,
    PropositionGapLifecycle,
    PropositionResearchGapType,
    RequiredEvidenceCharacteristics,
)
from app.domain.run_state import canonical_hash

RESEARCH_WORK_SCHEMA = "research-work-v1"
WORK_ATTEMPT_SCHEMA = "work-attempt-v1"
_WORK_NAMESPACE = UUID("ed9d9562-a369-5ca7-8e21-2b87a1ed5084")


class ResearchWorkKind(StrEnum):
    LEGACY_RESEARCH = "LEGACY_RESEARCH"
    PROPOSITION_GAP = "PROPOSITION_GAP"


class ResearchWorkStatus(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    IN_PROGRESS = "IN_PROGRESS"
    AWAITING_REASSESSMENT = "AWAITING_REASSESSMENT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    SUPERSEDED = "SUPERSEDED"


class WorkAttemptState(StrEnum):
    RESERVED = "RESERVED"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


class WorkResumePoint(StrEnum):
    """Derived recovery location; never persisted as lifecycle state."""

    TARGET_NOT_STARTED = "TARGET_NOT_STARTED"
    QUERY_DURABLE = "QUERY_DURABLE"
    SEARCH_RESULT_DURABLE = "SEARCH_RESULT_DURABLE"
    READER_RESULT_DURABLE = "READER_RESULT_DURABLE"
    EVIDENCE_DURABLE = "EVIDENCE_DURABLE"
    ATTEMPT_ALREADY_COMPLETE = "ATTEMPT_ALREADY_COMPLETE"


def resolve_work_resume_point(attempt: WorkAttempt) -> WorkResumePoint:
    """Derive a conservative resume point from durable attempt references."""

    if attempt.state is WorkAttemptState.COMPLETED:
        return WorkResumePoint.ATTEMPT_ALREADY_COMPLETE
    metadata = attempt.resume_metadata
    if attempt.produced_evidence_refs or metadata.get("evidence_durable") is True:
        return WorkResumePoint.EVIDENCE_DURABLE
    if metadata.get("reader_artifact_ref"):
        return WorkResumePoint.READER_RESULT_DURABLE
    if metadata.get("search_result_durable") is True:
        return WorkResumePoint.SEARCH_RESULT_DURABLE
    if attempt.query_refs or metadata.get("query_dispatch_started") is True:
        return WorkResumePoint.QUERY_DURABLE
    return WorkResumePoint.TARGET_NOT_STARTED


class LegacyResearchObjective(BaseModel):
    """Immutable snapshot of the planner-owned execution inputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: str = Field(min_length=1, max_length=50)
    question: str
    rationale: str
    requirements: tuple[str, ...]
    priority: int = Field(ge=1, le=3)
    search_hints: tuple[str, ...] = ()
    acceptance_context: tuple[tuple[str, str], ...] = ()
    exclusions: tuple[str, ...] = ()
    plan_version: int = Field(ge=1)
    bounds: dict[str, Any] = Field(default_factory=dict)


class PropositionGapResearchObjective(BaseModel):
    """Immutable, typed objective projected from one formal gap intent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    objective_kind: Literal["PROPOSITION_GAP"] = "PROPOSITION_GAP"
    objective_schema_version: str = "proposition-gap-research-objective-v1"
    gap_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    intent_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    objective_scope_ref: str = Field(min_length=1, max_length=256)
    target_group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_proposition_refs: tuple[UUID, ...] = Field(min_length=1)
    parent_claim_refs: tuple[UUID, ...] = ()
    gap_type: PropositionResearchGapType
    required_evidence: RequiredEvidenceCharacteristics
    # A durable, non-synthetic question identity resolved during materialization
    # from Claim→Question or a single-question run plan relation.
    question_id: str | None = Field(default=None, min_length=1, max_length=50)
    question_lineage_source: Literal[
        "PARENT_CLAIM_QUESTION", "SINGLE_RUN_PLAN_ITEM"
    ] | None = None
    source_semantic_snapshot_ref: str
    source_semantic_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_gap_projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    bounds: dict[str, Any] = Field(default_factory=lambda: {"max_attempts": 3})

    @model_validator(mode="after")
    def canonical_refs(self) -> PropositionGapResearchObjective:
        if self.target_proposition_refs != tuple(
            sorted(set(self.target_proposition_refs), key=str)
        ):
            raise ValueError("gap_objective_propositions_not_canonical")
        if self.parent_claim_refs != tuple(sorted(set(self.parent_claim_refs), key=str)):
            raise ValueError("gap_objective_claims_not_canonical")
        if self.required_evidence.target_group_ref != self.target_group_ref:
            raise ValueError("gap_objective_required_evidence_group_mismatch")
        if self.required_evidence.target_proposition_refs != self.target_proposition_refs:
            raise ValueError("gap_objective_required_evidence_proposition_mismatch")
        return self


class GapResearchExecutionContext(BaseModel):
    """Minimal execution projection of current authoritative Gap/Intent data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gap_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    intent_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    gap_type: PropositionResearchGapType
    searchability: GapSearchability
    lifecycle_state: PropositionGapLifecycle
    target_group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_proposition_refs: tuple[UUID, ...]
    parent_claim_refs: tuple[UUID, ...]
    required_evidence: RequiredEvidenceCharacteristics
    existing_evidence_refs: tuple[str, ...] = ()
    existing_owner_refs: tuple[str, ...] = ()
    existing_conflict_refs: tuple[str, ...] = ()
    objective_scope_ref: str = Field(min_length=1, max_length=256)
    source_semantic_snapshot_ref: str
    source_semantic_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_gap_projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_index: int = Field(ge=0)


ResearchObjective = LegacyResearchObjective | PropositionGapResearchObjective


class ResearchWork(BaseModel):
    """One durable logical task; semantic state remains outside this object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    work_ref: UUID
    run_ref: UUID
    producer_kind: str
    producer_ref: str
    work_kind: ResearchWorkKind
    objective: ResearchObjective
    objective_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    priority: int = Field(ge=1, le=3)
    constraints: dict[str, Any] = Field(default_factory=dict)
    execution_bounds: dict[str, Any] = Field(default_factory=dict)
    budget_attribution: dict[str, Any] = Field(default_factory=dict)
    source_run_state_ref: str
    source_run_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_run_state_version: int = Field(ge=1)
    status: ResearchWorkStatus = ResearchWorkStatus.ELIGIBLE
    schema_version: str = RESEARCH_WORK_SCHEMA
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def objective_matches_kind(self) -> ResearchWork:
        if (
            self.work_kind is ResearchWorkKind.LEGACY_RESEARCH
            and not isinstance(self.objective, LegacyResearchObjective)
        ) or (
            self.work_kind is ResearchWorkKind.PROPOSITION_GAP
            and not isinstance(self.objective, PropositionGapResearchObjective)
        ):
            raise ValueError("research_work_objective_kind_mismatch")
        return self


class WorkAttempt(BaseModel):
    """Durable reservation and outcome of one bounded Work execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    attempt_ref: UUID
    work_ref: UUID
    attempt_index: int = Field(ge=1)
    state: WorkAttemptState
    reservation_identity: str
    source_run_state_ref: str
    source_run_state_version: int = Field(ge=1)
    query_refs: tuple[UUID, ...] = ()
    tool_call_refs: tuple[UUID, ...] = ()
    produced_evidence_refs: tuple[UUID, ...] = ()
    reused_evidence_refs: tuple[UUID, ...] = ()
    budget_refs: tuple[str, ...] = ()
    usage_summary: dict[str, Any] = Field(default_factory=dict)
    failure_classification: str | None = None
    resume_metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    schema_version: str = WORK_ATTEMPT_SCHEMA


def legacy_work_identity(
    *,
    run_ref: UUID,
    plan_version: int,
    question_id: str,
    objective_hash: str,
) -> UUID:
    """Stable identity; query wording and execution attempts are not identity."""

    return uuid5(
        _WORK_NAMESPACE,
        f"{run_ref}:LEGACY_PLANNER:{plan_version}:{question_id}:{objective_hash}:{RESEARCH_WORK_SCHEMA}",
    )


def legacy_objective_hash(objective: LegacyResearchObjective) -> str:
    return canonical_hash(objective)


def proposition_gap_objective_hash(
    objective: PropositionGapResearchObjective,
) -> str:
    """Hash logical objective semantics, excluding attempt and snapshot identity."""

    return canonical_hash(
        {
            "objective_schema_version": objective.objective_schema_version,
            "gap_ref": objective.gap_ref,
            "objective_scope_ref": objective.objective_scope_ref,
            "target_group_ref": objective.target_group_ref,
            "target_proposition_refs": objective.target_proposition_refs,
            "parent_claim_refs": objective.parent_claim_refs,
            "gap_type": objective.gap_type,
            "required_evidence": objective.required_evidence,
            "bounds": objective.bounds,
        }
    )


def proposition_gap_work_identity(
    *, run_ref: UUID, gap_ref: str, objective_hash: str
) -> UUID:
    return uuid5(
        _WORK_NAMESPACE,
        f"PROPOSITION_GAP:{run_ref}:{gap_ref}:{objective_hash}:{RESEARCH_WORK_SCHEMA}",
    )


def deterministic_attempt_ref(work_ref: UUID, attempt_index: int) -> UUID:
    if attempt_index < 1:
        raise ValueError("attempt_index_must_be_positive")
    return uuid5(_WORK_NAMESPACE, f"attempt:{work_ref}:{attempt_index}:{WORK_ATTEMPT_SCHEMA}")
