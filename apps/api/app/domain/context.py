"""Contracts for deterministic context budgeting and auditable manifests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class ContextItemType(StrEnum):
    INSTRUCTION = "instruction"
    TASK_BRIEF = "task_brief"
    STATE_SUMMARY = "state_summary"
    GAP = "gap"
    EVIDENCE_CARD = "evidence_card"
    SOURCE_CHUNK = "source_chunk"
    RECENT_ACTION = "recent_action"
    CONFLICT = "conflict"
    MEMORY = "memory"
    OUTPUT_SCHEMA = "output_schema"


class CompressionLevel(StrEnum):
    RAW = "raw"
    EXTRACTIVE = "extractive"
    SUMMARIZED = "summarized"


@dataclass(frozen=True, slots=True)
class ContextCandidate:
    item_type: ContextItemType
    content: str
    rank_score: float
    source_ref_type: str | None = None
    source_ref_id: str | None = None
    protected: bool = False
    compression_level: CompressionLevel = CompressionLevel.RAW
    selected_reason_code: str = "ranked_for_node"
    provenance_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ContextBudgetAllocation:
    context_window: int
    input_budget: int
    output_reserve: int
    safety_margin: int


@dataclass(frozen=True, slots=True)
class ContextEnvelope:
    manifest_id: UUID
    allocation: ContextBudgetAllocation
    selected: tuple[ContextCandidate, ...]
    rejected: tuple[ContextCandidate, ...]
    token_before: int
    token_after: int
    rendered_prompt_hash: str

    def selected_by_type(self, item_type: ContextItemType) -> tuple[ContextCandidate, ...]:
        return tuple(item for item in self.selected if item.item_type == item_type)


class ContextItemMetric(BaseModel):
    item_type: ContextItemType
    source_ref_type: str | None
    source_ref_id: str | None
    rank_score: float
    token_count: int = Field(ge=0)
    compression_level: CompressionLevel
    selected: bool
    protected: bool
    selected_reason_code: str
    content_hash: str = Field(min_length=64, max_length=64)
    compression_artifact_id: UUID | None = None


class CompressionArtifactMetric(BaseModel):
    artifact_id: UUID
    input_hash: str = Field(min_length=64, max_length=64)
    output_hash: str = Field(min_length=64, max_length=64)
    compression_level: CompressionLevel
    token_before: int = Field(ge=0)
    token_after: int = Field(ge=0)
    validation_status: str
    provenance_refs: tuple[str, ...] = ()


class ContextManifestView(BaseModel):
    manifest_id: UUID
    run_id: UUID
    node_name: str
    provider_adapter: str
    model: str
    context_window: int = Field(ge=1)
    input_budget: int = Field(ge=1)
    output_reserve: int = Field(ge=1)
    safety_margin: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    protected_count: int = Field(ge=0)
    compressed_count: int = Field(ge=0)
    token_before: int = Field(ge=0)
    token_after: int = Field(ge=0)
    compression_ratio: float = Field(ge=0.0, le=1.0)
    truncated: bool
    rendered_prompt_hash: str = Field(min_length=64, max_length=64)
    prompt_template_version: str
    created_at: datetime
    items: list[ContextItemMetric] = Field(default_factory=list)
    compression_artifacts: list[CompressionArtifactMetric] = Field(default_factory=list)
