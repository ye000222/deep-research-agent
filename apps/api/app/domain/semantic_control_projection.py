"""Immutable artifacts for the V3.2 semantic control projection."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.gap_research_intent import GapResearchIntent
from app.domain.proposition_research_gap import (
    PropositionGapDerivationIssue,
    PropositionResearchGap,
)
from app.domain.run_state import canonical_hash

GAP_CONTROL_ARTIFACT_VERSION = "semantic-gap-control-projection-v1"
INTENT_CONTROL_ARTIFACT_VERSION = "semantic-intent-control-projection-v1"


class SemanticGapControlArtifact(BaseModel):
    """Content-addressed gap result, including valid-empty and incomplete states."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = GAP_CONTROL_ARTIFACT_VERSION
    run_ref: UUID
    semantic_snapshot_ref: str
    semantic_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection_status: str = Field(pattern=r"^(DERIVED|INCOMPLETE|CAPABILITY_BLOCKED)$")
    gaps: tuple[PropositionResearchGap, ...] = ()
    issues: tuple[PropositionGapDerivationIssue, ...] = ()
    projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_ref: str

    @model_validator(mode="after")
    def validate_identity(self) -> SemanticGapControlArtifact:
        if self.schema_version != GAP_CONTROL_ARTIFACT_VERSION:
            raise ValueError("unknown_gap_control_artifact_version")
        if tuple(sorted(self.gaps, key=lambda item: item.priority_ordering_key)) != self.gaps:
            raise ValueError("gap_control_artifact_not_canonical")
        expected = canonical_hash([item.gap_ref for item in self.gaps])
        if self.projection_hash != expected:
            raise ValueError("gap_control_projection_hash_mismatch")
        if self.projection_status == "DERIVED" and self.issues:
            raise ValueError("derived_gap_projection_cannot_have_issues")
        if self.artifact_hash != canonical_hash(
            {
                key: value
                for key, value in self.model_dump(mode="json").items()
                if key not in {"artifact_hash", "artifact_ref"}
            }
        ):
            raise ValueError("gap_control_artifact_hash_mismatch")
        if self.artifact_ref != semantic_gap_artifact_ref(self.run_ref, self.artifact_hash):
            raise ValueError("gap_control_artifact_ref_mismatch")
        return self


class SemanticIntentControlArtifact(BaseModel):
    """Immutable intent projection explicitly bound to one gap projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = INTENT_CONTROL_ARTIFACT_VERSION
    run_ref: UUID
    semantic_snapshot_ref: str
    semantic_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    gap_artifact_ref: str
    gap_projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    intents: tuple[GapResearchIntent, ...] = ()
    issues: tuple[dict[str, Any], ...] = ()
    projection_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_ref: str

    @model_validator(mode="after")
    def validate_identity(self) -> SemanticIntentControlArtifact:
        if self.schema_version != INTENT_CONTROL_ARTIFACT_VERSION:
            raise ValueError("unknown_intent_control_artifact_version")
        if self.projection_hash != canonical_hash([item.intent_ref for item in self.intents]):
            raise ValueError("intent_control_projection_hash_mismatch")
        if self.artifact_hash != canonical_hash(
            {
                key: value
                for key, value in self.model_dump(mode="json").items()
                if key not in {"artifact_hash", "artifact_ref"}
            }
        ):
            raise ValueError("intent_control_artifact_hash_mismatch")
        if self.artifact_ref != semantic_intent_artifact_ref(self.run_ref, self.artifact_hash):
            raise ValueError("intent_control_artifact_ref_mismatch")
        if any(item.run_ref != self.run_ref for item in self.intents):
            raise ValueError("intent_control_run_mismatch")
        return self


def semantic_gap_artifact_ref(run_ref: UUID, projection_hash: str) -> str:
    return f"runs/{run_ref}/semantic-control/gaps/{projection_hash}.json"


def semantic_intent_artifact_ref(run_ref: UUID, projection_hash: str) -> str:
    return f"runs/{run_ref}/semantic-control/intents/{projection_hash}.json"
