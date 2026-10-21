"""Non-executable, proposition-scoped research intents for V2.4 Step 2."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.proposition_research_gap import (
    GapPriorityInputs,
    PropositionGapReasonCode,
    PropositionResearchGapType,
    RequiredEvidenceCharacteristics,
)

GAP_RESEARCH_INTENT_SCHEMA_VERSION = "gap-research-intent-v1"
GAP_INTENT_PROJECTION_SCHEMA_VERSION = "gap-intent-planning-projection-v1"


class DesiredEvidencePurpose(StrEnum):
    SUPPORT = "SUPPORT"
    INDEPENDENT_CORROBORATION = "INDEPENDENT_CORROBORATION"
    CONFLICT_DISAMBIGUATION = "CONFLICT_DISAMBIGUATION"
    CONTEXT_RESOLUTION = "CONTEXT_RESOLUTION"
    SCOPE_EXTENSION = "SCOPE_EXTENSION"


class GapResearchIntentIssueCode(StrEnum):
    UNKNOWN_GAP_VERSION = "UNKNOWN_GAP_VERSION"
    INVALID_GAP = "INVALID_GAP"
    UNSUPPORTED_GAP_TYPE = "UNSUPPORTED_GAP_TYPE"
    INCONSISTENT_REQUIRED_CHARACTERISTICS = "INCONSISTENT_REQUIRED_CHARACTERISTICS"
    INTENT_IDENTITY_CONFLICT = "INTENT_IDENTITY_CONFLICT"
    PROJECTION_FAILED = "PROJECTION_FAILED"


def deterministic_intent_ref(*, gap_ref: str, attempt_index: int) -> str:
    """Build identity from stable gap, explicit attempt and intent version only."""

    identity = {
        "schema_version": GAP_RESEARCH_INTENT_SCHEMA_VERSION,
        "gap_ref": gap_ref,
        "attempt_index": attempt_index,
    }
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class GapResearchIntent(BaseModel):
    """One bounded evidence-purpose description; no query or source text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_ref: UUID
    gap_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_proposition_refs: tuple[UUID, ...] = Field(min_length=1)
    parent_claim_refs: tuple[UUID, ...] = ()
    objective_scope_ref: str = Field(min_length=1, max_length=256)
    gap_type: PropositionResearchGapType
    desired_evidence_purpose: DesiredEvidencePurpose
    required_evidence: RequiredEvidenceCharacteristics
    existing_evidence_refs: tuple[str, ...] = ()
    existing_owner_refs: tuple[str, ...] = ()
    existing_conflict_refs: tuple[str, ...] = ()
    gap_reason_codes: tuple[PropositionGapReasonCode, ...] = Field(min_length=1)
    source_reliability_refs: tuple[str, ...] = ()
    source_reliability_reason_codes: tuple[str, ...] = ()
    attempt_index: int = Field(ge=0)
    priority_inputs: GapPriorityInputs
    budget_state_ref: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    budget_state: tuple[tuple[str, int], ...] | None = None
    schema_version: str = GAP_RESEARCH_INTENT_SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_intent(self) -> GapResearchIntent:
        if self.schema_version != GAP_RESEARCH_INTENT_SCHEMA_VERSION:
            raise ValueError("unknown_gap_research_intent_schema_version")
        if self.intent_ref != deterministic_intent_ref(
            gap_ref=self.gap_ref,
            attempt_index=self.attempt_index,
        ):
            raise ValueError("gap_research_intent_ref_mismatch")
        if self.required_evidence.target_group_ref != self.target_group_ref:
            raise ValueError("intent_required_evidence_group_mismatch")
        if self.required_evidence.target_proposition_refs != self.target_proposition_refs:
            raise ValueError("intent_required_evidence_proposition_mismatch")
        if self.priority_inputs.attempt_count != self.attempt_index:
            raise ValueError("intent_attempt_priority_mismatch")
        if self.target_proposition_refs != tuple(
            sorted(set(self.target_proposition_refs), key=str)
        ):
            raise ValueError("intent_target_propositions_must_be_canonical")
        if self.parent_claim_refs != tuple(sorted(set(self.parent_claim_refs), key=str)):
            raise ValueError("intent_parent_claims_must_be_canonical")
        for name in (
            "existing_evidence_refs",
            "existing_owner_refs",
            "existing_conflict_refs",
            "source_reliability_refs",
            "source_reliability_reason_codes",
        ):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"intent_{name}_must_be_canonical")
        return self

    @property
    def priority_ordering_key(self) -> tuple[int, int, int, str, str]:
        return (*self.priority_inputs.ordering_key, self.gap_ref, self.intent_ref)


class GapResearchIntentIssue(BaseModel):
    """Per-gap fail-closed issue; other valid gaps may still proceed."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gap_ref: str | None = None
    issue_code: GapResearchIntentIssueCode
    detail_code: str = Field(min_length=1, max_length=128)
    schema_version: str = GAP_RESEARCH_INTENT_SCHEMA_VERSION


class GapIntentPlanningProjection(BaseModel):
    """Typed advisory view carried by existing ResearchContext, not a query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = GAP_INTENT_PROJECTION_SCHEMA_VERSION
    projection_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    intents: tuple[GapResearchIntent, ...]
    legacy_context_ref: str | None = None

    @model_validator(mode="after")
    def canonical_projection(self) -> GapIntentPlanningProjection:
        if self.schema_version != GAP_INTENT_PROJECTION_SCHEMA_VERSION:
            raise ValueError("unknown_gap_intent_projection_version")
        if self.intents != tuple(
            sorted(self.intents, key=lambda item: item.priority_ordering_key)
        ):
            raise ValueError("projection_intents_must_follow_gap_priority")
        refs = tuple(item.intent_ref for item in self.intents)
        if len(refs) != len(set(refs)):
            raise ValueError("projection_intents_must_be_unique")
        if self.projection_ref != deterministic_projection_ref(
            intent_refs=refs,
            legacy_context_ref=self.legacy_context_ref,
        ):
            raise ValueError("gap_intent_projection_ref_mismatch")
        return self


def deterministic_projection_ref(
    *, intent_refs: tuple[str, ...], legacy_context_ref: str | None
) -> str:
    value = {
        "schema_version": GAP_INTENT_PROJECTION_SCHEMA_VERSION,
        "intent_refs": list(intent_refs),
        "legacy_context_ref": legacy_context_ref,
    }
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
