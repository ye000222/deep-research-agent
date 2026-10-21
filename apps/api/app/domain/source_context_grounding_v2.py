"""Versioned, non-persistent source semantic grounding contracts for V2.1R shadow use."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.source_context_extraction import SourceAttachmentRelation, SourceScopeKind
from app.domain.truth_condition_context import ContextType

SOURCE_CONTEXT_GROUNDING_SCHEMA_V2 = "source-context-extraction-schema-v2"
SOURCE_CONTEXT_GROUNDING_PROMPT_V2 = "source-context-extraction.generic.v2"


class GroundingDimensionStatus(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"
    MISSING_EVIDENCE = "MISSING_EVIDENCE"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_EVALUATED = "NOT_EVALUATED"


class GroundingStatusV2(StrEnum):
    VALID = "VALID"
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"
    AMBIGUOUS = "AMBIGUOUS"
    NOT_EVALUATED = "NOT_EVALUATED"
    UNRESOLVED = "UNRESOLVED"


class MaterialNodeUsability(StrEnum):
    GROUNDED_USABLE = "GROUNDED_USABLE"
    PARTIALLY_GROUNDED_UNRESOLVED = "PARTIALLY_GROUNDED_UNRESOLVED"
    UNGROUNDED_UNUSABLE = "UNGROUNDED_UNUSABLE"
    SEMANTIC_SUPPORT_AMBIGUOUS = "SEMANTIC_SUPPORT_AMBIGUOUS"
    NOT_EVALUATED = "NOT_EVALUATED"


class SemanticEvidenceRole(StrEnum):
    ASSERTION_EVIDENCE = "ASSERTION_EVIDENCE"
    CONTEXT_VALUE = "CONTEXT_VALUE"
    RELATION_OPERATOR = "RELATION_OPERATOR"
    EPISTEMIC_OPERATOR = "EPISTEMIC_OPERATOR"
    EPISTEMIC_AGENT = "EPISTEMIC_AGENT"
    MODALITY = "MODALITY"
    METHOD_RELATION = "METHOD_RELATION"
    METHOD_VALUE = "METHOD_VALUE"
    CONDITION = "CONDITION"
    CONFIGURATION_VALUE = "CONFIGURATION_VALUE"
    SCOPE_VALUE = "SCOPE_VALUE"
    POPULATION_VALUE = "POPULATION_VALUE"
    LOCATION_VALUE = "LOCATION_VALUE"
    SCENARIO_VALUE = "SCENARIO_VALUE"
    TEMPORAL_VALUE = "TEMPORAL_VALUE"
    VERSION_VALUE = "VERSION_VALUE"
    SAMPLE_SCOPE_VALUE = "SAMPLE_SCOPE_VALUE"
    MEASUREMENT_DEFINITION = "MEASUREMENT_DEFINITION"
    MEASUREMENT_VALUE = "MEASUREMENT_VALUE"
    MEASUREMENT_UNIT = "MEASUREMENT_UNIT"
    MEASUREMENT_COMPARATOR = "MEASUREMENT_COMPARATOR"
    ATTACHMENT = "ATTACHMENT"
    SCOPE_STRUCTURE = "SCOPE_STRUCTURE"


class EvidenceTargetKind(StrEnum):
    ASSERTION = "ASSERTION"
    CONTEXT = "CONTEXT"
    ATTACHMENT = "ATTACHMENT"
    SCOPE = "SCOPE"


class EvidenceSpanInput(BaseModel):
    """Provider-authored quote locator; offsets and hashes are never provider-authored."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    exact_text: str = Field(min_length=1, max_length=4_000)
    occurrence_index: int = Field(default=0, ge=0, le=1_000)


class SourceEvidenceSpan(BaseModel):
    """Resolved ephemeral source span; not a persistence entity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: str
    exact_text: str
    occurrence_index: int
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)
    claim_hash: str
    textual_grounding_status: GroundingDimensionStatus
    diagnostics: tuple[str, ...] = ()


class SemanticEvidenceBindingInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    binding_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    evidence_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    semantic_role: SemanticEvidenceRole
    target_kind: EvidenceTargetKind
    target_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    related_binding_ref: str | None = Field(default=None, max_length=64)


class DimensionGrounding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    textual_grounding: GroundingDimensionStatus
    semantic_support: GroundingDimensionStatus
    structural_grounding: GroundingDimensionStatus
    diagnostics: tuple[str, ...] = ()


class SemanticEvidenceBinding(BaseModel):
    """Validated binding. support status is always derived by the local validator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    binding_ref: str
    evidence_ref: str
    semantic_role: SemanticEvidenceRole
    target_kind: EvidenceTargetKind
    target_ref: str
    related_binding_ref: str | None = None
    support_status: GroundingDimensionStatus
    dimensions: DimensionGrounding
    diagnostics: tuple[str, ...] = ()


class SourceAssertionV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_assertion_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=32)
    assertion_status: str = Field(pattern=r"^(ASSERTED|ATTRIBUTED|CONDITIONAL|UNKNOWN)$")
    source_scope_ref: str | None = Field(default=None, max_length=64)


class SourceContextV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    context_type: ContextType
    semantic_value: dict[str, Any] = Field(default_factory=dict)
    lexical_value: str | None = Field(default=None, max_length=2_000)
    materiality: str = Field(default="UNKNOWN", pattern=r"^(MATERIAL|DESCRIPTIVE|UNKNOWN)$")


class SourceAttachmentV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attachment_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    context_ref: str
    source_assertion_refs: tuple[str, ...] = Field(default=(), max_length=64)
    relation: SourceAttachmentRelation
    status: str = Field(pattern=r"^(RESOLVED|UNKNOWN)$")
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=32)
    source_scope_ref: str | None = Field(default=None, max_length=64)


class SourceScopeV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_scope_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    kind: SourceScopeKind
    source_assertion_refs: tuple[str, ...] = Field(default=(), max_length=64)
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=32)


class SourceExtractionResponseV2(BaseModel):
    """Provider-facing v2 schema. Grounding results are deliberately absent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^source-context-extraction-schema-v2$")
    status: str = Field(pattern=r"^(RESOLVED|UNRESOLVED)$")
    source_assertions: tuple[SourceAssertionV2, ...] = Field(default=(), max_length=64)
    source_contexts: tuple[SourceContextV2, ...] = Field(default=(), max_length=128)
    source_attachments: tuple[SourceAttachmentV2, ...] = Field(default=(), max_length=256)
    source_scopes: tuple[SourceScopeV2, ...] = Field(default=(), max_length=64)
    evidence_spans: tuple[EvidenceSpanInput, ...] = Field(default=(), max_length=512)
    evidence_bindings: tuple[SemanticEvidenceBindingInput, ...] = Field(
        default=(), max_length=1_024
    )
    unresolved_items: tuple[str, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def status_shape(self) -> SourceExtractionResponseV2:
        if self.status == "RESOLVED" and not self.source_assertions:
            raise ValueError("resolved_response_requires_assertion")
        if self.status == "UNRESOLVED" and not self.unresolved_items:
            raise ValueError("unresolved_response_requires_reason")
        return self


class NodeGroundingAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_kind: EvidenceTargetKind
    target_ref: str
    dimensions: DimensionGrounding
    usability: MaterialNodeUsability


class SourceContextGroundingResultV2(BaseModel):
    """Ephemeral shadow result. It has no repository or database writer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_ref: str | None
    claim_hash: str
    source_assertions: tuple[SourceAssertionV2, ...] = ()
    source_contexts: tuple[SourceContextV2, ...] = ()
    source_attachments: tuple[SourceAttachmentV2, ...] = ()
    source_scopes: tuple[SourceScopeV2, ...] = ()
    evidence_spans: tuple[SourceEvidenceSpan, ...] = ()
    evidence_bindings: tuple[SemanticEvidenceBinding, ...] = ()
    node_assessments: tuple[NodeGroundingAssessment, ...] = ()
    unresolved_items: tuple[str, ...] = ()
    status: GroundingStatusV2
    diagnostics: tuple[str, ...] = ()
    schema_version: str = SOURCE_CONTEXT_GROUNDING_SCHEMA_V2
    prompt_version: str = SOURCE_CONTEXT_GROUNDING_PROMPT_V2
    model_call_count: int = Field(default=0, ge=0, le=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
