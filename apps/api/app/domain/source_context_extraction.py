"""Non-persistent source-side context extraction contracts for V2.1R shadow work."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.truth_condition_context import (
    ContextOrigin,
    ContextScope,
    ContextScopeKind,
    ContextType,
    TruthConditionContext,
    canonical_context_value,
    normalize_semantic_value,
)

SOURCE_CONTEXT_EXTRACTION_SCHEMA_VERSION = "source-context-extraction-schema-v1"
SOURCE_CONTEXT_EXTRACTION_PROMPT_VERSION = "source-context-extraction.generic.v1"
SOURCE_CONTEXT_NORMALIZER_VERSION = "truth-condition-context-canonicalizers-v1"


class AssertionStatus(StrEnum):
    ASSERTED = "ASSERTED"
    ATTRIBUTED = "ATTRIBUTED"
    CONDITIONAL = "CONDITIONAL"
    UNKNOWN = "UNKNOWN"


class GroundingStatus(StrEnum):
    MODEL_OUTPUT_VALID = "MODEL_OUTPUT_VALID"
    GROUNDING_VALID = "GROUNDING_VALID"
    GROUNDING_PARTIAL = "GROUNDING_PARTIAL"
    GROUNDING_INVALID = "GROUNDING_INVALID"
    SOURCE_EXTRACTION_UNRESOLVED = "SOURCE_EXTRACTION_UNRESOLVED"
    STRUCTURED_OUTPUT_INVALID = "STRUCTURED_OUTPUT_INVALID"
    MODEL_ERROR = "MODEL_ERROR"
    CALL_BUDGET_EXHAUSTED = "CALL_BUDGET_EXHAUSTED"


class SourceScopeKind(StrEnum):
    SINGLE_ASSERTION = "SINGLE_ASSERTION"
    ASSERTION_SET = "ASSERTION_SET"
    CLAUSE = "CLAUSE"
    ASSERTION_GROUP = "ASSERTION_GROUP"
    CLAIM = "CLAIM"
    UNKNOWN = "UNKNOWN"


class SourceAttachmentRelation(StrEnum):
    APPLIES_TO = "APPLIES_TO"
    CONDITIONS = "CONDITIONS"
    ATTRIBUTES = "ATTRIBUTES"
    UNKNOWN = "UNKNOWN"


class SourceAttachmentStatus(StrEnum):
    RESOLVED = "RESOLVED"
    UNKNOWN = "UNKNOWN"


class SpanEvidence(BaseModel):
    """Exact quote locator. occurrence_index is zero-based among exact matches."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exact_text: str = Field(min_length=1, max_length=4_000)
    occurrence_index: int = Field(default=0, ge=0, le=1_000)
    advisory_start: int | None = Field(default=None, ge=0)
    advisory_end: int | None = Field(default=None, ge=0)


class GroundedSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    exact_text: str
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    occurrence_index: int = Field(ge=0)


class SourceAssertionAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_assertion_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    source_evidence: SpanEvidence
    grounded_span: GroundedSpan | None = None
    clause_ref: str | None = Field(default=None, max_length=64)
    source_scope_ref: str | None = Field(default=None, max_length=64)
    predicate_evidence: SpanEvidence | None = None
    grounded_predicate_span: GroundedSpan | None = None
    assertion_status: AssertionStatus
    grounding_status: str = "PENDING"
    diagnostics: tuple[str, ...] = ()


class SourceContextNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    context_type: ContextType
    semantic_value: dict[str, Any] = Field(default_factory=dict)
    canonical_value: dict[str, Any] = Field(default_factory=dict)
    lexical_value: str | None = Field(default=None, max_length=2_000)
    source_evidence: SpanEvidence
    grounded_span: GroundedSpan | None = None
    materiality: str = Field(default="UNKNOWN", pattern=r"^(MATERIAL|DESCRIPTIVE|UNKNOWN)$")


class SourceContextAttachment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context_ref: str
    source_assertion_refs: tuple[str, ...] = ()
    relation: SourceAttachmentRelation
    status: SourceAttachmentStatus
    evidence: SpanEvidence | None = None
    grounded_evidence: GroundedSpan | None = None
    reason_code: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unknown_has_no_targets(self) -> SourceContextAttachment:
        if (
            self.status is SourceAttachmentStatus.UNKNOWN
            and self.relation is not SourceAttachmentRelation.UNKNOWN
        ):
            raise ValueError("unknown_attachment_requires_unknown_relation")
        if self.status is SourceAttachmentStatus.RESOLVED and not self.source_assertion_refs:
            raise ValueError("resolved_attachment_requires_assertion_refs")
        return self


class SourceScopeNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_scope_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    kind: SourceScopeKind
    source_assertion_refs: tuple[str, ...] = ()
    source_evidence: SpanEvidence | None = None
    grounded_evidence: GroundedSpan | None = None


class SourceAssertionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_assertion_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    source_evidence: SpanEvidence
    clause_ref: str | None = Field(default=None, max_length=64)
    source_scope_ref: str | None = Field(default=None, max_length=64)
    predicate_evidence: SpanEvidence | None = None
    assertion_status: AssertionStatus


class SourceContextOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    context_type: ContextType
    semantic_value: dict[str, Any] = Field(default_factory=dict)
    lexical_value: str | None = Field(default=None, max_length=2_000)
    source_evidence: SpanEvidence
    materiality: str = Field(default="UNKNOWN", pattern=r"^(MATERIAL|DESCRIPTIVE|UNKNOWN)$")


class SourceAttachmentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context_ref: str
    source_assertion_refs: tuple[str, ...] = ()
    relation: SourceAttachmentRelation
    status: SourceAttachmentStatus
    evidence: SpanEvidence | None = None
    reason_code: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unknown_has_no_targets(self) -> SourceAttachmentOutput:
        if (
            self.status is SourceAttachmentStatus.UNKNOWN
            and self.relation is not SourceAttachmentRelation.UNKNOWN
        ):
            raise ValueError("unknown_attachment_requires_unknown_relation")
        if self.status is SourceAttachmentStatus.RESOLVED and not self.source_assertion_refs:
            raise ValueError("resolved_attachment_requires_assertion_refs")
        return self


class SourceScopeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_scope_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    kind: SourceScopeKind
    source_assertion_refs: tuple[str, ...] = ()
    evidence: SpanEvidence | None = None


class SourceExtractionResponse(BaseModel):
    """Provider-facing, versioned schema; no grounded offsets or output-side fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^source-context-extraction-schema-v1$")
    status: str = Field(pattern=r"^(RESOLVED|UNRESOLVED)$")
    source_assertions: tuple[SourceAssertionOutput, ...] = Field(max_length=64)
    source_contexts: tuple[SourceContextOutput, ...] = Field(max_length=128)
    source_attachments: tuple[SourceAttachmentOutput, ...] = Field(max_length=256)
    source_scopes: tuple[SourceScopeOutput, ...] = Field(max_length=64)
    unresolved_items: tuple[str, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def status_shape(self) -> SourceExtractionResponse:
        if self.status == "RESOLVED" and not self.source_assertions:
            raise ValueError("resolved_response_requires_assertion")
        if self.status == "UNRESOLVED" and not self.unresolved_items:
            raise ValueError("unresolved_response_requires_reason")
        return self


class SourceContextExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_ref: str | None
    claim_hash: str
    source_assertions: tuple[SourceAssertionAnchor, ...] = ()
    source_contexts: tuple[SourceContextNode, ...] = ()
    source_attachments: tuple[SourceContextAttachment, ...] = ()
    source_scopes: tuple[SourceScopeNode, ...] = ()
    unresolved_items: tuple[str, ...] = ()
    model_status: str
    grounding_status: GroundingStatus
    extractor_version: str = SOURCE_CONTEXT_EXTRACTION_PROMPT_VERSION
    schema_version: str = SOURCE_CONTEXT_EXTRACTION_SCHEMA_VERSION
    prompt_version: str = SOURCE_CONTEXT_EXTRACTION_PROMPT_VERSION
    normalizer_versions: tuple[str, ...] = (SOURCE_CONTEXT_NORMALIZER_VERSION,)
    diagnostics: tuple[str, ...] = ()
    auxiliary_detector_comparison: tuple[str, ...] = ()
    model_call_count: int = Field(default=0, ge=0, le=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)


def normalize_source_semantic_value(value: object) -> object:
    """Use the approved shared exact-value canonicalizer; do not convert units."""
    return normalize_semantic_value(value)


def canonicalize_source_context(
    context_type: ContextType,
    semantic_value: dict[str, Any],
    lexical_value: str | None,
) -> dict[str, Any]:
    """Delegate supported temporal/version semantics to the existing comparator contract."""
    normalized = normalize_semantic_value(semantic_value)
    normalized_dict = normalized if isinstance(normalized, dict) else {"value": normalized}
    projected = TruthConditionContext(
        context_ref="source-context-normalization",
        semantic_signature="diagnostic-only",
        context_type=context_type,
        semantic_value=normalized_dict,
        lexical_value=lexical_value,
        scope=ContextScope(kind=ContextScopeKind.UNKNOWN),
        origin=ContextOrigin.SOURCE,
    )
    return canonical_context_value(projected)
