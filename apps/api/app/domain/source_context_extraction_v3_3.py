"""Isolated v3.3 shadow contract for explicit Measurement identity semantics."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, model_validator

from app.domain.source_context_extraction_v3_1 import (
    GraphNodeKindV31,
    MeasurementBasisV31,
    MeasurementThresholdV31,
    ObservedValueComparatorV31,
    ObservedValueV31,
    SemanticEvidenceBindingV31,
    SemanticRelationStatusV31,
    SemanticRelationTypeV31,
    SourceAssertionV31,
    SourceAttachmentV31,
    SourceEvidenceSpanV31,
    SourceScopeV31,
)
from app.domain.truth_condition_context import ContextType

SOURCE_CONTEXT_EXTRACTION_SCHEMA_V33: Literal["source-context-extraction-schema-v3.3"] = (
    "source-context-extraction-schema-v3.3"
)
SOURCE_CONTEXT_EXTRACTION_DOMAIN_V33: Literal["source-context-extraction-domain-v3.3"] = (
    "source-context-extraction-domain-v3.3"
)
SOURCE_CONTEXT_EXTRACTION_PROMPT_V33: Literal["source-context-extraction.generic.v3.3"] = (
    "source-context-extraction.generic.v3.3"
)


class MeasurementIdentityTypeV33(StrEnum):
    MEASURABLE_PROPERTY = "MEASURABLE_PROPERTY"
    NAMED_METRIC = "NAMED_METRIC"
    SCORE_OR_INDEX = "SCORE_OR_INDEX"
    COUNT = "COUNT"
    RATE = "RATE"
    SIZE_OR_CAPACITY = "SIZE_OR_CAPACITY"
    TEMPORAL_MEASURE = "TEMPORAL_MEASURE"
    QUALITATIVE_PROPERTY = "QUALITATIVE_PROPERTY"
    UNKNOWN = "UNKNOWN"


class MeasurementAdmissibilityV33(StrEnum):
    ADMISSIBLE = "ADMISSIBLE"
    NOT_MEASUREMENT = "NOT_MEASUREMENT"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class SourceSpanReferenceV33(BaseModel):
    """Reference an existing exact source span; it does not carry inferred text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    occurrence_index: int = Field(default=0, ge=0, le=1_000)


class MeasurementIdentityV33(BaseModel):
    """Semantic identity and provenance; optional fields permit fail-closed review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str | None = Field(default=None, min_length=1, max_length=500)
    identity_type: MeasurementIdentityTypeV33 = MeasurementIdentityTypeV33.UNKNOWN
    source_span: SourceSpanReferenceV33 | None = None


class MeasurementSemanticsV33(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    measurement_identity: MeasurementIdentityV33 | None = None
    formal_definition: str | None = Field(default=None, min_length=1, max_length=2_000)
    observed_value: ObservedValueV31 | None = None
    unit: str | None = Field(default=None, min_length=1, max_length=200)
    comparator: ObservedValueComparatorV31 | None = None
    threshold: MeasurementThresholdV31 | None = None
    basis_ref: str | None = Field(default=None, pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    scope_category: str | None = Field(default=None, min_length=1, max_length=500)
    target_object: str | None = Field(default=None, min_length=1, max_length=500)
    qualifiers: tuple[str, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def threshold_comparator_consistency(self) -> MeasurementSemanticsV33:
        if (
            self.comparator is not None
            and self.threshold is not None
            and self.comparator != self.threshold.comparator
        ):
            raise ValueError("measurement_threshold_comparator_conflict")
        return self


class GenericContextSemanticValueV33(RootModel[dict[str, Any]]):
    model_config = ConfigDict(frozen=True)


class SourceContextV33(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    context_type: ContextType
    semantic_value: MeasurementSemanticsV33 | GenericContextSemanticValueV33 = Field(
        default_factory=lambda: GenericContextSemanticValueV33({})
    )
    lexical_value: str | None = Field(default=None, max_length=2_000)
    materiality: str = Field(default="UNKNOWN", pattern=r"^(MATERIAL|DESCRIPTIVE|UNKNOWN)$")

    @model_validator(mode="before")
    @classmethod
    def dispatch_by_context_type(cls, value: Any) -> Any:
        if isinstance(value, cls) or not isinstance(value, dict):
            return value
        raw_type = value.get("context_type")
        if raw_type is None:
            return value
        try:
            context_type = ContextType(raw_type)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported_context_type_discriminator") from exc
        raw_semantics = value.get("semantic_value", {})
        parsed = (
            MeasurementSemanticsV33.model_validate(raw_semantics)
            if context_type is ContextType.MEASUREMENT
            else GenericContextSemanticValueV33.model_validate(raw_semantics)
        )
        return {**value, "context_type": context_type, "semantic_value": parsed}

    @model_validator(mode="after")
    def payload_matches_context_type(self) -> SourceContextV33:
        is_measurement = isinstance(self.semantic_value, MeasurementSemanticsV33)
        is_generic = isinstance(self.semantic_value, GenericContextSemanticValueV33)
        if self.context_type is ContextType.MEASUREMENT and not is_measurement:
            raise ValueError("measurement_context_requires_typed_measurement_semantics")
        if self.context_type is not ContextType.MEASUREMENT and not is_generic:
            raise ValueError("typed_measurement_semantics_requires_measurement_context")
        return self


class SourceSemanticRelationV33(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relation_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    relation_type: SemanticRelationTypeV31
    source_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    source_kind: GraphNodeKindV31
    target_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    target_kind: GraphNodeKindV31
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=32)
    status: SemanticRelationStatusV31
    schema_version: Literal["source-context-extraction-schema-v3.3"] = (
        SOURCE_CONTEXT_EXTRACTION_SCHEMA_V33
    )
    semantic_payload: dict[str, Any] = Field(default_factory=dict)


class SourceExtractionResponseV33(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["source-context-extraction-schema-v3.3"] = (
        SOURCE_CONTEXT_EXTRACTION_SCHEMA_V33
    )
    status: Literal["RESOLVED", "UNRESOLVED"]
    source_assertions: tuple[SourceAssertionV31, ...] = Field(default=(), max_length=64)
    source_contexts: tuple[SourceContextV33, ...] = Field(default=(), max_length=128)
    source_attachments: tuple[SourceAttachmentV31, ...] = Field(default=(), max_length=256)
    source_scopes: tuple[SourceScopeV31, ...] = Field(default=(), max_length=64)
    evidence_spans: tuple[SourceEvidenceSpanV31, ...] = Field(default=(), max_length=512)
    evidence_bindings: tuple[SemanticEvidenceBindingV31, ...] = Field(default=(), max_length=1_024)
    semantic_relations: tuple[SourceSemanticRelationV33, ...] = Field(default=(), max_length=256)
    measurement_bases: tuple[MeasurementBasisV31, ...] = Field(default=(), max_length=128)
    unresolved_items: tuple[str, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def status_shape(self) -> SourceExtractionResponseV33:
        if self.status == "RESOLVED" and not self.source_assertions:
            raise ValueError("resolved_response_requires_assertion")
        if self.status == "UNRESOLVED" and not self.unresolved_items:
            raise ValueError("unresolved_response_requires_reason")
        return self
