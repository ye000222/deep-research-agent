"""v3.2 shadow contract with intrinsic Measurement admissibility."""

from __future__ import annotations

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
    SourceContextExtractionResultV31,
    SourceEvidenceSpanV31,
    SourceExtractionResponseV31,
    SourceScopeV31,
    validate_source_context_v31,
)
from app.domain.truth_condition_context import ContextType

SOURCE_CONTEXT_EXTRACTION_SCHEMA_V32: Literal["source-context-extraction-schema-v3.2"] = (
    "source-context-extraction-schema-v3.2"
)
SOURCE_CONTEXT_EXTRACTION_PROMPT_V32 = "source-context-extraction.generic.v3.2"
SOURCE_CONTEXT_EXTRACTION_DOMAIN_V32 = "source-context-extraction-domain-v3.2"


class MeasurementSemanticsV32(BaseModel):
    """v3.1 typed Measurement facts with a v3.2 intrinsic-anchor invariant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    measurement_identity: str | None = Field(default=None, min_length=1, max_length=500)
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
    def require_intrinsic_measurement_anchor(self) -> MeasurementSemanticsV32:
        if not any(
            value is not None
            for value in (
                self.measurement_identity,
                self.formal_definition,
                self.observed_value,
                self.threshold,
            )
        ):
            raise ValueError("measurement_requires_intrinsic_semantic_anchor")
        if (
            self.comparator is not None
            and self.threshold is not None
            and self.comparator != self.threshold.comparator
        ):
            raise ValueError("measurement_threshold_comparator_conflict")
        return self


class GenericContextSemanticValueV32(RootModel[dict[str, Any]]):
    """Generic non-Measurement payload retained under explicit context dispatch."""

    model_config = ConfigDict(frozen=True)


class SourceContextV32(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    context_type: ContextType
    semantic_value: MeasurementSemanticsV32 | GenericContextSemanticValueV32 = Field(
        default_factory=lambda: GenericContextSemanticValueV32({})
    )
    lexical_value: str | None = Field(default=None, max_length=2_000)
    materiality: str = Field(default="UNKNOWN", pattern=r"^(MATERIAL|DESCRIPTIVE|UNKNOWN)$")

    @model_validator(mode="before")
    @classmethod
    def dispatch_by_declared_context_type(cls, value: Any) -> Any:
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
            MeasurementSemanticsV32.model_validate(raw_semantics)
            if context_type is ContextType.MEASUREMENT
            else GenericContextSemanticValueV32.model_validate(raw_semantics)
        )
        return {**value, "context_type": context_type, "semantic_value": parsed}

    @model_validator(mode="after")
    def payload_matches_declared_context_type(self) -> SourceContextV32:
        is_measurement = isinstance(self.semantic_value, MeasurementSemanticsV32)
        is_generic = isinstance(self.semantic_value, GenericContextSemanticValueV32)
        if self.context_type is ContextType.MEASUREMENT and not is_measurement:
            raise ValueError("measurement_context_requires_typed_measurement_semantics")
        if self.context_type is not ContextType.MEASUREMENT and not is_generic:
            raise ValueError("typed_measurement_semantics_requires_measurement_context")
        return self


class SourceSemanticRelationV32(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relation_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    relation_type: SemanticRelationTypeV31
    source_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    source_kind: GraphNodeKindV31
    target_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    target_kind: GraphNodeKindV31
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=32)
    status: SemanticRelationStatusV31
    schema_version: Literal["source-context-extraction-schema-v3.2"] = (
        SOURCE_CONTEXT_EXTRACTION_SCHEMA_V32
    )
    semantic_payload: dict[str, Any] = Field(default_factory=dict)


class SourceExtractionResponseV32(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["source-context-extraction-schema-v3.2"] = (
        SOURCE_CONTEXT_EXTRACTION_SCHEMA_V32
    )
    status: Literal["RESOLVED", "UNRESOLVED"]
    source_assertions: tuple[SourceAssertionV31, ...] = Field(default=(), max_length=64)
    source_contexts: tuple[SourceContextV32, ...] = Field(default=(), max_length=128)
    source_attachments: tuple[SourceAttachmentV31, ...] = Field(default=(), max_length=256)
    source_scopes: tuple[SourceScopeV31, ...] = Field(default=(), max_length=64)
    evidence_spans: tuple[SourceEvidenceSpanV31, ...] = Field(default=(), max_length=512)
    evidence_bindings: tuple[SemanticEvidenceBindingV31, ...] = Field(default=(), max_length=1_024)
    semantic_relations: tuple[SourceSemanticRelationV32, ...] = Field(default=(), max_length=256)
    measurement_bases: tuple[MeasurementBasisV31, ...] = Field(default=(), max_length=128)
    unresolved_items: tuple[str, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def status_shape(self) -> SourceExtractionResponseV32:
        if self.status == "RESOLVED" and not self.source_assertions:
            raise ValueError("resolved_response_requires_assertion")
        if self.status == "UNRESOLVED" and not self.unresolved_items:
            raise ValueError("unresolved_response_requires_reason")
        return self


def validate_source_context_v32(
    response: SourceExtractionResponseV32, claim_text: str, *, claim_ref: str | None = None
) -> SourceContextExtractionResultV31:
    """Apply the frozen v3.1 graph checks after v3.2 semantic admission."""
    legacy_wire = response.model_dump(mode="python")
    legacy_wire["schema_version"] = "source-context-extraction-schema-v3.1"
    for relation in legacy_wire["semantic_relations"]:
        relation["schema_version"] = "source-context-extraction-schema-v3.1"
    legacy_response = SourceExtractionResponseV31.model_validate(legacy_wire)
    result = validate_source_context_v31(legacy_response, claim_text, claim_ref=claim_ref)
    return result.model_copy(
        update={
            "prompt_version": SOURCE_CONTEXT_EXTRACTION_PROMPT_V32,
            "schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V32,
        }
    )
