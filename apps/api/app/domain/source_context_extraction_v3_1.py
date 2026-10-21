"""Versioned v3.1 shadow contract for source extraction.

This module is deliberately separate from the historical v2/v3 contracts.
It validates graph structure and evidence references only; semantic support
is never inferred from structural validity.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StrictInt,
    field_validator,
    model_validator,
)

from app.domain.source_context_extraction import (
    AssertionStatus,
    SourceAttachmentRelation,
    SourceAttachmentStatus,
    SourceScopeKind,
)
from app.domain.truth_condition_context import ContextType

SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31: Literal["source-context-extraction-schema-v3.1"] = (
    "source-context-extraction-schema-v3.1"
)
SOURCE_CONTEXT_EXTRACTION_PROMPT_V31 = "source-context-extraction.generic.v3.1"
SOURCE_CONTEXT_EXTRACTION_DOMAIN_V31 = "source-context-extraction-domain-v3.1"


class GraphNodeKindV31(StrEnum):
    ASSERTION = "ASSERTION"
    CONTEXT = "CONTEXT"
    ATTACHMENT = "ATTACHMENT"
    SCOPE = "SCOPE"
    RELATION = "RELATION"


class SemanticEvidenceRoleV31(StrEnum):
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
    MEASUREMENT_IDENTITY = "MEASUREMENT_IDENTITY"
    MEASUREMENT_DEFINITION = "MEASUREMENT_DEFINITION"
    MEASUREMENT_VALUE = "MEASUREMENT_VALUE"
    MEASUREMENT_UNIT = "MEASUREMENT_UNIT"
    MEASUREMENT_COMPARATOR = "MEASUREMENT_COMPARATOR"
    MEASUREMENT_BASIS = "MEASUREMENT_BASIS"
    MEASUREMENT_TARGET = "MEASUREMENT_TARGET"
    MEASUREMENT_QUALIFIER = "MEASUREMENT_QUALIFIER"
    ATTACHMENT = "ATTACHMENT"
    SEMANTIC_RELATION = "SEMANTIC_RELATION"


class SemanticRelationTypeV31(StrEnum):
    CAUSES = "CAUSES"
    RESULTS_IN = "RESULTS_IN"
    PURPOSE = "PURPOSE"
    COMPATIBLE_WITH = "COMPATIBLE_WITH"
    UNKNOWN = "UNKNOWN"


class SemanticRelationStatusV31(StrEnum):
    RESOLVED = "RESOLVED"
    UNKNOWN = "UNKNOWN"


class MeasurementBasisKindV31(StrEnum):
    DENOMINATOR = "DENOMINATOR"
    NORMALIZATION = "NORMALIZATION"
    UNIT_OF_ANALYSIS = "UNIT_OF_ANALYSIS"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class ScopeResolutionStatusV31(StrEnum):
    RESOLVED = "RESOLVED"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"


class ObservedValueComparatorV31(StrEnum):
    EQ = "EQ"
    LT = "LT"
    LTE = "LTE"
    GT = "GT"
    GTE = "GTE"


class IntegerObservedValueV31(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["INTEGER"]
    raw_value: str = Field(min_length=1, max_length=500)
    value: StrictInt


class DecimalObservedValueV31(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["DECIMAL"]
    raw_value: str = Field(min_length=1, max_length=500)
    value: Decimal

    @field_validator("value", mode="before")
    @classmethod
    def reject_boolean_numeric_value(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("boolean_is_not_numeric_measurement")
        return value


class RangeObservedValueV31(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["RANGE"]
    raw_value: str = Field(min_length=1, max_length=500)
    lower_bound: Decimal
    upper_bound: Decimal

    @field_validator("lower_bound", "upper_bound", mode="before")
    @classmethod
    def reject_boolean_bounds(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("boolean_is_not_numeric_measurement")
        return value

    @model_validator(mode="after")
    def ordered_bounds(self) -> RangeObservedValueV31:
        if self.lower_bound > self.upper_bound:
            raise ValueError("range_lower_bound_exceeds_upper_bound")
        return self


class ApproximateObservedValueV31(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["APPROXIMATE"]
    raw_value: str = Field(min_length=1, max_length=500)
    value: Decimal

    @field_validator("value", mode="before")
    @classmethod
    def reject_boolean_numeric_value(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("boolean_is_not_numeric_measurement")
        return value


class BoundObservedValueV31(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["BOUND"]
    raw_value: str = Field(min_length=1, max_length=500)
    comparator: ObservedValueComparatorV31
    value: Decimal

    @field_validator("value", mode="before")
    @classmethod
    def reject_boolean_numeric_value(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("boolean_is_not_numeric_measurement")
        return value


class TextObservedValueV31(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["TEXT"]
    raw_value: str = Field(min_length=1, max_length=500)
    value: str = Field(min_length=1, max_length=500)


type ObservedValueV31 = Annotated[
    IntegerObservedValueV31
    | DecimalObservedValueV31
    | RangeObservedValueV31
    | ApproximateObservedValueV31
    | BoundObservedValueV31
    | TextObservedValueV31,
    Field(discriminator="kind"),
]


class MeasurementThresholdV31(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    comparator: ObservedValueComparatorV31
    value: ObservedValueV31
    raw_text: str = Field(min_length=1, max_length=500)


class MeasurementSemanticsV31(BaseModel):
    """Typed, source-supported measurement facts; every semantic field is optional."""

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
    def require_some_measurement_semantics(self) -> MeasurementSemanticsV31:
        fields = (
            self.measurement_identity,
            self.formal_definition,
            self.observed_value,
            self.unit,
            self.comparator,
            self.threshold,
            self.basis_ref,
            self.scope_category,
            self.target_object,
        )
        if not any(value is not None for value in fields) and not self.qualifiers:
            raise ValueError("measurement_context_requires_at_least_one_typed_fact")
        if (
            self.comparator is not None
            and self.threshold is not None
            and self.comparator is not self.threshold.comparator
        ):
            raise ValueError("measurement_threshold_comparator_conflict")
        return self


class GenericContextSemanticValueV31(RootModel[dict[str, Any]]):
    """Explicit domain branch for non-measurement context payloads.

    The root representation serializes as the original JSON object, preserving
    the v3.1 wire contract while making the selected parser branch explicit.
    """

    model_config = ConfigDict(frozen=True)


class SourceAssertionV31(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_assertion_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=32)
    assertion_status: AssertionStatus
    source_scope_ref: str | None = Field(default=None, max_length=64)


class SourceContextV31(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    context_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    context_type: ContextType
    semantic_value: MeasurementSemanticsV31 | GenericContextSemanticValueV31 = Field(
        default_factory=lambda: GenericContextSemanticValueV31({})
    )
    lexical_value: str | None = Field(default=None, max_length=2_000)
    materiality: str = Field(default="UNKNOWN", pattern=r"^(MATERIAL|DESCRIPTIVE|UNKNOWN)$")

    @model_validator(mode="before")
    @classmethod
    def dispatch_semantic_value_by_context_type(cls, value: Any) -> Any:
        """Select exactly one payload parser using the canonical discriminator."""
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            return value

        context_type_value = value.get("context_type")
        if context_type_value is None:
            # Leave the required-field error to Pydantic; never infer a type.
            return value
        try:
            context_type = ContextType(context_type_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported_context_type_discriminator") from exc

        raw_semantics = value.get("semantic_value", {})
        if context_type is ContextType.MEASUREMENT:
            parsed_semantics: MeasurementSemanticsV31 | GenericContextSemanticValueV31 = (
                MeasurementSemanticsV31.model_validate(raw_semantics)
            )
        else:
            parsed_semantics = GenericContextSemanticValueV31.model_validate(raw_semantics)

        dispatched = dict(value)
        dispatched["context_type"] = context_type
        dispatched["semantic_value"] = parsed_semantics
        return dispatched

    @model_validator(mode="after")
    def measurement_uses_typed_payload(self) -> SourceContextV31:
        measurement = isinstance(self.semantic_value, MeasurementSemanticsV31)
        generic = isinstance(self.semantic_value, GenericContextSemanticValueV31)
        if self.context_type is ContextType.MEASUREMENT and not measurement:
            raise ValueError("measurement_context_requires_typed_measurement_semantics")
        if self.context_type is not ContextType.MEASUREMENT and not generic:
            raise ValueError("typed_measurement_semantics_requires_measurement_context")
        return self


class SourceAttachmentV31(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    attachment_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    context_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    source_assertion_refs: tuple[str, ...] = Field(default=(), max_length=64)
    relation: SourceAttachmentRelation
    status: SourceAttachmentStatus
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=32)
    source_scope_ref: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def resolved_attachment_has_assertions(self) -> SourceAttachmentV31:
        if self.status is SourceAttachmentStatus.RESOLVED and not self.source_assertion_refs:
            raise ValueError("resolved_attachment_requires_assertion_refs")
        if self.status is SourceAttachmentStatus.RESOLVED and not self.evidence_refs:
            raise ValueError("resolved_attachment_requires_evidence_refs")
        return self


class SourceScopeV31(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source_scope_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    kind: SourceScopeKind
    source_assertion_refs: tuple[str, ...] = Field(min_length=1, max_length=64)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=32)
    resolution_status: ScopeResolutionStatusV31 = ScopeResolutionStatusV31.RESOLVED


class SourceEvidenceSpanV31(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    evidence_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    exact_text: str = Field(min_length=1, max_length=4_000)
    occurrence_index: int = Field(default=0, ge=0, le=1_000)


class SemanticEvidenceBindingV31(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    binding_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    evidence_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    semantic_role: SemanticEvidenceRoleV31
    target_kind: GraphNodeKindV31
    target_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    related_binding_ref: str | None = Field(default=None, max_length=64)


class SourceSemanticRelationV31(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    relation_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    relation_type: SemanticRelationTypeV31
    source_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    source_kind: GraphNodeKindV31
    target_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    target_kind: GraphNodeKindV31
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=32)
    status: SemanticRelationStatusV31
    schema_version: Literal["source-context-extraction-schema-v3.1"] = (
        SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31
    )
    semantic_payload: dict[str, Any] = Field(default_factory=dict)


class MeasurementBasisV31(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    basis_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    measurement_context_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    basis_kind: MeasurementBasisKindV31
    semantic_value: dict[str, Any] = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=32)


class SourceExtractionResponseV31(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["source-context-extraction-schema-v3.1"] = (
        SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31
    )
    status: Literal["RESOLVED", "UNRESOLVED"]
    source_assertions: tuple[SourceAssertionV31, ...] = Field(default=(), max_length=64)
    source_contexts: tuple[SourceContextV31, ...] = Field(default=(), max_length=128)
    source_attachments: tuple[SourceAttachmentV31, ...] = Field(default=(), max_length=256)
    source_scopes: tuple[SourceScopeV31, ...] = Field(default=(), max_length=64)
    evidence_spans: tuple[SourceEvidenceSpanV31, ...] = Field(default=(), max_length=512)
    evidence_bindings: tuple[SemanticEvidenceBindingV31, ...] = Field(default=(), max_length=1_024)
    semantic_relations: tuple[SourceSemanticRelationV31, ...] = Field(default=(), max_length=256)
    measurement_bases: tuple[MeasurementBasisV31, ...] = Field(default=(), max_length=128)
    unresolved_items: tuple[str, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def status_shape(self) -> SourceExtractionResponseV31:
        if self.status == "RESOLVED" and not self.source_assertions:
            raise ValueError("resolved_response_requires_assertion")
        if self.status == "UNRESOLVED" and not self.unresolved_items:
            raise ValueError("unresolved_response_requires_reason")
        return self


class V31ValidationStatus(StrEnum):
    RESOLVED = "RESOLVED"
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"
    UNRESOLVED = "UNRESOLVED"


class ScopeOwnershipAssessmentV31(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: V31ValidationStatus
    diagnostics: tuple[str, ...] = ()


class V31BindingAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    binding_ref: str
    evidence_ref: str
    semantic_role: SemanticEvidenceRoleV31
    target_kind: GraphNodeKindV31
    target_ref: str
    textual_grounding: str
    structural_grounding: str
    semantic_support: Literal["NOT_EVALUATED"] = "NOT_EVALUATED"
    diagnostics: tuple[str, ...] = ()


class SourceContextExtractionResultV31(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    claim_ref: str | None = None
    claim_hash: str
    status: V31ValidationStatus
    diagnostics: tuple[str, ...] = ()
    source_assertions: tuple[SourceAssertionV31, ...] = ()
    source_contexts: tuple[SourceContextV31, ...] = ()
    source_attachments: tuple[SourceAttachmentV31, ...] = ()
    source_scopes: tuple[SourceScopeV31, ...] = ()
    evidence_spans: tuple[SourceEvidenceSpanV31, ...] = ()
    evidence_bindings: tuple[V31BindingAssessment, ...] = ()
    semantic_relations: tuple[SourceSemanticRelationV31, ...] = ()
    measurement_bases: tuple[MeasurementBasisV31, ...] = ()
    semantic_support: Literal["NOT_EVALUATED"] = "NOT_EVALUATED"
    prompt_version: str = SOURCE_CONTEXT_EXTRACTION_PROMPT_V31
    schema_version: str = SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31
    model_call_count: int = Field(default=0, ge=0, le=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)


def _binding_target(role: SemanticEvidenceRoleV31) -> GraphNodeKindV31:
    if role is SemanticEvidenceRoleV31.ASSERTION_EVIDENCE:
        return GraphNodeKindV31.ASSERTION
    if role is SemanticEvidenceRoleV31.ATTACHMENT:
        return GraphNodeKindV31.ATTACHMENT
    if role is SemanticEvidenceRoleV31.SEMANTIC_RELATION:
        return GraphNodeKindV31.RELATION
    return GraphNodeKindV31.CONTEXT


def _measurement_required_roles(
    semantics: MeasurementSemanticsV31,
) -> tuple[SemanticEvidenceRoleV31, ...]:
    roles: list[SemanticEvidenceRoleV31] = []
    if semantics.measurement_identity is not None:
        roles.append(SemanticEvidenceRoleV31.MEASUREMENT_IDENTITY)
    if semantics.formal_definition is not None:
        roles.append(SemanticEvidenceRoleV31.MEASUREMENT_DEFINITION)
    if semantics.observed_value is not None:
        roles.append(SemanticEvidenceRoleV31.MEASUREMENT_VALUE)
    if semantics.unit is not None:
        roles.append(SemanticEvidenceRoleV31.MEASUREMENT_UNIT)
    if semantics.comparator is not None or semantics.threshold is not None:
        roles.append(SemanticEvidenceRoleV31.MEASUREMENT_COMPARATOR)
    if semantics.basis_ref is not None:
        roles.append(SemanticEvidenceRoleV31.MEASUREMENT_BASIS)
    if semantics.scope_category is not None:
        roles.append(SemanticEvidenceRoleV31.SCOPE_VALUE)
    if semantics.target_object is not None:
        roles.append(SemanticEvidenceRoleV31.MEASUREMENT_TARGET)
    if semantics.qualifiers:
        roles.append(SemanticEvidenceRoleV31.MEASUREMENT_QUALIFIER)
    return tuple(roles)


def validate_scope_ownership_v31(
    response: SourceExtractionResponseV31, claim_text: str
) -> ScopeOwnershipAssessmentV31:
    """Validate only canonical scope graph ownership for offline scope replay."""
    valid_evidence: set[str] = set()
    for span in response.evidence_spans:
        matches = [
            offset
            for offset in range(len(claim_text))
            if claim_text.startswith(span.exact_text, offset)
        ]
        if span.occurrence_index < len(matches):
            valid_evidence.add(span.evidence_ref)
    assertion_refs = {item.source_assertion_ref for item in response.source_assertions}
    diagnostics: list[str] = []
    unresolved = False
    for scope in response.source_scopes:
        if len(scope.source_assertion_refs) != len(set(scope.source_assertion_refs)):
            diagnostics.append("scope_duplicate_member_ref")
        if any(ref not in assertion_refs for ref in scope.source_assertion_refs):
            diagnostics.append("scope_member_ref_invalid")
        if not scope.evidence_refs or any(ref not in valid_evidence for ref in scope.evidence_refs):
            diagnostics.append("scope_evidence_ref_invalid")
        if scope.resolution_status is not ScopeResolutionStatusV31.RESOLVED:
            unresolved = True
            diagnostics.append(f"scope_{scope.resolution_status.value.lower()}")
    unique = tuple(dict.fromkeys(diagnostics))
    invalid = any(
        item.endswith("_invalid") or item == "scope_duplicate_member_ref" for item in unique
    )
    status = (
        V31ValidationStatus.INVALID
        if invalid
        else V31ValidationStatus.PARTIAL
        if unresolved
        else V31ValidationStatus.RESOLVED
    )
    return ScopeOwnershipAssessmentV31(status=status, diagnostics=unique)


def validate_source_context_v31(
    response: SourceExtractionResponseV31, claim_text: str, *, claim_ref: str | None = None
) -> SourceContextExtractionResultV31:
    """Check exact spans, references, canonical scope ownership and typed roles only."""
    import hashlib

    diagnostics: list[str] = []
    claim_hash = hashlib.sha256(claim_text.encode("utf-8")).hexdigest()
    valid_evidence: set[str] = set()
    for span in response.evidence_spans:
        occurrences = sum(
            1 for offset in range(len(claim_text)) if claim_text.startswith(span.exact_text, offset)
        )
        if span.occurrence_index < occurrences:
            valid_evidence.add(span.evidence_ref)
        else:
            diagnostics.append("evidence_span_not_in_claim")

    refs_by_kind: dict[GraphNodeKindV31, set[str]] = {
        GraphNodeKindV31.ASSERTION: {
            item.source_assertion_ref for item in response.source_assertions
        },
        GraphNodeKindV31.CONTEXT: {item.context_ref for item in response.source_contexts},
        GraphNodeKindV31.ATTACHMENT: {item.attachment_ref for item in response.source_attachments},
        GraphNodeKindV31.SCOPE: {item.source_scope_ref for item in response.source_scopes},
        GraphNodeKindV31.RELATION: {item.relation_ref for item in response.semantic_relations},
    }
    all_refs: list[str] = [ref for values in refs_by_kind.values() for ref in values]
    all_refs.extend(span.evidence_ref for span in response.evidence_spans)
    all_refs.extend(binding.binding_ref for binding in response.evidence_bindings)
    all_refs.extend(basis.basis_ref for basis in response.measurement_bases)
    if len(all_refs) != len(set(all_refs)):
        diagnostics.append("duplicate_reference")

    def node_exists(kind: GraphNodeKindV31, ref: str) -> bool:
        return ref in refs_by_kind[kind]

    bindings = response.evidence_bindings
    binding_keys = {(item.target_kind, item.target_ref, item.semantic_role) for item in bindings}
    assessments: list[V31BindingAssessment] = []
    for binding in bindings:
        local: list[str] = []
        textual = binding.evidence_ref in valid_evidence
        structural = node_exists(binding.target_kind, binding.target_ref)
        if not textual:
            local.append("binding_evidence_ref_invalid")
            diagnostics.append("binding_evidence_ref_invalid")
        if not structural:
            local.append("binding_target_ref_invalid")
            diagnostics.append("binding_target_ref_invalid")
        if binding.target_kind is not _binding_target(binding.semantic_role):
            local.append("binding_role_target_mismatch")
            diagnostics.append("binding_role_target_mismatch")
        assessments.append(
            V31BindingAssessment(
                binding_ref=binding.binding_ref,
                evidence_ref=binding.evidence_ref,
                semantic_role=binding.semantic_role,
                target_kind=binding.target_kind,
                target_ref=binding.target_ref,
                textual_grounding="VALID" if textual else "INVALID",
                structural_grounding="VALID" if structural else "INVALID",
                diagnostics=tuple(local),
            )
        )

    for assertion in response.source_assertions:
        if any(ref not in valid_evidence for ref in assertion.evidence_refs):
            diagnostics.append("assertion_evidence_ref_invalid")
        if not any(
            binding.target_kind is GraphNodeKindV31.ASSERTION
            and binding.target_ref == assertion.source_assertion_ref
            and binding.semantic_role is SemanticEvidenceRoleV31.ASSERTION_EVIDENCE
            and binding.evidence_ref in assertion.evidence_refs
            for binding in bindings
        ):
            diagnostics.append("assertion_evidence_binding_missing")
        if assertion.source_scope_ref:
            matching = next(
                (
                    scope
                    for scope in response.source_scopes
                    if scope.source_scope_ref == assertion.source_scope_ref
                ),
                None,
            )
            if (
                matching is None
                or assertion.source_assertion_ref not in matching.source_assertion_refs
            ):
                diagnostics.append("assertion_scope_ref_invalid")

    scope_assessment = validate_scope_ownership_v31(response, claim_text)
    scope_partial = scope_assessment.status is V31ValidationStatus.PARTIAL
    diagnostics.extend(scope_assessment.diagnostics)

    for attachment in response.source_attachments:
        if attachment.context_ref not in refs_by_kind[GraphNodeKindV31.CONTEXT]:
            diagnostics.append("attachment_context_ref_invalid")
        if any(
            ref not in refs_by_kind[GraphNodeKindV31.ASSERTION]
            for ref in attachment.source_assertion_refs
        ):
            diagnostics.append("attachment_assertion_ref_invalid")
        if any(ref not in valid_evidence for ref in attachment.evidence_refs):
            diagnostics.append("attachment_evidence_ref_invalid")
        if (
            attachment.source_scope_ref
            and attachment.source_scope_ref not in refs_by_kind[GraphNodeKindV31.SCOPE]
        ):
            diagnostics.append("attachment_scope_ref_invalid")
        if attachment.source_scope_ref:
            scope = next(
                (
                    item
                    for item in response.source_scopes
                    if item.source_scope_ref == attachment.source_scope_ref
                ),
                None,
            )
            if scope is not None and any(
                ref not in scope.source_assertion_refs for ref in attachment.source_assertion_refs
            ):
                diagnostics.append("attachment_scope_membership_invalid")
        if (
            attachment.status is SourceAttachmentStatus.RESOLVED
            and attachment.evidence_refs
            and not any(
                binding.target_kind is GraphNodeKindV31.ATTACHMENT
                and binding.target_ref == attachment.attachment_ref
                and binding.semantic_role is SemanticEvidenceRoleV31.ATTACHMENT
                and binding.evidence_ref in attachment.evidence_refs
                for binding in bindings
            )
        ):
            diagnostics.append("attachment_binding_missing")

    role_semantic_fields: dict[tuple[str, SemanticEvidenceRoleV31], bool] = {}
    for context in response.source_contexts:
        if context.context_type is ContextType.MEASUREMENT:
            semantics = context.semantic_value
            assert isinstance(semantics, MeasurementSemanticsV31)
            for role in _measurement_required_roles(semantics):
                role_semantic_fields[(context.context_ref, role)] = True
                if (GraphNodeKindV31.CONTEXT, context.context_ref, role) not in binding_keys:
                    diagnostics.append(f"measurement_binding_missing:{role.value.lower()}")
            if semantics.basis_ref and semantics.basis_ref not in {
                b.basis_ref for b in response.measurement_bases
            }:
                diagnostics.append("measurement_basis_ref_invalid")
            if semantics.basis_ref:
                basis = next(
                    (
                        item
                        for item in response.measurement_bases
                        if item.basis_ref == semantics.basis_ref
                    ),
                    None,
                )
                if basis is not None and basis.measurement_context_ref != context.context_ref:
                    diagnostics.append("measurement_basis_context_mismatch")
            if not any(
                edge.context_ref == context.context_ref for edge in response.source_attachments
            ):
                diagnostics.append("measurement_context_attachment_missing")
        else:
            if not any(
                edge.context_ref == context.context_ref for edge in response.source_attachments
            ):
                diagnostics.append("context_attachment_missing")

    for binding in bindings:
        if (
            binding.target_kind is GraphNodeKindV31.CONTEXT
            and binding.semantic_role
            in {
                SemanticEvidenceRoleV31.MEASUREMENT_IDENTITY,
                SemanticEvidenceRoleV31.MEASUREMENT_DEFINITION,
                SemanticEvidenceRoleV31.MEASUREMENT_VALUE,
                SemanticEvidenceRoleV31.MEASUREMENT_UNIT,
                SemanticEvidenceRoleV31.MEASUREMENT_COMPARATOR,
                SemanticEvidenceRoleV31.MEASUREMENT_BASIS,
                SemanticEvidenceRoleV31.MEASUREMENT_TARGET,
                SemanticEvidenceRoleV31.MEASUREMENT_QUALIFIER,
            }
            and (binding.target_ref, binding.semantic_role) not in role_semantic_fields
        ):
            diagnostics.append(
                f"measurement_role_without_matching_field:{binding.semantic_role.value.lower()}"
            )
        if binding.related_binding_ref and binding.related_binding_ref not in {
            b.binding_ref for b in bindings
        }:
            diagnostics.append("related_binding_ref_invalid")

    for relation in response.semantic_relations:
        if not node_exists(relation.source_kind, relation.source_ref):
            diagnostics.append("relation_source_ref_invalid")
        if not node_exists(relation.target_kind, relation.target_ref):
            diagnostics.append("relation_target_ref_invalid")
        if any(ref not in valid_evidence for ref in relation.evidence_refs):
            diagnostics.append("relation_evidence_ref_invalid")
        if relation.status is SemanticRelationStatusV31.RESOLVED and not any(
            binding.target_kind is GraphNodeKindV31.RELATION
            and binding.target_ref == relation.relation_ref
            and binding.semantic_role is SemanticEvidenceRoleV31.SEMANTIC_RELATION
            and binding.evidence_ref in relation.evidence_refs
            for binding in bindings
        ):
            diagnostics.append("relation_binding_missing")
        if relation.status is SemanticRelationStatusV31.UNKNOWN:
            scope_partial = True
            diagnostics.append("semantic_relation_unknown")

    for basis in response.measurement_bases:
        if basis.measurement_context_ref not in refs_by_kind[GraphNodeKindV31.CONTEXT]:
            diagnostics.append("measurement_basis_context_ref_invalid")
        if any(ref not in valid_evidence for ref in basis.evidence_refs):
            diagnostics.append("measurement_basis_evidence_ref_invalid")
        if not any(
            binding.target_kind is GraphNodeKindV31.CONTEXT
            and binding.target_ref == basis.measurement_context_ref
            and binding.semantic_role is SemanticEvidenceRoleV31.MEASUREMENT_BASIS
            and binding.evidence_ref in basis.evidence_refs
            for binding in bindings
        ):
            diagnostics.append("measurement_basis_binding_missing")

    unique = tuple(dict.fromkeys(diagnostics))
    invalid_markers = (
        "_invalid",
        "_missing",
        "duplicate_reference",
        "scope_duplicate_member_ref",
        "measurement_role_without_matching_field",
        "measurement_binding_missing",
    )
    invalid = any(
        item.endswith(invalid_markers)
        or item.startswith("measurement_binding_missing:")
        or item.startswith("measurement_role_without_matching_field:")
        for item in unique
    )
    if response.status == "UNRESOLVED" and not invalid:
        status = V31ValidationStatus.UNRESOLVED
    elif invalid:
        status = V31ValidationStatus.INVALID
    elif scope_partial:
        status = V31ValidationStatus.PARTIAL
    else:
        status = V31ValidationStatus.RESOLVED

    return SourceContextExtractionResultV31(
        claim_ref=claim_ref,
        claim_hash=claim_hash,
        status=status,
        diagnostics=unique,
        source_assertions=response.source_assertions,
        source_contexts=response.source_contexts,
        source_attachments=response.source_attachments,
        source_scopes=response.source_scopes,
        evidence_spans=response.evidence_spans,
        evidence_bindings=tuple(assessments),
        semantic_relations=response.semantic_relations,
        measurement_bases=response.measurement_bases,
    )
