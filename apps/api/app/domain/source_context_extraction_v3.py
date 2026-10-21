"""Versioned v3 source-semantic extraction shadow contract.

This module is isolated from the v2 production/shadow contracts. Structural
grounding is deterministic; semantic entailment is deliberately not inferred.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.source_context_extraction import (
    AssertionStatus,
    SourceAttachmentRelation,
    SourceAttachmentStatus,
    SourceScopeKind,
)
from app.domain.source_context_grounding_v2 import GroundingDimensionStatus
from app.domain.truth_condition_context import ContextType

SOURCE_CONTEXT_EXTRACTION_SCHEMA_V3 = "source-context-extraction-schema-v3"
SOURCE_CONTEXT_EXTRACTION_PROMPT_V3 = "source-context-extraction.generic.v3"


class GraphNodeKindV3(StrEnum):
    ASSERTION = "ASSERTION"
    CONTEXT = "CONTEXT"
    ATTACHMENT = "ATTACHMENT"
    SCOPE = "SCOPE"
    RELATION = "RELATION"


class SemanticRelationTypeV3(StrEnum):
    CAUSES = "CAUSES"
    RESULTS_IN = "RESULTS_IN"
    PURPOSE = "PURPOSE"
    COMPATIBLE_WITH = "COMPATIBLE_WITH"
    UNKNOWN = "UNKNOWN"


class SemanticRelationStatusV3(StrEnum):
    RESOLVED = "RESOLVED"
    UNKNOWN = "UNKNOWN"


class MeasurementBasisKindV3(StrEnum):
    DENOMINATOR = "DENOMINATOR"
    NORMALIZATION = "NORMALIZATION"
    UNIT_OF_ANALYSIS = "UNIT_OF_ANALYSIS"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class SemanticEvidenceRoleV3(StrEnum):
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
    MEASUREMENT_BASIS = "MEASUREMENT_BASIS"
    ATTACHMENT = "ATTACHMENT"
    SCOPE_STRUCTURE = "SCOPE_STRUCTURE"
    SEMANTIC_RELATION = "SEMANTIC_RELATION"


class SourceAssertionV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_assertion_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=32)
    assertion_status: AssertionStatus
    source_scope_ref: str | None = Field(default=None, max_length=64)


class SourceContextV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    context_type: ContextType
    semantic_value: dict[str, Any] = Field(default_factory=dict)
    lexical_value: str | None = Field(default=None, max_length=2_000)
    materiality: str = Field(default="UNKNOWN", pattern=r"^(MATERIAL|DESCRIPTIVE|UNKNOWN)$")


class SourceAttachmentV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    attachment_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    context_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    source_assertion_refs: tuple[str, ...] = Field(default=(), max_length=64)
    relation: SourceAttachmentRelation
    status: SourceAttachmentStatus
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=32)
    source_scope_ref: str | None = Field(default=None, max_length=64)


class SourceScopeV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_scope_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    kind: SourceScopeKind
    source_assertion_refs: tuple[str, ...] = Field(default=(), max_length=64)
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=32)


class SourceEvidenceSpanV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    exact_text: str = Field(min_length=1, max_length=4_000)
    occurrence_index: int = Field(default=0, ge=0, le=1_000)


class SemanticEvidenceBindingV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    binding_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    evidence_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    semantic_role: SemanticEvidenceRoleV3
    target_kind: GraphNodeKindV3
    target_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    related_binding_ref: str | None = Field(default=None, max_length=64)


class SourceSemanticRelationV3(BaseModel):
    """Directed generic semantic edge; source_ref → target_ref is explicit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    relation_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    relation_type: SemanticRelationTypeV3
    source_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    source_kind: GraphNodeKindV3
    target_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    target_kind: GraphNodeKindV3
    evidence_refs: tuple[str, ...] = Field(default=(), max_length=32)
    status: SemanticRelationStatusV3
    schema_version: str = Field(pattern=r"^source-context-extraction-schema-v3$")
    semantic_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def resolved_relation_has_evidence(self) -> SourceSemanticRelationV3:
        if self.status is SemanticRelationStatusV3.RESOLVED and not self.evidence_refs:
            raise ValueError("resolved_semantic_relation_requires_evidence")
        return self


class MeasurementBasisV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    basis_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    measurement_context_ref: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    basis_kind: MeasurementBasisKindV3
    semantic_value: dict[str, Any] = Field(min_length=1)
    evidence_refs: tuple[str, ...] = Field(min_length=1, max_length=32)


class SourceExtractionResponseV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^source-context-extraction-schema-v3$")
    status: str = Field(pattern=r"^(RESOLVED|UNRESOLVED)$")
    source_assertions: tuple[SourceAssertionV3, ...] = Field(default=(), max_length=64)
    source_contexts: tuple[SourceContextV3, ...] = Field(default=(), max_length=128)
    source_attachments: tuple[SourceAttachmentV3, ...] = Field(default=(), max_length=256)
    source_scopes: tuple[SourceScopeV3, ...] = Field(default=(), max_length=64)
    evidence_spans: tuple[SourceEvidenceSpanV3, ...] = Field(default=(), max_length=512)
    evidence_bindings: tuple[SemanticEvidenceBindingV3, ...] = Field(default=(), max_length=1_024)
    semantic_relations: tuple[SourceSemanticRelationV3, ...] = Field(default=(), max_length=256)
    measurement_bases: tuple[MeasurementBasisV3, ...] = Field(default=(), max_length=128)
    unresolved_items: tuple[str, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def status_shape(self) -> SourceExtractionResponseV3:
        if self.status == "RESOLVED" and not self.source_assertions:
            raise ValueError("resolved_response_requires_assertion")
        if self.status == "UNRESOLVED" and not self.unresolved_items:
            raise ValueError("unresolved_response_requires_reason")
        return self


# Explicit v3 names mirror the established source graph vocabulary without
# aliasing or reinterpreting any historical v2 model.
SourceAssertionAnchorV3 = SourceAssertionV3
SourceContextNodeV3 = SourceContextV3
SourceScopeNodeV3 = SourceScopeV3


class V3ValidationStatus(StrEnum):
    RESOLVED = "RESOLVED"
    PARTIAL = "PARTIAL"
    INVALID = "INVALID"
    UNRESOLVED = "UNRESOLVED"


class V3Diagnostic(StrEnum):
    DUPLICATE_REFERENCE = "duplicate_reference"
    DANGLING_EVIDENCE_REF = "dangling_evidence_ref"
    DANGLING_TARGET_REF = "dangling_target_ref"
    ROLE_TARGET_MISMATCH = "role_target_mismatch"
    ASSERTION_EVIDENCE_MISSING = "assertion_evidence_missing"
    INVALID_SCOPE_GRAPH = "invalid_scope_graph"
    ATTACHMENT_BINDING_MISSING = "attachment_binding_missing"
    RELATION_BINDING_MISSING = "relation_binding_missing"
    EPISTEMIC_TARGET_MISMATCH = "epistemic_target_mismatch"
    METHOD_RELATION_UNLINKED = "method_relation_unlinked"
    MEASUREMENT_BASIS_REF_INVALID = "measurement_basis_ref_invalid"
    SEMANTIC_RELATION_UNRESOLVED = "semantic_relation_unresolved"


class V3BindingAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    binding_ref: str
    evidence_ref: str
    semantic_role: SemanticEvidenceRoleV3
    target_kind: GraphNodeKindV3
    target_ref: str
    textual_grounding: GroundingDimensionStatus
    structural_grounding: GroundingDimensionStatus
    semantic_support: GroundingDimensionStatus = GroundingDimensionStatus.NOT_EVALUATED


class SourceContextExtractionResultV3(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_ref: str | None
    claim_hash: str
    status: V3ValidationStatus
    response: SourceExtractionResponseV3 | None = None
    binding_assessments: tuple[V3BindingAssessment, ...] = ()
    diagnostics: tuple[str, ...] = ()
    downstream_not_yet_consumed: tuple[str, ...] = ()
    prompt_version: str = SOURCE_CONTEXT_EXTRACTION_PROMPT_V3
    schema_version: str = SOURCE_CONTEXT_EXTRACTION_SCHEMA_V3
    model_call_count: int = Field(default=0, ge=0, le=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)


_CONTEXT_ROLES: dict[str, SemanticEvidenceRoleV3] = {
    "TEMPORAL": SemanticEvidenceRoleV3.TEMPORAL_VALUE,
    "VERSION": SemanticEvidenceRoleV3.VERSION_VALUE,
    "SCOPE": SemanticEvidenceRoleV3.SCOPE_VALUE,
    "CONFIGURATION": SemanticEvidenceRoleV3.CONFIGURATION_VALUE,
    "SCENARIO": SemanticEvidenceRoleV3.SCENARIO_VALUE,
    "CONDITION": SemanticEvidenceRoleV3.CONDITION,
    "SAMPLE_SCOPE": SemanticEvidenceRoleV3.SAMPLE_SCOPE_VALUE,
}
_ROLE_TARGETS: dict[SemanticEvidenceRoleV3, GraphNodeKindV3] = {
    SemanticEvidenceRoleV3.ASSERTION_EVIDENCE: GraphNodeKindV3.ASSERTION,
    SemanticEvidenceRoleV3.ATTACHMENT: GraphNodeKindV3.ATTACHMENT,
    SemanticEvidenceRoleV3.SCOPE_STRUCTURE: GraphNodeKindV3.SCOPE,
    SemanticEvidenceRoleV3.SEMANTIC_RELATION: GraphNodeKindV3.RELATION,
}
for _role in SemanticEvidenceRoleV3:
    if _role not in _ROLE_TARGETS:
        _ROLE_TARGETS[_role] = GraphNodeKindV3.CONTEXT


def _all_refs(response: SourceExtractionResponseV3) -> dict[GraphNodeKindV3, set[str]]:
    return {
        GraphNodeKindV3.ASSERTION: {x.source_assertion_ref for x in response.source_assertions},
        GraphNodeKindV3.CONTEXT: {x.context_ref for x in response.source_contexts},
        GraphNodeKindV3.ATTACHMENT: {x.attachment_ref for x in response.source_attachments},
        GraphNodeKindV3.SCOPE: {x.source_scope_ref for x in response.source_scopes},
        GraphNodeKindV3.RELATION: {x.relation_ref for x in response.semantic_relations},
    }


def validate_source_context_v3(
    response: SourceExtractionResponseV3, claim_text: str, *, claim_ref: str | None = None
) -> SourceContextExtractionResultV3:
    """Validate graph/ref/binding completeness without asserting semantic entailment."""
    diagnostics: list[str] = []
    claim_hash = hashlib.sha256(claim_text.encode("utf-8")).hexdigest()
    refs = _all_refs(response)
    refs_by_kind = {kind: set(values) for kind, values in refs.items()}
    # RELATION is an evidence target kind local to v3; v2 enums stay unchanged.
    known_kind_values = {kind.value for kind in refs_by_kind}
    all_object_refs: list[str] = []
    for node_refs in refs_by_kind.values():
        all_object_refs.extend(node_refs)
    all_object_refs.extend(x.basis_ref for x in response.measurement_bases)
    all_object_refs.extend(x.evidence_ref for x in response.evidence_spans)
    all_object_refs.extend(x.binding_ref for x in response.evidence_bindings)
    if len(all_object_refs) != len(set(all_object_refs)):
        diagnostics.append(V3Diagnostic.DUPLICATE_REFERENCE.value)
    spans = {x.evidence_ref: x for x in response.evidence_spans}
    valid_evidence = {
        span.evidence_ref
        for span in response.evidence_spans
        if span.occurrence_index
        < sum(
            1 for offset in range(len(claim_text)) if claim_text.startswith(span.exact_text, offset)
        )
    }
    if len(valid_evidence) != len(spans):
        diagnostics.append(V3Diagnostic.DANGLING_EVIDENCE_REF.value)

    def target_exists(kind: str, ref: str) -> bool:
        return kind in known_kind_values and ref in refs_by_kind[GraphNodeKindV3(kind)]

    bindings = response.evidence_bindings
    binding_keys = {(b.target_kind.value, b.target_ref, b.semantic_role) for b in bindings}
    assessments: list[V3BindingAssessment] = []
    for binding in bindings:
        structural = GroundingDimensionStatus.VALID
        if binding.evidence_ref not in valid_evidence or not target_exists(
            binding.target_kind.value, binding.target_ref
        ):
            structural = GroundingDimensionStatus.INVALID
            diagnostics.append(V3Diagnostic.DANGLING_TARGET_REF.value)
        expected_kind = _ROLE_TARGETS[binding.semantic_role]
        if binding.semantic_role is SemanticEvidenceRoleV3.SEMANTIC_RELATION:
            expected_kind_value = "RELATION"
        else:
            expected_kind_value = expected_kind.value
        if binding.target_kind.value != expected_kind_value:
            structural = GroundingDimensionStatus.INVALID
            diagnostics.append(V3Diagnostic.ROLE_TARGET_MISMATCH.value)
        if (
            binding.semantic_role
            in {
                SemanticEvidenceRoleV3.EPISTEMIC_OPERATOR,
                SemanticEvidenceRoleV3.EPISTEMIC_AGENT,
            }
            and binding.target_kind is not GraphNodeKindV3.CONTEXT
        ):
            diagnostics.append(V3Diagnostic.EPISTEMIC_TARGET_MISMATCH.value)
        assessments.append(
            V3BindingAssessment(
                binding_ref=binding.binding_ref,
                evidence_ref=binding.evidence_ref,
                semantic_role=binding.semantic_role,
                target_kind=binding.target_kind,
                target_ref=binding.target_ref,
                textual_grounding=(
                    GroundingDimensionStatus.VALID
                    if binding.evidence_ref in valid_evidence
                    else GroundingDimensionStatus.MISSING_EVIDENCE
                ),
                structural_grounding=structural,
            )
        )

    for assertion in response.source_assertions:
        if not assertion.evidence_refs or any(
            x not in valid_evidence for x in assertion.evidence_refs
        ):
            diagnostics.append(V3Diagnostic.ASSERTION_EVIDENCE_MISSING.value)
        if not any(
            b.target_kind is GraphNodeKindV3.ASSERTION
            and b.target_ref == assertion.source_assertion_ref
            and b.semantic_role is SemanticEvidenceRoleV3.ASSERTION_EVIDENCE
            and b.evidence_ref in assertion.evidence_refs
            for b in bindings
        ):
            diagnostics.append(V3Diagnostic.ASSERTION_EVIDENCE_MISSING.value)
        if (
            assertion.source_scope_ref
            and assertion.source_scope_ref not in refs[GraphNodeKindV3.SCOPE]
        ):
            diagnostics.append(V3Diagnostic.INVALID_SCOPE_GRAPH.value)
    for scope in response.source_scopes:
        if (
            not scope.source_assertion_refs
            or any(x not in refs[GraphNodeKindV3.ASSERTION] for x in scope.source_assertion_refs)
            or any(x not in valid_evidence for x in scope.evidence_refs)
        ):
            diagnostics.append(V3Diagnostic.INVALID_SCOPE_GRAPH.value)
        if len(scope.source_assertion_refs) > 1 and not any(
            b.target_kind is GraphNodeKindV3.SCOPE
            and b.target_ref == scope.source_scope_ref
            and b.semantic_role is SemanticEvidenceRoleV3.SCOPE_STRUCTURE
            and b.evidence_ref in scope.evidence_refs
            for b in bindings
        ):
            diagnostics.append(V3Diagnostic.INVALID_SCOPE_GRAPH.value)
    for attachment in response.source_attachments:
        if (
            attachment.context_ref not in refs[GraphNodeKindV3.CONTEXT]
            or (
                attachment.status is SourceAttachmentStatus.RESOLVED
                and not attachment.source_assertion_refs
            )
            or any(
                x not in refs[GraphNodeKindV3.ASSERTION] for x in attachment.source_assertion_refs
            )
            or any(x not in valid_evidence for x in attachment.evidence_refs)
            or (
                attachment.source_scope_ref is not None
                and attachment.source_scope_ref not in refs[GraphNodeKindV3.SCOPE]
            )
        ):
            diagnostics.append(V3Diagnostic.DANGLING_TARGET_REF.value)
        if (
            attachment.status is SourceAttachmentStatus.RESOLVED
            and attachment.evidence_refs
            and not any(
                b.target_kind is GraphNodeKindV3.ATTACHMENT
                and b.target_ref == attachment.attachment_ref
                and b.semantic_role is SemanticEvidenceRoleV3.ATTACHMENT
                and b.evidence_ref in attachment.evidence_refs
                for b in bindings
            )
        ):
            diagnostics.append(V3Diagnostic.ATTACHMENT_BINDING_MISSING.value)
    for relation in response.semantic_relations:
        if (
            not target_exists(relation.source_kind.value, relation.source_ref)
            or not target_exists(relation.target_kind.value, relation.target_ref)
            or any(x not in valid_evidence for x in relation.evidence_refs)
        ):
            diagnostics.append(V3Diagnostic.DANGLING_TARGET_REF.value)
        if relation.status is SemanticRelationStatusV3.RESOLVED and not any(
            b.target_kind.value == "RELATION"
            and b.target_ref == relation.relation_ref
            and b.semantic_role is SemanticEvidenceRoleV3.SEMANTIC_RELATION
            and b.evidence_ref in relation.evidence_refs
            for b in bindings
        ):
            diagnostics.append(V3Diagnostic.RELATION_BINDING_MISSING.value)
        if relation.status is SemanticRelationStatusV3.UNKNOWN:
            diagnostics.append(V3Diagnostic.SEMANTIC_RELATION_UNRESOLVED.value)
    for context in response.source_contexts:
        semantic = context.semantic_value
        required: list[SemanticEvidenceRoleV3] = []
        if context.context_type is ContextType.EPISTEMIC:
            if any(k in semantic for k in ("operator", "epistemic_operator")):
                required.append(SemanticEvidenceRoleV3.EPISTEMIC_OPERATOR)
            if any(k in semantic for k in ("agent", "source", "attribution")):
                required.append(SemanticEvidenceRoleV3.EPISTEMIC_AGENT)
            if any(k in semantic for k in ("modality", "certainty")):
                required.append(SemanticEvidenceRoleV3.MODALITY)
        if context.context_type is ContextType.METHOD:
            if any(k in semantic for k in ("method", "value", "tool", "module")):
                required.append(SemanticEvidenceRoleV3.METHOD_VALUE)
            if any(k in semantic for k in ("method_relation", "relation", "operation")):
                required.append(SemanticEvidenceRoleV3.METHOD_RELATION)
        if context.context_type is ContextType.MEASUREMENT:
            if "value" in semantic:
                required.append(SemanticEvidenceRoleV3.MEASUREMENT_VALUE)
            if "unit" in semantic:
                required.append(SemanticEvidenceRoleV3.MEASUREMENT_UNIT)
            if "basis" in semantic:
                required.append(SemanticEvidenceRoleV3.MEASUREMENT_BASIS)
        if context.context_type.value in _CONTEXT_ROLES and semantic:
            required.append(_CONTEXT_ROLES[context.context_type.value])
        for role in required:
            if (GraphNodeKindV3.CONTEXT.value, context.context_ref, role) not in binding_keys:
                diagnostics.append(V3Diagnostic.ROLE_TARGET_MISMATCH.value)
        if not any(edge.context_ref == context.context_ref for edge in response.source_attachments):
            diagnostics.append(V3Diagnostic.DANGLING_TARGET_REF.value)
        context_edges = [
            edge for edge in response.source_attachments if edge.context_ref == context.context_ref
        ]
        for binding in bindings:
            if (
                binding.target_kind is GraphNodeKindV3.CONTEXT
                and binding.target_ref == context.context_ref
                and not any(binding.evidence_ref in edge.evidence_refs for edge in context_edges)
            ):
                diagnostics.append(V3Diagnostic.DANGLING_EVIDENCE_REF.value)
    binding_by_ref = {binding.binding_ref: binding for binding in bindings}
    for binding in bindings:
        if binding.related_binding_ref and binding.related_binding_ref not in binding_by_ref:
            diagnostics.append(V3Diagnostic.DANGLING_TARGET_REF.value)
        if binding.semantic_role is SemanticEvidenceRoleV3.METHOD_RELATION:
            related = binding_by_ref.get(binding.related_binding_ref or "")
            if not (
                related
                and related.semantic_role is SemanticEvidenceRoleV3.METHOD_VALUE
                and related.target_kind is GraphNodeKindV3.CONTEXT
                and related.target_ref == binding.target_ref
            ):
                diagnostics.append(V3Diagnostic.METHOD_RELATION_UNLINKED.value)
    for basis in response.measurement_bases:
        if basis.measurement_context_ref not in refs[GraphNodeKindV3.CONTEXT] or any(
            x not in valid_evidence for x in basis.evidence_refs
        ):
            diagnostics.append(V3Diagnostic.MEASUREMENT_BASIS_REF_INVALID.value)
        if not any(
            b.semantic_role is SemanticEvidenceRoleV3.MEASUREMENT_BASIS
            and b.target_kind is GraphNodeKindV3.CONTEXT
            and b.target_ref == basis.measurement_context_ref
            and b.evidence_ref in basis.evidence_refs
            for b in bindings
        ):
            diagnostics.append(V3Diagnostic.MEASUREMENT_BASIS_REF_INVALID.value)

    unique = tuple(dict.fromkeys(diagnostics))
    hard_errors = {
        V3Diagnostic.DUPLICATE_REFERENCE.value,
        V3Diagnostic.DANGLING_EVIDENCE_REF.value,
        V3Diagnostic.DANGLING_TARGET_REF.value,
        V3Diagnostic.ROLE_TARGET_MISMATCH.value,
        V3Diagnostic.ASSERTION_EVIDENCE_MISSING.value,
        V3Diagnostic.INVALID_SCOPE_GRAPH.value,
        V3Diagnostic.EPISTEMIC_TARGET_MISMATCH.value,
        V3Diagnostic.MEASUREMENT_BASIS_REF_INVALID.value,
    }
    completeness = {
        V3Diagnostic.ATTACHMENT_BINDING_MISSING.value,
        V3Diagnostic.RELATION_BINDING_MISSING.value,
        V3Diagnostic.METHOD_RELATION_UNLINKED.value,
        V3Diagnostic.SEMANTIC_RELATION_UNRESOLVED.value,
    }
    if any(x in hard_errors for x in unique):
        status = V3ValidationStatus.INVALID
    elif any(x in completeness for x in unique):
        status = V3ValidationStatus.PARTIAL
    elif response.status == "UNRESOLVED":
        status = V3ValidationStatus.UNRESOLVED
    else:
        status = V3ValidationStatus.RESOLVED
    unconsumed = tuple(
        kind
        for kind, present in (
            ("SEMANTIC_RELATION", bool(response.semantic_relations)),
            ("MEASUREMENT_BASIS", bool(response.measurement_bases)),
        )
        if present
    )
    return SourceContextExtractionResultV3(
        claim_ref=claim_ref,
        claim_hash=claim_hash,
        status=status,
        response=response,
        binding_assessments=tuple(assessments),
        diagnostics=unique,
        downstream_not_yet_consumed=unconsumed,
    )


def project_v3_to_v2_1_read_only(
    result: SourceContextExtractionResultV3,
) -> dict[str, Any]:
    """Project unchanged common facts; explicitly retain v3-only facts as unconsumed."""
    response = result.response
    if response is None:
        return {
            "projection_status": "UNAVAILABLE",
            "facts": {},
            "diagnostics": list(result.diagnostics),
        }
    return {
        "projection_status": "READ_ONLY_PARTIAL",
        "facts": {
            "source_assertions": [x.model_dump(mode="json") for x in response.source_assertions],
            "source_contexts": [x.model_dump(mode="json") for x in response.source_contexts],
            "source_attachments": [x.model_dump(mode="json") for x in response.source_attachments],
            "source_scopes": [x.model_dump(mode="json") for x in response.source_scopes],
        },
        "downstream_not_yet_consumed": list(result.downstream_not_yet_consumed),
        "semantic_promotion": False,
    }
