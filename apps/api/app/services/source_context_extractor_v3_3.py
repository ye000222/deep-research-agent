"""Offline v3.3 Measurement admissibility evaluation; no Provider or persistence."""

# Prompt clauses are intentionally kept as complete reviewable instructions.
# ruff: noqa: E501

from __future__ import annotations

from collections import defaultdict

from app.domain.source_context_extraction_v3_1 import (
    SemanticEvidenceBindingV31,
    SemanticEvidenceRoleV31,
)
from app.domain.source_context_extraction_v3_3 import (
    MeasurementAdmissibilityV33,
    MeasurementIdentityTypeV33,
    MeasurementSemanticsV33,
    SourceExtractionResponseV33,
)

SOURCE_CONTEXT_V33_INSTRUCTIONS = "\n".join(
    (
        "Extract only source-supported semantic structure; use no external knowledge or common-sense completion.",
        "For each Measurement identity, provide name, broad identity_type, and a reference to an exact existing source span.",
        "Use only the versioned identity_type values. QUALITATIVE_PROPERTY is not a Measurement; use UNKNOWN when uncertain.",
        "A Measurement requires a source-linked observed_value, formal_definition, or threshold explicitly attached to the identity.",
        "Units, numbers, comparators, basis, target, scope, and qualifiers do not independently establish Measurement identity.",
        "Never invent a value, definition, unit, basis, source span, or identity. Preserve unsupported propositions as assertions and fail closed.",
        "This contract does not use keyword lists, metric catalogs, unit allowlists, domain rules, or Claim-specific rules.",
    )
)


def _occurrence_count(claim_text: str, span_text: str) -> int:
    return sum(
        claim_text.startswith(span_text, offset)
        for offset in range(len(claim_text))
    )


def evaluate_measurement_admissibility_v33(
    response: SourceExtractionResponseV33, claim_text: str
) -> dict[str, MeasurementAdmissibilityV33]:
    """Evaluate shape/provenance only; semantic truth is not inferred here."""
    spans = {item.evidence_ref: item for item in response.evidence_spans}
    bindings_by_context: dict[str, list[SemanticEvidenceBindingV31]] = defaultdict(list)
    for binding in response.evidence_bindings:
        if binding.target_kind.value == "CONTEXT":
            bindings_by_context[binding.target_ref].append(binding)

    results: dict[str, MeasurementAdmissibilityV33] = {}
    for context in response.source_contexts:
        if context.context_type.value != "MEASUREMENT":
            continue
        semantics = context.semantic_value
        if not isinstance(semantics, MeasurementSemanticsV33):
            results[context.context_ref] = MeasurementAdmissibilityV33.UNKNOWN
            continue
        identity = semantics.measurement_identity
        if identity is None or not identity.name:
            results[context.context_ref] = MeasurementAdmissibilityV33.UNKNOWN
            continue
        if identity.identity_type is MeasurementIdentityTypeV33.QUALITATIVE_PROPERTY:
            results[context.context_ref] = MeasurementAdmissibilityV33.NOT_MEASUREMENT
            continue
        if identity.identity_type is MeasurementIdentityTypeV33.UNKNOWN:
            results[context.context_ref] = MeasurementAdmissibilityV33.UNKNOWN
            continue
        if identity.source_span is None:
            results[context.context_ref] = MeasurementAdmissibilityV33.UNKNOWN
            continue
        span = spans.get(identity.source_span.evidence_ref)
        if (
            span is None
            or identity.source_span.occurrence_index
            >= _occurrence_count(claim_text, span.exact_text)
        ):
            results[context.context_ref] = MeasurementAdmissibilityV33.UNKNOWN
            continue

        context_bindings = bindings_by_context.get(context.context_ref, [])
        required_roles: list[SemanticEvidenceRoleV31] = []
        if semantics.observed_value is not None:
            required_roles.append(SemanticEvidenceRoleV31.MEASUREMENT_VALUE)
        if semantics.formal_definition is not None:
            required_roles.append(SemanticEvidenceRoleV31.MEASUREMENT_DEFINITION)
        if semantics.threshold is not None:
            required_roles.append(SemanticEvidenceRoleV31.MEASUREMENT_COMPARATOR)
        linked_basis = any(
            any(
                getattr(binding, "semantic_role", None) is role
                and getattr(binding, "evidence_ref", None) in spans
                and _occurrence_count(claim_text, spans[binding.evidence_ref].exact_text)
                > spans[binding.evidence_ref].occurrence_index
                for binding in context_bindings
            )
            for role in required_roles
        )
        if not linked_basis:
            results[context.context_ref] = MeasurementAdmissibilityV33.UNKNOWN
            continue
        results[context.context_ref] = MeasurementAdmissibilityV33.ADMISSIBLE
    return results
