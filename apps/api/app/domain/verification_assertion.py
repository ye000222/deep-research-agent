"""V2.2-A1 read-only assertion projection and comparison contracts.

This module defines data contracts only. It does not evaluate proposition
comparability or relation, persist semantic state, or invoke external services.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.assertion_composition import (
    AssertionComposition,
    AssertionCompositionValidationError,
    VerificationEligibility,
    proposition_verification_eligibility,
    validate_assertion_composition,
)
from app.domain.atomic_proposition import (
    AtomicProposition,
    MaterialQualifier,
    PropositionPolarity,
    PropositionRelation,
    PropositionValue,
)
from app.domain.source_context_extraction_v3_3 import (
    MeasurementSemanticsV33,
    SourceContextV33,
)
from app.domain.source_context_grounding_v2 import (
    DimensionGrounding,
    MaterialNodeUsability,
)
from app.domain.source_context_grounding_v2 import (
    NodeGroundingAssessment as SourceNodeGroundingAssessment,
)
from app.domain.truth_condition_context import (
    NUMERIC_NORMALIZATION_VERSION,
    AttachmentStatus,
    ContextAttachment,
    ContextGraph,
    ContextScopeKind,
    ContextType,
    TruthConditionContext,
)

VERIFICATION_ASSERTION_SCHEMA_VERSION: Literal["verification-assertion-v1"] = (
    "verification-assertion-v1"
)
PROPOSITION_COMPARISON_SCHEMA_VERSION: Literal["proposition-comparison-contract-v1"] = (
    "proposition-comparison-contract-v1"
)
SEMANTIC_IDENTITY_SCHEMA_VERSION: Literal["semantic-identity-v1"] = "semantic-identity-v1"


class ComparisonEligibility(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    UNKNOWN = "UNKNOWN"


class SemanticIdentityStatus(StrEnum):
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"


class SemanticIdentityBasis(StrEnum):
    """Auditable origin labels; this module does not create or verify identities."""

    EXPLICIT_CANONICAL_REFERENCE = "EXPLICIT_CANONICAL_REFERENCE"
    UPSTREAM_RESOLVER = "UPSTREAM_RESOLVER"
    HUMAN_CURATED = "HUMAN_CURATED"
    UNSPECIFIED = "UNSPECIFIED"


class SemanticIdentity(BaseModel):
    """Opaque, versioned identity metadata supplied by an upstream authority.

    Raw surface strings are intentionally not normalized or converted into keys
    here. A resolved identity must name its namespace, normalization contract,
    and basis so a later evaluator can fail closed across incompatible systems.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: SemanticIdentityStatus = SemanticIdentityStatus.UNRESOLVED
    key: str | None = Field(default=None, min_length=1, max_length=512)
    namespace: str | None = Field(default=None, min_length=1, max_length=200)
    schema_version: Literal["semantic-identity-v1"] = SEMANTIC_IDENTITY_SCHEMA_VERSION
    normalization_version: str | None = Field(default=None, min_length=1, max_length=100)
    basis: SemanticIdentityBasis = SemanticIdentityBasis.UNSPECIFIED
    provenance_ref: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def enforce_resolution_fields(self) -> SemanticIdentity:
        if self.status is SemanticIdentityStatus.UNRESOLVED:
            if self.key is not None:
                raise ValueError("unresolved_semantic_identity_must_not_have_key")
            return self
        if self.key is None:
            raise ValueError("resolved_semantic_identity_requires_key")
        if self.namespace is None:
            raise ValueError("resolved_semantic_identity_requires_namespace")
        if self.normalization_version is None:
            raise ValueError("resolved_semantic_identity_requires_normalization_version")
        if self.basis is SemanticIdentityBasis.UNSPECIFIED:
            raise ValueError("resolved_semantic_identity_requires_auditable_basis")
        return self

    @classmethod
    def unresolved(cls) -> SemanticIdentity:
        return cls(status=SemanticIdentityStatus.UNRESOLVED)


class ComparabilityStatus(StrEnum):
    COMPARABLE = "COMPARABLE"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    UNKNOWN = "UNKNOWN"


class DimensionCompatibility(StrEnum):
    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ComparisonDimension(StrEnum):
    SUBJECT = "SUBJECT"
    PREDICATE = "PREDICATE"
    VALUE = "VALUE"
    POLARITY = "POLARITY"
    TIME = "TIME"
    VERSION = "VERSION"
    SCOPE = "SCOPE"
    POPULATION = "POPULATION"
    LOCATION = "LOCATION"
    CONDITION = "CONDITION"
    SCENARIO = "SCENARIO"
    CONFIGURATION = "CONFIGURATION"
    MEASUREMENT = "MEASUREMENT"
    UNIT = "UNIT"
    CURRENCY = "CURRENCY"
    BASELINE = "BASELINE"
    COMPARISON_TARGET = "COMPARISON_TARGET"
    METHOD = "METHOD"
    ATTRIBUTION = "ATTRIBUTION"
    MODALITY = "MODALITY"
    COMPOSITION = "COMPOSITION"
    GROUNDING = "GROUNDING"


class ComparisonReasonCode(StrEnum):
    SUBJECT_UNKNOWN = "SUBJECT_UNKNOWN"
    SUBJECT_MISMATCH = "SUBJECT_MISMATCH"
    PREDICATE_UNKNOWN = "PREDICATE_UNKNOWN"
    PREDICATE_MISMATCH = "PREDICATE_MISMATCH"
    VALUE_TYPE_MISMATCH = "VALUE_TYPE_MISMATCH"
    VALUE_UNKNOWN = "VALUE_UNKNOWN"
    POLARITY_MISMATCH = "POLARITY_MISMATCH"
    TIME_MISMATCH = "TIME_MISMATCH"
    TIME_UNKNOWN = "TIME_UNKNOWN"
    VERSION_MISMATCH = "VERSION_MISMATCH"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    POPULATION_MISMATCH = "POPULATION_MISMATCH"
    LOCATION_MISMATCH = "LOCATION_MISMATCH"
    CONDITION_MISMATCH = "CONDITION_MISMATCH"
    SCENARIO_MISMATCH = "SCENARIO_MISMATCH"
    CONFIGURATION_MISMATCH = "CONFIGURATION_MISMATCH"
    MEASUREMENT_MISMATCH = "MEASUREMENT_MISMATCH"
    MEASUREMENT_UNKNOWN = "MEASUREMENT_UNKNOWN"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"
    BASELINE_MISMATCH = "BASELINE_MISMATCH"
    COMPARISON_TARGET_MISMATCH = "COMPARISON_TARGET_MISMATCH"
    METHOD_MISMATCH = "METHOD_MISMATCH"
    ATTRIBUTION_MISMATCH = "ATTRIBUTION_MISMATCH"
    MODALITY_MISMATCH = "MODALITY_MISMATCH"
    CONTEXT_ATTACHMENT_AMBIGUOUS = "CONTEXT_ATTACHMENT_AMBIGUOUS"
    COMPOSITION_INELIGIBLE = "COMPOSITION_INELIGIBLE"
    COMPOSITION_UNKNOWN = "COMPOSITION_UNKNOWN"
    INSUFFICIENT_GROUNDING = "INSUFFICIENT_GROUNDING"
    UNSUPPORTED_VALUE_TYPE = "UNSUPPORTED_VALUE_TYPE"
    EXPLICIT_SEMANTIC_ASSOCIATION = "EXPLICIT_SEMANTIC_ASSOCIATION"
    RELATION_NOT_EVALUATED = "RELATION_NOT_EVALUATED"
    SAME_TYPED_VALUE = "SAME_TYPED_VALUE"
    DIRECT_POLARITY_CONFLICT = "DIRECT_POLARITY_CONFLICT"
    DIRECT_BOOLEAN_VALUE_CONFLICT = "DIRECT_BOOLEAN_VALUE_CONFLICT"
    RELATION_SEMANTICS_INSUFFICIENT = "RELATION_SEMANTICS_INSUFFICIENT"


class ContextAttachmentResolution(StrEnum):
    RESOLVED = "RESOLVED"
    AMBIGUOUS = "AMBIGUOUS"


class AssertionGroundingUsability(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    UNKNOWN = "UNKNOWN"


class MaterialContextProjection(BaseModel):
    """Existing context plus its attachment certainty; never repairs scope."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    context: TruthConditionContext
    attachment_resolution: ContextAttachmentResolution
    attachment_refs: tuple[str, ...] = ()
    measurement_semantics: MeasurementSemanticsV33 | None = None
    reason_codes: tuple[ComparisonReasonCode, ...] = ()


class VerificationAssertion(BaseModel):
    """Versioned read-only view derived from an existing Atomic Proposition."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    assertion_ref: str = Field(min_length=1, max_length=128)
    source_proposition_id: UUID
    subject: str | None
    subject_identity: SemanticIdentity = Field(default_factory=SemanticIdentity.unresolved)
    predicate: str | None
    predicate_identity: SemanticIdentity = Field(default_factory=SemanticIdentity.unresolved)
    typed_value: PropositionValue
    polarity: PropositionPolarity
    material_qualifiers: tuple[MaterialQualifier, ...] = ()
    modality: str | None = None
    attribution: str | None = None
    material_contexts: tuple[MaterialContextProjection, ...] = ()
    composition_eligibility: ComparisonEligibility = ComparisonEligibility.UNKNOWN
    grounding_usability: AssertionGroundingUsability = AssertionGroundingUsability.UNKNOWN
    grounding_dimensions: DimensionGrounding | None = None
    source_grounding_assessment: SourceNodeGroundingAssessment | None = None
    source_context_detection_incomplete: bool = False
    schema_version: Literal["verification-assertion-v1"] = VERIFICATION_ASSERTION_SCHEMA_VERSION
    normalization_version: str = NUMERIC_NORMALIZATION_VERSION

    @property
    def material_context_unknown(self) -> bool:
        return self.source_context_detection_incomplete or any(
            item.attachment_resolution is ContextAttachmentResolution.AMBIGUOUS
            or item.context.materiality == "UNKNOWN"
            for item in self.material_contexts
        )


class DimensionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: ComparisonDimension
    compatibility: DimensionCompatibility
    reason_codes: tuple[ComparisonReasonCode, ...] = ()


class PropositionPairAssessment(BaseModel):
    """Contract result envelope; A1 leaves evaluation to later phases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    left_assertion_ref: str = Field(min_length=1, max_length=128)
    right_assertion_ref: str = Field(min_length=1, max_length=128)
    comparability: ComparabilityStatus = ComparabilityStatus.UNKNOWN
    relation: PropositionRelation = PropositionRelation.UNKNOWN
    relation_eligible: bool | None = Field(
        default=None,
        description=(
            "Whether the available pair information permits relation classification; "
            "independent of comparability. None means not evaluated."
        ),
    )
    dimension_assessments: tuple[DimensionAssessment, ...] = ()
    reason_codes: tuple[ComparisonReasonCode, ...] = (ComparisonReasonCode.RELATION_NOT_EVALUATED,)
    schema_version: Literal["proposition-comparison-contract-v1"] = (
        PROPOSITION_COMPARISON_SCHEMA_VERSION
    )
    normalization_version: str = NUMERIC_NORMALIZATION_VERSION

    @model_validator(mode="after")
    def enforce_relation_contract(self) -> PropositionPairAssessment:
        if self.relation is PropositionRelation.SAME and (
            self.comparability is not ComparabilityStatus.COMPARABLE
        ):
            raise ValueError("same_requires_comparable_assertions")
        if self.relation is PropositionRelation.CONTRADICTS and (
            self.comparability is not ComparabilityStatus.COMPARABLE
        ):
            raise ValueError("contradicts_requires_comparable_assertions")
        if self.relation is PropositionRelation.RELATED and (
            ComparisonReasonCode.EXPLICIT_SEMANTIC_ASSOCIATION not in self.reason_codes
        ):
            raise ValueError("related_requires_explicit_semantic_association_reason")
        return self


_MATERIALITY_INCLUDED = {"MATERIAL", "UNKNOWN"}


def _context_targets_proposition(
    context: TruthConditionContext,
    proposition_ref: str,
    graph: ContextGraph,
) -> tuple[bool, bool, tuple[str, ...]]:
    """Return (targeted, ambiguous, matching attachment refs)."""
    matching: list[ContextAttachment] = []
    for attachment in context.attachments:
        target = attachment.target_ref
        is_target = target == proposition_ref or (
            target is not None and proposition_ref in graph.scope_expansions.get(target, ())
        )
        unknown_global = attachment.status is AttachmentStatus.UNKNOWN and target is None
        if is_target or unknown_global:
            matching.append(attachment)

    scope_targets = proposition_ref in context.scope.target_refs
    scope_known = context.scope.kind in {
        ContextScopeKind.PROPOSITION,
        ContextScopeKind.PROPOSITION_SET,
    }
    if not matching and scope_targets:
        return True, not scope_known, (context.context_ref,)
    if not matching:
        return False, False, ()

    all_resolved = all(
        item.status is AttachmentStatus.RESOLVED
        and item.relation.value != "UNKNOWN"
        and item.target_ref is not None
        for item in matching
    )
    scope_ambiguous = context.scope.kind is ContextScopeKind.UNKNOWN
    return (
        True,
        not all_resolved or scope_ambiguous,
        tuple(sorted({context.context_ref for _ in matching})),
    )


def _grounding_projection(
    assessment: SourceNodeGroundingAssessment | None,
) -> tuple[AssertionGroundingUsability, DimensionGrounding | None]:
    if assessment is None:
        return AssertionGroundingUsability.UNKNOWN, None
    usability = assessment.usability
    if usability is MaterialNodeUsability.GROUNDED_USABLE:
        projected = AssertionGroundingUsability.SUFFICIENT
    elif usability in {
        MaterialNodeUsability.UNGROUNDED_UNUSABLE,
        MaterialNodeUsability.PARTIALLY_GROUNDED_UNRESOLVED,
    }:
        projected = AssertionGroundingUsability.INSUFFICIENT
    else:
        projected = AssertionGroundingUsability.UNKNOWN
    return projected, assessment.dimensions


def project_verification_assertion(
    proposition: AtomicProposition,
    *,
    proposition_ref: str,
    output_context_graph: ContextGraph,
    source_context_graph: ContextGraph | None = None,
    composition: AssertionComposition | None = None,
    claim_id: UUID | None = None,
    claim_text: str | None = None,
    proposition_ids: set[UUID] | frozenset[UUID] | None = None,
    source_grounding_assessment: SourceNodeGroundingAssessment | None = None,
    measurement_contexts: Mapping[str, SourceContextV33] | None = None,
    subject_identity: SemanticIdentity | None = None,
    predicate_identity: SemanticIdentity | None = None,
) -> VerificationAssertion:
    """Project existing typed inputs without changing or persisting source state.

    Composition eligibility is only trusted after whole-graph validation. Missing
    or invalid composition metadata fails closed to UNKNOWN.
    """
    eligibility = ComparisonEligibility.UNKNOWN
    if (
        composition is not None
        and claim_id is not None
        and composition.claim_id == claim_id
        and claim_text is not None
        and proposition_ids is not None
    ):
        try:
            validate_assertion_composition(
                composition,
                proposition_ids=proposition_ids,
                claim_text=claim_text,
            )
            declared = proposition_verification_eligibility(composition, proposition.proposition_id)
            eligibility = {
                VerificationEligibility.INDEPENDENTLY_ASSERTED: ComparisonEligibility.ELIGIBLE,
                VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED: (
                    ComparisonEligibility.INELIGIBLE
                ),
                VerificationEligibility.UNKNOWN: ComparisonEligibility.UNKNOWN,
            }[declared]
        except (AssertionCompositionValidationError, KeyError, ValueError):
            eligibility = ComparisonEligibility.UNKNOWN

    contexts: list[MaterialContextProjection] = []
    graphs = tuple(graph for graph in (source_context_graph, output_context_graph) if graph)
    for graph in graphs:
        for context in graph.contexts:
            if context.materiality.upper() not in _MATERIALITY_INCLUDED:
                continue
            targeted, ambiguous, attachment_refs = _context_targets_proposition(
                context, proposition_ref, graph
            )
            if not targeted:
                continue
            measurement = None
            missing_measurement_contract = False
            if context.context_type is ContextType.MEASUREMENT:
                candidate = (measurement_contexts or {}).get(context.context_ref)
                if candidate is not None and candidate.context_ref == context.context_ref:
                    semantic_value = candidate.semantic_value
                    if isinstance(semantic_value, MeasurementSemanticsV33):
                        measurement = semantic_value
                missing_measurement_contract = measurement is None
            reasons: tuple[ComparisonReasonCode, ...] = tuple(
                reason
                for condition, reason in (
                    (ambiguous, ComparisonReasonCode.CONTEXT_ATTACHMENT_AMBIGUOUS),
                    (missing_measurement_contract, ComparisonReasonCode.MEASUREMENT_UNKNOWN),
                )
                if condition
            )
            contexts.append(
                MaterialContextProjection(
                    context=context,
                    attachment_resolution=(
                        ContextAttachmentResolution.AMBIGUOUS
                        if ambiguous
                        else ContextAttachmentResolution.RESOLVED
                    ),
                    attachment_refs=attachment_refs,
                    measurement_semantics=measurement,
                    reason_codes=reasons,
                )
            )

    # A detector-limited source graph cannot prove absence. Preserve that fact as
    # projection metadata rather than synthesizing a context node.
    source_context_detection_incomplete = (
        source_context_graph is not None
        and not source_context_graph.source_detection_complete
        and source_context_graph.context_presence.value == "DETECTOR_CANNOT_DECIDE"
    )

    grounding, dimensions = _grounding_projection(source_grounding_assessment)
    return VerificationAssertion(
        assertion_ref=proposition_ref,
        source_proposition_id=proposition.proposition_id,
        subject=proposition.subject,
        subject_identity=subject_identity or SemanticIdentity.unresolved(),
        predicate=proposition.predicate,
        predicate_identity=predicate_identity or SemanticIdentity.unresolved(),
        typed_value=proposition.value,
        polarity=proposition.polarity,
        material_qualifiers=proposition.qualifiers,
        modality=proposition.modality,
        attribution=proposition.attribution,
        material_contexts=tuple(contexts),
        composition_eligibility=eligibility,
        grounding_usability=grounding,
        grounding_dimensions=dimensions,
        source_grounding_assessment=source_grounding_assessment,
        source_context_detection_incomplete=source_context_detection_incomplete,
    )
