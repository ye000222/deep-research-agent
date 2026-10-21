"""Claim-local, deterministic Step 2 bridge from source assertions to AP links.

This is a pure projection/evaluation layer. It does not persist, call providers,
or alter existing V2.2 / ResearchLoop behavior.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.atomic_proposition import (
    AtomicProposition,
    ClaimPropositionLink,
    EvidencePropositionRelation,
    PropositionEvidenceLink,
    PropositionRelation,
)
from app.domain.claim_architecture import CanonicalClaim, ClaimEvidenceLink
from app.domain.comparability_evaluator import ComparabilityEvaluator
from app.domain.relation_evaluator import ExplicitSemanticAssociation, RelationEvaluator
from app.domain.semantic_identity_alignment import (
    SemanticIdentityAlignmentFact,
    SemanticIdentityFieldRole,
    project_pair_identities,
)
from app.domain.semantic_runtime_budget import SemanticRuntimeBudget, SemanticRuntimeBudgetState
from app.domain.source_atomic_assertion import (
    SourceAssertionEligibility,
    SourceAssertionField,
    SourceAtomicAssertion,
    SourceGroundingState,
)
from app.domain.source_context_extraction_v3_3 import (
    MeasurementAdmissibilityV33,
    MeasurementSemanticsV33,
    SourceContextV33,
)
from app.domain.truth_condition_context import (
    AttachmentStatus,
    ContextScopeKind,
    TruthConditionContext,
)
from app.domain.verification_assertion import (
    AssertionGroundingUsability,
    ComparabilityStatus,
    ComparisonEligibility,
    ContextAttachmentResolution,
    MaterialContextProjection,
    PropositionPairAssessment,
    SemanticIdentity,
    SemanticIdentityStatus,
    VerificationAssertion,
)

PROPOSITION_EVIDENCE_DIRECTION_SCHEMA_VERSION = "proposition-evidence-direction-v1"
PROPOSITION_EVIDENCE_AGGREGATION_SCHEMA_VERSION = "proposition-evidence-aggregation-v1"
SEMANTIC_DIRECTION_METHOD = "v2_3d_semantic_runtime_step2"


class DirectionStatus(StrEnum):
    RESOLVED_SUPPORT = "RESOLVED_SUPPORT"
    RESOLVED_CONTRADICTION = "RESOLVED_CONTRADICTION"
    RESOLVED_MENTION = "RESOLVED_MENTION"
    UNRESOLVED = "UNRESOLVED"
    INVALID_INPUT = "INVALID_INPUT"
    NOT_PROCESSED_BUDGET = "NOT_PROCESSED_BUDGET"


class DirectionReason(StrEnum):
    SAME_RELATION = "SAME_RELATION"
    CONTRADICTS_RELATION = "CONTRADICTS_RELATION"
    EXPLICIT_MENTION_BASIS = "EXPLICIT_MENTION_BASIS"
    IDENTITY_UNRESOLVED = "IDENTITY_UNRESOLVED"
    GROUNDING_UNUSABLE = "GROUNDING_UNUSABLE"
    COMPOSITION_INELIGIBLE = "COMPOSITION_INELIGIBLE"
    MEASUREMENT_INADMISSIBLE = "MEASUREMENT_INADMISSIBLE"
    COMPARABILITY_UNKNOWN = "COMPARABILITY_UNKNOWN"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    RELATION_UNKNOWN = "RELATION_UNKNOWN"
    RELATED_WITHOUT_MENTION_BASIS = "RELATED_WITHOUT_MENTION_BASIS"
    INTRA_EVIDENCE_DIRECTION_CONFLICT = "INTRA_EVIDENCE_DIRECTION_CONFLICT"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    INVALID_PROVENANCE = "INVALID_PROVENANCE"
    MATERIAL_CONTEXT_UNRESOLVED = "MATERIAL_CONTEXT_UNRESOLVED"


class SourceAssertionEvidenceBinding(BaseModel):
    """Explicit caller attestation connecting one source assertion to one Evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_assertion_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_id: UUID
    source_id: UUID
    source_ref: str = Field(min_length=1, max_length=2_000)
    snapshot_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_evidence_span_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_owner_key: str | None = Field(default=None, min_length=1, max_length=255)


class MeasurementAdmissibilityBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    context_ref: str = Field(min_length=1, max_length=128)
    admissibility: MeasurementAdmissibilityV33


class PropositionDirectionTarget(BaseModel):
    """A target A1 projection and its formal Claim-to-proposition membership."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    membership: ClaimPropositionLink
    proposition: AtomicProposition
    assertion: VerificationAssertion

    @model_validator(mode="after")
    def validate_target_identity(self) -> PropositionDirectionTarget:
        if self.membership.proposition_id != self.proposition.proposition_id:
            raise ValueError("claim_proposition_membership_id_mismatch")
        if self.assertion.source_proposition_id != self.proposition.proposition_id:
            raise ValueError("verification_assertion_proposition_id_mismatch")
        if self.membership.decomposition_status != self.proposition.decomposition.status:
            raise ValueError("claim_proposition_decomposition_status_mismatch")
        return self


class PropositionMentionBasis(BaseModel):
    """Explicit proposition/evidence-relative basis authorizing only MENTIONS."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposition_ref: UUID
    evidence_ref: UUID
    source_assertion_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    basis_ref: str = Field(min_length=1, max_length=512)
    association: ExplicitSemanticAssociation


class EvidenceDirectionScope(BaseModel):
    """One explicitly provenance-scoped claim/evidence input; never globally joined."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    claim: CanonicalClaim
    claim_evidence: ClaimEvidenceLink
    targets: tuple[PropositionDirectionTarget, ...] = ()
    source_assertions: tuple[SourceAtomicAssertion, ...] = ()
    source_bindings: tuple[SourceAssertionEvidenceBinding, ...] = ()
    truth_condition_contexts: tuple[TruthConditionContext, ...] = ()
    source_contexts: tuple[SourceContextV33, ...] = ()
    measurement_admissibility: tuple[MeasurementAdmissibilityBinding, ...] = ()
    explicit_semantic_associations: tuple[ExplicitSemanticAssociation, ...] = ()
    mention_associations: tuple[PropositionMentionBasis, ...] = ()

    @model_validator(mode="after")
    def validate_explicit_scope(self) -> EvidenceDirectionScope:
        if self.claim_evidence.claim_id != self.claim.claim_id:
            raise ValueError("claim_evidence_scope_mismatch")
        if len({item.proposition.proposition_id for item in self.targets}) != len(self.targets):
            raise ValueError("duplicate_claim_proposition_target")
        if any(item.membership.claim_id != self.claim.claim_id for item in self.targets):
            raise ValueError("claim_proposition_target_scope_mismatch")
        assertion_by_ref = {item.assertion_ref: item for item in self.source_assertions}
        binding_by_ref = {item.source_assertion_ref: item for item in self.source_bindings}
        if len(assertion_by_ref) != len(self.source_assertions):
            raise ValueError("duplicate_source_assertion_ref")
        if len(binding_by_ref) != len(self.source_bindings):
            raise ValueError("duplicate_source_assertion_binding")
        if set(assertion_by_ref) != set(binding_by_ref):
            raise ValueError("source_assertion_binding_set_mismatch")
        context_refs = {item.context_ref for item in self.truth_condition_contexts}
        if len(context_refs) != len(self.truth_condition_contexts):
            raise ValueError("duplicate_truth_condition_context_ref")
        context_by_ref = {item.context_ref: item for item in self.truth_condition_contexts}
        if any(item.origin.value != "SOURCE" for item in self.truth_condition_contexts):
            raise ValueError("source_context_origin_invalid")
        if any(
            not set(assertion.truth_condition_context_refs).issubset(context_refs)
            for assertion in self.source_assertions
        ):
            raise ValueError("source_assertion_context_binding_missing")
        source_context_refs = {item.context_ref for item in self.source_contexts}
        if len(source_context_refs) != len(self.source_contexts):
            raise ValueError("duplicate_source_context_ref")
        measurement_refs = tuple(item.context_ref for item in self.measurement_admissibility)
        if len(set(measurement_refs)) != len(measurement_refs):
            raise ValueError("duplicate_measurement_admissibility_ref")
        if any(
            ref not in context_by_ref or context_by_ref[ref].context_type.value != "MEASUREMENT"
            for ref in measurement_refs
        ):
            raise ValueError("measurement_admissibility_context_mismatch")
        source_context_by_ref = {item.context_ref: item for item in self.source_contexts}
        if any(
            ref in source_context_by_ref
            and source_context_by_ref[ref].context_type.value
            != context_by_ref[ref].context_type.value
            for ref in source_context_by_ref.keys() & context_by_ref.keys()
        ):
            raise ValueError("source_context_type_mismatch")
        association_pairs = tuple(
            (item.left_assertion_ref, item.right_assertion_ref)
            for item in self.explicit_semantic_associations
        )
        if len(set(association_pairs)) != len(association_pairs):
            raise ValueError("duplicate_explicit_semantic_association")
        for ref, assertion in assertion_by_ref.items():
            binding = binding_by_ref[ref]
            if (
                binding.evidence_id != self.claim_evidence.evidence_id
                or binding.source_id != self.claim_evidence.source_id
                or binding.source_owner_key != self.claim_evidence.source_owner_key
                or binding.source_ref != assertion.source_ref
                or binding.snapshot_ref != assertion.snapshot_ref
                or binding.canonical_evidence_span_ref != assertion.canonical_evidence_span_ref
            ):
                raise ValueError("source_assertion_evidence_provenance_mismatch")
        self.claim_evidence.validate_scope(
            claim=self.claim,
            run_id=self.claim.run_id,
            question_id=self.claim.question_id,
            dimension_key=self.claim.dimension_key,
            evidence_source_id=self.claim_evidence.source_id,
            evidence_owner_key=self.claim_evidence.source_owner_key,
        )
        return self


class PropositionEvidenceDirectionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    assessment_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_ref: UUID
    proposition_ref: UUID
    source_assertion_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_ref: UUID
    source_ref: str
    snapshot_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    a2_result_ref: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    a2_comparability: ComparabilityStatus | None = None
    a3_result_ref: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    a3_relation: PropositionRelation | None = None
    mention_basis_ref: str | None = Field(default=None, min_length=1, max_length=512)
    grounding_usability: AssertionGroundingUsability
    grounding_basis_refs: tuple[str, ...] = ()
    composition_eligibility: ComparisonEligibility
    measurement_eligibility: MeasurementAdmissibilityV33
    direction_status: DirectionStatus
    reason_codes: tuple[DirectionReason, ...] = ()
    schema_version: str = PROPOSITION_EVIDENCE_DIRECTION_SCHEMA_VERSION

    @model_validator(mode="after")
    def enforce_evaluator_refs(self) -> PropositionEvidenceDirectionAssessment:
        if (self.a2_result_ref is None) != (self.a2_comparability is None):
            raise ValueError("a2_result_ref_and_status_must_cooccur")
        if (self.a3_result_ref is None) != (self.a3_relation is None):
            raise ValueError("a3_result_ref_and_status_must_cooccur")
        if self.direction_status is DirectionStatus.NOT_PROCESSED_BUDGET and (
            self.a2_result_ref is not None or self.a3_result_ref is not None
        ):
            raise ValueError("budget_not_processed_must_not_claim_evaluator_results")
        return self


class PropositionEvidenceDirectionAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    aggregate_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_ref: UUID
    proposition_ref: UUID
    evidence_ref: UUID
    source_assertion_refs: tuple[str, ...] = ()
    direction_status: DirectionStatus
    reason_codes: tuple[DirectionReason, ...] = ()
    schema_version: str = PROPOSITION_EVIDENCE_AGGREGATION_SCHEMA_VERSION


class PropositionEvidenceDirectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    assessments: tuple[PropositionEvidenceDirectionAssessment, ...]
    aggregates: tuple[PropositionEvidenceDirectionAggregate, ...]
    links: tuple[PropositionEvidenceLink, ...]
    budget_state: SemanticRuntimeBudgetState
    deferred_candidate_count: int = Field(default=0, ge=0)


def _stable_ref(*parts: str) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _projection_grounding(
    assertion: SourceAtomicAssertion,
    contexts: dict[str, TruthConditionContext] | None = None,
) -> AssertionGroundingUsability:
    core = {
        SourceAssertionField.SUBJECT,
        SourceAssertionField.PREDICATE,
        SourceAssertionField.VALUE,
        SourceAssertionField.POLARITY,
    }
    required = set(core)
    for context_ref in assertion.truth_condition_context_refs:
        context = (contexts or {}).get(context_ref)
        if context is None:
            return AssertionGroundingUsability.UNKNOWN
        if context.materiality.upper() == "DESCRIPTIVE":
            continue
        if context.context_type.value == "EPISTEMIC":
            required.update({SourceAssertionField.ATTRIBUTION, SourceAssertionField.MODALITY})
        else:
            required.add(SourceAssertionField.CONTEXT)
    if assertion.attribution_context_refs:
        required.add(SourceAssertionField.ATTRIBUTION)
    if assertion.modality_context_refs:
        required.add(SourceAssertionField.MODALITY)
    by_field = {item.field: item for item in assertion.field_grounding}
    if not required.issubset(by_field):
        return AssertionGroundingUsability.UNKNOWN
    states = [
        state
        for field in required
        for state in (
            by_field[field].textual_grounding,
            by_field[field].semantic_support,
            by_field[field].structural_grounding,
        )
    ]
    if SourceGroundingState.INVALID in states:
        return AssertionGroundingUsability.INSUFFICIENT
    if all(state is SourceGroundingState.VALID for state in states):
        return AssertionGroundingUsability.SUFFICIENT
    return AssertionGroundingUsability.UNKNOWN


def _composition_eligibility(
    assertion: SourceAtomicAssertion,
) -> ComparisonEligibility:
    return {
        SourceAssertionEligibility.INDEPENDENTLY_ASSERTED: ComparisonEligibility.ELIGIBLE,
        SourceAssertionEligibility.NOT_INDEPENDENTLY_ASSERTED: ComparisonEligibility.INELIGIBLE,
        SourceAssertionEligibility.UNKNOWN: ComparisonEligibility.UNKNOWN,
    }[assertion.verification_eligibility]


def _grounding_basis_refs(assertion: SourceAtomicAssertion) -> tuple[str, ...]:
    return tuple(sorted({ref for item in assertion.field_grounding for ref in item.evidence_refs}))


def _project_source_assertion(
    source: SourceAtomicAssertion,
    *,
    proposition_id: UUID,
    contexts: dict[str, TruthConditionContext],
    source_contexts: dict[str, SourceContextV33],
    measurement_admissibility: dict[str, MeasurementAdmissibilityV33],
) -> tuple[VerificationAssertion, ComparisonEligibility, MeasurementAdmissibilityV33]:
    context_projections: list[MaterialContextProjection] = []
    measurement_states: list[MeasurementAdmissibilityV33] = []
    for context_ref in source.truth_condition_context_refs:
        context = contexts.get(context_ref)
        if context is None:
            raise ValueError("source_assertion_context_ref_missing")
        if context.context_ref != context_ref:
            raise ValueError("source_assertion_context_ref_mismatch")
        if context.materiality.upper() == "DESCRIPTIVE":
            continue
        scope_resolved = (
            context.scope.kind in {ContextScopeKind.PROPOSITION, ContextScopeKind.PROPOSITION_SET}
            and source.assertion_ref in context.scope.target_refs
        )
        attachments_consistent = not context.attachments or all(
            item.status is AttachmentStatus.RESOLVED and item.target_ref == source.assertion_ref
            for item in context.attachments
        )
        attachment_resolved = bool(context.attachments) and attachments_consistent
        resolved = (scope_resolved or attachment_resolved) and attachments_consistent
        if context.scope.kind is ContextScopeKind.UNKNOWN or context.materiality == "UNKNOWN":
            resolved = False
        measurement_semantics: MeasurementSemanticsV33 | None = None
        if context.context_type.value == "MEASUREMENT":
            typed_context = source_contexts.get(context_ref)
            if (
                typed_context is not None
                and typed_context.context_ref == context_ref
                and isinstance(typed_context.semantic_value, MeasurementSemanticsV33)
            ):
                measurement_semantics = typed_context.semantic_value
        context_projections.append(
            MaterialContextProjection(
                context=context,
                attachment_resolution=(
                    ContextAttachmentResolution.RESOLVED
                    if resolved
                    else ContextAttachmentResolution.AMBIGUOUS
                ),
                attachment_refs=(context_ref,) if resolved else (),
                measurement_semantics=measurement_semantics,
                reason_codes=(),
            )
        )
        if context.context_type.value == "MEASUREMENT":
            measurement_states.append(
                measurement_admissibility.get(context_ref, MeasurementAdmissibilityV33.UNKNOWN)
            )

    if not measurement_states:
        measurement = MeasurementAdmissibilityV33.NOT_APPLICABLE
    elif all(item is MeasurementAdmissibilityV33.ADMISSIBLE for item in measurement_states):
        measurement = MeasurementAdmissibilityV33.ADMISSIBLE
    elif MeasurementAdmissibilityV33.NOT_MEASUREMENT in measurement_states:
        measurement = MeasurementAdmissibilityV33.NOT_MEASUREMENT
    else:
        measurement = MeasurementAdmissibilityV33.UNKNOWN

    composition = _composition_eligibility(source)
    projection = VerificationAssertion(
        assertion_ref=source.assertion_ref,
        source_proposition_id=proposition_id,
        subject=source.subject.source_representation,
        subject_identity=source.subject.semantic_identity,
        predicate=source.predicate.source_representation,
        predicate_identity=source.predicate.semantic_identity,
        typed_value=source.value,
        polarity=source.polarity,
        material_contexts=tuple(context_projections),
        composition_eligibility=composition,
        grounding_usability=_projection_grounding(source, contexts),
        source_context_detection_incomplete=False,
    )
    return projection, composition, measurement


def _assessment_ref(
    claim_ref: UUID,
    proposition_ref: UUID,
    source_ref: str,
    evidence_ref: UUID,
    *versions: str,
) -> str:
    return _stable_ref(
        str(claim_ref), str(proposition_ref), source_ref, str(evidence_ref), *versions
    )


def _evaluate_candidate(
    scope: EvidenceDirectionScope,
    target: PropositionDirectionTarget,
    source: SourceAtomicAssertion,
    binding: SourceAssertionEvidenceBinding,
    alignment_facts: tuple[SemanticIdentityAlignmentFact, ...] = (),
) -> tuple[PropositionEvidenceDirectionAssessment, PropositionPairAssessment | None]:
    proposition_id = target.proposition.proposition_id
    projection: VerificationAssertion | None = None
    comparison: PropositionPairAssessment | None = None
    relation_result = None
    reasons: set[DirectionReason] = set()
    status = DirectionStatus.UNRESOLVED
    context_map = {item.context_ref: item for item in scope.truth_condition_contexts}
    grounding = _projection_grounding(source, context_map)
    composition = _composition_eligibility(source)
    measurement = MeasurementAdmissibilityV33.NOT_APPLICABLE
    target_assertion = target.assertion
    try:
        contexts = {item.context_ref: item for item in scope.truth_condition_contexts}
        source_contexts = {item.context_ref: item for item in scope.source_contexts}
        projection, composition, measurement = _project_source_assertion(
            source,
            proposition_id=proposition_id,
            contexts=contexts,
            source_contexts=source_contexts,
            measurement_admissibility={
                item.context_ref: item.admissibility for item in scope.measurement_admissibility
            },
        )
        roles = [item.field_role for item in alignment_facts]
        if len(roles) != len(set(roles)):
            raise ValueError("duplicate_identity_alignment_field")
        fact_by_role = {item.field_role: item for item in alignment_facts}
        subject_fact = fact_by_role.get(SemanticIdentityFieldRole.SUBJECT)
        predicate_fact = fact_by_role.get(SemanticIdentityFieldRole.PREDICATE)
        # Runtime pair facts are the sole authority when supplied. Pure-domain
        # callers without them retain the frozen A1/A2 projection behavior.
        if alignment_facts:
            target_assertion = target.assertion.model_copy(
                update={
                    "subject_identity": SemanticIdentity.unresolved(),
                    "predicate_identity": SemanticIdentity.unresolved(),
                }
            )
            projection = projection.model_copy(
                update={
                    "subject_identity": SemanticIdentity.unresolved(),
                    "predicate_identity": SemanticIdentity.unresolved(),
                }
            )
        for fact in alignment_facts:
            if (
                fact.target_proposition_ref != str(proposition_id)
                or fact.source_assertion_ref != source.assertion_ref
            ):
                raise ValueError("identity_alignment_pair_mismatch")
        if subject_fact is not None:
            target_subject, source_subject = project_pair_identities(subject_fact)
            target_assertion = target_assertion.model_copy(
                update={"subject_identity": target_subject}
            )
            projection = projection.model_copy(update={"subject_identity": source_subject})
        if predicate_fact is not None:
            target_predicate, source_predicate = project_pair_identities(predicate_fact)
            target_assertion = target_assertion.model_copy(
                update={"predicate_identity": target_predicate}
            )
            projection = projection.model_copy(update={"predicate_identity": source_predicate})
    except (AttributeError, KeyError, TypeError, ValueError):
        reasons.add(DirectionReason.INVALID_PROVENANCE)
        return (
            PropositionEvidenceDirectionAssessment(
                assessment_ref=_assessment_ref(
                    scope.claim.claim_id,
                    proposition_id,
                    source.assertion_ref,
                    binding.evidence_id,
                    "comparability-contract-v1",
                    "relation-evaluator-v1",
                ),
                claim_ref=scope.claim.claim_id,
                proposition_ref=proposition_id,
                source_assertion_ref=source.assertion_ref,
                evidence_ref=binding.evidence_id,
                source_ref=source.source_ref,
                snapshot_ref=source.snapshot_ref,
                grounding_usability=grounding,
                grounding_basis_refs=_grounding_basis_refs(source),
                composition_eligibility=composition,
                measurement_eligibility=measurement,
                direction_status=DirectionStatus.INVALID_INPUT,
                reason_codes=tuple(sorted(reasons, key=lambda item: item.value)),
            ),
            None,
        )

    comparison = ComparabilityEvaluator.evaluate(projection, target_assertion)
    association = next(
        (
            item
            for item in scope.explicit_semantic_associations
            if item.matches(projection.assertion_ref, target_assertion.assertion_ref)
            and item.established
        ),
        None,
    )
    mention_basis_ref = next(
        (
            item.basis_ref
            for item in scope.mention_associations
            if item.proposition_ref == proposition_id
            and item.evidence_ref == binding.evidence_id
            and item.source_assertion_ref == source.assertion_ref
            and item.association.matches(projection.assertion_ref, target_assertion.assertion_ref)
            and item.association.established
        ),
        None,
    )
    relation_result = RelationEvaluator.evaluate(
        projection, target_assertion, comparison, association=association
    )
    a2_ref = _stable_ref(
        "A2",
        projection.assertion_ref,
        target_assertion.assertion_ref,
        comparison.model_dump_json(exclude_none=False),
    )
    a3_ref = _stable_ref(
        "A3",
        projection.assertion_ref,
        target_assertion.assertion_ref,
        relation_result.model_dump_json(exclude_none=False),
    )

    if (
        projection.subject_identity.status is SemanticIdentityStatus.UNRESOLVED
        or projection.predicate_identity.status is SemanticIdentityStatus.UNRESOLVED
        or target_assertion.subject_identity.status is SemanticIdentityStatus.UNRESOLVED
        or target_assertion.predicate_identity.status is SemanticIdentityStatus.UNRESOLVED
    ):
        reasons.add(DirectionReason.IDENTITY_UNRESOLVED)
    if projection.grounding_usability is not AssertionGroundingUsability.SUFFICIENT or (
        target_assertion.grounding_usability is not AssertionGroundingUsability.SUFFICIENT
    ):
        reasons.add(DirectionReason.GROUNDING_UNUSABLE)
    if projection.composition_eligibility is not ComparisonEligibility.ELIGIBLE or (
        target_assertion.composition_eligibility is not ComparisonEligibility.ELIGIBLE
    ):
        reasons.add(DirectionReason.COMPOSITION_INELIGIBLE)
    if (
        any(
            item.attachment_resolution is not ContextAttachmentResolution.RESOLVED
            or item.context.materiality == "UNKNOWN"
            for item in projection.material_contexts
        )
        or target_assertion.material_context_unknown
    ):
        reasons.add(DirectionReason.MATERIAL_CONTEXT_UNRESOLVED)
    if measurement not in {
        MeasurementAdmissibilityV33.ADMISSIBLE,
        MeasurementAdmissibilityV33.NOT_APPLICABLE,
    }:
        reasons.add(DirectionReason.MEASUREMENT_INADMISSIBLE)
    if relation_result.comparability is ComparabilityStatus.NOT_COMPARABLE:
        reasons.add(DirectionReason.NOT_COMPARABLE)
    elif relation_result.comparability is ComparabilityStatus.UNKNOWN:
        reasons.add(DirectionReason.COMPARABILITY_UNKNOWN)

    eligible_for_truth_direction = not reasons
    eligible_for_mention = (
        mention_basis_ref is not None
        and projection.grounding_usability is AssertionGroundingUsability.SUFFICIENT
        and not any(
            item.attachment_resolution is not ContextAttachmentResolution.RESOLVED
            or item.context.materiality == "UNKNOWN"
            for item in projection.material_contexts
        )
        and measurement
        in {
            MeasurementAdmissibilityV33.ADMISSIBLE,
            MeasurementAdmissibilityV33.NOT_APPLICABLE,
        }
    )
    if eligible_for_truth_direction and relation_result.relation is PropositionRelation.SAME:
        status = DirectionStatus.RESOLVED_SUPPORT
        reasons.add(DirectionReason.SAME_RELATION)
    elif (
        eligible_for_truth_direction and relation_result.relation is PropositionRelation.CONTRADICTS
    ):
        status = DirectionStatus.RESOLVED_CONTRADICTION
        reasons.add(DirectionReason.CONTRADICTS_RELATION)
    elif eligible_for_mention and relation_result.relation is PropositionRelation.RELATED:
        status = DirectionStatus.RESOLVED_MENTION
        reasons.add(DirectionReason.EXPLICIT_MENTION_BASIS)
    else:
        if relation_result.relation is PropositionRelation.RELATED:
            reasons.add(DirectionReason.RELATED_WITHOUT_MENTION_BASIS)
        elif relation_result.relation is PropositionRelation.UNKNOWN:
            reasons.add(DirectionReason.RELATION_UNKNOWN)

    assessment = PropositionEvidenceDirectionAssessment(
        assessment_ref=_assessment_ref(
            scope.claim.claim_id,
            proposition_id,
            source.assertion_ref,
            binding.evidence_id,
            comparison.schema_version,
            relation_result.schema_version,
        ),
        claim_ref=scope.claim.claim_id,
        proposition_ref=proposition_id,
        source_assertion_ref=source.assertion_ref,
        evidence_ref=binding.evidence_id,
        source_ref=source.source_ref,
        snapshot_ref=source.snapshot_ref,
        a2_result_ref=a2_ref,
        a2_comparability=relation_result.comparability,
        a3_result_ref=a3_ref,
        a3_relation=relation_result.relation,
        mention_basis_ref=(
            mention_basis_ref if status is DirectionStatus.RESOLVED_MENTION else None
        ),
        grounding_usability=projection.grounding_usability,
        grounding_basis_refs=_grounding_basis_refs(source),
        composition_eligibility=projection.composition_eligibility,
        measurement_eligibility=measurement,
        direction_status=status,
        reason_codes=tuple(sorted(reasons, key=lambda item: item.value)),
    )
    return assessment, relation_result


def _aggregate_key(
    item: PropositionEvidenceDirectionAssessment,
) -> tuple[str, str, str]:
    return str(item.claim_ref), str(item.proposition_ref), str(item.evidence_ref)


def _link(
    scope: EvidenceDirectionScope,
    proposition_ref: UUID,
    relation: EvidencePropositionRelation,
) -> PropositionEvidenceLink:
    return PropositionEvidenceLink(
        proposition_id=proposition_ref,
        evidence_id=scope.claim_evidence.evidence_id,
        claim_id=scope.claim.claim_id,
        relation=relation,
        source_id=scope.claim_evidence.source_id,
        source_owner_key=scope.claim_evidence.source_owner_key,
        quote_reference=scope.claim_evidence.quote_reference,
        method=SEMANTIC_DIRECTION_METHOD,
        run_id=scope.claim.run_id,
        question_id=scope.claim.question_id,
        dimension_key=scope.claim.dimension_key,
    )


def evaluate_proposition_evidence_directions(
    scopes: Iterable[EvidenceDirectionScope],
    *,
    budget: SemanticRuntimeBudget,
    initial_state: SemanticRuntimeBudgetState | None = None,
    identity_alignments: Iterable[SemanticIdentityAlignmentFact] = (),
) -> PropositionEvidenceDirectionResult:
    """Generate only explicit claim/evidence-local pairs and evaluate in stable order."""

    ordered_scopes = tuple(
        sorted(scopes, key=lambda x: (str(x.claim.claim_id), str(x.claim_evidence.evidence_id)))
    )
    if len({(x.claim.claim_id, x.claim_evidence.evidence_id) for x in ordered_scopes}) != len(
        ordered_scopes
    ):
        raise ValueError("duplicate_claim_evidence_scope")
    state = initial_state or SemanticRuntimeBudgetState()
    alignment_by_pair: dict[tuple[str, str], list[SemanticIdentityAlignmentFact]] = defaultdict(
        list
    )
    for fact in identity_alignments:
        alignment_by_pair[(fact.target_proposition_ref, fact.source_assertion_ref)].append(fact)
    planned: list[
        tuple[
            EvidenceDirectionScope,
            PropositionDirectionTarget,
            SourceAtomicAssertion,
            SourceAssertionEvidenceBinding,
            bool,
        ]
    ] = []
    for scope in ordered_scopes:
        targets = tuple(sorted(scope.targets, key=lambda x: str(x.proposition.proposition_id)))
        assertions = tuple(sorted(scope.source_assertions, key=lambda x: x.assertion_ref))
        state, scope_within_budget = state.consume_evidence_scope(
            budget,
            propositions=len(targets),
            source_assertions=len(assertions),
        )
        for target in targets:
            for source in assertions:
                binding = next(
                    item
                    for item in scope.source_bindings
                    if item.source_assertion_ref == source.assertion_ref
                )
                planned.append((scope, target, source, binding, not scope_within_budget))

    assessments: list[PropositionEvidenceDirectionAssessment] = []
    scope_by_key: dict[tuple[str, str, str], EvidenceDirectionScope] = {}
    evaluated = 0
    deferred = 0
    for scope, target, source, binding, input_blocked in planned:
        key = (
            str(scope.claim.claim_id),
            str(target.proposition.proposition_id),
            str(scope.claim_evidence.evidence_id),
        )
        scope_by_key[key] = scope
        if (
            input_blocked
            or state.candidate_pairs >= budget.max_candidate_pairs
            or state.direction_assessments >= budget.max_direction_assessments
        ):
            deferred += 1
            assessments.append(
                PropositionEvidenceDirectionAssessment(
                    assessment_ref=_assessment_ref(
                        scope.claim.claim_id,
                        target.proposition.proposition_id,
                        source.assertion_ref,
                        scope.claim_evidence.evidence_id,
                        "budget-not-processed-v1",
                    ),
                    claim_ref=scope.claim.claim_id,
                    proposition_ref=target.proposition.proposition_id,
                    source_assertion_ref=source.assertion_ref,
                    evidence_ref=scope.claim_evidence.evidence_id,
                    source_ref=source.source_ref,
                    snapshot_ref=source.snapshot_ref,
                    grounding_usability=_projection_grounding(
                        source,
                        {item.context_ref: item for item in scope.truth_condition_contexts},
                    ),
                    composition_eligibility=_composition_eligibility(source),
                    measurement_eligibility=MeasurementAdmissibilityV33.UNKNOWN,
                    direction_status=DirectionStatus.NOT_PROCESSED_BUDGET,
                    reason_codes=(DirectionReason.BUDGET_EXHAUSTED,),
                )
            )
            continue
        state, reserved = state.consume_direction_pair(budget)
        if not reserved:
            deferred += 1
            assessments.append(
                PropositionEvidenceDirectionAssessment(
                    assessment_ref=_assessment_ref(
                        scope.claim.claim_id,
                        target.proposition.proposition_id,
                        source.assertion_ref,
                        scope.claim_evidence.evidence_id,
                        "budget-not-processed-v1",
                    ),
                    claim_ref=scope.claim.claim_id,
                    proposition_ref=target.proposition.proposition_id,
                    source_assertion_ref=source.assertion_ref,
                    evidence_ref=scope.claim_evidence.evidence_id,
                    source_ref=source.source_ref,
                    snapshot_ref=source.snapshot_ref,
                    grounding_usability=_projection_grounding(
                        source,
                        {item.context_ref: item for item in scope.truth_condition_contexts},
                    ),
                    grounding_basis_refs=_grounding_basis_refs(source),
                    composition_eligibility=_composition_eligibility(source),
                    measurement_eligibility=MeasurementAdmissibilityV33.UNKNOWN,
                    direction_status=DirectionStatus.NOT_PROCESSED_BUDGET,
                    reason_codes=(DirectionReason.BUDGET_EXHAUSTED,),
                )
            )
            continue
        evaluated += 1
        pair_facts = tuple(
            sorted(
                alignment_by_pair.get(
                    (str(target.proposition.proposition_id), source.assertion_ref), ()
                ),
                key=lambda item: item.field_role.value,
            )
        )
        assessment, _ = _evaluate_candidate(scope, target, source, binding, pair_facts)
        assessments.append(assessment)

    by_pair: dict[tuple[str, str, str], list[PropositionEvidenceDirectionAssessment]] = defaultdict(
        list
    )
    for item in assessments:
        by_pair[_aggregate_key(item)].append(item)
    aggregates: list[PropositionEvidenceDirectionAggregate] = []
    links: list[PropositionEvidenceLink] = []
    for key, members in sorted(by_pair.items()):
        statuses = {item.direction_status for item in members}
        reasons: set[DirectionReason] = set()
        refs = tuple(sorted({item.source_assertion_ref for item in members}))
        assertion_map = {item.assertion_ref: item for item in scope_by_key[key].source_assertions}
        if (
            DirectionStatus.RESOLVED_SUPPORT in statuses
            and DirectionStatus.RESOLVED_CONTRADICTION in statuses
        ):
            aggregate_status = DirectionStatus.UNRESOLVED
            reasons.add(DirectionReason.INTRA_EVIDENCE_DIRECTION_CONFLICT)
        elif DirectionStatus.NOT_PROCESSED_BUDGET in statuses and statuses.intersection(
            {
                DirectionStatus.RESOLVED_SUPPORT,
                DirectionStatus.RESOLVED_CONTRADICTION,
                DirectionStatus.RESOLVED_MENTION,
            }
        ):
            aggregate_status = DirectionStatus.NOT_PROCESSED_BUDGET
            reasons.add(DirectionReason.BUDGET_EXHAUSTED)
        elif DirectionStatus.RESOLVED_SUPPORT in statuses:
            risky_unknown = any(
                item.direction_status in {DirectionStatus.UNRESOLVED, DirectionStatus.INVALID_INPUT}
                and any(
                    item.source_assertion_ref == resolved.source_assertion_ref
                    or (
                        assertion_map[item.source_assertion_ref].composition_ref is not None
                        and assertion_map[item.source_assertion_ref].composition_ref
                        == assertion_map[resolved.source_assertion_ref].composition_ref
                    )
                    for resolved in members
                    if resolved.direction_status is DirectionStatus.RESOLVED_SUPPORT
                )
                for item in members
            )
            aggregate_status = (
                DirectionStatus.UNRESOLVED if risky_unknown else DirectionStatus.RESOLVED_SUPPORT
            )
        elif DirectionStatus.RESOLVED_CONTRADICTION in statuses:
            aggregate_status = DirectionStatus.RESOLVED_CONTRADICTION
        elif DirectionStatus.RESOLVED_MENTION in statuses:
            aggregate_status = DirectionStatus.RESOLVED_MENTION
        elif DirectionStatus.NOT_PROCESSED_BUDGET in statuses:
            aggregate_status = DirectionStatus.NOT_PROCESSED_BUDGET
        else:
            aggregate_status = DirectionStatus.UNRESOLVED
        for member in members:
            reasons.update(member.reason_codes)
        aggregate = PropositionEvidenceDirectionAggregate(
            aggregate_ref=_stable_ref("aggregate", *key, *refs),
            claim_ref=UUID(key[0]),
            proposition_ref=UUID(key[1]),
            evidence_ref=UUID(key[2]),
            source_assertion_refs=refs,
            direction_status=aggregate_status,
            reason_codes=tuple(sorted(reasons, key=lambda item: item.value)),
        )
        aggregates.append(aggregate)
        scope = scope_by_key[key]
        if aggregate_status is DirectionStatus.RESOLVED_SUPPORT:
            links.append(
                _link(scope, aggregate.proposition_ref, EvidencePropositionRelation.SUPPORTS)
            )
        elif aggregate_status is DirectionStatus.RESOLVED_CONTRADICTION:
            links.append(
                _link(scope, aggregate.proposition_ref, EvidencePropositionRelation.CONTRADICTS)
            )
        elif aggregate_status is DirectionStatus.RESOLVED_MENTION:
            links.append(
                _link(scope, aggregate.proposition_ref, EvidencePropositionRelation.MENTIONS)
            )

    return PropositionEvidenceDirectionResult(
        assessments=tuple(sorted(assessments, key=lambda x: x.assessment_ref)),
        aggregates=tuple(aggregates),
        links=tuple(
            sorted(
                links,
                key=lambda x: (
                    str(x.claim_id),
                    str(x.proposition_id),
                    str(x.evidence_id),
                    x.relation.value,
                ),
            )
        ),
        budget_state=state,
        deferred_candidate_count=deferred,
    )
