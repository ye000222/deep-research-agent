from __future__ import annotations

from decimal import Decimal
from uuid import UUID

import pytest
from app.domain.assertion_composition import (
    AssertionComposition,
    AssertionGroup,
    AssertionGroupMember,
    AssertionGroupOperator,
    VerificationEligibility,
)
from app.domain.atomic_proposition import (
    AtomicProposition,
    DecompositionMetadata,
    DecompositionStatus,
    NumericOperator,
    NumericValue,
    PropositionPolarity,
    PropositionRelation,
    PropositionValue,
    PropositionValueKind,
)
from app.domain.source_context_grounding_v2 import (
    DimensionGrounding,
    EvidenceTargetKind,
    GroundingDimensionStatus,
    MaterialNodeUsability,
    NodeGroundingAssessment,
)
from app.domain.truth_condition_context import (
    AttachmentRelation,
    AttachmentStatus,
    ContextAttachment,
    ContextGraph,
    ContextPresenceConclusion,
    ContextScope,
    ContextScopeKind,
    ContextType,
    TruthConditionContext,
)
from app.domain.verification_assertion import (
    AssertionGroundingUsability,
    ComparabilityStatus,
    ComparisonDimension,
    ComparisonEligibility,
    ComparisonReasonCode,
    ContextAttachmentResolution,
    DimensionAssessment,
    DimensionCompatibility,
    PropositionPairAssessment,
    SemanticIdentity,
    SemanticIdentityBasis,
    SemanticIdentityStatus,
    project_verification_assertion,
)
from pydantic import ValidationError

CLAIM_ID = UUID("10000000-0000-0000-0000-000000000001")
PROP_ID = UUID("20000000-0000-0000-0000-000000000001")
PROP2_ID = UUID("20000000-0000-0000-0000-000000000002")


def _proposition(*, numeric: bool = False) -> AtomicProposition:
    value = (
        PropositionValue(
            kind=PropositionValueKind.QUANTITY,
            numeric=NumericValue(operator=NumericOperator.EXACT, value=Decimal("10.00"), unit="kg"),
        )
        if numeric
        else PropositionValue(kind=PropositionValueKind.TEXT, text="stable")
    )
    return AtomicProposition(
        proposition_id=PROP_ID,
        original_text="Atlas is stable",
        subject="Atlas",
        predicate="is",
        value=value,
        polarity=PropositionPolarity.ASSERTED,
        decomposition=DecompositionMetadata(status=DecompositionStatus.SINGLE_PROPOSITION),
    )


def _context(
    context_type: ContextType,
    *,
    ref: str = "ctx1",
    materiality: str = "MATERIAL",
    status: AttachmentStatus = AttachmentStatus.RESOLVED,
    target_ref: str | None = "p0",
    scope_kind: ContextScopeKind = ContextScopeKind.PROPOSITION,
    value: dict[str, object] | None = None,
) -> TruthConditionContext:
    edge = ContextAttachment(
        context_ref=ref,
        target_ref=target_ref,
        target_kind="proposition",
        relation=AttachmentRelation.APPLIES_TO,
        status=status,
        reason_code="test_attachment",
    )
    return TruthConditionContext(
        context_ref=ref,
        semantic_signature=f"{context_type.value.lower()}-signature",
        context_type=context_type,
        semantic_value=value or {"value": "2025"},
        lexical_value="2025",
        source_span="2025",
        materiality=materiality,
        scope=ContextScope(kind=scope_kind, target_refs=("p0",)),
        attachments=(edge,),
        origin="OUTPUT",
    )


def _graph(*contexts: TruthConditionContext) -> ContextGraph:
    return ContextGraph(
        origin="OUTPUT",
        claim_ref=str(CLAIM_ID),
        contexts=contexts,
        context_presence=ContextPresenceConclusion.CONTEXT_DETECTED,
        source_detection_complete=True,
    )


def _project(
    proposition: AtomicProposition | None = None,
    graph: ContextGraph | None = None,
    **kwargs: object,
):
    return project_verification_assertion(
        proposition or _proposition(),
        proposition_ref="p0",
        output_context_graph=graph or _graph(),
        **kwargs,
    )


def test_simple_atomic_proposition_projects_without_material_context() -> None:
    assertion = _project(graph=_graph())
    assert assertion.source_proposition_id == PROP_ID
    assert assertion.subject == "Atlas"
    assert assertion.subject_identity.status is SemanticIdentityStatus.UNRESOLVED
    assert assertion.subject_identity.key is None
    assert assertion.predicate_identity.status is SemanticIdentityStatus.UNRESOLVED
    assert assertion.predicate_identity.key is None
    assert assertion.material_contexts == ()
    assert assertion.composition_eligibility is ComparisonEligibility.UNKNOWN


def _resolved_identity(
    key: str,
    *,
    namespace: str = "catalog:entity",
    basis: SemanticIdentityBasis = SemanticIdentityBasis.EXPLICIT_CANONICAL_REFERENCE,
) -> SemanticIdentity:
    return SemanticIdentity(
        status=SemanticIdentityStatus.RESOLVED,
        key=key,
        namespace=namespace,
        normalization_version="catalog-id-v1",
        basis=basis,
        provenance_ref="source-record-1",
    )


def test_projection_preserves_raw_subject_and_predicate_with_identities() -> None:
    proposition = _proposition()
    subject_identity = _resolved_identity("entity:42")
    predicate_identity = _resolved_identity("relation:stability", namespace="catalog:relation")

    assertion = _project(
        proposition,
        subject_identity=subject_identity,
        predicate_identity=predicate_identity,
    )

    assert assertion.subject == "Atlas"
    assert assertion.predicate == "is"
    assert assertion.subject_identity is subject_identity
    assert assertion.predicate_identity is predicate_identity


@pytest.mark.parametrize("field", ["subject_identity", "predicate_identity"])
def test_same_surface_does_not_fabricate_resolved_identity(field: str) -> None:
    first = _project(_proposition())
    second = _project(_proposition())
    assert getattr(first, field).status is SemanticIdentityStatus.UNRESOLVED
    assert getattr(second, field).status is SemanticIdentityStatus.UNRESOLVED
    assert getattr(first, field).key is None
    assert getattr(second, field).key is None


def test_different_raw_subjects_do_not_automatically_create_identity_mismatch() -> None:
    first = _project(_proposition())
    second_proposition = _proposition().model_copy(update={"subject": "Atlas Labs"})
    second = _project(second_proposition)

    assert first.subject != second.subject
    assert first.subject_identity.status is SemanticIdentityStatus.UNRESOLVED
    assert second.subject_identity.status is SemanticIdentityStatus.UNRESOLVED
    assert first.subject_identity.key is second.subject_identity.key is None


def test_different_raw_predicates_do_not_automatically_create_identity_mismatch() -> None:
    first = _project(_proposition())
    second_proposition = _proposition().model_copy(update={"predicate": "remains"})
    second = _project(second_proposition)

    assert first.predicate != second.predicate
    assert first.predicate_identity.status is SemanticIdentityStatus.UNRESOLVED
    assert second.predicate_identity.status is SemanticIdentityStatus.UNRESOLVED
    assert first.predicate_identity.key is second.predicate_identity.key is None


def test_semantic_identity_keys_represent_same_subject_across_raw_surfaces() -> None:
    identity = _resolved_identity("entity:42")
    first = _project(_proposition(), subject_identity=identity)
    second_proposition = _proposition().model_copy(update={"subject": "Atlas Corporation"})
    second = _project(second_proposition, subject_identity=identity)

    assert first.subject != second.subject
    assert first.subject_identity == second.subject_identity
    assert first.subject_identity.key == second.subject_identity.key


def test_semantic_identity_keys_represent_distinct_subjects() -> None:
    first = _project(_proposition(), subject_identity=_resolved_identity("entity:42"))
    second = _project(
        _proposition(), subject_identity=_resolved_identity("entity:84")
    )

    assert first.subject_identity.key != second.subject_identity.key
    assert first.subject_identity.namespace == second.subject_identity.namespace


def test_semantic_identity_keys_represent_same_predicate_across_raw_surfaces() -> None:
    identity = _resolved_identity("relation:stability", namespace="catalog:relation")
    first = _project(_proposition(), predicate_identity=identity)
    second_proposition = _proposition().model_copy(update={"predicate": "remains stable"})
    second = _project(second_proposition, predicate_identity=identity)

    assert first.predicate != second.predicate
    assert first.predicate_identity == second.predicate_identity


def test_semantic_identity_keys_represent_distinct_predicates() -> None:
    first = _project(
        _proposition(),
        predicate_identity=_resolved_identity("relation:stability", namespace="catalog:relation"),
    )
    second = _project(
        _proposition(),
        predicate_identity=_resolved_identity("relation:growth", namespace="catalog:relation"),
    )

    assert first.predicate_identity.key != second.predicate_identity.key
    assert first.predicate_identity.namespace == second.predicate_identity.namespace


def test_namespace_is_preserved_and_part_of_identity_contract() -> None:
    first = _resolved_identity("123", namespace="namespace:a")
    second = _resolved_identity("123", namespace="namespace:b")

    assert first.key == second.key
    assert first.namespace != second.namespace
    assert first != second


@pytest.mark.parametrize(
    ("identity", "message"),
    [
        (
            {"status": "RESOLVED", "namespace": "catalog:entity"},
            "resolved_semantic_identity_requires_key",
        ),
        (
            {"status": "RESOLVED", "key": "entity:1"},
            "resolved_semantic_identity_requires_namespace",
        ),
        (
            {
                "status": "RESOLVED",
                "key": "entity:1",
                "namespace": "catalog:entity",
                "normalization_version": "catalog-id-v1",
            },
            "resolved_semantic_identity_requires_auditable_basis",
        ),
        (
            {
                "status": "RESOLVED",
                "key": "entity:1",
                "namespace": "catalog:entity",
                "basis": "EXPLICIT_CANONICAL_REFERENCE",
            },
            "resolved_semantic_identity_requires_normalization_version",
        ),
        (
            {"status": "UNRESOLVED", "key": "entity:1"},
            "unresolved_semantic_identity_must_not_have_key",
        ),
    ],
)
def test_semantic_identity_rejects_inconsistent_resolution_fields(
    identity: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        SemanticIdentity(**identity)


def test_projection_is_deterministic_with_explicit_identity_inputs() -> None:
    proposition = _proposition()
    before = proposition.model_dump(mode="json")
    identity = _resolved_identity("entity:42")

    first = _project(proposition, subject_identity=identity)
    second = _project(proposition, subject_identity=identity)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert proposition.model_dump(mode="json") == before


@pytest.mark.parametrize(
    ("context_type", "expected"),
    [
        (ContextType.TEMPORAL, "TIME"),
        (ContextType.SCOPE, "SCOPE"),
        (ContextType.CONDITION, "CONDITION"),
        (ContextType.EPISTEMIC, "ATTRIBUTION"),
    ],
)
def test_material_context_families_are_retained(context_type: ContextType, expected: str) -> None:
    context = _context(context_type, value={"operator": "REPORTS", "agent": "Atlas Research"})
    assertion = _project(graph=_graph(context))
    assert len(assertion.material_contexts) == 1
    assert assertion.material_contexts[0].context.context_type is context_type
    assert expected in {item.value for item in ComparisonDimension}


def test_modality_and_attribution_core_fields_are_not_dropped() -> None:
    proposition = _proposition().model_copy(
        update={"modality": "possibly", "attribution": "Atlas Research"}
    )
    assertion = _project(proposition)
    assert assertion.modality == "possibly"
    assert assertion.attribution == "Atlas Research"


def test_measurement_semantics_reuses_v33_typed_contract() -> None:
    from app.domain.source_context_extraction_v3_3 import (
        MeasurementIdentityTypeV33,
        MeasurementSemanticsV33,
        SourceContextV33,
    )

    semantics = MeasurementSemanticsV33(
        measurement_identity={
            "name": "latency",
            "identity_type": MeasurementIdentityTypeV33.NAMED_METRIC,
            "source_span": {"evidence_ref": "ev1", "occurrence_index": 0},
        },
        observed_value={"kind": "INTEGER", "raw_value": "12", "value": 12},
        unit="ms",
    )
    typed_context = SourceContextV33(
        context_ref="ctx1", context_type=ContextType.MEASUREMENT, semantic_value=semantics
    )
    context = _context(ContextType.MEASUREMENT)
    assertion = _project(
        graph=_graph(context), measurement_contexts={"ctx1": typed_context}
    )
    assert assertion.material_contexts[0].measurement_semantics == semantics
    assert isinstance(assertion.material_contexts[0].measurement_semantics, MeasurementSemanticsV33)


def test_unknown_context_materiality_is_retained() -> None:
    assertion = _project(graph=_graph(_context(ContextType.VERSION, materiality="UNKNOWN")))
    assert len(assertion.material_contexts) == 1
    assert assertion.material_context_unknown is True


def test_attachment_ambiguity_is_preserved_without_repair() -> None:
    context = _context(
        ContextType.TEMPORAL,
        status=AttachmentStatus.UNKNOWN,
        target_ref=None,
        scope_kind=ContextScopeKind.UNKNOWN,
    )
    assertion = _project(graph=_graph(context))
    assert (
        assertion.material_contexts[0].attachment_resolution
        is ContextAttachmentResolution.AMBIGUOUS
    )
    assert assertion.material_contexts[0].reason_codes == (
        ComparisonReasonCode.CONTEXT_ATTACHMENT_AMBIGUOUS,
    )


@pytest.mark.parametrize("operator", [AssertionGroupOperator.OR, AssertionGroupOperator.XOR])
def test_or_xor_descendant_is_not_comparison_eligible(operator: AssertionGroupOperator) -> None:
    group_id = UUID("30000000-0000-0000-0000-000000000001")
    second = _proposition().model_copy(update={"proposition_id": PROP2_ID})
    group = AssertionGroup(
        group_id=group_id,
        operator=operator,
        source_span="Atlas is stable or Atlas is unstable",
        members=(
            AssertionGroupMember(
                member_type="proposition",
                position=0,
                proposition_id=PROP_ID,
                verification_eligibility=VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED,
                source_span="Atlas is stable",
            ),
            AssertionGroupMember(
                member_type="proposition",
                position=1,
                proposition_id=PROP2_ID,
                verification_eligibility=VerificationEligibility.NOT_INDEPENDENTLY_ASSERTED,
                source_span="Atlas is unstable",
            ),
        ),
    )
    composition = AssertionComposition(claim_id=CLAIM_ID, root_group_id=group_id, groups=(group,))
    assertion = _project(
        graph=_graph(),
        composition=composition,
        claim_id=CLAIM_ID,
        claim_text="Atlas is stable or Atlas is unstable",
        proposition_ids={PROP_ID, PROP2_ID},
    )
    assert assertion.composition_eligibility is ComparisonEligibility.INELIGIBLE
    assert second.proposition_id == PROP2_ID


def test_unknown_composition_fails_closed() -> None:
    assertion = _project(graph=_graph(), composition=AssertionComposition.model_construct())
    assert assertion.composition_eligibility is ComparisonEligibility.UNKNOWN


def test_insufficient_grounding_is_retained_and_not_promoted() -> None:
    assessment = NodeGroundingAssessment(
        target_kind=EvidenceTargetKind.ASSERTION,
        target_ref="p0",
        dimensions=DimensionGrounding(
            textual_grounding=GroundingDimensionStatus.VALID,
            semantic_support=GroundingDimensionStatus.MISSING_EVIDENCE,
            structural_grounding=GroundingDimensionStatus.VALID,
        ),
        usability=MaterialNodeUsability.PARTIALLY_GROUNDED_UNRESOLVED,
    )
    assertion = _project(graph=_graph(), source_grounding_assessment=assessment)
    assert assertion.grounding_usability is AssertionGroundingUsability.INSUFFICIENT
    assert assertion.grounding_dimensions == assessment.dimensions


def test_decimal_and_range_values_remain_typed() -> None:
    proposition = _proposition(numeric=True)
    assertion = _project(proposition)
    assert assertion.typed_value.kind is PropositionValueKind.QUANTITY
    assert assertion.typed_value.numeric is not None
    assert assertion.typed_value.numeric.value == Decimal("10.00")
    ranged = PropositionValue(
        kind=PropositionValueKind.RANGE,
        numeric=NumericValue(
            operator=NumericOperator.RANGE,
            lower_bound=Decimal("1.2"),
            upper_bound=Decimal("4.8"),
        ),
    )
    range_assertion = _project(proposition.model_copy(update={"value": ranged}))
    assert range_assertion.typed_value.kind is PropositionValueKind.RANGE
    assert range_assertion.typed_value.numeric is not None
    assert range_assertion.typed_value.numeric.lower_bound == Decimal("1.2")


def test_repeated_projection_is_deterministic_and_does_not_mutate_source() -> None:
    proposition = _proposition(numeric=True)
    before = proposition.model_dump(mode="json")
    first = _project(proposition, _graph(_context(ContextType.TEMPORAL)))
    second = _project(proposition, _graph(_context(ContextType.TEMPORAL)))
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert proposition.model_dump(mode="json") == before


def test_projection_is_ephemeral_and_does_not_create_persistence_state() -> None:
    assertion = _project()
    assert "id" not in type(assertion).model_fields
    assert "created_at" not in type(assertion).model_fields
    assert "repository" not in type(assertion).model_fields


def test_comparability_and_relation_contract_vocabularies_are_separate() -> None:
    assert {item.value for item in ComparabilityStatus} == {
        "COMPARABLE",
        "NOT_COMPARABLE",
        "UNKNOWN",
    }
    assert {item.value for item in PropositionRelation} == {
        "same",
        "related",
        "contradicts",
        "unknown",
    }


def _pair(
    comparability: ComparabilityStatus,
    relation: PropositionRelation,
    *,
    reason_codes: tuple[ComparisonReasonCode, ...] = (),
) -> PropositionPairAssessment:
    return PropositionPairAssessment(
        left_assertion_ref="a",
        right_assertion_ref="b",
        comparability=comparability,
        relation=relation,
        reason_codes=reason_codes,
    )


@pytest.mark.parametrize(
    ("comparability", "relation"),
    [
        (ComparabilityStatus.COMPARABLE, PropositionRelation.SAME),
        (ComparabilityStatus.COMPARABLE, PropositionRelation.CONTRADICTS),
    ],
)
def test_same_and_contradicts_are_legal_only_when_comparable(
    comparability: ComparabilityStatus, relation: PropositionRelation
) -> None:
    assert _pair(comparability, relation).relation is relation


@pytest.mark.parametrize(
    ("comparability", "relation", "error"),
    [
        (ComparabilityStatus.NOT_COMPARABLE, PropositionRelation.SAME, "same_requires"),
        (ComparabilityStatus.UNKNOWN, PropositionRelation.SAME, "same_requires"),
        (
            ComparabilityStatus.NOT_COMPARABLE,
            PropositionRelation.CONTRADICTS,
            "contradicts_requires",
        ),
        (
            ComparabilityStatus.UNKNOWN,
            PropositionRelation.CONTRADICTS,
            "contradicts_requires",
        ),
    ],
)
def test_same_and_contradicts_reject_non_comparable_states(
    comparability: ComparabilityStatus, relation: PropositionRelation, error: str
) -> None:
    with pytest.raises(ValidationError, match=error):
        _pair(comparability, relation)


@pytest.mark.parametrize(
    "comparability",
    [ComparabilityStatus.COMPARABLE, ComparabilityStatus.NOT_COMPARABLE],
)
def test_related_requires_explicit_association_but_not_comparability(
    comparability: ComparabilityStatus,
) -> None:
    assessment = _pair(
        comparability,
        PropositionRelation.RELATED,
        reason_codes=(ComparisonReasonCode.EXPLICIT_SEMANTIC_ASSOCIATION,),
    )
    assert assessment.relation is PropositionRelation.RELATED


def test_unknown_comparability_can_coexist_with_explicit_related_association() -> None:
    assessment = _pair(
        ComparabilityStatus.UNKNOWN,
        PropositionRelation.RELATED,
        reason_codes=(ComparisonReasonCode.EXPLICIT_SEMANTIC_ASSOCIATION,),
    )
    assert assessment.comparability is ComparabilityStatus.UNKNOWN
    assert assessment.relation is PropositionRelation.RELATED


@pytest.mark.parametrize("comparability", list(ComparabilityStatus))
def test_related_without_explicit_association_is_illegal(
    comparability: ComparabilityStatus,
) -> None:
    with pytest.raises(ValidationError, match="related_requires_explicit"):
        _pair(
            comparability,
            PropositionRelation.RELATED,
            reason_codes=(ComparisonReasonCode.TIME_MISMATCH,),
        )


@pytest.mark.parametrize("comparability", list(ComparabilityStatus))
def test_unknown_relation_can_coexist_with_any_comparability(
    comparability: ComparabilityStatus,
) -> None:
    assessment = _pair(comparability, PropositionRelation.UNKNOWN)
    assert assessment.relation is PropositionRelation.UNKNOWN


def test_a2_style_assessment_can_leave_relation_unevaluated() -> None:
    assessment = _pair(ComparabilityStatus.NOT_COMPARABLE, PropositionRelation.UNKNOWN)
    assert assessment.comparability is ComparabilityStatus.NOT_COMPARABLE
    assert assessment.relation is PropositionRelation.UNKNOWN


def test_pair_assessment_defaults_to_legal_fail_closed_unknown() -> None:
    pair = PropositionPairAssessment(left_assertion_ref="a", right_assertion_ref="b")
    assert pair.comparability is ComparabilityStatus.UNKNOWN
    assert pair.relation is PropositionRelation.UNKNOWN
    assert pair.relation_eligible is None
    dimension = DimensionAssessment(
        dimension=ComparisonDimension.MEASUREMENT,
        compatibility=DimensionCompatibility.UNKNOWN,
        reason_codes=(ComparisonReasonCode.MEASUREMENT_UNKNOWN,),
    )
    assert dimension.compatibility is DimensionCompatibility.UNKNOWN
