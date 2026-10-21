from __future__ import annotations

from decimal import Decimal
from uuid import UUID

import pytest
from app.domain.atomic_proposition import (
    NumericOperator,
    NumericValue,
    PropositionPolarity,
    PropositionRelation,
    PropositionValue,
    PropositionValueKind,
)
from app.domain.comparability_evaluator import ComparabilityEvaluator
from app.domain.relation_evaluator import (
    ExplicitSemanticAssociation,
    RelationEvaluator,
)
from app.domain.source_context_extraction_v3_3 import (
    MeasurementIdentityTypeV33,
    MeasurementSemanticsV33,
)
from app.domain.source_typed_value_grounding import (
    SourceEvidenceSpanInput,
    SourceGroundingState,
    TypedValueComponentRole,
    TypedValueEvidenceBinding,
    assess_typed_value_grounding,
)
from app.domain.truth_condition_context import (
    ContextAttachment,
    ContextOrigin,
    ContextScope,
    ContextScopeKind,
    ContextType,
    TruthConditionContext,
)
from app.domain.verification_assertion import (
    AssertionGroundingUsability,
    ComparabilityStatus,
    ComparisonEligibility,
    ComparisonReasonCode,
    ContextAttachmentResolution,
    SemanticIdentity,
    SemanticIdentityBasis,
    SemanticIdentityStatus,
    VerificationAssertion,
)
from app.domain.verification_assertion import (
    MaterialContextProjection as AssertionMaterialContextProjection,
)


def _identity(key: str, namespace: str) -> SemanticIdentity:
    return SemanticIdentity(
        status=SemanticIdentityStatus.RESOLVED,
        key=key,
        namespace=namespace,
        normalization_version="test-id-v1",
        basis=SemanticIdentityBasis.EXPLICIT_CANONICAL_REFERENCE,
    )


def _assertion(
    ref: str,
    value: PropositionValue,
    *,
    polarity: PropositionPolarity = PropositionPolarity.ASSERTED,
) -> VerificationAssertion:
    return VerificationAssertion(
        assertion_ref=ref,
        source_proposition_id=UUID("20000000-0000-0000-0000-000000000001"),
        subject="surface subject is not compared",
        subject_identity=_identity("subject-1", "test.subject"),
        predicate="surface predicate is not compared",
        predicate_identity=_identity("predicate-1", "test.predicate"),
        typed_value=value,
        polarity=polarity,
        composition_eligibility=ComparisonEligibility.ELIGIBLE,
        grounding_usability=AssertionGroundingUsability.SUFFICIENT,
    )


def _numeric(value: str, *, unit: str = "%") -> PropositionValue:
    return PropositionValue(
        kind=PropositionValueKind.QUANTITY,
        numeric=NumericValue(
            operator=NumericOperator.EXACT,
            value=Decimal(value),
            unit=unit,
        ),
    )


def _assessment(left: VerificationAssertion, right: VerificationAssertion):
    return ComparabilityEvaluator.evaluate(left, right)


def _evaluate(left, right, assessment=None, association=None):
    return RelationEvaluator.evaluate(
        left,
        right,
        assessment or _assessment(left, right),
        association=association,
    )


def _association(left: str = "a", right: str = "b", *, established: bool = True):
    return ExplicitSemanticAssociation(
        left_assertion_ref=left,
        right_assertion_ref=right,
        established=established,
        provenance_ref="upstream-association-1",
    )


def test_same_exact_numeric_values_and_decimal_canonicalization() -> None:
    left = _assertion("a", _numeric("38.40"))
    right = _assertion("b", _numeric("38.4"))
    result = _evaluate(left, right)
    assert result.comparability is ComparabilityStatus.COMPARABLE
    assert result.relation is PropositionRelation.SAME
    assert result.relation_eligible is True
    assert ComparisonReasonCode.SAME_TYPED_VALUE in result.reason_codes


def test_same_resolved_boolean_value() -> None:
    left = _assertion("a", PropositionValue(kind=PropositionValueKind.BOOLEAN, boolean=True))
    right = _assertion("b", PropositionValue(kind=PropositionValueKind.BOOLEAN, boolean=True))
    assert _evaluate(left, right).relation is PropositionRelation.SAME


@pytest.mark.parametrize("kind", [PropositionValueKind.TEXT, PropositionValueKind.ENTITY])
def test_raw_text_or_entity_equality_never_establishes_same(kind) -> None:
    left = _assertion("a", PropositionValue(kind=kind, text="same surface value"))
    right = _assertion("b", PropositionValue(kind=kind, text="same surface value"))
    result = _evaluate(left, right)
    assert result.relation is PropositionRelation.UNKNOWN
    assert result.relation_eligible is False


def test_exact_range_representation_can_be_same() -> None:
    value = PropositionValue(
        kind=PropositionValueKind.RANGE,
        numeric=NumericValue(
            operator=NumericOperator.RANGE,
            lower_bound=Decimal("1.0"),
            upper_bound=Decimal("3.00"),
            unit="kg",
        ),
    )
    assert _evaluate(_assertion("a", value), _assertion("b", value)).relation is (
        PropositionRelation.SAME
    )


def test_v34_grounded_range_flows_to_frozen_a3_same_without_overlap_logic() -> None:
    value = PropositionValue(
        kind=PropositionValueKind.RANGE,
        numeric=NumericValue(
            operator=NumericOperator.RANGE,
            lower_bound=Decimal("10"),
            upper_bound=Decimal("20"),
            unit="kg",
        ),
    )
    span_text = "10\u201320 kg"
    source_ref = "source:range-test"
    snapshot_ref = "a" * 64
    roles = (
        TypedValueComponentRole.LOWER_BOUND,
        TypedValueComponentRole.UPPER_BOUND,
        TypedValueComponentRole.UNIT,
    )
    bindings = tuple(
        TypedValueEvidenceBinding(
            binding_ref=f"range-{role.value.lower()}",
            assertion_ref="A1",
            evidence_ref="range-span",
            occurrence_index=0,
            char_start=0,
            char_end=len(span_text),
            component_role=role,
            source_ref=source_ref,
            snapshot_ref=snapshot_ref,
        )
        for role in roles
    )
    grounding = assess_typed_value_grounding(
        assertion_ref="A1",
        value=value,
        bindings=bindings,
        spans=(
            SourceEvidenceSpanInput(
                evidence_ref="range-span",
                exact_text=span_text,
                occurrence_index=0,
                char_start=0,
                char_end=len(span_text),
            ),
        ),
        source_ref=source_ref,
        snapshot_ref=snapshot_ref,
    )
    assert grounding.status is SourceGroundingState.VALID
    assert grounding.normalized_value is not None
    assert (
        _evaluate(_assertion("a", grounding.normalized_value), _assertion("b", value)).relation
        is PropositionRelation.SAME
    )


def test_v34_grounded_bound_requires_same_operator_for_frozen_a3_same() -> None:
    value = PropositionValue(
        kind=PropositionValueKind.QUANTITY,
        numeric=NumericValue(operator=NumericOperator.GTE, value=Decimal("10")),
    )
    span_text = ">= 10"
    source_ref = "source:bound-test"
    snapshot_ref = "b" * 64
    roles = (
        TypedValueComponentRole.NUMERIC_MAGNITUDE,
        TypedValueComponentRole.NUMERIC_OPERATOR,
    )
    bindings = tuple(
        TypedValueEvidenceBinding(
            binding_ref=f"bound-{role.value.lower()}",
            assertion_ref="A1",
            evidence_ref="bound-span",
            occurrence_index=0,
            char_start=0,
            char_end=len(span_text),
            component_role=role,
            source_ref=source_ref,
            snapshot_ref=snapshot_ref,
        )
        for role in roles
    )
    grounding = assess_typed_value_grounding(
        assertion_ref="A1",
        value=value,
        bindings=bindings,
        spans=(
            SourceEvidenceSpanInput(
                evidence_ref="bound-span",
                exact_text=span_text,
                occurrence_index=0,
                char_start=0,
                char_end=len(span_text),
            ),
        ),
        source_ref=source_ref,
        snapshot_ref=snapshot_ref,
    )
    assert grounding.status is SourceGroundingState.VALID
    assert grounding.normalized_value is not None
    assert (
        _evaluate(_assertion("a", grounding.normalized_value), _assertion("b", value)).relation
        is PropositionRelation.SAME
    )
    strict = PropositionValue(
        kind=PropositionValueKind.QUANTITY,
        numeric=NumericValue(operator=NumericOperator.GT, value=Decimal("10")),
    )
    assert (
        _evaluate(_assertion("a", grounding.normalized_value), _assertion("c", strict)).relation
        is not PropositionRelation.SAME
    )


@pytest.mark.parametrize(
    "status", [ComparabilityStatus.NOT_COMPARABLE, ComparabilityStatus.UNKNOWN]
)
def test_same_value_cannot_override_non_comparable_or_unknown(status) -> None:
    left = _assertion("a", _numeric("4"))
    right = _assertion("b", _numeric("4"))
    assessment = _assessment(left, right).model_copy(update={"comparability": status})
    result = _evaluate(left, right, assessment)
    assert result.relation is PropositionRelation.UNKNOWN
    assert result.relation_eligible is False


def test_same_value_opposite_polarity_is_not_same_and_directly_contradicts() -> None:
    left = _assertion("a", _numeric("4"), polarity=PropositionPolarity.ASSERTED)
    right = _assertion("b", _numeric("4"), polarity=PropositionPolarity.NEGATED)
    result = _evaluate(left, right)
    assert result.relation is PropositionRelation.CONTRADICTS
    assert result.relation_eligible is True
    assert ComparisonReasonCode.DIRECT_POLARITY_CONFLICT in result.reason_codes


def test_boolean_true_vs_false_is_direct_conflict_for_same_comparable_target() -> None:
    left = _assertion("a", PropositionValue(kind=PropositionValueKind.BOOLEAN, boolean=True))
    right = _assertion("b", PropositionValue(kind=PropositionValueKind.BOOLEAN, boolean=False))
    result = _evaluate(left, right)
    assert result.relation is PropositionRelation.CONTRADICTS
    assert ComparisonReasonCode.DIRECT_BOOLEAN_VALUE_CONFLICT in result.reason_codes


@pytest.mark.parametrize(
    "status", [ComparabilityStatus.NOT_COMPARABLE, ComparabilityStatus.UNKNOWN]
)
def test_opposite_polarity_does_not_contradict_unless_comparable(status) -> None:
    left = _assertion("a", _numeric("4"))
    right = _assertion("b", _numeric("4"), polarity=PropositionPolarity.NEGATED)
    assessment = _assessment(left, right).model_copy(update={"comparability": status})
    result = _evaluate(left, right, assessment)
    assert result.relation is PropositionRelation.UNKNOWN


@pytest.mark.parametrize(
    ("left_value", "right_value"),
    [
        (
            PropositionValue(kind=PropositionValueKind.TEXT, text="A"),
            PropositionValue(kind=PropositionValueKind.TEXT, text="B"),
        ),
        (_numeric("100"), _numeric("120")),
    ],
)
def test_different_generic_values_are_not_contradictions(left_value, right_value) -> None:
    result = _evaluate(_assertion("a", left_value), _assertion("b", right_value))
    assert result.relation is PropositionRelation.UNKNOWN
    assert result.relation_eligible is False


def test_v33_measurement_context_has_no_single_valued_axiom() -> None:
    # The frozen v3.3 semantics only type an observation; they do not assert
    # that all observations for this measurement are mutually exclusive.
    def with_measurement(ref: str, value: str) -> VerificationAssertion:
        semantics = MeasurementSemanticsV33.model_validate(
            {
                "measurement_identity": {
                    "name": "metric-key",
                    "identity_type": MeasurementIdentityTypeV33.NAMED_METRIC,
                    "source_span": {"evidence_ref": "ev1", "occurrence_index": 0},
                },
                "observed_value": {
                    "kind": "DECIMAL",
                    "raw_value": value,
                    "value": value,
                },
                "unit": "%",
            }
        )
        context = TruthConditionContext(
            context_ref="measurement-context",
            semantic_signature="measurement-context",
            context_type=ContextType.MEASUREMENT,
            semantic_value={"identity": "measurement"},
            lexical_value=None,
            source_span=None,
            materiality="MATERIAL",
            scope=ContextScope(kind=ContextScopeKind.PROPOSITION, target_refs=(ref,)),
            attachments=(
                ContextAttachment(
                    context_ref="measurement-context",
                    target_ref=ref,
                    target_kind="proposition",
                    relation="APPLIES_TO",
                    status="RESOLVED",
                    reason_code="test",
                ),
            ),
            origin=ContextOrigin.OUTPUT,
        )
        return _assertion(
            ref,
            _numeric(value),
            polarity=PropositionPolarity.ASSERTED,
        ).model_copy(
            update={
                "material_contexts": (
                    AssertionMaterialContextProjection(
                        context=context,
                        attachment_resolution=ContextAttachmentResolution.RESOLVED,
                        measurement_semantics=semantics,
                    ),
                ),
            }
        )

    left, right = with_measurement("a", "100"), with_measurement("b", "120")
    assert _evaluate(left, right).relation is PropositionRelation.UNKNOWN


@pytest.mark.parametrize(
    ("comparability", "expected"),
    [
        (ComparabilityStatus.NOT_COMPARABLE, PropositionRelation.RELATED),
        (ComparabilityStatus.UNKNOWN, PropositionRelation.RELATED),
        (ComparabilityStatus.COMPARABLE, PropositionRelation.RELATED),
    ],
)
def test_explicit_association_is_only_related_carrier(comparability, expected) -> None:
    left = _assertion("a", _numeric("4"))
    right = _assertion("b", _numeric("5"))
    assessment = _assessment(left, right).model_copy(update={"comparability": comparability})
    association = _association("b", "a")  # undirected/canonical pair binding
    result = _evaluate(left, right, assessment, association)
    assert result.relation is expected
    assert result.relation_eligible is True
    assert ComparisonReasonCode.EXPLICIT_SEMANTIC_ASSOCIATION in result.reason_codes


def test_no_association_same_subject_source_or_claim_never_infers_related() -> None:
    left = _assertion("a", _numeric("4"))
    right = _assertion("b", _numeric("5"))
    assert _evaluate(left, right).relation is PropositionRelation.UNKNOWN


def test_false_or_mismatched_association_does_not_infer_related() -> None:
    left = _assertion("a", _numeric("4"))
    right = _assertion("b", _numeric("5"))
    with pytest.raises(ValueError, match="semantic_association_pair_mismatch"):
        _evaluate(left, right, association=_association("a", "other"))
    assert _evaluate(left, right, association=_association(established=False)).relation is (
        PropositionRelation.UNKNOWN
    )


def test_same_and_contradiction_precede_generic_association() -> None:
    same_left = _assertion("a", _numeric("4"))
    same_right = _assertion("b", _numeric("4"))
    assert _evaluate(same_left, same_right, association=_association()).relation is (
        PropositionRelation.SAME
    )

    conflict_left = _assertion("a", _numeric("4"))
    conflict_right = _assertion("b", _numeric("4"), polarity=PropositionPolarity.NEGATED)
    assert _evaluate(conflict_left, conflict_right, association=_association()).relation is (
        PropositionRelation.CONTRADICTS
    )


def test_a2_pair_refs_must_match_and_a2_is_not_rerun_or_mutated() -> None:
    left = _assertion("a", _numeric("4"))
    right = _assertion("b", _numeric("4"))
    assessment = _assessment(left, right)
    before = assessment.model_dump(mode="json")
    result = _evaluate(left, right, assessment)
    assert result.comparability is assessment.comparability
    assert assessment.model_dump(mode="json") == before

    wrong_pair = assessment.model_copy(update={"left_assertion_ref": "other"})
    with pytest.raises(ValueError, match="comparability_assessment_pair_mismatch"):
        _evaluate(left, right, wrong_pair)


def test_non_comparable_never_same_or_contradicts_even_with_equal_value() -> None:
    left = _assertion("a", _numeric("4"))
    right = _assertion("b", _numeric("4"))
    assessment = _assessment(left, right).model_copy(
        update={"comparability": ComparabilityStatus.NOT_COMPARABLE}
    )
    result = _evaluate(left, right, assessment)
    assert result.relation not in {PropositionRelation.SAME, PropositionRelation.CONTRADICTS}
    assert result.comparability is ComparabilityStatus.NOT_COMPARABLE


def test_relation_evaluation_is_deterministic_and_does_not_mutate_assertions() -> None:
    left = _assertion("a", _numeric("4"))
    right = _assertion("b", _numeric("4"))
    before = (left.model_dump(mode="json"), right.model_dump(mode="json"))
    first = _evaluate(left, right)
    second = _evaluate(left, right)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert (left.model_dump(mode="json"), right.model_dump(mode="json")) == before


def test_explicit_association_requires_provenance_and_distinct_pair() -> None:
    with pytest.raises(ValueError):
        ExplicitSemanticAssociation(
            left_assertion_ref="a",
            right_assertion_ref="a",
            established=True,
            provenance_ref="p",
        )
    with pytest.raises(ValueError):
        ExplicitSemanticAssociation(
            left_assertion_ref="a",
            right_assertion_ref="b",
            established=True,
            provenance_ref="",
        )


def test_unknown_polarity_never_creates_same_or_contradiction() -> None:
    value = PropositionValue(kind=PropositionValueKind.BOOLEAN, boolean=True)
    left = _assertion("a", value, polarity=PropositionPolarity.UNKNOWN)
    right = _assertion("b", value)
    result = _evaluate(left, right)
    assert result.relation is PropositionRelation.UNKNOWN
    assert result.relation_eligible is False
