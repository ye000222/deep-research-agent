from __future__ import annotations

from decimal import Decimal
from uuid import UUID

import pytest
from app.domain.atomic_proposition import (
    MaterialQualifier,
    NumericOperator,
    NumericValue,
    PropositionPolarity,
    PropositionRelation,
    PropositionValue,
    PropositionValueKind,
    QualifierCategory,
    QualifierMateriality,
)
from app.domain.comparability_evaluator import ComparabilityEvaluator
from app.domain.source_context_extraction_v3_3 import (
    MeasurementIdentityTypeV33,
    MeasurementSemanticsV33,
)
from app.domain.source_context_grounding_v2 import (
    DimensionGrounding,
    GroundingDimensionStatus,
)
from app.domain.truth_condition_context import (
    ContextOrigin,
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
    DimensionCompatibility,
    MaterialContextProjection,
    SemanticIdentity,
    SemanticIdentityBasis,
    SemanticIdentityStatus,
    VerificationAssertion,
)


def _identity(
    key: str, *, namespace: str = "test.entity", version: str = "id-v1"
) -> SemanticIdentity:
    return SemanticIdentity(
        status=SemanticIdentityStatus.RESOLVED,
        key=key,
        namespace=namespace,
        normalization_version=version,
        basis=SemanticIdentityBasis.EXPLICIT_CANONICAL_REFERENCE,
    )


def _context(
    kind: ContextType,
    value: dict[str, object],
    *,
    materiality: str = "MATERIAL",
    attachment: ContextAttachmentResolution = ContextAttachmentResolution.RESOLVED,
    ref: str = "ctx",
    measurement: MeasurementSemanticsV33 | None = None,
) -> MaterialContextProjection:
    context = TruthConditionContext(
        context_ref=ref,
        semantic_signature=f"typed:{kind.value}:{ref}",
        context_type=kind,
        semantic_value=value,
        lexical_value=None,
        source_span=None,
        materiality=materiality,
        scope=ContextScope(kind=ContextScopeKind.PROPOSITION, target_refs=("p",)),
        attachments=(),
        origin=ContextOrigin.OUTPUT,
    )
    return MaterialContextProjection(
        context=context,
        attachment_resolution=attachment,
        measurement_semantics=measurement,
    )


def _assertion(
    *,
    subject: str | None = "surface subject",
    predicate: str | None = "surface predicate",
    subject_identity: SemanticIdentity | None = None,
    predicate_identity: SemanticIdentity | None = None,
    value: PropositionValue | None = None,
    polarity: PropositionPolarity = PropositionPolarity.ASSERTED,
    contexts: tuple[MaterialContextProjection, ...] = (),
    eligibility: ComparisonEligibility = ComparisonEligibility.ELIGIBLE,
    grounding: AssertionGroundingUsability = AssertionGroundingUsability.SUFFICIENT,
    modality: str | None = None,
    attribution: str | None = None,
    incomplete_context_detection: bool = False,
    ref: str = "assertion",
) -> VerificationAssertion:
    return VerificationAssertion(
        assertion_ref=ref,
        source_proposition_id=UUID("20000000-0000-0000-0000-000000000001"),
        subject=subject,
        subject_identity=subject_identity or SemanticIdentity.unresolved(),
        predicate=predicate,
        predicate_identity=predicate_identity or SemanticIdentity.unresolved(),
        typed_value=value or PropositionValue(kind=PropositionValueKind.BOOLEAN, boolean=True),
        polarity=polarity,
        material_contexts=contexts,
        modality=modality,
        attribution=attribution,
        composition_eligibility=eligibility,
        grounding_usability=grounding,
        grounding_dimensions=DimensionGrounding(
            textual_grounding=GroundingDimensionStatus.VALID,
            semantic_support=GroundingDimensionStatus.VALID,
            structural_grounding=GroundingDimensionStatus.VALID,
        ),
        source_context_detection_incomplete=incomplete_context_detection,
    )


def _resolved_pair(**kwargs: object) -> tuple[VerificationAssertion, VerificationAssertion]:
    common = {
        "subject_identity": _identity("entity-1"),
        "predicate_identity": _identity("predicate-1", namespace="test.predicate"),
    }
    common.update(kwargs)
    return _assertion(**common), _assertion(**common, ref="right")


def _evaluate(left: VerificationAssertion, right: VerificationAssertion):
    return ComparabilityEvaluator.evaluate(left, right)


def _dimension(result, dimension: ComparisonDimension) -> DimensionCompatibility:
    values = [
        item.compatibility for item in result.dimension_assessments if item.dimension is dimension
    ]
    priority = {
        DimensionCompatibility.NOT_APPLICABLE: 0,
        DimensionCompatibility.COMPATIBLE: 1,
        DimensionCompatibility.UNKNOWN: 2,
        DimensionCompatibility.INCOMPATIBLE: 3,
    }
    return max(values, key=priority.__getitem__)


def test_resolved_target_and_compatible_truth_conditions_are_comparable() -> None:
    left, right = _resolved_pair()
    result = _evaluate(left, right)
    assert result.comparability is ComparabilityStatus.COMPARABLE
    assert result.relation is PropositionRelation.UNKNOWN
    assert result.relation_eligible is None


def test_different_resolved_subject_is_not_comparable() -> None:
    left, right = _resolved_pair()
    right = right.model_copy(update={"subject_identity": _identity("entity-2")})
    assert _evaluate(left, right).comparability is ComparabilityStatus.NOT_COMPARABLE


def test_unresolved_subject_is_unknown_even_when_raw_strings_match() -> None:
    left, right = _resolved_pair()
    left = left.model_copy(update={"subject_identity": SemanticIdentity.unresolved()})
    right = right.model_copy(update={"subject_identity": SemanticIdentity.unresolved()})
    result = _evaluate(left, right)
    assert _dimension(result, ComparisonDimension.SUBJECT) is DimensionCompatibility.UNKNOWN
    assert result.comparability is ComparabilityStatus.UNKNOWN


def test_different_raw_subjects_do_not_override_same_resolved_identity() -> None:
    left, right = _resolved_pair()
    right = right.model_copy(update={"subject": "another display surface"})
    result = _evaluate(left, right)
    assert _dimension(result, ComparisonDimension.SUBJECT) is DimensionCompatibility.COMPATIBLE


def test_different_resolved_predicate_is_not_comparable() -> None:
    left, right = _resolved_pair()
    right = right.model_copy(
        update={"predicate_identity": _identity("predicate-2", namespace="test.predicate")}
    )
    assert _evaluate(left, right).comparability is ComparabilityStatus.NOT_COMPARABLE


def test_unresolved_predicate_is_unknown_even_when_raw_strings_match() -> None:
    left, right = _resolved_pair()
    left = left.model_copy(update={"predicate_identity": SemanticIdentity.unresolved()})
    right = right.model_copy(update={"predicate_identity": SemanticIdentity.unresolved()})
    assert _evaluate(left, right).comparability is ComparabilityStatus.UNKNOWN


def test_different_raw_predicates_do_not_override_same_resolved_identity() -> None:
    left, right = _resolved_pair()
    right = right.model_copy(update={"predicate": "another surface form"})
    result = _evaluate(left, right)
    assert _dimension(result, ComparisonDimension.PREDICATE) is DimensionCompatibility.COMPATIBLE


def test_different_numeric_values_remain_comparable() -> None:
    left, right = _resolved_pair(
        value=PropositionValue(
            kind=PropositionValueKind.QUANTITY,
            numeric=NumericValue(operator=NumericOperator.EXACT, value=Decimal("100"), unit="kg"),
        )
    )
    right = right.model_copy(
        update={
            "typed_value": PropositionValue(
                kind=PropositionValueKind.QUANTITY,
                numeric=NumericValue(
                    operator=NumericOperator.EXACT, value=Decimal("120"), unit="kg"
                ),
            )
        }
    )
    assert _evaluate(left, right).comparability is ComparabilityStatus.COMPARABLE


def test_opposite_polarity_remains_comparable() -> None:
    left, right = _resolved_pair()
    right = right.model_copy(update={"polarity": PropositionPolarity.NEGATED})
    result = _evaluate(left, right)
    assert result.comparability is ComparabilityStatus.COMPARABLE
    assert _dimension(result, ComparisonDimension.POLARITY) is DimensionCompatibility.COMPATIBLE
    assert ComparisonReasonCode.POLARITY_MISMATCH in result.reason_codes


@pytest.mark.parametrize(
    ("kind", "left_value", "right_value", "expected"),
    [
        (ContextType.TEMPORAL, {"year": 2025}, {"year": 2026}, ComparabilityStatus.NOT_COMPARABLE),
        (
            ContextType.VERSION,
            {"subtype": "release", "value": "v1"},
            {"subtype": "release", "value": "v2"},
            ComparabilityStatus.NOT_COMPARABLE,
        ),
        (
            ContextType.SCOPE,
            {"subtype": "geography", "payload": "global"},
            {"subtype": "geography", "payload": "US-only"},
            ComparabilityStatus.NOT_COMPARABLE,
        ),
        (
            ContextType.SAMPLE_SCOPE,
            {"subtype": "population", "payload": "all users"},
            {"subtype": "population", "payload": "paid users"},
            ComparabilityStatus.NOT_COMPARABLE,
        ),
        (
            ContextType.CONDITION,
            {"subtype": "load", "payload": "high"},
            {"subtype": "load", "payload": "normal"},
            ComparabilityStatus.NOT_COMPARABLE,
        ),
        (
            ContextType.SCENARIO,
            {"subtype": "scenario", "payload": "A"},
            {"subtype": "scenario", "payload": "B"},
            ComparabilityStatus.NOT_COMPARABLE,
        ),
        (
            ContextType.CONFIGURATION,
            {"subtype": "config", "payload": "A"},
            {"subtype": "config", "payload": "B"},
            ComparabilityStatus.NOT_COMPARABLE,
        ),
    ],
)
def test_resolved_context_mismatches_are_not_comparable(
    kind, left_value, right_value, expected
) -> None:
    left = _assertion(
        subject_identity=_identity("entity-1"),
        predicate_identity=_identity("predicate-1", namespace="test.predicate"),
        contexts=(_context(kind, left_value),),
    )
    right = _assertion(
        subject_identity=_identity("entity-1"),
        predicate_identity=_identity("predicate-1", namespace="test.predicate"),
        contexts=(_context(kind, right_value),),
        ref="right",
    )
    assert _evaluate(left, right).comparability is expected


@pytest.mark.parametrize(
    "kind,value",
    [
        (ContextType.TEMPORAL, {"year": 2025}),
        (ContextType.SCOPE, {"subtype": "region", "payload": "global"}),
    ],
)
def test_missing_material_context_is_unknown(kind, value) -> None:
    left, right = _resolved_pair()
    left = left.model_copy(update={"material_contexts": (_context(kind, value),)})
    result = _evaluate(left, right)
    assert result.comparability is ComparabilityStatus.UNKNOWN


def test_missing_material_scope_is_unknown() -> None:
    left, right = _resolved_pair()
    left = left.model_copy(
        update={
            "material_contexts": (
                _context(ContextType.SCOPE, {"subtype": "region", "payload": "global"}),
            )
        }
    )
    result = _evaluate(left, right)
    assert result.comparability is ComparabilityStatus.UNKNOWN
    assert _dimension(result, ComparisonDimension.SCOPE) is DimensionCompatibility.UNKNOWN


def test_source_context_detector_incompleteness_forces_unknown() -> None:
    left, right = _resolved_pair()
    right = right.model_copy(update={"source_context_detection_incomplete": True})
    assert _evaluate(left, right).comparability is ComparabilityStatus.UNKNOWN


def _measurement(
    name: str = "latency",
    identity_type: MeasurementIdentityTypeV33 = MeasurementIdentityTypeV33.NAMED_METRIC,
    *,
    definition: str | None = None,
    observed: dict[str, object] | None = None,
    unit: str | None = "ms",
    **extra: object,
) -> MeasurementSemanticsV33:
    return MeasurementSemanticsV33(
        measurement_identity={
            "name": name,
            "identity_type": identity_type,
            "source_span": {"evidence_ref": "ev1", "occurrence_index": 0},
        },
        formal_definition=definition,
        observed_value=observed,
        unit=unit,
        **extra,
    )


def _measurement_assertion(semantics: MeasurementSemanticsV33, ref: str) -> VerificationAssertion:
    return _assertion(
        subject_identity=_identity("entity-1"),
        predicate_identity=_identity("predicate-1", namespace="test.predicate"),
        contexts=(
            _context(
                ContextType.MEASUREMENT,
                {"subtype": "metric", "payload": {"identity": "typed-v33"}},
                measurement=semantics,
                ref="measurement",
            ),
        ),
        ref=ref,
    )


def test_measurement_identity_mismatch_is_not_comparable() -> None:
    result = _evaluate(
        _measurement_assertion(_measurement("latency"), "a"),
        _measurement_assertion(_measurement("throughput"), "b"),
    )
    assert result.comparability is ComparabilityStatus.NOT_COMPARABLE


def test_measurement_identity_unknown_is_unknown() -> None:
    left = _measurement_assertion(_measurement("latency"), "a")
    unknown = MeasurementSemanticsV33(unit="ms")
    result = _evaluate(left, _measurement_assertion(unknown, "b"))
    assert result.comparability is ComparabilityStatus.UNKNOWN


def test_measurement_definition_vs_observation_is_not_comparable() -> None:
    left = _measurement_assertion(_measurement(definition="elapsed processing time"), "a")
    right = _measurement_assertion(
        _measurement(observed={"kind": "INTEGER", "raw_value": "12", "value": 12}), "b"
    )
    assert _evaluate(left, right).comparability is ComparabilityStatus.NOT_COMPARABLE


def test_measurement_numeric_values_are_not_equality_checked() -> None:
    left = _measurement_assertion(
        _measurement(observed={"kind": "DECIMAL", "raw_value": "38.40", "value": "38.40"}),
        "a",
    )
    right = _measurement_assertion(
        _measurement(observed={"kind": "DECIMAL", "raw_value": "38.4", "value": "38.4"}),
        "b",
    )
    assert _evaluate(left, right).comparability is ComparabilityStatus.COMPARABLE


def test_measurement_definition_or_qualifier_text_difference_fails_closed() -> None:
    left = _measurement_assertion(
        _measurement(definition="elapsed processing time", qualifiers=("under load A",)),
        "a",
    )
    right = _measurement_assertion(
        _measurement(definition="processing duration", qualifiers=("under load B",)),
        "b",
    )
    assert _evaluate(left, right).comparability is ComparabilityStatus.UNKNOWN


def test_percentage_38_40_and_38_4_are_comparable() -> None:
    left, right = _resolved_pair(
        value=PropositionValue(
            kind=PropositionValueKind.QUANTITY,
            numeric=NumericValue(operator=NumericOperator.EXACT, value=Decimal("38.40"), unit="%"),
        )
    )
    right = right.model_copy(
        update={
            "typed_value": PropositionValue(
                kind=PropositionValueKind.QUANTITY,
                numeric=NumericValue(
                    operator=NumericOperator.EXACT, value=Decimal("38.4"), unit="%"
                ),
            )
        }
    )
    assert _evaluate(left, right).comparability is ComparabilityStatus.COMPARABLE


def test_different_resolved_units_are_not_comparable() -> None:
    left, right = _resolved_pair(
        value=PropositionValue(
            kind=PropositionValueKind.QUANTITY,
            numeric=NumericValue(operator=NumericOperator.EXACT, value=Decimal("1"), unit="ms"),
        )
    )
    right = right.model_copy(
        update={
            "typed_value": PropositionValue(
                kind=PropositionValueKind.QUANTITY,
                numeric=NumericValue(operator=NumericOperator.EXACT, value=Decimal("1"), unit="s"),
            )
        }
    )
    assert _evaluate(left, right).comparability is ComparabilityStatus.NOT_COMPARABLE


def test_missing_material_unit_is_unknown() -> None:
    left, right = _resolved_pair(
        value=PropositionValue(
            kind=PropositionValueKind.QUANTITY,
            numeric=NumericValue(operator=NumericOperator.EXACT, value=Decimal("1"), unit="ms"),
        )
    )
    right = right.model_copy(
        update={
            "typed_value": PropositionValue(
                kind=PropositionValueKind.QUANTITY,
                numeric=NumericValue(operator=NumericOperator.EXACT, value=Decimal("1")),
            )
        }
    )
    assert _evaluate(left, right).comparability is ComparabilityStatus.UNKNOWN


def test_different_resolved_currency_is_not_comparable() -> None:
    left, right = _resolved_pair(
        value=PropositionValue(
            kind=PropositionValueKind.QUANTITY,
            numeric=NumericValue(
                operator=NumericOperator.EXACT, value=Decimal("10"), currency="USD"
            ),
        )
    )
    right = right.model_copy(
        update={
            "typed_value": PropositionValue(
                kind=PropositionValueKind.QUANTITY,
                numeric=NumericValue(
                    operator=NumericOperator.EXACT, value=Decimal("10"), currency="EUR"
                ),
            )
        }
    )
    assert _evaluate(left, right).comparability is ComparabilityStatus.NOT_COMPARABLE


@pytest.mark.parametrize(
    ("left_context", "right_context", "dimension"),
    [
        (
            (ContextType.SCOPE, {"subtype": "region", "payload": "all products"}),
            (ContextType.SCOPE, {"subtype": "region", "payload": "product A"}),
            ComparisonDimension.SCOPE,
        ),
        (
            (ContextType.SAMPLE_SCOPE, {"subtype": "population", "payload": "adults"}),
            (ContextType.SAMPLE_SCOPE, {"subtype": "population", "payload": "all users"}),
            ComparisonDimension.POPULATION,
        ),
        (
            (ContextType.CONDITION, {"subtype": "treatment", "payload": "with"}),
            (ContextType.CONDITION, {"subtype": "treatment", "payload": "without"}),
            ComparisonDimension.CONDITION,
        ),
        (
            (ContextType.SCENARIO, {"subtype": "scenario", "payload": "A"}),
            (ContextType.SCENARIO, {"subtype": "scenario", "payload": "B"}),
            ComparisonDimension.SCENARIO,
        ),
        (
            (ContextType.CONFIGURATION, {"subtype": "configuration", "payload": "A"}),
            (ContextType.CONFIGURATION, {"subtype": "configuration", "payload": "B"}),
            ComparisonDimension.CONFIGURATION,
        ),
        (
            (ContextType.METHOD, {"subtype": "method", "payload": "A"}),
            (ContextType.METHOD, {"subtype": "method", "payload": "B"}),
            ComparisonDimension.METHOD,
        ),
    ],
)
def test_context_mismatch_has_dimension_assessment(left_context, right_context, dimension) -> None:
    left, right = _resolved_pair()
    left = left.model_copy(update={"material_contexts": (_context(*left_context),)})
    right = right.model_copy(update={"material_contexts": (_context(*right_context),)})
    result = _evaluate(left, right)
    assert result.comparability is ComparabilityStatus.NOT_COMPARABLE
    assert _dimension(result, dimension) is DimensionCompatibility.INCOMPATIBLE


def test_material_context_sets_are_compared_without_picking_first() -> None:
    left, right = _resolved_pair()
    left_contexts = (
        _context(ContextType.SCOPE, {"subtype": "region", "payload": "A"}, ref="a"),
        _context(ContextType.SCOPE, {"subtype": "region", "payload": "B"}, ref="b"),
    )
    right_contexts = tuple(reversed(left_contexts))
    left = left.model_copy(update={"material_contexts": left_contexts})
    right = right.model_copy(update={"material_contexts": right_contexts})
    assert (
        _dimension(_evaluate(left, right), ComparisonDimension.SCOPE)
        is DimensionCompatibility.COMPATIBLE
    )


def test_material_context_multiplicity_is_preserved() -> None:
    left, right = _resolved_pair()
    one = _context(ContextType.SCOPE, {"subtype": "region", "payload": "A"}, ref="a")
    two = _context(ContextType.SCOPE, {"subtype": "region", "payload": "A"}, ref="b")
    left = left.model_copy(update={"material_contexts": (one, two)})
    right = right.model_copy(update={"material_contexts": (one,)})
    assert (
        _dimension(_evaluate(left, right), ComparisonDimension.SCOPE)
        is DimensionCompatibility.UNKNOWN
    )


def test_same_cardinality_context_value_replacement_is_incompatible() -> None:
    left, right = _resolved_pair()
    left = left.model_copy(
        update={
            "material_contexts": (
                _context(ContextType.SCOPE, {"subtype": "region", "payload": "A"}),
                _context(ContextType.SCOPE, {"subtype": "region", "payload": "B"}, ref="b"),
            )
        }
    )
    right = right.model_copy(
        update={
            "material_contexts": (
                _context(ContextType.SCOPE, {"subtype": "region", "payload": "A"}),
                _context(ContextType.SCOPE, {"subtype": "region", "payload": "C"}, ref="c"),
            )
        }
    )
    assert (
        _dimension(_evaluate(left, right), ComparisonDimension.SCOPE)
        is DimensionCompatibility.INCOMPATIBLE
    )


def test_ambiguous_context_attachment_is_unknown() -> None:
    left, right = _resolved_pair()
    context = _context(
        ContextType.TEMPORAL, {"year": 2025}, attachment=ContextAttachmentResolution.AMBIGUOUS
    )
    left = left.model_copy(update={"material_contexts": (context,)})
    right = right.model_copy(update={"material_contexts": (context,)})
    assert _evaluate(left, right).comparability is ComparabilityStatus.UNKNOWN


def test_unknown_materiality_is_unknown() -> None:
    left, right = _resolved_pair()
    context = _context(ContextType.TEMPORAL, {"year": 2025}, materiality="UNKNOWN")
    left = left.model_copy(update={"material_contexts": (context,)})
    right = right.model_copy(update={"material_contexts": (context,)})
    assert _evaluate(left, right).comparability is ComparabilityStatus.UNKNOWN


def test_material_qualifier_missing_on_one_side_is_unknown() -> None:
    left, right = _resolved_pair()
    qualifier = MaterialQualifier(
        category=QualifierCategory.SCOPE,
        key="region",
        value="global",
        materiality=QualifierMateriality.MATERIAL,
    )
    left = left.model_copy(update={"material_qualifiers": (qualifier,)})
    result = _evaluate(left, right)
    assert _dimension(result, ComparisonDimension.SCOPE) is DimensionCompatibility.UNKNOWN


def test_material_qualifier_difference_is_not_comparable() -> None:
    left, right = _resolved_pair()
    left = left.model_copy(
        update={
            "material_qualifiers": (
                MaterialQualifier(
                    category=QualifierCategory.BASELINE,
                    key="baseline",
                    value="A",
                    materiality=QualifierMateriality.MATERIAL,
                ),
            )
        }
    )
    right = right.model_copy(
        update={
            "material_qualifiers": (
                MaterialQualifier(
                    category=QualifierCategory.BASELINE,
                    key="baseline",
                    value="B",
                    materiality=QualifierMateriality.MATERIAL,
                ),
            )
        }
    )
    assert (
        _dimension(_evaluate(left, right), ComparisonDimension.BASELINE)
        is DimensionCompatibility.INCOMPATIBLE
    )


def test_different_material_attribution_is_not_comparable() -> None:
    left, right = _resolved_pair(attribution="Company A reports")
    right = right.model_copy(update={"attribution": "Regulator confirms"})
    result = _evaluate(left, right)
    assert result.comparability is ComparabilityStatus.NOT_COMPARABLE
    assert (
        _dimension(result, ComparisonDimension.ATTRIBUTION) is DimensionCompatibility.INCOMPATIBLE
    )


def test_different_material_modality_is_not_comparable() -> None:
    left, right = _resolved_pair(modality="will")
    right = right.model_copy(update={"modality": "may"})
    result = _evaluate(left, right)
    assert result.comparability is ComparabilityStatus.NOT_COMPARABLE
    assert _dimension(result, ComparisonDimension.MODALITY) is DimensionCompatibility.INCOMPATIBLE


def test_measurement_scope_target_baseline_and_unit_are_separate_dimensions() -> None:
    left = _measurement_assertion(
        _measurement(
            scope_category="population:A",
            target_object="system:A",
            basis_ref="basisA",
        ),
        "a",
    )
    right = _measurement_assertion(
        _measurement(
            scope_category="population:B",
            target_object="system:B",
            basis_ref="basisB",
        ),
        "b",
    )
    result = _evaluate(left, right)
    assert _dimension(result, ComparisonDimension.SCOPE) is DimensionCompatibility.INCOMPATIBLE
    assert (
        _dimension(result, ComparisonDimension.COMPARISON_TARGET)
        is DimensionCompatibility.INCOMPATIBLE
    )
    assert _dimension(result, ComparisonDimension.BASELINE) is DimensionCompatibility.INCOMPATIBLE


def test_measurement_namespace_or_role_difference_is_fail_closed() -> None:
    left = _measurement_assertion(
        _measurement("latency", MeasurementIdentityTypeV33.NAMED_METRIC), "a"
    )
    right = _measurement_assertion(
        _measurement("latency", MeasurementIdentityTypeV33.QUALITATIVE_PROPERTY), "b"
    )
    assert _evaluate(left, right).comparability is ComparabilityStatus.NOT_COMPARABLE


@pytest.mark.parametrize(
    "eligibility", [ComparisonEligibility.INELIGIBLE, ComparisonEligibility.UNKNOWN]
)
def test_composition_not_eligible_is_unknown(eligibility) -> None:
    left, right = _resolved_pair()
    right = right.model_copy(update={"composition_eligibility": eligibility})
    assert _evaluate(left, right).comparability is ComparabilityStatus.UNKNOWN


@pytest.mark.parametrize(
    "grounding", [AssertionGroundingUsability.INSUFFICIENT, AssertionGroundingUsability.UNKNOWN]
)
def test_grounding_not_sufficient_is_unknown(grounding) -> None:
    left, right = _resolved_pair()
    right = right.model_copy(update={"grounding_usability": grounding})
    assert _evaluate(left, right).comparability is ComparabilityStatus.UNKNOWN


def test_incompatible_dimension_takes_precedence_over_unknown() -> None:
    left, right = _resolved_pair()
    left = left.model_copy(
        update={"material_contexts": (_context(ContextType.TEMPORAL, {"year": 2025}),)}
    )
    right = right.model_copy(
        update={
            "subject_identity": SemanticIdentity.unresolved(),
            "material_contexts": (_context(ContextType.TEMPORAL, {"year": 2026}),),
        }
    )
    assert _evaluate(left, right).comparability is ComparabilityStatus.NOT_COMPARABLE


def test_namespace_or_normalization_version_mismatch_is_unknown() -> None:
    left, right = _resolved_pair()
    right = right.model_copy(
        update={"subject_identity": _identity("entity-1", namespace="other.namespace")}
    )
    assert (
        _dimension(_evaluate(left, right), ComparisonDimension.SUBJECT)
        is DimensionCompatibility.UNKNOWN
    )
    right = right.model_copy(update={"subject_identity": _identity("entity-1", version="id-v2")})
    assert (
        _dimension(_evaluate(left, right), ComparisonDimension.SUBJECT)
        is DimensionCompatibility.UNKNOWN
    )


def test_relation_is_never_classified_and_evaluation_is_deterministic() -> None:
    left, right = _resolved_pair()
    before_left = left.model_dump(mode="json")
    before_right = right.model_dump(mode="json")
    first = _evaluate(left, right)
    second = _evaluate(left, right)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.relation is PropositionRelation.UNKNOWN
    assert first.relation_eligible is None
    assert left.model_dump(mode="json") == before_left
    assert right.model_dump(mode="json") == before_right


def test_boolean_vs_numeric_typed_value_is_incompatible_without_flattening() -> None:
    left, right = _resolved_pair(
        value=PropositionValue(kind=PropositionValueKind.BOOLEAN, boolean=True)
    )
    right = right.model_copy(
        update={
            "typed_value": PropositionValue(
                kind=PropositionValueKind.QUANTITY,
                numeric=NumericValue(operator=NumericOperator.EXACT, value=Decimal("1")),
            )
        }
    )
    result = _evaluate(left, right)
    assert result.comparability is ComparabilityStatus.NOT_COMPARABLE
    assert _dimension(result, ComparisonDimension.VALUE) is DimensionCompatibility.INCOMPATIBLE


@pytest.mark.parametrize(
    ("left_value", "right_value", "expected"),
    [
        (
            PropositionValue(kind=PropositionValueKind.TEXT, text="alpha"),
            PropositionValue(kind=PropositionValueKind.TEXT, text="beta"),
            ComparabilityStatus.COMPARABLE,
        ),
        (
            PropositionValue(kind=PropositionValueKind.ENTITY, text="entity A"),
            PropositionValue(kind=PropositionValueKind.ENTITY, text="entity B"),
            ComparabilityStatus.COMPARABLE,
        ),
        (
            PropositionValue(kind=PropositionValueKind.SET, members=("a", "b")),
            PropositionValue(kind=PropositionValueKind.SET, members=("c", "d")),
            ComparabilityStatus.COMPARABLE,
        ),
        (
            PropositionValue(
                kind=PropositionValueKind.TEMPORAL_CHANGE,
                temporal_from="period A",
                temporal_to="period B",
                numeric=NumericValue(operator=NumericOperator.EXACT, value=Decimal("1")),
            ),
            PropositionValue(
                kind=PropositionValueKind.TEMPORAL_CHANGE,
                temporal_from="period A",
                temporal_to="period B",
                numeric=NumericValue(operator=NumericOperator.EXACT, value=Decimal("2")),
            ),
            ComparabilityStatus.COMPARABLE,
        ),
        (
            PropositionValue(
                kind=PropositionValueKind.TEMPORAL_CHANGE,
                temporal_from="A",
                temporal_to="B",
                numeric=NumericValue(operator=NumericOperator.EXACT, value=Decimal("1")),
            ),
            PropositionValue(
                kind=PropositionValueKind.TEMPORAL_CHANGE,
                temporal_from="A",
                temporal_to="C",
                numeric=NumericValue(operator=NumericOperator.EXACT, value=Decimal("1")),
            ),
            ComparabilityStatus.NOT_COMPARABLE,
        ),
        (
            PropositionValue(kind=PropositionValueKind.UNKNOWN),
            PropositionValue(kind=PropositionValueKind.UNKNOWN),
            ComparabilityStatus.UNKNOWN,
        ),
    ],
)
def test_supported_typed_value_families_are_not_flattened(
    left_value: PropositionValue,
    right_value: PropositionValue,
    expected: ComparabilityStatus,
) -> None:
    left, right = _resolved_pair(value=left_value)
    right = right.model_copy(update={"typed_value": right_value})
    assert _evaluate(left, right).comparability is expected
