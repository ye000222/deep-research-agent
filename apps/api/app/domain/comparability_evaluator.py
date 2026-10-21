"""Pure, deterministic V2.2-A2 comparability evaluation.

This module compares only already-projected VerificationAssertion data. It
never classifies proposition relations, resolves text identities, or performs
I/O. Missing semantic certainty is represented as UNKNOWN.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence

from app.domain.atomic_proposition import (
    NumericValue,
    PropositionPolarity,
    PropositionRelation,
    PropositionValue,
    PropositionValueKind,
)
from app.domain.source_context_extraction_v3_3 import (
    MeasurementIdentityTypeV33,
    MeasurementSemanticsV33,
)
from app.domain.truth_condition_context import (
    ContextType,
    TruthConditionContext,
    canonical_context_value,
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
    MaterialContextProjection,
    PropositionPairAssessment,
    SemanticIdentity,
    SemanticIdentityStatus,
    VerificationAssertion,
)

_CONTEXT_DIMENSIONS: dict[ContextType, ComparisonDimension] = {
    ContextType.TEMPORAL: ComparisonDimension.TIME,
    ContextType.VERSION: ComparisonDimension.VERSION,
    ContextType.SCOPE: ComparisonDimension.SCOPE,
    ContextType.SAMPLE_SCOPE: ComparisonDimension.POPULATION,
    ContextType.CONDITION: ComparisonDimension.CONDITION,
    ContextType.SCENARIO: ComparisonDimension.SCENARIO,
    ContextType.CONFIGURATION: ComparisonDimension.CONFIGURATION,
    ContextType.METHOD: ComparisonDimension.METHOD,
}

_QUALIFIER_DIMENSIONS = {
    "time": ComparisonDimension.TIME,
    "date": ComparisonDimension.TIME,
    "period": ComparisonDimension.TIME,
    "version": ComparisonDimension.VERSION,
    "location": ComparisonDimension.LOCATION,
    "scope": ComparisonDimension.SCOPE,
    "population": ComparisonDimension.POPULATION,
    "condition": ComparisonDimension.CONDITION,
    "scenario": ComparisonDimension.SCENARIO,
    "configuration": ComparisonDimension.CONFIGURATION,
    "measurement": ComparisonDimension.MEASUREMENT,
    "unit": ComparisonDimension.UNIT,
    "currency": ComparisonDimension.CURRENCY,
    "baseline": ComparisonDimension.BASELINE,
    "comparison_target": ComparisonDimension.COMPARISON_TARGET,
    "method": ComparisonDimension.METHOD,
    "modality": ComparisonDimension.MODALITY,
    "attribution": ComparisonDimension.ATTRIBUTION,
}

_NUMERIC_KINDS = {PropositionValueKind.QUANTITY, PropositionValueKind.RANGE}
_TEXT_KINDS = {PropositionValueKind.TEXT, PropositionValueKind.ENTITY}


def _assessment(
    dimension: ComparisonDimension,
    compatibility: DimensionCompatibility,
    *reasons: ComparisonReasonCode,
) -> DimensionAssessment:
    return DimensionAssessment(
        dimension=dimension,
        compatibility=compatibility,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def _identity_assessment(
    dimension: ComparisonDimension,
    left: SemanticIdentity,
    right: SemanticIdentity,
) -> DimensionAssessment:
    unknown_reason = (
        ComparisonReasonCode.SUBJECT_UNKNOWN
        if dimension is ComparisonDimension.SUBJECT
        else ComparisonReasonCode.PREDICATE_UNKNOWN
    )
    mismatch_reason = (
        ComparisonReasonCode.SUBJECT_MISMATCH
        if dimension is ComparisonDimension.SUBJECT
        else ComparisonReasonCode.PREDICATE_MISMATCH
    )
    if (
        left.status is not SemanticIdentityStatus.RESOLVED
        or right.status is not SemanticIdentityStatus.RESOLVED
    ):
        return _assessment(dimension, DimensionCompatibility.UNKNOWN, unknown_reason)
    if (
        left.namespace != right.namespace
        or left.schema_version != right.schema_version
        or left.normalization_version is None
        or right.normalization_version is None
        or left.normalization_version != right.normalization_version
    ):
        return _assessment(dimension, DimensionCompatibility.UNKNOWN, unknown_reason)
    if left.key == right.key:
        return _assessment(dimension, DimensionCompatibility.COMPATIBLE)
    return _assessment(dimension, DimensionCompatibility.INCOMPATIBLE, mismatch_reason)


def _typed_value_assessment(left: PropositionValue, right: PropositionValue) -> DimensionAssessment:
    if PropositionValueKind.UNKNOWN in {left.kind, right.kind}:
        return _assessment(
            ComparisonDimension.VALUE,
            DimensionCompatibility.UNKNOWN,
            ComparisonReasonCode.VALUE_UNKNOWN,
        )
    if left.kind == right.kind:
        return _assessment(ComparisonDimension.VALUE, DimensionCompatibility.COMPATIBLE)
    if left.kind in _NUMERIC_KINDS and right.kind in _NUMERIC_KINDS:
        return _assessment(ComparisonDimension.VALUE, DimensionCompatibility.COMPATIBLE)
    if left.kind in _TEXT_KINDS and right.kind in _TEXT_KINDS:
        return _assessment(ComparisonDimension.VALUE, DimensionCompatibility.COMPATIBLE)
    return _assessment(
        ComparisonDimension.VALUE,
        DimensionCompatibility.INCOMPATIBLE,
        ComparisonReasonCode.VALUE_TYPE_MISMATCH,
    )


def _optional_scalar_assessment(
    dimension: ComparisonDimension,
    left: object | None,
    right: object | None,
    mismatch_reason: ComparisonReasonCode,
) -> DimensionAssessment:
    if left is None and right is None:
        return _assessment(dimension, DimensionCompatibility.NOT_APPLICABLE)
    if left is None or right is None:
        return _assessment(dimension, DimensionCompatibility.UNKNOWN, mismatch_reason)
    if left == right:
        return _assessment(dimension, DimensionCompatibility.COMPATIBLE)
    return _assessment(dimension, DimensionCompatibility.INCOMPATIBLE, mismatch_reason)


def _numeric_assessments(
    left: NumericValue | None, right: NumericValue | None
) -> list[DimensionAssessment]:
    result: list[DimensionAssessment] = []
    for dimension, left_value, right_value, reason in (
        (
            ComparisonDimension.UNIT,
            left.unit if left else None,
            right.unit if right else None,
            ComparisonReasonCode.UNIT_MISMATCH,
        ),
        (
            ComparisonDimension.CURRENCY,
            left.currency if left else None,
            right.currency if right else None,
            ComparisonReasonCode.CURRENCY_MISMATCH,
        ),
    ):
        assessment = _optional_scalar_assessment(dimension, left_value, right_value, reason)
        # A unit/currency present on only one side establishes that the omitted
        # counterpart is unresolved rather than demonstrably incompatible.
        result.append(assessment)
    for name in ("measurement_basis", "extremum", "change_type"):
        left_value = getattr(left, name) if left else None
        right_value = getattr(right, name) if right else None
        if left_value is None and right_value is None:
            continue
        reason = ComparisonReasonCode.MEASUREMENT_MISMATCH
        result.append(
            _optional_scalar_assessment(
                ComparisonDimension.MEASUREMENT,
                left_value,
                right_value,
                reason,
            )
        )
    return result


def _comparison_value_assessments(
    left: PropositionValue, right: PropositionValue
) -> list[DimensionAssessment]:
    result: list[DimensionAssessment] = []
    if (
        left.kind is PropositionValueKind.COMPARISON
        and right.kind is PropositionValueKind.COMPARISON
    ):
        result.append(
            _optional_scalar_assessment(
                ComparisonDimension.COMPARISON_TARGET,
                left.comparison_target,
                right.comparison_target,
                ComparisonReasonCode.COMPARISON_TARGET_MISMATCH,
            )
        )
        if (
            left.comparison_metric_status is None
            or right.comparison_metric_status is None
            or left.comparison_metric_status.value == "unspecified"
            or right.comparison_metric_status.value == "unspecified"
        ):
            result.append(
                _assessment(
                    ComparisonDimension.MEASUREMENT,
                    DimensionCompatibility.UNKNOWN,
                    ComparisonReasonCode.MEASUREMENT_UNKNOWN,
                )
            )
        else:
            result.append(
                _optional_scalar_assessment(
                    ComparisonDimension.MEASUREMENT,
                    left.comparison_metric,
                    right.comparison_metric,
                    ComparisonReasonCode.MEASUREMENT_MISMATCH,
                )
            )
    if (
        left.kind is PropositionValueKind.TEMPORAL_CHANGE
        and right.kind is PropositionValueKind.TEMPORAL_CHANGE
    ):
        for left_value, right_value in (
            (left.temporal_from, right.temporal_from),
            (left.temporal_to, right.temporal_to),
        ):
            result.append(
                _optional_scalar_assessment(
                    ComparisonDimension.TIME,
                    left_value,
                    right_value,
                    ComparisonReasonCode.TIME_MISMATCH,
                )
            )
    return result


def _polarity_assessment(
    left: PropositionPolarity, right: PropositionPolarity
) -> DimensionAssessment:
    if PropositionPolarity.UNKNOWN in {left, right}:
        return _assessment(
            ComparisonDimension.POLARITY,
            DimensionCompatibility.UNKNOWN,
            ComparisonReasonCode.VALUE_UNKNOWN,
        )
    if left is right:
        return _assessment(ComparisonDimension.POLARITY, DimensionCompatibility.COMPATIBLE)
    # Polarity disagreement is precisely a possible future contradiction; it
    # does not prevent the two assertions from being comparable.
    return _assessment(
        ComparisonDimension.POLARITY,
        DimensionCompatibility.COMPATIBLE,
        ComparisonReasonCode.POLARITY_MISMATCH,
    )


def _surface_field_assessment(
    dimension: ComparisonDimension,
    left: str | None,
    right: str | None,
    reason: ComparisonReasonCode,
) -> DimensionAssessment:
    if left is None and right is None:
        return _assessment(dimension, DimensionCompatibility.NOT_APPLICABLE)
    if left is None or right is None:
        return _assessment(
            dimension, DimensionCompatibility.UNKNOWN, ComparisonReasonCode.VALUE_UNKNOWN
        )
    if left == right:
        return _assessment(dimension, DimensionCompatibility.COMPATIBLE)
    return _assessment(dimension, DimensionCompatibility.INCOMPATIBLE, reason)


def _stable_value(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _contains_unknown(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in {"", "unknown", "unresolved", "not_evaluated"}
    if isinstance(value, dict):
        return not value or any(_contains_unknown(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return not value or any(_contains_unknown(item) for item in value)
    return False


def _context_key(context: TruthConditionContext) -> object:
    return canonical_context_value(context)


def _compare_context_multisets(
    dimension: ComparisonDimension,
    left: list[MaterialContextProjection],
    right: list[MaterialContextProjection],
    reason: ComparisonReasonCode,
) -> DimensionAssessment:
    unknown_reason = _unknown_reason_for_dimension(dimension)
    if not left and not right:
        return _assessment(dimension, DimensionCompatibility.NOT_APPLICABLE)
    if not left or not right:
        return _assessment(dimension, DimensionCompatibility.UNKNOWN, unknown_reason)
    for item in (*left, *right):
        if item.attachment_resolution is ContextAttachmentResolution.AMBIGUOUS:
            return _assessment(
                dimension,
                DimensionCompatibility.UNKNOWN,
                ComparisonReasonCode.CONTEXT_ATTACHMENT_AMBIGUOUS,
            )
        if item.context.materiality == "UNKNOWN" or item.context.scope.kind.value == "UNKNOWN":
            return _assessment(dimension, DimensionCompatibility.UNKNOWN, unknown_reason)
    left_values = [_context_key(item.context) for item in left]
    right_values = [_context_key(item.context) for item in right]
    if any(_contains_unknown(value) for value in (*left_values, *right_values)):
        return _assessment(dimension, DimensionCompatibility.UNKNOWN, unknown_reason)
    left_counts = Counter(_stable_value(value) for value in left_values)
    right_counts = Counter(_stable_value(value) for value in right_values)
    if left_counts == right_counts:
        return _assessment(dimension, DimensionCompatibility.COMPATIBLE)
    if len(left_values) == len(right_values):
        return _assessment(dimension, DimensionCompatibility.INCOMPATIBLE, reason)
    return _assessment(dimension, DimensionCompatibility.UNKNOWN, unknown_reason)


def _measurement_assessments(
    left: list[MaterialContextProjection], right: list[MaterialContextProjection]
) -> list[DimensionAssessment]:
    dimension = ComparisonDimension.MEASUREMENT
    reason = ComparisonReasonCode.MEASUREMENT_UNKNOWN
    mismatch = ComparisonReasonCode.MEASUREMENT_MISMATCH
    if not left and not right:
        return [_assessment(dimension, DimensionCompatibility.NOT_APPLICABLE)]
    if not left or not right:
        return [_assessment(dimension, DimensionCompatibility.UNKNOWN, reason)]
    left_semantics = [item.measurement_semantics for item in left]
    right_semantics = [item.measurement_semantics for item in right]
    if any(item is None for item in (*left_semantics, *right_semantics)):
        return [_assessment(dimension, DimensionCompatibility.UNKNOWN, reason)]
    if any(
        item.context.materiality == "UNKNOWN"
        or item.attachment_resolution is ContextAttachmentResolution.AMBIGUOUS
        for item in (*left, *right)
    ):
        return [_assessment(dimension, DimensionCompatibility.UNKNOWN, reason)]

    def identity_signature(
        semantics: MeasurementSemanticsV33,
    ) -> tuple[MeasurementIdentityTypeV33, str] | None:
        identity = semantics.measurement_identity
        if (
            identity is None
            or identity.name is None
            or identity.identity_type is MeasurementIdentityTypeV33.UNKNOWN
        ):
            return None
        return identity.identity_type, identity.name

    left_signatures = [identity_signature(item) for item in left_semantics if item]
    right_signatures = [identity_signature(item) for item in right_semantics if item]
    if any(item is None for item in (*left_signatures, *right_signatures)):
        return [_assessment(dimension, DimensionCompatibility.UNKNOWN, reason)]
    if Counter(left_signatures) != Counter(right_signatures):
        return [_assessment(dimension, DimensionCompatibility.INCOMPATIBLE, mismatch)]

    # Measurement roles are determined by explicitly typed field presence, not
    # by inferring from values or names.
    left_roles = [
        (
            item.formal_definition is not None,
            item.observed_value is not None,
            item.threshold is not None,
        )
        for item in left_semantics
        if item
    ]
    right_roles = [
        (
            item.formal_definition is not None,
            item.observed_value is not None,
            item.threshold is not None,
        )
        for item in right_semantics
        if item
    ]
    if Counter(left_roles) != Counter(right_roles):
        return [_assessment(dimension, DimensionCompatibility.INCOMPATIBLE, mismatch)]
    result = [_assessment(dimension, DimensionCompatibility.COMPATIBLE)]
    left_definitions = [item.formal_definition for item in left_semantics if item]
    right_definitions = [item.formal_definition for item in right_semantics if item]
    if Counter(_stable_value(item) for item in left_definitions) != Counter(
        _stable_value(item) for item in right_definitions
    ):
        result.append(
            _assessment(
                dimension,
                DimensionCompatibility.UNKNOWN,
                ComparisonReasonCode.MEASUREMENT_UNKNOWN,
            )
        )
    left_qualifiers = [tuple(sorted(item.qualifiers)) for item in left_semantics if item]
    right_qualifiers = [tuple(sorted(item.qualifiers)) for item in right_semantics if item]
    if Counter(_stable_value(item) for item in left_qualifiers) != Counter(
        _stable_value(item) for item in right_qualifiers
    ):
        result.append(
            _assessment(
                dimension,
                DimensionCompatibility.UNKNOWN,
                ComparisonReasonCode.MEASUREMENT_UNKNOWN,
            )
        )
    return result


def _context_assessments(
    left: VerificationAssertion, right: VerificationAssertion
) -> tuple[list[DimensionAssessment], bool]:
    left_by_type: dict[ContextType, list[MaterialContextProjection]] = {}
    right_by_type: dict[ContextType, list[MaterialContextProjection]] = {}
    unknown_context = False
    for assertion, destination in ((left, left_by_type), (right, right_by_type)):
        for item in assertion.material_contexts:
            context_type = item.context.context_type
            destination.setdefault(context_type, []).append(item)
            if context_type in {ContextType.UNKNOWN, ContextType.OTHER}:
                unknown_context = True

    assessments: list[DimensionAssessment] = []
    for context_type, dimension in _CONTEXT_DIMENSIONS.items():
        assessments.append(
            _compare_context_multisets(
                dimension,
                left_by_type.get(context_type, []),
                right_by_type.get(context_type, []),
                _reason_for_dimension(dimension),
            )
        )

    epistemic_left = left_by_type.get(ContextType.EPISTEMIC, [])
    epistemic_right = right_by_type.get(ContextType.EPISTEMIC, [])
    for dimension, field_name, reason in (
        (
            ComparisonDimension.ATTRIBUTION,
            "agent",
            ComparisonReasonCode.ATTRIBUTION_MISMATCH,
        ),
        (
            ComparisonDimension.MODALITY,
            "modality",
            ComparisonReasonCode.MODALITY_MISMATCH,
        ),
    ):
        assessments.append(
            _compare_epistemic_field(dimension, field_name, epistemic_left, epistemic_right, reason)
        )

    measurement_left = left_by_type.get(ContextType.MEASUREMENT, [])
    measurement_right = right_by_type.get(ContextType.MEASUREMENT, [])
    assessments.extend(_measurement_assessments(measurement_left, measurement_right))
    if measurement_left or measurement_right:
        assessments.extend(
            _measurement_subdimension_assessments(measurement_left, measurement_right)
        )
    return assessments, unknown_context


def _compare_epistemic_field(
    dimension: ComparisonDimension,
    field_name: str,
    left: list[MaterialContextProjection],
    right: list[MaterialContextProjection],
    reason: ComparisonReasonCode,
) -> DimensionAssessment:
    if not left and not right:
        return _assessment(dimension, DimensionCompatibility.NOT_APPLICABLE)
    if not left or not right:
        return _assessment(
            dimension, DimensionCompatibility.UNKNOWN, ComparisonReasonCode.VALUE_UNKNOWN
        )

    def values(items: Iterable[MaterialContextProjection]) -> list[object]:
        found: list[object] = []
        for item in items:
            if (
                item.context.materiality == "UNKNOWN"
                or item.attachment_resolution is ContextAttachmentResolution.AMBIGUOUS
            ):
                return [None]
            canonical = canonical_context_value(item.context)
            value = canonical.get(field_name)
            if value is not None:
                found.append(value)
        return found

    left_values = values(left)
    right_values = values(right)
    if (
        not left_values
        or not right_values
        or any(_contains_unknown(value) for value in (*left_values, *right_values))
    ):
        return _assessment(
            dimension, DimensionCompatibility.UNKNOWN, ComparisonReasonCode.VALUE_UNKNOWN
        )
    left_counts = Counter(_stable_value(value) for value in left_values)
    right_counts = Counter(_stable_value(value) for value in right_values)
    if left_counts == right_counts:
        return _assessment(dimension, DimensionCompatibility.COMPATIBLE)
    if len(left_values) == len(right_values):
        return _assessment(dimension, DimensionCompatibility.INCOMPATIBLE, reason)
    return _assessment(dimension, DimensionCompatibility.UNKNOWN, reason)


def _measurement_subdimension_assessments(
    left: list[MaterialContextProjection], right: list[MaterialContextProjection]
) -> list[DimensionAssessment]:
    result: list[DimensionAssessment] = []
    for dimension, field_name, reason in (
        (
            ComparisonDimension.UNIT,
            "unit",
            ComparisonReasonCode.UNIT_MISMATCH,
        ),
        (
            ComparisonDimension.SCOPE,
            "scope_category",
            ComparisonReasonCode.SCOPE_MISMATCH,
        ),
        (
            ComparisonDimension.COMPARISON_TARGET,
            "target_object",
            ComparisonReasonCode.COMPARISON_TARGET_MISMATCH,
        ),
        (
            ComparisonDimension.BASELINE,
            "basis_ref",
            ComparisonReasonCode.BASELINE_MISMATCH,
        ),
    ):
        left_values = [
            getattr(item.measurement_semantics, field_name)
            for item in left
            if item.measurement_semantics is not None
        ]
        right_values = [
            getattr(item.measurement_semantics, field_name)
            for item in right
            if item.measurement_semantics is not None
        ]
        result.append(_compare_optional_multisets(dimension, left_values, right_values, reason))
    return result


def _compare_optional_multisets(
    dimension: ComparisonDimension,
    left: Sequence[object | None],
    right: Sequence[object | None],
    reason: ComparisonReasonCode,
) -> DimensionAssessment:
    if not left and not right:
        return _assessment(dimension, DimensionCompatibility.NOT_APPLICABLE)
    if not left or not right:
        return _assessment(dimension, DimensionCompatibility.UNKNOWN, reason)
    if all(value is None for value in left) and all(value is None for value in right):
        return _assessment(dimension, DimensionCompatibility.NOT_APPLICABLE)
    if any(value is None for value in (*left, *right)):
        return _assessment(dimension, DimensionCompatibility.UNKNOWN, reason)
    left_counts = Counter(_stable_value(value) for value in left)
    right_counts = Counter(_stable_value(value) for value in right)
    if left_counts == right_counts:
        return _assessment(dimension, DimensionCompatibility.COMPATIBLE)
    if len(left) == len(right):
        return _assessment(dimension, DimensionCompatibility.INCOMPATIBLE, reason)
    return _assessment(dimension, DimensionCompatibility.UNKNOWN, reason)


def _reason_for_dimension(dimension: ComparisonDimension) -> ComparisonReasonCode:
    return {
        ComparisonDimension.TIME: ComparisonReasonCode.TIME_MISMATCH,
        ComparisonDimension.VERSION: ComparisonReasonCode.VERSION_MISMATCH,
        ComparisonDimension.SCOPE: ComparisonReasonCode.SCOPE_MISMATCH,
        ComparisonDimension.POPULATION: ComparisonReasonCode.POPULATION_MISMATCH,
        ComparisonDimension.LOCATION: ComparisonReasonCode.LOCATION_MISMATCH,
        ComparisonDimension.CONDITION: ComparisonReasonCode.CONDITION_MISMATCH,
        ComparisonDimension.SCENARIO: ComparisonReasonCode.SCENARIO_MISMATCH,
        ComparisonDimension.CONFIGURATION: ComparisonReasonCode.CONFIGURATION_MISMATCH,
        ComparisonDimension.METHOD: ComparisonReasonCode.METHOD_MISMATCH,
        ComparisonDimension.MEASUREMENT: ComparisonReasonCode.MEASUREMENT_MISMATCH,
    }.get(dimension, ComparisonReasonCode.VALUE_UNKNOWN)


def _qualifier_assessments(
    left: VerificationAssertion, right: VerificationAssertion
) -> list[DimensionAssessment]:
    result: list[DimensionAssessment] = []
    for category, dimension in _QUALIFIER_DIMENSIONS.items():
        left_items = [item for item in left.material_qualifiers if item.category.value == category]
        right_items = [
            item for item in right.material_qualifiers if item.category.value == category
        ]
        left_material = [item for item in left_items if item.materiality.value == "material"]
        right_material = [item for item in right_items if item.materiality.value == "material"]
        has_unknown = any(
            item.materiality.value == "unknown" for item in (*left_items, *right_items)
        )
        reason = _reason_for_dimension(dimension)
        if has_unknown:
            result.append(_assessment(dimension, DimensionCompatibility.UNKNOWN, reason))
            continue
        if not left_material and not right_material:
            continue
        if not left_material or not right_material:
            result.append(_assessment(dimension, DimensionCompatibility.UNKNOWN, reason))
            continue
        left_values = [(item.key, type(item.value).__name__, item.value) for item in left_material]
        right_values = [
            (item.key, type(item.value).__name__, item.value) for item in right_material
        ]
        result.append(_compare_optional_multisets(dimension, left_values, right_values, reason))
    return result


def _reason_for_materiality_unknown() -> ComparisonReasonCode:
    # A1 has no generic material-context-unknown code; VALUE_UNKNOWN is the
    # closest existing fail-closed diagnostic and does not claim a mismatch.
    return ComparisonReasonCode.VALUE_UNKNOWN


def _unknown_reason_for_dimension(
    dimension: ComparisonDimension,
) -> ComparisonReasonCode:
    return {
        ComparisonDimension.TIME: ComparisonReasonCode.TIME_UNKNOWN,
        ComparisonDimension.MEASUREMENT: ComparisonReasonCode.MEASUREMENT_UNKNOWN,
    }.get(dimension, ComparisonReasonCode.VALUE_UNKNOWN)


class ComparabilityEvaluator:
    """Evaluate only whether two projected assertions can be related later."""

    @staticmethod
    def evaluate(
        left: VerificationAssertion, right: VerificationAssertion
    ) -> PropositionPairAssessment:
        dimensions: list[DimensionAssessment] = [
            _identity_assessment(
                ComparisonDimension.SUBJECT, left.subject_identity, right.subject_identity
            ),
            _identity_assessment(
                ComparisonDimension.PREDICATE,
                left.predicate_identity,
                right.predicate_identity,
            ),
            _typed_value_assessment(left.typed_value, right.typed_value),
            _polarity_assessment(left.polarity, right.polarity),
        ]
        dimensions.extend(_numeric_assessments(left.typed_value.numeric, right.typed_value.numeric))
        dimensions.extend(_comparison_value_assessments(left.typed_value, right.typed_value))
        dimensions.extend(
            (
                _surface_field_assessment(
                    ComparisonDimension.ATTRIBUTION,
                    left.attribution,
                    right.attribution,
                    ComparisonReasonCode.ATTRIBUTION_MISMATCH,
                ),
                _surface_field_assessment(
                    ComparisonDimension.MODALITY,
                    left.modality,
                    right.modality,
                    ComparisonReasonCode.MODALITY_MISMATCH,
                ),
            )
        )

        context_assessments, has_unclassifiable_context = _context_assessments(left, right)
        dimensions.extend(context_assessments)
        dimensions.extend(_qualifier_assessments(left, right))
        dimensions.append(_composition_assessment(left, right))
        dimensions.append(_grounding_assessment(left, right))

        global_reasons: list[ComparisonReasonCode] = []
        if has_unclassifiable_context:
            global_reasons.append(_reason_for_materiality_unknown())
        if left.source_context_detection_incomplete or right.source_context_detection_incomplete:
            global_reasons.append(_reason_for_materiality_unknown())

        comparability = _aggregate(dimensions, force_unknown=bool(global_reasons))
        reasons = list(
            dict.fromkeys(
                [reason for item in dimensions for reason in item.reason_codes]
                + global_reasons
                + [ComparisonReasonCode.RELATION_NOT_EVALUATED]
            )
        )
        return PropositionPairAssessment(
            left_assertion_ref=left.assertion_ref,
            right_assertion_ref=right.assertion_ref,
            comparability=comparability,
            relation=PropositionRelation.UNKNOWN,
            relation_eligible=None,
            dimension_assessments=tuple(dimensions),
            reason_codes=tuple(reasons),
        )


def _composition_assessment(
    left: VerificationAssertion, right: VerificationAssertion
) -> DimensionAssessment:
    if (
        left.composition_eligibility is ComparisonEligibility.ELIGIBLE
        and right.composition_eligibility is ComparisonEligibility.ELIGIBLE
    ):
        return _assessment(ComparisonDimension.COMPOSITION, DimensionCompatibility.COMPATIBLE)
    if ComparisonEligibility.INELIGIBLE in {
        left.composition_eligibility,
        right.composition_eligibility,
    }:
        return _assessment(
            ComparisonDimension.COMPOSITION,
            DimensionCompatibility.UNKNOWN,
            ComparisonReasonCode.COMPOSITION_INELIGIBLE,
        )
    return _assessment(
        ComparisonDimension.COMPOSITION,
        DimensionCompatibility.UNKNOWN,
        ComparisonReasonCode.COMPOSITION_UNKNOWN,
    )


def _grounding_assessment(
    left: VerificationAssertion, right: VerificationAssertion
) -> DimensionAssessment:
    if (
        left.grounding_usability is AssertionGroundingUsability.SUFFICIENT
        and right.grounding_usability is AssertionGroundingUsability.SUFFICIENT
    ):
        return _assessment(ComparisonDimension.GROUNDING, DimensionCompatibility.COMPATIBLE)
    if AssertionGroundingUsability.INSUFFICIENT in {
        left.grounding_usability,
        right.grounding_usability,
    }:
        return _assessment(
            ComparisonDimension.GROUNDING,
            DimensionCompatibility.UNKNOWN,
            ComparisonReasonCode.INSUFFICIENT_GROUNDING,
        )
    return _assessment(
        ComparisonDimension.GROUNDING,
        DimensionCompatibility.UNKNOWN,
        ComparisonReasonCode.VALUE_UNKNOWN,
    )


def _aggregate(
    dimensions: Iterable[DimensionAssessment], *, force_unknown: bool = False
) -> ComparabilityStatus:
    items = tuple(dimensions)
    if any(item.compatibility is DimensionCompatibility.INCOMPATIBLE for item in items):
        return ComparabilityStatus.NOT_COMPARABLE
    if force_unknown or any(item.compatibility is DimensionCompatibility.UNKNOWN for item in items):
        return ComparabilityStatus.UNKNOWN
    return ComparabilityStatus.COMPARABLE
