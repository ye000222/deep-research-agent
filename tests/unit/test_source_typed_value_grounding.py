from __future__ import annotations

from decimal import Decimal

import pytest
from app.domain.atomic_proposition import (
    ComparisonMetricStatus,
    NumericOperator,
    NumericValue,
    PropositionValue,
    PropositionValueKind,
)
from app.domain.source_context_extraction_v3_4 import (
    SOURCE_CONTEXT_EXTRACTION_SCHEMA_V34,
    SourceExtractionResponseV34,
)
from app.domain.source_typed_value_grounding import (
    SOURCE_TYPED_VALUE_GROUNDING_CONTRACT_VERSION,
    SOURCE_TYPED_VALUE_PARSER_VERSION,
    SourceEvidenceSpanInput,
    SourceGroundingState,
    SourceTypedValueGroundingAssessment,
    TypedValueComponentRole,
    TypedValueEvidenceBinding,
    assess_typed_value_grounding,
    parse_numeric_span,
)

SOURCE = "source:synthetic"
SNAPSHOT = "a" * 64


def _value(
    operator: NumericOperator = NumericOperator.EXACT,
    *,
    value: str | None = "12.4",
    lower: str | None = None,
    upper: str | None = None,
    unit: str | None = "%",
) -> PropositionValue:
    return PropositionValue(
        kind=(
            PropositionValueKind.RANGE
            if operator is NumericOperator.RANGE
            else PropositionValueKind.QUANTITY
        ),
        numeric=NumericValue(
            operator=operator,
            value=Decimal(value) if value is not None else None,
            lower_bound=Decimal(lower) if lower is not None else None,
            upper_bound=Decimal(upper) if upper is not None else None,
            unit=unit,
        ),
    )


def _binding(
    role: TypedValueComponentRole,
    ref: str,
    *,
    assertion: str = "A1",
    text: str = "12.4%",
    start: int = 20,
    end: int | None = None,
    source: str = SOURCE,
    snapshot: str = SNAPSHOT,
) -> TypedValueEvidenceBinding:
    return TypedValueEvidenceBinding(
        binding_ref=f"b-{role.value.lower()}",
        assertion_ref=assertion,
        evidence_ref=ref,
        occurrence_index=0,
        char_start=start,
        char_end=end if end is not None else start + len(text),
        component_role=role,
        source_ref=source,
        snapshot_ref=snapshot,
    )


def _assessment(
    value: PropositionValue,
    span_text: str,
    roles: tuple[TypedValueComponentRole, ...],
    *,
    span_ref: str = "e1",
    assertion_ref: str = "A1",
    binding_assertion_ref: str | None = None,
    source_ref: str = SOURCE,
    binding_source_ref: str = SOURCE,
    snapshot_ref: str = SNAPSHOT,
    binding_snapshot_ref: str = SNAPSHOT,
    start: int = 20,
    binding_start: int | None = None,
):
    end = start + len(span_text)
    return assess_typed_value_grounding(
        assertion_ref=assertion_ref,
        value=value,
        bindings=tuple(
            _binding(
                role,
                span_ref,
                assertion=binding_assertion_ref or assertion_ref,
                text=span_text,
                start=binding_start if binding_start is not None else start,
                end=(binding_start if binding_start is not None else start) + len(span_text),
                source=binding_source_ref,
                snapshot=binding_snapshot_ref,
            )
            for role in roles
        ),
        spans=(
            SourceEvidenceSpanInput(
                evidence_ref=span_ref,
                exact_text=span_text,
                occurrence_index=0,
                char_start=start,
                char_end=end,
            ),
        ),
        source_ref=source_ref,
        snapshot_ref=snapshot_ref,
    )


def test_v34_schema_is_new_version_and_preserves_v33_shape() -> None:
    assert SOURCE_CONTEXT_EXTRACTION_SCHEMA_V34 == "source-context-extraction-schema-v3.4"
    schema = SourceExtractionResponseV34.model_json_schema()
    assert schema["properties"]["typed_value_bindings"]
    assert "typed_value_bindings" in schema["properties"]
    assert schema["properties"]["schema_version"]["const"] == (SOURCE_CONTEXT_EXTRACTION_SCHEMA_V34)


def test_assessment_is_versioned_and_json_round_trip_is_stable() -> None:
    result = _assessment(
        _value(),
        "12.4%",
        (TypedValueComponentRole.NUMERIC_MAGNITUDE, TypedValueComponentRole.UNIT),
    )
    from app.services.semantic_research_shadow import _restore_persisted_decimal_values

    artifact_json = result.model_dump(mode="json", exclude_unset=True)
    restored = SourceTypedValueGroundingAssessment.model_validate(
        _restore_persisted_decimal_values(artifact_json)
    )
    assert result.contract_version == SOURCE_TYPED_VALUE_GROUNDING_CONTRACT_VERSION
    assert result.parser_version == SOURCE_TYPED_VALUE_PARSER_VERSION
    assert restored == result


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12", NumericValue(operator=NumericOperator.EXACT, value=Decimal("12"))),
        ("12.4", NumericValue(operator=NumericOperator.EXACT, value=Decimal("12.4"))),
        ("1,234", NumericValue(operator=NumericOperator.EXACT, value=Decimal("1234"))),
        ("12.4%", NumericValue(operator=NumericOperator.EXACT, value=Decimal("12.4"), unit="%")),
        (">= 10", NumericValue(operator=NumericOperator.GTE, value=Decimal("10"))),
        (
            "10\u201320",
            NumericValue(
                operator=NumericOperator.RANGE, lower_bound=Decimal("10"), upper_bound=Decimal("20")
            ),
        ),
        (
            "10\u201320 kg",
            NumericValue(
                operator=NumericOperator.RANGE,
                lower_bound=Decimal("10"),
                upper_bound=Decimal("20"),
                unit="kg",
            ),
        ),
        ("12.4 kg", NumericValue(operator=NumericOperator.EXACT, value=Decimal("12.4"), unit="kg")),
    ],
)
def test_numeric_parser_accepts_only_frozen_full_span_grammar(
    text: str, expected: NumericValue
) -> None:
    assert parse_numeric_span(text) == expected


@pytest.mark.parametrize(
    "text",
    ["1.0k", "12,4", "12.4 percent", "about 12.4%", "1e3", "10-20", "12.4 USD"],
)
def test_numeric_parser_fails_closed_on_unsupported_or_ambiguous_surface(text: str) -> None:
    assert parse_numeric_span(text) is None


def test_explicit_percentage_requires_magnitude_and_unit_basis() -> None:
    roles = (TypedValueComponentRole.NUMERIC_MAGNITUDE, TypedValueComponentRole.UNIT)
    result = _assessment(_value(), "12.4%", roles)
    assert result.status is SourceGroundingState.VALID
    assert result.parser_version == SOURCE_TYPED_VALUE_PARSER_VERSION
    assert result.normalized_value == _value()
    assert {item.component_role for item in result.components} == set(roles)


def test_percentage_without_percent_surface_is_not_grounded() -> None:
    roles = (TypedValueComponentRole.NUMERIC_MAGNITUDE, TypedValueComponentRole.UNIT)
    result = _assessment(_value(), "12.4", roles)
    assert result.status is SourceGroundingState.INVALID
    assert result.normalized_value is None


def test_bound_requires_operator_and_magnitude() -> None:
    value = _value(NumericOperator.GTE, unit=None)
    roles = (TypedValueComponentRole.NUMERIC_MAGNITUDE, TypedValueComponentRole.NUMERIC_OPERATOR)
    assert _assessment(value, ">= 12.4", roles).status is SourceGroundingState.VALID
    assert (
        _assessment(value, ">= 12.4", (TypedValueComponentRole.NUMERIC_MAGNITUDE,)).status
        is SourceGroundingState.UNKNOWN
    )
    assert _assessment(value, "12.4", roles).status is SourceGroundingState.INVALID


def test_dimensions_are_reported_independently_on_typed_value_mismatch() -> None:
    roles = (TypedValueComponentRole.NUMERIC_MAGNITUDE, TypedValueComponentRole.UNIT)
    result = _assessment(_value(), "12.5%", roles)
    assert result.status is SourceGroundingState.INVALID
    assert all(item.textual_grounding is SourceGroundingState.VALID for item in result.components)
    semantic_by_role = {item.component_role: item.semantic_support for item in result.components}
    assert (
        semantic_by_role[TypedValueComponentRole.NUMERIC_MAGNITUDE] is SourceGroundingState.INVALID
    )
    assert semantic_by_role[TypedValueComponentRole.UNIT] is SourceGroundingState.VALID
    assert all(
        item.structural_grounding is SourceGroundingState.VALID for item in result.components
    )


def test_range_requires_both_bounds_and_full_operator_semantics() -> None:
    value = _value(NumericOperator.RANGE, value=None, lower="10", upper="20", unit=None)
    roles = (TypedValueComponentRole.LOWER_BOUND, TypedValueComponentRole.UPPER_BOUND)
    assert _assessment(value, "10\u201320", roles).status is SourceGroundingState.VALID
    assert _assessment(value, "10\u201319", roles).status is SourceGroundingState.INVALID


def test_explicit_occurrence_ref_selects_the_second_identical_number() -> None:
    exact = "12.4% then 12.4%"
    start = exact.rindex("12.4%")
    roles = (TypedValueComponentRole.NUMERIC_MAGNITUDE, TypedValueComponentRole.UNIT)
    bindings = tuple(
        _binding(role, "e-second", text="12.4%", start=start).model_copy(
            update={"occurrence_index": 1}
        )
        for role in roles
    )
    result = assess_typed_value_grounding(
        assertion_ref="A1",
        value=_value(),
        bindings=bindings,
        spans=(
            SourceEvidenceSpanInput(
                evidence_ref="e-second",
                exact_text="12.4%",
                occurrence_index=1,
                char_start=start,
                char_end=start + len("12.4%"),
            ),
        ),
        source_ref=SOURCE,
        snapshot_ref=SNAPSHOT,
    )
    assert result.status is SourceGroundingState.VALID


def test_value_binding_does_not_search_other_supplied_or_unbound_spans() -> None:
    roles = (TypedValueComponentRole.NUMERIC_MAGNITUDE, TypedValueComponentRole.UNIT)
    bindings = tuple(_binding(role, "e-wrong", text="12.5%") for role in roles)
    result = assess_typed_value_grounding(
        assertion_ref="A1",
        value=_value(),
        bindings=bindings,
        spans=(
            SourceEvidenceSpanInput(
                evidence_ref="e-wrong",
                exact_text="12.5%",
                occurrence_index=0,
                char_start=20,
                char_end=25,
            ),
            SourceEvidenceSpanInput(
                evidence_ref="e-unbound",
                exact_text="12.4%",
                occurrence_index=0,
                char_start=40,
                char_end=45,
            ),
        ),
        source_ref=SOURCE,
        snapshot_ref=SNAPSHOT,
    )
    assert result.status is SourceGroundingState.INVALID


def test_wrong_value_unit_source_snapshot_occurrence_or_assertion_fails_closed() -> None:
    roles = (TypedValueComponentRole.NUMERIC_MAGNITUDE, TypedValueComponentRole.UNIT)
    expected = _value()
    assert _assessment(expected, "12.5%", roles).status is SourceGroundingState.INVALID
    assert _assessment(expected, "12.4 m", roles).status is SourceGroundingState.INVALID
    assert (
        _assessment(expected, "12.4%", roles, binding_assertion_ref="A2").status
        is SourceGroundingState.INVALID
    )
    assert (
        _assessment(expected, "12.4%", roles, binding_source_ref="source:other").status
        is SourceGroundingState.INVALID
    )
    assert (
        _assessment(expected, "12.4%", roles, binding_snapshot_ref="b" * 64).status
        is SourceGroundingState.INVALID
    )
    assert (
        _assessment(expected, "12.4%", roles, binding_start=21).status
        is SourceGroundingState.INVALID
    )


@pytest.mark.parametrize(
    "value",
    (
        PropositionValue(kind=PropositionValueKind.BOOLEAN, boolean=True),
        PropositionValue(kind=PropositionValueKind.TEXT, text="12.4%"),
        PropositionValue(kind=PropositionValueKind.ENTITY, text="12.4%"),
        PropositionValue(kind=PropositionValueKind.SET, members=("12.4%",)),
        PropositionValue(
            kind=PropositionValueKind.COMPARISON,
            comparison_target="baseline",
            comparison_metric_status=ComparisonMetricStatus.UNSPECIFIED,
        ),
        PropositionValue(
            kind=PropositionValueKind.TEMPORAL_CHANGE,
            temporal_from="2024",
            temporal_to="2025",
            numeric=NumericValue(operator=NumericOperator.EXACT, value=Decimal("12.4")),
        ),
        PropositionValue(kind=PropositionValueKind.UNKNOWN),
    ),
)
def test_unsupported_types_remain_unknown_without_text_fallback(value: PropositionValue) -> None:
    result = assess_typed_value_grounding(
        assertion_ref="A1",
        value=value,
        bindings=(),
        spans=(),
        source_ref=SOURCE,
        snapshot_ref=SNAPSHOT,
    )
    assert result.status is SourceGroundingState.UNKNOWN
    assert result.reason_codes == ("UNSUPPORTED_VALUE_KIND",)
