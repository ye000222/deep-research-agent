from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.domain.atomic_proposition import (
    ATOMIC_PROPOSITION_SCHEMA_VERSION,
    ContradictionComparability,
    GuardResult,
    NumericChangeType,
    NumericExtremum,
    NumericMeasurementBasis,
    PropositionValue,
    contradiction_comparability_guard,
    exact_structured_identity_guard,
)
from app.services.atomic_proposition_decomposer import ATOMIC_DECOMPOSITION_INSTRUCTIONS
from app.services.atomic_proposition_validation import (
    ATOMIC_DECOMPOSITION_PROMPT_VERSION,
    ShadowDecompositionResponse,
    validate_decomposition_response,
)
from pydantic import ValidationError

FIXTURE_PATH = Path("tests/fixtures/semantic_proposition_formation_v2_1r.json")


def _cases() -> dict[str, dict[str, object]]:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert fixture["synthetic_only"] is True
    return {case["case_id"]: case for case in fixture["cases"]}


def test_generic_synthetic_fixture_matrix_validates_without_provider_calls() -> None:
    cases = _cases()
    assert len(cases) >= 18
    for case_id, case in cases.items():
        checked = validate_decomposition_response(
            case["response"], claim_text=case["claim_text"]
        )
        assert checked.valid, case_id
        if case["response"]["decomposition_status"] == "unresolved_decomposition":
            assert checked.propositions == (), case_id
        else:
            assert checked.propositions, case_id
            assert all(proposition.original_text for proposition in checked.propositions)


def test_qualitative_comparison_needs_no_magnitude_and_is_fail_closed_for_identity() -> None:
    case = _cases()["qualitative_comparison"]
    checked = validate_decomposition_response(case["response"], claim_text=case["claim_text"])
    value = checked.propositions[0].value

    assert value.comparison_target == "Unit B"
    assert value.comparison_metric is None
    assert value.comparison_metric_status.value == "unspecified"
    assert value.numeric is None
    proposition = checked.propositions[0]
    assert (
        exact_structured_identity_guard(proposition, proposition).value
        == GuardResult.UNKNOWN.value
    )
    assert (
        contradiction_comparability_guard(proposition, proposition)
        .value
        == ContradictionComparability.UNKNOWN.value
    )


def test_quantitative_comparison_preserves_only_asserted_magnitude_and_metric() -> None:
    case = _cases()["quantitative_comparison"]
    checked = validate_decomposition_response(case["response"], claim_text=case["claim_text"])
    value = checked.propositions[0].value

    assert value.comparison_metric_status.value == "explicit"
    assert value.comparison_metric == "metric M"
    assert value.numeric is not None and value.numeric.value == 4


@pytest.mark.parametrize(
    "value",
    [
        {
            "kind": "comparison",
            "comparison_target": "Unit B",
            "comparison_metric": "latency",
            "comparison_metric_status": "unspecified",
        },
        {
            "kind": "comparison",
            "comparison_target": "Unit B",
            "comparison_metric_status": "explicit",
        },
        {"kind": "comparison", "comparison_target": "Unit B", "comparison_metric": "latency"},
    ],
)
def test_comparison_metric_status_prevents_invented_or_unclassified_metric(value: dict) -> None:
    with pytest.raises(ValidationError):
        PropositionValue.model_validate(value)


def test_context_resolved_metric_is_distinct_from_explicit_metric() -> None:
    case = _cases()["context_resolved_comparison_metric"]
    checked = validate_decomposition_response(case["response"], claim_text=case["claim_text"])
    value = checked.propositions[0].value
    assert value.comparison_metric == "throughput"
    assert value.comparison_metric_status.value == "context_resolved"


def test_numeric_operators_remain_mathematical_and_context_is_orthogonal() -> None:
    cases = _cases()
    expected = {
        "exact_numeric": ("exact", None, None, None),
        "approximate_numeric": ("approx", None, None, None),
        "lower_bound": ("gte", None, None, None),
        "upper_bound": ("lte", NumericMeasurementBasis.OBSERVED, None, None),
        "interval_range": ("range", NumericMeasurementBasis.OBSERVED, None, None),
        "sample_configuration_condition": (
            "exact",
            NumericMeasurementBasis.ATTAINABLE,
            None,
            None,
        ),
        "reported_peak_result": (
            "exact",
            NumericMeasurementBasis.REPORTED,
            NumericExtremum.PEAK,
            None,
        ),
        "best_result": ("exact", None, NumericExtremum.BEST, None),
        "absolute_change": ("exact", None, None, NumericChangeType.ABSOLUTE),
        "relative_change": ("approx", None, None, NumericChangeType.RELATIVE),
    }
    for case_id, wanted in expected.items():
        case = cases[case_id]
        checked = validate_decomposition_response(case["response"], claim_text=case["claim_text"])
        numeric = checked.propositions[0].value.numeric
        assert numeric is not None
        actual = (
            numeric.operator.value,
            numeric.measurement_basis,
            numeric.extremum,
            numeric.change_type,
        )
        assert actual == wanted


def test_subject_stays_entity_bearer_and_conditions_are_qualifiers() -> None:
    cases = _cases()
    condition_case = cases["explicit_condition"]
    condition = validate_decomposition_response(
        condition_case["response"], claim_text=condition_case["claim_text"]
    ).propositions[0]
    assert condition.subject == "Module M"
    assert condition.qualifiers[0].category.value == "condition"
    assert condition.qualifiers[0].value == "C"

    sample_case = cases["sample_configuration_condition"]
    sample = validate_decomposition_response(
        sample_case["response"], claim_text=sample_case["claim_text"]
    ).propositions[0]
    assert sample.subject == "Unit A"
    assert sample.qualifiers[0].category.value == "configuration"
    assert sample.qualifiers[0].value == "three"


def test_missing_subject_and_ambiguous_composite_fail_closed_as_whole_claim() -> None:
    cases = _cases()
    for case_id in ("missing_subject", "partially_ambiguous_composite_whole_claim_unresolved"):
        case = cases[case_id]
        response = case["response"]
        checked = validate_decomposition_response(response, claim_text=case["claim_text"])
        assert checked.valid
        assert checked.decomposition_status.value == "unresolved_decomposition"
        assert checked.propositions == ()
        assert response["propositions"] == []
        assert response["ambiguity_reason"]


def test_proposition_text_contract_requires_independent_truth_condition_context() -> None:
    cases = _cases()
    forecast_case = cases["forecast_modality"]
    forecast = validate_decomposition_response(
        forecast_case["response"], claim_text=forecast_case["claim_text"]
    ).propositions[0]
    attributed_case = cases["attributed_assertion"]
    attributed = validate_decomposition_response(
        attributed_case["response"], claim_text=attributed_case["claim_text"]
    ).propositions[0]

    assert "forecasts" in forecast.original_text
    assert forecast.modality == "forecast"
    assert forecast.attribution == "the analyst"
    assert "According to Review R" in attributed.original_text
    assert attributed.attribution == "Review R"
    assert "truth-condition-complete" in ATOMIC_DECOMPOSITION_INSTRUCTIONS


def test_material_qualifiers_modality_and_numeric_form_change_identity() -> None:
    cases = _cases()
    condition_case = cases["explicit_condition"]
    left = validate_decomposition_response(
        condition_case["response"], claim_text=condition_case["claim_text"]
    ).propositions[0]
    other_condition = left.model_copy(
        update={
            "qualifiers": (
                left.qualifiers[0].model_copy(update={"value": "D"}),
            )
        }
    )
    assert exact_structured_identity_guard(left, other_condition).value == "not_same"

    forecast_case = cases["forecast_modality"]
    forecast = validate_decomposition_response(
        forecast_case["response"], claim_text=forecast_case["claim_text"]
    ).propositions[0]
    observed = forecast.model_copy(update={"modality": "observed"})
    assert exact_structured_identity_guard(forecast, observed).value == "not_same"

    qualitative_case = cases["qualitative_comparison"]
    quantitative_case = cases["quantitative_comparison"]
    qualitative = validate_decomposition_response(
        qualitative_case["response"], claim_text=qualitative_case["claim_text"]
    ).propositions[0]
    quantitative = validate_decomposition_response(
        quantitative_case["response"], claim_text=quantitative_case["claim_text"]
    ).propositions[0]
    assert exact_structured_identity_guard(qualitative, quantitative).value == "unknown"


def test_schema_and_prompt_versions_are_bumped_for_material_contract_change() -> None:
    assert ATOMIC_PROPOSITION_SCHEMA_VERSION == "atomic-proposition-schema-v2"
    assert ATOMIC_DECOMPOSITION_PROMPT_VERSION == "atomic-decomposition.generic.v2"
    old_version_payload = {
        "schema_version": "atomic-proposition-schema-v1",
        "decomposition_status": "unresolved_decomposition",
        "propositions": [],
        "ambiguity_reason": "legacy version must not masquerade as current",
    }
    assert not validate_decomposition_response(
        old_version_payload, claim_text="Synthetic assertion."
    ).valid
    schema_version = ShadowDecompositionResponse.model_json_schema()["properties"][
        "schema_version"
    ]["const"]
    assert schema_version == ATOMIC_PROPOSITION_SCHEMA_VERSION
