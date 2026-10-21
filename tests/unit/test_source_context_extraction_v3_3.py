from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from app.domain.source_context_extraction_v3_3 import (
    MeasurementAdmissibilityV33,
    MeasurementIdentityTypeV33,
    MeasurementIdentityV33,
    MeasurementSemanticsV33,
    SourceContextV33,
    SourceExtractionResponseV33,
)
from app.services.source_context_extractor_v3_3 import evaluate_measurement_admissibility_v33
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/measurement_identity_v3_3_cases.json"


def _response(case: dict[str, Any]) -> SourceExtractionResponseV33:
    context_ref = "M1"
    spans = [{"evidence_ref": "e1", "exact_text": case["identity_span"]}]
    bindings = [
        {
            "binding_ref": "b1",
            "evidence_ref": "e1",
            "semantic_role": "MEASUREMENT_IDENTITY",
            "target_kind": "CONTEXT",
            "target_ref": context_ref,
        }
    ]
    semantic_value: dict[str, Any] = {"measurement_identity": case["identity"]}
    if case["observed_value"] is not None:
        spans.append({"evidence_ref": "e2", "exact_text": case["value_span"]})
        bindings.append(
            {
                "binding_ref": "b2",
                "evidence_ref": "e2",
                "semantic_role": "MEASUREMENT_VALUE",
                "target_kind": "CONTEXT",
                "target_ref": context_ref,
            }
        )
        semantic_value["observed_value"] = case["observed_value"]
    return SourceExtractionResponseV33.model_validate(
        {
            "schema_version": "source-context-extraction-schema-v3.3",
            "status": "RESOLVED",
            "source_assertions": [
                {
                    "source_assertion_ref": "A1",
                    "evidence_refs": ["e1"],
                    "assertion_status": "ASSERTED",
                }
            ],
            "source_contexts": [
                {
                    "context_ref": context_ref,
                    "context_type": "MEASUREMENT",
                    "semantic_value": semantic_value,
                }
            ],
            "evidence_spans": spans,
            "evidence_bindings": bindings,
        }
    )


def test_identity_type_enum_matches_d4o_contract() -> None:
    assert {item.value for item in MeasurementIdentityTypeV33} == {
        "MEASURABLE_PROPERTY",
        "NAMED_METRIC",
        "SCORE_OR_INDEX",
        "COUNT",
        "RATE",
        "SIZE_OR_CAPACITY",
        "TEMPORAL_MEASURE",
        "QUALITATIVE_PROPERTY",
        "UNKNOWN",
    }


@pytest.mark.parametrize("case_index", range(6))
def test_d4o_synthetic_cases_follow_explicit_structured_contract(case_index: int) -> None:
    cases = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
    case = cases[case_index]
    response = _response(case)
    result = evaluate_measurement_admissibility_v33(response, case["claim_text"])
    assert result["M1"].value == case["expected"]


def test_identity_alone_unit_alone_and_number_alone_fail_closed() -> None:
    semantics = MeasurementSemanticsV33.model_validate(
        {
            "measurement_identity": {
                "name": "an unnamed property",
                "identity_type": "MEASURABLE_PROPERTY",
                "source_span": {"evidence_ref": "e1"},
            },
            "unit": "ms",
        }
    )
    response = SourceExtractionResponseV33.model_validate(
        {
            "schema_version": "source-context-extraction-schema-v3.3",
            "status": "RESOLVED",
            "source_assertions": [
                {
                    "source_assertion_ref": "A1",
                    "evidence_refs": ["e1"],
                    "assertion_status": "ASSERTED",
                }
            ],
            "source_contexts": [
                {
                    "context_ref": "M1",
                    "context_type": "MEASUREMENT",
                    "semantic_value": semantics.model_dump(mode="json"),
                }
            ],
            "evidence_spans": [{"evidence_ref": "e1", "exact_text": "opaque source wording"}],
        }
    )
    assert (
        evaluate_measurement_admissibility_v33(response, "opaque source wording")["M1"]
        is MeasurementAdmissibilityV33.UNKNOWN
    )


def test_invalid_identity_span_and_unlinked_threshold_fail_closed() -> None:
    payload = {
        "schema_version": "source-context-extraction-schema-v3.3",
        "status": "RESOLVED",
        "source_assertions": [
            {"source_assertion_ref": "A1", "evidence_refs": ["e1"], "assertion_status": "ASSERTED"}
        ],
        "source_contexts": [
            {
                "context_ref": "M1",
                "context_type": "MEASUREMENT",
                "semantic_value": {
                    "measurement_identity": {
                        "name": "latency",
                        "identity_type": "TEMPORAL_MEASURE",
                        "source_span": {"evidence_ref": "missing"},
                    },
                    "threshold": {
                        "comparator": "LTE",
                        "value": {"kind": "INTEGER", "raw_value": "10", "value": 10},
                        "raw_text": "at most 10",
                    },
                },
            }
        ],
        "evidence_spans": [{"evidence_ref": "e1", "exact_text": "latency at most 10"}],
    }
    response = SourceExtractionResponseV33.model_validate(payload)
    assert (
        evaluate_measurement_admissibility_v33(response, "latency at most 10")["M1"]
        is MeasurementAdmissibilityV33.UNKNOWN
    )


@pytest.mark.parametrize(
    ("basis_field", "basis_role", "basis_payload", "basis_span"),
    [
        (
            "formal_definition",
            "MEASUREMENT_DEFINITION",
            "elapsed time from request to response",
            "elapsed time from request to response",
        ),
        (
            "threshold",
            "MEASUREMENT_COMPARATOR",
            {
                "comparator": "LTE",
                "value": {"kind": "INTEGER", "raw_value": "10", "value": 10},
                "raw_text": "at most 10 ms",
            },
            "at most 10 ms",
        ),
    ],
)
def test_formal_definition_and_identity_linked_threshold_are_admissible(
    basis_field: str,
    basis_role: str,
    basis_payload: Any,
    basis_span: str,
) -> None:
    semantic_value: dict[str, Any] = {
        "measurement_identity": {
            "name": "latency",
            "identity_type": "TEMPORAL_MEASURE",
            "source_span": {"evidence_ref": "e1"},
        }
    }
    semantic_value[basis_field] = basis_payload
    response = SourceExtractionResponseV33.model_validate(
        {
            "schema_version": "source-context-extraction-schema-v3.3",
            "status": "RESOLVED",
            "source_assertions": [
                {
                    "source_assertion_ref": "A1",
                    "evidence_refs": ["e1", "e2"],
                    "assertion_status": "ASSERTED",
                }
            ],
            "source_contexts": [
                {
                    "context_ref": "M1",
                    "context_type": "MEASUREMENT",
                    "semantic_value": semantic_value,
                }
            ],
            "evidence_spans": [
                {"evidence_ref": "e1", "exact_text": "latency"},
                {"evidence_ref": "e2", "exact_text": basis_span},
            ],
            "evidence_bindings": [
                {
                    "binding_ref": "b1",
                    "evidence_ref": "e2",
                    "semantic_role": basis_role,
                    "target_kind": "CONTEXT",
                    "target_ref": "M1",
                }
            ],
        }
    )
    result = evaluate_measurement_admissibility_v33(
        response, f"latency; {basis_span}"
    )
    assert result["M1"] is MeasurementAdmissibilityV33.ADMISSIBLE


def test_qualitative_type_is_not_measurement_even_if_a_value_is_present() -> None:
    case = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"][1]
    case["observed_value"] = {"kind": "INTEGER", "raw_value": "3", "value": 3}
    case["value_span"] = "excellent"
    result = evaluate_measurement_admissibility_v33(_response(case), case["claim_text"])
    assert result["M1"] is MeasurementAdmissibilityV33.NOT_MEASUREMENT


def test_context_dispatch_keeps_non_measurement_payload_generic() -> None:
    context = SourceContextV33.model_validate(
        {
            "context_ref": "C1",
            "context_type": "CONDITION",
            "semantic_value": {"qualifier": "as stated"},
        }
    )
    assert context.context_type.value == "CONDITION"
    assert context.semantic_value.root == {"qualifier": "as stated"}


def test_unknown_identity_is_not_repaired_and_wrong_type_is_rejected() -> None:
    identity = MeasurementIdentityV33.model_validate({"identity_type": "UNKNOWN"})
    assert identity.name is None
    assert identity.source_span is None
    with pytest.raises(ValidationError):
        MeasurementSemanticsV33.model_validate(
            {"measurement_identity": {"identity_type": "METRIC_CATALOG_ENTRY"}}
        )
