from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from app.domain.providers import AdapterType, CanonicalModelResult, TokenUsage, UsageAccuracy
from app.services.atomic_proposition_decomposer import (
    ATOMIC_DECOMPOSITION_INSTRUCTIONS,
    AtomicPropositionDecomposer,
    ShadowDecompositionStatus,
)
from app.services.atomic_proposition_validation import (
    ATOMIC_DECOMPOSITION_PROMPT_VERSION,
    QualifierCheckStatus,
    validate_decomposition_response,
)
from pydantic import SecretStr

FIXTURE_PATH = Path("tests/fixtures/atomic_proposition_decomposition_v2_1r.json")


class FakeGateway:
    def __init__(
        self,
        parsed_object: dict[str, Any] | None = None,
        *,
        text: str | None = None,
        error: Exception | None = None,
    ):
        self.parsed_object = parsed_object
        self.text = text
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def generate_structured(self, **kwargs: Any) -> CanonicalModelResult:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return CanonicalModelResult(
            text=self.text,
            parsed_object=self.parsed_object,
            usage=TokenUsage(
                input_tokens=7, output_tokens=11, total_tokens=18, accuracy=UsageAccuracy.ESTIMATED
            ),
            provider_request_id="synthetic-request",
        )


def _decomposer(gateway: FakeGateway, *, max_calls: int = 1) -> AtomicPropositionDecomposer:
    return AtomicPropositionDecomposer(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://not-contacted.invalid/v1",
        api_key=SecretStr("synthetic-only"),
        model="synthetic-model",
        provider_profile_id="synthetic-profile",
        max_calls=max_calls,
    )


def test_synthetic_shape_matrix_a_to_t_and_shared_p_fixtures() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert len(fixture["cases"]) == 22
    expected = {
        "R_ambiguous": ShadowDecompositionStatus.UNRESOLVED_DECOMPOSITION,
        "S_duplicate": ShadowDecompositionStatus.STRUCTURED_OUTPUT_INVALID,
        "T_schema_mismatch": ShadowDecompositionStatus.SCHEMA_INVALID,
    }
    for case in fixture["cases"]:
        checked = validate_decomposition_response(case["response"], claim_text=case["claim_text"])
        wanted = expected.get(case["case_id"], ShadowDecompositionStatus.RESOLVED)
        if wanted == ShadowDecompositionStatus.RESOLVED:
            assert checked.valid, case["case_id"]
            assert checked.propositions, case["case_id"]
        elif case["case_id"] == "R_ambiguous":
            assert checked.valid
            assert checked.decomposition_status.value == "unresolved_decomposition"
        else:
            assert not checked.valid, case["case_id"]
            assert checked.validation_issues

    shared_a = next(item for item in fixture["cases"] if item["case_id"] == "shared_P_plus_Q")
    shared_b = next(item for item in fixture["cases"] if item["case_id"] == "shared_P_plus_R")
    assert (
        shared_a["response"]["propositions"][0]["proposition_text"]
        == shared_b["response"]["propositions"][0]["proposition_text"]
    )


def test_gateway_is_called_once_with_frozen_versions_and_no_regeneration() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    response = fixture["cases"][0]["response"]
    gateway = FakeGateway(response)
    decomposer = _decomposer(gateway)
    result = asyncio.run(
        decomposer.decompose_claim(
            fixture["cases"][0]["claim_text"],
            {"claim_id": "synthetic-claim", "api_key": "must-not-be-forwarded"},
            claim_id="synthetic-claim",
        )
    )

    assert result.status == ShadowDecompositionStatus.RESOLVED
    assert result.schema_version == "atomic-proposition-schema-v2"
    assert result.prompt_version == ATOMIC_DECOMPOSITION_PROMPT_VERSION
    assert result.model_call_count == 1
    assert result.provenance == {"claim_id": "synthetic-claim"}
    assert (result.input_tokens, result.output_tokens) == (7, 11)
    assert len(gateway.calls) == 1
    assert gateway.calls[0]["allow_regeneration"] is False
    request = gateway.calls[0]["request"]
    assert request.metadata["prompt_version"] == ATOMIC_DECOMPOSITION_PROMPT_VERSION
    assert "must-not-be-forwarded" not in str(request.content_parts)
    assert "atomic" in ATOMIC_DECOMPOSITION_INSTRUCTIONS.lower()


def test_malformed_json_fails_closed_and_never_retries() -> None:
    gateway = FakeGateway(text="{not-json")
    decomposer = _decomposer(gateway)
    first = asyncio.run(decomposer.decompose_claim("Synthetic assertion."))
    second = asyncio.run(decomposer.decompose_claim("Second synthetic assertion."))
    assert first.status == ShadowDecompositionStatus.STRUCTURED_OUTPUT_INVALID
    assert second.status == ShadowDecompositionStatus.CALL_BUDGET_EXHAUSTED
    assert len(gateway.calls) == 1
    assert gateway.calls[0]["allow_regeneration"] is False


def test_gateway_exception_fails_closed_without_retry_or_secret_leak() -> None:
    gateway = FakeGateway(error=TimeoutError("private endpoint and token must not leak"))
    result = asyncio.run(_decomposer(gateway).decompose_claim("Synthetic assertion."))
    assert result.status == ShadowDecompositionStatus.MODEL_ERROR
    assert result.model_call_count == 1
    assert len(gateway.calls) == 1
    assert "private endpoint" not in json.dumps(result.model_dump(mode="json"))


def test_explicit_qualifier_loss_is_reported_not_repaired() -> None:
    claim = (
        "The projected value for version 3.2 is not above $25 USD in 2024 compared with group B."
    )
    response = {
        "schema_version": "atomic-proposition-schema-v2",
        "decomposition_status": "single_proposition",
        "propositions": [
            {
                "proposition_text": "The value is above a threshold.",
                "subject": "value",
                "predicate": "is above",
                "typed_value": {"kind": "text", "text": "a threshold"},
                "polarity": "asserted",
            }
        ],
    }
    output = validate_decomposition_response(response, claim_text=claim)
    assert output.valid
    states = {item.category: item.status for item in output.qualifier_checks}
    assert states["version"] is QualifierCheckStatus.POTENTIAL_LOSS
    assert states["currency"] is QualifierCheckStatus.POTENTIAL_LOSS
    assert states["date_time"] is QualifierCheckStatus.POTENTIAL_LOSS
    assert states["modality"] is QualifierCheckStatus.POTENTIAL_LOSS
    assert states["negation"] is QualifierCheckStatus.POTENTIAL_LOSS
    assert output.validation_issues


def test_diagnostic_retention_distinguishes_explicit_model_unresolved() -> None:
    response = {
        "schema_version": "atomic-proposition-schema-v2",
        "decomposition_status": "unresolved_decomposition",
        "propositions": [],
        "ambiguity_reason": "The relation attachment is ambiguous.",
    }
    result = asyncio.run(_decomposer(FakeGateway(response)).decompose_claim("A synthetic claim."))
    retained = result.model_structured_result

    assert result.status is ShadowDecompositionStatus.UNRESOLVED_DECOMPOSITION
    assert retained is not None
    assert retained["model_returned_status"] == "unresolved_decomposition"
    assert retained["model_returned_proposition_count"] == 0
    assert retained["ambiguity_reason"] == "The relation attachment is ambiguous."
    assert [step["stage"] for step in result.status_transition_trace] == [
        "GATEWAY_RESULT",
        "PARSED_RESULT",
        "VALIDATED_RESULT",
        "FINAL_SHADOW_RESULT",
    ]
    assert result.status_transition_trace[1]["status"] == "PARSED"
    assert result.status_transition_trace[2]["proposition_count"] == 0


def test_diagnostic_retention_keeps_one_and_multiple_valid_propositions() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    one = next(item for item in fixture["cases"] if item["case_id"] == "A_simple_fact")
    multiple = next(
        item for item in fixture["cases"] if item["case_id"] == "B_independent_properties"
    )

    one_result = asyncio.run(
        _decomposer(FakeGateway(one["response"])).decompose_claim(one["claim_text"])
    )
    multi_result = asyncio.run(
        _decomposer(FakeGateway(multiple["response"])).decompose_claim(multiple["claim_text"])
    )

    assert one_result.model_structured_result["model_returned_proposition_count"] == 1
    assert one_result.validator_result["input_proposition_count"] == 1
    assert one_result.validator_result["output_proposition_count"] == 1
    assert one_result.status_transition_trace[-1]["status"] == "RESOLVED"
    assert multi_result.model_structured_result["model_returned_proposition_count"] == 2
    assert multi_result.validator_result["output_proposition_count"] == 2


def test_diagnostic_retention_distinguishes_malformed_json_from_schema_rejection() -> None:
    malformed = asyncio.run(_decomposer(FakeGateway(text="{not-json")).decompose_claim("A claim."))
    invalid_schema = {
        "schema_version": "atomic-proposition-schema-v2",
        "decomposition_status": "single_proposition",
        "propositions": [
            {
                "proposition_text": "A statement.",
                "predicate": "is stated",
                "typed_value": {"kind": "text", "text": "statement"},
                "polarity": "asserted",
            }
        ],
    }
    rejected = asyncio.run(_decomposer(FakeGateway(invalid_schema)).decompose_claim("A statement."))

    assert malformed.parser_result["status"] == "REJECTED"
    assert malformed.parser_result["reason_codes"] == ["structured_output_invalid"]
    assert malformed.model_structured_result is None
    assert rejected.parser_result["status"] == "PARSED"
    assert rejected.validator_result["status"] == "REJECTED"
    assert rejected.status is ShadowDecompositionStatus.SCHEMA_INVALID
    assert rejected.status_transition_trace[-2]["proposition_count"] == 0
    assert rejected.status_transition_trace[-1]["validation_issue_codes"] == ["schema_invalid"]


def test_field_diagnostics_preserve_safe_pydantic_location_and_contract() -> None:
    invalid = {
        "schema_version": "atomic-proposition-schema-v2",
        "decomposition_status": "single_proposition",
        "propositions": [
            {
                "proposition_text": "A synthetic proposition.",
                "predicate": "describes",
                "typed_value": {"kind": "entity", "text": "a synthetic object"},
                "polarity": "asserted",
            }
        ],
    }
    result = asyncio.run(_decomposer(FakeGateway(invalid)).decompose_claim("A synthetic claim."))

    diagnostic = result.validator_result["validation_diagnostics"][0]
    assert result.status is ShadowDecompositionStatus.SCHEMA_INVALID
    assert result.pre_validation_proposition_count == 1
    assert result.model_structured_result["model_returned_proposition_count"] == 1
    assert diagnostic["validation_stage"] == "PROPOSITION_SCHEMA"
    assert diagnostic["field_path"] == "propositions[0].subject"
    assert diagnostic["expected_contract"] == "Non-empty string, at most 1000 characters."
    assert diagnostic["observed_type"] == "null_or_missing"
    assert diagnostic["issue_code"] == "missing"


def test_diagnostic_retention_records_qualifier_loss_without_changing_decision() -> None:
    response = {
        "schema_version": "atomic-proposition-schema-v2",
        "decomposition_status": "single_proposition",
        "propositions": [
            {
                "proposition_text": "The rate is high.",
                "subject": "rate",
                "predicate": "is",
                "typed_value": {"kind": "text", "text": "high"},
                "polarity": "asserted",
            }
        ],
    }
    result = asyncio.run(
        _decomposer(FakeGateway(response)).decompose_claim("The rate may reach 99% in 2025.")
    )

    assert result.model_structured_result["model_returned_proposition_count"] == 1
    assert result.validator_result["input_proposition_count"] == 1
    assert result.validator_result["output_proposition_count"] == 1
    assert "potential_unit_loss" in result.validator_result["validation_issue_codes"]
    assert result.status is ShadowDecompositionStatus.UNRESOLVED_DECOMPOSITION


def test_diagnostic_retention_redacts_fake_gateway_secrets() -> None:
    error = RuntimeError(
        "api_key=fake-api-key Authorization:Bearer fake-auth secret_header=fake-header"
    )
    result = asyncio.run(_decomposer(FakeGateway(error=error)).decompose_claim("A claim."))
    serialized = json.dumps(result.model_dump(mode="json"))

    assert result.status is ShadowDecompositionStatus.MODEL_ERROR
    assert "fake-api-key" not in serialized
    assert "fake-auth" not in serialized
    assert "fake-header" not in serialized
    assert "Authorization" not in serialized
    assert result.status_transition_trace[0]["status"] == "ERROR"


def test_schema_rejects_unknown_fields_and_invalid_numeric_shape() -> None:
    response = {
        "schema_version": "atomic-proposition-schema-v2",
        "decomposition_status": "single_proposition",
        "unexpected": "not allowed",
        "propositions": [],
    }
    output = validate_decomposition_response(response, claim_text="Synthetic assertion.")
    assert not output.valid

    malformed = {
        "schema_version": "atomic-proposition-schema-v2",
        "decomposition_status": "single_proposition",
        "propositions": [
            {
                "proposition_text": "The value ranges.",
                "subject": "value",
                "predicate": "ranges",
                "typed_value": {
                    "kind": "range",
                    "numeric": {"operator": "range", "lower_bound": 9, "upper_bound": 2},
                },
                "polarity": "asserted",
            }
        ],
    }
    assert not validate_decomposition_response(malformed, claim_text="Synthetic assertion.").valid

    incompatible_kind = {
        "schema_version": "atomic-proposition-schema-v2",
        "decomposition_status": "single_proposition",
        "propositions": [
            {
                "proposition_text": "The value is textual.",
                "subject": "value",
                "predicate": "is",
                "typed_value": {"kind": "text", "text": "text", "boolean": True},
                "polarity": "asserted",
            }
        ],
    }
    assert not validate_decomposition_response(
        incompatible_kind, claim_text="Synthetic assertion."
    ).valid


def test_structured_slots_preserve_negation_modality_attribution_and_comparison() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    by_id = {item["case_id"]: item for item in fixture["cases"]}
    negated = validate_decomposition_response(
        by_id["G_negation"]["response"], claim_text=by_id["G_negation"]["claim_text"]
    ).propositions[0]
    modal = validate_decomposition_response(
        by_id["O_modality"]["response"], claim_text=by_id["O_modality"]["claim_text"]
    ).propositions[0]
    attributed = validate_decomposition_response(
        by_id["P_attribution"]["response"], claim_text=by_id["P_attribution"]["claim_text"]
    ).propositions[0]
    comparison = validate_decomposition_response(
        by_id["L_comparison"]["response"], claim_text=by_id["L_comparison"]["claim_text"]
    ).propositions[0]

    assert negated.polarity.value == "negated"
    assert modal.modality == "may"
    assert attributed.attribution == "the report"
    assert comparison.value.comparison_target == "system B"
    assert comparison.value.comparison_metric == "metric M"
