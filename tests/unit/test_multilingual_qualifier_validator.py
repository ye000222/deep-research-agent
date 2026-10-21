from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from app.domain.providers import AdapterType, CanonicalModelResult, TokenUsage, UsageAccuracy
from app.services.atomic_proposition_decomposer import AtomicPropositionDecomposer
from app.services.atomic_proposition_validation import (
    QualifierCheckStatus,
    inspect_qualifier_fields,
    validate_decomposition_response,
)
from pydantic import SecretStr

FIXTURE = Path("tests/fixtures/multilingual_qualifier_v2_1r.json")
CANARY_ARTIFACT = Path("artifacts/v2_1r_semantic_canary_20260927T021637269654Z_77988f2f.json")


class FakeGateway:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0

    async def generate_structured(self, **_: Any) -> CanonicalModelResult:
        self.calls += 1
        return CanonicalModelResult(
            text=None,
            parsed_object=self.payload,
            usage=TokenUsage(
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                accuracy=UsageAccuracy.ESTIMATED,
            ),
        )


def _fixture() -> dict[str, Any]:
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert value["synthetic_only"] is True
    return value


def test_multilingual_fixture_categories_are_preserved_consistently() -> None:
    cases = _fixture()["cases"]
    results: dict[str, dict[str, QualifierCheckStatus]] = {}
    for case in cases:
        checks = inspect_qualifier_fields(case["claim_text"], case["propositions"])
        statuses = {item.category: item.status for item in checks}
        if not case["case_id"].endswith("_structural"):
            results[case["language"]] = statuses
        for category in case["must_preserve"]:
            assert statuses[category] is QualifierCheckStatus.PRESERVED, (
                case["case_id"],
                category,
                statuses[category],
            )

    linguistic_cases = [item for item in cases if not item["case_id"].endswith("_structural")]
    shared_categories = set.intersection(*(set(item["must_preserve"]) for item in linguistic_cases))
    assert shared_categories
    for category in shared_categories:
        assert {result[category] for result in results.values()} == {QualifierCheckStatus.PRESERVED}


def test_hard_negative_text_does_not_trigger_english_substring_false_positives() -> None:
    for case in _fixture()["hard_negatives"]:
        checks = {item.category: item.status for item in inspect_qualifier_fields(case["text"], [])}
        assert all(status is QualifierCheckStatus.NOT_APPLICABLE for status in checks.values())
        for category in case["not_detected"]:
            assert checks[category] is QualifierCheckStatus.NOT_APPLICABLE, (
                case["text"],
                category,
            )


def test_prose_token_elsewhere_is_not_enough_to_claim_negation_preserved() -> None:
    claim = "The component is not active."
    output = [
        {
            "proposition_text": "The word not appears in this unrelated description.",
            "subject": "component",
            "predicate": "is active",
            "polarity": "asserted",
        }
    ]
    checks = {item.category: item.status for item in inspect_qualifier_fields(claim, output)}
    assert checks["negation"] is QualifierCheckStatus.POTENTIAL_LOSS


def test_ambiguous_multi_proposition_scope_fails_closed() -> None:
    claim = "Under setting B, device A runs; device C remains idle."
    response = {
        "schema_version": "atomic-proposition-schema-v2",
        "decomposition_status": "decomposed",
        "propositions": [
            {
                "proposition_text": "Device A runs under setting B.",
                "subject": "device A",
                "predicate": "runs",
                "typed_value": {"kind": "boolean", "boolean": True},
                "polarity": "asserted",
                "material_qualifiers": [
                    {
                        "category": "configuration",
                        "key": "setting",
                        "value": "setting B",
                        "materiality": "material",
                    }
                ],
            },
            {
                "proposition_text": "Device C remains idle.",
                "subject": "device C",
                "predicate": "remains",
                "typed_value": {"kind": "text", "text": "idle"},
                "polarity": "asserted",
            },
        ],
    }
    checked = validate_decomposition_response(response, claim_text=claim)
    statuses = {item.category: item.status for item in checked.qualifier_checks}
    assert checked.valid
    assert statuses["condition"] is QualifierCheckStatus.AMBIGUOUS
    assert any(issue.code == "ambiguous_condition_scope" for issue in checked.validation_issues)


def test_offline_replay_of_persisted_v2_canary_uses_safe_fields_only() -> None:
    artifact = json.loads(CANARY_ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["status"] == "COMPLETED"
    source_claim = artifact["claim"]["claim_text"]
    safe_props = artifact["decomposition"]["model_structured_result"]["propositions"]
    checks = {
        item.category: item.status for item in inspect_qualifier_fields(source_claim, safe_props)
    }

    assert checks["date_time"] is QualifierCheckStatus.PRESERVED
    assert checks["comparison"] is QualifierCheckStatus.PRESERVED
    # These sentence-level signals share one explicit nested source span across all
    # three propositions; shared attribution/modality is semantically compatible.
    assert checks["modality"] is QualifierCheckStatus.PRESERVED
    assert checks["attribution"] is QualifierCheckStatus.PRESERVED
    assert checks["unit"] is QualifierCheckStatus.PRESERVED
    assert checks["sample_count"] is QualifierCheckStatus.PRESERVED
    # The captured source span covers the sample-count phrase but not the separate
    # accuracy obligation; preserving the latter cannot be inferred from proximity.
    assert checks["measurement"] is QualifierCheckStatus.AMBIGUOUS


def test_safe_v2_diagnostics_retain_comparison_and_numeric_semantics_offline() -> None:
    payload = {
        "schema_version": "atomic-proposition-schema-v2",
        "decomposition_status": "single_proposition",
        "propositions": [
            {
                "proposition_text": "A unit has a measured value.",
                "subject": "A unit",
                "predicate": "has",
                "typed_value": {
                    "kind": "comparison",
                    "comparison_target": "unit B",
                    "comparison_metric": "latency",
                    "comparison_metric_status": "explicit",
                    "numeric": {
                        "operator": "approx",
                        "value": 4,
                        "unit": "ms",
                        "measurement_basis": "reported",
                        "extremum": "peak",
                        "change_type": "relative",
                    },
                },
                "polarity": "asserted",
            }
        ],
    }
    gateway = FakeGateway(payload)
    decomposer = AtomicPropositionDecomposer(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://offline.invalid/v1",
        api_key=SecretStr("synthetic-secret"),
        model="synthetic-model",
    )
    result = asyncio.run(decomposer.decompose_claim("A synthetic assertion."))

    assert gateway.calls == 1
    retained = result.model_structured_result["propositions"][0]["typed_value"]
    assert retained["comparison_metric_status"] == "explicit"
    assert retained["numeric"]["measurement_basis"] == "reported"
    assert retained["numeric"]["extremum"] == "peak"
    assert retained["numeric"]["change_type"] == "relative"
