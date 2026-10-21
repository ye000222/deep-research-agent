from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.atomic_proposition_validation import (
    QualifierCheckStatus,
    inspect_qualifier_fields,
)

FIXTURE = Path("tests/fixtures/version_qualifier_precision_v2_1r.json")


def _fixture() -> dict[str, Any]:
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert value["synthetic_only"] is True
    return value


def _statuses(claim_text: str, propositions: list[dict[str, Any]]) -> dict[str, str]:
    return {
        item.category: item.status.value
        for item in inspect_qualifier_fields(claim_text, propositions)
    }


def _version_proposition(value: str, source_span: str) -> dict[str, Any]:
    return {
        "proposition_text": f"The component uses version {value}.",
        "subject": "The component",
        "predicate": "uses",
        "typed_value": {"kind": "entity", "text": f"version {value}"},
        "polarity": "asserted",
        "source_span": source_span,
        "material_qualifiers": [
            {
                "category": "version",
                "key": "version",
                "value": value,
                "source_span": source_span,
                "materiality": "material",
            }
        ],
    }


def test_generic_version_positive_evidence_is_detected_and_preserved() -> None:
    for case in _fixture()["positive_versions"]:
        statuses = _statuses(
            case["claim_text"],
            [_version_proposition(case["value"], case["claim_text"])],
        )
        assert statuses["version"] == QualifierCheckStatus.PRESERVED.value, case["case_id"]


def test_bare_decimals_and_quantitative_forms_are_not_versions() -> None:
    for case in _fixture()["hard_negative_quantities"]:
        statuses = _statuses(case["claim_text"], [])
        assert statuses["version"] == QualifierCheckStatus.NOT_APPLICABLE.value, case["case_id"]


def test_unmarked_two_segment_decimal_is_not_a_version_qualifier() -> None:
    statuses = _statuses("A result of 12.34 was recorded.", [])
    assert statuses["version"] == QualifierCheckStatus.NOT_APPLICABLE.value


def test_four_digit_sample_count_is_not_a_date_or_version() -> None:
    statuses = _statuses("The dataset contains 5354 images.", [])

    assert statuses["date_time"] == QualifierCheckStatus.NOT_APPLICABLE.value
    assert statuses["version"] == QualifierCheckStatus.NOT_APPLICABLE.value


def test_dropped_real_version_still_fails_closed() -> None:
    claim = "The component supports version 2.3."
    proposition = {
        "proposition_text": "The component is supported.",
        "subject": "The component",
        "predicate": "is supported",
        "typed_value": {"kind": "boolean", "boolean": True},
        "polarity": "asserted",
        "source_span": "The component is supported.",
    }

    statuses = _statuses(claim, [proposition])

    assert statuses["version"] == QualifierCheckStatus.POTENTIAL_LOSS.value


def test_generic_chinese_attribution_and_frequency_modality_are_applicable() -> None:
    attribution = _statuses(
        "该研究声称设备已经上线。",
        [
            {
                "proposition_text": "该研究声称设备已经上线。",
                "subject": "设备",
                "predicate": "已经上线",
                "typed_value": {"kind": "boolean", "boolean": True},
                "polarity": "asserted",
                "attribution": "该研究",
                "material_qualifiers": [
                    {
                        "category": "attribution",
                        "key": "attributed_to",
                        "value": "该研究",
                        "source_span": "该研究声称",
                        "materiality": "material",
                    }
                ],
            }
        ],
    )
    typicality = _statuses(
        "The system typically reduces latency.",
        [
            {
                "proposition_text": "The system typically reduces latency.",
                "subject": "The system",
                "predicate": "reduces latency",
                "typed_value": {"kind": "text", "text": "latency"},
                "polarity": "asserted",
                "material_qualifiers": [
                    {
                        "category": "modality",
                        "key": "frequency",
                        "value": "typically",
                        "source_span": "typically",
                        "materiality": "material",
                    }
                ],
            }
        ],
    )
    predicate_attribution = _statuses(
        "该研究提出一种通用方法。",
        [
            {
                "proposition_text": "该研究提出一种通用方法。",
                "subject": "该研究",
                "predicate": "提出",
                "typed_value": {"kind": "entity", "text": "一种通用方法"},
                "polarity": "asserted",
            }
        ],
    )

    assert attribution["attribution"] == QualifierCheckStatus.PRESERVED.value
    assert predicate_attribution["attribution"] == QualifierCheckStatus.PRESERVED.value
    assert typicality["modality"] == QualifierCheckStatus.PRESERVED.value


def test_condition_value_is_preserved_without_repeating_source_connective() -> None:
    statuses = _statuses(
        "Under input size 128x128, the device runs.",
        [
            {
                "proposition_text": "The device runs at input size 128x128.",
                "subject": "The device",
                "predicate": "runs",
                "typed_value": {"kind": "boolean", "boolean": True},
                "polarity": "asserted",
                "material_qualifiers": [
                    {
                        "category": "condition",
                        "key": "input_resolution",
                        "value": "128x128",
                        "materiality": "material",
                    }
                ],
            }
        ],
    )

    assert statuses["condition"] == QualifierCheckStatus.PRESERVED.value


def test_attribution_reporting_verbs_are_recognized_across_supported_scripts() -> None:
    cases = (
        ("The study finds a result.", "finds"),
        ("该研究认为方法有效。", "认为"),
        ("研究は新しい方法を提案する。", "提案する"),
        ("El estudio propone un método.", "propone"),
    )
    for claim, predicate in cases:
        statuses = _statuses(
            claim,
            [
                {
                    "proposition_text": claim,
                    "subject": "study",
                    "predicate": predicate,
                    "typed_value": {"kind": "text", "text": "a method"},
                    "polarity": "asserted",
                }
            ],
        )
        assert statuses["attribution"] == QualifierCheckStatus.PRESERVED.value, claim


def test_parameter_count_is_measurement_not_a_product_version_or_bare_unit() -> None:
    claim = "Model-X-640M has approximately 7 million parameters."
    proposition = {
        "proposition_text": claim,
        "subject": "Model-X-640M",
        "predicate": "has approximately 7 million parameters",
        "typed_value": {"kind": "text", "text": "approximately 7 million parameters"},
        "polarity": "asserted",
        "material_qualifiers": [
            {
                "category": "measurement",
                "key": "parameter_count",
                "value": "approximately 7 million",
                "source_span": "approximately 7 million parameters",
                "materiality": "material",
            }
        ],
    }

    statuses = _statuses(claim, [proposition])

    assert statuses["measurement"] == QualifierCheckStatus.PRESERVED.value
    assert statuses["version"] == QualifierCheckStatus.NOT_APPLICABLE.value
    assert statuses["unit"] == QualifierCheckStatus.NOT_APPLICABLE.value


def test_measurement_preserved_in_comparison_target_without_duplicate_category() -> None:
    claim = "Results compare with systems whose parameter count reaches several billion."
    proposition = {
        "proposition_text": claim,
        "subject": "Results",
        "predicate": "compare with systems",
        "typed_value": {
            "kind": "comparison",
            "comparison_target": "systems whose parameter count reaches several billion",
            "comparison_metric_status": "unspecified",
        },
        "polarity": "asserted",
        "material_qualifiers": [
            {
                "category": "comparison_target",
                "key": "parameter range",
                "value": "several billion",
                "materiality": "material",
            }
        ],
    }

    statuses = _statuses(claim, [proposition])

    assert statuses["measurement"] == QualifierCheckStatus.PRESERVED.value
