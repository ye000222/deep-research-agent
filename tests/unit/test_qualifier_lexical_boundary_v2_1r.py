from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.services.atomic_proposition_validation import (
    QualifierCheckStatus,
    inspect_qualifier_fields,
)

FIXTURE = Path("tests/fixtures/qualifier_lexical_boundary_v2_1r.json")


def _fixture() -> dict[str, Any]:
    value = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert value["synthetic_only"] is True
    return value


def test_short_cjk_attribution_markers_require_source_lexical_boundary() -> None:
    fixture = _fixture()
    for case in fixture["positive_attribution"]:
        checks = {
            item.category: item.status
            for item in inspect_qualifier_fields(case["claim_text"], case["propositions"])
        }
        assert checks["attribution"] is QualifierCheckStatus.PRESERVED, case["case_id"]

    for case in fixture["hard_negatives"]:
        checks = {
            item.category: item.status
            for item in inspect_qualifier_fields(case["text"], [])
        }
        for category in case["not_detected"]:
            assert checks[category] is QualifierCheckStatus.NOT_APPLICABLE, case["case_id"]


def test_other_short_qualifier_forms_keep_existing_structural_behavior() -> None:
    claim = (
        "报告R称\uFF0C在配置B条件下\uFF0C采用方法K测得的比率可能达到8\uFF05\uFF0C"
        "而不适用于配置A。"
    )
    proposition = {
        "proposition_text": claim,
        "subject": "比率",
        "predicate": "可能达到",
        "polarity": "negated",
        "modality": "可能",
        "attribution": "报告R",
        "typed_value": {"kind": "quantity", "numeric": {"value": 8, "unit": "%"}},
        "material_qualifiers": [
            {"category": "configuration", "value": "配置B"},
            {"category": "condition", "value": "配置B条件下"},
            {"category": "method", "value": "方法K"},
            {"category": "measurement", "value": "比率"},
        ],
    }
    statuses = {
        item.category: item.status
        for item in inspect_qualifier_fields(claim, [proposition])
    }
    assert statuses["attribution"] is QualifierCheckStatus.PRESERVED
    assert statuses["modality"] is QualifierCheckStatus.PRESERVED
    assert statuses["negation"] is QualifierCheckStatus.PRESERVED
    assert statuses["condition"] is QualifierCheckStatus.PRESERVED
    assert statuses["method"] is QualifierCheckStatus.PRESERVED


def test_numeric_unit_inside_hyphenated_identifier_is_not_a_measurement() -> None:
    checks = {
        item.category: item.status
        for item in inspect_qualifier_fields(
            "Atlas-500M 是紧凑模型。",
            [],
        )
    }
    assert checks["unit"] is QualifierCheckStatus.NOT_APPLICABLE
