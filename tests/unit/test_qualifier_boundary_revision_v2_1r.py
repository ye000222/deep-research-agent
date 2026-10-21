from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from app.services.atomic_proposition_validation import (
    QualifierCheckStatus,
    inspect_qualifier_fields,
)

FIXTURE = Path("tests/fixtures/qualifier_boundary_revision_v2_1r.json")


def _fixture() -> dict[str, Any]:
    value = cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert value["synthetic_only"] is True
    return value


def _checks(claim_text: str, propositions: list[dict[str, Any]]) -> dict[str, str]:
    return {
        check.category: check.status.value
        for check in inspect_qualifier_fields(claim_text, propositions)
    }


def test_cjk_negation_is_detected_when_structured_polarity_supports_it() -> None:
    for case in _fixture()["negation_positive"]:
        checks = _checks(case["claim_text"], case["propositions"])
        assert checks["negation"] == QualifierCheckStatus.PRESERVED.value, case["case_id"]


def test_cjk_negation_substrings_inside_lexical_items_are_not_negation() -> None:
    for case in _fixture()["negation_hard_negatives"]:
        checks = _checks(case["claim_text"], case["propositions"])
        assert checks["negation"] == QualifierCheckStatus.NOT_APPLICABLE.value, case["case_id"]


def test_method_relation_in_predicate_counts_only_with_same_proposition_object() -> None:
    for case in _fixture()["method_equivalence"]:
        checks = _checks(case["claim_text"], case["propositions"])
        assert checks["method"] == case["expected"], case["case_id"]
