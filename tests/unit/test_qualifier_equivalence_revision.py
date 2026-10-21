from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from app.services.atomic_proposition_validation import (
    QualifierCheckStatus,
    inspect_qualifier_fields,
)

FIXTURE = Path("tests/fixtures/qualifier_equivalence_v2_1r.json")


def _fixture() -> dict[str, Any]:
    value = cast(dict[str, Any], json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert value["synthetic_only"] is True
    return value


def test_category_specific_semantic_equivalence_fixture() -> None:
    for case in _fixture()["cases"]:
        checks = {
            item.category: item.status.value
            for item in inspect_qualifier_fields(case["claim_text"], case["propositions"])
        }
        for category, expected in case["expected"].items():
            assert checks[category] == expected, (case["case_id"], category, checks)


def test_category_specific_hard_negatives_fail_closed_without_false_qualifiers() -> None:
    for case in _fixture()["hard_negatives"]:
        checks = {
            item.category: item.status
            for item in inspect_qualifier_fields(case["claim_text"], [])
        }
        for category, expected in case["expected"].items():
            assert checks[category] is QualifierCheckStatus(expected), (case["case_id"], checks)
