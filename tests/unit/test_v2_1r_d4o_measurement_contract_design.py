"""Contract-example integrity tests; these do not implement classification."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "artifacts/v2_1r_d4o/synthetic_measurement_cases.json"


def test_d4o_synthetic_contract_examples_are_complete_and_explicit() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cases = data["cases"]

    assert len(cases) == 6
    assert {case["case_id"] for case in cases} == {
        "SYN-MEAS-01",
        "SYN-MEAS-02",
        "SYN-MEAS-03",
        "SYN-MEAS-04",
        "SYN-MEAS-05",
        "SYN-MEAS-06",
    }
    assert [case["expected"] for case in cases] == [
        "MEASUREMENT",
        "NOT_MEASUREMENT",
        "UNKNOWN",
        "MEASUREMENT",
        "NOT_MEASUREMENT",
        "UNKNOWN",
    ]
    assert all(case["text"] and case["reason"] for case in cases)
    assert "No text matching" in data["evaluation_boundary"]
