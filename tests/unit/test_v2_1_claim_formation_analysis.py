from __future__ import annotations

from scripts.analyze_v2_1_claim_formation import (
    build_reports,
    deterministic_sample,
    generic_features,
    holdout_membership_only,
    overlap_sample,
    structural_overlap,
)


def _claim(claim_id: str, run_id: str, text: str) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "run_id": run_id,
        "question_id": "q-generic",
        "dimension_key": "d-generic",
        "claim_text": text,
        "claim_type": "factual",
        "evidence": [],
    }


def test_sample_is_deterministic_and_stratified_across_runs() -> None:
    rows = [
        _claim(f"{run}-{index:03d}", run, f"An independently verifiable statement {index}.")
        for run in ("run-a", "run-b", "run-c")
        for index in range(80)
    ]
    first = deterministic_sample(rows, size=30)
    second = deterministic_sample(rows, size=30)
    assert [item["claim_id"] for item in first] == [item["claim_id"] for item in second]
    assert {item["_run_bucket"] for item in first} == {"run-01", "run-02", "run-03"}


def test_generic_features_identify_composition_and_qualifier_cues() -> None:
    features = generic_features(
        "The value was 12% in 2022 and 15% in 2024, causing a measurable increase."
    )
    assert features["numeric"] is True
    assert features["multi_metric"] is True
    assert features["multi_time"] is True
    assert features["composite_heuristic"] is True
    assert features["bundled_fact_consequence"] is True


def test_decimal_point_alone_does_not_mark_a_claim_composite() -> None:
    features = generic_features("Accuracy was 37.4% in 2022.")
    assert features["numeric"] is True
    assert features["multi_metric"] is False
    assert features["composite_heuristic"] is False


def test_overlap_screen_is_structural_not_relation_prediction() -> None:
    assert (
        structural_overlap("System has 10 units in 2022", "System has 12 units in 2022")
        == "POTENTIAL_CONTRADICTION"
    )
    assert (
        structural_overlap(
            "The system has feature Y", "The system has feature Y under condition C using method Z"
        )
        == "GENERAL_SPECIFIC"
    )
    assert structural_overlap(
        "A distinct proposition about capacity", "A separate statement on unrelated staffing"
    ) in {"DISJOINT_RELATED", "INSUFFICIENT"}


def test_holdout_membership_helper_returns_only_ids() -> None:
    pairs, claims = holdout_membership_only(
        {
            "holdout_ids": ["holdout-pair"],
            "review_view": {
                "items": [
                    {
                        "pair_id": "holdout-pair",
                        "claim_a_id": "claim-a",
                        "claim_b_id": "claim-b",
                        "claim_a": "must not be returned",
                        "relation": "must not be returned",
                    },
                    {
                        "pair_id": "development-pair",
                        "claim_a_id": "claim-c",
                        "claim_b_id": "claim-d",
                    },
                ]
            },
        }
    )
    assert pairs == {"holdout-pair"}
    assert claims == {"claim-a", "claim-b"}


def test_report_decision_and_cardinality_are_stable() -> None:
    rows = [
        {
            **_claim(
                f"claim-{i:03d}", f"run-{i % 3}", f"Statement {i} has a value of {i + 1} units."
            ),
            "evidence": [
                {
                    "evidence_id": f"ev-{i}",
                    "source_id": f"source-{i}",
                    "quote": f"Statement {i} has a value of {i + 1} units.",
                    "accepted": i % 2 == 0,
                    "chunk_id": None,
                    "owner_key": f"owner-{i}",
                    "source_type": "generic",
                }
            ],
        }
        for i in range(120)
    ]
    baseline = {"claim_count": 120, "run_ids": ["run-a", "run-b", "run-c"]}
    reports = build_reports(rows, baseline, pair_count=25, holdout_claim_count=48)
    assert reports[0]["sample_size"] == 100
    assert reports[3]["decision"] == "REDESIGN_CLAIM_PLUS_ATOMIC_PROPOSITION"
    assert reports[1]["provenance_cardinality"]["claims_with_multiple_evidence"] == 0
    assert reports[0]["holdout_content_used"] is False
    assert overlap_sample(deterministic_sample(rows)) == overlap_sample(deterministic_sample(rows))
