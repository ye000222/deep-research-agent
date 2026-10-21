from __future__ import annotations

from typing import Any
from uuid import uuid4

from scripts.analyze_v2_1_claim_equivalence import (
    compare_history,
    generalization_audit,
    make_real_claim_eval_dataset,
    real_claim_eval_markdown,
    scoped_pairs,
    unknown_reason_counts,
)


def _row(
    claim_id: str,
    run_id: str,
    text: str,
    *,
    question_id: str = "q1",
    dimension_key: str = "q1:d1",
    evidence_count: int = 1,
    owners: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "run_id": run_id,
        "question_id": question_id,
        "dimension_key": dimension_key,
        "atomic_claim": text,
        "claim_type": "factual",
        "accepted_support_evidence_count": evidence_count,
        "owner_keys": owners or [],
    }


def test_pair_generation_blocks_cross_scope_pairs_before_comparison() -> None:
    rows = [
        _row(str(uuid4()), str(uuid4()), "Revenue was $20B in 2024."),
        _row(str(uuid4()), str(uuid4()), "Revenue was $20B in 2024."),
        _row(str(uuid4()), str(uuid4()), "Revenue was $20B in 2024.", question_id="q2"),
    ]
    pairs, cross_scope_filtered, cap_filtered, scoped_count = scoped_pairs(rows)
    assert pairs == []
    assert cross_scope_filtered == 3
    assert cap_filtered == 0
    assert scoped_count == 0


def test_candidate_pairing_is_scoped_to_run_question_dimension() -> None:
    run_id = str(uuid4())
    rows = [
        _row(str(uuid4()), run_id, "Same claim."),
        _row(str(uuid4()), run_id, "Same claim."),
        _row(str(uuid4()), run_id, "Same claim.", dimension_key="q1:d2"),
    ]
    replay = compare_history(rows)
    assert replay["candidate_pair_count"] == 1
    assert replay["pairs_filtered_cross_run_question_dimension"] == 2
    assert replay["relation_counts"]["same"] == 1
    assert replay["semantic_model_calls"] == 0


def test_clusters_require_high_confidence_same_and_keep_owner_metadata() -> None:
    run_id = str(uuid4())
    rows = [
        _row(str(uuid4()), run_id, "The company was founded in 2018.", owners=["owner-a"]),
        _row(str(uuid4()), run_id, "The company was founded in 2018.", owners=["owner-b"]),
        _row(str(uuid4()), run_id, "The company was founded in 2019.", owners=["owner-c"]),
    ]
    replay = compare_history(rows)
    assert replay["semantic_cluster_count"] == 1
    cluster = replay["clusters"][0]
    assert len(cluster["claim_ids"]) == 2
    assert cluster["accepted_support_evidence_count"] == 2
    assert cluster["independent_owner_count"] == 2
    assert replay["potential_multi_evidence_cluster_count"] == 1
    assert replay["potential_multi_owner_cluster_count"] == 1


def test_real_eval_dataset_has_blind_review_view_and_frozen_split() -> None:
    run_id = str(uuid4())
    pairs = [
        {
            "left_claim_id": str(uuid4()),
            "right_claim_id": str(uuid4()),
            "run_id": run_id,
            "question_id": "q1",
            "dimension_key": "q1:d1",
            "left_text": f"Revenue was {index} million in 2024.",
            "right_text": f"Revenue was {index + 1} million in 2024.",
            "lexical_overlap_jaccard": 0.8,
            "relation": "unknown",
            "confidence": 0.25,
            "reason_codes": ["INSUFFICIENT_INFORMATION"],
        }
        for index in range(80)
    ]

    first = make_real_claim_eval_dataset(pairs)
    second = make_real_claim_eval_dataset(pairs)

    assert len(first["development_ids"]) == 45
    assert len(first["holdout_ids"]) == 25
    assert first["holdout_ids"] == second["holdout_ids"]
    assert first["holdout_membership_sha256"] == second["holdout_membership_sha256"]
    assert first["review_view"]["matcher_predictions_included"] is False
    assert all("matcher_prediction" not in row for row in first["review_view"]["items"])
    assert all(row["human_label"] is None for row in first["review_view"]["items"])
    assert real_claim_eval_markdown(first).find("unknown") == -1


def test_real_eval_dataset_excludes_prior_prediction_exposure_from_both_splits() -> None:
    run_id = str(uuid4())
    pairs = [
        {
            "left_claim_id": str(uuid4()),
            "right_claim_id": str(uuid4()),
            "run_id": run_id,
            "question_id": "q1",
            "dimension_key": "q1:d1",
            "left_text": f"A factual proposition numbered {index} in 2024.",
            "right_text": f"A different proposition numbered {index + 1} in 2025.",
            "lexical_overlap_jaccard": 0.5,
            "relation": "unknown",
            "confidence": 0.25,
            "reason_codes": ["INSUFFICIENT_INFORMATION"],
        }
        for index in range(160)
    ]
    previously_exposed = make_real_claim_eval_dataset(pairs)
    exposed_ids = set(previously_exposed["development_ids"] + previously_exposed["holdout_ids"])

    regenerated = make_real_claim_eval_dataset(pairs, excluded_pair_ids=exposed_ids)
    regenerated_split_ids = set(regenerated["development_ids"] + regenerated["holdout_ids"])

    assert regenerated["prior_prediction_exposure_excluded_from_both_splits"] == len(exposed_ids)
    assert regenerated["excluded_prior_prediction_exposure_pair_ids"] == sorted(exposed_ids)
    assert not exposed_ids & regenerated_split_ids
    assert len(regenerated["development_ids"]) == 45
    assert len(regenerated["holdout_ids"]) == 25


def test_unknown_reason_diagnostics_are_bounded_and_non_labeling() -> None:
    reasons = unknown_reason_counts(
        [
            {
                "relation": "unknown",
                "left_text": "Claim A.",
                "right_text": "Claim B.",
                "signals": {
                    "subject_equal": None,
                    "predicate_equal": None,
                    "numeric_values_equal": None,
                },
                "qualifier_compatibility": "compatible",
                "reason_codes": ["INSUFFICIENT_INFORMATION"],
            }
        ]
    )
    assert reasons["MISSING_STRUCTURED_FIELDS"] == 1
    assert reasons["ENTITY_UNRESOLVED"] == 1
    assert reasons["PREDICATE_UNRESOLVED"] == 1
    assert reasons["VALUE_UNRESOLVED"] == 1


def test_generalization_audit_has_no_forbidden_production_literals() -> None:
    audit = generalization_audit()
    assert audit["result"] == "NO_FORBIDDEN_SPECIAL_CASES"
    assert audit["holdout_influenced"] is False
    assert all(not item["holdout_influenced"] for item in audit["mechanisms"])
