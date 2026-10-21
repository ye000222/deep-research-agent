from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / "artifacts/v2_1r_truth_condition_context_c1_audit_selection.json"
MANUAL_AUDIT = ROOT / "artifacts/v2_1r_truth_condition_context_c1_manual_audit.json"
SHARED_AUDIT = ROOT / "artifacts/v2_1r_truth_condition_context_c1_shared_context_reaudit.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_audit_selection_is_deterministic_development_only_and_diverse() -> None:
    selection = _load(SELECTION)
    assert selection["selected_count"] == 40
    assert selection["target_count"] == 40
    assert selection["development_only"] is True
    assert selection["holdout_content_read"] is False
    assert selection["provider_calls"] == 0
    ids = [entry["claim_id"] for entry in selection["selected_claims"]]
    assert len(ids) == len(set(ids)) == 40
    assert {entry["cohort"] for entry in selection["selected_claims"]} == {
        "bounded_shadow",
        "fresh_validation",
        "final_unseen",
    }
    assert {entry["stratum"][1] for entry in selection["selected_claims"]} == {
        "detected",
        "zero_detected",
    }
    assert selection["manual_review_started"] is False


def test_manual_audit_has_one_record_for_every_frozen_claim_and_rates_are_support_aware() -> None:
    selection = _load(SELECTION)
    audit = _load(MANUAL_AUDIT)
    assert audit["selection_manifest_sha256"]
    assert audit["claims_total"] == 40
    assert {row["claim_id"] for row in audit["claims"]} == {
        row["claim_id"] for row in selection["selected_claims"]
    }
    assert all(row["manual_source_contexts"] is not None for row in audit["claims"])
    assert audit["provider_calls"] == 0
    assert audit["holdout_content_read"] is False
    assert audit["aggregate"]["false_attribution_from_retrieval_metadata"] == 0
    assert audit["by_family"]["TEMPORAL"]["sample_support"] == "REPORT_RATE"
    assert audit["by_family"]["CONFIGURATION"]["sample_support"] == "COUNTS_ONLY_SMALL_FAMILY"
    assert audit["by_family"]["CONFIGURATION"]["precision"] is None
    assert audit["aggregate"]["matched"] == 15
    assert audit["aggregate"]["false_positive"] == 5
    assert audit["aggregate"]["false_negative"] == 108
    assert audit["manual_quality_review"]["semantic_payload_correctness"] == {
        "CORRECT": 10,
        "PARTIAL": 5,
    }
    assert audit["manual_quality_review"]["scope_correctness"] == {
        "EXACT_MATCH": 10,
        "TARGET_SET_MISMATCH": 5,
    }


def test_shared_context_reaudit_covers_all_23_and_includes_zero_count_categories() -> None:
    audit = _load(SHARED_AUDIT)
    assert audit["candidates_total"] == 23
    assert audit["replay_rerun"] is False
    required = {
        "CORRECT_SHARED_CONTEXT",
        "TRUE_SCOPE_UNDEREXPANSION",
        "TRUE_SCOPE_OVEREXPANSION",
        "TRUE_ATTACHMENT_AMBIGUITY",
        "SOURCE_SCOPE_DETECTOR_ERROR",
        "OUTPUT_PROJECTION_ERROR",
        "CANONICALIZATION_ERROR",
        "DETECTOR_LIMIT",
    }
    assert set(audit["classification_counts"]) == required
    assert sum(audit["classification_counts"].values()) == 23
    assert audit["classification_counts"]["CORRECT_SHARED_CONTEXT"] == 6


def test_audit_claim_ids_do_not_leak_into_runtime_detector_implementation() -> None:
    source = (ROOT / "apps/api/app/domain/truth_condition_context.py").read_text(encoding="utf-8")
    assert "01a0c84e-9da9-7c02-b932-d5b8aca2bd1e" not in source
    assert "01a0c3f5-e702-7de6-9b2d-95d70e3c98e0" not in source
