from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"


def _load(name: str) -> dict[str, Any]:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def _claim_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        found.add(value)
    elif isinstance(value, list):
        for item in value:
            found.update(_claim_ids(item))
    elif isinstance(value, dict):
        if isinstance(value.get("claim_id"), str):
            found.add(value["claim_id"])
        for key in (
            "claim_replays",
            "claims",
            "claim_ids",
            "selected_claim_ids",
            "selected_claims",
            "strata",
            "excluded_claim_ids",
        ):
            if key in value:
                found.update(_claim_ids(value[key]))
    return found


def test_d4a_selection_is_unique_hashed_and_development_only() -> None:
    manifest = _load("v2_1r_d4a_selection_manifest.json")
    source = _load("v2_1_dev_extension_v2_candidates.json")
    claims = manifest["claims"]
    assert manifest["development_only"] is True
    assert manifest["claim_text_in_manifest"] is False
    assert manifest["human_semantic_labels_in_selection"] is False
    assert len(claims) == 8
    assert len({item["claim_id"] for item in claims}) == 8
    assert len({item["source_pair_id"] for item in claims}) == 8
    assert all(len(item["claim_sha256"]) == 64 for item in claims)

    source_text: dict[str, str] = {}
    for pair in source["pairs"]:
        source_text[pair["claim_a_id"]] = pair["claim_a_text"]
        source_text[pair["claim_b_id"]] = pair["claim_b_text"]
    for item in claims:
        assert item["claim_id"] in source_text
        digest = hashlib.sha256(source_text[item["claim_id"]].encode("utf-8")).hexdigest()
        assert item["claim_sha256"] == digest


def test_d4a_selection_has_no_overlap_with_frozen_nonholdout_cohorts() -> None:
    manifest = _load("v2_1r_d4a_selection_manifest.json")
    selected = {item["claim_id"] for item in manifest["claims"]}
    exclusion = _load("v2_1r_d4a_exclusion_audit.json")
    assert all(count == 0 for count in exclusion["selected_overlap_by_cohort"].values())

    cohorts = [
        ("v2_1r_truth_condition_context_128_reconciled_replay.json", "claim_replays"),
        ("v2_1r_source_context_extraction_40_eval_manifest.json", "claims"),
        ("v2_1r_source_context_extraction_real_canary_manifest.json", "claim_ids"),
        (
            "v2_1r_source_semantic_grounding_evidence_v2_real_development_canary_selection_manifest.json",
            "selection",
        ),
        ("v2_1r_bounded_shadow_100_selection_manifest.json", "selected_claims"),
        (
            "v2_1r_qualifier_semantic_preservation_fresh_validation_12_selection_manifest.json",
            "selected_claims",
        ),
        ("v2_1r_final_fresh_qualifier_check_selection_manifest.json", "selected_claims"),
        ("v2_1r_final_fresh_mini_shadow_3/pre_provider_manifest.json", "selected_claim_ids"),
        ("v2_1r_final_fresh_mini_shadow_4/mini_shadow_4_execution_summary.json", "claims"),
        ("v2_1r_fresh_mini_shadow_2_selection.json", "strata"),
        ("v2_1r_mini_shadow_selection.json", "strata"),
    ]
    for filename, key in cohorts:
        artifact = _load(filename)
        assert not selected.intersection(_claim_ids(artifact.get(key)))

    v2_manifest = _load(
        "v2_1r_source_semantic_grounding_evidence_v2_real_development_canary_selection_manifest.json"
    )
    assert not selected.intersection(
        _claim_ids(v2_manifest.get("cohort_exclusions", {}).get("final_unseen_16", {}))
    )

    # Holdout content/labels are intentionally not loaded by this test.
    assert exclusion["holdout_content_or_labels_read"] is False
    assert exclusion["source_pool_frozen_holdout_claim_id_overlap"] == 0


def test_d4a_normalized_manifest_and_component_hashes_match() -> None:
    manifest = _load("v2_1r_d4a_selection_manifest.json")
    recorded = manifest["normalized_manifest_sha256"]
    normalized = dict(manifest)
    normalized.pop("normalized_manifest_sha256")
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == recorded

    freeze = _load("v2_1r_d4a_semantic_freeze.json")
    for relative, expected in freeze["components"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected


def test_d4a_result_schema_contract_is_versioned_and_complete() -> None:
    schema = _load("v2_1r_d4a_result_schema.json")
    required = set(schema["required"])
    properties = schema["properties"]
    assert schema["schema_version"] == "v2_1r_hybrid_canary_result.v1"
    assert required <= properties.keys()
    assert {
        "claim_id",
        "claim_sha256",
        "extractor",
        "source_graph_counts",
        "grounding",
        "requirements",
        "hybrid_support",
        "calls",
        "provider",
        "components",
    } <= required
    assert all(metric in schema["safety_metrics_required"] for metric in (
        "fabricated_source_span_count",
        "unsupported_semantic_promoted_count",
        "unsafe_node_promotion_count",
        "provenance_defect_count",
    ))
