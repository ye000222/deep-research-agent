from __future__ import annotations

import json
from pathlib import Path

ARTIFACT_ROOT = Path("artifacts/v2_1r_grounding_requirement_d3_4")


def test_d3_4_closure_replay_artifacts_are_consistent_and_offline() -> None:
    report = json.loads((ARTIFACT_ROOT / "closure_replay.json").read_text(encoding="utf-8"))
    manifest = json.loads((ARTIFACT_ROOT / "hash_manifest.json").read_text(encoding="utf-8"))
    nodes = json.loads((ARTIFACT_ROOT / "node_ledger.json").read_text(encoding="utf-8"))["rows"]
    obligations = json.loads(
        (ARTIFACT_ROOT / "obligation_ledger.json").read_text(encoding="utf-8")
    )["rows"]

    assert report["claims"] == 8
    assert report["nodes"] == len(nodes) == 72
    assert report["bindings"] == 75
    assert (
        report["contract_version"]
        == manifest["requirement_contract_version"]
        == "grounding-requirements.generic.v2"
    )
    assert (
        report["matcher_version"] == manifest["matcher_version"] == "grounding-matcher.generic.v2"
    )
    assert report["attachment"]["target_applicability_missing_count"] == 1
    assert report["attachment"]["c6_at1_true_omission_remains_missing"] is True
    assert report["method"]["method_relation_obligation_count"] == 0
    assert report["method"]["duplicate_relational_obligation_count"] == 0
    assert report["safety_counters"] == {
        "unsafe_promotion": 0,
        "unsafe_failure_masking": 0,
        "unsafe_vacuous_promotion": 0,
        "undeclared_semantic_reuse": 0,
        "provenance_defects": 0,
        "requiredness_stability_violations": 0,
    }
    assert report["scope_boundary"]["provider_calls"] == 0
    assert report["scope_boundary"]["external_http"] == 0
    assert report["scope_boundary"]["semantic_verifier_calls"] == 0
    assert report["scope_boundary"]["extractor_calls"] == 0
    assert report["scope_boundary"]["holdout_accessed"] is False
    assert report["comparator_reporting_fix"]["legacy_false_positive_discrepancies_removed"] == 18
    assert len({(row["claim_id"], row["node_ref"]) for row in nodes}) == 72
    assert len(
        {(row["claim_id"], row["node_ref"], row["obligation_ref"]) for row in obligations}
    ) == len(obligations)
    assert all(
        row["provenance"].get("matcher_version") == report["matcher_version"] for row in obligations
    )
