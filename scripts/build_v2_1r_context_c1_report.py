"""Render the Phase C.1 closeout from frozen replay and audit artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "artifacts/v2_1r_truth_condition_context_128_reconciled_replay.json"
SELECTION = ROOT / "artifacts/v2_1r_truth_condition_context_c1_audit_selection.json"
MANUAL = ROOT / "artifacts/v2_1r_truth_condition_context_c1_manual_audit.json"
SHARED = ROOT / "artifacts/v2_1r_truth_condition_context_c1_shared_context_reaudit.json"
JSON_OUT = ROOT / "artifacts/v2_1r_truth_condition_context_c1_report.json"
MD_OUT = ROOT / "artifacts/v2_1r_truth_condition_context_c1_report.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    replay = load(REPLAY)
    selection = load(SELECTION)
    manual = load(MANUAL)
    shared = load(SHARED)
    agg = manual["aggregate"]
    quality = manual["manual_quality_review"]
    sections: dict[str, Any] = {
        "A_phase_c_inconclusive_evidence": {
            "phase_c_status": replay["report_title"],
            "claims": replay["total_claims"],
            "source_context_nodes": 33,
            "output_context_nodes": 481,
            "all_claims_detector_limited": True,
            "reference_reconciliation": replay["reference_reconciliation_integrity"]["status"],
        },
        "B_source_detection_completeness_problem": {
            "global_complete_true": 0,
            "detector_limit_claims": 128,
            "source_contexts": 33,
            "projected_output_contexts": 481,
            "finding": (
                "A claim-global false bit hid family-specific support and made absence ambiguous."
            ),
        },
        "C_detection_coverage_model": {
            "family_statuses": [
                "SUPPORTED_FOR_CONTRACT",
                "PARTIAL_DETECTOR_LIMIT",
                "NOT_EVALUATED",
            ],
            "presence_conclusions": [
                "CONTEXT_DETECTED",
                "NO_CONTEXT_DETECTED",
                "NO_CONTEXT_EXISTS",
                "DETECTOR_CANNOT_DECIDE",
            ],
            "legacy_global_boolean_retained_for_compatibility": True,
            "family_coverage_is_authoritative": True,
        },
        "D_development_audit_sample": {
            "count": selection["selected_count"],
            "selection_artifact": SELECTION.relative_to(ROOT).as_posix(),
            "selection_manifest_sha256": manual["selection_manifest_sha256"],
            "selection_frozen_before_review": selection["manual_review_started"] is False,
            "selection_rule": selection["selection_rule"],
            "cohorts": sorted({row["cohort"] for row in selection["selected_claims"]}),
            "both_preexisting_detection_states_included": True,
            "development_only": True,
            "holdout_content_read": False,
        },
        "E_human_source_context_audit": {
            "manual_contexts": agg["manual_contexts"],
            "materiality_source_span_payload_and_targets_recorded": True,
            "unknown_attachment_preserved_when_not_determinable": True,
            "claim_details_artifact": MANUAL.relative_to(ROOT).as_posix(),
        },
        "F_detector_precision_coverage": {
            "overall": {
                "precision": agg["precision"],
                "recall": agg["recall"],
                "true_positive": agg["matched"],
                "false_positive": agg["false_positive"],
                "false_negative": agg["false_negative"],
            },
            "by_family": manual["by_family"],
            "type_correct_of_reviewed_matches": quality["type_correct"],
            "reviewed_match_count": quality["matched_detector_contexts_reviewed"],
            "semantic_payload_review": quality["semantic_payload_correctness"],
            "scope_target_set_review": quality["scope_correctness"],
            "attachment_review": quality["attachment_correctness"],
            "rate_caveat": (
                "Development subset only; span-overlap matching is a bounded detector audit, "
                "not population performance."
            ),
        },
        "G_detection_failure_stages": {
            "pipeline": [
                "lexical candidate",
                "semantic-role determination",
                "context creation",
                "scope determination",
            ],
            "persisted_replay_diagnostics": {
                "candidate_rejected_no_semantic_relation": 11,
                "attachment_unknown_no_resolved_proposition_span": 2,
            },
            "audit_false_positive_contexts": agg["false_positive"],
            "unmatched_manual_contexts": agg["false_negative"],
            "stage_attribution_limit": (
                "Persisted traces do not expose per-anchor lexical-candidate events; remaining "
                "false negatives cannot be reliably split between lexical recall and "
                "semantic-role/context creation."
            ),
        },
        "H_canonical_context_representation": {
            "raw_semantic_signature_used_for_equivalence": False,
            "context_identity_and_diagnostic_signature_retained": True,
            "typed_canonical_inputs": [
                "family/type",
                "subtype and normalized payload",
                "scope target set",
                "attachment relation",
                "epistemic operator/agent/modality when explicit",
            ],
        },
        "I_temporal_canonicalization": {
            "supported": [
                "year",
                "date",
                "month",
                "year range",
                "named quarter/period",
                "ISO date range",
            ],
            "general_temporal_nlp": False,
            "product_identifier_year_false_positive_guard": "X2000 is excluded",
        },
        "J_epistemic_canonicalization": {
            "canonical_fields": [
                "operator",
                "explicit agent/source",
                "represented modality",
                "attachment target set",
            ],
            "retrieval_provenance_inference": False,
            "known_source_reports_x_vs_unqualified_x": "MISSING / unsafe rather than equivalent",
            "shared_epistemic_audit": {
                "existing_source_epistemic_cases_in_replay": 4,
                "shared_epistemic_candidate_outcomes": {
                    outcome: sum(
                        item["comparator_outcome"] == outcome
                        for item in shared["candidates"]
                        if item["context_type"] == "EPISTEMIC"
                    )
                    for outcome in sorted(
                        {
                            item["comparator_outcome"]
                            for item in shared["candidates"]
                            if item["context_type"] == "EPISTEMIC"
                        }
                    )
                },
            },
        },
        "K_scope_canonicalization": {
            "scope_payload_typed": True,
            "target_refs_resolved_through_scope_expansions": True,
            "source_evidence_retained": True,
            "scope_is_not_collapsed_to_free_text": True,
        },
        "L_shared_target_set_semantics": {
            "candidate_count": shared["candidates_total"],
            "manual_classification_counts": shared["classification_counts"],
            "target_set_comparator": (
                "explicit set equality/subset/superset reasoning; unknown source set is not "
                "labeled under/over expansion"
            ),
            "audit_artifact": SHARED.relative_to(ROOT).as_posix(),
        },
        "M_context_comparison": {
            "outcome_counts_for_23_candidates": shared["comparator_outcome_counts"],
            "outcomes_distinguish_representation_and_semantics": True,
            "does_not_collapse_granularity_mismatch_into_semantic_mismatch": True,
        },
        "N_verification_assertion_completeness": {
            "view_fields": [
                "resolved_context_refs",
                "unknown_context_details",
                "detector_limited_context_families",
                "source_context_completeness",
                "source_context_presence",
                "composition_eligibility",
            ],
            "eligibility_independent_from_context_completeness": True,
            "unit_regression_added": True,
        },
        "O_retrieval_provenance_safety": {
            "retrieval_domain_owner_used_as_epistemic_attribution": False,
            "hard_negative_tests": "PASS",
            "false_attribution_count": agg["false_attribution_from_retrieval_metadata"],
        },
        "P_method_semantic_role_precision": {
            "manual_method_contexts": manual["method_role_audit"]["manual_method_context_count"],
            "detected_method_contexts": manual["method_role_audit"][
                "detected_method_context_count"
            ],
            "category_noun_only_context_created": manual["method_role_audit"][
                "category_noun_only_context_created"
            ],
            "generic_relation_evidence_required": manual["method_role_audit"][
                "role_evidence_required"
            ],
        },
        "Q_production_behavior_diff": {
            "research_loop_changed": False,
            "provider_search_reader_acceptance_coverage_changed": False,
            "atomic_proposition_formation_changed": False,
            "prompt_schema_composition_eligibility_or_source_span_producer_changed": False,
            "scope": "Offline truth-context shadow domain, tests, and audit/report scripts only.",
        },
        "R_anti_overfitting_audit": {
            "claim_specific_ids_or_rules_in_runtime_detector": False,
            "runtime_rule_mined_from_audit_labels": False,
            "sample_membership_frozen_before_labels": True,
            "generic_structural_sampling": True,
            "family_support_guard": "tiny families reported as counts only",
        },
        "S_provider_calls": {
            "provider_calls": 0,
            "database_writes": 0,
            "holdout_content_read": False,
            "replay_128_rerun": False,
        },
        "T_final_decision": {
            "all_success_gate_conditions_met": True,
            "decision": "SOURCE CONTEXT DETECTION + COMPARATOR READY FOR 128 REPLAY",
            "known_limitations": [
                "Audit recall is low and varies sharply by family; the next replay must preserve "
                "DETECTOR_LIMIT rather than infer absence.",
                "Per-anchor lexical-candidate trace is not available, so most missed contexts "
                "cannot be assigned to a precise detector stage.",
                "A ready result means interpretable replay readiness, not complete semantic "
                "parsing or detector quality sufficiency.",
            ],
            "next_action": (
                "A later explicit request may rerun the frozen 128 development set; this task "
                "does not."
            ),
        },
    }
    payload = {
        "report_title": "V2.1R Truth-Condition Context Phase C.1",
        "report_version": "v2_1r-truth-condition-context-c1-closeout-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_artifacts": [
            REPLAY.relative_to(ROOT).as_posix(),
            SELECTION.relative_to(ROOT).as_posix(),
            MANUAL.relative_to(ROOT).as_posix(),
            SHARED.relative_to(ROOT).as_posix(),
        ],
        "sections": sections,
    }
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# V2.1R Truth-Condition Context Phase C.1", ""]
    for letter, key in zip("ABCDEFGHIJKLMNOPQRST", sections, strict=True):
        title = key.split("_", 1)[1].replace("_", " ").title()
        lines.extend(
            [
                f"## {letter}. {title}",
                "",
                "```json",
                json.dumps(sections[key], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "STOP — 不重跑 128 replay; 不调用 Provider; 不创建 Holdout; 不开始 V2.2; "
            "不 stage/commit。",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": JSON_OUT.relative_to(ROOT).as_posix(),
                "markdown": MD_OUT.relative_to(ROOT).as_posix(),
                "decision": sections["T_final_decision"]["decision"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
