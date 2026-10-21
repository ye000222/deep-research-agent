"""Offline replay #2 of the frozen 128 Phase C development outputs."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import replay_v2_1r_truth_condition_context as phase_c
from app.domain import truth_condition_context as tcc
from truth_condition_context_replay2_view_adapter import normalize_verification_view

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "artifacts/v2_1r_truth_condition_context_phase_c_replay2_freeze.json"
JSON_OUT = ROOT / "artifacts/v2_1r_truth_condition_context_128_replay2.json"
MD_OUT = ROOT / "artifacts/v2_1r_truth_condition_context_128_replay2.md"
HUMAN_REVIEW = ROOT / "artifacts/v2_1r_truth_condition_context_128_replay2_manual_review.json"
COMPAT_REPAIR = (
    ROOT / "artifacts/v2_1r_truth_condition_context_phase_c_replay2_compat_repair_v2.json"
)
PREVIOUS_COMPAT_REPAIR = (
    ROOT / "artifacts/v2_1r_truth_condition_context_phase_c_replay2_compat_repair.json"
)
COMPAT_AUDIT_OUT = (
    ROOT / "artifacts/v2_1r_truth_condition_context_phase_c_replay2_compat_audit_v2.json"
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_frozen_state() -> dict[str, Any]:
    frozen = phase_c.load_json(FREEZE)
    repair = phase_c.load_json(COMPAT_REPAIR)
    expected_repair_files = {
        "scripts/replay_v2_1r_truth_condition_context_phase_c_replay2.py",
        "scripts/truth_condition_context_replay2_view_adapter.py",
        "tests/unit/test_v2_1r_truth_condition_context_replay2.py",
    }
    allowed_repair_files = set(repair["analyzer_only_files"])
    current_source = {relative: _hash(ROOT / relative) for relative in frozen["source_file_sha256"]}
    semantic_source_paths = set(frozen["source_file_sha256"]) - allowed_repair_files
    current_inputs = {
        key: _hash(path) for key, path in sorted(phase_c.input_artifact_paths().items())
    }
    extra_paths = {
        "c1_selection": ROOT / "artifacts/v2_1r_truth_condition_context_c1_audit_selection.json",
        "c1_manual_audit": ROOT / "artifacts/v2_1r_truth_condition_context_c1_manual_audit.json",
        "c1_shared_reaudit": ROOT
        / "artifacts/v2_1r_truth_condition_context_c1_shared_context_reaudit.json",
        "c1_closeout": ROOT / "artifacts/v2_1r_truth_condition_context_c1_report.json",
        "phase_c_manual_review": ROOT
        / "artifacts/v2_1r_truth_condition_context_128_manual_review.md",
    }
    current_reviews = {key: _hash(path) for key, path in sorted(extra_paths.items())}
    current_repair_files = {
        relative: _hash(ROOT / relative) for relative in sorted(allowed_repair_files)
    }
    checks = {
        "semantic_source_files": all(
            current_source[path] == frozen["source_file_sha256"][path]
            for path in semantic_source_paths
        ),
        "analyzer_only_repair_files": current_repair_files == repair["repaired_file_sha256"],
        "compat_repair_freeze_link": repair["base_freeze_sha256"] == _hash(FREEZE),
        "previous_compat_repair_link": repair["previous_compat_repair_sha256"]
        == _hash(PREVIOUS_COMPAT_REPAIR),
        "inputs_128": current_inputs == frozen["input_artifact_sha256"],
        "review_protocol": current_reviews == frozen["review_protocol_artifact_sha256"],
        "repair_scope": repair["semantic_behavior_change"] is False,
        "repair_allowlist_exact": allowed_repair_files == expected_repair_files,
        "original_analyzer_hashes_attested": all(
            repair["original_frozen_analyzer_sha256"].get(path)
            == frozen["source_file_sha256"].get(path)
            for path in expected_repair_files
            if path in frozen["source_file_sha256"]
        ),
        "repair_attestation_tool": repair["attestation_tool_sha256"]
        == _hash(ROOT / repair["attestation_tool"]),
    }
    if not all(checks.values()):
        raise SystemExit(f"freeze verification failed; replay stopped: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "freeze": FREEZE.relative_to(ROOT).as_posix(),
        "compatibility_repair": COMPAT_REPAIR.relative_to(ROOT).as_posix(),
    }


def pre_replay_compatibility_audit(cohorts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    variants: Counter[str] = Counter()
    diagnostics: Counter[str] = Counter()
    eligibility_input: Counter[str] = Counter()
    eligibility_output: Counter[str] = Counter()
    current_views = 0
    current_by_cohort: Counter[str] = Counter()
    current_failures: list[dict[str, str]] = []
    for cohort_name, rows in cohorts.items():
        for row in rows:
            source_graph = tcc.project_source_context_graph(
                row["claim_text"],
                row["propositions"],
                claim_ref=row["claim_id"],
                composition_groups=row["composition_groups"],
            )
            output_graph = tcc.project_output_context_graph(
                row["propositions"], claim_ref=row["claim_id"]
            )
            preservation = tcc.compare_context_graphs(source_graph, output_graph)
            views = tcc.build_verification_assertion_views(
                row["propositions"],
                output_graph,
                eligibility_by_ref=row["eligibility_by_ref"],
                source_graph=source_graph,
            )
            output_by_ref = {context.context_ref: context for context in output_graph.contexts}
            for view in views:
                current_views += 1
                current_by_cohort[cohort_name] += 1
                native = view.model_dump(mode="json")
                resolved = [
                    output_by_ref[ref].model_dump(mode="json")
                    for ref in view.resolved_context_refs
                    if ref in output_by_ref
                ]
                legacy = phase_c.supplemental_view(view, source_graph, output_graph, preservation)
                payload = {
                    **native,
                    **legacy,
                    "resolved_material_contexts": resolved,
                    "epistemic_contexts": [
                        context for context in resolved if context["context_type"] == "EPISTEMIC"
                    ],
                }
                normalized = normalize_verification_view(
                    payload,
                    replay_preservation_completeness=legacy.get("context_completeness"),
                )
                variants["CURRENT_VIEW_NATIVE"] += 1
                diagnostics.update(
                    diagnostic
                    for diagnostic in normalized.compatibility_diagnostics
                    if diagnostic.startswith(
                        ("VIEW_FIELD", "VIEW_SCHEMA", "COMPLETENESS_FIELD", "VIEW_FIELD_INVALID")
                    )
                )
                eligibility_input[view.composition_eligibility] += 1
                eligibility_output[normalized.composition_eligibility] += 1
                if (
                    normalized.preservation_completeness is None
                    or normalized.source_context_completeness == "UNRESOLVED"
                    or normalized.material_context_unknown is None
                    or normalized.detector_limited_families is None
                    or normalized.resolved_material_contexts is None
                    or normalized.epistemic_contexts is None
                    or "VIEW_SCHEMA_VERSION_UNKNOWN" in normalized.compatibility_diagnostics
                ):
                    current_failures.append(
                        {
                            "cohort": cohort_name,
                            "claim_id": row["claim_id"],
                            "proposition_ref": view.proposition_ref,
                            "diagnostics": ",".join(normalized.compatibility_diagnostics),
                        }
                    )

    historical = phase_c.load_json(
        ROOT / "artifacts/v2_1r_truth_condition_context_128_reconciled_replay.json"
    )
    legacy_adapted = 0
    legacy_by_cohort: Counter[str] = Counter()
    legacy_missing: Counter[str] = Counter()
    for claim in historical["claim_replays"]:
        cohort_name = claim.get("cohort", "UNKNOWN_COHORT")
        for raw in claim.get("verification_assertion_views", []):
            normalized = normalize_verification_view(raw)
            if "LEGACY_VIEW_ADAPTED" in normalized.compatibility_diagnostics:
                legacy_adapted += 1
                legacy_by_cohort[cohort_name] += 1
            legacy_missing.update(normalized.compatibility_diagnostics)

    expected_current_views = sum(
        len(row["propositions"]) for rows in cohorts.values() for row in rows
    )
    eligibility_preserved = eligibility_input == eligibility_output
    current_integrity = current_views == expected_current_views and not current_failures
    return {
        "passed": current_integrity and eligibility_preserved,
        "cohorts": {name: len(rows) for name, rows in cohorts.items()},
        "views_total_current": current_views,
        "expected_views_current": expected_current_views,
        "historical_persisted_views_audited": sum(
            len(claim.get("verification_assertion_views", []))
            for claim in historical["claim_replays"]
        ),
        "view_layout_variants": {
            **dict(sorted(variants.items())),
            "HISTORICAL_LEGACY_ADAPTED": legacy_adapted,
        },
        "native_current_views": current_views,
        "legacy_adapted_views": legacy_adapted,
        "per_cohort_view_counts": {
            cohort: {
                "current_native_views": current_by_cohort.get(cohort, 0),
                "historical_legacy_adapted_views": legacy_by_cohort.get(cohort, 0),
            }
            for cohort in cohorts
        },
        "missing_required_fields_current": dict(sorted(diagnostics.items())),
        "historical_legacy_diagnostics": dict(sorted(legacy_missing.items())),
        "unresolved_current_compatibility_cases": len(current_failures),
        "adapter_failures": current_failures,
        "eligibility_preserved_independently": eligibility_preserved,
        "eligibility_counts": dict(sorted(eligibility_output.items())),
        "legacy_source_completeness_remains_unresolved_by_design": legacy_adapted > 0,
    }


def _expanded_targets(context: Any, graph: Any) -> list[str]:
    targets: list[str] = []
    for edge in context.attachments:
        if edge.target_ref:
            targets.extend(graph.scope_expansions.get(edge.target_ref, (edge.target_ref,)))
    if not targets:
        targets.extend(context.scope.target_refs)
    return sorted(set(targets))


def build_view_rows(claim: dict[str, Any], input_row: dict[str, Any]) -> list[dict[str, Any]]:
    source_graph = tcc.ContextGraph.model_validate(claim["source_context_graph"])
    output_graph = tcc.ContextGraph.model_validate(claim["output_context_graph"])
    eligibility = {
        item["local_ref"]: item.get("eligibility", "UNKNOWN") for item in input_row["propositions"]
    }
    views = tcc.build_verification_assertion_views(
        input_row["propositions"],
        output_graph,
        eligibility_by_ref=eligibility,
        source_graph=source_graph,
    )
    preservation_results = tcc.compare_context_graphs(source_graph, output_graph)
    output_by_ref = {context.context_ref: context for context in output_graph.contexts}
    result = []
    for view in views:
        native_view = view.model_dump(mode="json")
        resolved = [
            output_by_ref[ref].model_dump(mode="json")
            for ref in view.resolved_context_refs
            if ref in output_by_ref
        ]
        legacy_analysis = phase_c.supplemental_view(
            view, source_graph, output_graph, preservation_results
        )
        analyzer_payload = {
            **native_view,
            **legacy_analysis,
            "resolved_material_contexts": resolved,
            "epistemic_contexts": [
                context for context in resolved if context["context_type"] == "EPISTEMIC"
            ],
        }
        normalized = normalize_verification_view(
            analyzer_payload,
            replay_preservation_completeness=legacy_analysis.get("context_completeness"),
        )
        view_dict = normalized.to_summary_payload()
        view_dict["material_unknown_contexts"] = list(view.unknown_context_details)
        limited = list(normalized.detector_limited_families or ())
        view_dict["detector_limit_impact"] = {
            "limited_family_count": len(limited),
            "limited_family_cardinality": (
                "NONE"
                if not limited
                else "ONE_FAMILY"
                if len(limited) == 1
                else "MULTIPLE_FAMILIES"
            ),
            "affirmative_future_semantic_comparison_safe": (
                normalized.source_context_completeness == "SUPPORTED_FOR_CONTRACT"
                and not limited
                and normalized.material_context_unknown is False
            ),
            "compatibility_diagnostics": list(normalized.compatibility_diagnostics),
        }
        result.append(view_dict)
    return result


def _coverage_rows(claims: list[dict[str, Any]]) -> dict[str, Any]:
    contexts_by_family: Counter[str] = Counter()
    coverage_by_family: Counter[str] = Counter()
    presence: Counter[str] = Counter()
    for claim in claims:
        graph = claim["source_context_graph"]
        presence[graph["context_presence"]] += 1
        contexts_by_family.update(item["context_type"] for item in graph["contexts"])
        coverage_by_family.update(
            f"{family}:{status}" for family, status in graph["source_detection_coverage"].items()
        )
    manual = phase_c.load_json(
        ROOT / "artifacts/v2_1r_truth_condition_context_c1_manual_audit.json"
    )
    return {
        "presence_conclusions_by_claim": dict(sorted(presence.items())),
        "detected_contexts_by_family": dict(sorted(contexts_by_family.items())),
        "coverage_status_claim_counts_by_family": dict(sorted(coverage_by_family.items())),
        "development_manual_evidence_by_family": manual["by_family"],
        "non_detection_is_not_absence": True,
    }


def _view_impact(claims: list[dict[str, Any]]) -> dict[str, Any]:
    all_views = [
        {"cohort": claim["cohort"], "claim_id": claim["claim_id"], **view}
        for claim in claims
        for view in claim["verification_assertion_views"]
    ]
    cardinality = Counter(
        view["detector_limit_impact"]["limited_family_cardinality"] for view in all_views
    )
    unsafe = [
        view
        for view in all_views
        if not view["detector_limit_impact"]["affirmative_future_semantic_comparison_safe"]
    ]
    return {
        "views_total": len(all_views),
        "views_no_known_material_detector_limit": cardinality.get("NONE", 0),
        "views_one_limited_family": cardinality.get("ONE_FAMILY", 0),
        "views_multiple_limited_families": cardinality.get("MULTIPLE_FAMILIES", 0),
        "views_affirmative_comparison_unsafe_due_to_detector_limits_or_unknown": len(unsafe),
        "context_completeness_by_view": dict(
            sorted(Counter(view["source_context_completeness"] for view in all_views).items())
        ),
        "eligibility_by_view": dict(
            sorted(Counter(view["composition_eligibility"] for view in all_views).items())
        ),
        "per_view": all_views,
    }


def _shared_context_occurrences(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"cohort": claim["cohort"], "claim_id": claim["claim_id"], **item}
        for claim in claims
        for item in claim["shared_contexts"]
    ]


def _human_review_protocol(claims: list[dict[str, Any]]) -> dict[str, Any]:
    manual_rows = phase_c.load_json(
        ROOT / "artifacts/v2_1r_truth_condition_context_c1_shared_context_reaudit.json"
    )["candidates"]
    manual_by_key = {(row["claim_id"], row["context_ref"]): row for row in manual_rows}
    required_shared = _shared_context_occurrences(claims)
    matched_manual = []
    unmatched = []
    claims_by_id = {claim["claim_id"]: claim for claim in claims}
    for item in required_shared:
        key = (item["claim_id"], item["context_ref"])
        previous = manual_by_key.get(key)
        claim = claims_by_id[key[0]]
        context = next(
            row for row in claim["source_context_graph"]["contexts"] if row["context_ref"] == key[1]
        )
        same_semantics = previous is not None and (
            previous.get("source_canonical_value")
            == tcc.canonical_context_value(tcc.TruthConditionContext.model_validate(context))
            and previous.get("source_target_set") == item["source_targets"]
            and previous.get("output_target_set") == item["output_targets"]
            and previous.get("comparator_outcome") == item["outcome"]
        )
        if not same_semantics:
            unmatched.append(
                {
                    "claim_id": key[0],
                    "context_ref": key[1],
                    "reason": (
                        "no prior review key"
                        if previous is None
                        else "prior key exists but semantic/target/outcome payload changed"
                    ),
                }
            )
        elif previous is not None:
            matched_manual.append(
                {
                    **item,
                    "manual_classification": previous["classification"],
                    "manual_rationale": previous["manual_rationale"],
                }
            )
    source_context_rows = _source_outcomes(claims)["material_source_contexts"]
    review_queue = []
    for row in source_context_rows:
        context = row["context"]
        comparison = row["comparison"]
        cohort_claim = claims_by_id[row["claim_id"]]
        shared = next(
            (
                item
                for item in cohort_claim["shared_contexts"]
                if item["context_ref"] == context["context_ref"]
            ),
            None,
        )
        reasons = []
        if comparison.get("outcome") != "PRESERVED":
            reasons.append("ALL_NON_PRESERVED_CONTEXTS_FALSE_BLOCK_REVIEW")
        if context["context_type"] == "EPISTEMIC":
            reasons.append("ALL_EPISTEMIC_CONTEXTS")
        if shared is not None:
            reasons.append("ALL_SHARED_CONTEXTS")
        suspicious = context["context_type"] in {
            "TEMPORAL",
            "VERSION",
            "CONDITION",
            "MEASUREMENT",
            "SAMPLE_SCOPE",
            "METHOD",
            "EPISTEMIC",
            "SCOPE",
            "CONFIGURATION",
            "SCENARIO",
        }
        if comparison.get("outcome") == "PRESERVED" and suspicious:
            reasons.append("SUSPICIOUS_PRESERVED")
        review_queue.append(
            {
                "cohort": row["cohort"],
                "claim_id": row["claim_id"],
                "claim_text": cohort_claim["claim_text"],
                "source_context": context,
                "comparison": comparison,
                "shared_context": shared,
                "required_review_reasons": reasons,
                "manual_disposition": "PENDING",
                "root_cause": "PENDING",
                "rationale": None,
            }
        )
    ordinary_preserved = [
        row
        for row in review_queue
        if row["comparison"].get("outcome") == "PRESERVED" and not row["required_review_reasons"]
    ]
    ordinary_preserved.sort(
        key=lambda row: hashlib.sha256(
            f"{row['claim_id']}:{row['source_context']['context_ref']}".encode()
        ).hexdigest()
    )
    for row in ordinary_preserved[: max(1, (len(ordinary_preserved) + 9) // 10)]:
        row["required_review_reasons"].append("DETERMINISTIC_ORDINARY_PRESERVED_SAMPLE")
    return {
        "c1_shared_review_recomputed_against_current_result": True,
        "shared_context_occurrences_current": len(required_shared),
        "exact_review_key_matches": len(matched_manual),
        "requires_new_manual_label": unmatched,
        "reviewed_shared_contexts": matched_manual,
        "shared_context_prior_labels_reused_only_when_payload_exact": True,
        "source_context_manual_review_queue": review_queue,
        "source_context_review_queue_count": len(review_queue),
        "manual_review_complete": False,
        "previous_phase_c_manual_review_is_reference_not_a_substitute": True,
    }


def _source_outcomes(claims: list[dict[str, Any]]) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    rows = []
    for claim in claims:
        result_by_ref = {item["source_context_ref"]: item for item in claim["context_preservation"]}
        for context in claim["source_context_graph"]["contexts"]:
            result = result_by_ref.get(context["context_ref"], {})
            counter[result.get("outcome", "UNMAPPED")] += 1
            rows.append(
                {
                    "cohort": claim["cohort"],
                    "claim_id": claim["claim_id"],
                    "context": context,
                    "comparison": result,
                }
            )
    return {"outcome_counts": dict(sorted(counter.items())), "material_source_contexts": rows}


def run() -> dict[str, Any]:
    freeze_check = verify_frozen_state()
    preflight = phase_c.load_json(COMPAT_AUDIT_OUT)
    if not preflight.get("passed"):
        raise SystemExit("pre-replay compatibility audit is not PASS; replay stopped")
    cohorts = phase_c.load_cohorts()
    integrity = phase_c.reference_integrity_audit(cohorts)
    if not integrity["passed"]:
        raise SystemExit("reference reconciliation failed; replay stopped before projection")
    compat_audit = preflight["view_compatibility_audit"]
    claims = []
    for _cohort, rows in cohorts.items():
        for row in rows:
            claim = phase_c.claim_replay(row)
            claim["verification_assertion_views"] = build_view_rows(claim, row)
            claims.append(claim)
    if len(claims) != 128 or len({row["claim_id"] for row in claims}) != 128:
        raise SystemExit("expected exactly 128 unique persisted development outputs")
    summaries = {
        name: phase_c.summarize_claims([row for row in claims if row["cohort"] == name])
        for name in cohorts
    }
    all_summary = phase_c.summarize_claims(claims)
    shared = [item for claim in claims for item in claim["shared_contexts"]]
    epistemic = [item for claim in claims for item in claim["epistemic_audit"]]
    temporal = [
        row
        for claim in claims
        for row in claim["source_context_graph"]["contexts"]
        if row["context_type"] == "TEMPORAL"
    ]
    method = [
        row
        for claim in claims
        for row in claim["source_context_graph"]["contexts"]
        if row["context_type"] == "METHOD"
    ]
    numeric = phase_c.numeric_contract_audit()
    data = {
        "report_title": "V2.1R Truth-Condition Context Phase C Replay #2",
        "replay_id": "v2_1r_truth_condition_context_phase_c_replay2_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "freeze_verification": freeze_check,
        "reference_reconciliation_integrity": integrity,
        "pre_replay_compatibility_audit": compat_audit,
        "dataset_governance": {
            "cohorts": {name: len(rows) for name, rows in cohorts.items()},
            "total_claims": len(claims),
            "unique_claim_ids": len({row["claim_id"] for row in claims}),
            "claim_ids_disjoint": True,
            "persisted_outputs_regenerated": False,
            "holdout_content_read": False,
            "provider_calls": 0,
            "database_read_or_write": False,
        },
        "source_detection_coverage": _coverage_rows(claims),
        "detector_limit_impact": _view_impact(claims),
        "preservation": _source_outcomes(claims),
        "shared_contexts": shared,
        "shared_context_outcome_counts": dict(
            sorted(Counter(item["outcome"] for item in shared).items())
        ),
        "shared_context_classification_counts": dict(
            sorted(Counter(item["classification"] for item in shared).items())
        ),
        "epistemic": epistemic,
        "epistemic_outcome_counts": dict(
            sorted(Counter(item["outcome"] for item in epistemic).items())
        ),
        "temporal": temporal,
        "temporal_outcome_counts": dict(
            sorted(
                Counter(
                    result["outcome"]
                    for claim in claims
                    for result in claim["context_preservation"]
                    if any(
                        context["context_ref"] == result["source_context_ref"]
                        and context["context_type"] == "TEMPORAL"
                        for context in claim["source_context_graph"]["contexts"]
                    )
                ).items()
            )
        ),
        "method_contexts": method,
        "method_candidate_rejected_count": sum(
            diagnostic == "candidate_rejected_no_semantic_relation"
            for claim in claims
            for diagnostic in claim["source_projection_diagnostics"]
        ),
        "retrieval_provenance_safety": {
            "source_records_with_retrieval_provenance": all_summary[
                "retrieval_provenance_source_records"
            ],
            "retrieval_fields_forwarded_to_source_projector": all_summary[
                "retrieval_provenance_metadata_claim_count"
            ],
            "retrieval_provenance_as_attribution": 0,
            "hard_negative_test_expected": True,
        },
        "numeric_normalization": numeric,
        "non_context_failures": phase_c.non_context_failures(cohorts),
        "cohort_summaries": summaries,
        "combined_summary": all_summary,
        "manual_review_protocol": _human_review_protocol(claims),
        "claims": claims,
    }
    if HUMAN_REVIEW.exists() or MD_OUT.exists():
        raise SystemExit("refusing to overwrite pre-existing Replay #2 review/report artifact")
    JSON_OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    HUMAN_REVIEW.write_text(
        json.dumps(data["manual_review_protocol"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "replay": data["replay_id"],
                "claims": len(claims),
                "propositions": all_summary["propositions"],
                "views": data["detector_limit_impact"]["views_total"],
                "json": JSON_OUT.relative_to(ROOT).as_posix(),
                "provider_calls": 0,
            },
            ensure_ascii=False,
        )
    )
    return data


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--preflight", action="store_true", help="audit frozen view layouts before replay"
    )
    action.add_argument(
        "--execute", action="store_true", help="execute the one frozen offline replay"
    )
    args = parser.parse_args()
    if args.preflight:
        if COMPAT_AUDIT_OUT.exists():
            raise SystemExit(f"refusing to overwrite pre-replay audit: {COMPAT_AUDIT_OUT}")
        freeze_check = verify_frozen_state()
        cohorts = phase_c.load_cohorts()
        integrity = phase_c.reference_integrity_audit(cohorts)
        audit = pre_replay_compatibility_audit(cohorts)
        preflight = {
            "freeze_verification": freeze_check,
            "reference_reconciliation_integrity": integrity,
            "view_compatibility_audit": audit,
            "passed": bool(freeze_check["passed"] and integrity["passed"] and audit["passed"]),
            "provider_calls": 0,
            "holdout_content_read": False,
        }
        COMPAT_AUDIT_OUT.write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(preflight, ensure_ascii=False))
        if not preflight["passed"]:
            raise SystemExit("pre-replay integrity gate failed; full replay was not run")
        return
    if JSON_OUT.exists():
        raise SystemExit(f"refusing to overwrite replay2 output: {JSON_OUT}")
    if HUMAN_REVIEW.exists():
        raise SystemExit(f"refusing to overwrite replay2 review queue: {HUMAN_REVIEW}")
    if not COMPAT_AUDIT_OUT.is_file():
        raise SystemExit("run --preflight and review its PASS before --execute")
    run()


if __name__ == "__main__":
    main()
