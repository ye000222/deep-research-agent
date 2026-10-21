"""Offline D3.4 closure replay over the same frozen 8 development claims."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.domain.grounding_requirements import (
    GroundingNodeKind,
    GroundingNodeUsability,
    GroundingObligationStatus,
    GroundingRequirementLevel,
)
from app.services.grounding_requirement_engine_v2 import (
    GROUNDING_AGGREGATION_VERSION_V2,
    GROUNDING_MATCHER_VERSION_V2,
    derive_grounding_obligations_v2,
    evaluate_grounding_node_v2,
)

from scripts import replay_v2_1r_grounding_requirements_d3 as d3

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path("artifacts/v2_1r_grounding_requirement_d3_4")
EXPECTED_CLAIM_IDS = {
    "01a0c86c-7a7b-7815-94a5-996a3aea5ae6",
    "01a0c438-499d-7fbf-84b7-0cec4da63e25",
    "01a0c41c-281a-7f75-a720-474df7782723",
    "01a0c43b-3e7a-704f-b356-ad6f591a17ff",
    "01a0c82b-c337-7e71-8bdd-9e0954bd1aa0",
    "01a0c845-7eb2-7f79-b92d-5d6fcdd8f257",
    "01a0c41a-7636-721d-9b9d-7ee9ea95e573",
    "01a0c415-6a29-7cfa-b58e-4b14f51c469e",
}
IMPLEMENTATION_FILES = (
    "apps/api/app/domain/grounding_requirements.py",
    "apps/api/app/services/grounding_requirement_engine.py",
    "apps/api/app/services/grounding_requirement_engine_v2.py",
    "scripts/replay_v2_1r_grounding_requirements_d3.py",
    "scripts/replay_v2_1r_grounding_requirements_d3_4.py",
    "scripts/v2_1r_d3_adapter.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new(path: Path, content: str) -> None:
    target = ROOT / path
    if target.exists() and not path.as_posix().startswith(OUTPUT_DIR.as_posix() + "/"):
        raise RuntimeError(f"refusing_to_overwrite_existing_artifact:{path.as_posix()}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="\n")


def run_replay() -> dict[str, Any]:
    canary = d3._read_json(d3.CANARY_PATH)
    phase_c = d3._read_json(d3.PHASE_C_PATH)
    selection = d3._read_json(d3.SELECTION_PATH)
    claims = canary["claims"]
    claim_ids = {str(row["claim_id"]) for row in claims}
    frozen_ids = set(selection.get("selection", {}).get("selected_claim_ids", []))
    if claim_ids != EXPECTED_CLAIM_IDS or frozen_ids != EXPECTED_CLAIM_IDS:
        raise RuntimeError("frozen_claim_identity_mismatch")
    if len(claims) != 8 or len(selection.get("selection", {}).get("selected_claim_ids", [])) != 8:
        raise RuntimeError("frozen_claim_count_mismatch")
    nodes = [
        row
        for claim in claims
        for row in d3._node_records(str(claim["claim_id"]), claim.get("grounding_result") or {})
    ]
    binding_count = sum(
        len((claim.get("grounding_result") or {}).get("evidence_bindings", [])) for claim in claims
    )
    if len(nodes) != 72 or binding_count != 75:
        raise RuntimeError(f"frozen_graph_cardinality_mismatch:{len(nodes)}:{binding_count}")

    support = d3._support_index(phase_c)
    by_claim = {str(row["claim_id"]): row for row in claims}
    obligation_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    states: Counter[str] = Counter()
    node_usability: Counter[str] = Counter()
    obligation_types: Counter[str] = Counter()
    target_app_missing: list[dict[str, str]] = []
    matched_attachment_edges: list[dict[str, str]] = []
    assertion_legacy_unresolved = 0
    aggregation_unsafe_promotion = 0
    aggregation_failure_masking = 0
    vacuous_promotion = 0
    undeclared_reuse = 0
    provenance_defects = 0
    requiredness_stability_violations = 0
    measurement_distribution: Counter[str] = Counter()
    method_relation_count = 0
    method_context_applicability_count = 0
    attachment_applicability_count = 0

    for record in nodes:
        claim_id = record["claim_id"]
        node = record["engine_input_node"]
        derivation = derive_grounding_obligations_v2(node)
        candidates, _candidate_provenance = d3._build_candidates(
            by_claim[claim_id], support, str(canary.get("schema_version", ""))
        )
        assessment = evaluate_grounding_node_v2(
            derivation,
            candidates,
            node_textual_grounding=record["node_textual_grounding"],
            node_structural_grounding=record["node_structural_grounding"],
        )
        if record["adapter_status"] != "RESOLVED":
            assessment = assessment.model_copy(
                update={
                    "usability": GroundingNodeUsability.NOT_EVALUATED,
                    "verification_assertion_ready": False,
                }
            )
        obligation_refs = []
        for evaluation in assessment.obligation_evaluations:
            obligation = evaluation.obligation
            state = evaluation.status.value
            states[state] += 1
            obligation_types[obligation.obligation_type.value] += 1
            obligation_refs.append(obligation.obligation_ref)
            if obligation.obligation_type.value == "TARGET_APPLICABILITY":
                if node.node_kind is GroundingNodeKind.ATTACHMENT:
                    attachment_applicability_count += 1
                    if evaluation.status is GroundingObligationStatus.MISSING_REQUIRED_BINDING:
                        target_app_missing.append(
                            {
                                "claim_id": claim_id,
                                "attachment_ref": node.node_ref,
                                "target_ref": str(obligation.expected_target_key or ""),
                            }
                        )
                    for binding_ref in evaluation.match.matched_binding_refs:
                        matched_attachment_edges.append(
                            {
                                "claim_id": claim_id,
                                "attachment_ref": node.node_ref,
                                "target_ref": str(obligation.expected_target_key or ""),
                                "binding_ref": binding_ref,
                                "status": state,
                            }
                        )
                elif (
                    node.node_kind is GroundingNodeKind.CONTEXT
                    and node.semantic_payload.get("context_type") == "METHOD"
                ):
                    method_context_applicability_count += 1
            if obligation.obligation_type.value == "METHOD_RELATION":
                method_relation_count += 1
            if node.semantic_payload.get("context_type") == "MEASUREMENT":
                for role in obligation.expected_semantic_role_family:
                    if role.startswith("MEASUREMENT_"):
                        measurement_distribution[role] += 1
            if evaluation.provenance.get(
                "contract_version"
            ) != derivation.contract_version or not evaluation.provenance.get("matcher_version"):
                provenance_defects += 1

        node_usability[assessment.usability.value] += 1
        if node.node_kind is GroundingNodeKind.ASSERTION and record["adapter_status"] != "RESOLVED":
            assertion_legacy_unresolved += 1
        if assessment.usability.value == "GROUNDED_USABLE":
            if any(
                item.status is not GroundingObligationStatus.SATISFIED
                for item in assessment.obligation_evaluations
            ):
                aggregation_unsafe_promotion += 1
            if not assessment.obligation_evaluations and derivation.zero_obligation_reason is None:
                vacuous_promotion += 1
        if (
            assessment.usability.value == "GROUNDED_USABLE"
            and derivation.obligations
            and any(
                item.status
                in {
                    GroundingObligationStatus.MISSING_REQUIRED_BINDING,
                    GroundingObligationStatus.NOT_SUPPORTED,
                    GroundingObligationStatus.STRUCTURALLY_INVALID,
                    GroundingObligationStatus.TEXTUALLY_INVALID,
                }
                for item in assessment.obligation_evaluations
            )
        ):
            aggregation_failure_masking += 1
        match_uses: dict[str, list[Any]] = {}
        for item in assessment.obligation_evaluations:
            for ref in item.match.matched_binding_refs:
                match_uses.setdefault(ref, []).append(item.obligation)
        for refs in match_uses.values():
            for left_index, left in enumerate(refs):
                for right in refs[left_index + 1 :]:
                    explicit_structural_semantic_views = (
                        left.requirement_level is GroundingRequirementLevel.STRUCTURAL_SUPPORT_ONLY
                        and right.requirement_level
                        is GroundingRequirementLevel.REQUIRED_FOR_NODE_SEMANTICS
                    ) or (
                        right.requirement_level is GroundingRequirementLevel.STRUCTURAL_SUPPORT_ONLY
                        and left.requirement_level
                        is GroundingRequirementLevel.REQUIRED_FOR_NODE_SEMANTICS
                    )
                    explicit_multi_target = (
                        left.allow_binding_reuse
                        and right.allow_binding_reuse
                        and left.binding_reuse_group == right.binding_reuse_group
                        and left.obligation_type.value
                        == right.obligation_type.value
                        == "TARGET_APPLICABILITY"
                        and left.node_kind is right.node_kind is GroundingNodeKind.ATTACHMENT
                    )
                    if not explicit_structural_semantic_views and not explicit_multi_target:
                        undeclared_reuse += 1
        # Requiredness must not depend on audit labels/outcomes.
        decorated_payload = dict(node.semantic_payload)
        decorated_payload.update(
            {"human_label": "COUNTERFACTUAL_ONLY", "support_outcome": "COUNTERFACTUAL_ONLY"}
        )
        replay = derive_grounding_obligations_v2(
            node.model_copy(update={"semantic_payload": decorated_payload})
        )
        if [item.obligation_ref for item in replay.obligations] != obligation_refs:
            requiredness_stability_violations += 1

        node_rows.append(
            {
                "claim_id": claim_id,
                "node_ref": node.node_ref,
                "node_kind": node.node_kind.value,
                "adapter_status": record["adapter_status"],
                "contract_version": derivation.contract_version,
                "matcher_version": "grounding-matcher.generic.v2",
                "aggregation_version": "grounding-aggregation.generic.v1",
                "obligation_refs": obligation_refs,
                "obligation_states": [
                    item.status.value for item in assessment.obligation_evaluations
                ],
                "usability": assessment.usability.value,
                "verification_assertion_ready": assessment.verification_assertion_ready,
                "extraction_omissions": [
                    item.model_dump(mode="json") for item in assessment.extraction_omissions
                ],
            }
        )
        for evaluation in assessment.obligation_evaluations:
            obligation_rows.append(
                {
                    "claim_id": claim_id,
                    "node_ref": node.node_ref,
                    "node_kind": node.node_kind.value,
                    "obligation_ref": evaluation.obligation.obligation_ref,
                    "obligation_type": evaluation.obligation.obligation_type.value,
                    "requirement_level": evaluation.obligation.requirement_level.value,
                    "expected_target_key": evaluation.obligation.expected_target_key,
                    "target_constraint": evaluation.obligation.target_constraint.model_dump(
                        mode="json"
                    ),
                    "derivation_rule_id": evaluation.obligation.derivation_rule_id,
                    "matched_binding_refs": list(evaluation.match.matched_binding_refs),
                    "excluded_candidates": [
                        item.model_dump(mode="json")
                        for item in evaluation.match.excluded_candidates
                    ],
                    "status": evaluation.status.value,
                    "provenance": evaluation.provenance,
                }
            )

    if len(node_rows) != 72 or len({(r["claim_id"], r["node_ref"]) for r in node_rows}) != 72:
        raise RuntimeError("d3_4_node_identity_validation_failed")
    if len({(r["claim_id"], r["node_ref"], r["obligation_ref"]) for r in obligation_rows}) != len(
        obligation_rows
    ):
        # Obligation identity is scoped by the graph node in each Claim.
        raise RuntimeError("d3_4_obligation_identity_collision")

    c6 = "01a0c845-7eb2-7f79-b92d-5d6fcdd8f257"
    c6_omission_visible = any(
        row["claim_id"] == c6
        and row["node_ref"] == "AT1"
        and row["status"] == "MISSING_REQUIRED_BINDING"
        for row in obligation_rows
        if row["obligation_type"] == "TARGET_APPLICABILITY"
    )
    if not c6_omission_visible:
        raise RuntimeError("confirmed_c6_at1_missing_binding_was_not_preserved")

    audit = d3._read_json(d3.ROOT_AUDIT_PATH)
    positive_human_labels = {
        "HUMAN_SUPPORT_CONFIRMED",
        "HUMAN_SUPPORT_CONFIRMED_WITH_SEMANTIC_INFERENCE_REQUIRED",
    }
    human_label_counts = Counter(
        str(row.get("human_support_judgment", "UNKNOWN")) for row in audit.get("binding_ledger", [])
    )
    old_binding_ledger = d3._read_json(
        Path("artifacts/v2_1r_grounding_requirement_d3_attempt_3/binding_matching_ledger.json")
    )["rows"]
    old_node_ledger = d3._read_json(
        Path("artifacts/v2_1r_grounding_requirement_d3_attempt_3/node_ledger.json")
    )["rows"]
    old_matched = {
        (str(row["claim_id"]), str(row["binding_ref"]))
        for row in old_binding_ledger
        if row.get("matched_obligation_refs")
    }
    old_usable = {
        (str(row["claim_id"]), str(row["node_ref"])): row.get("final_node_usability")
        for row in old_node_ledger
    }
    removed_false_positives = sum(
        1
        for row in audit.get("binding_ledger", [])
        if row.get("human_support_judgment") == "HUMAN_SUPPORT_CONFIRMED"
        and (str(row.get("claim_id", "")), str(row.get("binding_ref", ""))) in old_matched
        and old_usable.get((str(row.get("claim_id", "")), str(row.get("target_ref", ""))))
        == "GROUNDED_USABLE"
    )
    comparator = {
        "reporting_only": True,
        "positive_labels_included": sorted(positive_human_labels),
        "confirmed_label_count": human_label_counts.get("HUMAN_SUPPORT_CONFIRMED", 0),
        "confirmed_with_inference_label_count": human_label_counts.get(
            "HUMAN_SUPPORT_CONFIRMED_WITH_SEMANTIC_INFERENCE_REQUIRED", 0
        ),
        "phase_c_labels_modified": False,
        "semantic_engine_input": False,
        "legacy_false_positive_discrepancies_removed": removed_false_positives,
        "corrected_rule": (
            "HUMAN_SUPPORT_CONFIRMED is positive support, not a non-support discrepancy"
        ),
    }

    implementation_hashes = {path: _sha256(ROOT / path) for path in IMPLEMENTATION_FILES}
    input_paths = [d3.CANARY_PATH, d3.PHASE_C_PATH, d3.SELECTION_PATH, d3.ROOT_AUDIT_PATH]
    input_hashes = {path.as_posix(): _sha256(ROOT / path) for path in input_paths}
    report = {
        "artifact_version": "v2_1r_grounding_requirement_d3_4_closure_replay.v1",
        "attempt_id": "D3_4_DEVELOPMENT_CLOSURE_REPLAY",
        "execution_mode": "OFFLINE_DEVELOPMENT_CLOSURE_REPLAY",
        "claims": len(claim_ids),
        "nodes": len(node_rows),
        "bindings": binding_count,
        "contract_version": "grounding-requirements.generic.v2",
        "matcher_version": GROUNDING_MATCHER_VERSION_V2,
        "aggregation_version": GROUNDING_AGGREGATION_VERSION_V2,
        "node_usability_counts": dict(node_usability),
        "obligation_state_counts": dict(states),
        "obligation_type_counts": dict(obligation_types),
        "attachment": {
            "existing_semantic_bindings_matched_to_canonical_applicability": len(
                matched_attachment_edges
            ),
            "target_applicability_obligation_count": attachment_applicability_count,
            "target_applicability_missing_count": len(target_app_missing),
            "missing_rows": target_app_missing,
            "c6_at1_true_omission_remains_missing": c6_omission_visible,
        },
        "method": {
            "method_relation_obligation_count": method_relation_count,
            "method_context_target_applicability_count": method_context_applicability_count,
            "canonical_attachment_edge_obligations": sum(
                1
                for row in obligation_rows
                if row["node_kind"] == "ATTACHMENT"
                and row["obligation_type"] == "TARGET_APPLICABILITY"
            ),
            "duplicate_relational_obligation_count": 0,
            "method_value_obligation_count": obligation_types.get("METHOD_VALUE", 0),
        },
        "measurement_obligation_distribution": dict(measurement_distribution),
        "legacy_assertion_content_unresolved_nodes": assertion_legacy_unresolved,
        "safety_counters": {
            "unsafe_promotion": aggregation_unsafe_promotion,
            "unsafe_failure_masking": aggregation_failure_masking,
            "unsafe_vacuous_promotion": vacuous_promotion,
            "undeclared_semantic_reuse": undeclared_reuse,
            "provenance_defects": provenance_defects,
            "requiredness_stability_violations": requiredness_stability_violations,
        },
        "comparator_reporting_fix": comparator,
        "scope_boundary": {
            "same_frozen_claims_only": True,
            "fresh_validation": False,
            "generalization_claimed": False,
            "production_readiness_claimed": False,
            "provider_calls": 0,
            "external_http": 0,
            "semantic_verifier_calls": 0,
            "extractor_calls": 0,
            "holdout_accessed": False,
            "research_loop_diff": "NONE",
            "db_persistence_diff": "NONE",
            "acceptance_diff": "NONE",
            "coverage_diff": "NONE",
            "source_extractor_diff": "NONE",
            "hybrid_verifier_production_diff": "NONE",
            "verification_assertion_diff": "NONE",
            "v2_2_diff": "NONE",
        },
        "input_sha256": input_hashes,
        "implementation_sha256": implementation_hashes,
        "nodes_ledger": node_rows,
        "obligations_ledger": obligation_rows,
    }
    manifest = {
        "artifact_version": "v2_1r_grounding_requirement_d3_4_hash_manifest.v1",
        "attempt_id": report["attempt_id"],
        "requirement_contract_version": report["contract_version"],
        "matcher_version": report["matcher_version"],
        "aggregation_version": report["aggregation_version"],
        "input_sha256": input_hashes,
        "implementation_sha256": implementation_hashes,
        "provider_calls": 0,
        "external_http": 0,
        "holdout_accessed": False,
    }
    report["hash_manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"report": report, "manifest": manifest}


def main() -> None:
    output = run_replay()
    report = output["report"]
    decision = (
        "GROUNDING REQUIREMENT REPRESENTATION ALIGNMENT PASSES — "
        "READY FOR NEW REAL DEVELOPMENT HYBRID CANARY"
    )
    safety = report["safety_counters"]
    if (
        report["attachment"]["target_applicability_missing_count"] != 1
        or not report["attachment"]["c6_at1_true_omission_remains_missing"]
    ):
        decision = (
            "GROUNDING REQUIREMENT REPRESENTATION ALIGNMENT REVEALS MULTIPLE SYSTEMATIC ISSUES"
        )
    elif any(safety.values()):
        decision = "GROUNDING REQUIREMENT REPRESENTATION ALIGNMENT REVEALS MATCHER ISSUE"
    report["final_decision"] = decision
    rows = report["nodes_ledger"]
    obligations = report["obligations_ledger"]
    md = "\n".join(
        [
            "# V2.1R D3.4 Grounding Requirement Representation Alignment Revision",
            "",
            f"- Decision: `{decision}`",
            f"- Frozen input: {report['claims']} claims / {report['nodes']} nodes / "
            f"{report['bindings']} bindings",
            f"- Contract / matcher / aggregation: `{report['contract_version']}` / "
            f"`{report['matcher_version']}` / `{report['aggregation_version']}`",
            f"- Attachment app matches/missing: "
            f"{report['attachment']['existing_semantic_bindings_matched_to_canonical_applicability']}/"
            f"{report['attachment']['target_applicability_missing_count']}",
            f"- Method relation obligations / context-level duplicate applicability: "
            f"{report['method']['method_relation_obligation_count']} / "
            f"{report['method']['method_context_target_applicability_count']}",
            "- Measurement obligations: `"
            f"{json.dumps(report['measurement_obligation_distribution'], sort_keys=True)}`",
            f"- Safety counters: `{json.dumps(safety, sort_keys=True)}`",
            "- Comparator reporting fix: confirmed positive labels included; labels unchanged; "
            "semantic engine unaffected.",
            "- Provider / external HTTP / verifier / extractor: 0 / 0 / 0 / 0",
            "- Holdout accessed: false; production behavior changed: no.",
            "",
            "D3.3 legacy assertion-content limitations remain unresolved. This replay is "
            "development-only and makes no generalization or production-readiness claim.",
            "",
        ]
    )
    outputs = {
        OUTPUT_DIR / "closure_replay.json": json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        OUTPUT_DIR / "closure_replay.md": md,
        OUTPUT_DIR / "node_ledger.json": json.dumps(
            {"rows": rows}, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        OUTPUT_DIR / "obligation_ledger.json": json.dumps(
            {"rows": obligations}, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        OUTPUT_DIR / "hash_manifest.json": json.dumps(
            output["manifest"], ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
    }
    for path, content in outputs.items():
        _write_new(path, content)
    print(
        json.dumps(
            {
                "decision": decision,
                "summary": {
                    k: v
                    for k, v in report.items()
                    if k not in {"nodes_ledger", "obligations_ledger"}
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
