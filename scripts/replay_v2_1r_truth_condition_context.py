# ruff: noqa: RUF001
"""Offline replay of frozen V2.1R truth-condition context projections.

This script is deliberately limited to local, previously persisted development
outputs. It has no database, network, model, or ResearchLoop integration.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))

from app.domain import truth_condition_context as tcc  # noqa: E402
from truth_condition_context_reference_reconciliation import (  # noqa: E402
    reconcile_member_ref,
    reconcile_references,
)

ARTIFACTS = ROOT / "artifacts"
FREEZE_PATH = ARTIFACTS / "v2_1r_truth_condition_context_128_reconciled_freeze.json"
JSON_OUT = ARTIFACTS / "v2_1r_truth_condition_context_128_reconciled_replay.json"
MD_OUT = ARTIFACTS / "v2_1r_truth_condition_context_128_reconciled_replay.md"
INTEGRITY_JSON = ARTIFACTS / "v2_1r_truth_condition_context_128_reference_integrity.json"
INTEGRITY_MD = ARTIFACTS / "v2_1r_truth_condition_context_128_reference_integrity.md"
INVALID_RUN_MARKER = ARTIFACTS / "v2_1r_truth_condition_context_128_invalid_run_status.json"

PATHS = {
    "bounded100_report": ARTIFACTS / "v2_1r_bounded_shadow_100_report.json",
    "bounded100_manifest": ARTIFACTS / "v2_1r_bounded_shadow_100_selection_manifest.json",
    "fresh12_manifest": ARTIFACTS
    / "v2_1r_qualifier_semantic_preservation_fresh_validation_12_selection_manifest.json",
    "final16_manifest": ARTIFACTS / "v2_1r_qualifier_final_unseen_16" / "selection_manifest.json",
    "final16_results": ARTIFACTS
    / "v2_1r_qualifier_final_unseen_16"
    / "final_unseen_validation_results.json",
    "context_projection_contract_fixture": ROOT
    / "tests/fixtures/truth_condition_context_shadow_v1.json",
}

HASHED_SOURCE_FILES = (
    "apps/api/app/domain/truth_condition_context.py",
    "apps/api/app/domain/atomic_proposition.py",
    "apps/api/app/domain/assertion_composition.py",
    "apps/api/app/services/atomic_proposition_validation.py",
    "apps/api/app/services/logical_composition_v3.py",
    "apps/api/app/services/atomic_proposition_decomposer.py",
    "apps/api/app/services/atomic_proposition_shadow.py",
    "scripts/replay_v2_1r_truth_condition_context.py",
    "scripts/truth_condition_context_reference_reconciliation.py",
)

FROZEN_SYMBOLS = (
    "project_source_context_graph",
    "project_output_context_graph",
    "compare_context_graphs",
    "normalize_semantic_value",
    "VerificationAssertionView",
    "adapt_legacy_qualifier_inputs",
    "build_verification_assertion_views",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )


def symbol_hashes() -> dict[str, str]:
    source_path = ROOT / "apps/api/app/domain/truth_condition_context.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = set(FROZEN_SYMBOLS)
    found: dict[str, str] = {}
    for node in tree.body:
        name = getattr(node, "name", None)
        if name in wanted:
            found[name] = sha256_bytes(ast.dump(node, include_attributes=False).encode())
    missing = wanted - found.keys()
    if missing:
        raise RuntimeError(f"frozen projection symbols missing: {sorted(missing)}")
    return found


def source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in HASHED_SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"required frozen source missing: {relative}")
        result[relative] = sha256_file(path)
    return result


def fresh_raw_files() -> list[Path]:
    directory = (
        ARTIFACTS
        / "v2_1r_qualifier_semantic_preservation_fresh_validation_12"
        / "canary_diagnostics"
    )
    return sorted(directory.glob("*/v2_1r_semantic_canary_*.json"))


def input_artifact_paths() -> dict[str, Path]:
    paths = dict(PATHS)
    fresh = fresh_raw_files()
    if len(fresh) != 12:
        raise RuntimeError(f"expected 12 persisted Fresh-12 outputs, found {len(fresh)}")
    paths.update({f"fresh12_output_{index:02d}": path for index, path in enumerate(fresh, 1)})
    return paths


def freeze_payload() -> dict[str, Any]:
    artifacts = input_artifact_paths()
    return {
        "freeze_version": "v2_1r-truth-condition-context-128-reconciled-freeze-v1",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "scope": "offline local replay; no provider/database/research execution",
        "projection_module_sha256": sha256_file(
            ROOT / "apps/api/app/domain/truth_condition_context.py"
        ),
        "projection_symbol_ast_sha256": symbol_hashes(),
        "source_file_sha256": source_hashes(),
        "input_artifact_sha256": {
            key: sha256_file(path) for key, path in sorted(artifacts.items())
        },
        "cohort_counts_expected": {
            "bounded_shadow": 100,
            "fresh_validation": 12,
            "final_unseen": 16,
        },
        "provider_calls": 0,
        "database_writes": 0,
        "holdout_content_read": False,
        "reference_contract": "model_local_references maps local_ref -> runtime proposition ID",
    }


def verify_freeze() -> dict[str, Any]:
    frozen = load_json(FREEZE_PATH)
    current = freeze_payload()
    comparisons = {
        "projection_module_sha256": frozen["projection_module_sha256"]
        == current["projection_module_sha256"],
        "projection_symbol_ast_sha256": frozen["projection_symbol_ast_sha256"]
        == current["projection_symbol_ast_sha256"],
        "source_file_sha256": frozen["source_file_sha256"] == current["source_file_sha256"],
        "input_artifact_sha256": frozen["input_artifact_sha256"]
        == current["input_artifact_sha256"],
    }
    if not all(comparisons.values()):
        changed = [key for key, passed in comparisons.items() if not passed]
        raise RuntimeError(f"frozen inputs/implementation changed; replay prohibited: {changed}")
    return {
        "passed": True,
        "checks": comparisons,
        "freeze_path": str(FREEZE_PATH.relative_to(ROOT)),
    }


def map_qualifiers(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list | tuple):
        return []
    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "category": item.get("category", item.get("type", "other")),
                "key": item.get("key", item.get("name", "qualifier")),
                "value": item.get("value", item.get("normalized_value")),
                "source_span": item.get("source_span", item.get("span")),
            }
        )
    return result


def normalize_proposition(raw: dict[str, Any], index: int) -> dict[str, Any]:
    ref = raw.get("local_ref", raw.get("proposition_id", raw.get("proposition_ref", f"p{index}")))
    decomposition = raw.get("decomposition")
    span = raw.get("source_span")
    if span is None and isinstance(decomposition, dict):
        span = decomposition.get("source_span")
    typed_value = raw.get("typed_value", raw.get("value"))
    return {
        "local_ref": str(ref),
        "runtime_proposition_id": raw.get("runtime_proposition_id", raw.get("proposition_id")),
        "model_local_ref": raw.get("model_local_ref", raw.get("local_ref")),
        "reference_identity_kind": raw.get("reference_identity_kind"),
        "eligibility": raw.get("eligibility", "UNKNOWN"),
        "eligibility_source_key": raw.get("eligibility_source_key"),
        "eligibility_reason": raw.get("eligibility_reason"),
        "source_span": span,
        "subject": raw.get("subject"),
        "predicate": raw.get("predicate", raw.get("relation")),
        "typed_value": typed_value,
        "polarity": raw.get("polarity"),
        "modality": raw.get("modality"),
        "attribution": raw.get("attribution"),
        "material_qualifiers": map_qualifiers(
            raw.get("material_qualifiers", raw.get("qualifiers"))
        ),
        "proposition_text": raw.get(
            "proposition_text", raw.get("canonical_text", raw.get("original_text"))
        ),
    }


def normalize_groups(
    composition: Any, local_to_runtime: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    if not isinstance(composition, dict):
        return []
    groups = composition.get("groups", [])
    if not isinstance(groups, list):
        return []
    normalized = []
    for index, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        local_ref = group.get("local_ref", group.get("group_id", f"g{index}"))
        members = group.get("member_refs", group.get("members", []))
        refs = []
        for member in members if isinstance(members, list) else []:
            if isinstance(member, str):
                refs.append(reconcile_member_ref(member, local_to_runtime or {}))
            elif isinstance(member, dict):
                member_ref = member.get("ref")
                member_ref = member_ref or member.get("proposition_id") or member.get("group_id")
                if member_ref:
                    member_type = str(member.get("member_type", "")).lower()
                    refs.append(
                        str(member_ref)
                        if member_type == "group"
                        else reconcile_member_ref(str(member_ref), local_to_runtime or {})
                    )
        normalized.append(
            {
                "local_ref": str(local_ref),
                "member_refs": refs,
                "source_span": group.get("source_span"),
                "operator": group.get("operator"),
            }
        )
    return normalized


def composition_reference_issues(
    composition: Any,
    local_to_runtime: dict[str, str],
    propositions: list[dict[str, Any]],
) -> list[dict[str, str]]:
    if not isinstance(composition, dict) or not isinstance(composition.get("groups", []), list):
        return []
    prop_ids = {
        str(item.get("runtime_proposition_id", item.get("local_ref"))) for item in propositions
    }
    group_rows = [item for item in composition.get("groups", []) if isinstance(item, dict)]
    group_ids = {
        str(item.get("local_ref", item.get("group_id")))
        for item in group_rows
        if item.get("local_ref", item.get("group_id")) is not None
    }
    issues: list[dict[str, str]] = []
    for group_index, group in enumerate(group_rows):
        members = group.get("members", group.get("member_refs", []))
        if not isinstance(members, list):
            continue
        for member_index, member in enumerate(members):
            ref: Any
            if isinstance(member, str):
                ref, member_type = member, ""
            elif isinstance(member, dict):
                ref = member.get("ref", member.get("proposition_id", member.get("group_id")))
                member_type = str(member.get("member_type", "")).lower()
            else:
                continue
            if ref is None:
                continue
            ref = str(ref)
            if member_type == "group" or (not member_type and ref in group_ids):
                if ref not in group_ids:
                    issues.append({"code": "COMPOSITION_GROUP_REF_UNRESOLVED", "detail": ref})
                continue
            canonical = reconcile_member_ref(ref, local_to_runtime)
            if canonical not in prop_ids:
                issues.append(
                    {
                        "code": "COMPOSITION_MEMBER_REF_UNRESOLVED",
                        "detail": f"group:{group_index}/member:{member_index}/{ref}",
                    }
                )
    return issues


def eligibility_map(record: dict[str, Any], cohort: str) -> dict[str, str]:
    if cohort == "bounded_shadow":
        items = record.get("verification_eligibility", [])
        return {
            str(i.get("ref")): str(i.get("eligibility", "UNKNOWN"))
            for i in items
            if isinstance(i, dict)
        }
    dec = record.get("decomposition", {}) if cohort != "bounded_shadow" else {}
    if cohort == "fresh_validation":
        dec = dec.get("validator_result", {})
    raw = dec.get("derived_member_eligibility", {}) if isinstance(dec, dict) else {}
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    if isinstance(raw, list):
        return {
            str(item.get("ref", item.get("proposition_ref"))): str(
                item.get("eligibility", item.get("status", "UNKNOWN"))
            )
            for item in raw
            if isinstance(item, dict)
        }
    return {}


def normalize_record(record: dict[str, Any], cohort: str) -> dict[str, Any]:
    if cohort == "bounded_shadow":
        claim_text = record["claim_text"]
        raw_props = record.get("propositions", [])
        composition = record.get("assertion_composition", {})
        claim_id = record["claim_id"]
        identity_hash = record.get("claim_identity_sha256")
        result_meta = {
            "model_status": record.get("model_returned_status"),
            "parser_status": record.get("parser_status"),
            "final_status": record.get("final_shadow_status"),
            "tokens": record.get("tokens", {}),
            "latency_ms": record.get("latency_ms"),
            "provider_request_id": record.get("provider_request_id"),
            "qualifier_status_counts": record.get("qualifier_status_counts", {}),
            "legacy_qualifier_status_counts": record.get("qualifier_check_status_counts", {}),
        }
        model_local_references = None
        raw_eligibility = record.get("verification_eligibility", [])
    elif cohort == "fresh_validation":
        claim_text = record["claim"]["claim_text"]
        dec = record["decomposition"]
        structured = dec.get("model_structured_result", {})
        raw_props = structured.get("propositions", [])
        composition = structured.get("assertion_composition", {})
        claim_id = record["claim"]["claim_id"]
        identity_hash = None
        result_meta = {
            "model_status": structured.get(
                "model_returned_status", structured.get("decomposition_status")
            ),
            "parser_status": dec.get("parser_result", {}).get("status"),
            "final_status": dec.get("final_shadow_result", {}).get("status"),
            "tokens": record.get("tokens", {}),
            "latency_ms": record.get("latency_ms"),
            "provider_request_id": record.get("provider_model", {}).get("provider_request_id"),
            "qualifier_status_counts": dec.get("validator_result", {}).get("qualifier_checks", []),
            "legacy_qualifier_status_counts": {},
        }
        model_local_references = dec.get("model_local_references")
        raw_eligibility = dec.get("validator_result", {}).get("derived_member_eligibility", {})
    else:
        claim_text = record["claim_text"]
        dec = record["decomposition"]
        raw_props = dec.get("propositions", [])
        composition = dec.get("assertion_composition", {})
        claim_id = record["claim_id"]
        identity_hash = None
        result_meta = {
            "model_status": dec.get("status"),
            "parser_status": dec.get("parser_result", {}).get("status")
            if isinstance(dec.get("parser_result"), dict)
            else None,
            "final_status": record.get("runner_status"),
            "tokens": {"input": dec.get("input_tokens"), "output": dec.get("output_tokens")},
            "latency_ms": record.get("wall_latency_ms"),
            "provider_request_id": dec.get("model_metadata", {}).get("provider_request_id"),
            "qualifier_status_counts": dec.get("qualifier_checks", []),
            "legacy_qualifier_status_counts": {},
        }
        model_local_references = dec.get("model_local_references")
        raw_eligibility = dec.get("derived_member_eligibility", {})
    reconciliation = reconcile_references(raw_props, model_local_references, raw_eligibility)
    propositions = [
        normalize_proposition(item, i)
        for i, item in enumerate(reconciliation["propositions"])
        if isinstance(item, dict)
    ]
    provenance = record.get("provenance", {})
    group_issues = composition_reference_issues(
        composition, reconciliation["local_to_runtime"], propositions
    )
    retrieval_provenance_present = isinstance(provenance, dict) and any(
        provenance.get(key) is not None
        for key in ("source_id", "source_owner_key", "source_domain", "source_url", "evidence_id")
    )
    return {
        "cohort": cohort,
        "claim_id": str(claim_id),
        "claim_text": str(claim_text),
        "identity_hash": identity_hash,
        "propositions": propositions,
        "composition_groups": normalize_groups(composition, reconciliation["local_to_runtime"]),
        "eligibility_by_ref": reconciliation["eligibility_by_ref"],
        "reference_reconciliation": reconciliation,
        "composition_reference_issues": group_issues,
        "metadata": result_meta,
        "raw_record": record,
        "retrieval_provenance_present": retrieval_provenance_present,
    }


def load_cohorts() -> dict[str, list[dict[str, Any]]]:
    report = load_json(PATHS["bounded100_report"])
    manifest100 = load_json(PATHS["bounded100_manifest"])
    manifest12 = load_json(PATHS["fresh12_manifest"])
    manifest16 = load_json(PATHS["final16_manifest"])
    results16 = load_json(PATHS["final16_results"])

    selected100 = {item["claim_id"]: item for item in manifest100["selected_claims"]}
    report100 = report["claim_records"]
    if len(report100) != 100:
        raise RuntimeError(f"bounded-shadow report record count is {len(report100)}, expected 100")
    for row in report100:
        selected = selected100.get(row["claim_id"])
        if selected is None or selected.get("identity_sha256") != row.get("claim_identity_sha256"):
            raise RuntimeError(f"bounded-shadow selected identity mismatch: {row.get('claim_id')}")

    fresh_manifest = {item["claim_id"]: item for item in manifest12["selected_claims"]}
    fresh_by_id: dict[str, dict[str, Any]] = {}
    for path in fresh_raw_files():
        item = load_json(path)
        cid = item.get("claim", {}).get("claim_id")
        if cid in fresh_by_id:
            raise RuntimeError(f"duplicate Fresh-12 output for {cid}")
        fresh_by_id[cid] = item
    if len(fresh_manifest) != 12 or set(fresh_by_id) != set(fresh_manifest):
        raise RuntimeError("Fresh-12 selected IDs do not exactly match persisted outputs")
    for cid, row in fresh_by_id.items():
        expected = fresh_manifest[cid].get("claim_text_sha256")
        if expected != sha256_bytes(row["claim"]["claim_text"].encode("utf-8")):
            raise RuntimeError(f"Fresh-12 claim text hash mismatch for {cid}")

    selected16 = {item["claim_id"]: item for item in manifest16["selected_claims"]}
    rows16 = results16["records"]
    if len(rows16) != 16 or len(selected16) != 16:
        raise RuntimeError("Final-Unseen-16 record/manifest count mismatch")
    if results16.get("provider_outbound_count") != 16 or results16.get("database_writes") != 0:
        raise RuntimeError(
            "Final-Unseen-16 persisted run metadata violates expected offline source constraints"
        )
    for row in rows16:
        selected = selected16.get(row["claim_id"])
        if selected is None or selected.get("claim_text_sha256") != row.get("claim_text_sha256"):
            raise RuntimeError(f"Final-Unseen-16 selected identity mismatch: {row.get('claim_id')}")
        if sha256_bytes(row["claim_text"].encode("utf-8")) != row.get("claim_text_sha256"):
            raise RuntimeError(
                f"Final-Unseen-16 claim text hash does not verify: {row['claim_id']}"
            )

    cohorts = {
        "bounded_shadow": [normalize_record(row, "bounded_shadow") for row in report100],
        "fresh_validation": [
            normalize_record(fresh_by_id[cid], "fresh_validation") for cid in fresh_manifest
        ],
        "final_unseen": [normalize_record(row, "final_unseen") for row in rows16],
    }
    ids = {name: [row["claim_id"] for row in rows] for name, rows in cohorts.items()}
    if any(len(values) != len(set(values)) for values in ids.values()):
        raise RuntimeError("duplicate Claim IDs within a cohort")
    overlaps = {
        f"{left}__{right}": sorted(set(ids[left]) & set(ids[right]))
        for left, right in (
            ("bounded_shadow", "fresh_validation"),
            ("bounded_shadow", "final_unseen"),
            ("fresh_validation", "final_unseen"),
        )
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Claim IDs overlap across cohorts: {overlaps}")
    if sum(map(len, cohorts.values())) != 128:
        raise RuntimeError("cohort sizes do not total 128")
    return cohorts


def reference_integrity_audit(cohorts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    blocking_codes = {
        "LOCAL_REF_MAPPING_MISSING",
        "RUNTIME_ID_MAPPING_MISSING",
        "DUPLICATE_LOCAL_REF",
        "DUPLICATE_RUNTIME_ID",
        "ORPHAN_ELIGIBILITY_ENTRY",
        "ELIGIBILITY_KEY_UNRESOLVED",
        "COMPOSITION_MEMBER_REF_UNRESOLVED",
        "COMPOSITION_GROUP_REF_UNRESOLVED",
    }
    by_cohort: dict[str, Any] = {}
    all_issues: list[dict[str, str]] = []
    for cohort, rows in cohorts.items():
        counts: Counter[str] = Counter()
        issues: list[dict[str, Any]] = []
        for row in rows:
            reconciliation = row["reference_reconciliation"]
            counts.update(reconciliation["counts"])
            for issue in reconciliation["issues"]:
                enriched = {
                    "cohort": cohort,
                    "claim_id": row["claim_id"],
                    **issue,
                }
                issues.append(enriched)
                all_issues.append(enriched)
            for issue in row.get("composition_reference_issues", []):
                enriched = {"cohort": cohort, "claim_id": row["claim_id"], **issue}
                issues.append(enriched)
                all_issues.append(enriched)
        blocking = [item for item in issues if item["code"] in blocking_codes]
        by_cohort[cohort] = {
            **dict(counts),
            "issue_counts": dict(sorted(Counter(item["code"] for item in issues).items())),
            "blocking_issue_count": len(blocking),
            "issues": issues,
        }
    passed = all(item["blocking_issue_count"] == 0 for item in by_cohort.values())
    return {
        "audit_version": "v2_1r-reference-reconciliation-integrity-v1",
        "status": "PASS" if passed else "STOP_MAPPING_AMBIGUITY",
        "passed": passed,
        "provider_calls": 0,
        "database_writes": 0,
        "holdout_content_read": False,
        "canonical_identity": "runtime proposition ID where persisted; otherwise source-local ref",
        "model_local_references_contract": "local_ref -> runtime proposition ID",
        "cohorts": by_cohort,
        "blocking_issue_codes": sorted(blocking_codes),
        "nonblocking_source_absence_code": "PROPOSITION_WITHOUT_ELIGIBILITY_MAPPING",
        "all_issues": all_issues,
    }


def _target_refs(context: Any, graph: Any) -> tuple[str, ...]:
    refs: list[str] = []
    for edge in context.attachments:
        if edge.target_ref is None:
            continue
        expanded = graph.scope_expansions.get(edge.target_ref)
        refs.extend(expanded or (edge.target_ref,))
    if not refs:
        refs.extend(context.scope.target_refs)
    return tuple(dict.fromkeys(refs))


def supplemental_view(
    view: Any, source_graph: Any, output_graph: Any, comparison: Any
) -> dict[str, Any]:
    attached = []
    for context in output_graph.contexts:
        if view.proposition_ref in _target_refs(context, output_graph):
            attached.append(context)
    epistemic = [
        context.semantic_signature
        for context in attached
        if context.context_type.value == "EPISTEMIC"
    ]
    unknown = [
        context.context_ref
        for context in attached
        if context.materiality.upper() == "UNKNOWN"
        or context.scope.kind.value == "UNKNOWN"
        or any(edge.status.value == "UNKNOWN" for edge in context.attachments)
    ]
    source_context_results = {
        item.source_context_ref: item for item in comparison if item.source_context_ref
    }
    source_attached = [
        context
        for context in source_graph.contexts
        if view.proposition_ref in _target_refs(context, source_graph)
    ]
    source_statuses = [
        source_context_results[context.context_ref].outcome.value
        for context in source_attached
        if context.context_ref in source_context_results
    ]
    if not source_graph.source_detection_complete:
        completeness = "DETECTOR_LIMIT"
    elif (
        any(status in {"ATTACHMENT_AMBIGUOUS", "DETECTOR_LIMIT"} for status in source_statuses)
        or unknown
    ):
        completeness = "MATERIAL_UNKNOWN"
    elif any(
        status in {"MISSING", "SCOPE_MISMATCH", "SEMANTIC_MISMATCH"} for status in source_statuses
    ):
        completeness = "MATERIAL_MISSING"
    else:
        completeness = "COMPLETE"
    epistemic_source = [
        context for context in source_attached if context.context_type.value == "EPISTEMIC"
    ]
    epistemic_statuses = [
        source_context_results[context.context_ref].outcome.value
        for context in epistemic_source
        if context.context_ref in source_context_results
    ]
    epistemic_completeness = (
        "DETECTOR_LIMIT"
        if not source_graph.source_detection_complete
        else "MATERIAL_UNKNOWN"
        if any(
            status in {"ATTACHMENT_AMBIGUOUS", "DETECTOR_LIMIT"} for status in epistemic_statuses
        )
        else "MISSING"
        if any(
            status in {"MISSING", "SCOPE_MISMATCH", "SEMANTIC_MISMATCH"}
            for status in epistemic_statuses
        )
        else "PRESERVED"
        if epistemic_source
        else "NO_SOURCE_EPISTEMIC_CONTEXT_DETECTED"
    )
    return {
        "base_verification_assertion_view": view.model_dump(mode="json"),
        "normalized_proposition_core": view.proposition_core,
        "material_context_signatures": list(view.material_context_signatures),
        "resolved_material_contexts": [
            {
                "context_ref": context.context_ref,
                "context_type": context.context_type.value,
                "semantic_signature": context.semantic_signature,
                "outcome": source_context_results[context.context_ref].outcome.value,
            }
            for context in source_attached
            if context.context_ref in source_context_results
            and source_context_results[context.context_ref].outcome.value == "PRESERVED"
        ],
        "context_completeness": completeness,
        "material_context_unknown": bool(unknown)
        or not source_graph.source_detection_complete
        or any(status in {"ATTACHMENT_AMBIGUOUS", "DETECTOR_LIMIT"} for status in source_statuses),
        "epistemic_completeness": epistemic_completeness,
        "epistemic_context_signatures": sorted(epistemic),
        "output_attached_context_refs": sorted(context.context_ref for context in attached),
        "unknown_context_details": sorted(set(unknown) | set(view.unknown_context_details)),
        "composition_eligibility": view.composition_eligibility,
    }


def claim_replay(row: dict[str, Any]) -> dict[str, Any]:
    source_graph = tcc.project_source_context_graph(
        row["claim_text"],
        row["propositions"],
        claim_ref=row["claim_id"],
        composition_groups=row["composition_groups"],
    )
    output_graph = tcc.project_output_context_graph(row["propositions"], claim_ref=row["claim_id"])
    comparison = tcc.compare_context_graphs(source_graph, output_graph)
    views = tcc.build_verification_assertion_views(
        row["propositions"], output_graph, eligibility_by_ref=row["eligibility_by_ref"]
    )
    outcomes = Counter(result.outcome.value for result in comparison)
    unknown_source = any(
        context.scope.kind.value == "UNKNOWN"
        or any(edge.status.value == "UNKNOWN" for edge in context.attachments)
        for context in source_graph.contexts
        if context.materiality.upper() in {"MATERIAL", "UNKNOWN"}
    )
    detector_limit = not source_graph.source_detection_complete
    loss = any(
        value > 0
        for key, value in outcomes.items()
        if key in {"MISSING", "SCOPE_MISMATCH", "SEMANTIC_MISMATCH"}
    )
    if detector_limit:
        completeness = "DETECTOR_LIMIT"
    elif (
        unknown_source
        or outcomes.get("ATTACHMENT_AMBIGUOUS", 0)
        or outcomes.get("DETECTOR_LIMIT", 0)
    ):
        completeness = "MATERIAL_UNKNOWN"
    elif loss:
        completeness = "MATERIAL_MISSING_OR_MISMATCH"
    else:
        completeness = "COMPLETE_AS_PROJECTED"
    eligibility_counts = Counter(
        row["eligibility_by_ref"].get(p["local_ref"], "UNKNOWN") for p in row["propositions"]
    )
    shared = []
    for context in source_graph.contexts:
        targets = _target_refs(context, source_graph)
        if len(targets) > 1:
            matching = next(
                (item for item in comparison if item.source_context_ref == context.context_ref),
                None,
            )
            output_targets = sorted(
                {
                    member
                    for output_ref in (matching.output_context_refs if matching else ())
                    for output_context in output_graph.contexts
                    if output_context.context_ref == output_ref
                    for member in _target_refs(output_context, output_graph)
                }
            )
            if matching is None or matching.outcome.value == "DETECTOR_LIMIT":
                shared_classification = "DETECTOR_LIMIT"
            elif matching.outcome.value == "ATTACHMENT_AMBIGUOUS":
                shared_classification = "TRUE_ATTACHMENT_AMBIGUITY"
            elif output_targets == sorted(targets):
                shared_classification = "CORRECT_SHARED_CONTEXT"
            elif set(output_targets) > set(targets):
                shared_classification = "SCOPE_OVEREXPANSION"
            elif set(output_targets) < set(targets):
                shared_classification = "SCOPE_UNDEREXPANSION"
            else:
                shared_classification = "SCOPE_MISMATCH"
            shared.append(
                {
                    "context_ref": context.context_ref,
                    "context_type": context.context_type.value,
                    "source_targets": list(targets),
                    "outcome": matching.outcome.value if matching else "UNMAPPED",
                    "output_context_refs": list(matching.output_context_refs) if matching else [],
                    "output_targets": output_targets,
                    "classification": shared_classification,
                }
            )
    retrieval_fields = {
        "evidence_source_domain",
        "source_owner_key",
        "source_type",
        "source_id",
        "evidence_id",
    }
    if row["cohort"] == "fresh_validation":
        raw_props = (
            row["raw_record"]
            .get("decomposition", {})
            .get("model_structured_result", {})
            .get("propositions", [])
        )
    elif row["cohort"] == "final_unseen":
        raw_props = row["raw_record"].get("decomposition", {}).get("propositions", [])
    else:
        raw_props = row["raw_record"].get("propositions", [])
    retrieval_only_metadata_present = any(
        retrieval_fields & proposition.keys()
        for proposition in raw_props
        if isinstance(proposition, dict)
    )
    review_reasons = []
    if any(context.context_type.value == "EPISTEMIC" for context in source_graph.contexts):
        review_reasons.append("SOURCE_EPISTEMIC_CONTEXT")
    if any(
        item.outcome.value
        in {"MISSING", "SCOPE_MISMATCH", "SEMANTIC_MISMATCH", "ATTACHMENT_AMBIGUOUS"}
        for item in comparison
    ):
        review_reasons.append("NON_PRESERVED_MATERIAL_CONTEXT")
    if any(
        item.context_type.value
        in {"TEMPORAL", "VERSION", "CONDITION", "MEASUREMENT", "SAMPLE_SCOPE", "METHOD"}
        for item in source_graph.contexts
    ):
        review_reasons.append("SUSPICIOUS_CONTEXT_REVIEW")
    if any(item.outcome.value == "PRESERVED" for item in comparison):
        review_reasons.append("PRESERVED_CONTEXT_MANUAL_AUDIT")
    return {
        "cohort": row["cohort"],
        "claim_id": row["claim_id"],
        "claim_text": row["claim_text"],
        "identity_hash": row["identity_hash"],
        "metadata": row["metadata"],
        "propositions": row["propositions"],
        "reference_reconciliation": row.get("reference_reconciliation", {}),
        "source_context_graph": source_graph.model_dump(mode="json"),
        "output_context_graph": output_graph.model_dump(mode="json"),
        "context_preservation": [item.model_dump(mode="json") for item in comparison],
        "verification_assertion_views": [
            supplemental_view(view, source_graph, output_graph, comparison) for view in views
        ],
        "context_completeness": completeness,
        "context_outcome_counts": dict(sorted(outcomes.items())),
        "epistemic_source_context_count": sum(
            context.context_type.value == "EPISTEMIC" for context in source_graph.contexts
        ),
        "shared_contexts": shared,
        "composition_eligibility_counts_independent_of_context": dict(
            sorted(eligibility_counts.items())
        ),
        "source_projection_diagnostics": list(source_graph.diagnostics),
        "source_detection_complete": source_graph.source_detection_complete,
        "retrieval_provenance_metadata_present_in_claim_output": retrieval_only_metadata_present,
        "retrieval_provenance_present_in_cohort_record": row.get(
            "retrieval_provenance_present", False
        ),
        "review_reasons": sorted(set(review_reasons)),
        "source_scope_counts": dict(
            sorted(Counter(context.scope.kind.value for context in source_graph.contexts).items())
        ),
        "source_attachment_status_counts": dict(
            sorted(
                Counter(
                    edge.status.value
                    for context in source_graph.contexts
                    for edge in context.attachments
                ).items()
            )
        ),
        "output_context_type_counts": dict(
            sorted(Counter(context.context_type.value for context in output_graph.contexts).items())
        ),
        "output_attachment_status_counts": dict(
            sorted(
                Counter(
                    edge.status.value
                    for context in output_graph.contexts
                    for edge in context.attachments
                ).items()
            )
        ),
        "output_numeric_normalized_values": [
            {
                "proposition_ref": proposition["local_ref"],
                "typed_value": tcc.normalize_semantic_value(proposition.get("typed_value")),
            }
            for proposition in row["propositions"]
            if isinstance(proposition.get("typed_value"), dict)
            and (
                "numeric" in proposition["typed_value"]
                or proposition["typed_value"].get("kind") in {"quantity", "comparison", "range"}
            )
        ],
        "epistemic_audit": [
            {
                "source_context_ref": context.context_ref,
                "source_signature": context.semantic_signature,
                "scope_kind": context.scope.kind.value,
                "source_targets": list(_target_refs(context, source_graph)),
                "outcome": next(
                    (
                        item.outcome.value
                        for item in comparison
                        if item.source_context_ref == context.context_ref
                    ),
                    "UNMAPPED",
                ),
                "output_context_refs": list(
                    next(
                        (
                            item.output_context_refs
                            for item in comparison
                            if item.source_context_ref == context.context_ref
                        ),
                        (),
                    )
                ),
                "source_epistemic_detected": True,
                "output_classification": (
                    "OUTPUT_EPISTEMIC_CONTEXT_PRESERVED"
                    if any(
                        item.source_context_ref == context.context_ref
                        and item.outcome.value == "PRESERVED"
                        for item in comparison
                    )
                    else "OUTPUT_EPISTEMIC_CONTEXT_MISSING"
                    if any(
                        item.source_context_ref == context.context_ref
                        and item.outcome.value in {"MISSING", "SEMANTIC_MISMATCH", "SCOPE_MISMATCH"}
                        for item in comparison
                    )
                    else "ATTACHMENT_AMBIGUOUS_OR_DETECTOR_LIMIT"
                ),
            }
            for context in source_graph.contexts
            if context.context_type.value == "EPISTEMIC"
        ],
    }


def summarize_claims(claims: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes: Counter[str] = Counter()
    contexts: Counter[str] = Counter()
    completeness: Counter[str] = Counter()
    eligibility: Counter[str] = Counter()
    diagnostics: Counter[str] = Counter()
    source_scopes: Counter[str] = Counter()
    source_attachments: Counter[str] = Counter()
    output_types: Counter[str] = Counter()
    output_attachments: Counter[str] = Counter()
    shared_classes: Counter[str] = Counter()
    epistemic_outcomes: Counter[str] = Counter()
    source_types_by_claim: Counter[str] = Counter()
    unknown_context_refs = 0
    unknown_scope_refs = 0
    numeric_outputs = 0
    legacy_warning_claims = 0
    legacy_statuses: Counter[str] = Counter()
    detector_limit_claims = 0
    material_unknown_views = 0
    missing_views = 0
    shared_count = 0
    epistemic = 0
    prop_count = 0
    for claim in claims:
        outcomes.update(claim["context_outcome_counts"])
        completeness[claim["context_completeness"]] += 1
        contexts.update(item["context_type"] for item in claim["source_context_graph"]["contexts"])
        eligibility.update(claim["composition_eligibility_counts_independent_of_context"])
        diagnostics.update(claim["source_projection_diagnostics"])
        source_scopes.update(claim["source_scope_counts"])
        source_attachments.update(claim["source_attachment_status_counts"])
        output_types.update(claim["output_context_type_counts"])
        output_attachments.update(claim["output_attachment_status_counts"])
        shared_classes.update(item["classification"] for item in claim["shared_contexts"])
        epistemic_outcomes.update(
            item["output_classification"] for item in claim["epistemic_audit"]
        )
        source_types_by_claim.update(
            context["context_type"] for context in claim["source_context_graph"]["contexts"]
        )
        unknown_context_refs += sum(
            edge["status"] == "UNKNOWN"
            for graph_context in claim["source_context_graph"]["contexts"]
            for edge in graph_context["attachments"]
        )
        unknown_scope_refs += sum(
            graph_context["scope"]["kind"] == "UNKNOWN"
            for graph_context in claim["source_context_graph"]["contexts"]
        )
        numeric_outputs += len(claim["output_numeric_normalized_values"])
        detector_limit_claims += claim["context_completeness"] == "DETECTOR_LIMIT"
        material_unknown_views += sum(
            view["context_completeness"] == "MATERIAL_UNKNOWN"
            for view in claim["verification_assertion_views"]
        )
        missing_views += sum(
            view["context_completeness"] == "MATERIAL_MISSING"
            for view in claim["verification_assertion_views"]
        )
        legacy = claim["metadata"].get("legacy_qualifier_status_counts", {})
        if isinstance(legacy, dict):
            legacy_statuses.update(
                {str(key): int(value) for key, value in legacy.items() if isinstance(value, int)}
            )
            legacy_warning_claims += any(
                str(key).upper() in {"POTENTIAL_LOSS", "AMBIGUOUS", "FAILED", "BLOCKED"} and value
                for key, value in legacy.items()
                if isinstance(value, int)
            )
        shared_count += len(claim["shared_contexts"])
        epistemic += claim["epistemic_source_context_count"]
        prop_count += len(claim["propositions"])
    review = [claim for claim in claims if claim["review_reasons"]]
    return {
        "claims": len(claims),
        "propositions": prop_count,
        "source_context_counts_by_type": dict(sorted(contexts.items())),
        "preservation_outcomes": dict(sorted(outcomes.items())),
        "context_completeness_by_claim": dict(sorted(completeness.items())),
        "composition_eligibility_counts_independent_of_context": dict(sorted(eligibility.items())),
        "source_projection_diagnostics": dict(sorted(diagnostics.items())),
        "source_scope_counts": dict(sorted(source_scopes.items())),
        "source_attachment_status_counts": dict(sorted(source_attachments.items())),
        "output_context_counts_by_type": dict(sorted(output_types.items())),
        "output_attachment_status_counts": dict(sorted(output_attachments.items())),
        "shared_context_classifications": dict(sorted(shared_classes.items())),
        "epistemic_context_classifications": dict(sorted(epistemic_outcomes.items())),
        "unknown_attachment_count": unknown_context_refs,
        "unknown_scope_count": unknown_scope_refs,
        "detector_limit_claim_count": detector_limit_claims,
        "material_unknown_view_count": material_unknown_views,
        "material_missing_view_count": missing_views,
        "normalized_numeric_typed_value_count": numeric_outputs,
        "legacy_qualifier_status_counts_available_in_source_artifacts": dict(
            sorted(legacy_statuses.items())
        ),
        "legacy_warning_claim_count_when_aggregates_available": legacy_warning_claims,
        "source_epistemic_contexts": epistemic,
        "shared_context_occurrences": shared_count,
        "review_queue_claim_count": len(review),
        "review_queue_claim_ids": [claim["claim_id"] for claim in review],
        "retrieval_provenance_metadata_claim_count": sum(
            c["retrieval_provenance_metadata_present_in_claim_output"] for c in claims
        ),
        "retrieval_provenance_source_records": sum(
            c["retrieval_provenance_present_in_cohort_record"] for c in claims
        ),
    }


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# V2.1R Truth-Condition Context Development Replay — Phase C",
        "",
        f"- Replay ID: `{payload['replay_id']}`",
        f"- Freeze verified: `{payload['freeze_verification']['passed']}`",
        f"- Cohort total: **{payload['total_claims']}** (provider calls: **0**, DB writes: **0**)",
        "- Holdout content read: **No**",
        "",
        "## Cohort summary",
        "",
        "| Cohort | Claims | Propositions | Outcomes | Completeness | Epistemic | Review queue |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for name, summary in payload["cohort_summaries"].items():
        lines.append(
            f"| {name} | {summary['claims']} | {summary['propositions']} | "
            f"PRESERVED {summary['preservation_outcomes'].get('PRESERVED', 0)}; "
            f"MISSING {summary['preservation_outcomes'].get('MISSING', 0)} | "
            f"DETECTOR_LIMIT {summary['context_completeness_by_claim'].get('DETECTOR_LIMIT', 0)} | "
            f"{summary['source_epistemic_contexts']} | {summary['review_queue_claim_count']} |"
        )
    lines += [
        "",
        "## Required sections A–X",
        "",
        "Claim/context-level rows are retained in the paired JSON artifact.",
        "",
    ]
    section_order = [
        ("A", "Frozen Versions", "A_frozen_versions"),
        ("B", "Dataset Governance", "B_dataset_governance"),
        ("C", "Source Context Graph Metrics", "C_source_context_graph_metrics"),
        ("D", "Output Context Graph Metrics", "D_output_context_graph_metrics"),
        ("E", "Shared Context", "E_shared_context"),
        ("F", "Temporal Context", "F_temporal_context"),
        ("G", "Numeric Normalization", "G_numeric_normalization"),
        ("H", "Epistemic Context", "H_epistemic_context"),
        ("I", "Retrieval Provenance Safety", "I_retrieval_provenance_safety"),
        ("J", "Method / Semantic Role Precision", "J_method_semantic_role"),
        ("K", "Preservation Comparison", "K_preservation_comparison"),
        ("L", "Legacy vs Context", "L_legacy_vs_context"),
        ("M", "Final-16 Failure Mapping", "M_final16_failure_mapping"),
        ("N", "False-Preservation Audit", "N_false_preservation_audit"),
        ("O", "UNKNOWN Quality", "O_unknown_quality"),
        ("P", "Context Completeness", "P_context_completeness"),
        ("Q", "Verification Assertion View", "Q_verification_assertion_view"),
        ("R", "V2.2 Interface Diagnostic", "R_v2_2_interface_diagnostic"),
        ("S", "Non-Context Failures", "S_non_context_failures"),
        ("T", "Cohort Comparison", "T_cohort_comparison"),
        ("U", "Anti-Overfitting Audit", "U_anti_overfitting"),
        ("V", "Provider Calls", "V_provider_calls"),
        ("W", "Tests", "W_test_results"),
        ("X", "Final Decision", "X_final_decision"),
    ]
    verbose_keys = {
        "cases",
        "per_claim_and_context_detail",
        "actual_claim_normalized_values",
        "per_view",
        "per_proposition_views",
        "suspicious_preserved_cases",
        "false_preservation_review_cases",
        "ordinary_preserved_sample_claim_ids",
        "legacy_warning_cases",
        "method_context_cases",
    }
    for letter, title, key in section_order:
        value = payload["required_sections"].get(key, {})
        if isinstance(value, dict):
            concise = {field: item for field, item in value.items() if field not in verbose_keys}
            for field in verbose_keys & value.keys():
                item = value[field]
                concise[f"{field}_count"] = len(item) if isinstance(item, list | dict) else item
            value = concise
        lines += [
            f"### {letter}. {title}",
            "",
            "```json",
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
        ]
    lines += [
        "",
        "## Cross-cohort findings",
        "",
        "```json",
        json.dumps(payload["cross_cohort_findings"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Manual review queue",
        "",
        "此队列来自确定性规则；它不是人工通过/失败标签。"
        "人工复核状态仍为待审，不将 projection 结果冒充人工语义结论。",
        "",
    ]
    for claim in payload["manual_review_queue"]:
        lines.append(
            f"- `{claim['cohort']}` / `{claim['claim_id']}` — {', '.join(claim['review_reasons'])}"
        )
    lines += [
        "",
        "## Per-claim replay detail",
        "",
        "详尽 source/output graph、span、attachment、preservation outcome 与派生 assertion view "
        "见配套 JSON artifact。",
        "",
    ]
    return "\n".join(lines)


def numeric_contract_audit() -> dict[str, Any]:
    fixture_path = ROOT / "tests/fixtures/truth_condition_context_shadow_v1.json"
    fixture = load_json(fixture_path)
    cases = {item["case_id"]: item for item in fixture["cases"]}
    pair = cases["L_numeric_canonical_equivalence"]["normalization_pair"]
    equal = tcc.normalize_semantic_value(pair[0]) == tcc.normalize_semantic_value(pair[1])
    percent = tcc.normalize_semantic_value({"value": 38.4, "unit": "%"})
    fraction = tcc.normalize_semantic_value({"value": 38.4, "unit": "fraction"})
    currency_a = tcc.normalize_semantic_value({"value": 10, "unit": "USD"})
    currency_b = tcc.normalize_semantic_value({"value": 10, "unit": "EUR"})
    return {
        "diagnostic_fixture": str(fixture_path.relative_to(ROOT)),
        "same_typed_value_trailing_zero_pair": "CANONICALLY_EQUAL"
        if equal
        else "CANONICALLY_DIFFERENT",
        "different_units_percent_vs_fraction": "UNIT_MISMATCH"
        if percent != fraction
        else "CANONICALLY_EQUAL",
        "different_currency_units": "UNIT_MISMATCH"
        if currency_a != currency_b
        else "CANONICALLY_EQUAL",
        "general_unit_conversion_performed": False,
        "tolerance_inference_performed": False,
        "actual_cohort_numeric_values_are_normalized_but_not_source_reparsed": True,
        "unsupported_normalization_cases": (
            "Per-record diagnostics; typed values remain core values, not source reparses."
        ),
    }


def non_context_failures(cohorts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cohort, rows in cohorts.items():
        categories: Counter[str] = Counter()
        zero_props = 0
        for row in rows:
            record = row["raw_record"]
            props = row["propositions"]
            if not props:
                zero_props += 1
                categories["ZERO_PROPOSITION_OR_ABSTENTION"] += 1
            serialized = json.dumps(record, ensure_ascii=False, sort_keys=True).lower()
            if any(
                token in serialized
                for token in ("schema_invalid", "schema validation failed", "schema_error")
            ):
                categories["SCHEMA_INVALID"] += 1
            if any(
                token in serialized
                for token in (
                    "composition_invalid",
                    "composition_validation_failed",
                    "composition_reference_invalid",
                )
            ):
                categories["COMPOSITION_INVALID"] += 1
            if any(
                token in serialized
                for token in (
                    "source_span_invalid",
                    "source_span_mismatch",
                    "span_validation_failed",
                )
            ):
                categories["SOURCE_SPAN_INVALID"] += 1
            if any(
                token in serialized for token in ("logical_weakening", "weakened_truth_condition")
            ):
                categories["LOGICAL_WEAKENING_REPORTED"] += 1
        result[cohort] = {
            "claims": len(rows),
            "zero_proposition_claims": zero_props,
            "failure_signal_counts": dict(sorted(categories.items())),
        }
    return result


def required_sections(
    all_claims: list[dict[str, Any]],
    cohorts: dict[str, list[dict[str, Any]]],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_context_rows: list[dict[str, Any]] = []
    output_context_rows: list[dict[str, Any]] = []
    for claim in all_claims:
        outcomes = {item["source_context_ref"]: item for item in claim["context_preservation"]}
        for context in claim["source_context_graph"]["contexts"]:
            match = outcomes.get(context["context_ref"])
            source_context_rows.append(
                {
                    "cohort": claim["cohort"],
                    "claim_id": claim["claim_id"],
                    "context_ref": context["context_ref"],
                    "context_type": context["context_type"],
                    "scope_kind": context["scope"]["kind"],
                    "scope_targets": context["scope"]["target_refs"],
                    "scope_evidence": context["scope"].get("evidence"),
                    "attachment_statuses": [edge["status"] for edge in context["attachments"]],
                    "source_span": context["source_span"],
                    "semantic_value": context["semantic_value"],
                    "outcome": match["outcome"] if match else "UNMAPPED",
                    "reason_codes": match["reason_codes"] if match else [],
                    "output_context_refs": match["output_context_refs"] if match else [],
                }
            )
        for context in claim["output_context_graph"]["contexts"]:
            output_context_rows.append(
                {
                    "cohort": claim["cohort"],
                    "claim_id": claim["claim_id"],
                    "context_ref": context["context_ref"],
                    "context_type": context["context_type"],
                    "target_refs": [edge["target_ref"] for edge in context["attachments"]],
                    "attachment_statuses": [edge["status"] for edge in context["attachments"]],
                    "semantic_value": context["semantic_value"],
                    "detection_evidence": context["detection_evidence"],
                    "source_span": context["source_span"],
                }
            )

    temporal = [row for row in source_context_rows if row["context_type"] == "TEMPORAL"]
    methods = [row for row in source_context_rows if row["context_type"] == "METHOD"]
    epistemic = [row for row in source_context_rows if row["context_type"] == "EPISTEMIC"]
    preserved_suspicious = [
        row
        for row in source_context_rows
        if row["outcome"] == "PRESERVED"
        and row["context_type"]
        in {
            "TEMPORAL",
            "VERSION",
            "CONDITION",
            "MEASUREMENT",
            "SAMPLE_SCOPE",
            "METHOD",
            "EPISTEMIC",
            "SCOPE",
        }
    ]
    selected_claim_ids = {
        claim["claim_id"]
        for claim in all_claims
        if any(
            reason.startswith("ORDINARY_PRESERVED_SAMPLE:") for reason in claim["review_reasons"]
        )
    }

    final16_claims = [claim for claim in all_claims if claim["cohort"] == "final_unseen"]
    final16_mapping = {
        "shared_year_over_multiple_propositions": [
            claim["claim_id"]
            for claim in final16_claims
            if any(
                item["context_type"] == "TEMPORAL" and len(item["source_targets"]) > 1
                for item in claim["shared_contexts"]
            )
        ],
        "exact_numeric_semantic_equality": [
            claim["claim_id"]
            for claim in final16_claims
            if claim["output_numeric_normalized_values"]
        ],
        "attribution_or_epistemic_context": [
            claim["claim_id"] for claim in final16_claims if claim["epistemic_audit"]
        ],
        "method_role_or_candidate_rejection": [
            claim["claim_id"]
            for claim in final16_claims
            if any(
                item["context_type"] == "METHOD"
                for item in claim["source_context_graph"]["contexts"]
            )
            or "candidate_rejected_no_semantic_relation" in claim["source_projection_diagnostics"]
        ],
    }
    context_types = [item.value for item in tcc.ContextType]
    source_type_counts = Counter(row["context_type"] for row in source_context_rows)
    output_type_counts = Counter(row["context_type"] for row in output_context_rows)
    source_outcomes = Counter(row["outcome"] for row in source_context_rows)
    scope_counts = Counter(row["scope_kind"] for row in source_context_rows)
    source_attachment_counts = Counter(
        status for row in source_context_rows for status in row["attachment_statuses"]
    )
    output_attachment_counts = Counter(
        status for row in output_context_rows for status in row["attachment_statuses"]
    )
    qualifier_evidence_counts = Counter(
        reason for row in output_context_rows for reason in row["detection_evidence"]
    )
    no_context_claims = [
        claim for claim in all_claims if not claim["source_context_graph"]["contexts"]
    ]
    legacy_warning_claims = []
    for claim in all_claims:
        legacy = claim["metadata"].get("legacy_qualifier_status_counts", {})
        statuses = []
        if isinstance(legacy, dict):
            statuses = [str(key).upper() for key, count in legacy.items() if count]
        elif isinstance(legacy, list):
            statuses = [
                str(item.get("status", "")).upper() for item in legacy if isinstance(item, dict)
            ]
        if any(
            status in {"POTENTIAL_LOSS", "AMBIGUOUS", "FAILED", "BLOCKED"} for status in statuses
        ):
            legacy_warning_claims.append(
                {
                    "cohort": claim["cohort"],
                    "claim_id": claim["claim_id"],
                    "legacy_statuses": statuses,
                    "context_outcomes": claim["context_outcome_counts"],
                    "provisional_migration_classification": "REQUIRES_CASE_REVIEW",
                }
            )
    return {
        "C_source_context_graph_metrics": {
            "source_context_nodes_total": len(source_context_rows),
            "by_context_type_including_zero": {
                key: source_type_counts.get(key, 0) for key in context_types
            },
            "source_scope_counts": dict(sorted(scope_counts.items())),
            "source_attachment_status_counts": dict(sorted(source_attachment_counts.items())),
            "contexts_created": len(source_context_rows),
            "candidates_rejected": dict(
                sorted(
                    Counter(
                        diagnostic
                        for claim in all_claims
                        for diagnostic in claim["source_projection_diagnostics"]
                        if diagnostic.startswith("candidate_rejected")
                    ).items()
                )
            ),
            "detector_limit_claims": sum(
                not claim["source_detection_complete"] for claim in all_claims
            ),
            "zero_source_context_claims": len(no_context_claims),
            "scope_unknown": scope_counts.get("UNKNOWN", 0),
            "attachment_unknown": source_attachment_counts.get("UNKNOWN", 0),
            "source_detection_complete_true": sum(
                claim["source_detection_complete"] for claim in all_claims
            ),
        },
        "D_output_context_graph_metrics": {
            "output_context_nodes_total": len(output_context_rows),
            "by_context_type_including_zero": {
                key: output_type_counts.get(key, 0) for key in context_types
            },
            "resolved_proposition_attachments": output_attachment_counts.get("RESOLVED", 0),
            "shared_attachments": sum(len(row["target_refs"]) > 1 for row in output_context_rows),
            "unknown_attachments": output_attachment_counts.get("UNKNOWN", 0),
            "epistemic_nodes": output_type_counts.get("EPISTEMIC", 0),
            "structured_contexts_from_v3_fields": dict(sorted(qualifier_evidence_counts.items())),
            "numeric_normalized_values_in_proposition_core": sum(
                len(claim["output_numeric_normalized_values"]) for claim in all_claims
            ),
            "source_semantics_copied_into_output": False,
        },
        "E_shared_context": {
            "cases": [item for claim in all_claims for item in claim["shared_contexts"]],
            "classification_counts": dict(
                sorted(
                    Counter(
                        item["classification"]
                        for claim in all_claims
                        for item in claim["shared_contexts"]
                    ).items()
                )
            ),
            "generic_shared_year_case_count": sum(
                item["context_type"] == "TEMPORAL"
                for claim in all_claims
                for item in claim["shared_contexts"]
            ),
        },
        "F_temporal_context": {
            "cases": temporal,
            "outcome_counts": dict(sorted(Counter(row["outcome"] for row in temporal).items())),
            "shared_temporal_contexts": sum(
                row["context_type"] == "TEMPORAL"
                for claim in all_claims
                for row in claim["shared_contexts"]
            ),
            "date_range_support": (
                "Not independently inferred; only frozen-projection contexts counted."
            ),
            "false_blocks": [
                row
                for row in temporal
                if row["outcome"] in {"MISSING", "SCOPE_MISMATCH", "SEMANTIC_MISMATCH"}
            ],
            "false_preservation_review_cases": [
                row for row in preserved_suspicious if row["context_type"] == "TEMPORAL"
            ],
        },
        "G_numeric_normalization": {
            **numeric_contract_audit(),
            "actual_claim_normalized_values": [
                {"cohort": claim["cohort"], "claim_id": claim["claim_id"], **item}
                for claim in all_claims
                for item in claim["output_numeric_normalized_values"]
            ],
            "actual_claim_source_reparse": False,
            "actual_claim_value_match_classification": (
                "NOT_INFERRED_WITHOUT_INDEPENDENT_SOURCE_VALUE_PARSER"
            ),
        },
        "H_epistemic_context": {
            "cases": epistemic,
            "source_epistemic_detected_count": len(epistemic),
            "output_classification_counts": dict(
                sorted(
                    Counter(
                        item["output_classification"]
                        for claim in all_claims
                        for item in claim["epistemic_audit"]
                    ).items()
                )
            ),
            "human_review_every_case_required": True,
        },
        "I_retrieval_provenance_safety": {
            "source_records_with_retrieval_provenance": sum(
                claim["retrieval_provenance_present_in_cohort_record"] for claim in all_claims
            ),
            "retrieval_provenance_fields_forwarded_to_projection": sum(
                claim["retrieval_provenance_metadata_present_in_claim_output"]
                for claim in all_claims
            ),
            "RETRIEVAL_PROVENANCE_AS_ATTRIBUTION": 0,
            "basis": (
                "Projectors receive Claim text and proposition structure only. "
                "Cohort provenance is not supplied."
            ),
        },
        "J_method_semantic_role": {
            "method_context_cases": methods,
            "created_method_context_count": len(methods),
            "rejected_candidate_count": sum(
                value
                for claim in all_claims
                for key, value in Counter(claim["source_projection_diagnostics"]).items()
                if key == "candidate_rejected_no_semantic_relation"
            ),
            "method_candidate_classifications": (
                "TRUE_METHOD_RELATION / REFERENTIAL_METHOD_NOUN / "
                "NON_METHOD_LEXICAL_COLLISION require manual review. "
                "Detector rejection is not proof of a false positive."
            ),
            "human_review_claim_ids": sorted({row["claim_id"] for row in methods}),
        },
        "K_preservation_comparison": {
            "all_material_source_context_outcomes": dict(sorted(source_outcomes.items())),
            "per_claim_and_context_detail": source_context_rows,
            "detector_limit_kept_distinct_from_not_applicable": True,
        },
        "L_legacy_vs_context": {
            "legacy_warning_cases": legacy_warning_claims,
            "migration_classification_counts": {"REQUIRES_CASE_REVIEW": len(legacy_warning_claims)},
            "legacy_adapter_executed_as_second_truth_source": False,
            "note": (
                "Only persisted legacy qualifier outcomes are compared. "
                "No legacy result is recomputed or treated as canonical truth."
            ),
        },
        "M_final16_failure_mapping": {
            "cases_by_mechanism": final16_mapping,
            "mechanism_disposition": {
                "shared_year_over_multiple_propositions": (
                    "REPRESENTS_CORRECTLY / REMAINS_LIMITED / INTRODUCES_NEW_ERROR "
                    "require rows and human review."
                ),
                "exact_numeric_semantic_equality": (
                    "Fixture contract tested; actual source-value equivalence was not reparsed."
                ),
                "attribution_missing_from_output": (
                    "MISSING/SEMANTIC_MISMATCH or attachment uncertainty per case."
                ),
                "method_source_semantic_role": (
                    "Detected methods and rejected candidates listed; "
                    "manual semantic-role review required."
                ),
            },
        },
        "N_false_preservation_audit": {
            "suspicious_preserved_cases": preserved_suspicious,
            "ordinary_preserved_sample_claim_ids": sorted(selected_claim_ids),
            "confirmed_false_preservation_count": None,
            "human_review_status": "PENDING",
        },
        "O_unknown_quality": {
            "unknown_attachment_count": source_attachment_counts.get("UNKNOWN", 0)
            + output_attachment_counts.get("UNKNOWN", 0),
            "unknown_scope_count": scope_counts.get("UNKNOWN", 0),
            "material_unknown_views": sum(
                view["material_context_unknown"]
                for claim in all_claims
                for view in claim["verification_assertion_views"]
            ),
            "detector_limit_claim_count": len(no_context_claims),
            "unknown_rate_claim_basis": sum(
                not claim["source_detection_complete"] for claim in all_claims
            )
            / max(1, len(all_claims)),
            "classification": (
                "Distinguish source ambiguity from bounded-detector/projection limits; "
                "per-claim details retained."
            ),
        },
        "P_context_completeness": {
            "per_view": [
                {"cohort": claim["cohort"], "claim_id": claim["claim_id"], **view}
                for claim in all_claims
                for view in claim["verification_assertion_views"]
            ],
            "by_claim": {
                name: summary["context_completeness_by_claim"]
                for name, summary in summaries.items()
            },
            "separate_from_composition_eligibility": True,
        },
        "Q_verification_assertion_view": {
            "per_proposition_views": [
                {"cohort": claim["cohort"], "claim_id": claim["claim_id"], **view}
                for claim in all_claims
                for view in claim["verification_assertion_views"]
            ],
            "persistence": False,
            "relation_matching": False,
        },
        "R_v2_2_interface_diagnostic": {
            "decision": "LIMITED_INTERFACE",
            "reasons": [
                "source_detection_complete is unset; zero-context claims remain DETECTOR_LIMIT",
                "epistemic matching is signature-based and bounded by the lexical detector",
                "numeric source values are not reparsed; typed outputs are normalized",
            ],
            "available_dimensions": [
                "core identity",
                "typed value",
                "time",
                "scope",
                "configuration",
                "condition",
                "epistemic context",
                "context completeness",
                "composition eligibility",
            ],
            "relation_matching_implemented": False,
        },
        "S_non_context_failures": non_context_failures(cohorts),
        "T_cohort_comparison": summaries,
        "U_anti_overfitting": {
            "claim_ids_unique_across_cohorts": len({claim["claim_id"] for claim in all_claims})
            == 128,
            "all_cohorts_development_evidence": True,
            "fresh_validation_and_pristine_holdout_must_exclude_all_128": True,
            "claim_specific_detector_rules_added": False,
            "provider_output_regenerated": False,
        },
        "V_provider_calls": 0,
        "W_test_results": "See execution transcript; test results filled after gates.",
        "X_final_decision": "TRUTH-CONDITION CONTEXT DEVELOPMENT REPLAY INCONCLUSIVE",
    }


def run_replay() -> dict[str, Any]:
    freeze_check = verify_freeze()
    cohorts_raw = load_cohorts()
    integrity = reference_integrity_audit(cohorts_raw)
    if not integrity["passed"]:
        raise RuntimeError("reference integrity gate failed; corrected replay prohibited")
    all_claims = [claim_replay(row) for rows in cohorts_raw.values() for row in rows]
    preserved_by_type: dict[str, list[dict[str, Any]]] = {}
    for claim in all_claims:
        for context in claim["source_context_graph"]["contexts"]:
            outcome = next(
                (
                    item["outcome"]
                    for item in claim["context_preservation"]
                    if item["source_context_ref"] == context["context_ref"]
                ),
                None,
            )
            if outcome == "PRESERVED":
                preserved_by_type.setdefault(context["context_type"], []).append(claim)
    ordinary_sample_claim_ids: set[str] = set()
    for context_type, claims_with_type in preserved_by_type.items():
        selected = min(
            claims_with_type,
            key=lambda claim: sha256_bytes(f"{claim['claim_id']}|{context_type}".encode()),
        )
        ordinary_sample_claim_ids.add(selected["claim_id"])
        selected["review_reasons"] = sorted(
            set(selected["review_reasons"]) | {f"ORDINARY_PRESERVED_SAMPLE:{context_type}"}
        )
    summaries = {
        name: summarize_claims([claim for claim in all_claims if claim["cohort"] == name])
        for name in cohorts_raw
    }
    total_summary = summarize_claims(all_claims)
    findings = {
        "total_claims": len(all_claims),
        "unique_claim_ids": len({claim["claim_id"] for claim in all_claims}),
        "provider_calls": 0,
        "database_writes": 0,
        "holdout_content_read": False,
        "composition_eligibility_is_separate_projection": True,
        "context_completeness": total_summary["context_completeness_by_claim"],
        "preservation_outcomes": total_summary["preservation_outcomes"],
        "source_context_types": total_summary["source_context_counts_by_type"],
        "source_diagnostics": total_summary["source_projection_diagnostics"],
        "composition_eligibility": total_summary[
            "composition_eligibility_counts_independent_of_context"
        ],
        "shared_context_occurrences": total_summary["shared_context_occurrences"],
        "epistemic_source_contexts": total_summary["source_epistemic_contexts"],
        "retrieval_provenance_metadata_claims": total_summary[
            "retrieval_provenance_metadata_claim_count"
        ],
        "legacy_qualifier_comparison_available": {
            name: sum(
                bool(claim["metadata"].get("legacy_qualifier_status_counts"))
                for claim in cohort_claims
            )
            for name, cohort_claims in (
                (n, [c for c in all_claims if c["cohort"] == n]) for n in cohorts_raw
            )
        },
        "source_span_availability": {
            "proposition_count": total_summary["propositions"],
            "with_source_span": sum(
                bool(prop.get("source_span"))
                for claim in all_claims
                for prop in claim["propositions"]
            ),
            "without_source_span": sum(
                not bool(prop.get("source_span"))
                for claim in all_claims
                for prop in claim["propositions"]
            ),
        },
        "output_context_types": total_summary["output_context_counts_by_type"],
        "source_scope_counts": total_summary["source_scope_counts"],
        "source_attachment_status_counts": total_summary["source_attachment_status_counts"],
        "output_attachment_status_counts": total_summary["output_attachment_status_counts"],
        "shared_context_classifications": total_summary["shared_context_classifications"],
        "epistemic_context_classifications": total_summary["epistemic_context_classifications"],
        "unknown_attachment_count": total_summary["unknown_attachment_count"],
        "unknown_scope_count": total_summary["unknown_scope_count"],
        "detector_limit_claim_count": total_summary["detector_limit_claim_count"],
        "material_unknown_view_count": total_summary["material_unknown_view_count"],
        "material_missing_view_count": total_summary["material_missing_view_count"],
        "numeric_normalization_contract": numeric_contract_audit(),
        "non_context_failures_by_cohort": non_context_failures(cohorts_raw),
        "development_claim_ids_sha256": canonical_hash(
            sorted(claim["claim_id"] for claim in all_claims)
        ),
    }
    governance = {
        "cohort_counts": {name: len(rows) for name, rows in cohorts_raw.items()},
        "claim_ids_unique_and_disjoint": True,
        "development_evidence_only": True,
        "future_fresh_validation_and_pristine_holdout_exclude_all_128": True,
        "manifest_files_read": [
            str(path.relative_to(ROOT))
            for path in (
                PATHS["bounded100_manifest"],
                PATHS["fresh12_manifest"],
                PATHS["final16_manifest"],
            )
        ],
        "holdout_content_read": False,
        "database_read_or_write": False,
        "provider_calls": 0,
    }
    sections = required_sections(all_claims, cohorts_raw, summaries)
    sections.update(
        {
            "A_frozen_versions": load_json(FREEZE_PATH),
            "B_dataset_governance": governance,
            "reference_reconciliation_integrity": integrity,
            "V_provider_calls": 0,
            "W_test_results": "See final report verification block.",
            "X_final_decision": "TRUTH-CONDITION CONTEXT DEVELOPMENT REPLAY INCONCLUSIVE",
        }
    )
    payload = {
        "report_title": "V2.1R Truth-Condition Context Development Replay — Phase C",
        "replay_id": "v2_1r_truth_condition_context_128_replay_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "freeze_verification": freeze_check,
        "reference_reconciliation_integrity": integrity,
        "frozen_implementation": load_json(FREEZE_PATH),
        "cohort_counts": {name: len(rows) for name, rows in cohorts_raw.items()},
        "total_claims": len(all_claims),
        "cohort_summaries": summaries,
        "cross_cohort_findings": findings,
        "dataset_governance": governance,
        "required_sections": sections,
        "manual_review_queue": [
            {
                "cohort": claim["cohort"],
                "claim_id": claim["claim_id"],
                "review_reasons": claim["review_reasons"],
            }
            for claim in all_claims
            if claim["review_reasons"]
        ],
        "manual_review_scope": {
            "all_source_epistemic_cases": True,
            "all_suspicious_preserved_cases": True,
            "ordinary_preserved_sample_rule": (
                "Lowest SHA256(claim_id + context_type) per context type; fixed before labels."
            ),
            "human_labels": "NOT_YET_ASSIGNED; deterministic queue is not a human judgment",
        },
        "claim_replays": all_claims,
    }
    if payload["total_claims"] != 128 or findings["unique_claim_ids"] != 128:
        raise RuntimeError("replay output does not contain exactly 128 unique claims")
    if not verify_freeze()["passed"]:
        raise RuntimeError("post-replay frozen hashes changed")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "freeze", "replay"))
    args = parser.parse_args()
    if args.mode == "audit":
        cohorts = load_cohorts()
        payload = reference_integrity_audit(cohorts)
        INTEGRITY_JSON.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        lines = [
            "# Phase C Reference Reconciliation Integrity Audit",
            "",
            f"- Status: **{payload['status']}**",
            "- Provider calls: **0**",
            "- DB writes: **0**",
            "- Holdout read: **No**",
            "- Mapping contract: `local_ref → runtime proposition ID`",
            "",
            "| Cohort | Props | Elig. entries | Local maps | Reconciled | Missing result | "
            "Orphans | Dups | Blocking |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, item in payload["cohorts"].items():
            lines.append(
                f"| {name} | {item.get('propositions', 0)} | "
                f"{item.get('eligibility_source_entries', 0)} | "
                f"{item.get('local_ref_mappings', 0)} | "
                f"{item.get('reconciled_propositions', 0)} | "
                f"{item.get('unresolved_eligibility', 0)} | "
                f"{item.get('orphan_eligibility_entries', 0)} | "
                f"{item.get('duplicate_mappings', 0)} | {item.get('blocking_issue_count', 0)} |"
            )
        INTEGRITY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        marker = {
            "status": "NON_FINAL",
            "decision_eligibility": "INVALID_FOR_PHASE_C_DECISION",
            "reason": "ANALYZER_REFERENCE_MAPPING_BUG",
            "preserved_outputs": [
                "artifacts/v2_1r_truth_condition_context_128_replay.json",
                "artifacts/v2_1r_truth_condition_context_128_replay.md",
            ],
            "replacement_outputs_created": False,
        }
        INVALID_RUN_MARKER.write_text(
            json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "json": str(INTEGRITY_JSON.relative_to(ROOT)),
                    "markdown": str(INTEGRITY_MD.relative_to(ROOT)),
                    "old_outputs_preserved": True,
                    "provider_calls": 0,
                },
                ensure_ascii=False,
            )
        )
        return 0 if payload["passed"] else 2
    if args.mode == "freeze":
        if FREEZE_PATH.exists():
            raise SystemExit(f"freeze already exists; refusing overwrite: {FREEZE_PATH}")
        payload = freeze_payload()
        FREEZE_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "frozen": True,
                    "path": str(FREEZE_PATH),
                    "input_files": len(payload["input_artifact_sha256"]),
                    "source_files": len(payload["source_file_sha256"]),
                    "provider_calls": 0,
                },
                ensure_ascii=False,
            )
        )
        return 0
    payload = run_replay()
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MD_OUT.write_text(markdown_report(payload) + "\n", encoding="utf-8")
    marker = load_json(INVALID_RUN_MARKER)
    marker["replacement_outputs_created"] = True
    marker["corrected_outputs"] = [
        str(JSON_OUT.relative_to(ROOT)),
        str(MD_OUT.relative_to(ROOT)),
    ]
    INVALID_RUN_MARKER.write_text(
        json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "replay_completed": True,
                "claims": payload["total_claims"],
                "json": str(JSON_OUT),
                "markdown": str(MD_OUT),
                "provider_calls": 0,
                "database_writes": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
