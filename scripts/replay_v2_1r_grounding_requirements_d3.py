"""Analysis-only D3 replay for frozen V2.1R development artifacts.

This adapter is intentionally read-only with respect to production systems.
It consumes only frozen JSON artifacts, calls the deterministic shadow
requirement engine, and writes new D3 analysis artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.domain.grounding_requirements import (
    GROUNDING_REQUIREMENT_CONTRACT_VERSION,
    GroundingBindingCandidate,
    GroundingDerivationStatus,
    GroundingNodeInput,
    GroundingNodeKind,
    GroundingNodeUsability,
    GroundingObligationStatus,
    GroundingRequirementLevel,
    ZeroObligationReason,
)
from app.domain.semantic_support_verification import (
    SEMANTIC_SUPPORT_REQUIREDNESS_CONTRACT_VERSION,
    SEMANTIC_SUPPORT_ROUTER_VERSION,
    SEMANTIC_SUPPORT_VERIFIER_PROMPT_VERSION,
    SEMANTIC_SUPPORT_VERIFIER_SCHEMA_VERSION,
)
from app.domain.source_context_extraction import (
    SOURCE_CONTEXT_EXTRACTION_PROMPT_VERSION,
    SOURCE_CONTEXT_EXTRACTION_SCHEMA_VERSION,
    SOURCE_CONTEXT_NORMALIZER_VERSION,
)
from app.domain.source_context_grounding_v2 import (
    SOURCE_CONTEXT_GROUNDING_PROMPT_V2,
    SOURCE_CONTEXT_GROUNDING_SCHEMA_V2,
    GroundingDimensionStatus,
)
from app.services.grounding_requirement_engine import (
    derive_grounding_obligations,
    evaluate_grounding_node,
)

from scripts.v2_1r_d3_adapter import (
    EXPECTED_SOURCE_SCHEMA,
    NORMALIZATION_CONTRACT_VERSION,
    NormalizedReplayBindingView,
    normalize_replay_binding,
    normalized_view_from_dict,
    resolve_omission_role,
)
from scripts.v2_1r_d3_report_contracts import (
    canonical_sha256,
    cardinality_report,
    jsonable,
    validate_report_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
ATTEMPT_ID = "D3_FORMAL_ATTEMPT_2"
ADAPTER_VERSION = "grounding-requirements-d3-analysis-adapter.v2"
CANARY_PATH = Path(
    "artifacts/v2_1r_source_semantic_grounding_evidence_v2_real_development_canary_result.json"
)
SELECTION_PATH = Path(
    "artifacts/v2_1r_source_semantic_grounding_evidence_v2_real_development_canary_selection_manifest.json"
)
ROOT_AUDIT_PATH = Path("artifacts/v2_1r_semantic_support_root_cause_audit.json")
PHASE_C_PATH = Path("artifacts/v2_1r_hybrid_semantic_support_phase_c.json")
PHASE_C_HASHES_PATH = Path("artifacts/v2_1r_hybrid_semantic_support_phase_c_hashes.json")
OUTPUTS = {
    "node_ledger": Path("artifacts/v2_1r_grounding_requirement_d3_node_ledger.json"),
    "obligation_ledger": Path("artifacts/v2_1r_grounding_requirement_d3_obligation_ledger.json"),
    "binding_ledger": Path("artifacts/v2_1r_grounding_requirement_d3_binding_matching_ledger.json"),
    "omission_comparison": Path(
        "artifacts/v2_1r_grounding_requirement_d3_known_omission_comparison.json"
    ),
    "cardinality": Path("artifacts/v2_1r_grounding_requirement_d3_cardinality_report.json"),
    "requiredness": Path("artifacts/v2_1r_grounding_requirement_d3_requiredness_stability.json"),
    "usability": Path("artifacts/v2_1r_grounding_requirement_d3_node_usability_summary.json"),
    "readiness": Path("artifacts/v2_1r_grounding_requirement_d3_verification_readiness.json"),
    "coverage": Path("artifacts/v2_1r_grounding_requirement_d3_contract_coverage.json"),
    "provenance": Path("artifacts/v2_1r_grounding_requirement_d3_provenance_audit.json"),
    "report_json": Path("artifacts/v2_1r_grounding_requirement_d3_offline_replay.json"),
    "report_markdown": Path("artifacts/v2_1r_grounding_requirement_d3_offline_replay.md"),
}
OUTPUTS = {key: Path(str(path).replace("_d3_", "_d3_attempt_2_")) for key, path in OUTPUTS.items()}
HASH_MANIFEST = Path("artifacts/v2_1r_grounding_requirement_d3_attempt_2_hash_manifest.json")
PREVIOUS_HASH_MANIFEST = Path("artifacts/v2_1r_grounding_requirement_d3_hash_manifest.json")

IMPLEMENTATION_PATHS = (
    Path("apps/api/app/domain/grounding_requirements.py"),
    Path("apps/api/app/services/grounding_requirement_engine.py"),
    Path("apps/api/app/domain/semantic_support_verification.py"),
    Path("apps/api/app/services/hybrid_semantic_support.py"),
    Path("apps/api/app/domain/source_context_grounding_v2.py"),
    Path("apps/api/app/domain/source_context_extraction.py"),
    Path("apps/api/app/services/source_context_extractor_v2.py"),
    Path("scripts/v2_1r_d3_adapter.py"),
    Path("scripts/replay_v2_1r_grounding_requirements_d3.py"),
)
SEMANTIC_IMPLEMENTATION_PATHS = IMPLEMENTATION_PATHS[:7]
ARTIFACT_CONTRACT_INVENTORY = {
    "source_node_replay_row": {
        "fields": [
            "claim_id",
            "source_node_ref",
            "source_node_kind",
            "source_semantic_payload",
            "engine_input_node",
        ],
        "source": "frozen Canary claim.grounding_result source node collections",
        "semantic_role_direct": False,
    },
    "normalized_binding_replay_row": {
        "fields": list(NormalizedReplayBindingView.__dataclass_fields__),
        "source": "binding_ref lookup in frozen Canary evidence_bindings; Phase C support lookup",
        "semantic_role_direct": True,
    },
    "obligation_provenance_row": {
        "fields": [
            "node_ref",
            "contract_version",
            "derivation_rule_id",
            "obligation_ref",
            "obligation_status",
        ],
        "source": "GroundingObligationEvaluation.provenance",
        "semantic_role_direct": False,
    },
    "binding_to_obligation_match_row": {
        "fields": [
            "candidate_bindings",
            "matched_bindings",
            "excluded_candidates",
            "cardinality_satisfied",
        ],
        "source": "deterministic requirement engine match result",
        "semantic_role_direct": False,
    },
    "omission_comparison_row": {
        "fields": ["known_omission_kind", "canonical_role_resolution", "classification"],
        "source": "frozen audit annotations compared after derivation",
        "semantic_role_direct": False,
    },
    "phase_c_binding_row": {
        "fields": [
            "claim_id",
            "binding_ref",
            "semantic_role",
            "final_oracle_simulated_semantic_support",
            "oracle_provenance",
        ],
        "source": "frozen Phase C routing_ledger",
        "semantic_role_direct": True,
    },
    "root_cause_audit_binding_row": {
        "fields": [
            "claim_id",
            "binding_ref",
            "semantic_role",
            "target_kind",
            "target_ref",
            "human_support_judgment",
        ],
        "source": "frozen root-cause audit binding_ledger; comparison-only",
        "semantic_role_direct": True,
    },
}
INPUT_PATHS = (
    CANARY_PATH,
    SELECTION_PATH,
    ROOT_AUDIT_PATH,
    PHASE_C_PATH,
    PHASE_C_HASHES_PATH,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    return canonical_sha256(value)


def _read_json(path: Path) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    destination = ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if path in OUTPUTS.values() and isinstance(payload, dict):
        payload.setdefault("attempt_id", ATTEMPT_ID)
        payload.setdefault("analysis_adapter_version", ADAPTER_VERSION)
        payload.setdefault("normalization_contract_version", NORMALIZATION_CONTRACT_VERSION)
    destination.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _versions() -> dict[str, str]:
    return {
        "grounding_requirement_contract": GROUNDING_REQUIREMENT_CONTRACT_VERSION,
        "hybrid_router": SEMANTIC_SUPPORT_ROUTER_VERSION,
        "semantic_support_requiredness": SEMANTIC_SUPPORT_REQUIREDNESS_CONTRACT_VERSION,
        "semantic_support_verifier_prompt": SEMANTIC_SUPPORT_VERIFIER_PROMPT_VERSION,
        "semantic_support_verifier_schema": SEMANTIC_SUPPORT_VERIFIER_SCHEMA_VERSION,
        "source_grounding_v2_prompt": SOURCE_CONTEXT_GROUNDING_PROMPT_V2,
        "source_grounding_v2_schema": SOURCE_CONTEXT_GROUNDING_SCHEMA_V2,
        "source_extraction_prompt": SOURCE_CONTEXT_EXTRACTION_PROMPT_VERSION,
        "source_extraction_schema": SOURCE_CONTEXT_EXTRACTION_SCHEMA_VERSION,
        "source_context_normalizer": SOURCE_CONTEXT_NORMALIZER_VERSION,
        "d3_analysis_adapter": ADAPTER_VERSION,
        "d3_normalization_contract": NORMALIZATION_CONTRACT_VERSION,
    }


def freeze_manifest() -> dict[str, Any]:
    implementation_hashes = {
        str(path).replace("\\", "/"): _sha256(path) for path in IMPLEMENTATION_PATHS
    }
    input_hashes = {str(path).replace("\\", "/"): _sha256(path) for path in INPUT_PATHS}
    phase_c = _read_json(PHASE_C_PATH)
    phase_c_frozen = phase_c["frozen_versions"]
    previous = _read_json(PREVIOUS_HASH_MANIFEST)
    previous_hashes = previous["implementation_hashes"]
    semantic_hashes = {
        str(path).replace("\\", "/"): implementation_hashes[str(path).replace("\\", "/")]
        for path in SEMANTIC_IMPLEMENTATION_PATHS
    }
    previous_semantic_hashes = {
        str(path).replace("\\", "/"): previous_hashes[str(path).replace("\\", "/")]
        for path in SEMANTIC_IMPLEMENTATION_PATHS
    }
    if semantic_hashes != previous_semantic_hashes:
        raise RuntimeError("semantic_implementation_freeze_changed_since_attempt_1")
    fingerprint_payload = {
        "implementation_hashes": implementation_hashes,
        "input_hashes": input_hashes,
        "versions": _versions(),
        "phase_c_frozen_versions": phase_c_frozen,
    }
    manifest = {
        "artifact_version": "v2_1r_grounding_requirement_d3_attempt_2_hash_manifest.v1",
        "attempt_id": ATTEMPT_ID,
        "previous_attempt": {
            "attempt_id": "D3_FORMAL_ATTEMPT_1",
            "status": "INVALID_FOR_SEMANTIC_DECISION",
            "reason": "ANALYSIS_ADAPTER_CONTRACT_FAILURE",
            "stage": "KNOWN_OMISSION_COMPARISON",
            "error_category": "MISSING_EXPECTED_REPORT_FIELD",
        },
        "adapter_version": ADAPTER_VERSION,
        "frozen_before_formal_replay": True,
        "provider_calls": 0,
        "external_http": 0,
        "semantic_verifier_calls": 0,
        "extractor_calls": 0,
        "implementation_hashes": implementation_hashes,
        "input_hashes": input_hashes,
        "versions": _versions(),
        "phase_c_frozen_versions": phase_c_frozen,
        "normalized_implementation_fingerprint": _canonical_hash(fingerprint_payload),
        "semantic_implementation_fingerprint": _canonical_hash(semantic_hashes),
        "semantic_implementation_hashes": semantic_hashes,
        "previous_attempt_semantic_fingerprint": _canonical_hash(previous_semantic_hashes),
        "semantic_freeze_unchanged": True,
        "normalization_contract_version": NORMALIZATION_CONTRACT_VERSION,
        "artifact_contract_inventory": ARTIFACT_CONTRACT_INVENTORY,
        "replay_script_sha256": implementation_hashes[
            "scripts/replay_v2_1r_grounding_requirements_d3.py"
        ],
        "normalization_helper_sha256": implementation_hashes["scripts/v2_1r_d3_adapter.py"],
        "selection_manifest_normalized_sha256": (
            "4aaa89330b5212537a4ac389f570038e8a3bd27bf9bc3fb6a58161d1c01c511a"
        ),
    }
    _write_json(HASH_MANIFEST, manifest)
    return manifest


def _verify_frozen_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = _read_json(HASH_MANIFEST)
    if not manifest.get("frozen_before_formal_replay"):
        raise RuntimeError("hash_manifest_not_frozen_before_replay")
    for relpath, expected in manifest["implementation_hashes"].items():
        if _sha256(Path(relpath)) != expected:
            raise RuntimeError(f"frozen_implementation_hash_mismatch:{relpath}")
    for relpath, expected in manifest["input_hashes"].items():
        if _sha256(Path(relpath)) != expected:
            raise RuntimeError(f"frozen_input_hash_mismatch:{relpath}")
    return manifest


def _status(value: Any) -> GroundingDimensionStatus:
    try:
        return GroundingDimensionStatus(str(value))
    except ValueError:
        return GroundingDimensionStatus.NOT_EVALUATED


def _jsonable(value: Any) -> Any:
    return jsonable(value)


def _graph_maps(grounding: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {
        "ASSERTION": {},
        "CONTEXT": {},
        "ATTACHMENT": {},
        "SCOPE": {},
    }
    for field, kind, key in (
        ("source_assertions", "ASSERTION", "source_assertion_ref"),
        ("source_contexts", "CONTEXT", "context_ref"),
        ("source_attachments", "ATTACHMENT", "attachment_ref"),
        ("source_scopes", "SCOPE", "source_scope_ref"),
    ):
        for raw in grounding.get(field, []):
            result[kind][str(raw[key])] = raw
    return result


def _context_links(
    grounding: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    links: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attachment in grounding.get("source_attachments", []):
        links[str(attachment.get("context_ref", ""))].append(attachment)
    return links


def _node_records(
    claim_id: str,
    grounding: dict[str, Any],
) -> list[dict[str, Any]]:
    graph = _graph_maps(grounding)
    links = _context_links(grounding)
    records: list[dict[str, Any]] = []
    for source_kind, nodes in graph.items():
        for node_ref, raw in nodes.items():
            adapter_status = "RESOLVED"
            adapter_notes: list[str] = []
            if source_kind == "ASSERTION":
                input_kind = GroundingNodeKind.ASSERTION
                payload = dict(raw)
                if "assertion_content" not in raw:
                    adapter_status = "LEGACY_INPUT_INSUFFICIENT"
                    adapter_notes.append("historical_source_assertion_missing_assertion_content")
                graph_role = "SOURCE_ASSERTION_NODE"
            elif source_kind == "CONTEXT":
                input_kind = GroundingNodeKind.CONTEXT
                payload = dict(raw)
                context_type = str(raw.get("context_type", ""))
                semantic_value = raw.get("semantic_value")
                if context_type == "METHOD" and isinstance(semantic_value, dict):
                    if "method" in semantic_value:
                        payload["method_value"] = semantic_value["method"]
                    context_attachments = links.get(node_ref, [])
                    relations = sorted({str(item.get("relation")) for item in context_attachments})
                    targets = sorted(
                        {
                            str(target)
                            for item in context_attachments
                            for target in item.get("source_assertion_refs", [])
                        }
                    )
                    if len(relations) == 1 and targets:
                        payload["target_relation"] = relations[0]
                        payload["target_refs"] = targets
                    elif len(relations) > 1:
                        adapter_status = "INPUT_ADAPTER_UNRESOLVED"
                        adapter_notes.append("method_context_has_multiple_attachment_relations")
                graph_role = "SOURCE_CONTEXT_NODE"
            elif source_kind == "ATTACHMENT":
                input_kind = GroundingNodeKind.ATTACHMENT
                payload = dict(raw)
                targets = raw.get("source_assertion_refs", [])
                payload["target_refs"] = list(targets) if isinstance(targets, list) else []
                if not payload["target_refs"]:
                    adapter_status = "INPUT_ADAPTER_UNRESOLVED"
                    adapter_notes.append("attachment_has_no_explicit_assertion_target")
                graph_role = "SOURCE_ATTACHMENT_NODE"
            else:
                # SourceScopeV2 records graph membership, not an independent semantic value.
                input_kind = GroundingNodeKind.GROUP
                members = raw.get("source_assertion_refs", [])
                payload = {
                    "group_type": "STRUCTURAL_GROUP",
                    "member_refs": list(members) if isinstance(members, list) else [],
                    "source_scope_kind": raw.get("kind"),
                }
                if not payload["member_refs"]:
                    adapter_status = "INPUT_ADAPTER_UNRESOLVED"
                    adapter_notes.append("source_scope_has_no_explicit_member_refs")
                graph_role = "SOURCE_SCOPE_MEMBERSHIP"

            input_node = GroundingNodeInput(
                node_ref=node_ref,
                node_kind=input_kind,
                semantic_payload=payload,
                graph_role=graph_role,
            )
            assessment = next(
                (
                    item
                    for item in grounding.get("node_assessments", [])
                    if str(item.get("target_kind")) == source_kind
                    and str(item.get("target_ref")) == node_ref
                ),
                None,
            )
            dimensions = (assessment or {}).get("dimensions", {})
            records.append(
                {
                    "claim_id": claim_id,
                    "source_node_ref": node_ref,
                    "source_node_kind": source_kind,
                    "source_semantic_payload": _jsonable(raw),
                    "engine_input_node": input_node,
                    "graph_role": graph_role,
                    "adapter_status": adapter_status,
                    "adapter_notes": adapter_notes,
                    "node_textual_grounding": _status(dimensions.get("textual_grounding")),
                    "node_structural_grounding": _status(dimensions.get("structural_grounding")),
                    "source_node_assessment_present": assessment is not None,
                }
            )
    return records


def _support_index(phase_c: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row["claim_id"]), str(row["binding_ref"])): row
        for row in phase_c.get("routing_ledger", [])
    }


def _candidate_payload(
    binding: dict[str, Any], graph: dict[str, dict[str, dict[str, Any]]]
) -> tuple[dict[str, Any], str | None, str | None, list[str]]:
    kind = str(binding.get("target_kind", ""))
    target_ref = str(binding.get("target_ref", ""))
    raw = graph.get(kind, {}).get(target_ref)
    if raw is None:
        return {}, None, None, ["binding_target_not_found_in_frozen_source_graph"]
    payload = dict(raw)
    relation: str | None = None
    scope_ref = raw.get("source_scope_ref")
    notes: list[str] = []
    if kind == "CONTEXT":
        if raw.get("context_type") == "METHOD":
            semantic_value = raw.get("semantic_value")
            if isinstance(semantic_value, dict) and "method" in semantic_value:
                # Exact field projection from the frozen node, not semantic inference.
                payload["semantic_value"] = semantic_value["method"]
    elif kind == "ATTACHMENT":
        relation = str(raw.get("relation")) if raw.get("relation") is not None else None
        targets = raw.get("source_assertion_refs", [])
        if isinstance(targets, list) and len(targets) == 1:
            # A scalar target is exposed only for an explicit singleton edge.
            payload["target_ref"] = targets[0]
        elif isinstance(targets, list) and len(targets) > 1:
            notes.append("multi_target_attachment_not_scalarized_or_cloned")
    return payload, relation, str(scope_ref) if scope_ref else None, notes


def _build_candidates(
    claim: dict[str, Any],
    phase_c_support: dict[tuple[str, str], dict[str, Any]],
    source_schema_version: str,
) -> tuple[tuple[GroundingBindingCandidate, ...], list[dict[str, Any]]]:
    claim_id = str(claim["claim_id"])
    grounding = claim.get("grounding_result") or {}
    graph = _graph_maps(grounding)
    candidates: list[GroundingBindingCandidate] = []
    provenance: list[dict[str, Any]] = []
    phase_c_rows = [
        row
        for (support_claim_id, _), row in phase_c_support.items()
        if support_claim_id == claim_id
    ]
    for binding in grounding.get("evidence_bindings", []):
        binding_ref = str(binding["binding_ref"])
        support = phase_c_support.get((claim_id, binding_ref))
        normalized = normalize_replay_binding(
            claim_id=claim_id,
            binding_ref=binding_ref,
            canonical_bindings=grounding.get("evidence_bindings", []),
            phase_c_rows=phase_c_rows,
            source_artifact=str(CANARY_PATH).replace("\\", "/"),
            schema_version=source_schema_version,
        )
        if normalized.normalization_status != "RESOLVED" or not normalized.semantic_role:
            raise RuntimeError(
                f"normalized_binding_unresolved:{claim_id}:{binding_ref}:"
                f"{normalized.normalization_status}"
            )
        payload, relation, scope_ref, adapter_notes = _candidate_payload(binding, graph)
        textual = _status(binding.get("dimensions", {}).get("textual_grounding"))
        structural = _status(binding.get("dimensions", {}).get("structural_grounding"))
        support_status = _status((support or {}).get("final_oracle_simulated_semantic_support"))
        support_source = str((support or {}).get("oracle_provenance", "MISSING_FROZEN_SUPPORT"))
        if support is None:
            support_status = GroundingDimensionStatus.NOT_EVALUATED
        candidates.append(
            GroundingBindingCandidate(
                binding_ref=binding_ref,
                evidence_ref=str(binding["evidence_ref"]),
                semantic_role=normalized.semantic_role,
                target_kind=str(binding["target_kind"]),
                target_ref=str(binding["target_ref"]),
                canonical_payload=payload,
                relation=relation,
                scope_ref=scope_ref,
                member_key=None,
                value_key=None,
                textual_grounding=textual,
                structural_grounding=structural,
                semantic_support=support_status,
            )
        )
        provenance.append(
            {
                "claim_id": claim_id,
                "binding_ref": binding_ref,
                "semantic_role": normalized.semantic_role,
                "target_kind": normalized.target_kind,
                "target_ref": normalized.target_ref,
                "normalized_role_view": normalized.to_dict(),
                "normalization_status": normalized.normalization_status,
                "normalization_source_artifact": normalized.source_artifact,
                "normalization_source_ref": normalized.source_ref,
                "normalization_schema_version": normalized.schema_version,
                "evidence_ref": str(binding["evidence_ref"]),
                "hybrid_semantic_support": support_status.value,
                "hybrid_support_provenance": support_source,
                "support_result_present": support is not None,
                "textual_grounding": textual.value,
                "structural_grounding": structural.value,
                "source_binding_support_status_not_used_as_hybrid": binding.get("support_status"),
                "adapter_notes": adapter_notes,
                "candidate_payload_source": "frozen_source_graph_target_object",
            }
        )
    return tuple(candidates), provenance


def _write_json_ledgers(
    *,
    manifest: dict[str, Any],
    claims: list[dict[str, Any]],
    phase_c: dict[str, Any],
    source_schema_version: str,
    publish: bool = True,
    compute_semantic_decision: bool = True,
    execution_mode: str = "FORMAL",
) -> dict[str, Any]:
    expected_ids = set(claims_item["claim_id"] for claims_item in claims)
    selection = _read_json(SELECTION_PATH)
    selected = set(selection.get("selection", {}).get("selected_claim_ids", []))
    if selected and selected != expected_ids:
        raise RuntimeError("frozen_canary_ids_do_not_match_selection_manifest")
    if len(claims) != 8 or len(expected_ids) != 8:
        raise RuntimeError("expected_exactly_eight_frozen_development_claims")

    phase_support = _support_index(phase_c)
    phase_support_keys = set(phase_support)
    node_records: list[dict[str, Any]] = []
    binding_records: list[dict[str, Any]] = []
    for claim in claims:
        claim_id = str(claim["claim_id"])
        grounding = claim.get("grounding_result") or {}
        binding_candidates, support_provenance = _build_candidates(
            claim, phase_support, source_schema_version
        )
        by_binding_ref = {item.binding_ref: item for item in binding_candidates}
        node_records.extend(_node_records(claim_id, grounding))
        binding_records.extend(support_provenance)
        if len(by_binding_ref) != len(binding_candidates):
            raise RuntimeError(f"duplicate_binding_ref_within_claim:{claim_id}")

    if len(node_records) != 72:
        raise RuntimeError(f"expected_72_source_nodes_got_{len(node_records)}")
    if len(binding_records) != 75:
        raise RuntimeError(f"expected_75_source_bindings_got_{len(binding_records)}")
    source_binding_keys = {
        (str(claim["claim_id"]), str(binding["binding_ref"]))
        for claim in claims
        for binding in (claim.get("grounding_result") or {}).get("evidence_bindings", [])
    }
    if source_binding_keys != phase_support_keys:
        raise RuntimeError("source_binding_and_phase_c_support_identity_mismatch")

    obligations_rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    derivation_statuses: Counter[str] = Counter()
    obligation_states: Counter[str] = Counter()
    usability_states: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    support_by_binding: dict[tuple[str, str], dict[str, Any]] = {
        (row["claim_id"], row["binding_ref"]): row for row in binding_records
    }
    binding_usage: dict[tuple[str, str], dict[str, Any]] = {
        (row["claim_id"], row["binding_ref"]): {
            "presented_to_obligations": 0,
            "candidate_obligation_refs": [],
            "matched_obligation_refs": [],
            "reuse_groups": [],
            "excluded_reasons": Counter(),
        }
        for row in binding_records
    }
    claims_by_id = {str(claim["claim_id"]): claim for claim in claims}

    # Formal first pass starts here. The known omission comparison files are not read
    # until every node and binding has been derived and evaluated.
    for record in node_records:
        claim_id = record["claim_id"]
        source_node = record["engine_input_node"]
        derivation = derive_grounding_obligations(source_node)
        derivation_statuses[derivation.status.value] += 1
        shape_counts[f"{record['source_node_kind']}->{source_node.node_kind.value}"] += 1
        candidates, _ = _build_candidates(
            claims_by_id[claim_id], phase_support, source_schema_version
        )
        assessment = evaluate_grounding_node(
            derivation,
            candidates,
            node_textual_grounding=record["node_textual_grounding"],
            node_structural_grounding=record["node_structural_grounding"],
        )
        raw_engine_usability = assessment.usability.value
        final_usability = raw_engine_usability
        if record["adapter_status"] != "RESOLVED":
            # Keep the engine result for audit, but do not let a lossy adapter pass.
            final_usability = GroundingNodeUsability.NOT_EVALUATED.value

        counts = Counter(
            item.obligation.requirement_level.value for item in assessment.obligation_evaluations
        )
        matched_bindings = sorted(
            {
                binding_ref
                for item in assessment.obligation_evaluations
                for binding_ref in item.match.matched_binding_refs
            }
        )
        unmatched_obligations = [
            item.obligation.obligation_ref
            for item in assessment.obligation_evaluations
            if not item.match.matched_binding_refs
        ]
        candidate_refs = sorted({item.binding_ref for item in candidates})
        node_obligation_states = Counter(
            item.status.value for item in assessment.obligation_evaluations
        )
        obligation_states.update(node_obligation_states)
        usability_states[final_usability] += 1

        cardinality_rows: list[dict[str, Any]] = []
        for evaluation in assessment.obligation_evaluations:
            obligation = evaluation.obligation
            status = evaluation.status.value
            match = evaluation.match
            obligation_states[status] += 0
            support_sources = [
                support_by_binding.get((claim_id, binding_ref), {}).get(
                    "hybrid_support_provenance", "MISSING"
                )
                for binding_ref in match.matched_binding_refs
            ]
            row = {
                "claim_id": claim_id,
                "obligation_ref": obligation.obligation_ref,
                "node_ref": obligation.node_ref,
                "node_kind": obligation.node_kind.value,
                "obligation_type": obligation.obligation_type.value,
                "requirement_level": obligation.requirement_level.value,
                "semantic_role_family": list(obligation.expected_semantic_role_family),
                "target_constraint": obligation.target_constraint.model_dump(mode="json"),
                "cardinality_policy": obligation.cardinality_policy.value,
                "expected_member_key": obligation.expected_member_key,
                "expected_target_key": obligation.expected_target_key,
                "expected_value_key": obligation.expected_value_key,
                "derivation_rule_id": obligation.derivation_rule_id,
                "candidate_bindings": list(match.candidate_binding_refs),
                "matched_bindings": list(match.matched_binding_refs),
                "excluded_candidates": [
                    item.model_dump(mode="json") for item in match.excluded_candidates
                ],
                "cardinality_satisfied": match.cardinality_satisfied,
                "duplicate_evidence_refs": list(match.duplicate_evidence_refs),
                "textual_grounding_state": [item[1] for item in evaluation.binding_statuses],
                "structural_grounding_state": [item[2] for item in evaluation.binding_statuses],
                "hybrid_semantic_support_state": [item[3] for item in evaluation.binding_statuses],
                "hybrid_support_provenance": support_sources,
                "final_obligation_state": status,
                "omission_diagnostic": next(
                    (
                        item.model_dump(mode="json")
                        for item in assessment.extraction_omissions
                        if item.obligation_ref == obligation.obligation_ref
                    ),
                    None,
                ),
                "provenance": _jsonable(evaluation.provenance),
            }
            obligations_rows.append(row)
            cardinality_rows.append(
                {
                    "obligation_ref": obligation.obligation_ref,
                    "policy": obligation.cardinality_policy.value,
                    "state": status,
                    "satisfied": match.cardinality_satisfied,
                    "expected_key": obligation.expected_member_key
                    or obligation.expected_target_key
                    or obligation.expected_value_key,
                }
            )
            for candidate_ref in match.candidate_binding_refs:
                usage = binding_usage.get((claim_id, candidate_ref))
                if usage is not None:
                    usage["presented_to_obligations"] += 1
                    usage["candidate_obligation_refs"].append(obligation.obligation_ref)
            for binding_ref in match.matched_binding_refs:
                usage = binding_usage.get((claim_id, binding_ref))
                if usage is not None:
                    usage["matched_obligation_refs"].append(obligation.obligation_ref)
                    if obligation.allow_binding_reuse and obligation.binding_reuse_group:
                        usage["reuse_groups"].append(obligation.binding_reuse_group)
            for exclusion in match.excluded_candidates:
                usage = binding_usage.get((claim_id, exclusion.binding_ref))
                if usage is not None:
                    usage["excluded_reasons"].update([exclusion.reason.value])

        node_rows.append(
            {
                "claim_id": claim_id,
                "node_ref": record["source_node_ref"],
                "node_kind": record["source_node_kind"],
                "engine_node_kind": source_node.node_kind.value,
                "semantic_payload": record["source_semantic_payload"],
                "engine_input_payload": _jsonable(source_node.semantic_payload),
                "graph_role": record["graph_role"],
                "requirement_contract_version": derivation.contract_version,
                "input_adapter_status": record["adapter_status"],
                "input_adapter_notes": record["adapter_notes"],
                "derivation_status": derivation.status.value,
                "derived_obligation_count": len(derivation.obligations),
                "required_obligation_count": counts[
                    GroundingRequirementLevel.REQUIRED_FOR_NODE_SEMANTICS.value
                ],
                "optional_obligation_count": counts[
                    GroundingRequirementLevel.OPTIONAL_CORROBORATION.value
                ],
                "structural_only_obligation_count": counts[
                    GroundingRequirementLevel.STRUCTURAL_SUPPORT_ONLY.value
                ],
                "zero_obligation_reason": derivation.zero_obligation_reason.value
                if derivation.zero_obligation_reason
                else None,
                "candidate_bindings": candidate_refs,
                "matched_bindings": matched_bindings,
                "unmatched_obligations": unmatched_obligations,
                "cardinality_results": cardinality_rows,
                "hybrid_semantic_support_results": [
                    support_by_binding[(claim_id, binding_ref)]
                    for binding_ref in candidate_refs
                    if (claim_id, binding_ref) in support_by_binding
                ],
                "obligation_states": dict(node_obligation_states),
                "extraction_omission_diagnostics": [
                    item.model_dump(mode="json") for item in assessment.extraction_omissions
                ],
                "engine_node_usability": raw_engine_usability,
                "final_node_usability": final_usability,
                "verification_assertion_ready": assessment.verification_assertion_ready
                and record["adapter_status"] == "RESOLVED",
                "aggregation_precedence": list(assessment.aggregation_precedence),
                "diagnostics": list(assessment.diagnostics) + record["adapter_notes"],
                # An unresolved legacy field remains fully traceable because
                # its adapter status and diagnostic are persisted explicitly.
                "provenance_complete": record["source_node_assessment_present"],
            }
        )

    # Comparison-only inputs are loaded only after obligation derivation/evaluation.
    root_audit = _read_json(ROOT_AUDIT_PATH)
    omission_comparison = _compare_known_omissions(
        root_audit=root_audit,
        phase_c=phase_c,
        node_rows=node_rows,
        obligation_rows=obligations_rows,
        binding_records=binding_records,
    )
    binding_rows = _finalize_binding_rows(binding_records, binding_usage)
    cardinality = _cardinality_report(obligations_rows)
    requiredness = _requiredness_stability(node_records)
    usability = _usability_report(node_rows, obligations_rows)
    readiness = _readiness_report(node_rows, obligations_rows)
    coverage = _contract_coverage(node_rows, obligations_rows)
    provenance = _provenance_report(node_rows, obligations_rows, binding_rows)
    fail_closed = _fail_closed_report(node_rows, obligations_rows)
    safety = _safety_review(node_rows, obligations_rows)

    decision: str | None = None
    if compute_semantic_decision:
        unsafe_promotions = safety["counts"].get("UNSAFE_PROMOTION", 0)
        unsafe_masking = safety["counts"].get("UNSAFE_FAILURE_MASKING", 0)
        if unsafe_promotions or unsafe_masking:
            decision = "GROUNDING REQUIREMENT OFFLINE REPLAY REVEALS AGGREGATION SAFETY ISSUE"
        elif any(
            row["classification"] == "LEGACY_INPUT_INSUFFICIENT"
            for row in omission_comparison["rows"]
        ) and (
            coverage["unsupported_node_shape_count"] > 0
            or any(
                row["classification"]
                in {"MISSED_BY_REQUIREMENT_CONTRACT", "DETECTED_AS_TARGET_MISMATCH"}
                for row in omission_comparison["rows"]
            )
        ):
            decision = "GROUNDING REQUIREMENT OFFLINE REPLAY REVEALS MULTIPLE SYSTEMATIC ISSUES"
        elif coverage["unsupported_node_shape_count"] > 0:
            decision = "GROUNDING REQUIREMENT OFFLINE REPLAY REVEALS CONTRACT COVERAGE ISSUE"
        elif any(
            row["classification"] == "LEGACY_INPUT_INSUFFICIENT"
            for row in omission_comparison["rows"]
        ):
            decision = (
                "GROUNDING REQUIREMENT OFFLINE REPLAY REVEALS LEGACY-INPUT RECONSTRUCTION ISSUE"
            )
        elif any(
            row["classification"]
            in {"MISSED_BY_REQUIREMENT_CONTRACT", "DETECTED_AS_TARGET_MISMATCH"}
            for row in omission_comparison["rows"]
        ):
            decision = "GROUNDING REQUIREMENT OFFLINE REPLAY REVEALS CONTRACT COVERAGE ISSUE"
        elif provenance["provenance_defect_count"]:
            decision = "GROUNDING REQUIREMENT OFFLINE REPLAY INCONCLUSIVE"
        else:
            decision = (
                "GROUNDING REQUIREMENT OFFLINE REPLAY PASSES — READY FOR NEW REAL DEVELOPMENT "
                "HYBRID CANARY"
            )

    binding_ref_count = Counter((row["claim_id"], row["binding_ref"]) for row in binding_rows)
    if any(count != 1 for count in binding_ref_count.values()):
        raise RuntimeError("binding_ledger_identity_not_unique")
    if len({(row["claim_id"], row["node_ref"]) for row in node_rows}) != 72:
        raise RuntimeError("node_ledger_identity_not_unique")
    if len({row["obligation_ref"] for row in obligations_rows}) != len(obligations_rows):
        # Obligation refs are node-scoped; claim is part of node provenance.
        keys = [(row["claim_id"], row["obligation_ref"]) for row in obligations_rows]
        if len(set(keys)) != len(keys):
            raise RuntimeError("obligation_ledger_identity_not_unique_within_claim")

    artifacts: dict[str, Any] = {
        "node_ledger": {
            "artifact_version": "v2_1r_grounding_requirement_d3_node_ledger.v1",
            "rows": node_rows,
        },
        "obligation_ledger": {
            "artifact_version": "v2_1r_grounding_requirement_d3_obligation_ledger.v1",
            "rows": obligations_rows,
        },
        "binding_ledger": {
            "artifact_version": "v2_1r_grounding_requirement_d3_binding_ledger.v1",
            "rows": binding_rows,
        },
        "omission_comparison": omission_comparison,
        "cardinality": cardinality,
        "requiredness": requiredness,
        "usability": usability,
        "readiness": readiness,
        "coverage": coverage,
        "provenance": provenance,
    }
    report = _build_report(
        manifest=manifest,
        node_rows=node_rows,
        obligation_rows=obligations_rows,
        binding_rows=binding_rows,
        artifacts=artifacts,
        omission_comparison=omission_comparison,
        fail_closed=fail_closed,
        safety=safety,
        decision=decision,
        claim_count=len(claims),
        semantic_decision_computed=compute_semantic_decision,
        execution_mode=execution_mode,
    )
    artifacts["report_json"] = report
    artifacts["report_markdown"] = _markdown_report(report)
    contract_check = validate_report_artifacts(artifacts)
    if contract_check.status != "PASS":
        raise RuntimeError(f"analysis_report_contract_invalid:{contract_check.errors}")
    if publish:
        _publish_artifact_set(artifacts)
    return report


def _publish_artifact_set(artifacts: dict[str, Any]) -> None:
    """Serialize, parse back, validate, then publish the complete artifact set."""
    expected_keys = set(OUTPUTS)
    if set(artifacts) != expected_keys:
        raise RuntimeError("analysis_artifact_set_incomplete")
    existing = [str(path) for path in OUTPUTS.values() if (ROOT / path).exists()]
    if existing:
        raise RuntimeError(f"refusing_to_overwrite_analysis_artifacts:{existing}")

    json_payloads: dict[str, str] = {}
    for name, payload in artifacts.items():
        if name == "report_markdown":
            if not isinstance(payload, str):
                raise TypeError("report_markdown_must_be_string")
            continue
        normalized = jsonable(payload)
        if isinstance(normalized, dict):
            normalized.setdefault("attempt_id", ATTEMPT_ID)
            normalized.setdefault("analysis_adapter_version", ADAPTER_VERSION)
            normalized.setdefault("normalization_contract_version", NORMALIZATION_CONTRACT_VERSION)
        json_payloads[name] = (
            json.dumps(
                normalized,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
    round_trip = {name: json.loads(value) for name, value in json_payloads.items()}
    round_trip["report_markdown"] = artifacts["report_markdown"]
    checked = validate_report_artifacts(round_trip)
    if checked.status != "PASS":
        raise RuntimeError(f"analysis_artifact_round_trip_invalid:{checked.errors}")

    artifact_root = ROOT / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".d3-artifact-stage-", dir=artifact_root) as temp_name:
        stage_root = Path(temp_name)
        staged: dict[str, Path] = {}
        for name, relative_path in OUTPUTS.items():
            destination = stage_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            content = (
                json_payloads[name] if name != "report_markdown" else artifacts["report_markdown"]
            )
            destination.write_text(content, encoding="utf-8")
            staged[name] = destination
        for name, relative_path in OUTPUTS.items():
            destination = ROOT / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged[name], destination)


def _compare_known_omissions(
    *,
    root_audit: dict[str, Any],
    phase_c: dict[str, Any],
    node_rows: list[dict[str, Any]],
    obligation_rows: list[dict[str, Any]],
    binding_records: list[dict[str, Any]],
) -> dict[str, Any]:
    nodes = {(row["claim_id"], row["node_ref"]): row for row in node_rows}
    normalized_views = [
        normalized_view_from_dict(row["normalized_role_view"]) for row in binding_records
    ]
    obligations_by_node: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in obligation_rows:
        obligations_by_node[(row["claim_id"], row["node_ref"])].append(row)
    rows: list[dict[str, Any]] = []
    for omission in root_audit.get("absent_required_binding_audit", []):
        claim_id = str(omission.get("claim_id"))
        target_ref = str(omission.get("target_ref"))
        role = str(omission.get("required_role"))
        node = nodes.get((claim_id, target_ref))
        role_resolution = resolve_omission_role(
            claim_id=claim_id,
            node_ref=target_ref,
            required_role=role,
            normalized_bindings=normalized_views,
        )
        matched = [
            item
            for item in obligations_by_node.get((claim_id, target_ref), [])
            if role in item["semantic_role_family"]
        ]
        state = next((item["final_obligation_state"] for item in matched), None)
        if state == GroundingObligationStatus.MISSING_REQUIRED_BINDING.value:
            classification = "DETECTED_AS_MISSING_REQUIRED_BINDING"
        elif state == GroundingObligationStatus.AMBIGUOUS.value:
            classification = "DETECTED_BY_CARDINALITY_FAILURE"
        elif node and node["derivation_status"] != GroundingDerivationStatus.RESOLVED.value:
            classification = "DETECTED_AS_DERIVATION_UNRESOLVED"
        elif role_resolution.status == "LEGACY_FIELD_UNRESOLVED":
            classification = "LEGACY_INPUT_INSUFFICIENT"
        elif not matched:
            classification = "NOT_REPRESENTED_BY_CURRENT_NODE_SEMANTICS"
        else:
            classification = "MISSED_BY_REQUIREMENT_CONTRACT"
        rows.append(
            {
                "claim_id": claim_id,
                "node_ref": target_ref,
                "known_omission_kind": role,
                "historical_audit_root_cause": omission.get("root_cause"),
                "human_judgment_used_for_derivation": False,
                "matching_obligation_refs": [item["obligation_ref"] for item in matched],
                "canonical_role_resolution": role_resolution.status,
                "canonical_role_binding_refs": list(role_resolution.matching_binding_refs),
                "unresolved_role_binding_refs": list(role_resolution.unresolved_binding_refs),
                "observed_obligation_state": state,
                "classification": classification,
            }
        )

    # Explicitly retain the historical schema limitation discovered during
    # reconstruction; it is comparison metadata, never a derivation input.
    for node in node_rows:
        if node["input_adapter_status"] == "LEGACY_INPUT_INSUFFICIENT":
            rows.append(
                {
                    "claim_id": node["claim_id"],
                    "node_ref": node["node_ref"],
                    "known_omission_kind": "LEGACY_ASSERTION_CONTENT_FIELD",
                    "known_omission_count": 1,
                    "human_judgment_used_for_derivation": False,
                    "observed_obligation_state": "DERIVATION_UNRESOLVED",
                    "classification": "LEGACY_INPUT_INSUFFICIENT",
                }
            )

    known_findings = phase_c.get("extraction_omission_ledger", [])
    for finding in known_findings:
        text = str(finding.get("finding", ""))
        claim_id = str(finding.get("claim_id", ""))
        if "METHOD_RELATION" in text:
            classification = (
                "DETECTED_AS_MISSING_REQUIRED_BINDING"
                if any(
                    row["claim_id"] == claim_id
                    and row["obligation_type"] == "METHOD_RELATION"
                    and row["final_obligation_state"]
                    == GroundingObligationStatus.MISSING_REQUIRED_BINDING.value
                    for row in obligation_rows
                )
                else "NOT_REPRESENTED_BY_CURRENT_NODE_SEMANTICS"
            )
            omitted_count = 5
            kind = "METHOD_RELATION"
        elif "12 category" in text or ("12" in text and "association" in text):
            classification = "NOT_REPRESENTED_BY_CURRENT_NODE_SEMANTICS"
            omitted_count = 12
            kind = "NUMERIC_MATRIX_CELL_AND_ASSOCIATION"
        elif "five category members" in text:
            classification = "NOT_REPRESENTED_BY_CURRENT_NODE_SEMANTICS"
            omitted_count = 5
            kind = "ENUMERATION_MEMBER"
        elif "EPISTEMIC" in text or "reporting agent" in text:
            matching_role_views = [
                row
                for row in binding_records
                if row["claim_id"] == claim_id
                and row["semantic_role"] in {"EPISTEMIC_OPERATOR", "EPISTEMIC_AGENT"}
            ]
            unresolved_role_views = [
                row
                for row in binding_records
                if row["claim_id"] == claim_id and row["normalization_status"] != "RESOLVED"
            ]
            if any(row["target_kind"] != "CONTEXT" for row in matching_role_views):
                classification = "DETECTED_AS_TARGET_MISMATCH"
            elif unresolved_role_views:
                classification = "LEGACY_INPUT_INSUFFICIENT"
            else:
                classification = "NOT_REPRESENTED_BY_CURRENT_NODE_SEMANTICS"
            omitted_count = 2
            kind = "EPISTEMIC_TARGET_AND_BINDING"
        else:
            classification = "OTHER"
            omitted_count = 1
            kind = "OTHER_KNOWN_EXTRACTION_OMISSION"
        rows.append(
            {
                "claim_id": claim_id,
                "node_ref": None,
                "known_omission_kind": kind,
                "known_omission_count": omitted_count,
                "human_judgment_used_for_derivation": False,
                "observed_obligation_state": None,
                "classification": classification,
            }
        )

    for finding in root_audit.get("additional_graph_findings", []):
        claim_id = str(finding.get("claim_id", ""))
        if finding.get("finding") == "attachment_evidence_missing":
            node_ref = str(finding.get("attachment_ref"))
            matches = obligations_by_node.get((claim_id, node_ref), [])
            if any(
                item["final_obligation_state"]
                == GroundingObligationStatus.MISSING_REQUIRED_BINDING.value
                for item in matches
            ):
                classification = "DETECTED_AS_MISSING_REQUIRED_BINDING"
            elif (
                nodes.get((claim_id, node_ref), {}).get("derivation_status")
                != GroundingDerivationStatus.RESOLVED.value
            ):
                classification = "DETECTED_AS_DERIVATION_UNRESOLVED"
            else:
                classification = "MISSED_BY_REQUIREMENT_CONTRACT"
            rows.append(
                {
                    "claim_id": claim_id,
                    "node_ref": node_ref,
                    "known_omission_kind": "DIRECT_ATTACHMENT_BINDING",
                    "known_omission_count": 1,
                    "human_judgment_used_for_derivation": False,
                    "matching_obligation_refs": [item["obligation_ref"] for item in matches],
                    "classification": classification,
                }
            )
        elif finding.get("finding") == "single-assertion scope node has no SCOPE_STRUCTURE binding":
            rows.append(
                {
                    "claim_id": claim_id,
                    "node_ref": str(finding.get("scope_ref")),
                    "known_omission_kind": "SINGLE_ASSERTION_SCOPE_STRUCTURE",
                    "known_omission_count": 1,
                    "human_judgment_used_for_derivation": False,
                    "classification": "AUDIT_OMISSION_NOT_ACTUALLY_REQUIRED",
                }
            )
    # Historical human judgments are comparison-only. They cannot influence
    # derivation, matching, or node aggregation.
    binding_to_matches: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for obligation in obligation_rows:
        for binding_ref in obligation["matched_bindings"]:
            binding_to_matches[(obligation["claim_id"], binding_ref)].append(obligation)
    judgment_counts: Counter[str] = Counter()
    discrepancies: list[dict[str, Any]] = []
    audited_bindings = root_audit.get("binding_ledger", [])
    for audit_row in audited_bindings:
        judgment = str(audit_row.get("human_support_judgment", "UNKNOWN"))
        judgment_counts[judgment] += 1
        key = (str(audit_row.get("claim_id", "")), str(audit_row.get("binding_ref", "")))
        matched_rows = binding_to_matches.get(key, [])
        node = nodes.get((key[0], str(audit_row.get("target_ref", ""))))
        human_non_support = judgment not in {
            "SUPPORTED",
            "SUPPORTS",
            "GROUNDED",
            "VALID",
            "UNKNOWN",
        }
        if (
            human_non_support
            and node
            and node["final_node_usability"] == GroundingNodeUsability.GROUNDED_USABLE.value
            and matched_rows
            and all(
                row["final_obligation_state"] == GroundingObligationStatus.SATISFIED.value
                for row in matched_rows
            )
        ):
            discrepancies.append(
                {
                    "claim_id": key[0],
                    "binding_ref": key[1],
                    "human_support_judgment": judgment,
                    "node_ref": node["node_ref"],
                    "matched_obligation_refs": [row["obligation_ref"] for row in matched_rows],
                    "classification": "D3_SATISFIED_WHERE_HUMAN_REVIEW_NON_SUPPORTS",
                }
            )

    return {
        "artifact_version": "v2_1r_grounding_requirement_d3_known_omission_comparison.v1",
        "comparison_only": True,
        "omission_ledger_read_after_derivation": True,
        "obligation_derivation_consulted_omission_ledger": False,
        "rows": rows,
        "classification_counts": dict(Counter(row["classification"] for row in rows)),
        "human_review_comparison": {
            "comparison_only": True,
            "human_judgment_used_for_derivation": False,
            "audited_binding_count": len(audited_bindings),
            "judgment_distribution": dict(judgment_counts),
            "potential_discrepancy_count": len(discrepancies),
            "potential_discrepancies": discrepancies,
        },
    }


def _finalize_binding_rows(
    binding_records: list[dict[str, Any]],
    usage: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for binding in binding_records:
        key = (binding["claim_id"], binding["binding_ref"])
        use = usage[key]
        matched = sorted(set(use["matched_obligation_refs"]))
        groups = sorted(set(use["reuse_groups"]))
        result.append(
            {
                **binding,
                "presented_to_obligation_match_count": use["presented_to_obligations"],
                "candidate_for_obligation_count": len(set(use["candidate_obligation_refs"])),
                "matched_obligation_refs": matched,
                "matches_multiple_obligations": len(matched) > 1,
                "reuse_permitted": bool(groups),
                "reuse_groups": groups,
                "undeclared_reuse": len(matched) > 1 and not groups,
                "unmatched": not matched,
                "unmatched_or_excluded_reasons": dict(use["excluded_reasons"]),
                "extraneous_optional_or_structural_classification": (
                    "MATCHED_STRUCTURAL_OR_SEMANTIC" if matched else "NO_MATCH_IN_FROZEN_CONTRACT"
                ),
            }
        )
    return result


def _cardinality_report(obligations: list[dict[str, Any]]) -> dict[str, Any]:
    return cardinality_report(obligations).to_dict()


def _requiredness_stability(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in records:
        node = record["engine_input_node"]
        original = derive_grounding_obligations(node)
        decorated_payload = dict(node.semantic_payload)
        decorated_payload.update(
            {
                "human_support_judgment": "COUNTERFACTUAL_ONLY",
                "phase_c_oracle_result": "COUNTERFACTUAL_ONLY",
                "verifier_verdict": "COUNTERFACTUAL_ONLY",
                "binding_support_outcome": "COUNTERFACTUAL_ONLY",
                "omission_annotation": "COUNTERFACTUAL_ONLY",
            }
        )
        decorated = node.model_copy(update={"semantic_payload": decorated_payload})
        replay = derive_grounding_obligations(decorated)
        original_refs = [item.obligation_ref for item in original.obligations]
        replay_refs = [item.obligation_ref for item in replay.obligations]
        rows.append(
            {
                "node_ref": node.node_ref,
                "node_kind": node.node_kind.value,
                "same_obligations": original_refs == replay_refs,
                "same_derivation_status": original.status is replay.status,
                "obligation_refs_sha256": _canonical_hash(original_refs),
                "omission_annotation_removed_vs_present_stable": original_refs == replay_refs,
            }
        )
    return {
        "artifact_version": "v2_1r_grounding_requirement_d3_requiredness_stability.v1",
        "nodes_tested": len(rows),
        "all_stable": all(
            row["same_obligations"] and row["same_derivation_status"] for row in rows
        ),
        "obligation_set_fingerprint": _canonical_hash(
            {row["node_ref"]: row["obligation_refs_sha256"] for row in rows}
        ),
        "requiredness_stability_violation_count": sum(
            not row["same_obligations"] or not row["same_derivation_status"] for row in rows
        ),
        "omission_ledger_used_for_derivation": False,
        "counterfactual_fields": [
            "human_support_judgment",
            "verifier_verdict",
            "binding_support_outcome",
            "omission_annotation",
        ],
        "rows": rows,
    }


def _usability_report(
    nodes: list[dict[str, Any]], obligations: list[dict[str, Any]]
) -> dict[str, Any]:
    counts = Counter(row["final_node_usability"] for row in nodes)
    missing_nodes = sorted(
        {
            row["node_ref"]
            for row in obligations
            if row["final_obligation_state"] == "MISSING_REQUIRED_BINDING"
        }
    )
    return {
        "artifact_version": "v2_1r_grounding_requirement_d3_node_usability.v1",
        "node_count": len(nodes),
        "distribution": dict(counts),
        "missing_required_binding_count": sum(
            row["final_obligation_state"] == "MISSING_REQUIRED_BINDING" for row in obligations
        ),
        "affected_node_count": len(missing_nodes),
        "affected_node_refs": missing_nodes,
        "unsafe_vacuous_promotion_count": sum(
            row["final_node_usability"] == "GROUNDED_USABLE"
            and row["derived_obligation_count"] == 0
            for row in nodes
        ),
        "verification_assertion_ready_count": sum(
            row["verification_assertion_ready"] for row in nodes
        ),
    }


def _readiness_report(
    nodes: list[dict[str, Any]], obligations: list[dict[str, Any]]
) -> dict[str, Any]:
    by_node: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in obligations:
        by_node[(row["claim_id"], row["node_ref"])].append(row)
    blockers: Counter[str] = Counter()
    for node in nodes:
        if node["verification_assertion_ready"]:
            continue
        states = {
            item["final_obligation_state"] for item in by_node[(node["claim_id"], node["node_ref"])]
        }
        if node["input_adapter_status"] != "RESOLVED":
            blockers["legacy_input_adapter_unresolved"] += 1
        elif node["derivation_status"] != "RESOLVED":
            blockers["derivation_failure"] += 1
        elif "MISSING_REQUIRED_BINDING" in states:
            blockers["missing_required_binding"] += 1
        elif "NOT_SUPPORTED" in states:
            blockers["unsupported_semantic_evidence"] += 1
        elif "AMBIGUOUS" in states:
            blockers["ambiguity"] += 1
        elif states.intersection(
            {"INSUFFICIENT_EVIDENCE", "TEXTUALLY_INVALID", "STRUCTURALLY_INVALID"}
        ):
            blockers["insufficient_textual_structural_or_semantic_evidence"] += 1
        else:
            blockers["not_ready_under_shadow_contract"] += 1
    return {
        "artifact_version": "v2_1r_grounding_requirement_d3_verification_readiness.v1",
        "ready_nodes": sum(row["verification_assertion_ready"] for row in nodes),
        "blocked_nodes_by_reason": dict(blockers),
        "verification_assertion_view_constructed": False,
    }


def _contract_coverage(
    nodes: list[dict[str, Any]], obligations: list[dict[str, Any]]
) -> dict[str, Any]:
    shapes: dict[str, dict[str, Any]] = {}
    for row in nodes:
        key = f"{row['node_kind']}->{row['engine_node_kind']}"
        item = shapes.setdefault(
            key,
            {
                "nodes": 0,
                "supported": 0,
                "partially_supported": 0,
                "unsupported": 0,
                "adapter_unresolved": 0,
            },
        )
        item["nodes"] += 1
        if row["input_adapter_status"] != "RESOLVED":
            item["adapter_unresolved"] += 1
            item["partially_supported"] += 1
        elif row["derivation_status"] == "UNSUPPORTED_NODE_SHAPE":
            item["unsupported"] += 1
        elif row["derivation_status"] == "DERIVATION_UNRESOLVED":
            item["partially_supported"] += 1
        else:
            item["supported"] += 1
    context_types = Counter(
        str(row["semantic_payload"].get("context_type", "UNKNOWN"))
        for row in nodes
        if row["node_kind"] == "CONTEXT"
    )
    return {
        "artifact_version": "v2_1r_grounding_requirement_d3_contract_coverage.v1",
        "source_node_shape_counts": dict(Counter(row["node_kind"] for row in nodes)),
        "engine_node_shape_coverage": shapes,
        "context_type_counts": dict(context_types),
        "unsupported_node_shape_count": sum(item["unsupported"] for item in shapes.values()),
        "partially_supported_node_count": sum(
            item["partially_supported"] for item in shapes.values()
        ),
        "supported_node_count": sum(item["supported"] for item in shapes.values()),
        "obligation_type_counts": dict(Counter(row["obligation_type"] for row in obligations)),
    }


def _provenance_report(
    nodes: list[dict[str, Any]], obligations: list[dict[str, Any]], bindings: list[dict[str, Any]]
) -> dict[str, Any]:
    node_complete = sum(bool(row["provenance_complete"]) for row in nodes)
    obligation_complete = sum(
        bool(row.get("provenance", {}).get("node_ref"))
        and bool(row.get("provenance", {}).get("contract_version"))
        and bool(row.get("provenance", {}).get("derivation_rule_id"))
        and bool(row.get("provenance", {}).get("obligation_ref"))
        and "obligation_status" in row.get("provenance", {})
        for row in obligations
    )
    defects = []
    for row in nodes:
        if not row["provenance_complete"]:
            defects.append(
                {
                    "claim_id": row["claim_id"],
                    "node_ref": row["node_ref"],
                    "defect": "source_node_assessment_missing",
                }
            )
    for row in bindings:
        if not row["support_result_present"]:
            defects.append(
                {
                    "claim_id": row["claim_id"],
                    "binding_ref": row["binding_ref"],
                    "defect": "frozen_hybrid_support_missing",
                }
            )
        if row["undeclared_reuse"]:
            defects.append(
                {
                    "claim_id": row["claim_id"],
                    "binding_ref": row["binding_ref"],
                    "defect": "undeclared_cross_obligation_reuse",
                }
            )
    return {
        "artifact_version": "v2_1r_grounding_requirement_d3_provenance.v1",
        "trace_fields": [
            "source_semantic_node",
            "contract_version",
            "derivation_rule",
            "obligation",
            "candidate_bindings",
            "binding_matching",
            "textual_grounding",
            "structural_grounding",
            "hybrid_semantic_support",
            "obligation_state",
            "aggregation",
            "node_usability",
            "verification_assertion_readiness",
        ],
        "nodes_total": len(nodes),
        "nodes_with_complete_trace": node_complete,
        "obligations_total": len(obligations),
        "obligations_with_complete_trace": obligation_complete,
        "provenance_defect_count": len(defects),
        "provenance_defects": defects,
        "zero_hidden_state_transitions_target_met": len(defects) == 0,
    }


def _fail_closed_report(
    nodes: list[dict[str, Any]], obligations: list[dict[str, Any]]
) -> dict[str, Any]:
    hard_cases: dict[str, bool | str] = {
        "missing_required_binding": any(
            row["final_obligation_state"] == "MISSING_REQUIRED_BINDING" and row["claim_id"]
            for row in obligations
        ),
        "historical_artifact_missing_assertion_content": all(
            row["engine_node_kind"] != "ASSERTION"
            or row["derivation_status"] == "DERIVATION_UNRESOLVED"
            for row in nodes
        ),
        "unknown_node_shape_fixture": "covered_by_requirement_engine_regression_tests",
        "ambiguous_candidate_fixture": "covered_by_requirement_engine_regression_tests",
        "multiple_candidate_fixture": "covered_by_requirement_engine_regression_tests",
        "structural_invalid_fixture": "covered_by_requirement_engine_regression_tests",
        "textual_invalid_fixture": "covered_by_requirement_engine_regression_tests",
        "derivation_unresolved_fixture": "covered_by_requirement_engine_regression_tests",
    }
    unsafe = [
        row
        for row in nodes
        if row["derived_obligation_count"] == 0
        and row["final_node_usability"] == GroundingNodeUsability.GROUNDED_USABLE.value
        and row["zero_obligation_reason"]
        not in {ZeroObligationReason.NO_SEMANTIC_GROUNDING_REQUIRED.value}
    ]
    return {
        "cases": hard_cases,
        "unsafe_promotion_count": sum(
            row["final_node_usability"] == GroundingNodeUsability.GROUNDED_USABLE.value
            and any(
                item["claim_id"] == row["claim_id"]
                and item["node_ref"] == row["node_ref"]
                and item["final_obligation_state"]
                in {
                    "MISSING_REQUIRED_BINDING",
                    "NOT_SUPPORTED",
                    "AMBIGUOUS",
                    "INSUFFICIENT_EVIDENCE",
                    "TEXTUALLY_INVALID",
                    "STRUCTURALLY_INVALID",
                    "DERIVATION_UNRESOLVED",
                }
                for item in obligations
            )
            for row in nodes
        ),
        "unsafe_failure_masking_count": sum(
            row["final_node_usability"] == GroundingNodeUsability.GROUNDED_USABLE.value
            and row["derivation_status"] != GroundingDerivationStatus.RESOLVED.value
            for row in nodes
        ),
        "unsafe_vacuous_promotion_count": len(unsafe),
        "unsafe_zero_obligation_promotions": len(unsafe),
        "unsafe_node_refs": [
            {"claim_id": row["claim_id"], "node_ref": row["node_ref"]} for row in unsafe
        ],
        "missing_required_binding_promoted_to_usable": any(
            node["final_node_usability"] == "GROUNDED_USABLE"
            and row["final_obligation_state"] == "MISSING_REQUIRED_BINDING"
            for node in nodes
            for row in obligations
            if row["claim_id"] == node["claim_id"] and row["node_ref"] == node["node_ref"]
        ),
    }


def _safety_review(
    nodes: list[dict[str, Any]], obligations: list[dict[str, Any]]
) -> dict[str, Any]:
    obligation_states: dict[tuple[str, str], set[str]] = defaultdict(set)
    for item in obligations:
        obligation_states[(item["claim_id"], item["node_ref"])].add(item["final_obligation_state"])
    rows: list[dict[str, Any]] = []
    for node in nodes:
        states = obligation_states[(node["claim_id"], node["node_ref"])]
        if node["input_adapter_status"] != "RESOLVED":
            classification = "INPUT_RECONSTRUCTION_UNRESOLVED"
        elif node["final_node_usability"] == "GROUNDED_USABLE" and states.intersection(
            {"MISSING_REQUIRED_BINDING", "NOT_SUPPORTED", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"}
        ):
            classification = "UNSAFE_PROMOTION"
        elif node["final_node_usability"] == "GROUNDED_USABLE":
            classification = "CORRECTLY_GROUNDED"
        elif node["final_node_usability"] in {
            "STRUCTURALLY_GROUNDED",
            "NO_SEMANTIC_GROUNDING_REQUIRED",
            "PARTIALLY_GROUNDED_UNRESOLVED",
            "UNGROUNDED_UNUSABLE",
            "SEMANTIC_SUPPORT_AMBIGUOUS",
            "NOT_EVALUATED",
        }:
            classification = "CORRECTLY_FAIL_CLOSED"
        else:
            classification = "OTHER"
        rows.append(
            {
                "claim_id": node["claim_id"],
                "node_ref": node["node_ref"],
                "node_kind": node["node_kind"],
                "classification": classification,
            }
        )
    return {"rows": rows, "counts": dict(Counter(row["classification"] for row in rows))}


def _build_report(
    *,
    manifest: dict[str, Any],
    node_rows: list[dict[str, Any]],
    obligation_rows: list[dict[str, Any]],
    binding_rows: list[dict[str, Any]],
    artifacts: dict[str, Any],
    omission_comparison: dict[str, Any],
    fail_closed: dict[str, Any],
    safety: dict[str, Any],
    decision: str | None,
    claim_count: int,
    semantic_decision_computed: bool,
    execution_mode: str,
) -> dict[str, Any]:
    derivation_counts = Counter(row["derivation_status"] for row in node_rows)
    obligation_counts = Counter(row["final_obligation_state"] for row in obligation_rows)
    return {
        "artifact_version": "v2_1r_grounding_requirement_d3_attempt_2_offline_replay.v1",
        "attempt_id": ATTEMPT_ID,
        "execution_mode": execution_mode,
        "semantic_decision_computed": semantic_decision_computed,
        "analysis_adapter_version": ADAPTER_VERSION,
        "normalization_contract_version": NORMALIZATION_CONTRACT_VERSION,
        "artifact_contract_inventory": ARTIFACT_CONTRACT_INVENTORY,
        "title": "V2.1R Grounding Requirement Offline Development Replay",
        "first_pass": True,
        "frozen_hash_manifest": str(HASH_MANIFEST).replace("\\", "/"),
        "normalized_implementation_fingerprint": manifest["normalized_implementation_fingerprint"],
        "claim_count": claim_count,
        "node_count": len(node_rows),
        "obligation_count": len(obligation_rows),
        "binding_count": len(binding_rows),
        "derivation_distribution": dict(derivation_counts),
        "obligation_state_distribution": dict(obligation_counts),
        "node_usability_distribution": artifacts["usability"]["distribution"],
        "contract_coverage": artifacts["coverage"],
        "missing_required_binding_count": artifacts["usability"]["missing_required_binding_count"],
        "verification_assertion_ready_count": artifacts["readiness"]["ready_nodes"],
        "known_omission_classification_counts": omission_comparison["classification_counts"],
        "requiredness_stability": artifacts["requiredness"]["all_stable"],
        "provenance_complete_nodes": artifacts["provenance"]["nodes_with_complete_trace"],
        "provenance_complete_obligations": artifacts["provenance"][
            "obligations_with_complete_trace"
        ],
        "provenance_defect_count": artifacts["provenance"]["provenance_defect_count"],
        "fail_closed_audit": fail_closed,
        "node_safety_review_counts": safety["counts"],
        "unsafe_promotion_count": fail_closed["unsafe_promotion_count"],
        "unsafe_failure_masking_count": fail_closed["unsafe_failure_masking_count"],
        "unsafe_vacuous_promotion_count": fail_closed["unsafe_vacuous_promotion_count"],
        "undeclared_binding_reuse_count": sum(row["undeclared_reuse"] for row in binding_rows),
        "requiredness_stability_violation_count": artifacts["requiredness"][
            "requiredness_stability_violation_count"
        ],
        "provider_calls": 0,
        "external_http": 0,
        "semantic_verifier_calls": 0,
        "extractor_calls": 0,
        "holdout_accessed": False,
        "production_behavior_changed": False,
        "decision": decision,
        "output_artifacts": {key: str(path).replace("\\", "/") for key, path in OUTPUTS.items()},
    }


def _markdown_report(report: dict[str, Any]) -> str:
    artifacts = report["output_artifacts"]
    lines = [
        "# V2.1R Grounding Requirement Offline Development Replay",
        "",
        "## A. Frozen Inputs / Implementation",
        f"- Claims/nodes/bindings: {report['claim_count']}/{report['node_count']}/"
        f"{report['binding_count']}",
        f"- Fingerprint: `{report['normalized_implementation_fingerprint']}`",
        f"- Hash manifest: `{report['frozen_hash_manifest']}`",
        "- Phase C development support retained its provenance labels; no verifier ran.",
        "",
        "## B. Replay Reconstruction",
        "- First formal offline pass; deterministic, versioned, analysis-only adapter.",
        "- No missing assertion content was reconstructed from Claim text or notes.",
        "- Omission annotations/human judgments were comparison-only.",
        "",
        "## C. Node Contract Coverage",
        f"- `{json.dumps(report['contract_coverage'], ensure_ascii=False)}`",
        f"- Artifact: `{artifacts['coverage']}`",
        "",
        "## D. Obligation Derivation",
        f"- `{json.dumps(report['derivation_distribution'], ensure_ascii=False)}`",
        f"- Ledger: `{artifacts['obligation_ledger']}`",
        "",
        "## E. Obligation Distribution",
        f"- Total {report['obligation_count']}: "
        f"`{json.dumps(report['obligation_state_distribution'], ensure_ascii=False)}`",
        "",
        "## F. Binding Matching",
        f"- All frozen bindings: `{artifacts['binding_ledger']}`",
        "- Candidate, matched, excluded, and reuse counts are distinct fields.",
        "",
        "## G. Cardinality",
        f"- Per-policy results: `{artifacts['cardinality']}`",
        "",
        "## H. Missing Required Bindings",
        f"- Missing-required state count: {report['missing_required_binding_count']}.",
        "- Promotion safety is separately checked by the fail-closed audit.",
        "",
        "## I. Known Omission Comparison",
        f"- Comparison artifact: `{artifacts['omission_comparison']}`",
        "- Loaded only after derivation/evaluation; it did not alter obligations.",
        "",
        "## J. Method Findings",
        "- METHOD_VALUE, METHOD_RELATION and target applicability appear separately.",
        "",
        "## K. Epistemic Findings",
        "- Operator, agent, target and modality are not inferred from unrelated bindings.",
        "",
        "## L. Attachment Findings",
        "- Structural reference validity is separate from semantic applicability.",
        "",
        "## M. Scope / Shared Context Findings",
        "- Per-target obligations and sibling isolation are recorded in the ledgers.",
        "",
        "## N. Numeric / Pairing Findings",
        "- Value, metric, scope and association requirements remain distinct.",
        "",
        "## O. Enumeration Findings",
        "- Members absent from frozen node semantics were not inferred from Claim text.",
        "",
        "## P. Optional Corroboration",
        "- Optional failures remain distinct from required semantic obligations.",
        "",
        "## Q. Structural-Only / Zero-Obligation Findings",
        "- Zero-obligation reasons and structural-only states are explicit per node.",
        "",
        "## R. Obligation State Distribution",
        f"- `{json.dumps(report['obligation_state_distribution'], ensure_ascii=False)}`",
        "",
        "## S. Node Usability",
        f"- `{json.dumps(report['node_usability_distribution'], ensure_ascii=False)}`",
        f"- Summary: `{artifacts['usability']}`",
        "",
        "## T. VerificationAssertion Readiness",
        f"- Ready nodes: {report['verification_assertion_ready_count']}; no view constructed.",
        f"- Readiness: `{artifacts['readiness']}`",
        "",
        "## U. Node-Level Safety Review",
        f"- `{json.dumps(report['node_safety_review_counts'], ensure_ascii=False)}`",
        "",
        "## V. Requiredness Stability",
        f"- Stable: {report['requiredness_stability']}; `{artifacts['requiredness']}`",
        "",
        "## W. Binding Reuse",
        "- Permission, reuse group, matching and undeclared reuse are per binding.",
        "",
        "## X. Provenance",
        f"- Complete nodes/obligations: {report['provenance_complete_nodes']}/"
        f"{report['provenance_complete_obligations']}; defects: "
        f"{report['provenance_defect_count']}.",
        f"- Audit: `{artifacts['provenance']}`",
        "",
        "## Y. Fail-Closed Audit",
        f"- `{json.dumps(report['fail_closed_audit'], ensure_ascii=False)}`",
        "",
        "## Z. Architecture Ownership of Remaining Failures",
        "- Ownership is separated in the omission comparison and node diagnostics.",
        "- No Hybrid/router or production implementation revision was made.",
        "",
        "## AA. Production Behavior",
        "- ResearchLoop, persistence, acceptance, coverage, extraction: NONE changed.",
        "",
        "## AB. Anti-Overfitting Audit",
        "- Frozen development Claims only; no Holdout access or sample reselection.",
        "",
        "## AC. Provider Calls",
        "- Provider/external HTTP/verifier/extractor calls: 0/0/0/0.",
        "",
        "## AD. Quality Gates",
        "- Artifact validation is part of replay; remaining gates are recorded at handoff.",
        "",
        "## AE. Artifacts",
        *[f"- `{path}`" for path in artifacts.values()],
        "",
        "## AF. Final Decision",
        (
            f"`{report['decision']}`"
            if report.get("semantic_decision_computed")
            else "`HARNESS_ONLY — NO FORMAL D3 SEMANTIC DECISION`"
        ),
        "",
    ]
    return "\n".join(lines)


def reconstruction_dry_run(*, check_output_collisions: bool = True) -> dict[str, Any]:
    """Validate addressability and artifact contracts without deriving obligations."""
    canary = _read_json(CANARY_PATH)
    phase_c = _read_json(PHASE_C_PATH)
    root_audit = _read_json(ROOT_AUDIT_PATH)
    selection = _read_json(SELECTION_PATH)
    claims = canary.get("claims", [])
    selected_ids = set(selection.get("selection", {}).get("selected_claim_ids", []))
    claim_ids = {str(claim["claim_id"]) for claim in claims}
    if selected_ids != claim_ids:
        raise RuntimeError("dry_run_selection_identity_mismatch")

    nodes = [
        node
        for claim in claims
        for node in _node_records(str(claim["claim_id"]), claim.get("grounding_result") or {})
    ]
    source_binding_keys = {
        (str(claim["claim_id"]), str(binding.get("binding_ref", "")))
        for claim in claims
        for binding in (claim.get("grounding_result") or {}).get("evidence_bindings", [])
    }
    phase_index = _support_index(phase_c)
    phase_keys = set(phase_index)
    normalized = []
    duplicate_binding_refs = []
    for claim in claims:
        grounding = claim.get("grounding_result") or {}
        seen: set[str] = set()
        claim_id = str(claim["claim_id"])
        phase_rows = [
            row for (row_claim_id, _), row in phase_index.items() if row_claim_id == claim_id
        ]
        for binding in grounding.get("evidence_bindings", []):
            binding_ref = str(binding.get("binding_ref", ""))
            if binding_ref in seen:
                duplicate_binding_refs.append({"claim_id": claim_id, "binding_ref": binding_ref})
            seen.add(binding_ref)
            normalized.append(
                normalize_replay_binding(
                    claim_id=claim_id,
                    binding_ref=binding_ref,
                    canonical_bindings=grounding.get("evidence_bindings", []),
                    phase_c_rows=phase_rows,
                    source_artifact=str(CANARY_PATH).replace("\\", "/"),
                    schema_version=str(canary.get("schema_version", "")),
                )
            )

    known_omission_shape = {
        "absent_required_binding_audit": {"claim_id", "target_ref", "required_role"},
        "additional_graph_findings": {"claim_id", "finding"},
        "phase_c_extraction_omission_ledger": {"claim_id", "finding"},
    }
    sources = {
        "absent_required_binding_audit": root_audit.get("absent_required_binding_audit", []),
        "additional_graph_findings": root_audit.get("additional_graph_findings", []),
        "phase_c_extraction_omission_ledger": phase_c.get("extraction_omission_ledger", []),
    }
    schema_errors = [
        name
        for name, required_fields in known_omission_shape.items()
        if any(not required_fields.issubset(set(row)) for row in sources[name])
    ]
    if canary.get("schema_version") != EXPECTED_SOURCE_SCHEMA:
        schema_errors.append("canary_source_schema_version")
    output_collisions = (
        [str(path) for path in [HASH_MANIFEST, *OUTPUTS.values()] if (ROOT / path).exists()]
        if check_output_collisions
        else []
    )
    duplicate_node_refs = len(nodes) - len(
        {(row["claim_id"], row["source_node_ref"]) for row in nodes}
    )
    assertion_nodes = [row for row in nodes if row["source_node_kind"] == "ASSERTION"]
    unresolved_legacy_fields = [
        {
            "claim_id": row["claim_id"],
            "node_ref": row["source_node_ref"],
            "field": "assertion_content",
        }
        for row in assertion_nodes
        if "assertion_content" not in row["source_semantic_payload"]
    ]
    unresolved_legacy_fields.extend(
        {
            "claim_id": row.claim_id,
            "binding_ref": row.binding_ref,
            "status": row.normalization_status,
        }
        for row in normalized
        if row.normalization_status != "RESOLVED"
    )
    orphan_source_refs = sorted(source_binding_keys - phase_keys)
    orphan_phase_refs = sorted(phase_keys - source_binding_keys)
    result = {
        "status": "PASS"
        if not (
            len(claims) != 8
            or len(nodes) != 72
            or len(source_binding_keys) != 75
            or len(normalized) != 75
            or any(row.normalization_status != "RESOLVED" for row in normalized)
            or any(row.support_resolution_status != "RESOLVED" for row in normalized)
            or duplicate_binding_refs
            or duplicate_node_refs
            or orphan_source_refs
            or orphan_phase_refs
            or schema_errors
            or output_collisions
        )
        else "FAIL",
        "decision_computed": False,
        "node_resolution_count": len(nodes),
        "binding_resolution_count": sum(
            row.normalization_status == "RESOLVED" for row in normalized
        ),
        "support_resolution_count": sum(
            row.support_resolution_status == "RESOLVED" for row in normalized
        ),
        "unresolved_legacy_fields": unresolved_legacy_fields,
        "duplicate_node_refs": duplicate_node_refs,
        "duplicate_binding_refs": duplicate_binding_refs,
        "orphan_source_refs": orphan_source_refs,
        "orphan_phase_refs": orphan_phase_refs,
        "known_omission_records_parsed": sum(len(rows) for rows in sources.values()),
        "known_omission_schema_errors": schema_errors,
        "required_artifact_schemas_readable": sorted(ARTIFACT_CONTRACT_INVENTORY),
        "attempt_2_output_collisions": output_collisions,
        "provider_calls": 0,
        "external_http": 0,
        "semantic_verifier_calls": 0,
        "extractor_calls": 0,
        "holdout_accessed": False,
    }
    return result


def validate_outputs() -> dict[str, Any]:
    required = [HASH_MANIFEST, *OUTPUTS.values()]
    absent = [str(path) for path in required if not (ROOT / path).is_file()]
    if absent:
        raise RuntimeError(f"missing_d3_outputs:{absent}")
    parsed: dict[str, Any] = {}
    for key, path in OUTPUTS.items():
        if path.suffix == ".json":
            parsed[key] = _read_json(path)
    node_rows = parsed["node_ledger"]["rows"]
    obligation_rows = parsed["obligation_ledger"]["rows"]
    binding_rows = parsed["binding_ledger"]["rows"]
    if len(node_rows) != 72 or len({(row["claim_id"], row["node_ref"]) for row in node_rows}) != 72:
        raise RuntimeError("d3_node_ledger_must_contain_72_unique_rows")
    if (
        len(binding_rows) != 75
        or len({(row["claim_id"], row["binding_ref"]) for row in binding_rows}) != 75
    ):
        raise RuntimeError("d3_binding_ledger_must_contain_75_unique_rows")
    if any(
        not row.get("obligation_ref") or "final_obligation_state" not in row
        for row in obligation_rows
    ):
        raise RuntimeError("d3_obligation_ledger_missing_required_fields")
    if (
        parsed["report_json"].get("provider_calls") != 0
        or parsed["report_json"].get("external_http") != 0
    ):
        raise RuntimeError("d3_external_execution_invariant_failed")
    if parsed["report_json"].get("attempt_id") != ATTEMPT_ID:
        raise RuntimeError("d3_attempt_identity_mismatch")
    if any(
        row.get("normalization_status") != "RESOLVED"
        or not row.get("semantic_role")
        or not row.get("normalization_source_ref")
        for row in binding_rows
    ):
        raise RuntimeError("d3_normalized_binding_provenance_incomplete")
    return {
        "status": "PASS",
        "node_rows": len(node_rows),
        "obligation_rows": len(obligation_rows),
        "binding_rows": len(binding_rows),
        "json_artifacts_parsed": len(parsed),
        "provider_calls": 0,
        "external_http": 0,
    }


HARNESS_VERSION = "grounding-requirements-d3-analysis-harness.v3"
HARNESS_REPORT_SCHEMA_VERSION = "grounding-requirements-d3-report-schema.v2"
HARNESS_JSON_PATH = Path("artifacts/v2_1r_grounding_requirement_d3_2_harness_validation.json")
HARNESS_MARKDOWN_PATH = Path("artifacts/v2_1r_grounding_requirement_d3_2_harness_validation.md")
HARNESS_REQUIRED_ARTIFACTS = (
    "node_ledger",
    "obligation_ledger",
    "binding_ledger",
    "omission_comparison",
    "cardinality",
    "requiredness",
    "usability",
    "readiness",
    "coverage",
    "provenance",
    "hash_manifest",
    "report_json",
    "report_markdown",
)


def _harness_freeze_manifest() -> dict[str, Any]:
    manifest = freeze_manifest()
    manifest.update(
        {
            "harness_version": HARNESS_VERSION,
            "report_schema_version": HARNESS_REPORT_SCHEMA_VERSION,
            "artifact_hash_policy": "SHA256 over canonical JSON values; raw bytes for source files",
        }
    )
    return manifest


def _run_isolated_harness_pipeline() -> dict[str, Any]:
    """Run all analysis/report builders in a temp root; never publish D3 results."""
    global ROOT, ATTEMPT_ID
    source_root = ROOT
    original_attempt = ATTEMPT_ID
    preflight = reconstruction_dry_run(check_output_collisions=False)
    if preflight["status"] != "PASS":
        return {
            "status": "HARNESS_NOT_READY",
            "stage": "RECONSTRUCTION_GATE",
            "preflight_status": preflight["status"],
            "missing_required_artifacts": list(HARNESS_REQUIRED_ARTIFACTS),
        }

    manifest = _harness_freeze_manifest()
    with tempfile.TemporaryDirectory(prefix="d3-harness-validation-") as temp_name:
        temp_root = Path(temp_name)
        for relative in INPUT_PATHS:
            source = source_root / relative
            destination = temp_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        hash_destination = temp_root / HASH_MANIFEST
        hash_destination.parent.mkdir(parents=True, exist_ok=True)
        hash_destination.write_text(
            json.dumps(jsonable(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        ROOT = temp_root
        ATTEMPT_ID = "D3_ANALYSIS_HARNESS_ONLY"
        try:
            canary = _read_json(CANARY_PATH)
            phase_c = _read_json(PHASE_C_PATH)
            _write_json_ledgers(
                manifest=manifest,
                claims=canary["claims"],
                phase_c=phase_c,
                source_schema_version=str(canary.get("schema_version", "")),
                publish=True,
                compute_semantic_decision=False,
                execution_mode="ISOLATED_HARNESS_ONLY",
            )
            parsed_validation = validate_outputs()
            parsed_artifacts: dict[str, Any] = {}
            for key, path in OUTPUTS.items():
                if path.suffix == ".json":
                    parsed_artifacts[key] = _read_json(path)
                else:
                    parsed_artifacts[key] = (ROOT / path).read_text(encoding="utf-8")
            contract_check = validate_report_artifacts(parsed_artifacts)
            report = parsed_artifacts["report_json"]
            if (
                report.get("semantic_decision_computed") is not False
                or report.get("decision") is not None
            ):
                raise RuntimeError("harness_semantic_output_not_quarantined")
            output_hashes = {
                key: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                for key, path in OUTPUTS.items()
            }
            hash_file_sha256 = hashlib.sha256((ROOT / HASH_MANIFEST).read_bytes()).hexdigest()
            if contract_check.status != "PASS" or parsed_validation["status"] != "PASS":
                raise RuntimeError("harness_artifact_validation_failed")
            return {
                "status": "HARNESS_READY",
                "harness_version": HARNESS_VERSION,
                "report_schema_version": HARNESS_REPORT_SCHEMA_VERSION,
                "failed_attempt_history": [
                    {
                        "attempt_id": "D3_FORMAL_ATTEMPT_1",
                        "status": "INVALID_FOR_SEMANTIC_DECISION",
                        "classification": "ANALYSIS_ADAPTER_CONTRACT_FAILURE",
                    },
                    {
                        "attempt_id": "D3_FORMAL_ATTEMPT_2",
                        "status": "INVALID_FOR_SEMANTIC_DECISION",
                        "classification": "ANALYSIS_REPORTING_AGGREGATION_FAILURE",
                    },
                ],
                "cardinality_failure_root_cause": {
                    "expression": "sum(predicate and row['matched_bindings'] for row in rows)",
                    "expected_type": "integer count of partially satisfied obligations",
                    "actual_type": "list[str] binding refs when predicate is true",
                    "producer": "_write_json_ledgers obligation row; "
                    "matched_bindings=list(match.matched_binding_refs)",
                    "consumer": "_cardinality_report partial_count sum",
                    "classification": "REPORT_ONLY_IMPLEMENTATION_BUG / LIST_VS_COUNT_AMBIGUITY",
                    "repair": "explicit typed count plus separate matched binding instance list",
                },
                "cardinality_report_contract_fields": [
                    "group_count",
                    "obligation_count",
                    "expected_item_count",
                    "matched_item_count",
                    "satisfied_count",
                    "partial_count",
                    "failed_count",
                    "unresolved_count",
                    "instance_lists",
                ],
                "report_builder_coverage": list(HARNESS_REQUIRED_ARTIFACTS),
                "serialization_types_covered": [
                    "Enum",
                    "tuple",
                    "set",
                    "Decimal",
                    "UUID",
                    "datetime",
                    "nested diagnostics",
                    "optional fields",
                    "lists of refs",
                    "cardinality structures",
                ],
                "edge_cases_covered": [
                    "empty reports",
                    "zero candidates",
                    "multiple candidates",
                    "partial PER_TARGET",
                    "partial PER_VALUE",
                    "missing ALL_EXPECTED_MEMBERS item",
                    "legacy missing role",
                    "missing support provenance",
                    "unsupported normalized row schema",
                ],
                "preflight": "PASS",
                "artifact_contract_validation": "PASS",
                "parse_back_validation": "PASS",
                "semantic_decision_computed": False,
                "semantic_output_quarantined": True,
                "required_artifacts": list(HARNESS_REQUIRED_ARTIFACTS),
                "artifact_hashes": output_hashes,
                "hash_manifest_sha256": hash_file_sha256,
                "input_sha256": manifest["input_hashes"],
                "semantic_implementation_fingerprint": manifest[
                    "semantic_implementation_fingerprint"
                ],
                "semantic_implementation_unchanged": manifest["semantic_freeze_unchanged"],
                "provider_calls": 0,
                "external_http": 0,
                "semantic_verifier_calls": 0,
                "extractor_calls": 0,
                "holdout_accessed": False,
                "formal_attempt_started": False,
                "isolated_output_root": "temporary_directory_removed_after_validation",
            }
        finally:
            ROOT = source_root
            ATTEMPT_ID = original_attempt


def validate_harness() -> dict[str, Any]:
    """Execute the full pipeline in isolation and persist only harness metadata."""
    result = _run_isolated_harness_pipeline()
    result["artifact_version"] = "v2_1r_grounding_requirement_d3_2_harness_validation.v1"
    json_path = ROOT / HARNESS_JSON_PATH
    markdown_path = ROOT / HARNESS_MARKDOWN_PATH
    markdown = "\n".join(
        [
            "# V2.1R D3.2 Replay Harness Validation",
            "",
            f"- Status: `{result['status']}`",
            f"- Harness: `{result.get('harness_version', HARNESS_VERSION)}`",
            "- Report schema: "
            f"`{result.get('report_schema_version', HARNESS_REPORT_SCHEMA_VERSION)}`",
            f"- Complete pipeline report builders: {len(result.get('required_artifacts', []))}",
            "- Formal semantic decision computed: "
            f"`{result.get('semantic_decision_computed', False)}`",
            f"- Provider / HTTP / verifier / extractor calls: {result.get('provider_calls', 0)}/"
            f"{result.get('external_http', 0)}/{result.get('semantic_verifier_calls', 0)}/"
            f"{result.get('extractor_calls', 0)}",
            f"- Holdout accessed: `{result.get('holdout_accessed', False)}`",
            f"- Formal D3 attempt started: `{result.get('formal_attempt_started', False)}`",
            "",
            "This artifact records harness readiness only. It is not a D3 development result.",
        ]
    )
    payload = json.dumps(jsonable(result), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    json.loads(payload)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".d3-harness-summary-", dir=json_path.parent
    ) as temp_name:
        stage = Path(temp_name)
        staged_json = stage / json_path.name
        staged_markdown = stage / markdown_path.name
        staged_json.write_text(payload, encoding="utf-8")
        staged_markdown.write_text(markdown, encoding="utf-8")
        json.loads(staged_json.read_text(encoding="utf-8"))
        os.replace(staged_json, json_path)
        os.replace(staged_markdown, markdown_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--validate-harness", action="store_true")
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--replay", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        result = reconstruction_dry_run()
        print(json.dumps(result, ensure_ascii=False))
        if result["status"] != "PASS":
            raise SystemExit(1)
        return
    if args.validate_harness:
        result = validate_harness()
        print(json.dumps(jsonable(result), ensure_ascii=False))
        if result["status"] != "HARNESS_READY":
            raise SystemExit(1)
        return
    if args.freeze_only:
        if (ROOT / HASH_MANIFEST).exists():
            raise SystemExit("refusing_to_overwrite_existing_hash_manifest")
        manifest = freeze_manifest()
        print(
            json.dumps(
                {
                    "status": "FROZEN",
                    "fingerprint": manifest["normalized_implementation_fingerprint"],
                    "hash_manifest": str(HASH_MANIFEST),
                },
                ensure_ascii=False,
            )
        )
        return
    if args.validate_only:
        print(json.dumps(validate_outputs(), ensure_ascii=False))
        return
    manifest = _verify_frozen_manifest()
    existing = [str(path) for path in OUTPUTS.values() if (ROOT / path).exists()]
    if existing:
        raise SystemExit(f"refusing_to_overwrite_existing_d3_artifacts:{existing}")
    canary = _read_json(CANARY_PATH)
    phase_c = _read_json(PHASE_C_PATH)
    report = _write_json_ledgers(
        manifest=manifest,
        claims=canary["claims"],
        phase_c=phase_c,
        source_schema_version=str(canary.get("schema_version", "")),
    )
    validation = validate_outputs()
    print(
        json.dumps({"decision": report["decision"], "validation": validation}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
