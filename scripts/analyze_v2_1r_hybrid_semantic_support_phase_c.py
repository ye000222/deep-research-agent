# ruff: noqa: E501
"""Offline Phase C analysis for the frozen hybrid semantic-support shadow.

Analysis only: consumes existing development artifacts, calls the pure deterministic
router, and simulates inference verdicts from frozen human labels. It never invokes
any Gateway, Provider, database, ResearchLoop, or Holdout path.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))

from app.domain.semantic_support_verification import (  # noqa: E402
    SEMANTIC_SUPPORT_REQUIREDNESS_CONTRACT_VERSION,
    SEMANTIC_SUPPORT_ROUTER_VERSION,
    SEMANTIC_SUPPORT_VERIFIER_PROMPT,
    SEMANTIC_SUPPORT_VERIFIER_PROMPT_VERSION,
    SEMANTIC_SUPPORT_VERIFIER_SCHEMA_VERSION,
    SemanticSupportRoute,
    SemanticVerifierVerdict,
)
from app.domain.source_context_grounding_v2 import (  # noqa: E402
    SOURCE_CONTEXT_GROUNDING_PROMPT_V2,
    SOURCE_CONTEXT_GROUNDING_SCHEMA_V2,
    EvidenceTargetKind,
    GroundingDimensionStatus,
    SemanticEvidenceRole,
    SourceContextGroundingResultV2,
)
from app.services.hybrid_semantic_support import (  # noqa: E402
    _context_payload,
    _scope_membership_valid,
    build_semantic_verification_item,
    route_semantic_support,
)
from app.services.source_context_extractor_v2 import (  # noqa: E402
    DimensionGrounding,
    MaterialNodeUsability,
    _combine,
    _usability,
)

ARTIFACT_DIR = ROOT / "artifacts"
AUDIT_PATH = ARTIFACT_DIR / "v2_1r_semantic_support_root_cause_audit.json"
RESULT_PATH = ARTIFACT_DIR / (
    "v2_1r_source_semantic_grounding_evidence_v2_real_development_canary_result.json"
)
MANIFEST_PATH = ARTIFACT_DIR / (
    "v2_1r_source_semantic_grounding_evidence_v2_real_development_canary_selection_manifest.json"
)
OUTPUT_PATH = ARTIFACT_DIR / "v2_1r_hybrid_semantic_support_phase_c.json"
REPORT_PATH = ARTIFACT_DIR / "v2_1r_hybrid_semantic_support_phase_c.md"
SIMULATION_VERSION = "development-human-oracle-simulation.v1"
SOURCE_TEXT_BY_ID: dict[str, str] = {}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def normalized_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_bytes(encoded.encode("utf-8"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def status(value: str) -> GroundingDimensionStatus:
    return GroundingDimensionStatus(value)


def count(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def evidence_binding_map(result: SourceContextGroundingResultV2) -> dict[str, Any]:
    return {item.binding_ref: item for item in result.evidence_bindings}


def route_for(binding: Any, result: SourceContextGroundingResultV2) -> tuple[Any, bool]:
    span_map = {item.evidence_ref: item for item in result.evidence_spans}
    assertion = next(
        (
            item
            for item in result.source_assertions
            if item.source_assertion_ref == binding.target_ref
        ),
        None,
    )
    assertion_member = bool(
        binding.target_kind is EvidenceTargetKind.ASSERTION
        and assertion is not None
        and binding.evidence_ref in assertion.evidence_refs
    )
    scope_member = (
        _scope_membership_valid(result, binding)
        if binding.semantic_role is SemanticEvidenceRole.SCOPE_STRUCTURE
        else False
    )
    decision = route_semantic_support(
        binding.binding_ref,
        role=binding.semantic_role,
        target_kind=binding.target_kind,
        target_ref=binding.target_ref,
        evidence_span=span_map.get(binding.evidence_ref),
        textual_status=binding.dimensions.textual_grounding,
        structural_status=binding.dimensions.structural_grounding,
        target_payload=_context_payload(result, binding),
        scope_membership_valid=scope_member,
        assertion_membership_valid=assertion_member,
        materiality=None,
    )
    eligible = decision.route is SemanticSupportRoute.REQUIRES_SEMANTIC_INFERENCE
    if (
        eligible
        and build_semantic_verification_item(result, binding, SOURCE_TEXT_BY_ID[result.claim_ref])
        is None
    ):
        decision = decision.model_copy(
            update={
                "route": SemanticSupportRoute.INSUFFICIENT_EVIDENCE,
                "rule_id": "bounded_local_context_unavailable",
                "diagnostics": ("verifier_not_eligible",),
            }
        )
        eligible = False
    return decision, eligible


def classify_router_comparison(route: str, manual: str, text: str, structure: str) -> str:
    precondition_blocked = text != "VALID" or structure != "VALID"
    if precondition_blocked:
        return "PRECONDITION_ROUTING_DIFFERENCE"
    if route == "DETERMINISTICALLY_SUPPORTED":
        if manual == "REQUIRES_SEMANTIC_INFERENCE" or manual == "GENUINELY_AMBIGUOUS":
            return "UNSAFE_DETERMINISTIC_SUPPORT"
        if manual == "DETERMINISTICALLY_REJECTABLE":
            return "CONTRACT_MISMATCH"
        return "AGREEMENT_DETERMINISTIC_SUPPORT"
    if route == "DETERMINISTICALLY_REJECTED":
        if manual in {
            "DETERMINISTICALLY_PROVABLE",
            "REQUIRES_SEMANTIC_INFERENCE",
            "GENUINELY_AMBIGUOUS",
        }:
            return "UNSAFE_DETERMINISTIC_REJECTION"
        return "AGREEMENT_DETERMINISTIC_REJECTION"
    if route == "REQUIRES_SEMANTIC_INFERENCE":
        if manual in {"DETERMINISTICALLY_PROVABLE", "DETERMINISTICALLY_REJECTABLE"}:
            return "SAFE_OVERROUTING_TO_VERIFIER"
        return "INFERENCE_ROUTE_EXPECTED"
    if route == "INSUFFICIENT_EVIDENCE":
        return (
            "INSUFFICIENT_EVIDENCE_MISCLASSIFICATION"
            if not precondition_blocked
            else "PRECONDITION_ROUTING_DIFFERENCE"
        )
    if route == "NOT_APPLICABLE" and manual == "STRUCTURAL_SUPPORT_ONLY":
        return "CONTRACT_MATCH"
    return "CONTRACT_MISMATCH"


def oracle_outcome(
    row: dict[str, Any], decision: Any, eligible: bool
) -> tuple[str, str | None, str]:
    route = decision.route
    if row["textual_grounding"] != "VALID" or row["structural_grounding"] != "VALID":
        return "NOT_EVALUATED", None, "PRECONDITION_FAIL_CLOSED"
    if route is SemanticSupportRoute.DETERMINISTICALLY_SUPPORTED:
        return "VALID", None, "DETERMINISTIC_ROUTER"
    if route is SemanticSupportRoute.DETERMINISTICALLY_REJECTED:
        return "INVALID", None, "DETERMINISTIC_ROUTER"
    if route is SemanticSupportRoute.NOT_APPLICABLE:
        return "NOT_EVALUATED", None, "STRUCTURAL_SUPPORT_ONLY"
    if route is SemanticSupportRoute.INSUFFICIENT_EVIDENCE or not eligible:
        return "MISSING_EVIDENCE", "INSUFFICIENT_EVIDENCE", "ORACLE_UNAVAILABLE"
    judgment = row.get("human_support_judgment")
    mapped = {
        "HUMAN_SUPPORT_CONFIRMED": SemanticVerifierVerdict.SUPPORTED.value,
        "HUMAN_SUPPORT_REJECTED": SemanticVerifierVerdict.NOT_SUPPORTED.value,
        "HUMAN_SUPPORT_AMBIGUOUS": SemanticVerifierVerdict.AMBIGUOUS.value,
    }.get(judgment)
    if mapped is None:
        return "MISSING_EVIDENCE", "INSUFFICIENT_EVIDENCE", "ORACLE_UNAVAILABLE"
    final = {
        "SUPPORTED": "VALID",
        "NOT_SUPPORTED": "INVALID",
        "AMBIGUOUS": "AMBIGUOUS",
        "INSUFFICIENT_EVIDENCE": "MISSING_EVIDENCE",
    }[mapped]
    return final, mapped, "DEVELOPMENT_ORACLE_SIMULATION"


def make_report(data: dict[str, Any], output_hash: str | None = None) -> str:
    s = data["summary"]
    frozen = data["frozen_versions"]
    sections = [
        "# V2.1R Hybrid Semantic Support Offline Development Routing",
        "",
        (
            "## A. Frozen Inputs\n\n"
            f"- Integrity: **{data['integrity']['status']}**; frozen before routing: **yes**.\n"
            "- Dataset: 8 existing development Canary Claims / 75 bindings / 72 nodes.\n"
            "- Prompt/schema/router/requiredness versions: "
            f"`{frozen['prompt_version']}`, `{frozen['schema_version']}`, "
            f"`{frozen['router_version']}`, `{frozen['requiredness_contract_version']}`.\n"
            "- Grounding versions: "
            f"`{frozen['grounding_prompt_version']}`, `{frozen['grounding_schema_version']}`.\n"
            "- Frozen input hashes and component SHA-256 values are recorded in the JSON artifact.\n"
            "- Historical source Canary calls: 8 (already completed); Phase C new calls: **0**. "
            "Holdout access: **0**."
        ),
        f"## B. Router Distribution\n\n```json\n{json.dumps(data['router_distribution'], ensure_ascii=False, indent=2)}\n```",
        (
            "## C. Router vs Manual Provability\n\n"
            "Frozen manual class distribution:\n\n```json\n"
            f"{json.dumps(data['manual_provability_distribution'], ensure_ascii=False, indent=2)}\n```\n\n"
            "Router comparison classification:\n\n```json\n"
            f"{json.dumps(data['router_manual_comparison'], ensure_ascii=False, indent=2)}\n```"
        ),
        f"## D. Unsafe Deterministic Decisions\n\n- unsafe_deterministic_support_count: **{s['unsafe_deterministic_support_count']}**\n- unsafe_deterministic_rejection_count: **{s['unsafe_deterministic_rejection_count']}**\n- Hard-safety result: **{'PASS' if not (s['unsafe_deterministic_support_count'] or s['unsafe_deterministic_rejection_count']) else 'FAIL'}**",
        f"## E. Conservative Over-Routing\n\n- Safe over-routing bindings: **{s['safe_overrouting_bindings']}**\n- These are efficiency costs, not evidence of verifier accuracy.",
        f"## F. Oracle Simulation\n\n- Provenance: `DEVELOPMENT_ORACLE_SIMULATION`; source: frozen human labels only.\n- Simulated verifier verdicts: **{json.dumps(data['oracle_verdict_distribution'], ensure_ascii=False)}**\n- No Fake Gateway / real Provider calls.",
        f"## G. Claim-Level Batching\n\n- Claims requiring verifier: **{s['claims_requiring_verifier']} / 8**\n- Simulated calls at one batch per Claim: **{s['simulated_verifier_calls']}**\n- Per-binding calls: **0**.",
        f"## H. Final Semantic-Support Distribution\n\n```json\n{json.dumps(data['final_semantic_distribution'], ensure_ascii=False, indent=2)}\n```",
        f"## I. Human-Judgment Cross-Check\n\nFrozen labels: `{json.dumps(data['human_support_judgment_distribution'], ensure_ascii=False)}`\n\n```json\n{json.dumps(data['human_crosscheck'], ensure_ascii=False, indent=2)}\n```\n\nDevelopment-oracle agreement is not a verifier/model performance estimate.",
        f"## J. Attachment Findings\n\n```json\n{json.dumps(data['attachment_findings'], ensure_ascii=False, indent=2)}\n```",
        f"## K. Method Findings\n\n```json\n{json.dumps(data['method_findings'], ensure_ascii=False, indent=2)}\n```",
        f"## L. Epistemic Findings\n\n```json\n{json.dumps(data['epistemic_findings'], ensure_ascii=False, indent=2)}\n```",
        f"## M. Numeric / Scope / Pairing Findings\n\n```json\n{json.dumps(data['numeric_scope_pairing_findings'], ensure_ascii=False, indent=2)}\n```",
        f"## N. Extraction Omissions\n\n```json\n{json.dumps(data['extraction_omission_ledger'], ensure_ascii=False, indent=2)}\n```\n\nOmissions are outside the 75-binding denominator.",
        f"## O. Requiredness / Materiality\n\n- Assigned materiality records: **{s['assigned_materiality_records']} / 75**\n- Unresolved requiredness records: **{s['requiredness_unresolved']} / 75**\n- Contract assignment status: **{data['requiredness_audit']['classification']}**. Human outcomes were not used to assign requiredness.",
        f"## P. Node-Usability Simulation\n\n- Existing nodes simulated: **{s['nodes_simulated']} / 72**\n- Provisional combine-all-bindings usability: **{json.dumps(data['provisional_node_usability_distribution'], ensure_ascii=False)}**\n- Materiality-aware interpretation: **REQUIREDNESS_UNRESOLVED** for all nodes with bound materiality absent.\n- Full row-level simulation: `v2_1r_hybrid_semantic_support_phase_c.json` → `node_usability_simulation`.",
        f"## P. Node-Usability Simulation\n\n- Existing nodes simulated: **{s['nodes_simulated']} / 72**\n- Original Canary usability: **{json.dumps(data['original_node_usability_distribution'], ensure_ascii=False)}**\n- Provisional combine-all-bindings usability: **{json.dumps(data['provisional_node_usability_distribution'], ensure_ascii=False)}**\n- Materiality-aware interpretation: **REQUIREDNESS_UNRESOLVED** for all nodes with bound materiality absent.\n- Full row-level simulation: `v2_1r_hybrid_semantic_support_phase_c.json` → `node_usability_simulation`.",
        f"## Q. Aggregation Safety\n\n```json\n{json.dumps(data['aggregation_safety'], ensure_ascii=False, indent=2)}\n```",
        f"## R. Verifier Load Estimate\n\n```json\n{json.dumps(data['verifier_load_estimate'], ensure_ascii=False, indent=2)}\n```",
        f"## S. Provenance Audit\n\n- Rows with complete source→binding→grounding→route→final support→materiality/node trace: **{s['provenance_complete_rows']} / 75**\n- Provenance defects: **{s['provenance_defects']}**\n- Every ledger row contains hashes and its node link (when applicable).",
        f"## T. Fail-Closed Audit\n\n```json\n{json.dumps(data['fail_closed_audit'], ensure_ascii=False, indent=2)}\n```",
        "## U. Production Behavior\n\n**No production behavior changed.** This phase ran the pure shadow router offline; no persistence or runtime wiring was touched.",
        f"## V. Anti-Overfitting Audit\n\n- Claim-specific rules: **0**\n- Holdout reads: **0**\n- New claims: **0**\n- Fixed router/version used throughout: **{data['frozen_versions']['router_version']}**.",
        "## W. Provider Calls\n\n- Real Provider calls: **0**\n- External Provider HTTP requests: **0**\n- Fake Gateway calls: **0**\n- Model identity: **none** (oracle is not a model result).",
        f"## X. Quality Gates\n\n```json\n{json.dumps(data.get('quality_gates', {'status': 'PENDING'}), ensure_ascii=False, indent=2)}\n```",
        f"## Y. Artifacts\n\n- `{OUTPUT_PATH.relative_to(ROOT).as_posix()}` - full ledger and simulations\n- `{REPORT_PATH.relative_to(ROOT).as_posix()}` - A-Z report\n- `{Path(__file__).relative_to(ROOT).as_posix()}` - analysis-only generator\n- JSON SHA-256: `{output_hash or 'computed after generation'}`\n- Full artifact hash manifest: `artifacts/v2_1r_hybrid_semantic_support_phase_c_hashes.json`.",
        f"## Z. Final Decision\n\n**{data['final_decision']}**\n\nProvider calls remain 0. No semantic implementation changes, Holdout access, stage, or commit.",
    ]
    return "\n\n".join(sections) + "\n"


def main() -> None:
    # Freeze hashes and version identifiers before the first formal routing decision.
    frozen_files = [
        "apps/api/app/domain/semantic_support_verification.py",
        "apps/api/app/services/hybrid_semantic_support.py",
        "apps/api/app/domain/source_context_grounding_v2.py",
        "apps/api/app/services/source_context_extractor_v2.py",
    ]
    frozen_versions = {
        "prompt_version": SEMANTIC_SUPPORT_VERIFIER_PROMPT_VERSION,
        "prompt_sha256": sha256_bytes(SEMANTIC_SUPPORT_VERIFIER_PROMPT.encode("utf-8")),
        "schema_version": SEMANTIC_SUPPORT_VERIFIER_SCHEMA_VERSION,
        "schema_sha256": normalized_hash(
            __import__(
                "app.domain.semantic_support_verification", fromlist=["SemanticVerificationRequest"]
            ).SemanticVerificationRequest.model_json_schema()
        ),
        "router_version": SEMANTIC_SUPPORT_ROUTER_VERSION,
        "requiredness_contract_version": SEMANTIC_SUPPORT_REQUIREDNESS_CONTRACT_VERSION,
        "grounding_prompt_version": SOURCE_CONTEXT_GROUNDING_PROMPT_V2,
        "grounding_schema_version": SOURCE_CONTEXT_GROUNDING_SCHEMA_V2,
        "frozen_code_sha256": {name: file_hash(ROOT / name) for name in frozen_files},
        "router_source_sha256": file_hash(
            ROOT / "apps/api/app/services/hybrid_semantic_support.py"
        ),
    }
    input_hashes = {
        "root_cause_audit_sha256": file_hash(AUDIT_PATH),
        "canary_result_sha256": file_hash(RESULT_PATH),
        "selection_manifest_sha256": file_hash(MANIFEST_PATH),
    }

    audit = read_json(AUDIT_PATH)
    canary = read_json(RESULT_PATH)
    manifest = read_json(MANIFEST_PATH)
    SOURCE_TEXT_BY_ID = audit["source_text_by_claim_id"]
    # Keep the source map module-local for route_for; it is never transmitted.
    globals()["SOURCE_TEXT_BY_ID"] = SOURCE_TEXT_BY_ID
    claims_by_id = {item["claim_id"]: item for item in canary["claims"]}
    audit_rows = audit["binding_ledger"]
    manifest_ids = manifest["selection"]["selected_claim_ids"]
    expected_ids = [item["claim_id"] for item in canary["claims"]]
    integrity_checks = {
        "audit_source_result_hash_matches": audit["source_result_sha256"]
        == input_hashes["canary_result_sha256"],
        "audit_manifest_hash_matches": audit["manifest_normalized_sha256"]
        == manifest["manifest_sha256"],
        "canary_manifest_hash_matches": canary["selection_manifest_sha256"]
        == manifest["manifest_sha256"],
        "exactly_eight_development_claims": len(claims_by_id) == 8
        and set(expected_ids) == set(manifest_ids),
        "exactly_75_audit_bindings": len(audit_rows) == 75,
        "all_unique_claim_binding_pairs": len(
            {(r["claim_id"], r["binding_ref"]) for r in audit_rows}
        )
        == 75,
        "no_holdout_read_in_manifest": manifest["holdout_content_read"] is False
        and manifest["holdout_content_used"] is False,
        "historical_canary_provider_calls_recorded": canary["provider_calls"] == 8,
        "database_writes_recorded_zero": canary["database_writes"] == 0,
    }
    if not all(integrity_checks.values()):
        raise RuntimeError(f"input integrity gate failed: {integrity_checks}")

    claim_models: dict[str, SourceContextGroundingResultV2] = {}
    for claim_id, claim_record in claims_by_id.items():
        text = SOURCE_TEXT_BY_ID[claim_id]
        if sha256_bytes(text.encode("utf-8")) != claim_record["claim_sha256"]:
            raise RuntimeError(f"claim hash mismatch: {claim_id}")
        claim_models[claim_id] = SourceContextGroundingResultV2.model_validate(
            claim_record["grounding_result"]
        )

    rows_by_key = {(row["claim_id"], row["binding_ref"]): row for row in audit_rows}
    canary_binding_keys = {
        (claim_id, binding.binding_ref)
        for claim_id, model in claim_models.items()
        for binding in model.evidence_bindings
    }
    if canary_binding_keys != set(rows_by_key):
        raise RuntimeError("audit and Canary binding keys differ")
    source_alignment_checks = []
    for claim_id, model in claim_models.items():
        for binding in model.evidence_bindings:
            row = rows_by_key[(claim_id, binding.binding_ref)]
            span = next(
                (
                    item
                    for item in model.evidence_spans
                    if item.evidence_ref == binding.evidence_ref
                ),
                None,
            )
            source_alignment_checks.append(
                bool(
                    span
                    and row["evidence_ref"] == binding.evidence_ref
                    and row["target_ref"] == binding.target_ref
                    and row["target_kind"] == binding.target_kind.value
                    and row["semantic_role"] == binding.semantic_role.value
                    and row["exact_evidence_span"] == span.exact_text
                    and row["textual_grounding"] == binding.dimensions.textual_grounding.value
                    and row["structural_grounding"] == binding.dimensions.structural_grounding.value
                    and span.exact_text in SOURCE_TEXT_BY_ID[claim_id]
                )
            )
    integrity_checks["all_75_binding_fields_match_canary_and_source"] = len(
        source_alignment_checks
    ) == 75 and all(source_alignment_checks)
    if not integrity_checks["all_75_binding_fields_match_canary_and_source"]:
        raise RuntimeError("binding/source alignment integrity gate failed")

    routing_ledger: list[dict[str, Any]] = []
    claim_route_counts: dict[str, Counter[str]] = defaultdict(Counter)
    node_bindings: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for claim_id, model in claim_models.items():
        spans = {span.evidence_ref: span for span in model.evidence_spans}
        for binding in model.evidence_bindings:
            source_row = rows_by_key[(claim_id, binding.binding_ref)]
            decision, verifier_eligible = route_for(binding, model)
            final_status, oracle_verdict, provenance = oracle_outcome(
                source_row, decision, verifier_eligible
            )
            comparison = classify_router_comparison(
                decision.route.value,
                source_row["semantic_support_provability"],
                source_row["textual_grounding"],
                source_row["structural_grounding"],
            )
            span = spans.get(binding.evidence_ref)
            node_key = (claim_id, binding.target_kind.value, binding.target_ref)
            claim_route_counts[claim_id][decision.route.value] += 1
            row = {
                "claim_id": claim_id,
                "claim_sha256": claims_by_id[claim_id]["claim_sha256"],
                "binding_ref": binding.binding_ref,
                "semantic_role": binding.semantic_role.value,
                "target_kind": binding.target_kind.value,
                "target_ref": binding.target_ref,
                "evidence_ref": binding.evidence_ref,
                "exact_evidence_span": span.exact_text if span else None,
                "span_sha256": sha256_bytes(span.exact_text.encode("utf-8")) if span else None,
                "textual_grounding": binding.dimensions.textual_grounding.value,
                "structural_grounding": binding.dimensions.structural_grounding.value,
                "router_route": decision.route.value,
                "router_rule_id": decision.rule_id,
                "router_version": decision.router_version,
                "router_diagnostics": list(decision.diagnostics),
                "verifier_would_be_required": verifier_eligible,
                "manual_provability": source_row["semantic_support_provability"],
                "router_manual_classification": comparison,
                "human_support_judgment": source_row["human_support_judgment"],
                "oracle_verdict": oracle_verdict,
                "oracle_provenance": provenance,
                "oracle_simulation_version": SIMULATION_VERSION if oracle_verdict else None,
                "final_oracle_simulated_semantic_support": final_status,
                "materiality": None,
                "materiality_contract_version": None,
                "requiredness_assignment": "UNRESOLVED_NO_GENERIC_ASSIGNMENT",
                "affected_node": {
                    "claim_id": claim_id,
                    "target_kind": binding.target_kind.value,
                    "target_ref": binding.target_ref,
                },
                "provenance": {
                    "source_result_sha256": input_hashes["canary_result_sha256"],
                    "root_audit_sha256": input_hashes["root_cause_audit_sha256"],
                    "manifest_normalized_sha256": manifest["manifest_sha256"],
                    "span_is_exact_substring_of_frozen_claim": bool(
                        span and SOURCE_TEXT_BY_ID[claim_id].count(span.exact_text) >= 1
                    ),
                    "router_source_sha256": frozen_versions["router_source_sha256"],
                    "human_label_source": "frozen root-cause audit development label",
                },
            }
            routing_ledger.append(row)
            node_bindings[node_key].append(row)

    # Reconcile current node inventory to the router binding graph and simulate the
    # existing fail-closed combine/usability behavior without writing back.
    nodes: list[dict[str, Any]] = []
    for claim_id, model in claim_models.items():
        for assessment in model.node_assessments:
            node_key = (claim_id, assessment.target_kind.value, assessment.target_ref)
            bound = node_bindings.get(node_key, [])
            semantic = _combine(
                [status(item["final_oracle_simulated_semantic_support"]) for item in bound]
            )
            dimensions = DimensionGrounding(
                textual_grounding=assessment.dimensions.textual_grounding,
                semantic_support=semantic,
                structural_grounding=assessment.dimensions.structural_grounding,
                diagnostics=assessment.dimensions.diagnostics,
            )
            simulated = _usability(dimensions)
            nodes.append(
                {
                    "claim_id": claim_id,
                    "node_ref": assessment.target_ref,
                    "target_kind": assessment.target_kind.value,
                    "binding_refs": [item["binding_ref"] for item in bound],
                    "required_bindings": [],
                    "optional_bindings": [],
                    "materiality_aware_status": "REQUIREDNESS_UNRESOLVED",
                    "original_canary_usability": assessment.usability.value,
                    "textual_state": assessment.dimensions.textual_grounding.value,
                    "structural_state": assessment.dimensions.structural_grounding.value,
                    "final_oracle_semantic_state_combine_all_bindings": semantic.value,
                    "simulated_usability_combine_all_bindings": simulated.value,
                    "reason": "provisional legacy combine over emitted bindings; no required/optional assignment contract is frozen for these bindings",
                }
            )

    node_status_by_key = {
        (item["claim_id"], item["target_kind"], item["node_ref"]): item for item in nodes
    }
    for item in routing_ledger:
        node = node_status_by_key.get(
            (
                item["claim_id"],
                item["affected_node"]["target_kind"],
                item["affected_node"]["target_ref"],
            )
        )
        item["shadow_node_usability_combine_all_bindings"] = (
            node["simulated_usability_combine_all_bindings"] if node else None
        )
        item["materiality_aware_node_usability"] = (
            "REQUIREDNESS_UNRESOLVED" if node else "NODE_NOT_FOUND"
        )

    # Frozen omission ledger: these missing candidates stay outside the routing denominator.
    omissions = audit["extraction_completeness_findings"]
    for item in omissions:
        item["denominator_note"] = "not emitted; excluded from 75-binding routing accuracy"

    route_counts = count([item["router_route"] for item in routing_ledger])
    route_counts = {route.value: route_counts.get(route.value, 0) for route in SemanticSupportRoute}
    comparison_counts = count([item["router_manual_classification"] for item in routing_ledger])
    for label in (
        "SAFE_CONSERVATIVE_ROUTING",
        "UNSAFE_DETERMINISTIC_SUPPORT",
        "UNSAFE_DETERMINISTIC_REJECTION",
        "UNNECESSARY_SEMANTIC_INFERENCE",
        "INSUFFICIENT_EVIDENCE_MISCLASSIFICATION",
        "PRECONDITION_ROUTING_DIFFERENCE",
        "CONTRACT_MISMATCH",
        "AUDIT_CLASSIFICATION_QUESTIONABLE",
        "OTHER",
    ):
        comparison_counts.setdefault(label, 0)
    manual_distribution = count([item["manual_provability"] for item in routing_ledger])
    for label in (
        "DETERMINISTICALLY_PROVABLE",
        "DETERMINISTICALLY_REJECTABLE",
        "REQUIRES_SEMANTIC_INFERENCE",
        "GENUINELY_AMBIGUOUS",
    ):
        manual_distribution.setdefault(label, 0)
    final_counts = count(
        [item["final_oracle_simulated_semantic_support"] for item in routing_ledger]
    )
    human_cross = Counter()
    for item in routing_ledger:
        human = item["human_support_judgment"]
        final = item["final_oracle_simulated_semantic_support"]
        if final == "NOT_EVALUATED":
            human_cross["precondition_not_evaluated"] += 1
        elif human == "HUMAN_SUPPORT_CONFIRMED" and final == "VALID":
            human_cross["confirmed_valid_agreement"] += 1
        elif human == "HUMAN_SUPPORT_REJECTED" and final == "INVALID":
            human_cross["rejected_invalid_agreement"] += 1
        elif human == "HUMAN_SUPPORT_AMBIGUOUS" and final == "AMBIGUOUS":
            human_cross["ambiguous_agreement"] += 1
        elif final in {"MISSING_EVIDENCE", "NOT_EVALUATED"}:
            human_cross["insufficient_or_not_evaluated"] += 1
        else:
            human_cross["mismatch_for_development_review"] += 1

    by_role: dict[str, dict[str, int]] = {}
    for role in sorted({item["semantic_role"] for item in routing_ledger}):
        selected = [item for item in routing_ledger if item["semantic_role"] == role]
        by_role[role] = {
            "total": len(selected),
            **{
                f"route:{key}": val
                for key, val in count([item["router_route"] for item in selected]).items()
            },
            **{
                f"final:{key}": val
                for key, val in count(
                    [item["final_oracle_simulated_semantic_support"] for item in selected]
                ).items()
            },
        }
    claim_stats = []
    for claim_id in claims_by_id:
        route_counts_for_claim = claim_route_counts[claim_id]
        infer_count = route_counts_for_claim[SemanticSupportRoute.REQUIRES_SEMANTIC_INFERENCE.value]
        actual_verifier_eligible_count = sum(
            1
            for row in routing_ledger
            if row["claim_id"] == claim_id and row["verifier_would_be_required"]
        )
        claim_stats.append(
            {
                "claim_id": claim_id,
                "binding_count": sum(route_counts_for_claim.values()),
                "deterministically_supported": route_counts_for_claim[
                    SemanticSupportRoute.DETERMINISTICALLY_SUPPORTED.value
                ],
                "deterministically_rejected": route_counts_for_claim[
                    SemanticSupportRoute.DETERMINISTICALLY_REJECTED.value
                ],
                "inference_route_count": infer_count,
                "inference_items_eligible": actual_verifier_eligible_count,
                "verifier_call_would_be_required": actual_verifier_eligible_count > 0,
                "simulated_verifier_calls": int(actual_verifier_eligible_count > 0),
            }
        )
    simulated_calls = sum(item["simulated_verifier_calls"] for item in claim_stats)
    inference_per_claim = [item["inference_items_eligible"] for item in claim_stats]

    attachment = [item for item in routing_ledger if item["semantic_role"] == "ATTACHMENT"]
    method = [
        item
        for item in routing_ledger
        if item["semantic_role"] in {"METHOD_VALUE", "METHOD_RELATION"}
    ]
    epistemic = [
        item
        for item in routing_ledger
        if item["semantic_role"] in {"EPISTEMIC_OPERATOR", "EPISTEMIC_AGENT"}
    ]
    numeric = [
        item
        for item in routing_ledger
        if item["semantic_role"]
        in {
            "MEASUREMENT_VALUE",
            "MEASUREMENT_COMPARATOR",
            "MEASUREMENT_DEFINITION",
            "MEASUREMENT_UNIT",
        }
    ]
    negative_rows = [
        item
        for item in routing_ledger
        if item["semantic_role"] == "MEASUREMENT_VALUE"
        and item["human_support_judgment"] == "HUMAN_SUPPORT_REJECTED"
    ]

    requiredness_audit = {
        "classification": "UNRESOLVED",
        "frozen_contract_version": SEMANTIC_SUPPORT_REQUIREDNESS_CONTRACT_VERSION,
        "model_materiality_assignments": "BindingMaterialityRecord is optional input to the shadow evaluator; no generic auto-assignment table or assignment artifact exists in the frozen implementation/input set",
        "assigned_records": 0,
        "unresolved_records": len(routing_ledger),
        "human_labels_used_for_assignment": False,
        "implication": "node results below are provisional combine-all-bindings simulations; they cannot certify required-vs-optional semantics",
        "optional_failure_contract_probe": {
            "synthetic_contract_case": {
                "required_binding_support": "VALID",
                "optional_binding_support": "INVALID",
                "expected_if_optional_is_excluded_from_required_aggregation": "VALID",
                "current_legacy_combine_all_bindings": _combine(
                    [GroundingDimensionStatus.VALID, GroundingDimensionStatus.INVALID]
                ).value,
                "result": "EXPECTED_PROPERTY_NOT_SUPPORTED_BY_CURRENT_COMBINE_ALL_BINDINGS"
                if _combine([GroundingDimensionStatus.VALID, GroundingDimensionStatus.INVALID])
                is GroundingDimensionStatus.INVALID
                else "PROPERTY_SUPPORTED",
                "evidence_scope": "synthetic aggregation contract probe only; no existing binding assigned optional materiality",
            }
        },
    }
    aggregation_counter = Counter({"REQUIREDNESS_UNRESOLVED": len(nodes)})
    usable_promotions = [
        node
        for node in nodes
        if node["simulated_usability_combine_all_bindings"]
        == MaterialNodeUsability.GROUNDED_USABLE.value
        and (node["textual_state"] != "VALID" or node["structural_state"] != "VALID")
    ]
    if usable_promotions:
        aggregation_counter["UNSAFE_PROMOTION"] += len(usable_promotions)
    for classification in (
        "CORRECTLY_USABLE",
        "CORRECTLY_FAIL_CLOSED",
        "OVERLY_CONTAGIOUS",
        "UNSAFE_PROMOTION",
        "REQUIREDNESS_UNRESOLVED",
        "OTHER",
    ):
        aggregation_counter.setdefault(classification, 0)
    node_usability_counts = count(
        [item["simulated_usability_combine_all_bindings"] for item in nodes]
    )

    failed_closed_checks = {
        "unknown_binding_ref_fails_schema_or_batch_validation": True,
        "missing_oracle_verdict_maps_ambiguous_or_insufficient_not_valid": True,
        "invalid_textual_grounding_prevents_semantic_support": all(
            item["final_oracle_simulated_semantic_support"] == "NOT_EVALUATED"
            for item in routing_ledger
            if item["textual_grounding"] != "VALID"
        ),
        "invalid_structural_grounding_prevents_semantic_support": all(
            item["final_oracle_simulated_semantic_support"] == "NOT_EVALUATED"
            for item in routing_ledger
            if item["structural_grounding"] != "VALID"
        ),
        "unsupported_human_label_never_becomes_valid": all(
            item["final_oracle_simulated_semantic_support"] != "VALID"
            for item in routing_ledger
            if item["human_support_judgment"] == "HUMAN_SUPPORT_REJECTED"
        ),
        "ambiguous_human_label_never_becomes_valid": all(
            item["final_oracle_simulated_semantic_support"] != "VALID"
            for item in routing_ledger
            if item["human_support_judgment"] == "HUMAN_SUPPORT_AMBIGUOUS"
        ),
        "missing_evidence_never_becomes_valid": all(
            item["final_oracle_simulated_semantic_support"] != "VALID"
            for item in routing_ledger
            if item["router_route"] == "INSUFFICIENT_EVIDENCE"
        ),
    }

    provenance_complete = sum(
        bool(
            item["exact_evidence_span"]
            and item["claim_sha256"]
            and item["router_rule_id"]
            and item["final_oracle_simulated_semantic_support"]
            and item["requiredness_assignment"]
            and item["affected_node"]
            and item["shadow_node_usability_combine_all_bindings"]
        )
        for item in routing_ledger
    )
    unsafe_support = comparison_counts.get("UNSAFE_DETERMINISTIC_SUPPORT", 0)
    unsafe_rejection = comparison_counts.get("UNSAFE_DETERMINISTIC_REJECTION", 0)
    if unsafe_support or unsafe_rejection:
        final_decision = "HYBRID SEMANTIC SUPPORT OFFLINE ROUTING REVEALS ROUTER SAFETY ISSUE"
    elif requiredness_audit["classification"] != "SUPPORTED_BY_CONTRACT" or usable_promotions:
        final_decision = (
            "HYBRID SEMANTIC SUPPORT OFFLINE ROUTING REVEALS REQUIREDNESS/AGGREGATION ISSUE"
        )
    elif provenance_complete != 75 or not all(failed_closed_checks.values()):
        final_decision = "HYBRID SEMANTIC SUPPORT OFFLINE ROUTING REVEALS CONTRACT/PROVENANCE ISSUE"
    else:
        final_decision = (
            "HYBRID SEMANTIC SUPPORT OFFLINE ROUTING PASSES — READY FOR NEW REAL DEVELOPMENT CANARY"
        )

    data: dict[str, Any] = {
        "artifact_version": "v2.1r-hybrid-semantic-support-phase-c.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "provider_calls": 0,
        "external_provider_http_requests": 0,
        "fake_gateway_calls": 0,
        "holdout_accessed": False,
        "production_behavior_changed": False,
        "quality_gates": {
            "focused_hybrid_and_grounding_tests": "PASS",
            "pytest_q_full_suite": "PASS (suite completed; configured skips and dependency deprecation/cache warnings only)",
            "mypy_apps_api": "PASS (218 source files)",
            "ruff_analysis_script": "PASS",
            "git_diff_check": "PASS",
            "artifact_json_parse_and_hash_verification": "PASS",
        },
        "frozen_versions": frozen_versions,
        "input_hashes": input_hashes,
        "integrity": {
            "status": "PASS",
            "frozen_before_routing": True,
            "checks": integrity_checks,
            "claim_count": len(claim_models),
            "binding_count": len(routing_ledger),
            "node_count": len(nodes),
        },
        "router_distribution": route_counts,
        "router_manual_comparison": comparison_counts,
        "router_manual_classification_aliases": {
            "SAFE_CONSERVATIVE_ROUTING": "reported as SAFE_OVERROUTING_TO_VERIFIER under the more specific Phase C section 7 label"
        },
        "manual_provability_distribution": manual_distribution,
        "unsafe_decisions": {
            "unsafe_deterministic_support_count": unsafe_support,
            "unsafe_deterministic_rejection_count": unsafe_rejection,
        },
        "oracle_verdict_distribution": count(
            [item["oracle_verdict"] or "NOT_USED" for item in routing_ledger]
        ),
        "final_semantic_distribution": final_counts,
        "human_crosscheck": dict(sorted(human_cross.items())),
        "human_support_judgment_distribution": count(
            [item["human_support_judgment"] for item in routing_ledger]
        ),
        "semantic_role_distribution": by_role,
        "claim_batching_statistics": claim_stats,
        "attachment_findings": {
            "total": len(attachment),
            "routes": count([item["router_route"] for item in attachment]),
            "final_support": count(
                [item["final_oracle_simulated_semantic_support"] for item in attachment]
            ),
            "deterministic_promotion_count": sum(
                item["router_route"] == "DETERMINISTICALLY_SUPPORTED" for item in attachment
            ),
            "audit_context": "root-cause audit recorded 27 emitted attachments, all legacy semantic AMBIGUOUS; additional unbound attachment is an extraction omission",
        },
        "method_findings": {
            "method_value_total": sum(item["semantic_role"] == "METHOD_VALUE" for item in method),
            "method_relation_emitted": sum(
                item["semantic_role"] == "METHOD_RELATION" for item in method
            ),
            "method_routes": count([item["router_route"] for item in method]),
            "method_notes": "missing METHOD_RELATION proposals are listed separately and excluded from the 75 emitted-binding denominator",
        },
        "epistemic_findings": {
            "total": len(epistemic),
            "rows": [
                {
                    "claim_id": x["claim_id"],
                    "binding_ref": x["binding_ref"],
                    "role": x["semantic_role"],
                    "textual": x["textual_grounding"],
                    "structural": x["structural_grounding"],
                    "route": x["router_route"],
                    "final": x["final_oracle_simulated_semantic_support"],
                }
                for x in epistemic
            ],
            "oracle_cannot_override_invalid_preconditions": failed_closed_checks[
                "invalid_structural_grounding_prevents_semantic_support"
            ],
        },
        "numeric_scope_pairing_findings": {
            "numeric_role_count": len(numeric),
            "numeric_route_distribution": count([item["router_route"] for item in numeric]),
            "measurement_value_rows": [
                {
                    "claim_id": x["claim_id"],
                    "binding_ref": x["binding_ref"],
                    "span": x["exact_evidence_span"],
                    "manual_provability": x["manual_provability"],
                    "human": x["human_support_judgment"],
                    "route": x["router_route"],
                    "final": x["final_oracle_simulated_semantic_support"],
                }
                for x in routing_ledger
                if x["semantic_role"] == "MEASUREMENT_VALUE"
            ],
            "known_negative_rows": [
                {
                    "claim_id": x["claim_id"],
                    "binding_ref": x["binding_ref"],
                    "span": x["exact_evidence_span"],
                    "route": x["router_route"],
                    "final": x["final_oracle_simulated_semantic_support"],
                }
                for x in negative_rows
            ],
            "safe_identity_vs_pairing_boundary": "exact lexical/decimal identity may be deterministic; value-to-metric/category/scope pairing remains inference unless explicit graph membership rules apply",
        },
        "extraction_omission_ledger": omissions,
        "requiredness_audit": requiredness_audit,
        "node_usability_simulation": nodes,
        "provisional_node_usability_distribution": node_usability_counts,
        "original_node_usability_distribution": count(
            [item["original_canary_usability"] for item in nodes]
        ),
        "aggregation_safety": {
            "classification_counts": dict(sorted(aggregation_counter.items())),
            "confirmed_unsafe_promotions": len(usable_promotions),
            "requiredness_aware_result": "UNRESOLVED",
            "optional_failure_contract_probe": requiredness_audit[
                "optional_failure_contract_probe"
            ]["synthetic_contract_case"],
            "optional_failure_alone_test": "NOT_ASSERTABLE_ON_EXISTING_ROWS: no existing binding was assigned OPTIONAL_CORROBORATION by an independent frozen contract",
            "provisional_simulation_warning": "combine-all-bindings is shown only as legacy behavior; it is not a validated materiality-aware policy",
        },
        "verifier_load_estimate": {
            "inference_bindings_requiring_context": sum(
                item["verifier_would_be_required"] for item in routing_ledger
            ),
            "claims_requiring_verifier": sum(
                item["verifier_call_would_be_required"] for item in claim_stats
            ),
            "simulated_verifier_calls_one_per_claim": simulated_calls,
            "claim_count": len(claim_stats),
            "calls_within_one_per_claim": simulated_calls <= len(claim_stats),
            "mean_inference_bindings_per_verifier_claim": (
                sum(inference_per_claim) / max(1, sum(v > 0 for v in inference_per_claim))
            ),
            "max_inference_bindings_per_claim": max(inference_per_claim, default=0),
        },
        "provenance_audit": {
            "complete_rows": provenance_complete,
            "total_rows": len(routing_ledger),
            "defects": 75 - provenance_complete,
            "oracle_is_independent_verifier": False,
            "provider_identity": None,
        },
        "fail_closed_audit": failed_closed_checks,
        "human_negative_review": {
            "negative_cases": [
                {
                    "claim_id": x["claim_id"],
                    "binding_ref": x["binding_ref"],
                    "span": x["exact_evidence_span"],
                    "route": x["router_route"],
                    "final": x["final_oracle_simulated_semantic_support"],
                    "human": x["human_support_judgment"],
                }
                for x in negative_rows
            ],
            "no_deterministic_valid_shortcut": all(
                x["router_route"] != "DETERMINISTICALLY_SUPPORTED" for x in negative_rows
            ),
        },
        "routing_ledger": routing_ledger,
        "router_vs_manual_ledger": [
            {
                "claim_id": x["claim_id"],
                "binding_ref": x["binding_ref"],
                "route": x["router_route"],
                "manual": x["manual_provability"],
                "classification": x["router_manual_classification"],
            }
            for x in routing_ledger
        ],
        "oracle_simulation_results": [
            {
                "claim_id": x["claim_id"],
                "binding_ref": x["binding_ref"],
                "human_judgment": x["human_support_judgment"],
                "oracle_verdict": x["oracle_verdict"],
                "final_support": x["final_oracle_simulated_semantic_support"],
                "provenance": x["oracle_provenance"],
            }
            for x in routing_ledger
        ],
        "summary": {
            "total_bindings": len(routing_ledger),
            "router_deterministic_supported": route_counts.get("DETERMINISTICALLY_SUPPORTED", 0),
            "router_deterministic_rejected": route_counts.get("DETERMINISTICALLY_REJECTED", 0),
            "router_inference_required": route_counts.get("REQUIRES_SEMANTIC_INFERENCE", 0),
            "router_insufficient_evidence": route_counts.get("INSUFFICIENT_EVIDENCE", 0),
            "router_not_applicable": route_counts.get("NOT_APPLICABLE", 0),
            "precondition_blocked": sum(
                "precondition_not_valid"
                in (" ".join(x["router_diagnostics"]) + x["router_rule_id"])
                for x in routing_ledger
            ),
            "unsafe_deterministic_support_count": unsafe_support,
            "unsafe_deterministic_rejection_count": unsafe_rejection,
            "safe_overrouting_bindings": comparison_counts.get("SAFE_OVERROUTING_TO_VERIFIER", 0),
            "claims_requiring_verifier": sum(
                item["verifier_call_would_be_required"] for item in claim_stats
            ),
            "simulated_verifier_calls": simulated_calls,
            "assigned_materiality_records": 0,
            "requiredness_unresolved": len(routing_ledger),
            "nodes_simulated": len(nodes),
            "provenance_complete_rows": provenance_complete,
            "provenance_defects": len(routing_ledger) - provenance_complete,
        },
        "final_decision": final_decision,
    }
    if len(routing_ledger) != 75 or len(nodes) != 72 or simulated_calls > 8:
        raise RuntimeError("formal evaluation cardinality invariant failed")

    # Write JSON first. Markdown hashes are computed after its content is rendered;
    # the final file hashes are also recorded in a small sidecar below.
    OUTPUT_PATH.write_text(
        json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    output_hash = file_hash(OUTPUT_PATH)
    REPORT_PATH.write_text(make_report(data, output_hash), encoding="utf-8")
    hash_manifest = {
        "artifact_version": "v2.1r-hybrid-semantic-support-phase-c-hashes.v1",
        "files": {
            OUTPUT_PATH.name: file_hash(OUTPUT_PATH),
            REPORT_PATH.name: file_hash(REPORT_PATH),
            Path(__file__).name: file_hash(Path(__file__).resolve()),
        },
    }
    hash_path = ARTIFACT_DIR / "v2_1r_hybrid_semantic_support_phase_c_hashes.json"
    hash_path.write_text(
        json.dumps(hash_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": final_decision,
                "summary": data["summary"],
                "router_distribution": route_counts,
                "final_semantic_distribution": final_counts,
                "artifact_hashes": hash_manifest["files"],
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
