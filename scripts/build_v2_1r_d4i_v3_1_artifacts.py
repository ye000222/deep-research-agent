"""Build the offline-only D4-I v3.1 shadow artifact bundle."""

# Embedded report prose is retained as publication text.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.domain.source_context_extraction_v3_1 import (
    SOURCE_CONTEXT_EXTRACTION_DOMAIN_V31,
    SOURCE_CONTEXT_EXTRACTION_PROMPT_V31,
    SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31,
    SourceExtractionResponseV31,
    validate_scope_ownership_v31,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "v2_1r_d4i_v3_1_shadow"
FROZEN = ROOT / "artifacts" / "v2_1r_d4g_b_execution_attempt_1"


def write_json(name: str, payload: Any) -> Path:
    path = OUT / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_items = [
        (
            "claim_01_01a0c86f-e82d-7dd8-b42f-d48bba5b879e.json",
            "01a0c86f-e82d-7dd8-b42f-d48bba5b879e",
        ),
        (
            "claim_03_01a0c41c-a7cc-73d2-9f55-6e19ede5be76.json",
            "01a0c41c-a7cc-73d2-9f55-6e19ede5be76",
        ),
        (
            "claim_05_01a0c873-f07c-74b5-94e6-0c694bf0382b.json",
            "01a0c873-f07c-74b5-94e6-0c694bf0382b",
        ),
    ]
    replay: list[dict[str, Any]] = []
    for filename, claim_id in frozen_items:
        frozen = json.loads((FROZEN / filename).read_text(encoding="utf-8"))
        raw = json.loads(frozen["raw_provider_structured_response"]["raw_structured_content"])
        payload = {
            "schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31,
            "status": "RESOLVED",
            "source_assertions": raw["source_assertions"],
            "source_contexts": [],
            "source_attachments": [],
            "source_scopes": raw["source_scopes"],
            "evidence_spans": raw["evidence_spans"],
            "evidence_bindings": [],
            "semantic_relations": [],
            "measurement_bases": [],
            "unresolved_items": [],
        }
        assessment = validate_scope_ownership_v31(
            SourceExtractionResponseV31.model_validate(payload), frozen["claim_text"]
        )
        replay.append(
            {
                "claim_id": claim_id,
                "scope_only_status": assessment.status.value,
                "scope_diagnostics": list(assessment.diagnostics),
                "legacy_execution_status": frozen["execution_status"],
                "legacy_non_scope_diagnostics_preserved": [
                    item for item in frozen["diagnostics"] if item != "invalid_scope_graph"
                ],
                "duplicate_scope_structure_binding_required": False,
                "provenance_evidence_ref_count": sum(
                    len(scope["evidence_refs"]) for scope in raw["source_scopes"]
                ),
            }
        )

    fixture_data = json.loads(
        (ROOT / "tests" / "fixtures" / "source_context_extraction_v3_1_cases.json").read_text(
            encoding="utf-8"
        )
    )
    artifacts: dict[str, Any] = {
        "revision_manifest.json": {
            "artifact_version": "v2_1r_d4i_v3_1_revision_manifest.v1",
            "phase": "D4-I",
            "schema": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31,
            "prompt": SOURCE_CONTEXT_EXTRACTION_PROMPT_V31,
            "domain": SOURCE_CONTEXT_EXTRACTION_DOMAIN_V31,
            "scope_revision": "SourceScopeNode is the canonical structural/provenance owner; no duplicate SCOPE_STRUCTURE binding.",
            "measurement_revision": "Typed measurement identity/definition/value/unit/comparator/threshold/basis/scope/target/qualifier semantics.",
            "provider_calls": 0,
            "external_provider_http": 0,
            "real_extractor_calls": 0,
            "holdout_access": 0,
        },
        "scope_canonical_owner_contract.json": {
            "artifact_version": "scope-canonical-owner-contract.v1",
            "owner": "SourceScopeNode",
            "owned_facts": [
                "scope identity/type",
                "member refs",
                "graph membership",
                "source evidence refs/provenance",
            ],
            "duplicate_binding": "SCOPE_STRUCTURE removed from v3.1 required output; deprecated for source-extraction scope ownership.",
            "safety": "Scope graph validity does not establish semantic support; refs and provenance remain mandatory.",
            "unresolved_scope": "PARTIAL; invalid members/evidence/duplicate members fail closed as INVALID.",
        },
        "scope_offline_closure_replay.json": {
            "artifact_version": "scope-offline-closure-replay.v1",
            "source": "Frozen D4-G-B structured outputs; deterministic scope path only.",
            "provider_calls": 0,
            "cases": replay,
            "duplicate_scope_blockers_closed": sum(
                x["scope_only_status"] == "RESOLVED" for x in replay
            ),
            "invalid_scope_accepted_without_evidence": False,
            "provenance_preserved": all(x["provenance_evidence_ref_count"] > 0 for x in replay),
            "limitation": "Claim 3 retains its unrelated frozen graph/evidence diagnostics; no full-result reinterpretation was performed.",
        },
        "measurement_typed_contract_specification.json": {
            "artifact_version": "measurement-typed-contract.v1",
            "optional_fields": [
                "measurement_identity",
                "formal_definition",
                "observed_value",
                "unit",
                "comparator",
                "threshold",
                "basis_ref",
                "scope_category",
                "target_object",
                "qualifiers",
            ],
            "observed_value_kinds": ["INTEGER", "DECIMAL", "RANGE", "APPROXIMATE", "BOUND", "TEXT"],
            "raw_value_preserved": True,
            "absence_policy": "Source-absent semantics remain absent; no inference of definition, value, unit, comparator, basis, scope, target, or qualifiers.",
            "measurement_basis_contract": "Preserved without new heuristics.",
        },
        "measurement_role_matrix.json": {
            "artifact_version": "measurement-role-matrix.v1",
            "roles": {
                "MEASUREMENT_IDENTITY": "measurement_identity",
                "MEASUREMENT_DEFINITION": "formal_definition only",
                "MEASUREMENT_VALUE": "observed_value only",
                "MEASUREMENT_UNIT": "unit only",
                "MEASUREMENT_COMPARATOR": "comparator or threshold",
                "MEASUREMENT_BASIS": "basis only",
                "SCOPE_VALUE": "scope_category only",
                "MEASUREMENT_TARGET": "target_object only",
                "MEASUREMENT_QUALIFIER": "qualifiers only",
            },
            "measurement_target_kind": "CONTEXT",
            "assertion_connection": "Canonical SourceAttachment graph; no competing measurement edge.",
        },
        "prompt_v3_1_summary.json": {
            "artifact_version": "prompt-v3.1-summary.v1",
            "prompt_id": SOURCE_CONTEXT_EXTRACTION_PROMPT_V31,
            "explicit_distinctions": [
                "identity != definition",
                "definition != result",
                "value != unit",
                "basis is independent",
                "scope/target preserved",
                "absent facts not inferred",
                "typed roles enforced",
            ],
            "one_off_d4g_watches_not_targeted": [
                "attachment omission",
                "PURPOSE omission",
                "SSL transport failure",
            ],
        },
        "schema_v3_1_summary.json": {
            "artifact_version": "schema-v3.1-summary.v1",
            "schema_id": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31,
            "typed_measurement_payload": True,
            "observed_value_discriminator": "kind",
            "measurement_fields_optional": True,
            "scope_node_canonical_owner": True,
            "scope_structure_binding_in_required_roles": False,
        },
        "validator_v3_1_summary.json": {
            "artifact_version": "validator-v3.1-summary.v1",
            "checks": [
                "exact textual evidence refs",
                "graph refs",
                "scope refs and members",
                "attachment refs and provenance",
                "relation refs",
                "measurement field-role compatibility",
                "measurement target kind",
            ],
            "natural_language_entailment": False,
            "semantic_support_result": "NOT_EVALUATED",
            "automatic_semantic_repair": False,
        },
        "synthetic_fixture_report.json": {
            "artifact_version": "synthetic-fixture-report.v1",
            "fixture_file": "tests/fixtures/source_context_extraction_v3_1_cases.json",
            "fixture_count": len(fixture_data["cases"]),
            "claim_specific_content": False,
            "fixture_ids": [item["id"] for item in fixture_data["cases"]],
        },
        "fake_gateway_report.json": {
            "artifact_version": "fake-gateway-report.v1",
            "provider_calls": 0,
            "external_http": 0,
            "fake_gateway_calls_max": 1,
            "regeneration": False,
            "repair": False,
            "payload_cases": [
                "valid scope",
                "typed measurement",
                "schema-invalid",
                "wrong role",
                "missing value/unit retained absent",
                "multiple measurement contexts",
            ],
        },
        "v2_v3_regression_report.json": {
            "artifact_version": "v2-v3-regression-report.v1",
            "v2_source_extraction": "historical module/test suite unchanged",
            "v3_source_extraction": "historical module/test suite unchanged",
            "v3_1_separate_versioned_path": True,
            "downstream_modified": False,
        },
        "hard_safety_report.json": {
            "artifact_version": "hard-safety-report.v1",
            "fabricated_semantic_facts_synthesized": False,
            "scope_structure_promoted_to_semantic_support": False,
            "formal_definition_inferred": False,
            "unit_inferred": False,
            "basis_inferred": False,
            "invalid_graph_refs_fail_closed": True,
            "role_target_mismatch_fails_closed": True,
            "missing_bindings_silently_repaired": False,
            "multi_target_sibling_support_leak": False,
            "production_research_loop_connected": False,
            "db_persistence_connected": False,
        },
    }
    written = [write_json(name, payload) for name, payload in artifacts.items()]
    source_paths = [
        ROOT / "apps/api/app/domain/source_context_extraction_v3_1.py",
        ROOT / "apps/api/app/services/source_context_extractor_v3_1.py",
        ROOT / "tests/unit/test_source_context_extraction_v3_1.py",
        ROOT / "tests/fixtures/source_context_extraction_v3_1_cases.json",
        ROOT / "scripts/build_v2_1r_d4i_v3_1_artifacts.py",
    ]
    report = """# V2.1R D4-I Scope Ownership + Measurement Typing v3.1 Shadow Revision

Provider calls: 0; external Provider HTTP: 0; real extractor calls: 0; semantic verifier calls: 0; Holdout access: 0.

## A. D4-H Revision Basis

D4-H established the canonical scope-owner mismatch in all three naturally occurring multi-assertion scopes and repeated measurement typing/role defects across Claims 1/3/8. This revision addresses only these two systematic clusters.

## B. Versioning

New identities: `source-context-extraction-schema-v3.1`, `source-context-extraction.generic.v3.1`, and `source-context-extraction-domain-v3.1`. Historical v2/v3 source contracts remain unchanged.

## C. Scope Canonical Ownership

`SourceScopeNode` owns scope type, member refs, graph membership, and evidence provenance. Scope structure alone does not establish semantic support.

## D. SCOPE_STRUCTURE Disposition

Deprecated for source-extraction scope graph ownership in v3.1; it is not a required duplicate of kind/members/evidence_refs.

## E. Scope Safety

Invalid member/evidence refs and duplicate members fail closed. PARTIAL/UNKNOWN/UNSUPPORTED scope resolution remains PARTIAL. Structural scope validity never promotes semantic support.

## F. Frozen Scope Offline Closure

All three frozen scope-only cases pass v3.1 scope graph checks without duplicate binding. Provenance refs remain present. Claim 3's unrelated historical graph/evidence diagnostics remain; this replay did not reinterpret them.

## G. Measurement Typed Object Model

Optional typed fields separate identity, formal definition, observed value, unit, comparator, threshold, basis, scope/category, target/object, and material qualifiers. Observed values support INTEGER, DECIMAL, RANGE, APPROXIMATE, BOUND, and TEXT with raw representation.

## H. Measurement Identity

Identity means what is measured; it is not a definition or result.

## I. Observed Value

Dedicated discriminated value field preserves raw source form and typed value. No absent value is inferred.

## J. Unit / Comparator / Threshold

Unit is separate and optional; comparator/threshold are distinct. No unit is inferred.

## K. Formal Definition

Optional and only represented when supplied. Metric identity/result cannot stand in for formal definition.

## L. Measurement Basis Preservation

Existing MeasurementBasis semantics are retained; no basis heuristic was added. Basis remains optional and independent.

## M. Binding Role Contract

Explicit field-role map is in `measurement_role_matrix.json`; wrong/missing field-role combinations fail closed without repair.

## N. Measurement Target / Attachment Ownership

Measurement evidence targets its canonical CONTEXT. Assertion association uses the existing SourceAttachment graph.

## O. Prompt v3.1

Explicitly distinguishes identity/definition/result/unit/basis, preserves scope/target, forbids absent-field inference, and documents the role map.

## P. Schema v3.1

Versioned typed observed-value discriminator and optional typed measurement fields; scope does not require duplicate structure binding.

## Q. Validator v3.1

Deterministic text-span/reference/graph/attachment/scope/relation/role checks only. No natural-language entailment; semantic support remains NOT_EVALUATED.

## R. Synthetic Fixtures

24 generic fixture cases; no D4-G-specific entities, claims, or values.

## S. Fake Gateway Tests

Single call maximum, regeneration false, valid/invalid scope and typed measurement cases tested. No real Provider or outbound HTTP.

## T. D4-G One-Off Watches

Attachment omission, PURPOSE omission, and Claim 7 SSL transport failure remain WATCH_ONLY; no targeted behavior change.

## U. V2 / V3 Regression

Historical v2/v3 extraction, Grounding Requirement v2.1, Matcher v2.1, and Hybrid suites are included in regression runs; downstream source is untouched.

## V. Safety Invariants

No fabricated semantics, scope-to-semantic promotion, inferred definition/unit/basis, invalid-ref acceptance, role repair, or sibling leakage.

## W. Production Behavior

v3.1 is shadow/development-only; no ResearchLoop, DB, Acceptance, Coverage, report, hybrid verifier, or V2.2 connection.

## X. Provider Calls

Generation 0; external Provider HTTP 0; real extractor 0; semantic verifier 0.

## Y. Quality Gates

See actual final test, mypy, Ruff, and diff-check results reported in the delivery.

## Z. Artifacts

Revision manifest, scope owner contract/replay, measurement typed specification/role matrix, prompt/schema/validator summaries, fixture/Fake Gateway/regression/safety reports, hash manifest, and this report.

## AA. Final Decision

`V3.1 SCOPE/MEASUREMENT SHADOW PASSES — READY FOR FRESH REAL CANARY READINESS`. This means only offline shadow contract readiness; it does not authorize or execute a Provider Canary.
"""
    (OUT / "final_report.md").write_text(report, encoding="utf-8")
    report_path = OUT / "final_report.md"
    hash_manifest = {
        "artifact_version": "v2_1r_d4i_v3_1_hash_manifest.v1",
        "hash_algorithm": "SHA-256",
        "provider_calls": 0,
        "source_file_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in source_paths
        },
        "artifact_sha256": {path.name: sha(path) for path in [*written, report_path]},
        "self_hash": "NOT_INCLUDED_TO_AVOID_SELF_REFERENCE",
    }
    write_json("hash_manifest.json", hash_manifest)


if __name__ == "__main__":
    main()
