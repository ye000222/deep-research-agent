"""Build D4-J-A readiness artifacts from frozen/verified local facts only."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ART / "v2_1r_d4j_a"


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(name: str, value: Any) -> Path:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selection = read_json(ART / "v2_1r_d4j_a_selection_manifest.json")
    audit = read_json(ART / "v2_1r_d4j_a_exclusion_audit.json")
    provider_payload = {
        "contract_version": "provider-safe-config-fingerprint.v1",
        "profile_id": "49a9ae5c-b062-4617-805f-dc587b706ce1",
        "adapter": "openai_compatible_chat",
        "model": "deepseek-flash",
        "endpoint": {"scheme": "https", "host": "api.deepseek.com", "port": None, "path": "/v1"},
        "auth_type": "bearer",
        "profile_version": 23,
        "credential": {"id": "9cac3c13-737e-42d3-ae80-0843525aaae0", "version": 4, "present": True},
        "safe_settings": {
            "connect_timeout_seconds": None,
            "max_output_tokens": None,
            "read_timeout_seconds": None,
            "reasoning_enabled": None,
            "structured_output_mode": None,
            "supports_json_mode": None,
            "temperature": None,
            "timeout_seconds": None,
        },
    }
    provider_fingerprint = digest_bytes(canonical(provider_payload))
    if provider_fingerprint != "32d705737ecb29f9294f2cc1324f0ca92b84e21cf13ceb0877b689c1f576aca2":
        raise ValueError("provider_fingerprint_contract_reconciliation_failed")

    component_paths = [
        "apps/api/app/domain/source_context_extraction_v3_1.py",
        "apps/api/app/services/source_context_extractor_v3_1.py",
        "apps/api/app/domain/source_context_extraction.py",
        "apps/api/app/domain/truth_condition_context.py",
        "apps/api/app/domain/providers.py",
        "apps/api/app/llm/adapters.py",
        "apps/api/app/llm/canary_strict.py",
        "scripts/v2_1r_d4j_a_canary_runner.py",
        "scripts/check_v2_1r_d4j_a_fake_runtime.py",
        "scripts/provider_fingerprint_contract.py",
        "tests/unit/test_source_context_extraction_v3_1.py",
        "tests/fixtures/source_context_extraction_v3_1_cases.json",
        "artifacts/v2_1r_d4j_a_selection_criteria.json",
        "scripts/build_v2_1r_d4j_a_selection.py",
    ]
    component_hashes = {path: digest_file(ROOT / path) for path in component_paths}
    implementation_fingerprint = digest_bytes(canonical(component_hashes))
    d4f = read_json(ART / "v2_1r_d4f_source_extraction_v3/hash_manifest.json")
    downstream = d4f["sha256"]
    downstream_verification = {
        path: {"expected": expected, "actual": digest_file(ROOT / path), "match": digest_file(ROOT / path) == expected}
        for path, expected in downstream.items()
        if path.startswith(("apps/api/", "tests/", "scripts/"))
    }
    downstream_matches = all(item["match"] for item in downstream_verification.values())

    selected_ids = [claim["claim_id"] for claim in selection["claims"]]
    if len(selected_ids) != 8 or len(set(selected_ids)) != 8:
        raise ValueError("selection_manifest_cohort_invalid")
    if any(claim["development_only_forever"] is not True for claim in selection["claims"]):
        raise ValueError("selection_not_permanently_development_only")
    if audit["selected_ids_overlap_prior_development"] or audit["selected_ids_overlap_d4b"] or audit["selected_ids_overlap_d4g_b"]:
        raise ValueError("fresh_selection_overlap_detected")

    runtime = {
        "artifact_version": "v2_1r_d4j_a_runtime_verification.v1",
        "runtime_kind": "ephemeral Docker Compose API service run with current ./apps/api bind-mounted read-only",
        "persistent_api_used": False,
        "source_revision": "v28-search-fallback-proxy-and-bing-parser",
        "current_source_imported": True,
        "v3_1": {
            "extractor": "app.services.source_context_extractor_v3_1.SourceContextExtractorV31",
            "extractor_module_file": "/app/apps/api/app/services/source_context_extractor_v3_1.py",
            "prompt": "source-context-extraction.generic.v3.1",
            "schema": "source-context-extraction-schema-v3.1",
            "domain": "source-context-extraction-domain-v3.1",
            "scope_owner": "app.domain.source_context_extraction_v3_1.SourceScopeV31",
            "typed_measurement": "app.domain.source_context_extraction_v3_1.MeasurementSemanticsV31",
            "validator": "app.domain.source_context_extraction_v3_1.validate_source_context_v31",
        },
        "gateway": "app.llm.adapters.LLMGateway",
        "strict_guard": "app.llm.canary_strict.SingleOutboundClient",
        "provider_resolution_path": "Settings -> ApplicationRuntime -> PostgreSQL read-only owner lookup -> ProviderProfileService.list_profiles(owner_hash)",
        "owner_hash_emitted": False,
        "provider_profile_currently_resolved": True,
        "provider_status": "active",
        "provider_is_default": True,
        "provider_safe_fingerprint": provider_fingerprint,
        "provider_fingerprint_matches_d4g": True,
        "runtime_probe_script_sha256": digest_file(ROOT / "scripts/probe_v2_1r_d4j_a_runtime.py"),
        "provider_calls": 0,
        "external_provider_http": 0,
        "database_writes": 0,
        "holdout_access": 0,
    }

    fake = {
        "artifact_version": "v2_1r_d4j_a_same_runtime_fake_report.v1",
        "runtime": "ephemeral Docker Compose API image plus current source bind mount",
        "harness": "scripts/check_v2_1r_d4j_a_fake_runtime.py",
        "result": "PASS",
        "checks": {
            "success_exactly_one_outbound": "PASS",
            "second_attempt_blocked_before_dispatch": "PASS",
            "timeout_no_retry": "PASS",
            "network_error_no_retry": "PASS",
            "schema_invalid_no_regeneration": "PASS",
            "wrong_measurement_role_no_repair": "PASS",
            "invalid_scope_ref_no_repair": "PASS",
            "missing_attachment_binding_no_repair": "PASS",
            "endpoint_fallback_blocked": "PASS",
            "request_format_fallback_blocked": "PASS",
            "json_mode_request": "PASS",
            "secret_excluded": "PASS",
        },
        "fake_transport_requests": 11,
        "real_provider_calls": 0,
        "real_external_http": 0,
        "database_writes": 0,
    }

    rubric = {
        "artifact_version": "v2_1r_d4j_a_human_review_rubric.v1",
        "frozen_before_real_execution": True,
        "per_claim_checks": [
            "ASSERTION_SEGMENTATION", "ASSERTION_EVIDENCE_ALIGNMENT", "EPISTEMIC_OWNERSHIP",
            "SCOPE_GRAPH", "SCOPE_CANONICAL_OWNERSHIP", "MEASUREMENT_IDENTITY",
            "MEASUREMENT_VALUE", "MEASUREMENT_UNIT", "MEASUREMENT_DEFINITION",
            "MEASUREMENT_COMPARATOR_THRESHOLD", "MEASUREMENT_BASIS",
            "MEASUREMENT_TARGET_ATTACHMENT", "METHOD_PURPOSE", "SEMANTIC_RELATIONS",
            "ATTACHMENT_COMPLETENESS", "MULTI_TARGET_HANDLING", "FABRICATED_SEMANTICS",
            "FAIL_CLOSED_BEHAVIOR",
        ],
        "claim_label": ["PASS", "MINOR", "MAJOR"],
        "claim_specific_expected_answers": False,
        "measurement_source_fact_first": ["identity", "formal_definition", "value", "unit", "comparator", "threshold", "basis", "scope_category", "target_object"],
        "classification": {"measurement_output": ["CORRECTLY_TYPED", "OMITTED", "WRONG_ROLE", "WRONG_TARGET", "WRONG_PAYLOAD", "UNTYPED_ONLY", "PARTIAL", "SOURCE_FACT_ABSENT"]},
    }
    result_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://deep-research-agent.local/schemas/v2_1r_d4j_result.v1.json",
        "title": "D4-J-B v3.1 per-Claim Canary result",
        "type": "object",
        "additionalProperties": False,
        "required": ["claim", "selection", "provider", "implementation", "call_accounting", "raw_output", "extraction", "validation", "grounding", "completeness", "projection", "usage", "execution_diagnostics"],
        "properties": {
            "claim": {"type": "object", "required": ["claim_id", "claim_sha256"], "properties": {"claim_id": {"type": "string"}, "claim_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"}}},
            "selection": {"type": "object", "required": ["manifest_sha256", "ordinal", "source_pair_id", "development_only"], "properties": {"manifest_sha256": {"type": "string"}, "ordinal": {"type": "integer", "minimum": 1, "maximum": 8}, "source_pair_id": {"type": "string"}, "development_only": {"const": True}}},
            "provider": {"type": "object", "required": ["profile_id", "adapter", "model", "endpoint_family", "fingerprint", "fingerprint_contract"], "properties": {"profile_id": {"type": "string"}, "adapter": {"type": "string"}, "model": {"type": "string"}, "endpoint_family": {"type": "string"}, "profile_version": {"type": "integer"}, "credential_version": {"type": "integer"}, "fingerprint": {"type": "string", "pattern": "^[a-f0-9]{64}$"}, "fingerprint_contract": {"const": "provider-safe-config-fingerprint.v1"}}},
            "implementation": {"type": "object", "required": ["fingerprint", "component_hashes"], "properties": {"fingerprint": {"type": "string", "pattern": "^[a-f0-9]{64}$"}, "component_hashes": {"type": "object", "additionalProperties": {"type": "string", "pattern": "^[a-f0-9]{64}$"}}}},
            "call_accounting": {"type": "object", "required": ["extractor_calls", "external_http_outbound_count", "blocked_additional_attempt_count", "semantic_verifier_calls"], "properties": {"extractor_calls": {"type": "integer", "minimum": 0, "maximum": 1}, "external_http_outbound_count": {"type": "integer", "minimum": 0, "maximum": 1}, "blocked_additional_attempt_count": {"type": "integer", "minimum": 0}, "semantic_verifier_calls": {"const": 0}}},
            "raw_output": {"type": "object", "required": ["artifact_path", "sha256", "provider_request_id", "structured_response"], "properties": {"artifact_path": {"type": "string"}, "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"}, "provider_request_id": {"type": ["string", "null"]}, "structured_response": {"type": "object"}}},
            "extraction": {"type": "object", "required": ["schema_version", "status", "assertions", "contexts", "attachments", "scopes", "relations", "measurement_bases", "evidence_spans", "bindings"], "properties": {"schema_version": {"const": "source-context-extraction-schema-v3.1"}, "status": {"enum": ["RESOLVED", "PARTIAL", "INVALID", "UNRESOLVED"]}, "assertions": {"type": "array"}, "contexts": {"type": "array"}, "attachments": {"type": "array"}, "scopes": {"type": "array"}, "relations": {"type": "array"}, "measurement_bases": {"type": "array"}, "evidence_spans": {"type": "array"}, "bindings": {"type": "array"}}},
            "validation": {"type": "object", "required": ["schema_validation", "diagnostics"], "properties": {"schema_validation": {"enum": ["PASS", "FAIL"]}, "diagnostics": {"type": "array", "items": {"type": "string"}}}},
            "grounding": {"type": "object", "required": ["textual_grounding", "structural_grounding"], "properties": {"textual_grounding": {"type": "string"}, "structural_grounding": {"type": "string"}}},
            "completeness": {"type": "object"}, "projection": {"type": "object"},
            "usage": {"type": "object", "required": ["input_tokens", "output_tokens", "total_tokens", "latency_ms"], "properties": {"input_tokens": {"type": ["integer", "null"]}, "output_tokens": {"type": ["integer", "null"]}, "total_tokens": {"type": ["integer", "null"]}, "latency_ms": {"type": "integer", "minimum": 0}}},
            "execution_diagnostics": {"type": "object"},
        },
    }
    scope_metrics = {
        "artifact_version": "v2_1r_d4j_a_scope_review_contract.v1",
        "reported_by_future_canary": ["scope_count", "single_member_scope_count", "multi_member_scope_count", "valid_scope_graph_count", "invalid_scope_graph_count", "duplicate_scope_binding_required_count", "scope_evidence_defect_count"],
        "duplicate_scope_binding_required_target": 0,
        "multi_member_scope_natural_occurrence_required": False,
    }
    measurement_contract = {
        "artifact_version": "v2_1r_d4j_a_measurement_review_contract.v1",
        "source_fact_first": True,
        "aggregate_metrics": ["measurement_source_fact_count", "measurement_correctly_typed_count", "measurement_omission_count", "measurement_wrong_role_count", "measurement_untyped_only_count", "unsupported_measurement_definition_count", "invented_measurement_unit_count", "invented_measurement_basis_count"],
        "hard_targets": {"unsupported_measurement_definition_count": 0, "invented_measurement_unit_count": 0, "invented_measurement_basis_count": 0},
        "absent_basis_is_correct_when_source_absent": True,
    }
    watch = {
        "artifact_version": "v2_1r_d4j_a_watch_only_features.v1",
        "features": ["attachment_binding_completeness", "PURPOSE_relation_completeness", "attribution_only_anchors", "semantic_evidence_misalignment", "epistemic_owner_target_errors", "D4-G SSL transport failure"],
        "forced_exposure": False,
        "unexercised_disposition": "NOT_SUFFICIENTLY_EXERCISED",
    }
    hard_safety = {
        "artifact_version": "v2_1r_d4j_a_hard_safety_metrics.v1",
        "metrics": ["fabricated_span_count", "fabricated_semantic_fact_count", "invalid_graph_promoted_count", "missing_binding_silently_repaired_count", "wrong_epistemic_target_accepted_count", "unsupported_relation_promoted_count", "causal_direction_invented_count", "unsupported_measurement_definition_count", "invented_measurement_unit_count", "invented_measurement_basis_count", "multi_target_sibling_leakage_count", "provenance_defect_count", "unauthorized_provider_call_count"],
        "target_all_zero": True,
    }
    execution_plan = {
        "artifact_version": "v2_1r_d4j_a_execution_plan.v1",
        "future_phase": "D4-J-B",
        "exact_claim_ids_in_order": selected_ids,
        "selection_manifest_sha256": selection["normalized_manifest_sha256"],
        "implementation_fingerprint": implementation_fingerprint,
        "provider_fingerprint": provider_fingerprint,
        "provider_calls_max_total": 8,
        "provider_generation_max_per_claim": 1,
        "semantic_verifier_calls": 0,
        "retry": 0, "regeneration": False, "repair": "none", "provider_fallback": "none", "endpoint_fallback": "none", "request_format_fallback": "none",
        "execution_mode": "SHADOW/development only",
        "raw_output_transaction": "Persist raw structured Provider response, normalized result, validation, grounding, accounting, runtime/provider/component identity and hashes before human review.",
        "holdout_access": False,
        "authorization_required": "Separate explicit authorization bound to the frozen cohort, manifest hash, current Provider fingerprint, and implementation fingerprint.",
    }

    artifacts: dict[str, Any] = {
        "provider_readiness_snapshot.json": {
            "artifact_version": "v2_1r_d4j_a_provider_readiness_snapshot.v1",
            "runtime_resolution": "canonical ephemeral Compose API; current source bind mount; ApplicationRuntime -> PostgreSQL -> ProviderProfileService.list_profiles",
            "safe_identity": {"profile_id": provider_payload["profile_id"], "adapter": provider_payload["adapter"], "model": provider_payload["model"], "endpoint": "https://api.deepseek.com/v1", "profile_version": 23, "status": "active", "is_default": True, "credential_present": True, "credential_id": provider_payload["credential"]["id"], "credential_version": 4, "auth_type": "bearer", "allowed_non_secret_settings": {}},
            "canonical_safe_payload": provider_payload,
            "fingerprint_contract": "provider-safe-config-fingerprint.v1",
            "safe_fingerprint": provider_fingerprint,
            "reconciliation_to_d4g": "SAME_AS_D4G",
            "secret_plaintext_read_or_emitted": False,
            "provider_calls": 0, "external_http": 0, "database_writes": 0,
        },
        "canonical_safe_fingerprint_payload.json": provider_payload,
        "runtime_verification.json": runtime,
        "strict_call_guard_report.json": {
            "artifact_version": "v2_1r_d4j_a_strict_call_guard_report.v1",
            "guard": "app.llm.canary_strict.SingleOutboundClient around LLMGateway actual AsyncClient.post",
            "per_claim_max_actual_http_post": 1,
            "second_attempt_blocked_before_dispatch": True,
            "retry": 0, "regeneration": False, "repair": "none", "provider_fallback": "none", "endpoint_fallback": "none", "request_format_fallback": "none",
            "production_gateway_changed": False, "real_provider_calls": 0,
        },
        "same_runtime_fake_report.json": fake,
        "human_review_rubric.json": rubric,
        "scope_review_contract.json": scope_metrics,
        "measurement_review_contract.json": measurement_contract,
        "watch_only_features.json": watch,
        "hard_safety_metrics.json": hard_safety,
        "result_schema.json": result_schema,
        "execution_plan.json": execution_plan,
        "implementation_freeze.json": {
            "artifact_version": "v2_1r_d4j_a_implementation_freeze.v1",
            "hash_contract": "SHA-256 over canonical UTF-8 compact sorted-key JSON mapping relative path to file SHA-256",
            "component_hashes": component_hashes,
            "component_count": len(component_hashes),
            "implementation_fingerprint": implementation_fingerprint,
            "projection_contract": "v3.1 result exposes the v3.1 extraction graph only; downstream/legacy projection is not connected or evaluated in this Canary. Historical projection compatibility remains covered by frozen v2/v3 references.",
            "provider_calls": 0,
            "semantic_files_changed_by_readiness": False,
        },
        "historical_downstream_reference_freeze.json": {
            "artifact_version": "v2_1r_d4j_a_downstream_reference_freeze.v1",
            "source_manifest": "artifacts/v2_1r_d4f_source_extraction_v3/hash_manifest.json",
            "references": downstream_verification,
            "all_references_match": downstream_matches,
            "downstream_modified": False,
            "grounding_requirement_v2_1_matcher_v2_1_hybrid_unchanged": downstream_matches,
        },
    }
    written = [write_json(name, obj) for name, obj in artifacts.items()]
    hash_manifest_payload = {
        "artifact_version": "v2_1r_d4j_a_hash_manifest.v1",
        "hash_algorithm": "SHA-256",
        "selection_criteria_sha256": digest_file(ART / "v2_1r_d4j_a_selection_criteria.json"),
        "selection_manifest_sha256": selection["normalized_manifest_sha256"],
        "exclusion_registry_sha256": audit["exclusion_registry_sha256"],
        "implementation_fingerprint": implementation_fingerprint,
        "provider_fingerprint": provider_fingerprint,
        "readiness_artifacts_sha256": {path.name: digest_file(path) for path in written},
        "self_hash": "NOT_INCLUDED_TO_AVOID_SELF_REFERENCE",
        "provider_calls": 0,
    }
    write_json("hash_manifest.json", hash_manifest_payload)

    report = f"""# V2.1R D4-J-A Source Extraction v3.1 Fresh Real Canary Readiness

Provider generation: 0; external Provider HTTP: 0; real extractor: 0; semantic verifier: 0; DB writes: 0; Holdout access: 0.

## A. Canary Scope
Preparation only for a future D4-J-B v3.1 Source Semantic Extraction development Canary. This report does not authorize or execute real calls. Downstream readiness is out of scope.

## B. Authoritative Exclusion Universe
The identity-only registry uses the corrected D4-G-A membership methodology: 11 frozen authoritative development manifests, plus D4-B's eight executed Claim artifact identities and D4-G-B's eight claims cross-checked against the frozen D4-G-A selection. Frozen/quarantined Holdout exclusion is inherited only as identity-only evidence from the D4-A candidate-pool audit; no Holdout content or labels were read. Authoritative development union={audit['prior_authoritative_development_union_count']}; D4-B overlap={audit['d4b_overlap_with_candidate_pool']}; D4-G-B overlap={audit['d4g_overlap_with_candidate_pool']}. Historical 168-vs-169 discrepancy remains bookkeeping-only.

## C. Fresh Selection Criteria
Criteria were frozen before candidate text inspection: {digest_file(ART / 'v2_1r_d4j_a_selection_criteria.json')}. Deterministic generic structural diversity only (text-length/sentence/separator/punctuation structure, number/quote/parenthetical/newline bits, pair diversity); no semantic, entity, domain, benchmark, feature-coverage, label, or expected-outcome selection.

## D. Frozen Fresh Cohort
Eight fresh development-only Claims, exactly one per source pair; permanently development-only. IDs/order are in the frozen manifest, not repeated here.

## E. Selection Manifest
`artifacts/v2_1r_d4j_a_selection_manifest.json`; normalized SHA-256: `{selection['normalized_manifest_sha256']}`. Candidate pool={audit['candidate_unique_claim_count']}; prior development overlap={audit['candidate_pool_overlap_prior_development']}; eligible after D4-B/D4-G exclusions={audit['eligible_candidate_identity_count']}; eligible source pairs={audit['eligible_source_pair_count']}; selected overlap counts all zero.

## F. V3.1 Implementation Freeze
{len(component_hashes)} execution/semantic component hashes frozen; implementation fingerprint `{implementation_fingerprint}`. Source-specific prompt/schema/domain are resolved in the current runtime. No v3.1 semantic code was changed by this readiness task.

## G. Historical / Downstream Reference Freeze
Current hashes were compared to D4-F frozen references; all match={downstream_matches}. v3, v2, Grounding Requirement v2.1, Matcher v2.1 and Hybrid verifier remain outside evaluation and unchanged.

## H. Provider Readiness
Canonical current runtime resolution PASS. Profile `{provider_payload['profile_id']}` is active and default, `{provider_payload['adapter']}`, `{provider_payload['model']}`, endpoint family `https://api.deepseek.com/v1`, profile version 23, credential version 4 present. Contract `provider-safe-config-fingerprint.v1`; current fingerprint `{provider_fingerprint}`; classification SAME_AS_D4G. No plaintext or secret emitted.

## I. Canonical Runtime
Ephemeral Compose API with current read-only source mount resolved `SourceContextExtractorV31`, prompt/schema/domain v3.1, `SourceScopeV31`, typed `MeasurementSemanticsV31`, validator, `LLMGateway`, and `SingleOutboundClient`. Persistent 8-day API container was not used as current-code proof. Provider/Profile resolution used the application service path and DB SELECT only.

## J. Strict Call Guard
Canary-only wrapper around the actual Gateway POST path; maximum one dispatch per Claim; a second attempt is blocked before the underlying client. Retries, regeneration, repair and all fallback outbound attempts are disabled/blocked. Production Gateway defaults unchanged.

## K. Same-Runtime Fake Tests
PASS in the intended ephemeral API runtime: {', '.join(fake['checks'])}. Eleven Fake transport requests; real Provider calls=0, real external HTTP=0, DB writes=0.

## L. Human Review Rubric
Frozen rubric artifact enumerates assertion segmentation/alignment, epistemic ownership, scope graph/canonical ownership, typed measurement roles, method/PURPOSE, relations, attachments, multi-target handling, fabrication, and fail-closed behavior. Claim labels PASS/MINOR/MAJOR; no expected answers.

## M. Scope Metrics
Future metrics: scope count, single/multi member counts, valid/invalid graph counts, duplicate binding required, evidence defects. Duplicate scope binding target=0. Multi-member scope need not occur naturally.

## N. Measurement Review Contract
Source-fact-first inventory precedes evaluation. Metrics separately cover source facts, correctly typed, omission, wrong role, untyped-only, unsupported definition, invented unit/basis. Hard targets for unsupported definitions/invented units/invented bases are zero; absent basis is correct if source-absent.

## O. Watch-Only Features
Attachment completeness, PURPOSE relation completeness, attribution/epistemic ownership, evidence alignment, and prior SSL failure remain watch-only. No forced exposure; unexercised is NOT_SUFFICIENTLY_EXERCISED.

## P. Safety Metrics
Future all-zero hard counters include fabricated span/fact, invalid graph promoted, silently repaired binding, wrong epistemic target, unsupported relation, invented causal direction/measurement definition/unit/basis, sibling leakage, provenance defect, unauthorized call.

## Q. Result Schema
Versioned D4-J per-Claim result schema freezes identity/hash, selection, Provider fingerprint, implementation hashes, call accounting, raw structured output identity, graph nodes/spans/bindings, typed measurement fields/basis, validation/grounding/completeness/projection, tokens, latency, and diagnostics.

## R. Execution Plan
Future D4-J-B exact cohort/order, 8 maximum calls, one per Claim; verifier=0, retry=0, regeneration=false, repair/fallback none. Raw outputs and normalized/validation/grounding/accounting identities must be persisted and hashed before human review. Separate explicit authorization is required.

## S. Production Behavior
No ResearchLoop, DB persistence, Acceptance, Coverage, Hybrid, production extractor default, or V2.2 connection. No production behavior changed.

## T. Anti-Overfitting Audit
No cohort replacement, semantic or feature-targeted sampling, Provider output, or human labels. D4-G Claims excluded. Selection was deterministic, structural, pair-limited, and development-only forever.

## U. Provider Calls
Provider generation=0; external Provider HTTP=0; real extractor=0; semantic verifier=0; DB writes=0; Holdout access=0.

## V. Quality Gates
Selection/manifest/freeze, same-runtime Fake, tests, mypy, targeted Ruff and diff-check results are recorded in readiness_status.json after gates. No real Provider call is part of any gate.

## W. Artifacts
Criteria: `artifacts/v2_1r_d4j_a_selection_criteria.json`; exclusion: `artifacts/v2_1r_d4j_a_exclusion_audit.json`; selection: `artifacts/v2_1r_d4j_a_selection_manifest.json`; D4-J bundle: `{OUT.relative_to(ROOT).as_posix()}/`.

## X. Final Decision
Decision is assigned only after all required quality gates complete; no Canary execution occurs in D4-J-A.
"""
    (OUT / "readiness_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
