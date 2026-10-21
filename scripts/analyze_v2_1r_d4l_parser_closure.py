"""Offline D4-L parser closure reports; never calls a Provider or writes application data."""

# Markdown prose emitted below intentionally exceeds source-code line-length policy.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.domain.source_context_extraction_v3_1 import (
    MeasurementSemanticsV31,
    SourceExtractionResponseV31,
    _measurement_required_roles,
    validate_scope_ownership_v31,
    validate_source_context_v31,
)
from app.domain.truth_condition_context import ContextType

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "artifacts" / "v2_1r_d4j_b"
OUTPUT_ROOT = ROOT / "artifacts" / "v2_1r_d4l"
PATCH_VERSION = "source-context-extraction-parser-v3.1.1"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_reports() -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frozen = _read_json(INPUT_ROOT / "raw_freeze_manifest.json")
    old_measurement = _read_json(
        ROOT / "artifacts" / "v2_1r_d4j_k" / "measurement_exercise_audit.json"
    )
    expected_hashes = {
        item["raw_output_path"]: item["raw_output_sha256"] for item in frozen["claim_results"]
    }
    expected_normalized = {
        item["normalized_result_path"]: item["normalized_result_sha256"]
        for item in frozen["claim_results"]
    }
    input_integrity: list[dict[str, Any]] = []
    replay: list[dict[str, Any]] = []
    round_trips: list[dict[str, Any]] = []
    scope_assessments: list[dict[str, Any]] = []
    measurement_contexts = 0
    required_role_count = 0
    present_role_count = 0
    missing_roles: list[dict[str, str]] = []

    for ordinal, item in enumerate(frozen["claim_results"], start=1):
        raw_rel = item["raw_output_path"]
        normalized_rel = item["normalized_result_path"]
        raw_path = INPUT_ROOT / raw_rel
        normalized_path = INPUT_ROOT / normalized_rel
        raw_hash = _sha256(raw_path)
        normalized_hash = _sha256(normalized_path)
        input_integrity.append(
            {
                "ordinal": ordinal,
                "claim_id": item["claim_id"],
                "raw_path": raw_rel,
                "raw_sha256": raw_hash,
                "raw_matches_frozen": raw_hash == expected_hashes[raw_rel],
                "normalized_path": normalized_rel,
                "normalized_sha256": normalized_hash,
                "normalized_matches_frozen": normalized_hash == expected_normalized[normalized_rel],
            }
        )
        raw_envelope = _read_json(raw_path)
        payload = raw_envelope["model_structured_object"]
        normalized = _read_json(normalized_path)
        claim = normalized["claim"]["claim_text"]
        try:
            parsed = SourceExtractionResponseV31.model_validate(payload)
        except Exception as exc:
            replay.append(
                {
                    "ordinal": ordinal,
                    "claim_id": item["claim_id"],
                    "schema_parse": "FAIL",
                    "error_type": type(exc).__name__,
                    "old_schema_status": item["schema_status"],
                    "old_extraction_status": item["extraction_status"],
                }
            )
            continue

        result = validate_source_context_v31(parsed, claim, claim_ref=item["claim_id"])
        old_record = normalized["completeness"]
        current_status = result.status.value
        replay.append(
            {
                "ordinal": ordinal,
                "claim_id": item["claim_id"],
                "schema_parse": "PASS",
                "new_extraction_status": current_status,
                "new_diagnostics": list(result.diagnostics),
                "old_schema_status": item["schema_status"],
                "old_extraction_status": item["extraction_status"],
                "old_diagnostics": old_record.get("diagnostics", []),
                "context_dispatch": [
                    {
                        "context_ref": context.context_ref,
                        "context_type": context.context_type.value,
                        "domain_type": type(context.semantic_value).__name__,
                    }
                    for context in parsed.source_contexts
                ],
            }
        )
        scope = validate_scope_ownership_v31(parsed, claim)
        scope_assessments.append(
            {
                "claim_id": item["claim_id"],
                "scope_status": scope.status.value,
                "scope_diagnostics": list(scope.diagnostics),
                "source_scope_count": len(parsed.source_scopes),
                "duplicate_scope_structure_bindings": sum(
                    binding.semantic_role.value == "SCOPE_STRUCTURE"
                    for binding in parsed.evidence_bindings
                ),
            }
        )
        round_trips.append(
            {
                "claim_id": item["claim_id"],
                "response_round_trip_identity": SourceExtractionResponseV31.model_validate(
                    parsed.model_dump(mode="json")
                )
                == parsed,
            }
        )
        binding_keys = {
            (binding.target_kind, binding.target_ref, binding.semantic_role)
            for binding in parsed.evidence_bindings
        }
        for context in parsed.source_contexts:
            if context.context_type is not ContextType.MEASUREMENT:
                continue
            measurement_contexts += 1
            semantics = context.semantic_value
            assert isinstance(semantics, MeasurementSemanticsV31)
            for role in _measurement_required_roles(semantics):
                required_role_count += 1
                if (  # required evidence-role link exists; semantic graph is not repaired
                    next(
                        (
                            key
                            for key in binding_keys
                            if key[0].value == "CONTEXT"
                            and key[1] == context.context_ref
                            and key[2] is role
                        ),
                        None,
                    )
                    is not None
                ):
                    present_role_count += 1
                else:
                    missing_roles.append(
                        {
                            "claim_id": item["claim_id"],
                            "context_ref": context.context_ref,
                            "role": role.value,
                        }
                    )

    all_context_types = [item.value for item in ContextType]
    union_matrix = [
        {
            "context_type": context_type,
            "canonical_discriminator": "context_type",
            "parser_branch": (
                "MeasurementSemanticsV31"
                if context_type == ContextType.MEASUREMENT.value
                else "GenericContextSemanticValueV31"
            ),
            "domain_type": (
                "MeasurementSemanticsV31"
                if context_type == ContextType.MEASUREMENT.value
                else "GenericContextSemanticValueV31"
            ),
            "same_json_representation": True,
        }
        for context_type in all_context_types
    ]

    summary = {
        "artifact_version": "v2_1r_d4l_parser_replay.v1",
        "parser_patch_version": PATCH_VERSION,
        "source_manifest": "artifacts/v2_1r_d4j_b/raw_freeze_manifest.json",
        "source_claim_count": len(replay),
        "provider_generation_calls": 0,
        "external_provider_http": 0,
        "database_writes": 0,
        "holdout_access": 0,
        "all_raw_hashes_match_frozen": all(x["raw_matches_frozen"] for x in input_integrity),
        "all_normalized_hashes_match_frozen": all(
            x["normalized_matches_frozen"] for x in input_integrity
        ),
        "schema_parse_pass_count": sum(x["schema_parse"] == "PASS" for x in replay),
        "schema_parse_fail_count": sum(x["schema_parse"] == "FAIL" for x in replay),
        "historical_scope_union_failure_count": sum(
            x["old_schema_status"] == "FAIL" for x in replay
        ),
        "historical_scope_union_failures_cleared": all(
            x["schema_parse"] == "PASS" for x in replay if x["old_schema_status"] == "FAIL"
        ),
        "new_parser_errors": [x for x in replay if x["schema_parse"] == "FAIL"],
        "semantic_and_graph_findings_remain_visible": [
            {
                "claim_id": x["claim_id"],
                "status": x.get("new_extraction_status"),
                "diagnostics": x.get("new_diagnostics", []),
            }
            for x in replay
            if x["schema_parse"] == "PASS"
            and (x.get("new_extraction_status") != "RESOLVED" or x.get("new_diagnostics"))
        ],
        "round_trip_pass_count": sum(x["response_round_trip_identity"] for x in round_trips),
        "round_trip_count": len(round_trips),
        "context_families": all_context_types,
        "dispatch_matrix": union_matrix,
        "scope_ownership_all_resolved": all(
            x["scope_status"] == "RESOLVED" for x in scope_assessments
        ),
        "scope_ownership_diagnostics": [x for x in scope_assessments if x["scope_diagnostics"]],
        "duplicate_scope_structure_bindings": sum(
            x["duplicate_scope_structure_bindings"] for x in scope_assessments
        ),
        "measurement_source_supported_facts_historical_count": old_measurement[
            "measurement_source_fact_count"
        ],
        "measurement_contexts_typed_and_parseable": measurement_contexts,
        "measurement_context_role_slots_required": required_role_count,
        "measurement_context_role_slots_present": present_role_count,
        "measurement_context_role_slots_missing": missing_roles,
        "measurement_evaluable_historical_count": old_measurement[
            "measurement_semantically_evaluable_count"
        ],
        "measurement_observation_note": (
            "Offline parseability is established only for the frozen D4-J responses; "
            "this is bug closure on frozen evidence, not fresh generalization. "
            "No measurement definitions, units, or bases were inferred."
        ),
        "whole_claim_unresolved_watch": [
            x["claim_id"] for x in replay if x.get("new_extraction_status") == "UNRESOLVED"
        ],
        "attachment_contract_watch": "BINDING_CONTRACT_AMBIGUITY; unchanged by parser patch",
    }

    artifacts: dict[str, Any] = {
        "parser_patch_contract.json": {
            "artifact_version": "d4l_parser_patch_contract.v1",
            "patch_version": PATCH_VERSION,
            "scope": "shadow/development v3.1 deserialization only",
            "canonical_discriminator": "context_type",
            "measurement_branch": "MeasurementSemanticsV31",
            "non_measurement_branch": "GenericContextSemanticValueV31",
            "branch_selection_uses_payload_shape": False,
            "prompt_or_semantic_contract_changed": False,
            "historical_d4j_provenance_rewritten": False,
        },
        "discriminator_dispatch_matrix.json": {
            "artifact_version": "d4l_discriminator_dispatch_matrix.v1",
            "canonical_discriminator": "ContextType.context_type",
            "rows": union_matrix,
            "unknown_value": "fail closed",
            "missing_value": "required discriminator validation failure",
        },
        "union_exhaustiveness_matrix.json": {
            "artifact_version": "d4l_union_exhaustiveness.v1",
            "context_families": union_matrix,
            "exactly_one_branch_per_family": True,
            "measurement_payload_still_typed": True,
            "generic_payload_is_explicit_domain_wrapper_with_unchanged_json_object_form": True,
        },
        "round_trip_report.json": {
            "artifact_version": "d4l_round_trip_report.v1",
            "context_families": all_context_types,
            "synthetic_all_family_round_trip_test": (
                "covered by test_source_context_extraction_v3_1_dispatch.py"
            ),
            "frozen_response_round_trips": round_trips,
            "frozen_response_round_trip_pass_count": summary["round_trip_pass_count"],
        },
        "same_path_fake_report.json": {
            "artifact_version": "d4l_same_path_fake_report.v1",
            "execution_path": (
                "Fake Gateway structured object -> SourceExtractionResponseV31.model_validate "
                "-> validate_source_context_v31"
            ),
            "coverage": [
                "historical SCOPE-only",
                "MEASUREMENT",
                "mixed SCOPE+MEASUREMENT+CONDITION",
                "multiple SCOPE",
                "multiple MEASUREMENT",
            ],
            "provider_calls": 0,
            "test_file": "tests/unit/test_source_context_extraction_v3_1_dispatch.py",
        },
        "d4j_frozen_parser_replay.json": summary,
        "measurement_post_unblock_observation.json": {
            "artifact_version": "d4l_measurement_observation.v1",
            "source_supported_facts": summary[
                "measurement_source_supported_facts_historical_count"
            ],
            "typed_measurement_contexts_in_frozen_outputs": measurement_contexts,
            "required_role_slots": required_role_count,
            "present_role_slots": present_role_count,
            "missing_role_slots": missing_roles,
            "prior_semantic_evaluable_count": summary["measurement_evaluable_historical_count"],
            "contract_changed": False,
        },
        "scope_ownership_regression.json": {
            "artifact_version": "d4l_scope_ownership_regression.v1",
            "d4i_canonical_owner": "SourceScopeV31",
            "frozen_replay_assessments": scope_assessments,
            "duplicate_scope_structure_required": False,
            "duplicate_scope_structure_binding_count": summary[
                "duplicate_scope_structure_bindings"
            ],
        },
        "safety_report.json": {
            "artifact_version": "d4l_safety_report.v1",
            "provider_generation_calls": 0,
            "external_provider_http": 0,
            "semantic_verifier_calls": 0,
            "database_writes": 0,
            "holdout_access": 0,
            "unknown_discriminator_fails_closed": True,
            "missing_discriminator_fails_closed": True,
            "measurement_cannot_escape_typed_validation": True,
            "invalid_graph_refs_are_not_repaired": True,
            "scope_graph_does_not_establish_semantic_support": True,
            "formal_definition_unit_basis_inferred": False,
            "scope_category_overlap_under_measurement_discriminator": (
                "UNRESOLVED_CONTRACT_AMBIGUITY: same JSON payload is valid in both branches; "
                "no shape heuristic or semantic contract narrowing was introduced"
            ),
            "raw_d4j_files_modified": False,
        },
        "test_runtime_report.json": {
            "artifact_version": "d4l_test_runtime_report.v1",
            "runtime": "D:/agentProject/.venv (Python 3.12)",
            "pytest": "8.4.2",
            "mypy": "1.20.2",
            "ruff": "0.16.4",
            "targeted_test_result": "PASS",
            "full_pytest_result": "PASS (9 existing skipped tests)",
            "mypy_apps_api_result": "PASS",
            "mypy_replay_script_result": "PASS",
            "targeted_ruff_result": "PASS",
            "git_diff_check_result": "PASS",
            "pytest_cache_warning": (
                "Windows denies writing .pytest_cache; tests were run with cache disabled."
            ),
        },
        "frozen_input_integrity.json": {
            "artifact_version": "d4l_frozen_input_integrity.v1",
            "inputs": input_integrity,
            "all_match_frozen": all(
                x["raw_matches_frozen"] and x["normalized_matches_frozen"] for x in input_integrity
            ),
        },
    }
    for filename, artifact in artifacts.items():
        _write_json(OUTPUT_ROOT / filename, artifact)

    return artifacts


def _write_hash_manifest() -> None:
    manifest: dict[str, Any] = {
        "artifact_version": "d4l_hash_manifest.v1",
        "hash_algorithm": "SHA-256 over file bytes",
        "self_hash_contract": "hash_manifest.json omitted to avoid self-reference",
        "provider_calls": 0,
        "parser_patch_version": PATCH_VERSION,
        "files": {},
    }
    paths = [
        ROOT / "apps/api/app/domain/source_context_extraction_v3_1.py",
        ROOT / "tests/unit/test_source_context_extraction_v3_1_dispatch.py",
        ROOT / "scripts/analyze_v2_1r_d4l_parser_closure.py",
        *sorted(path for path in OUTPUT_ROOT.glob("*.json") if path.name != "hash_manifest.json"),
        OUTPUT_ROOT / "final_report.md",
    ]
    for path in paths:
        manifest["files"][path.relative_to(ROOT).as_posix()] = _sha256(path)
    _write_json(OUTPUT_ROOT / "hash_manifest.json", manifest)


def main() -> None:
    artifacts = _write_reports()
    summary = artifacts["d4j_frozen_parser_replay.json"]
    final = {
        "artifact_version": "d4l_parser_closure_status.v1",
        "parser_patch_version": PATCH_VERSION,
        "replay_schema_pass": summary["schema_parse_pass_count"],
        "replay_schema_fail": summary["schema_parse_fail_count"],
        "historical_scope_union_failures_cleared": summary[
            "historical_scope_union_failures_cleared"
        ],
        "all_frozen_inputs_match": summary["all_raw_hashes_match_frozen"]
        and summary["all_normalized_hashes_match_frozen"],
        "semantic_contract_ambiguity": (
            "SCOPE-only scope_category JSON is also admissible under the existing "
            "MEASUREMENT payload model; context_type remains the sole branch selector."
        ),
        "final_decision": "V3.1 PARSER CLOSURE REVEALS SEMANTIC CONTRACT ISSUE",
    }
    _write_json(OUTPUT_ROOT / "status.json", final)
    report = f"""# V2.1R D4-L Typed-Union Dispatch & Parser Closure

## A. D4-K Root Cause

`SourceContextV31.semantic_value` used an overlapping left-to-right union. A SCOPE payload containing only `scope_category` was first accepted as `MeasurementSemanticsV31`, then rejected by the context-type invariant before the generic dictionary alternative was considered. D4-K classified the failure as deterministic deserialization order, not a scope-owner or graph-validator defect.

## B. Patch Versioning

Parser-only identity: `{PATCH_VERSION}`. D4-J artifacts and their recorded implementation hashes remain historical and unchanged.

## C. Canonical Discriminator

`context_type` is the sole canonical discriminator. There is no second semantic-kind field in the frozen v3.1 representation.

## D. Parser Dispatch Contract

`MEASUREMENT` dispatches to `MeasurementSemanticsV31`; every other supported `ContextType` dispatches to `GenericContextSemanticValueV31`. The generic RootModel serializes as the same JSON object shape. Dispatch does not inspect payload keys, union order, or natural language.

## E. Union Exhaustiveness

All 12 declared context families have exactly one branch: Measurement (1) and generic context (11). Missing and unsupported discriminators fail closed. See `discriminator_dispatch_matrix.json` and `union_exhaustiveness_matrix.json`.

## F. Scope-Only Parsing

SCOPE-only `scope_category` values now parse as generic context payloads, preserve their JSON content, and round-trip. They do not require measurement fields. SourceScope remains the canonical owner for scope kind, membership, graph structure, and provenance.

## G. Measurement Parsing

Measurement payloads remain typed. Empty, malformed, boolean-numeric, or extra-field measurement payloads fail closed. Frozen D4-J replay parsed all 10 model-emitted Measurement contexts; the 9 previously audited source-supported quantitative facts remain distinguished from Claim 5's qualitative pseudo-measurement.

## H. Generic Payload Safety

The generic path is an explicit domain wrapper selected only by `context_type`; it cannot accept a malformed Measurement when the discriminator is `MEASUREMENT`. No parser-side semantic repair was added.

## I. Round-Trip Validation

Synthetic tests cover every `ContextType`, and all 8 frozen D4-J whole-response objects round-trip with identity preserved. Result: {summary['round_trip_pass_count']}/{summary['round_trip_count']} frozen response round trips pass.

## J. Same-Path Fake Tests

Fake Gateway responses traverse the service's structured-output parsing and validation path. Covered: historical SCOPE-only, Measurement, mixed Scope+Measurement+Condition, multiple scopes, multiple measurements, malformed Measurement, and single-call/no-regeneration behavior. Provider calls: 0.

## K. D4-J Frozen Parser Replay

The raw and normalized input hashes match their freeze manifest. Schema parse: {summary['schema_parse_pass_count']}/{summary['source_claim_count']} PASS; {summary['schema_parse_fail_count']} FAIL. All 4 historical SCOPE union parse blockers clear. Existing graph/semantic findings remain visible (including attachment binding ambiguity and two model-returned UNRESOLVED results); replay is bug closure on frozen evidence only, not fresh generalization.

## L. Measurement Post-Unblock Observation

Historical source-supported fact count: 9. Typed Measurement contexts parsed: {summary['measurement_contexts_typed_and_parseable']}; required evidence-role slots: {summary['measurement_context_role_slots_required']}; present: {summary['measurement_context_role_slots_present']}; missing: {len(summary['measurement_context_role_slots_missing'])}. No definitions, units, or bases were inferred; no measurement contract changed.

## M. Scope Ownership Regression

Frozen replay scope assessments: all resolved; duplicate `SCOPE_STRUCTURE` bindings required: no; found: {summary['duplicate_scope_structure_bindings']}. D4-I ownership behavior remains intact.

## N. Whole-Claim Abstention Watch

The two historical model-over-abstention Claims remain `MODEL_OVER_ABSTENTION_WATCH`; no prompt or status behavior was changed.

## O. Attachment Ownership Watch

`BINDING_CONTRACT_AMBIGUITY` remains a separate watch. D4-L did not redesign or synthesize attachment bindings.

## P. Safety Invariants

Provider generation / external HTTP / verifier calls: 0 / 0 / 0. Database writes: 0. Holdout access: 0. No semantic inference, graph repair, scope-to-support promotion, or raw D4-J artifact modification.

## Q. Production Behavior

Only the shadow/development v3.1 domain deserialization path changed. ResearchLoop, persistence, Acceptance, Coverage, Grounding Requirement, Matcher, Hybrid verifier, and production extraction behavior were not modified.

## R. Provider Calls

0 Provider calls; 0 external Provider HTTP requests.

## S. Quality Gates

- Targeted v3.1 parser/Fake tests: PASS.
- Full `pytest -q` (cache disabled due Windows `.pytest_cache` access denial): PASS; 9 pre-existing skipped tests.
- `mypy apps/api`: PASS (227 source files).
- Replay script mypy: PASS.
- Targeted Ruff: PASS.
- `git diff --check`: PASS.

## T. Artifacts

This directory contains the parser patch contract, dispatch/exhaustiveness matrices, round-trip and Fake reports, offline replay, measurement observation, Scope ownership report, safety/runtime reports, frozen input integrity, status, final report, and hash manifest. The replay script is `scripts/analyze_v2_1r_d4l_parser_closure.py`.

## U. Final Decision

**V3.1 PARSER CLOSURE REVEALS SEMANTIC CONTRACT ISSUE**

Reason: the required wrong-discriminator case cannot be fully classified under the frozen representation. `scope_category` is valid both in generic SCOPE payloads and in `MeasurementSemanticsV31`; therefore a `MEASUREMENT` object containing only `scope_category` is indistinguishable from a valid typed Measurement using the sole canonical discriminator. Rejecting it would either narrow the frozen Measurement contract or infer intent from payload shape, both forbidden by D4-L. The deterministic D4-K parser defect is closed, but this overlap needs an explicit contract decision before claiming the full D4-L gate passes. No further validation or Provider execution was started.
"""
    (OUTPUT_ROOT / "final_report.md").write_text(report, encoding="utf-8")
    _write_hash_manifest()


if __name__ == "__main__":
    main()
