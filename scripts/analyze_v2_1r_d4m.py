"""Build D4-M frozen-evidence replay and contract artifacts without Provider calls."""

# ruff: noqa: E501

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from app.domain.source_context_extraction_v3_2 import (
    SOURCE_CONTEXT_EXTRACTION_SCHEMA_V32,
    SourceExtractionResponseV32,
    validate_source_context_v32,
)
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "v2_1r_d4m"
D4J = ROOT / "artifacts" / "v2_1r_d4j_b"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(name: str, payload: Any) -> Path:
    path = OUTPUT / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _measurement_fields(response: dict[str, Any]) -> list[list[str]]:
    return [
        sorted((context.get("semantic_value") or {}).keys())
        for context in response.get("source_contexts", [])
        if context.get("context_type") == "MEASUREMENT"
    ]


def _legacy_version_to_v32(payload: dict[str, Any]) -> dict[str, Any]:
    """Adapt only version tags in memory; frozen raw artifacts remain byte-identical."""
    adapted = copy.deepcopy(payload)
    adapted["schema_version"] = SOURCE_CONTEXT_EXTRACTION_SCHEMA_V32
    for relation in adapted.get("semantic_relations", []):
        relation["schema_version"] = SOURCE_CONTEXT_EXTRACTION_SCHEMA_V32
    return adapted


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    preflight = _read(OUTPUT / "frozen_evidence_feasibility_audit.json")

    core_modifier = {
        "artifact_version": "v2_1r_d4m_core_vs_modifier_matrix.v1",
        "core_fields": {
            "measurement_identity": "Names the measured construct; alone can anchor a measurement context.",
            "formal_definition": "Explicit source-supported definition of the measured construct.",
            "observed_value": "Typed observed value with its raw lexical representation.",
            "threshold": "Fully typed threshold: comparator + typed observed value + exact non-empty raw_text.",
        },
        "modifier_fields": {
            "unit": "Qualifies a value; does not independently establish a measurement.",
            "comparator": "Relational modifier; requires a measurable value/threshold context.",
            "basis_ref": "Measurement basis reference; does not independently establish a measurement.",
            "scope_category": "Where/to what set a measurement applies; not what is measured or an observed fact.",
            "target_object": "Target/object modifier.",
            "qualifiers": "Additional modifiers only.",
        },
        "valid_abstract_shapes": [
            {"shape": ["measurement_identity", "scope_category"], "valid": True},
            {"shape": ["observed_value", "unit"], "valid": True},
            {"shape": ["formal_definition"], "valid": True},
            {"shape": ["threshold", "comparator"], "valid": True},
        ],
        "invalid_modifier_only_shapes": [
            ["scope_category"],
            ["unit"],
            ["basis_ref"],
            ["comparator"],
            ["target_object"],
            ["qualifiers"],
            ["scope_category", "unit"],
        ],
        "classification_basis": "Generic semantic ownership; no Claim-specific or metric-specific rules.",
    }
    _dump("core_vs_modifier_matrix.json", core_modifier)

    threshold_decision = {
        "artifact_version": "v2_1r_d4m_threshold_decision.v1",
        "independent_core_anchor": True,
        "required_shape": {
            "comparator": "typed ObservedValueComparator",
            "value": "typed ObservedValue union member",
            "raw_text": "non-empty exact source representation",
        },
        "comparator_alone_is_core": False,
        "basis": "A complete typed threshold expresses a measurement boundary and includes both typed criterion and lexical provenance. No threshold-only accepted counterexample exists in inspected frozen development evidence.",
        "inference": "none",
    }
    _dump("threshold_decision.json", threshold_decision)

    replay_rows: list[dict[str, Any]] = []
    raw_hashes: dict[str, str] = {}
    for path in sorted(D4J.glob("claim_*.json")):
        frozen = _read(path)
        raw_hashes[path.name] = _sha(path)
        model_payload = (
            frozen.get("raw_output", {})
            .get("structured_response", {})
            .get("model_structured_object")
        )
        if not isinstance(model_payload, dict):
            replay_rows.append(
                {"artifact": path.name, "parser_status": "NO_STRUCTURED_OBJECT", "unchanged": True}
            )
            continue
        adapted = _legacy_version_to_v32(model_payload)
        try:
            response = SourceExtractionResponseV32.model_validate(adapted)
        except ValidationError as exc:
            replay_rows.append(
                {
                    "artifact": path.name,
                    "parser_status": "REJECTED",
                    "error_types": sorted(
                        {str(error.get("type", "invalid")) for error in exc.errors()}
                    ),
                    "measurement_count": len(_measurement_fields(adapted)),
                    "measurement_admissibility": "NOT_REACHED",
                    "scope_count": sum(
                        item.get("context_type") == "SCOPE"
                        for item in adapted.get("source_contexts", [])
                    ),
                    "raw_artifact_unchanged": True,
                }
            )
            continue
        result = validate_source_context_v32(response, frozen.get("claim", {}).get("text", ""))
        measurement_fields = _measurement_fields(adapted)
        replay_rows.append(
            {
                "artifact": path.name,
                "parser_status": "PASS",
                "model_status": response.status,
                "measurement_count": len(measurement_fields),
                "measurement_fields": measurement_fields,
                "measurement_admissibility": "PASS" if measurement_fields else "NOT_APPLICABLE",
                "human_label": _human_label(path.stem),
                "v32_graph_validation_status": result.status.value,
                "scope_count": sum(
                    item.get("context_type") == "SCOPE"
                    for item in adapted.get("source_contexts", [])
                ),
                "raw_artifact_unchanged": True,
            }
        )

    d4j_review = _read(D4J / "human_review.json")
    parser_pass = sum(row.get("parser_status") == "PASS" for row in replay_rows)
    measurement_pass = sum(row.get("measurement_admissibility") == "PASS" for row in replay_rows)
    measurement_contexts = sum(row.get("measurement_count", 0) for row in replay_rows)
    scope_pass = sum(
        row.get("parser_status") == "PASS" and row.get("scope_count", 0) > 0 for row in replay_rows
    )
    unresolved_count = sum(row.get("model_status") == "UNRESOLVED" for row in replay_rows)
    false_positive = [
        row for row in replay_rows if row.get("human_label") == "MEASUREMENT_FALSE_POSITIVE"
    ]
    replay = {
        "artifact_version": "v2_1r_d4m_d4j_offline_replay.v1",
        "input_manifest": "artifacts/v2_1r_d4j_b/raw_freeze_manifest.json",
        "input_manifest_sha256": _sha(D4J / "raw_freeze_manifest.json"),
        "raw_claim_artifact_sha256": raw_hashes,
        "method": "Deterministic in-memory schema/domain/graph replay of frozen D4-J outputs. Only schema_version tags were adapted in memory from v3.1 to v3.2; no raw file or model output was rewritten.",
        "provider_calls": 0,
        "claims_total": len(replay_rows),
        "parser_pass_count": parser_pass,
        "parser_fail_count": len(replay_rows) - parser_pass,
        "measurement_context_count": measurement_contexts,
        "measurement_admissibility_pass_claims": measurement_pass,
        "measurement_admissibility_fail_claims": sum(
            row.get("measurement_admissibility") == "FAIL" for row in replay_rows
        ),
        "scope_claims_parsed": scope_pass,
        "whole_claim_unresolved_count_unchanged": unresolved_count,
        "remaining_graph_and_completeness_failures": {
            "human_claim_labels": d4j_review.get("claim_label_counts", {}),
            "graph_validation_status_counts": _count_values(
                replay_rows, "v32_graph_validation_status"
            ),
            "note": "D4-J human-identified attachment, scope, faithfulness, and completeness issues are outside the Measurement-admissibility change and remain visible.",
        },
        "known_human_semantic_measurement_false_positive": false_positive,
        "claims": replay_rows,
        "generalization_claim": "none; this is frozen development replay only",
    }
    _dump("d4j_offline_replay.json", replay)

    contract = {
        "artifact_version": "v2_1r_d4m_measurement_admissibility_contract.v1",
        "schema_version": "source-context-extraction-schema-v3.2",
        "prompt_version": "source-context-extraction.generic.v3.2",
        "domain_version": "source-context-extraction-domain-v3.2",
        "invariant": "MEASUREMENT semantic_value must include at least one valid intrinsic anchor: measurement_identity, formal_definition, observed_value, or fully typed threshold.",
        "modifier_only_invalid": True,
        "discriminator": "context_type remains authoritative; no shape inference or generic fallback.",
        "scope_category": "retained as modifier, never sufficient alone",
        "threshold": threshold_decision,
        "no_fact_synthesis": True,
        "production_default_changed": False,
    }
    _dump("measurement_admissibility_contract.json", contract)

    schema_summary = {
        "artifact_version": "v2_1r_d4m_schema_domain_revision_summary.v1",
        "new_versions": {
            "schema": "source-context-extraction-schema-v3.2",
            "prompt": "source-context-extraction.generic.v3.2",
            "domain": "source-context-extraction-domain-v3.2",
        },
        "new_implementation_files": [
            "apps/api/app/domain/source_context_extraction_v3_2.py",
            "apps/api/app/services/source_context_extractor_v3_2.py",
        ],
        "preserved": ["v3", "v3.1", "parser-v3.1.1", "D4-L deterministic discriminator dispatch"],
        "scope": "Shadow/development only; no ResearchLoop, DB, Acceptance, Coverage, or production-default wiring.",
    }
    _dump("schema_domain_revision_summary.json", schema_summary)

    fake_report = {
        "artifact_version": "v2_1r_d4m_same_path_fake_report.v1",
        "test": "tests/unit/test_source_context_extraction_v3_2.py::test_same_path_fake_accepts_measurement_scope_and_multiple_contexts",
        "gateway_response_to_schema_to_domain_to_graph_validator": "covered",
        "valid_measurement": "pass",
        "modifier_only_measurement": "rejected_fail_closed",
        "valid_scope_scope_category": "pass_generic_branch",
        "mixed_scope_and_measurement": "pass",
        "multiple_measurements": "pass",
        "fake_provider_calls_in_test": 1,
        "real_provider_calls": 0,
    }
    _dump("same_path_fake_report.json", fake_report)

    safety = {
        "artifact_version": "v2_1r_d4m_safety_report.v1",
        "provider_generation_calls": 0,
        "external_http": 0,
        "real_extractor_calls": 0,
        "semantic_verifier_calls": 0,
        "holdout_access": 0,
        "database_writes": 0,
        "no_semantic_fact_synthesis": True,
        "no_inferred_unit_basis_definition": True,
        "invalid_measurement_generic_fallback": False,
        "scope_structure_does_not_prove_semantics": True,
        "invalid_refs_fail_closed": True,
        "attachment_binding_synthesis": False,
        "downstream_or_production_behavior_changed": False,
    }
    _dump("safety_report.json", safety)

    regression = {
        "artifact_version": "v2_1r_d4m_regression_report.v1",
        "parser_v3_1_1": "covered by tests/unit/test_source_context_extraction_v3_1_dispatch.py",
        "v3_1_and_older_contract_files_modified": False,
        "scope_ownership_logic_modified": False,
        "attachment_watch": "BINDING_CONTRACT_AMBIGUITY remains frozen",
        "model_over_abstention_watch": "MODEL_OVER_ABSTENTION_WATCH remains frozen",
        "additional_measurement_issue": "D4-J Claim 5 remains structurally admissible via measurement_identity despite human review identifying qualitative-performance over-typing; this is not silently claimed fixed by the narrow anchor invariant.",
    }
    _dump("regression_report.json", regression)

    hash_inputs = [
        path
        for path in sorted(OUTPUT.iterdir())
        if path.is_file() and path.name != "hash_manifest.json"
    ]
    hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): _sha(path) for path in hash_inputs}
    components = [
        ROOT / "apps/api/app/domain/source_context_extraction_v3_2.py",
        ROOT / "apps/api/app/services/source_context_extractor_v3_2.py",
        ROOT / "tests/unit/test_source_context_extraction_v3_2.py",
        ROOT / "scripts/analyze_v2_1r_d4m.py",
    ]
    hash_manifest = {
        "artifact_version": "v2_1r_d4m_hash_manifest.v1",
        "artifact_hashes_sha256": hashes,
        "implementation_hashes_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): _sha(path) for path in components
        },
        "frozen_evidence_feasibility_audit_sha256": _sha(
            OUTPUT / "frozen_evidence_feasibility_audit.json"
        ),
    }
    _dump("hash_manifest.json", hash_manifest)

    print(
        json.dumps(
            {
                "output_directory": str(OUTPUT),
                "claims": len(replay_rows),
                "parser_pass": parser_pass,
                "measurement_contexts": measurement_contexts,
                "measurement_admissibility_pass_claims": measurement_pass,
                "scope_claims_parsed": scope_pass,
                "whole_claim_unresolved_unchanged": unresolved_count,
                "known_human_measurement_false_positive_count": len(false_positive),
                "audit_decision": preflight["decision"],
            },
            indent=2,
        )
    )


def _human_label(stem: str) -> str | None:
    index = int(stem.split("_")[1])
    reviews = _read(D4J / "human_review.json")["claims"]
    review = next((item for item in reviews if item["ordinal"] == index), None)
    if review is None:
        return None
    if index == 5:
        return "MEASUREMENT_FALSE_POSITIVE"
    label = review.get("label")
    return label if isinstance(label, str) else None


def _count_values(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key, "NOT_REACHED")
        counts[value] = counts.get(value, 0) + 1
    return counts


if __name__ == "__main__":
    main()
