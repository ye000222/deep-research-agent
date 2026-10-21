"""Audit frozen D4-G/D4-J measurement identity semantics; no execution or mutation."""

# Audit annotations and rendered Markdown intentionally keep full explanations inline.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ARTIFACTS / "v2_1r_d4n"
D4G = ARTIFACTS / "v2_1r_d4g_b_execution_attempt_1"
D4J = ARTIFACTS / "v2_1r_d4j_b"

CLASSIFICATION: dict[tuple[str, str], tuple[str, str]] = {
    ("01a0c86f-e82d-7dd8-b42f-d48bba5b879e", "c1"): (
        "RANGE_OR_DISTANCE",
        "D4-G human review: Measurement Typing MAJOR; context only labels a range and omits typed values/units.",
    ),
    ("01a0c86f-e82d-7dd8-b42f-d48bba5b879e", "c2"): (
        "RANGE_OR_DISTANCE",
        "D4-G human review: Measurement Typing MAJOR; context only labels a range and omits typed values/units.",
    ),
    ("01a0c41c-a7cc-73d2-9f55-6e19ede5be76", "c2"): (
        "COUNT_OR_CARDINALITY",
        "D4-G human review: named parameter-count measure, but typed observed value/units are absent; Measurement Typing MAJOR.",
    ),
    ("01a0c41c-a7cc-73d2-9f55-6e19ede5be76", "c3"): (
        "SIZE_OR_CAPACITY",
        "D4-G human review: named model-size measure, but typed observed value/units are absent; Measurement Typing MAJOR.",
    ),
    ("01a0c41c-a7cc-73d2-9f55-6e19ede5be76", "c4"): (
        "NAMED_METRIC",
        "D4-G human review: GFLOPs identity is present but typed observed value/units are absent; Measurement Typing MAJOR.",
    ),
    ("01a0c406-6977-73df-aa7b-c2cdfd0e8ee8", "c1"): (
        "SCORE_OR_INDEX",
        "D4-G human review: valid metric identity/value semantics; minor issue is an evidence-role label, not the identity.",
    ),
    ("01a0c406-6977-73df-aa7b-c2cdfd0e8ee8", "c2"): (
        "SCORE_OR_INDEX",
        "D4-G human review: valid metric identity/value semantics; minor issue is an evidence-role label, not the identity.",
    ),
    ("01a0c406-6977-73df-aa7b-c2cdfd0e8ee8", "c3"): (
        "SCORE_OR_INDEX",
        "D4-G human review: valid metric identity/value semantics; minor issue is an evidence-role label, not the identity.",
    ),
    ("01a0c41c-a7d9-7ef5-bbde-b70d8d0787cd", "C1"): (
        "SIZE_OR_CAPACITY",
        "D4-J measurement subreview PASS for identity/value/unit; overall Claim issues concern predicate/scope.",
    ),
    ("01a0c43b-3e5a-7801-8bda-b8ade1fd7eb8", "C1"): (
        "SCORE_OR_INDEX",
        "D4-J human review PASS: average Image AUROC identity/value/scope preserved.",
    ),
    ("01a0c43b-3e5a-7801-8bda-b8ade1fd7eb8", "C2"): (
        "SCORE_OR_INDEX",
        "D4-J human review PASS: average Pixel AUROC identity/value/scope preserved.",
    ),
    ("01a0c405-03ed-79fd-9758-15029694fc0c", "C1"): (
        "COUNT_OR_CARDINALITY",
        "D4-J measurement subreview PASS; overall Claim issue is a separate attachment-binding defect.",
    ),
    ("01a0c405-03ed-79fd-9758-15029694fc0c", "C2"): (
        "COUNT_OR_CARDINALITY",
        "D4-J measurement subreview PASS; image counts and role qualifiers are source-supported.",
    ),
    ("01a0c405-03ed-79fd-9758-15029694fc0c", "C3"): (
        "COUNT_OR_CARDINALITY",
        "D4-J measurement subreview PASS; image counts and role qualifiers are source-supported.",
    ),
    ("01a0c834-62f9-7c30-8813-20d24892e2f4", "C2"): (
        "QUALITATIVE_PROPERTY",
        "D4-J human review: qualitative performance comparison only; no metric definition/value; false Measurement typing.",
    ),
    ("01a0c406-6970-7f63-be80-943c2a5822db", "C2"): (
        "SCORE_OR_INDEX",
        "D4-J measurement subreview PASS: I-AUROC identity/value/target preserved; separate SCOPE parse issue.",
    ),
    ("01a0c406-6970-7f63-be80-943c2a5822db", "C3"): (
        "SCORE_OR_INDEX",
        "D4-J measurement subreview PASS: I-AP identity/value/target preserved; separate SCOPE parse issue.",
    ),
    ("01a0c406-6970-7f63-be80-943c2a5822db", "C4"): (
        "SCORE_OR_INDEX",
        "D4-J measurement subreview PASS: I-F1-max identity/value/target preserved; separate SCOPE parse issue.",
    ),
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(name: str, data: Any) -> Path:
    path = OUTPUT / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _d4g_reviews() -> dict[str, dict[str, Any]]:
    data = _read(D4G / "human_review.json")
    return {row["claim_id"]: row for row in data["claim_reviews"]}


def _d4j_reviews() -> dict[str, dict[str, Any]]:
    data = _read(D4J / "human_review.json")
    return {row["claim_id"]: row for row in data["claims"]}


def _span_map(response: dict[str, Any], context_ref: str) -> tuple[list[str], list[str]]:
    evidence_refs: set[str] = set()
    for attachment in response.get("source_attachments", []):
        if attachment.get("context_ref") == context_ref:
            evidence_refs.update(attachment.get("evidence_refs", []))
    for binding in response.get("evidence_bindings", []):
        if binding.get("target_kind") == "CONTEXT" and binding.get("target_ref") == context_ref:
            evidence_ref = binding.get("evidence_ref")
            if evidence_ref:
                evidence_refs.add(evidence_ref)
    spans = [
        span.get("exact_text", "")
        for span in response.get("evidence_spans", [])
        if span.get("evidence_ref") in evidence_refs
    ]
    return sorted(evidence_refs), spans


def _ledger() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    g_reviews = _d4g_reviews()
    j_reviews = _d4j_reviews()

    for path in sorted(D4G.glob("claim_*.json")):
        frozen = _read(path)
        raw = frozen.get("raw_provider_structured_response") or {}
        response = raw.get("raw_structured_json") or {}
        claim_id = frozen["claim_id"]
        human = g_reviews.get(claim_id, {})
        for context in response.get("source_contexts", []):
            if context.get("context_type") != "MEASUREMENT":
                continue
            semantics = context.get("semantic_value") or {}
            identity_field = next(
                (name for name in ("measurement_identity", "measurement", "metric") if name in semantics),
                None,
            )
            if identity_field is None:
                continue
            context_ref = context["context_ref"]
            primary_class, judgment = CLASSIFICATION[(claim_id, context_ref)]
            evidence_refs, source_spans = _span_map(response, context_ref)
            identity_only = not any(
                semantics.get(name) is not None for name in ("observed_value", "formal_definition", "threshold")
            )
            rows.append(
                {
                    "source_phase": "D4-G-B v3 development Canary",
                    "artifact": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "artifact_sha256": _sha(path),
                    "claim_id": claim_id,
                    "claim_sha256": frozen.get("claim_text_sha256"),
                    "context_ref": context_ref,
                    "context_type": context.get("context_type"),
                    "raw_source_spans": source_spans,
                    "span_evidence_refs": evidence_refs,
                    "measurement_identity": semantics.get(identity_field),
                    "identity_source_field": identity_field,
                    "typed_fields_present": sorted(semantics),
                    "observed_value_present": semantics.get("observed_value") is not None,
                    "formal_definition_present": semantics.get("formal_definition") is not None,
                    "threshold_present": semantics.get("threshold") is not None,
                    "unit_present": semantics.get("unit") is not None,
                    "basis_present": semantics.get("basis_ref") is not None,
                    "scope_or_category_present": semantics.get("scope_category") is not None,
                    "target_or_object_present": semantics.get("target_object") is not None,
                    "primary_semantic_class": primary_class,
                    "human_semantic_judgment": judgment,
                    "human_claim_label": human.get("final_label"),
                    "human_measurement_dimension": human.get("dimensions", {}).get("MEASUREMENT_TYPING"),
                    "human_review_notes": human.get("notes", []),
                    "identity_only": identity_only,
                }
            )

    frozen_manifest = _read(D4J / "raw_freeze_manifest.json")
    claim_hashes = {row["claim_id"]: row["claim_sha256"] for row in frozen_manifest["claim_results"]}
    for path in sorted(D4J.glob("claim_*.json")):
        frozen = _read(path)
        raw_output = frozen.get("raw_output") or {}
        raw = raw_output.get("structured_response") or {}
        response = raw.get("model_structured_object") or {}
        claim = frozen.get("claim", {})
        claim_id = claim.get("claim_id")
        human = j_reviews.get(claim_id, {})
        for context in response.get("source_contexts", []):
            if context.get("context_type") != "MEASUREMENT":
                continue
            semantics = context.get("semantic_value") or {}
            if "measurement_identity" not in semantics:
                continue
            context_ref = context["context_ref"]
            primary_class, judgment = CLASSIFICATION[(claim_id, context_ref)]
            evidence_refs, source_spans = _span_map(response, context_ref)
            measurement_only = not any(
                semantics.get(name) is not None for name in ("observed_value", "formal_definition", "threshold")
            )
            rows.append(
                {
                    "source_phase": "D4-J-B v3.1 frozen development output",
                    "artifact": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "artifact_sha256": _sha(path),
                    "claim_id": claim_id,
                    "claim_sha256": claim_hashes.get(claim_id),
                    "context_ref": context_ref,
                    "context_type": context.get("context_type"),
                    "raw_source_spans": source_spans,
                    "span_evidence_refs": evidence_refs,
                    "measurement_identity": semantics.get("measurement_identity"),
                    "identity_source_field": "measurement_identity",
                    "typed_fields_present": sorted(semantics),
                    "observed_value_present": semantics.get("observed_value") is not None,
                    "formal_definition_present": semantics.get("formal_definition") is not None,
                    "threshold_present": semantics.get("threshold") is not None,
                    "unit_present": semantics.get("unit") is not None,
                    "basis_present": semantics.get("basis_ref") is not None,
                    "scope_or_category_present": semantics.get("scope_category") is not None,
                    "target_or_object_present": semantics.get("target_object") is not None,
                    "primary_semantic_class": primary_class,
                    "human_semantic_judgment": judgment,
                    "human_claim_label": human.get("label"),
                    "human_measurement_review": human.get("review", {}),
                    "human_review_findings": human.get("findings", []),
                    "identity_only": measurement_only,
                }
            )
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ledger = _ledger()

    evidence_paths = [
        D4G / "human_review.json",
        D4G / "execution_manifest.json",
        D4J / "human_review.json",
        D4J / "raw_freeze_manifest.json",
        ARTIFACTS / "v2_1r_d4j_k" / "measurement_exercise_audit.json",
        ARTIFACTS / "v2_1r_d4l" / "measurement_post_unblock_observation.json",
        ARTIFACTS / "v2_1r_d4m" / "frozen_evidence_feasibility_audit.json",
        ARTIFACTS / "v2_1r_d4m" / "d4j_offline_replay.json",
        ARTIFACTS / "v2_1r_d4c_systematic_failure_ownership" / "measurement_audit.json",
        ARTIFACTS / "v2_1r_d4b_execution_attempt_2" / "aggregate.json",
    ]
    source_inventory = [
        {
            "path": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": _sha(path),
            "use": (
                "primary frozen v3/v3.1 result or human review"
                if path.parent in (D4G, D4J)
                else "cross-check/derived record; deduplicated from primary ledger"
                if "d4j_k" in str(path) or "d4l" in str(path) or "d4m" in str(path)
                else "legacy v2 background only; excluded from v3/v3.1 identity counts"
            ),
        }
        for path in evidence_paths
    ]
    for directory in (D4G, D4J):
        for path in sorted(directory.glob("claim_*.json")):
            source_inventory.append(
                {
                    "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": _sha(path),
                    "use": "primary frozen development output; parsed once into ledger",
                }
            )

    _write(
        "measurement_identity_ledger.json",
        {
            "artifact_version": "v2_1r_d4n_measurement_identity_ledger.v1",
            "scope": "D4-G v3 and D4-J v3.1 frozen development-only outputs; D4-K/L/M cross-checks deduplicated.",
            "holdout_access": 0,
            "provider_calls": 0,
            "unique_context_count": len(ledger),
            "records": ledger,
            "source_inventory": source_inventory,
            "legacy_alias_note": "D4-G v3 used semantic key 'measurement' rather than 'measurement_identity'; these are retained as identity-like Measurement proposals and not conflated with exact v3.1 field counts.",
        },
    )

    exact_identity_rows = [
        row for row in ledger if row["identity_source_field"] == "measurement_identity"
    ]
    identity_only = [row for row in exact_identity_rows if row["identity_only"]]
    identity_only_audit = {
        "artifact_version": "v2_1r_d4n_identity_only_measurement_audit.v1",
        "valid_identity_only_measurement_count": 0,
        "false_identity_only_measurement_count": sum(
            row["primary_semantic_class"] in {"QUALITATIVE_PROPERTY", "EVALUATIVE_JUDGMENT", "RESULT_DESCRIPTION", "OTHER_NON_MEASUREMENT"}
            for row in identity_only
        ),
        "ambiguous_identity_only_count": sum(
            row["primary_semantic_class"] == "AMBIGUOUS" for row in identity_only
        ),
        "exact_identity_field_records": len(exact_identity_rows),
        "exact_identity_only_records": [
            {
                "claim_id": row["claim_id"],
                "claim_sha256": row["claim_sha256"],
                "context_ref": row["context_ref"],
                "identity": row["measurement_identity"],
                "primary_semantic_class": row["primary_semantic_class"],
                "human_judgment": row["human_semantic_judgment"],
                "source_spans": row["raw_source_spans"],
            }
            for row in identity_only
        ],
        "legacy_measurement_alias_only_candidates": [
            {
                "claim_id": row["claim_id"],
                "context_ref": row["context_ref"],
                "identity_like_value": row["measurement_identity"],
                "human_judgment": row["human_semantic_judgment"],
                "reason_not_counted_as_valid_identity_only": "D4-G human Measurement Typing was MAJOR because typed observations/units were omitted; these were not accepted valid outputs and use the legacy v3 field name.",
            }
            for row in ledger
            if row["identity_source_field"] == "measurement"
            and row["identity_only"]
        ],
        "interpretation": "No human-confirmed valid identity-only Measurement was found. The sole exact v3.1 identity-only record is the D4-J qualitative false positive. This small frozen sample does not prove genuine identity-only representations are never useful.",
    }
    _write("identity_only_measurement_audit.json", identity_only_audit)

    false_positive = next(
        row for row in ledger if row["primary_semantic_class"] == "QUALITATIVE_PROPERTY"
    )
    _write(
        "qualitative_false_positive_audit.json",
        {
            "artifact_version": "v2_1r_d4n_qualitative_false_positive_audit.v1",
            "source": {
                "claim_id": false_positive["claim_id"],
                "claim_sha256": false_positive["claim_sha256"],
                "context_ref": false_positive["context_ref"],
                "identity": false_positive["measurement_identity"],
                "fields_present": false_positive["typed_fields_present"],
                "source_spans": false_positive["raw_source_spans"],
                "human_findings": false_positive["human_review_findings"],
            },
            "human_primary_class": "QUALITATIVE_PROPERTY",
            "why_not_measurement": "The source asserts a qualitative comparative performance property relative to industrial requirements; the proposed context carries no observed value, formal definition, or threshold, and the human review says no metric is stated.",
            "not_claimed": [
                "This one observation establishes a failure frequency.",
                "A keyword or adjective list can reliably separate measurements from qualitative properties.",
            ],
            "contract_conclusion": "The free-text identity slot admits a generic class of non-measurement qualitative descriptions; systematic contract vulnerability is present, while observed frequency remains one frozen manifestation.",
        },
    )

    boundary = {
        "artifact_version": "v2_1r_d4n_measurement_boundary_matrix.v1",
        "rows": [
            {"semantic_case": "named score/index with reported score", "primary_class": "SCORE_OR_INDEX", "measurement": "yes", "minimum_semantic_anchor": "identity/classification linked to observed score", "caveat": "score-like name and numeric value still require source-grounded association"},
            {"semantic_case": "parameter cardinality or dataset image count", "primary_class": "COUNT_OR_CARDINALITY", "measurement": "yes when asserted as a count", "minimum_semantic_anchor": "count property plus source-supported count value or definition", "caveat": "a bare number may be a version, rank, year, or label"},
            {"semantic_case": "model memory/size or capacity", "primary_class": "SIZE_OR_CAPACITY", "measurement": "yes when a size/capacity quantity is asserted", "minimum_semantic_anchor": "identified property plus value/definition", "caveat": "a configured size parameter may instead be CONFIGURATION_PROPERTY"},
            {"semantic_case": "duration, rate, frequency, range, distance", "primary_class": "DURATION_OR_LATENCY / RATE_OR_FREQUENCY / RANGE_OR_DISTANCE", "measurement": "yes when describing an observation/metric", "minimum_semantic_anchor": "identified measurable property", "caveat": "time/year or range-as-category may not be a measurement"},
            {"semantic_case": "performance is high / below requirements", "primary_class": "QUALITATIVE_PROPERTY or EVALUATIVE_JUDGMENT", "measurement": "no by itself", "minimum_semantic_anchor": "asserted qualitative predicate/comparison", "caveat": "preserve exact proposition; do not invent metric or score"},
            {"semantic_case": "metric M score equals X / improved by X", "primary_class": "NAMED_METRIC or SCORE_OR_INDEX", "measurement": "yes", "minimum_semantic_anchor": "metric identity linked to result and comparison direction where applicable", "caveat": "improvement relation must not be lost"},
            {"semantic_case": "version number, year, rank, category ID", "primary_class": "CONFIGURATION_PROPERTY / CATEGORY_OR_SCOPE / OTHER_NON_MEASUREMENT", "measurement": "not automatically", "minimum_semantic_anchor": "semantic role, not numeric shape", "caveat": "number alone is insufficient"},
            {"semantic_case": "hardware/config setting such as configured input dimensions", "primary_class": "CONFIGURATION_PROPERTY", "measurement": "not unless source asserts an observed/measured quantity", "minimum_semantic_anchor": "distinguish setting from result", "caveat": "same number/unit may occur in either role"},
            {"semantic_case": "threshold comparator without identified property", "primary_class": "THRESHOLD_CRITERION", "measurement": "no standalone Measurement identity", "minimum_semantic_anchor": "attach criterion to named measurable property", "caveat": "threshold may be an acceptance condition, not an observation"},
        ],
        "no_lexical_heuristics": True,
    }
    _write("measurement_nonmeasurement_boundary_matrix.json", boundary)

    _write(
        "representability_audit.json",
        {
            "artifact_version": "v2_1r_d4n_representability_audit.v1",
            "classification": "EXISTING_ASSERTION_SUFFICIENT",
            "basis": "The exact qualitative proposition remains preservable as source-assertion evidence through its source span. It should not be forced into MEASUREMENT; an existing generic OTHER/CONDITION context can carry a source-supported qualitative qualifier where useful.",
            "limitation": "The current SourceAssertion is evidence/status-oriented and does not expose a dedicated typed comparison predicate such as BELOW_REQUIREMENT. Thus verbatim assertion preservation is lossless, but normalized comparison semantics are not guaranteed by the current type system.",
            "new_node_type_needed": False,
            "preferred_existing_representation": ["source assertion with exact evidence span", "optional existing OTHER or CONDITION context attached to the assertion"],
            "semantic_repair_or_inference": False,
        },
    )

    options = [
        {"option": "OPTION_A", "design": "Keep free-text measurement_identity; rely on prompt only.", "expressiveness": "HIGH", "false_positive_risk": "HIGH", "false_negative_risk": "LOW", "llm_burden": "LOW", "deterministic_validation": "WEAK", "schema_complexity": "LOW", "downstream_compatibility": "HIGH", "assessment": "Does not close the observed semantic gap."},
        {"option": "OPTION_B", "design": "Add a small typed semantic discriminator for identity nature.", "expressiveness": "MEDIUM_HIGH", "false_positive_risk": "MEDIUM; classification still model-authored", "false_negative_risk": "MEDIUM", "llm_burden": "MEDIUM", "deterministic_validation": "MEDIUM for allowed values/unknown fail-closed; semantic truth remains non-deterministic", "schema_complexity": "MEDIUM", "downstream_compatibility": "MEDIUM with explicit versioning", "assessment": "Useful boundary carrier; insufficient alone without evidence linkage."},
        {"option": "OPTION_C", "design": "Require identity plus independent measurement evidence.", "expressiveness": "MEDIUM", "false_positive_risk": "LOWER", "false_negative_risk": "MEDIUM_HIGH; may reject legitimate named identity/definition contexts", "llm_burden": "MEDIUM", "deterministic_validation": "MEDIUM for field/binding presence, not entailment", "schema_complexity": "MEDIUM", "downstream_compatibility": "MEDIUM", "assessment": "Promising for observed facts; do not require numeric observation when a source-supported formal definition is the semantic fact."},
        {"option": "OPTION_D", "design": "Permit identity-only but require explicit extractor semantic classification.", "expressiveness": "HIGH", "false_positive_risk": "MEDIUM; model may misclassify", "false_negative_risk": "LOW_MEDIUM", "llm_burden": "MEDIUM_HIGH", "deterministic_validation": "MEDIUM; verifies explicit class and fail-closed unknown, not class correctness", "schema_complexity": "MEDIUM", "downstream_compatibility": "MEDIUM", "assessment": "Preserves legitimate named metrics if confirmed, but needs source evidence and conservative UNKNOWN handling."},
        {"option": "OPTION_E", "design": "Defer Measurement qualification to later semantic verification.", "expressiveness": "HIGH", "false_positive_risk": "MEDIUM until verified", "false_negative_risk": "LOW", "llm_burden": "HIGH across pipeline", "deterministic_validation": "LOW without verifier", "schema_complexity": "LOW_MEDIUM", "downstream_compatibility": "LOW_MEDIUM; changes verification boundary", "assessment": "Can be part of hybrid flow but pushes an avoidable type ambiguity downstream."},
        {"option": "OPTION_F", "design": "Use a minimal typed identity class plus source-linked semantic evidence; UNKNOWN routes to assertion/generic context.", "expressiveness": "HIGH", "false_positive_risk": "LOWER, not zero", "false_negative_risk": "MEDIUM and measurable in fresh validation", "llm_burden": "MEDIUM", "deterministic_validation": "HYBRID: deterministic shape/reference checks plus semantic classification", "schema_complexity": "MEDIUM", "downstream_compatibility": "MEDIUM with shadow-only versioned adapter", "assessment": "Best bounded direction; retain an explicit unknown/non-measurement route and do not use lexical heuristics."},
    ]
    _write("contract_option_comparison.json", {"artifact_version": "v2_1r_d4n_contract_option_comparison.v1", "options": options})

    _write(
        "threshold_review.json",
        {
            "artifact_version": "v2_1r_d4n_threshold_review.v1",
            "d4m_decision_revisited": "THRESHOLD_ALONE_IS_NOT_A_MEASUREMENT_IDENTITY",
            "reason": "A typed comparator/value/raw_text can encode a criterion but does not identify the property to which the criterion applies. It may be an acceptance condition, configuration limit, or generic inequality.",
            "future_contract_direction": "Require a source-supported measurement identity/property association for threshold and observed-value contexts; a formal definition may independently qualify when explicitly source-supported.",
            "no_metric_specific_rule": True,
        },
    )

    _write(
        "configuration_boundary_review.json",
        {
            "artifact_version": "v2_1r_d4n_configuration_boundary_review.v1",
            "principles": [
                "Parameter count and image cardinality are quantitative properties when asserted as counts.",
                "Model size/capacity and FLOPs are measurement-like when the source presents them as measurable reported properties.",
                "Input dimensions, hardware settings, model versions, years, ranks, and identifiers are not Measurements merely because they are numeric or unit-bearing.",
                "The semantic role is observation/result versus setting/configuration versus identifier/category; numeric syntax and unit dictionaries are not proof.",
                "The same field may be measurement or configuration depending on source assertion; preserve uncertainty as UNKNOWN rather than infer.",
            ],
            "examples_are_generic": True,
            "lexical_heuristics": "REJECTED",
        },
    )

    _write(
        "recommended_revision_plan.json",
        {
            "artifact_version": "v2_1r_d4n_recommended_revision_plan.v1",
            "recommended_direction": "MULTI_PART_MINIMAL_REVISION",
            "contract_vulnerability": "SYSTEMATIC_CONTRACT_VULNERABILITY",
            "observed_failure_frequency": "one frozen qualitative false-positive manifestation; no frequency/generalization claim",
            "proposed_design_work_only": [
                "Define a small semantic distinction between measurable construct/metric and qualitative/evaluative property, plus UNKNOWN; exact runtime values remain for a separate contract proposal.",
                "Require source-supported association of identity with the relevant value, threshold, or explicit formal definition where the context asserts one; do not equate a bare number, unit, or threshold with a measurement.",
                "For UNKNOWN/non-measurement, preserve the original proposition through existing assertion evidence and an optional existing generic context; do not force Measurement.",
                "Keep the typed-class declaration as model semantic input but validate field/value/reference shape deterministically; semantic class truth remains subject to human or later hybrid validation.",
                "Retain identity-only as a possible representation only when explicitly classified as a source-supported named measurable construct; fresh development evidence must test both false positives and false negatives.",
                "Carry BINDING_CONTRACT_AMBIGUITY and MODEL_OVER_ABSTENTION_WATCH unchanged.",
            ],
            "do_not_do": [
                "keyword lists, metric-name dictionaries, unit whitelists, qualitative-adjective regexes, domain-specific catalogs, Claim-specific runtime rules",
                "change frozen v3/v3.1/v3.2 implementation in this audit phase",
                "tune prompts, run models, or access Holdout",
            ],
            "confidence": "DESIGN_DIRECTION_SUPPORTED; exact schema contract requires a separate review",
        },
    )

    _write(
        "downstream_impact_audit.json",
        {
            "artifact_version": "v2_1r_d4n_downstream_impact_audit.v1",
            "grounding_requirement_v2_1": "NO_REQUIRED_INTERFACE_CHANGE_IDENTIFIED",
            "matcher_v2_1": "NO_REQUIRED_INTERFACE_CHANGE_IDENTIFIED",
            "hybrid_verifier": "NO_REQUIRED_INTERFACE_CHANGE_IDENTIFIED; semantic classification can remain shadow-side until separately authorized",
            "verification_assertion": "NO_REQUIRED_INTERFACE_CHANGE_IDENTIFIED",
            "reason": "A future versioned extraction contract can keep existing assertion/context/binding output and fail closed before downstream projection. No downstream consumer requires a new node type for this audit's proposed boundary.",
            "attachment_watch": "BINDING_CONTRACT_AMBIGUITY carried unchanged",
            "model_over_abstention_watch": "carried unchanged; no prompt tuning",
            "production_or_downstream_changes_made": False,
        },
    )

    _write(
        "audit_summary.json",
        {
            "artifact_version": "v2_1r_d4n_audit_summary.v1",
            "ledger_context_count": len(ledger),
            "exact_v31_measurement_identity_count": len(exact_identity_rows),
            "valid_identity_only_measurement_count": identity_only_audit["valid_identity_only_measurement_count"],
            "false_identity_only_measurement_count": identity_only_audit["false_identity_only_measurement_count"],
            "ambiguous_identity_only_count": identity_only_audit["ambiguous_identity_only_count"],
            "d4m_false_positive_count": 1,
            "measurement_identity_boundary": "A source-grounded semantic designation of a property whose observation, comparison, threshold, or formal measurement definition is meaningful; not every evaluative property is measurable merely because the identity field is populated.",
            "structural_validation_limit": "REQUIRES_HYBRID_VALIDATION",
            "recommended_revision": "MULTI_PART_MINIMAL_REVISION",
            "systematicity": "The schema permits a class of false positives; one observed manifestation does not estimate frequency.",
            "provider_calls": 0,
            "external_http": 0,
            "extractor_calls": 0,
            "verifier_calls": 0,
            "holdout_access": 0,
        },
    )

    final_md = """# V2.1R D4-N — Measurement Identity Semantic Boundary Audit

## Final decision

**MEASUREMENT IDENTITY CONTRACT REQUIRES MULTI-PART REVISION**

## Finding

The generic semantic definition—an identity names a property for which an observation, comparison, threshold, or formal measurement definition is meaningful—covers the accepted development examples conceptually. But the current free-text `measurement_identity` field cannot enforce that distinction: a qualitative/evaluative property can occupy the field and pass the v3.2 structural anchor check. This is a systematic contract vulnerability, not evidence of a high observed failure rate; the frozen set contains one clear manifestation.

## Frozen identity-only audit

The de-duplicated D4-G/D4-J ledger contains 18 proposed Measurement contexts. Thirteen use the exact `measurement_identity` field; three earlier D4-G metric identities use the v3 field alias `measurement`. Among exact v3.1 identity-only contexts with no observed value, formal definition, or threshold: 0 were human-confirmed valid, 1 was a human-confirmed qualitative false positive, and 0 were ambiguous. The three D4-G alias-only contexts were human-rated major/incomplete because typed observations were omitted, so they are tracked separately and not counted as accepted identity-only cases. The small cohort cannot prove that legitimate identity-only representations are never needed.

## False-positive representation

The D4-J qualitative performance comparison should remain an asserted source proposition, not a Measurement. Existing assertion evidence can preserve the exact source span without adding a node type. A generic `OTHER`/`CONDITION` context may carry a source-supported qualitative qualifier, but the current graph does not provide a dedicated typed “below requirement” comparison predicate; normalized comparison semantics therefore remain a limitation.

## Boundary conclusions

- A number is not automatically a measurement: it may be a year, version, rank, category label, identifier, or configuration value.
- Units can strengthen a measurement interpretation but cannot prove identity; no unit whitelist is justified.
- A source-supported formal measurement definition may independently qualify.
- A threshold does not identify what property it constrains; D4-M's threshold-alone anchor is not semantically sufficient. Attach it to an identified measurable property.
- Qualitative “high/below requirements” differs from a named score/result or a metric change; preserve the source proposition without inventing a metric.
- Free-text identity validity is not structurally decidable. A future version needs hybrid validation: explicit semantic classification and source-linked structure, deterministic shape/reference checks, and UNKNOWN fail-closed handling.

## Recommended next revision

`MULTI_PART_MINIMAL_REVISION`: design a small typed distinction between measurable construct/metric, qualitative/evaluative property, and UNKNOWN; associate values/thresholds/definitions with a source-supported property; keep non-measurement propositions in existing assertion/context representations. Exact runtime schema remains for a separate design review. No lexical heuristics, Claim-specific rules, implementation, prompt tuning, Provider call, downstream change, or Holdout access occurred in D4-N.

## Scope and artifacts

Primary ledger sources are frozen D4-G v3 and D4-J v3.1 development outputs and their human reviews. D4-K/L/M are cross-checks, deduplicated. D4-B/C v2 results are listed only as legacy background and excluded from v3/v3.1 counts. See the JSON artifacts in this directory and `hash_manifest.json`.

Provider calls: 0 · External HTTP: 0 · Extractor: 0 · Verifier: 0 · Holdout access: 0.

STOP: no implementation, replay, broader validation, Holdout, V2.2, stage, or commit.
"""
    (OUTPUT / "final_report.md").write_text(final_md, encoding="utf-8")

    output_files = [p for p in sorted(OUTPUT.iterdir()) if p.is_file() and p.name != "hash_manifest.json"]
    manifest = {
        "artifact_version": "v2_1r_d4n_hash_manifest.v1",
        "generated_artifact_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): _sha(path) for path in output_files
        },
        "analysis_script": {
            "path": "scripts/analyze_v2_1r_d4n_measurement_identity.py",
            "sha256": _sha(Path(__file__)),
        },
        "primary_frozen_source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): _sha(path)
            for path in evidence_paths
            if path.exists()
        },
        "holdout_access": 0,
        "provider_calls": 0,
        "external_http": 0,
    }
    _write("hash_manifest.json", manifest)
    print(
        json.dumps(
            {
                "output_directory": str(OUTPUT),
                "ledger_context_count": len(ledger),
                "exact_identity_field_count": len(exact_identity_rows),
                "identity_only": {
                    "valid": identity_only_audit["valid_identity_only_measurement_count"],
                    "false": identity_only_audit["false_identity_only_measurement_count"],
                    "ambiguous": identity_only_audit["ambiguous_identity_only_count"],
                },
                "recommended_revision": "MULTI_PART_MINIMAL_REVISION",
                "final_decision": "MEASUREMENT IDENTITY CONTRACT REQUIRES MULTI-PART REVISION",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
