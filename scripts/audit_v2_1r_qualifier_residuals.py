"""Create a deterministic, offline root-cause ledger for qualifier warnings."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "api"))

from app.services.atomic_proposition_validation import (  # noqa: E402
    _proposition_semantic_view,
    inspect_qualifier_fields,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _status_map(items: list[dict[str, Any]]) -> dict[str, str]:
    return {str(item["category"]): str(item["status"]) for item in items}


def build_ledger(input_dir: Path, baseline_path: Path) -> dict[str, Any]:
    baseline = _read(baseline_path)
    classifications = baseline.get("qualifier", {}).get("event_classifications", [])
    manual_by_key = {
        (str(item["claim_id"]), str(item["category"])): item
        for item in classifications
    }
    records = {item["claim_id"]: item for item in baseline.get("claim_records", [])}
    paths = sorted(input_dir.glob("*.json"))
    if len(paths) != 100:
        raise ValueError(f"expected the frozen 100 diagnostics; found {len(paths)}")

    ledger: list[dict[str, Any]] = []
    claim_states: dict[str, dict[str, Any]] = {}
    provider_calls = 0
    for path in paths:
        artifact = _read(path)
        claim = artifact["claim"]
        decomposition = artifact["decomposition"]
        model_result = decomposition["model_structured_result"]
        provider_calls += int(decomposition.get("model_call_count", 0))
        propositions = model_result.get("propositions", [])
        old_checks = decomposition.get("validator_result", {}).get("qualifier_checks", [])
        old = _status_map(old_checks)
        parser_status = decomposition.get("parser_result", {}).get("status")
        validator_status = decomposition.get("validator_result", {}).get("status")
        final_status = decomposition.get("final_shadow_result", {}).get("status")
        if (
            parser_status == "PARSED"
            and validator_status == "VALID"
            and propositions
            and final_status
            not in {
                "SCHEMA_INVALID",
                "COMPOSITION_REFERENCE_INVALID",
                "COMPOSITION_SEMANTIC_INVALID",
                "SOURCE_SPAN_INVALID",
            }
        ):
            checks = inspect_qualifier_fields(claim["claim_text"], propositions)
        else:
            checks = ()
        claim_id = str(claim["claim_id"])
        claim_record = records.get(claim_id, {})
        old_final = claim_record.get(
            "final_shadow_status", decomposition.get("final_shadow_result", {}).get("status")
        )
        old_claim_warning = any(
            value in {"POTENTIAL_LOSS", "AMBIGUOUS"} for value in old.values()
        )
        new_claim_warning = any(
            item.status.value in {"POTENTIAL_LOSS", "AMBIGUOUS"} for item in checks
        )
        claim_states[claim_id] = {
            "old_final": old_final,
            "old_claim_warning": old_claim_warning,
            "new_claim_warning": new_claim_warning,
        }

        for check in checks:
            if check.status.value not in {"POTENTIAL_LOSS", "AMBIGUOUS"}:
                continue
            manual = manual_by_key.get((claim_id, check.category))
            evidence_rows = []
            for evidence in check.preservation_evidence:
                obligation = evidence.obligation.model_dump(mode="json")
                index = evidence.proposition_index
                views = [
                    _proposition_semantic_view(candidate_index, proposition)
                    for candidate_index, proposition in enumerate(propositions)
                ]
                if index is not None:
                    candidate_indexes = [index]
                else:
                    normalized_source = obligation["normalized_value"]
                    candidate_indexes = [
                        candidate_index
                        for candidate_index, view in enumerate(views)
                        if any(
                            normalized_source in " ".join(span.casefold().split())
                            for span in view.source_spans
                        )
                    ]
                    if not candidate_indexes:
                        candidate_indexes = [view.proposition_index for view in views]
                candidate_rows = []
                for candidate_index in candidate_indexes:
                    candidate = propositions[candidate_index]
                    candidate_view = views[candidate_index]
                    candidate_rows.append(
                        {
                            "local_ref": candidate.get("local_ref"),
                            "proposition_text": candidate.get("proposition_text"),
                            "source_span": candidate.get("source_span"),
                            "semantic_carriers": [
                                item.model_dump(mode="json") for item in candidate_view.carriers
                            ],
                        }
                    )
                proposition = propositions[index] if index is not None else None
                evidence_rows.append(
                    {
                        "source_obligation": obligation,
                        "normalized_obligation_payload": obligation["normalized_value"],
                        "source_detection_evidence": obligation["detection_evidence"],
                        "source_span": obligation["source_text"],
                        "materiality": obligation["materiality"],
                        "attachment_scope": obligation["attachment_context"],
                        "candidate_proposition_ref": (
                            proposition.get("local_ref") if proposition else None
                        ),
                        "candidate_proposition_index": index,
                        "candidate_propositions_considered": candidate_rows,
                        "semantic_carriers_found": [
                            carrier
                            for candidate in candidate_rows
                            for carrier in candidate["semantic_carriers"]
                        ],
                        "carrier_locations": list(evidence.carrier_paths),
                        "carrier_evidence_strength": evidence.evidence_strength.value,
                        "semantic_compatible": evidence.semantic_compatible,
                        "attachment_compatible": evidence.attachment_compatible,
                        "attachment_decision": evidence.reason_code,
                        "reason_code": evidence.reason_code,
                    }
                )
            old_status = old.get(check.category, "NOT_RECORDED")
            issue_codes = claim_record.get(
                "issue_codes", decomposition.get("validation_issue_codes", [])
            )
            human_classification = (
                manual.get("manual_classification", "NOT_PREVIOUSLY_HUMAN_REVIEWED")
                if manual
                else "NOT_PREVIOUSLY_HUMAN_REVIEWED"
            )
            ledger.append(
                {
                    "claim_id": claim_id,
                    "artifact": path.name,
                    "qualifier_category": check.category,
                    "cohort": "TO_BE_ASSIGNED_AT_CLAIM_LEVEL",
                    "old_validator_status": old_status,
                    "new_validator_status": check.status.value,
                    "source_applicability": check.applicability.value,
                    "source_reason_codes": list(check.reason_codes),
                    "source_obligations_and_candidate_evidence": evidence_rows,
                    "claim_final_status_before": old_final,
                    "claim_level_blocking_effect": {
                        "blocks_under_new_qualifier_validator": True,
                        "independent_issue_codes": list(issue_codes),
                    },
                    "human_semantic_classification": human_classification,
                    "historical_review_note": manual.get("review_note") if manual else None,
                    "reason_codes": list(check.reason_codes),
                }
            )

    if provider_calls != 100:
        raise ValueError(
            "expected 100 persisted provider calls in source artifacts, "
            f"got {provider_calls}"
        )
    for item in ledger:
        state = claim_states[item["claim_id"]]
        if state["old_final"] == "RESOLVED" and not state["old_claim_warning"]:
            item["cohort"] = "NEWLY_BLOCKED"
        elif state["old_claim_warning"] and state["new_claim_warning"]:
            item["cohort"] = "PERSISTENTLY_BLOCKED"
        else:
            item["cohort"] = "OTHER_BLOCKED"
    cohorts = Counter(item["cohort"] for item in ledger)
    warning_claim_counts: Counter[str] = Counter()
    for state in claim_states.values():
        if not state["new_claim_warning"]:
            continue
        if state["old_final"] == "RESOLVED" and not state["old_claim_warning"]:
            warning_claim_counts["NEWLY_BLOCKED"] += 1
        elif state["old_claim_warning"]:
            warning_claim_counts["PERSISTENTLY_BLOCKED"] += 1
        else:
            warning_claim_counts["OTHER_BLOCKED"] += 1
    if (
        len(ledger) != 22
        or warning_claim_counts["NEWLY_BLOCKED"] != 4
        or warning_claim_counts["PERSISTENTLY_BLOCKED"] != 15
    ):
        raise ValueError(
            "unexpected warning ledger/claim cohorts: "
            f"{len(ledger)} events, {dict(cohorts)} event cohorts, "
            f"{dict(warning_claim_counts)} claim cohorts"
        )

    warning_claims = {item["claim_id"] for item in ledger}
    newly_blocked_ids = sorted(
        {item["claim_id"] for item in ledger if item["cohort"] == "NEWLY_BLOCKED"}
    )
    persistent_claim_ids = sorted(
        {item["claim_id"] for item in ledger if item["cohort"] == "PERSISTENTLY_BLOCKED"}
    )
    historical_classes = Counter(
        item.get("manual_classification", "UNCLASSIFIED") for item in classifications
    )
    canonical_classes = Counter(
        item.get("manual_classification", "UNCLASSIFIED") for item in classifications
    )
    return {
        "audit_version": "v2.1r-qualifier-residual-warning-ledger-v1",
        "provider_calls_during_audit": 0,
        "persisted_provider_calls_in_inputs": provider_calls,
        "warning_count": len(ledger),
        "warning_claim_count": len(warning_claims),
        "cohort_warning_event_counts": dict(cohorts),
        "cohort_warning_claim_counts": dict(warning_claim_counts),
        "newly_blocked_claim_ids": newly_blocked_ids,
        "persistent_blocked_claim_ids": persistent_claim_ids,
        "ledger": ledger,
        "baseline_reconciliation": {
            "historical_summary_counts": {
                "STRUCTURED_EQUIVALENCE_MISSED": 30,
                "SOURCE_DETECTOR_FALSE_POSITIVE": 2,
                "APPLICABILITY_FALSE_POSITIVE": 3,
                "TRUE_ATTACHMENT_AMBIGUITY": 1,
            },
            "item_level_reconstructed_counts": dict(canonical_classes),
            "recorded_historical_item_level_counts": dict(historical_classes),
            "canonical_for_future_diagnostics": "item_level_reconstructed_counts",
            "historical_artifacts_modified": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "artifacts/v2_1r_bounded_shadow_100/canary_diagnostics",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "artifacts/v2_1r_bounded_shadow_100_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/v2_1r_qualifier_revision2_warning_ledger.json",
    )
    parser.add_argument(
        "--reconciliation",
        type=Path,
        default=ROOT / "artifacts/v2_1r_qualifier_revision2_baseline_reconciliation.json",
    )
    args = parser.parse_args()
    result = build_ledger(args.input_dir, args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.reconciliation.write_text(
        json.dumps(result["baseline_reconciliation"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "warning_count": result["warning_count"],
                "warning_claim_count": result["warning_claim_count"],
                "cohorts": result["cohort_warning_event_counts"],
                "provider_calls_during_audit": 0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
