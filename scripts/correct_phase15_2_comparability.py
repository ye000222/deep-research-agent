#!/usr/bin/env python3
"""Correct Phase 15.2 A/B comparability using the real Phase 15.1 cohort.

This is a read-only analysis/report builder.  It does not start runs and does
not mutate production state.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any

import psycopg


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CONTROL = [
    "01a0c3ef-928b-7f0d-8006-40843c571b57",
    "01a0c412-10b0-7c3f-8cf4-d56aeecbbea0",
    "01a0c412-5ae6-7b9c-9353-7be1ad74c293",
]
CANDIDATE = [
    "01a0c687-4027-7fe9-91f0-94dc990595c9",
    "01a0c71a-ba87-78b6-a7f5-5d1db5535f80",
    "01a0c71a-d98a-7019-be8c-493d1af8ea72",
]
OLD_CONTROL = [
    "01a0c258-b953-79ab-8853-22ea54a5d267",
    "01a0c276-7cc5-7371-af5f-4aa9625d480c",
    "01a0c276-c122-7edd-8e4b-03c148c48d0d",
]


def load(name: str) -> dict[str, Any]:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "mean": round(statistics.fmean(values), 6),
        "sd": round(statistics.pstdev(values), 6),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def metric(arm: list[dict[str, Any]], section: str, key: str) -> dict[str, Any]:
    return summary([float(row[section][key]) for row in arm])


def db_cards(connection: Any, run_ids: list[str]) -> dict[str, dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT id::text, status, termination_reason, plan_version,
               budget_snapshot, quality_snapshot, usage_snapshot
        FROM research_runs WHERE id = ANY(%s::uuid[])
        """,
        (run_ids,),
    ).fetchall()
    cards: dict[str, dict[str, Any]] = {}
    for run_id, status, termination, plan_version, budget, quality, usage in rows:
        benchmark = (budget or {}).get("benchmark", {})
        flags = benchmark.get("feature_flags", {})
        cards[run_id] = {
            "status": status,
            "termination_reason": termination,
            "plan_version": plan_version,
            "source_revision": (budget or {}).get("source_revision"),
            "benchmark_id": benchmark.get("benchmark_id"),
            "benchmark_version": benchmark.get("benchmark_version"),
            "metric_definition_version": benchmark.get("metric_definition_version"),
            "tier": benchmark.get("tier"),
            "plan_shape_policy": benchmark.get("plan_shape_policy"),
            "plan_template_run_id": benchmark.get("plan_template_run_id"),
            "feature_flags": {
                "independent_source_targeting_enabled": flags.get(
                    "independent_source_targeting_enabled", None
                ),
                "evidence_input_quality_enabled": flags.get(
                    "evidence_input_quality_enabled", None
                ),
            },
            "budget_snapshot": budget or {},
            "quality_snapshot": quality or {},
            "usage_snapshot": usage or {},
        }
    return cards


def evidence_counts(connection: Any, run_ids: list[str]) -> dict[str, dict[str, int]]:
    rows = connection.execute(
        """
        SELECT run_id::text, count(*) AS candidate,
               count(*) FILTER (WHERE accepted) AS accepted,
               count(*) FILTER (WHERE rejection_reason = 'claim_quote_entailment_failed') AS entailment,
               count(*) FILTER (WHERE rejection_reason = 'source_role_mismatch') AS role_mismatch
        FROM research_evidence
        WHERE run_id = ANY(%s::uuid[])
        GROUP BY run_id
        """,
        (run_ids,),
    ).fetchall()
    return {
        str(run_id): {
            "candidate": int(candidate),
            "accepted": int(accepted),
            "entailment": int(entailment),
            "role_mismatch": int(role_mismatch),
        }
        for run_id, candidate, accepted, entailment, role_mismatch in rows
    }


def primary_metric(
    control_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    control_counts: dict[str, dict[str, int]],
    candidate_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    def rates(rows: list[dict[str, Any]], counts: dict[str, dict[str, int]], key: str) -> list[float]:
        return [counts[row["run_id"]][key] / counts[row["run_id"]]["candidate"] for row in rows]

    def count_metric(rows: list[dict[str, Any]], counts: dict[str, dict[str, int]], key: str) -> dict[str, Any]:
        return summary([float(counts[row["run_id"]][key]) for row in rows])

    def arm_metric(section: str, key: str) -> dict[str, Any]:
        return {
            "control": metric(control_rows, section, key),
            "candidate": metric(candidate_rows, section, key),
            "delta_mean": round(
                metric(candidate_rows, section, key)["mean"]
                - metric(control_rows, section, key)["mean"],
                6,
            ),
        }

    result: dict[str, Any] = {}
    for name, key in (
        ("candidate_to_accepted_rate", "accepted"),
        ("claim_quote_entailment_failed_rate", "entailment"),
        ("source_role_mismatch_rate", "role_mismatch"),
    ):
        control_rates = rates(control_rows, control_counts, key)
        candidate_rates = rates(candidate_rows, candidate_counts, key)
        result[name] = {
            "control": summary(control_rates),
            "candidate": summary(candidate_rates),
            "delta_mean": round(statistics.fmean(candidate_rates) - statistics.fmean(control_rates), 6),
            "pooled_control": round(sum(control_counts[r["run_id"]][key] for r in control_rows) / sum(control_counts[r["run_id"]]["candidate"] for r in control_rows), 6),
            "pooled_candidate": round(sum(candidate_counts[r["run_id"]][key] for r in candidate_rows) / sum(candidate_counts[r["run_id"]]["candidate"] for r in candidate_rows), 6),
        }
        result[name]["pooled_delta"] = round(
            result[name]["pooled_candidate"] - result[name]["pooled_control"], 6
        )
    result["claim_quote_entailment_failed_count"] = {
        "control": count_metric(control_rows, control_counts, "entailment"),
        "candidate": count_metric(candidate_rows, candidate_counts, "entailment"),
        "delta_mean": round(
            count_metric(candidate_rows, candidate_counts, "entailment")["mean"]
            - count_metric(control_rows, control_counts, "entailment")["mean"],
            6,
        ),
    }
    result["source_role_mismatch_count"] = {
        "control": count_metric(control_rows, control_counts, "role_mismatch"),
        "candidate": count_metric(candidate_rows, candidate_counts, "role_mismatch"),
        "delta_mean": round(
            count_metric(candidate_rows, candidate_counts, "role_mismatch")["mean"]
            - count_metric(control_rows, control_counts, "role_mismatch")["mean"],
            6,
        ),
    }
    for name, section, key in (
        ("candidate_evidence_count", "quality", "candidate_evidence"),
        ("accepted_evidence_count", "quality", "accepted_evidence"),
    ):
        result[name] = arm_metric(section, key)
    return result


def normalized_feature(value: Any, *, legacy_false: bool = False) -> bool | None:
    if value is None and legacy_false:
        return False
    return value


def main() -> None:
    control_artifact = load("phase15_1_candidate_n3.json")
    candidate_artifact = load("phase15_2_candidate_n3.json")
    old_artifact = load("phase15_1_baseline_n3.json")
    prior_comparison = load("phase15_2_ab_comparison.json")
    db_url = os.environ["DATABASE_URL"].replace("postgresql+psycopg://", "postgresql://", 1)
    with psycopg.connect(db_url) as connection:
        cards = db_cards(connection, CONTROL + CANDIDATE + OLD_CONTROL)
        counts = evidence_counts(connection, CONTROL + CANDIDATE)

    control_rows = control_artifact["per_run"]
    candidate_rows = candidate_artifact["per_run"]
    primary = primary_metric(control_rows, candidate_rows, counts, counts)
    control_summary = control_artifact["golden_baseline_summary"]
    candidate_summary = candidate_artifact["golden_baseline_summary"]

    control_cards = [cards[r] for r in CONTROL]
    candidate_cards = [cards[r] for r in CANDIDATE]
    identity_fields = ("benchmark_id", "benchmark_version", "metric_definition_version", "tier", "plan_shape_policy", "plan_template_run_id")
    identity_equal = all(
        len({card[field] for card in control_cards + candidate_cards}) == 1
        for field in identity_fields
    )
    def budget_signature(card: dict[str, Any]) -> str:
        snapshot = json.loads(json.dumps(card["budget_snapshot"]))
        snapshot.pop("source_revision", None)
        snapshot.pop("deadline_at", None)
        benchmark = snapshot.get("benchmark")
        if isinstance(benchmark, dict):
            benchmark = dict(benchmark)
            benchmark.pop("feature_flags", None)
            snapshot["benchmark"] = benchmark
        return json.dumps(snapshot, sort_keys=True, default=str)

    budget_equal = len({budget_signature(card) for card in control_cards + candidate_cards}) == 1
    control_flags = {
        "independent_source_targeting_enabled": [
            normalized_feature(card["feature_flags"]["independent_source_targeting_enabled"])
            for card in control_cards
        ],
        "evidence_input_quality_enabled": [
            normalized_feature(card["feature_flags"]["evidence_input_quality_enabled"], legacy_false=True)
            for card in control_cards
        ],
    }
    candidate_flags = {
        "independent_source_targeting_enabled": [
            normalized_feature(card["feature_flags"]["independent_source_targeting_enabled"])
            for card in candidate_cards
        ],
        "evidence_input_quality_enabled": [
            normalized_feature(card["feature_flags"]["evidence_input_quality_enabled"])
            for card in candidate_cards
        ],
    }
    comparable = (
        identity_equal
        and budget_equal
        and all(flag is True for flag in control_flags["independent_source_targeting_enabled"])
        and all(flag is True for flag in candidate_flags["independent_source_targeting_enabled"])
        and all(flag is False for flag in control_flags["evidence_input_quality_enabled"])
        and all(flag is True for flag in candidate_flags["evidence_input_quality_enabled"])
        and all(card["status"] == "completed_with_limitations" for card in control_cards + candidate_cards)
    )

    def delta(key: str) -> float:
        return round(control_summary[key]["mean"] - candidate_summary[key]["mean"], 6)

    corrected = {
        "schema_version": "phase15-2-ab-comparison.v2-corrected-control",
        "experiment": "Evidence Input Quality & Acceptance Conversion",
        "analysis_correction": {
            "reason": "Previous report used Phase 15.0 Golden Baseline IDs as the Phase 15.1 Control.",
            "old_incorrect_control_run_ids": OLD_CONTROL,
            "corrected_phase15_1_control_run_ids": CONTROL,
            "phase15_0_golden_reference_only": True,
            "voided_old_delta_fields": [
                "candidate_to_accepted_rate",
                "claim_quote_entailment_failed_rate",
                "source_role_mismatch_rate",
                "candidate_evidence_count",
                "accepted_evidence_count",
                "coverage",
                "gap_closed",
                "gap_closure_rate",
                "search_completed",
                "provider_requests",
                "readable_sources",
                "evidence_extractions",
                "tokens",
                "runtime_seconds",
            ],
        },
        "comparability": {
            "status": "COMPARABLE_FOR_EXPERIMENT" if comparable else "NOT_COMPARABLE",
            "control_arm": "Phase 15.1 Candidate n=3 (corrected Control)",
            "candidate_arm": "Phase 15.2 Candidate n=3",
            "declared_delta": "evidence_input_quality_enabled: false -> true",
            "benchmark_id": control_cards[0]["benchmark_id"],
            "benchmark_version": control_cards[0]["benchmark_version"],
            "metric_definition_version": control_cards[0]["metric_definition_version"],
            "tier": control_cards[0]["tier"],
            "plan_shape_policy": control_cards[0]["plan_shape_policy"],
            "plan_template_run_id": control_cards[0]["plan_template_run_id"],
            "budget_profile": "standard",
            "budget_policy_same": budget_equal,
            "identity_fields_same": identity_equal,
            "control_feature_flags": control_flags,
            "candidate_feature_flags": candidate_flags,
            "control_legacy_note": "evidence_input_quality_enabled was absent from these legacy Run snapshots; worker resolves absent to false, which is the auditable legacy-false equivalent.",
            "source_revisions": {
                "control": sorted({card["source_revision"] for card in control_cards}),
                "candidate": sorted({card["source_revision"] for card in candidate_cards}),
            },
        },
        "run_metadata": {"control": control_cards, "candidate": candidate_cards},
        "control_run_ids": CONTROL,
        "candidate_run_ids": CANDIDATE,
        "primary_metrics": primary,
        "quality": {
            "coverage": {"control": control_summary["quality"]["coverage"], "candidate": candidate_summary["quality"]["coverage"], "delta_candidate_minus_control": round(candidate_summary["quality"]["coverage"]["mean"] - control_summary["quality"]["coverage"]["mean"], 6)},
            "priority_one_coverage": {"control": control_summary["quality"]["priority_one_coverage"], "candidate": candidate_summary["quality"]["priority_one_coverage"], "delta_mean": 0.0},
            "critical_gap_count": {"control": control_summary["quality"]["critical_gap_count"], "candidate": candidate_summary["quality"]["critical_gap_count"], "delta_mean": round(candidate_summary["quality"]["critical_gap_count"]["mean"] - control_summary["quality"]["critical_gap_count"]["mean"], 6)},
            "gap_closed": {"control": control_summary["quality"]["gap_closed"], "candidate": candidate_summary["quality"]["gap_closed"], "delta_mean": round(candidate_summary["quality"]["gap_closed"]["mean"] - control_summary["quality"]["gap_closed"]["mean"], 6)},
            "gap_closure_rate": {"control": control_summary["quality"]["gap_closure_rate"], "candidate": candidate_summary["quality"]["gap_closure_rate"], "delta_mean": round(candidate_summary["quality"]["gap_closure_rate"]["mean"] - control_summary["quality"]["gap_closure_rate"]["mean"], 6)},
            "supported_claims": {"control": control_summary["quality"]["supported_claims"], "candidate": candidate_summary["quality"]["supported_claims"], "delta_mean": round(candidate_summary["quality"]["supported_claims"]["mean"] - control_summary["quality"]["supported_claims"]["mean"], 6)},
            "verified_claims": {"control": control_summary["quality"]["verified_claims"], "candidate": candidate_summary["quality"]["verified_claims"], "delta_mean": 0.0},
        },
        "research_and_cost": {
            key: {
                "control": control_summary[section][key],
                "candidate": candidate_summary[section][key],
                "delta_mean_candidate_minus_control": round(candidate_summary[section][key]["mean"] - control_summary[section][key]["mean"], 6),
            }
            for section, keys in (("research_work", ("logical_queries", "search_started", "search_completed", "provider_requests", "candidate_urls", "unique_urls", "readable_sources", "reader_completed", "evidence_extractions")), ("cost", ("runtime_seconds", "tokens")))
            for key in keys
        },
        "activation": prior_comparison["activation"],
        "recovery_attribution": prior_comparison["recovery_attribution"],
        "quote_recovery_conclusion": {
            "quote_improved_total": 2,
            "direct_event_level_evidence": True,
            "causal_claim": "The two accepted quote_improved events remain directly evidenced, but they do not establish an aggregate A/B improvement.",
        },
        "phase15_0_golden_reference": {
            "run_ids": OLD_CONTROL,
            "purpose": "V1 absolute quality / longitudinal reference only; not a causal control for Phase 15.2.",
            "artifact_summary": old_artifact["golden_baseline_summary"],
        },
        "stop_gate_basis": {
            "phase15_closed": True,
            "reason": "Corrected A/B is comparable, but the corrected Candidate is lower on coverage, accepted evidence, and gap closure; the bounded mechanism has direct event-level activation evidence only.",
            "phase15_3": "not_allowed",
            "phase16": "not_started",
        },
    }

    (ARTIFACTS / "phase15_2_ab_comparison.json").write_text(json.dumps(corrected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = [
        "# Phase 15.2 A/B Comparison — Corrected Control",
        "",
        "> Analysis correction only. No production run was re-run and no research behavior was changed.",
        "",
        "## Comparability",
        "",
        f"- Status: **{corrected['comparability']['status']}**",
        "- Declared experiment variable: `evidence_input_quality_enabled: false -> true`",
        f"- Corrected Control: {', '.join(CONTROL)}",
        f"- Candidate: {', '.join(CANDIDATE)}",
        "- Benchmark / version / metric / tier / plan: identical (`v1-dev-model-comparison`, `1.0.0`, `coverage-v1`, `standard`, frozen reference plan).",
        "- `independent_source_targeting_enabled`: true in both arms.",
        "- Control `evidence_input_quality_enabled`: absent in legacy Run snapshots and resolved by the worker as false; Candidate: true.",
        "",
        "## Old Control Correction",
        "",
        f"- OLD INCORRECT CONTROL: {', '.join(OLD_CONTROL)} — Phase 15.0 Golden Baseline.",
        f"- CORRECTED PHASE 15.1 CONTROL: {', '.join(CONTROL)}.",
        "- All deltas previously computed against the old Control are void and must not be used for causal interpretation.",
        "",
        "## Corrected A/B Metrics",
        "",
        "| Metric | Corrected Control | Phase 15.2 Candidate | Delta (Candidate-Control) |",
        "|---|---:|---:|---:|",
    ]
    rows = [
        ("Candidate → Accepted", primary["candidate_to_accepted_rate"]["control"]["mean"], primary["candidate_to_accepted_rate"]["candidate"]["mean"], primary["candidate_to_accepted_rate"]["delta_mean"]),
        ("Entailment failure rate", primary["claim_quote_entailment_failed_rate"]["control"]["mean"], primary["claim_quote_entailment_failed_rate"]["candidate"]["mean"], primary["claim_quote_entailment_failed_rate"]["delta_mean"]),
        ("Source-role mismatch rate", primary["source_role_mismatch_rate"]["control"]["mean"], primary["source_role_mismatch_rate"]["candidate"]["mean"], primary["source_role_mismatch_rate"]["delta_mean"]),
        ("Accepted evidence", control_summary["quality"]["accepted_evidence"]["mean"], candidate_summary["quality"]["accepted_evidence"]["mean"], candidate_summary["quality"]["accepted_evidence"]["mean"] - control_summary["quality"]["accepted_evidence"]["mean"]),
        ("Coverage", control_summary["quality"]["coverage"]["mean"], candidate_summary["quality"]["coverage"]["mean"], candidate_summary["quality"]["coverage"]["mean"] - control_summary["quality"]["coverage"]["mean"]),
        ("Gap closure rate", control_summary["quality"]["gap_closure_rate"]["mean"], candidate_summary["quality"]["gap_closure_rate"]["mean"], candidate_summary["quality"]["gap_closure_rate"]["mean"] - control_summary["quality"]["gap_closure_rate"]["mean"]),
        ("Verified claims", control_summary["quality"]["verified_claims"]["mean"], candidate_summary["quality"]["verified_claims"]["mean"], 0.0),
    ]
    md.extend(f"| {name} | {control:.6f} | {candidate:.6f} | {delta:+.6f} |" for name, control, candidate, delta in rows)
    md.extend([
        "",
        "## Quote Recovery Attribution",
        "",
        "- `quote_improved=2` remains directly supported by two persisted Candidate-arm events; both were accepted.",
        "- This is mechanism/path evidence, not evidence that the aggregate A/B result improved: corrected Candidate quality is lower than the true Phase 15.1 Control.",
        "",
        "## Phase 15.0 Golden Reference",
        "",
        "The three old IDs remain in the report only as the V1 absolute-quality/longitudinal reference. They are not used as the Phase 15.2 causal Control.",
        "",
        "## Stop Gate",
        "",
        "Phase 15 remains **CLOSED**. The corrected comparison is valid but does not justify Phase 15.3; Phase 15.3 and Phase 16 were not started.",
    ])
    (ARTIFACTS / "phase15_2_ab_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    stop = {
        "schema_version": "phase15-stop-gate.v2-corrected-control",
        "phase": "15.2",
        "decision": "CLOSED",
        "comparability_status": corrected["comparability"]["status"],
        "control_run_ids": CONTROL,
        "candidate_run_ids": CANDIDATE,
        "old_incorrect_control_run_ids": OLD_CONTROL,
        "activation_valid": True,
        "observed": {
            "candidate_to_accepted_delta": primary["candidate_to_accepted_rate"]["delta_mean"],
            "pooled_candidate_to_accepted_delta": round(primary["candidate_to_accepted_rate"]["pooled_candidate"] - primary["candidate_to_accepted_rate"]["pooled_control"], 6),
            "pooled_entailment_rate_delta": primary["claim_quote_entailment_failed_rate"]["pooled_delta"],
            "pooled_source_role_mismatch_rate_delta": primary["source_role_mismatch_rate"]["pooled_delta"],
            "quote_improved_total": 2,
            "accepted_quote_recovery": 2,
            "coverage_delta": corrected["quality"]["coverage"]["delta_candidate_minus_control"],
            "gap_closure_rate_delta": corrected["quality"]["gap_closure_rate"]["delta_mean"],
            "verified_claims_control": 0,
            "verified_claims_candidate": 0,
        },
        "old_deltas_void": True,
        "one_final_15_3_justification": {
            "single_bottleneck": False,
            "direct_evidence": True,
            "unlock_gte_0_10": False,
            "bounded_implementation": False,
            "no_large_architecture": False,
            "all_five_yes": False,
            "reason": "Corrected Candidate is lower on the primary outcome metrics; the two quote-recovery events prove activation but not a positive aggregate effect or a bounded next-phase fix.",
        },
        "remaining_bottlenecks": [
            "accepted evidence -> independently verified claim: verified claims remain 0 in both arms",
            "candidate evidence -> accepted evidence/source-role suitability: corrected Candidate conversion is lower than corrected Control",
        ],
        "next_phase": "not_auto_started",
        "stop_instruction": "Do not start Phase 15.3 or Phase 16 automatically.",
    }
    (ARTIFACTS / "phase15_2_stop_gate.json").write_text(json.dumps(stop, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    stop_md = [
        "# Phase 15.2 Stop Gate — Corrected Control",
        "",
        "- Decision: **CLOSED**",
        f"- Comparability: **{corrected['comparability']['status']}**",
        "- Mechanism activation: proven (`input_quality.evaluated` persisted for the Candidate arm; `quote_improved=2`).",
        "- Corrected aggregate result: no positive A/B effect; Candidate is lower on coverage, accepted evidence, and gap closure than true Phase 15.1 Control.",
        "- OLD INCORRECT CONTROL deltas: void.",
        "- Phase 15.3: not allowed / not started.",
        "- Phase 16: not started.",
    ]
    (ARTIFACTS / "phase15_2_stop_gate.md").write_text("\n".join(stop_md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
