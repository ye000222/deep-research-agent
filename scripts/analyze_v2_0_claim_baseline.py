#!/usr/bin/env python3
"""Build a read-only V2 Claim funnel baseline from persisted V1 qualification runs.

Without ``--database-url`` this emits a provenance-only artifact inventory and
marks claim-level fields NOT_RECONSTRUCTABLE. It never launches a Research Run.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, cast

import psycopg

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SUITE_PATH = ARTIFACTS / "v1_release_quality_suite.json"
METRIC_VERSION = "claim-v2-baseline-v1"


def load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def artifact_inventory() -> list[dict[str, Any]]:
    suite = load_json(SUITE_PATH)
    runs: list[dict[str, Any]] = []
    for benchmark in suite.get("benchmarks", []):
        if not benchmark.get("selected_for_qualification"):
            continue
        family = benchmark.get("research_type", "unknown")
        for run_id in benchmark.get("historical_run_ids", []):
            runs.append(
                {
                    "run_id": run_id,
                    "benchmark_id": benchmark["benchmark_id"],
                    "task_family": family,
                    "benchmark_version": benchmark.get("benchmark_version"),
                    "tier": benchmark.get("tier"),
                    "metric_definition_version": benchmark.get("metric_version"),
                    "data_status": "NOT_RECONSTRUCTABLE",
                    "available_metrics": {
                        "benchmark_identity": "available",
                        "coverage_v1": "run-level values not retained in this inventory artifact",
                        "claim_funnel": "NOT_RECONSTRUCTABLE",
                        "cost": "NOT_RECONSTRUCTABLE",
                    },
                }
            )
    return runs


def database_baseline(database_url: str, runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not runs:
        return runs
    run_ids = [row["run_id"] for row in runs]
    by_id = {row["run_id"]: row for row in runs}
    url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    with psycopg.connect(url, options="-c default_transaction_read_only=on") as conn:
        evidence_rows = conn.execute(
            """
            SELECT e.run_id::text, count(*) AS candidates,
                   count(*) FILTER (WHERE e.accepted) AS accepted,
                   count(*) FILTER (WHERE e.claim_id IS NOT NULL) AS linked,
                   count(DISTINCT e.claim_id) FILTER (
                     WHERE e.accepted AND e.relation = 'supports' AND e.claim_id IS NOT NULL
                   )
                     AS supported_claims
            FROM research_evidence e WHERE e.run_id = ANY(%s::uuid[]) GROUP BY e.run_id
            """,
            (run_ids,),
        ).fetchall()
        claim_rows = conn.execute(
            """
            SELECT c.run_id::text, count(*) AS claim_records,
                   count(*) FILTER (WHERE c.status = 'disputed') AS disputed_claims,
                   count(*) FILTER (WHERE c.status = 'supported') AS legacy_supported_status
            FROM research_claims c WHERE c.run_id = ANY(%s::uuid[]) GROUP BY c.run_id
            """,
            (run_ids,),
        ).fetchall()
        link_rows = conn.execute(
            """
            SELECT e.run_id::text,
                   count(*) FILTER (WHERE e.link_count >= 2) AS multi_evidence_claims,
                   count(*) FILTER (WHERE e.owner_count >= 2) AS multi_owner_claims,
                   avg(e.owner_count) AS avg_owners_per_supported_claim,
                   sum(e.owner_count) AS total_owners_across_supported_claims
            FROM (
              SELECT ev.run_id, ev.claim_id, count(*) AS link_count,
                     count(DISTINCT src.source_owner_key) AS owner_count
              FROM research_evidence ev JOIN research_sources src ON src.id=ev.source_id
              WHERE ev.run_id = ANY(%s::uuid[])
                AND ev.accepted AND ev.relation = 'supports' AND ev.claim_id IS NOT NULL
              GROUP BY ev.run_id, ev.claim_id
            ) e GROUP BY e.run_id
            """,
            (run_ids,),
        ).fetchall()
        usage_rows = conn.execute(
            """
            SELECT id::text, started_at, finished_at, usage_snapshot, quality_snapshot
            FROM research_runs WHERE id = ANY(%s::uuid[])
            """,
            (run_ids,),
        ).fetchall()
        event_rows = conn.execute(
            """
            SELECT run_id::text,
              count(*) FILTER (
                WHERE event_type IN ('search.query.started','search.reused')
              ) AS searches,
              count(*) FILTER (
                WHERE event_type IN ('reader.execution.started','source.fetch_started')
              ) AS readers,
              count(*) FILTER (
                WHERE event_type IN (
                  'evidence.extraction.started', 'evidence.extraction_started'
                )
              ) AS extractions
            FROM agent_events WHERE run_id = ANY(%s::uuid[]) GROUP BY run_id
            """,
            (run_ids,),
        ).fetchall()

    evidence = {str(row[0]): row[1:] for row in evidence_rows}
    claims = {str(row[0]): row[1:] for row in claim_rows}
    links = {str(row[0]): row[1:] for row in link_rows}
    usage = {str(row[0]): row[1:] for row in usage_rows}
    events = {str(row[0]): row[1:] for row in event_rows}
    for run_id, item in by_id.items():
        candidates, accepted, linked, supported = evidence.get(run_id, (0, 0, 0, 0))
        claim_count, disputed, legacy_supported = claims.get(run_id, (0, 0, 0))
        multi_evidence, multi_owner, avg_owners, owner_total = links.get(run_id, (0, 0, None, 0))
        started, finished, usage_snapshot, quality_snapshot = usage.get(
            run_id, (None, None, {}, {})
        )
        searches, readers, extractions = events.get(run_id, (0, 0, 0))
        usage_snapshot = usage_snapshot if isinstance(usage_snapshot, dict) else {}
        quality_snapshot = quality_snapshot if isinstance(quality_snapshot, dict) else {}
        tokens = usage_snapshot.get("model_tokens", quality_snapshot.get("model_tokens"))
        llm_calls = usage_snapshot.get("llm_calls", quality_snapshot.get("llm_calls"))
        runtime = (
            round((finished - started).total_seconds(), 3)
            if started is not None and finished is not None
            else None
        )
        item.update(
            {
                "data_status": "RECONSTRUCTED_FROM_READ_ONLY_DB",
                "candidate_evidence": int(candidates),
                "accepted_evidence": int(accepted),
                "claim_linked_evidence": int(linked),
                "claim_records": int(claim_count),
                "supported_claims": int(supported),
                "legacy_supported_status_count": int(legacy_supported),
                "claims_with_multi_evidence": int(multi_evidence),
                "claims_with_multiple_independent_owners": int(multi_owner),
                "verified_claims": "NOT_RECONSTRUCTABLE",
                "conflicted_claims_legacy_graph_proxy": int(disputed),
                "unknown_verification_claims": "NOT_RECONSTRUCTABLE",
                "supported_claim_rate": (
                    round(int(supported) / int(claim_count), 4) if claim_count else None
                ),
                "candidate_to_accepted_rate": (
                    round(int(accepted) / int(candidates), 4) if candidates else None
                ),
                "multi_evidence_claim_rate": (
                    round(int(multi_evidence) / int(supported), 4) if supported else None
                ),
                "independently_corroborated_claim_rate": (
                    round(int(multi_owner) / int(supported), 4) if supported else None
                ),
                "verified_claim_rate": "NOT_RECONSTRUCTABLE",
                "conflicted_claim_rate_legacy_proxy": (
                    round(int(disputed) / int(claim_count), 4) if claim_count else None
                ),
                "unknown_verification_rate": "NOT_RECONSTRUCTABLE",
                "independent_owners_per_supported_claim": (
                    round(float(avg_owners), 4) if avg_owners is not None else None
                ),
                "total_owner_links_for_supported_claims": int(owner_total or 0),
                "total_tokens": tokens,
                "runtime_seconds": runtime,
                "llm_calls": llm_calls if isinstance(llm_calls, int) else "NOT_RECONSTRUCTABLE",
                "searches": int(searches),
                "reader_count": int(readers),
                "extraction_calls": int(extractions),
                "accepted_evidence_per_supported_claim": (
                    round(int(accepted) / int(supported), 4) if supported else None
                ),
                "tokens_per_accepted_evidence": (
                    round(float(tokens) / int(accepted), 4)
                    if tokens is not None and accepted
                    else None
                ),
                "tokens_per_supported_claim": (
                    round(float(tokens) / int(supported), 4)
                    if tokens is not None and supported
                    else None
                ),
                "runtime_seconds_per_supported_claim": (
                    round(float(runtime) / int(supported), 4)
                    if runtime is not None and supported
                    else None
                ),
                "searches_per_supported_claim": (
                    round(int(searches) / int(supported), 4) if supported else None
                ),
            }
        )
    return runs


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# V2.0 Claim Baseline",
        "",
        f"- Metric definition: `{report['metric_definition_version']}`",
        f"- Source: `{report['source']}`",
        f"- Historical qualification runs inventoried: **{len(report['runs'])}**",
        "- Coverage definition: `coverage-v1` unchanged; not re-scored here.",
        "",
        "## Run-level Claim funnel",
        "",
        "| Family | Run | Cand. | Accepted | Claims | Supported | Multi-evidence | Multi-owner | "
        "Verified | Conflict* | Data |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for run in report["runs"]:
        lines.append(
            f"| {run['task_family']} | `{run['run_id']}` | "
            f"{_cell(run, 'candidate_evidence')} | {_cell(run, 'accepted_evidence')} | "
            f"{_cell(run, 'claim_records')} | {_cell(run, 'supported_claims')} | "
            f"{_cell(run, 'claims_with_multi_evidence')} | "
            f"{_cell(run, 'claims_with_multiple_independent_owners')} | "
            f"{_cell(run, 'verified_claims')} | "
            f"{_cell(run, 'conflicted_claims_legacy_graph_proxy')} | "
            f"{run['data_status']} |"
        )
    lines.extend(
        [
            "",
            "`Conflicted*` is the legacy graph `disputed` status proxy, "
            "not a V2 claim-verification verdict.",
            "",
            "## Task-family aggregates",
            "",
            "| Family | Runs | Candidates | Accepted | Claims | Supported | "
            "Multi-evidence | Multi-owner | Verified | Tokens | Runtime s | "
            "Searches | Readers | Extraction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
            *[
                f"| {family['task_family']} | {family['run_count']} | "
                f"{_cell(family, 'candidate_evidence')} | {_cell(family, 'accepted_evidence')} | "
                f"{_cell(family, 'claim_records')} | {_cell(family, 'supported_claims')} | "
                f"{_cell(family, 'claims_with_multi_evidence')} | "
                f"{_cell(family, 'claims_with_multiple_independent_owners')} | "
                f"NOT_RECONSTRUCTABLE | {_cell(family, 'total_tokens')} | "
                f"{_cell(family, 'runtime_seconds')} | {_cell(family, 'searches')} | "
                f"{_cell(family, 'reader_count')} | {_cell(family, 'extraction_calls')} |"
                for family in report["family_summaries"]
            ],
            "",
            "Task-family rates and unit-cost ratios are in JSON; LLM calls and per-Claim "
            "verification remain NOT_RECONSTRUCTABLE.",
            "",
            "## Interpretation and data gaps",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
            "No run was created and no research behavior was changed.",
            "",
        ]
    )
    return "\n".join(lines)


def _cell(run: dict[str, Any], key: str) -> str:
    value = run.get(key)
    return "NOT_RECONSTRUCTABLE" if value is None else str(value)


def _family_summaries(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["task_family"], []).append(run)
    metric_names = (
        "candidate_evidence",
        "accepted_evidence",
        "claim_linked_evidence",
        "claim_records",
        "supported_claims",
        "claims_with_multi_evidence",
        "claims_with_multiple_independent_owners",
        "conflicted_claims_legacy_graph_proxy",
        "total_owner_links_for_supported_claims",
        "total_tokens",
        "runtime_seconds",
        "searches",
        "reader_count",
        "extraction_calls",
    )
    output: list[dict[str, Any]] = []
    for family, family_runs in sorted(grouped.items()):
        summary: dict[str, Any] = {"task_family": family, "run_count": len(family_runs)}
        for metric in metric_names:
            values = [run.get(metric) for run in family_runs]
            summary[metric] = (
                sum(values)
                if all(isinstance(value, (int, float)) for value in values)
                else "NOT_RECONSTRUCTABLE"
            )
        summary["verified_claims"] = "NOT_RECONSTRUCTABLE"
        summary["llm_calls"] = "NOT_RECONSTRUCTABLE"
        summary["verified_claim_rate"] = "NOT_RECONSTRUCTABLE"
        summary["unknown_verification_rate"] = "NOT_RECONSTRUCTABLE"
        claims = summary["claim_records"]
        supported = summary["supported_claims"]
        summary["candidate_to_accepted_rate"] = (
            round(summary["accepted_evidence"] / summary["candidate_evidence"], 4)
            if summary["candidate_evidence"]
            else None
        )
        summary["supported_claim_rate"] = round(supported / claims, 4) if claims else None
        summary["multi_evidence_claim_rate"] = (
            round(summary["claims_with_multi_evidence"] / supported, 4) if supported else None
        )
        summary["independently_corroborated_claim_rate"] = (
            round(summary["claims_with_multiple_independent_owners"] / supported, 4)
            if supported
            else None
        )
        summary["conflicted_claim_rate_legacy_proxy"] = (
            round(summary["conflicted_claims_legacy_graph_proxy"] / claims, 4) if claims else None
        )
        summary["independent_owners_per_supported_claim"] = (
            round(summary["total_owner_links_for_supported_claims"] / supported, 4)
            if supported
            else None
        )
        summary["evidence_per_supported_claim"] = (
            round(summary["accepted_evidence"] / supported, 4) if supported else None
        )
        summary["tokens_per_accepted_evidence"] = (
            round(summary["total_tokens"] / summary["accepted_evidence"], 4)
            if summary["accepted_evidence"] and isinstance(summary["total_tokens"], (int, float))
            else None
        )
        summary["tokens_per_supported_claim"] = (
            round(summary["total_tokens"] / supported, 4)
            if supported and isinstance(summary["total_tokens"], (int, float))
            else None
        )
        summary["runtime_per_supported_claim_seconds"] = (
            round(summary["runtime_seconds"] / supported, 4)
            if supported and isinstance(summary["runtime_seconds"], (int, float))
            else None
        )
        summary["searches_per_supported_claim"] = (
            round(summary["searches"] / supported, 4)
            if supported and isinstance(summary["searches"], (int, float))
            else None
        )
        output.append(summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database-url",
        default=os.environ.get("V2_CLAIM_BASELINE_DATABASE_URL"),
        help="optional read-only PostgreSQL URL; no research run is created",
    )
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()

    runs = artifact_inventory()
    source = "Phase 17 release-quality suite artifact inventory only"
    limitations = [
        "The configured DB hostname `postgres` is Docker-network-only; provide a reachable "
        "read-only V2_CLAIM_BASELINE_DATABASE_URL to query historical rows.",
        "Phase 17 artifacts retain run IDs/families, but omit raw Evidence/Claim rows, "
        "owner links, verification results and cost/event details.",
        "Absent historical funnel values are NOT_RECONSTRUCTABLE, not zero. "
        "V1.1 has no persisted per-Claim VERIFIED state.",
        "With DB access, corroboration requires accepted SUPPORTS links from two distinct "
        "source_owner_key values for one Claim.",
        "Legacy graph `supported` means owner corroboration; it is not V2 `VERIFIED`.",
    ]
    if args.database_url:
        runs = database_baseline(args.database_url, runs)
        source = "read-only PostgreSQL query over the Phase 17 persisted qualification run IDs"
        limitations = [
            "V1.1 has no persisted per-Claim verification result; "
            "verified/unknown counts remain NOT_RECONSTRUCTABLE.",
            "Legacy `disputed` is only a conflict proxy, not a final V2 contradiction verdict.",
            "Coverage-v1 is unchanged; this artifact reports Claim and efficiency fields only.",
        ]

    report = {
        "schema_version": "v2.0-claim-baseline.v1",
        "metric_definition_version": METRIC_VERSION,
        "source": source,
        "coverage_metric_definition": "coverage-v1 (unchanged; reference only)",
        "research_behavior_diff": "NONE",
        "new_live_runs": 0,
        "runs": runs,
        "task_family_count": len({row["task_family"] for row in runs}),
        "family_summaries": _family_summaries(runs),
        "claim_funnel_aggregate": {
            "candidate_evidence": "NOT_RECONSTRUCTABLE"
            if not args.database_url
            else "available in run rows",
            "accepted_evidence": "NOT_RECONSTRUCTABLE"
            if not args.database_url
            else "available in run rows",
            "claim_records": "NOT_RECONSTRUCTABLE"
            if not args.database_url
            else "available in run rows",
            "supported_claims": "NOT_RECONSTRUCTABLE"
            if not args.database_url
            else "available in run rows",
            "multi_evidence_claims": "NOT_RECONSTRUCTABLE"
            if not args.database_url
            else "available in run rows",
            "independently_corroborated_claims": "NOT_RECONSTRUCTABLE"
            if not args.database_url
            else "available in run rows",
            "verified_claims": "NOT_RECONSTRUCTABLE",
            "conflicted_claims": "legacy disputed-status proxy only with DB",
        },
        "limitations": limitations,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "v2_0_claim_baseline.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "v2_0_claim_baseline.md").write_text(
        build_markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {"runs": len(runs), "task_families": report["task_family_count"], "source": source},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
