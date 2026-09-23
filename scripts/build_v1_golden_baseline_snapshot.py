#!/usr/bin/env python3
"""Phase 15.0 Task Q — emit the formal V1 Golden Baseline snapshot (read-only).

This script turns the three registered Golden Baseline run ids into the durable
regression artifact ``artifacts/v1_golden_baseline.{json,md}``.  It performs no
writes to the database and changes no research behaviour; it only reads.

Composition (no re-implementation of existing rules):

* Task E metrics (mean / SD / min / max) come straight from the funnel analyzer
  (:func:`scripts.analyze_phase15_conversion_funnel.analyze`), so the coverage /
  acceptance / verification / gap numbers here are byte-identical to the funnel
  report.
* The VALID / PROVISIONAL / INVALID verdict reuses the Phase 14.5 baseline gate
  through :func:`app.domain.golden_baseline.evaluate_golden_baseline`: a baseline
  is VALID only when the manifest is fully eligible, every counted run is
  mutually COMPARABLE, the experiment-critical feature flags are runtime-
  confirmed, and at least three runs completed normally.  Fewer than three ->
  PROVISIONAL; an unregistered / non-comparable identity -> INVALID.

Run inside the repository virtualenv, e.g.::

    python scripts/build_v1_golden_baseline_snapshot.py \
        --run-ids <id1>,<id2>,<id3>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:  # psycopg is only needed for the live read; keep the import soft for lint.
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPOSITORY_ROOT / "apps" / "api"
# ``app.*`` resolves from the API root; the funnel analyzer lives under the
# repository-root ``scripts`` package, so both paths must be importable when
# this module is launched directly as ``python scripts/<name>.py``.
for _path in (str(REPOSITORY_ROOT), str(API_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.domain.benchmark_comparability import evaluate_baseline_requirements  # noqa: E402
from app.domain.golden_baseline import (  # noqa: E402
    BenchmarkBaselineManifest,
    comparison_key_from_run,
    evaluate_golden_baseline,
)

from scripts.analyze_phase15_conversion_funnel import analyze  # noqa: E402

#: Run statuses that count as a normal, usable completion for the baseline.
COMPLETED_STATUSES: frozenset[str] = frozenset({"completed", "completed_with_limitations"})

_MANIFEST_RELATIVE = Path("evals") / "benchmarks" / "v1_golden_baseline.v1.json"


def _database_uri(value: str) -> str:
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def _split_run_ids(raw: str) -> list[str]:
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def _load_manifest(path: Path) -> BenchmarkBaselineManifest:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return BenchmarkBaselineManifest.from_mapping(payload)


def _fetch_run_states(
    connection: Any, run_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Return ``run_id -> {status, normalized_goal, budget_snapshot}``."""

    states: dict[str, dict[str, Any]] = {}
    for run_id in run_ids:
        row = connection.execute(
            "SELECT status, normalized_goal, budget_snapshot FROM research_runs WHERE id = %s",
            (run_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"run {run_id} not found")
        status, goal, budget = row
        budget = budget if isinstance(budget, dict) else {}
        states[run_id] = {
            "status": str(status),
            "normalized_goal": str(goal) if goal is not None else None,
            "budget_snapshot": budget,
        }
    return states


def _verdict_summary(
    manifest: BenchmarkBaselineManifest,
    states: dict[str, dict[str, Any]],
    funnel_report: dict[str, Any],
    feature_flags_confirmed: bool,
) -> dict[str, Any]:
    run_keys = [comparison_key_from_run(states[run_id]) for run_id in states]
    completed = sum(1 for s in states.values() if s["status"] in COMPLETED_STATUSES)
    evaluation = evaluate_golden_baseline(
        manifest,
        run_keys,
        feature_flags_confirmed=feature_flags_confirmed,
        valid_run_count=completed,
    )
    manifest_gate = evaluate_baseline_requirements(
        manifest.to_comparison_key(), feature_flags_known=feature_flags_confirmed
    )
    return {
        "status": evaluation.status.value,
        "eligible": evaluation.eligible,
        "valid_run_count": evaluation.valid_run_count,
        "completed_run_count": completed,
        "run_count": len(states),
        "comparable_run_pairs": evaluation.comparable_run_pairs,
        "non_comparable_pairs": list(evaluation.non_comparable_pairs),
        "unmet_requirements": list(evaluation.unmet_requirements),
        "manifest_gate_unmet": list(manifest_gate.unmet),
        "reasons": list(evaluation.reasons),
    }


def _run_table(
    states: dict[str, dict[str, Any]], identities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    id_by_run = {item["run_id"]: item for item in identities}
    rows: list[dict[str, Any]] = []
    for run_id, state in states.items():
        ident = id_by_run.get(run_id, {})
        rows.append(
            {
                "run_id": run_id,
                "status": state["status"],
                "benchmark_id": ident.get("benchmark_id"),
                "benchmark_version": ident.get("benchmark_version"),
                "metric_definition_version": ident.get("metric_definition_version"),
            }
        )
    return rows


def build_snapshot(
    run_ids: list[str],
    database_url: str,
    manifest_path: Path,
    *,
    feature_flags_confirmed: bool,
) -> dict[str, Any]:
    if psycopg is None:  # pragma: no cover
        raise RuntimeError("psycopg is required to build the golden baseline snapshot")
    manifest = _load_manifest(manifest_path)
    funnel_report = analyze(run_ids, database_url)
    with psycopg.connect(_database_uri(database_url)) as connection:
        states = _fetch_run_states(connection, run_ids)
    verdict = _verdict_summary(manifest, states, funnel_report, feature_flags_confirmed)
    summary = funnel_report["golden_baseline_summary"]
    return {
        "schema_version": "phase15-golden-baseline.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": verdict["status"],
        "benchmark": manifest.as_dict(),
        "verdict": verdict,
        "runs": _run_table(states, funnel_report["identities"]),
        "quality_summary": summary["quality"],
        "research_work_summary": summary["research_work"],
        "cost_summary": summary["cost"],
        "first_major_loss": funnel_report.get("first_major_loss"),
        "primary_bottleneck": funnel_report.get("primary_bottleneck"),
        "coverage_funnel_reference": "artifacts/phase15_conversion_funnel.{json,md}",
        "notes": (
            "Task E aggregates are reused verbatim from the funnel analyzer; the "
            "status reuses the Phase 14.5 baseline gate. Diagnostic only — no "
            "quality rule was changed to influence these numbers."
        ),
    }


def _fmt_summary(block: dict[str, Any]) -> str:
    mean = block.get("mean")
    sd = block.get("sd")
    if mean is None:
        return "n/a"
    return f"{mean:.4g} ± {sd:.4g}"


def render_markdown(snapshot: dict[str, Any]) -> str:
    bench = snapshot["benchmark"]
    verdict = snapshot["verdict"]
    lines: list[str] = ["# Phase 15.0 — V1 Golden Baseline", ""]
    lines.append(f"**Status**: {snapshot['status']}")
    lines.append("")
    lines.append("## Benchmark Identity")
    lines.append(f"- benchmark_id: `{bench['benchmark_id']}`")
    lines.append(f"- benchmark_version: `{bench['benchmark_version']}`")
    lines.append(f"- tier: `{bench['tier']}`  budget_profile: `{bench['budget_profile']}`")
    lines.append(f"- metric_definition_version: `{bench['metric_definition_version']}`")
    lines.append(f"- plan_shape_policy: `{bench['plan_shape_policy']}`")
    lines.append(f"- plan_template_run_id: `{bench['plan_template_run_id']}`")
    flags = bench.get("feature_flags", {})
    flag_text = flags.get("evidence_aware_context_enabled")
    lines.append(f"- evidence_aware_context_enabled: `{flag_text}`")
    lines.append(f"- goal: {bench['normalized_goal']}")
    lines.append("")
    lines.append("## Runs")
    lines.append("| Run | Status | Benchmark | Version | Metric |")
    lines.append("| --- | --- | --- | --- | --- |")
    for run in snapshot["runs"]:
        lines.append(
            f"| `{run['run_id']}` | {run['status']} | {run['benchmark_id']} "
            f"| {run['benchmark_version']} | {run['metric_definition_version']} |"
        )
    lines.append("")
    lines.append(
        f"- valid/completed runs: {verdict['completed_run_count']}/{verdict['run_count']} "
        f"(comparable pairs: {verdict['comparable_run_pairs']})"
    )
    if verdict["reasons"]:
        lines.append("- reasons:")
        for reason in verdict["reasons"]:
            lines.append(f"  - {reason}")
    lines.append("")
    lines.append("## Quality (mean ± SD across golden runs)")
    quality = snapshot["quality_summary"]
    for field in ("coverage", "priority_one_coverage", "acceptance_rate", "verified_claims",
                  "gap_closed", "gap_closure_rate", "critical_gap_count"):
        lines.append(f"- **{field}**: {_fmt_summary(quality[field])}")
    lines.append("")
    lines.append("## Research Work (mean ± SD)")
    work = snapshot["research_work_summary"]
    work_fields = (
        "logical_queries",
        "search_completed",
        "readable_sources",
        "evidence_extractions",
    )
    for field in work_fields:
        lines.append(f"- **{field}**: {_fmt_summary(work[field])}")
    lines.append("")
    lines.append("## Cost (mean ± SD)")
    cost = snapshot["cost_summary"]
    for field in ("runtime_seconds", "tokens"):
        lines.append(f"- **{field}**: {_fmt_summary(cost[field])}")
    lines.append("")
    funnel = snapshot.get("primary_bottleneck") or {}
    if funnel.get("primary"):
        lines.append("## Diagnosis Pointer")
        label = funnel.get("primary_label")
        lines.append(f"- Primary bottleneck: **{funnel['primary']}** — {label}")
        lines.append(f"- Secondary: {funnel.get('secondary')}")
        lines.append("- Full funnel: see artifacts/phase15_conversion_funnel.md")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ids", required=True, help="comma-separated golden run ids")
    parser.add_argument(
        "--database-url",
        default="postgresql+psycopg://deep_research:deep_research@127.0.0.1:5432/deep_research",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / _MANIFEST_RELATIVE,
    )
    parser.add_argument(
        "--feature-flags-confirmed",
        action="store_true",
        help="assert the runtime-critical flags were observed as required",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "v1_golden_baseline",
        help="writes <prefix>.json and <prefix>.md",
    )
    args = parser.parse_args()
    run_ids = _split_run_ids(args.run_ids)
    if len(run_ids) < 1:
        raise SystemExit("at least one run id is required")
    snapshot = build_snapshot(
        run_ids,
        args.database_url,
        args.manifest,
        feature_flags_confirmed=args.feature_flags_confirmed,
    )
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.out_prefix.with_suffix(".json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    args.out_prefix.with_suffix(".md").write_text(render_markdown(snapshot), encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(
        f"wrote {args.out_prefix.with_suffix('.json')} and "
        f"{args.out_prefix.with_suffix('.md')} (status={snapshot['status']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
