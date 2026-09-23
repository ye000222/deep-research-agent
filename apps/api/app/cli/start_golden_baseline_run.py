"""Start one registered Phase 15.0 Golden Baseline run (idempotent).

This launcher reuses the ordinary :class:`ResearchRunService.create_run` path so
the golden baseline is produced by the *same* production flow as every other
run.  The only addition is a benchmark identity block stamped onto the run's
``budget_snapshot`` (Task A/B): it makes the run self-describing for the
Phase 14.5 comparability gate (Task C) and changes no research behaviour.

Before creating anything it enforces the eligibility gate: the manifest must
satisfy every Phase 14.5 baseline requirement and the experiment-critical
feature flag must be observed ``false`` in the runtime configuration.  If the
gate does not pass the command refuses (``INVALID``) instead of producing a
run that could not become part of a golden baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import UUID

from app.core.config import Settings
from app.domain.benchmark_comparability import evaluate_baseline_requirements
from app.domain.golden_baseline import (
    GOLDEN_BASELINE_TIER,
    BenchmarkBaselineManifest,
)
from app.infrastructure.runtime import ApplicationRuntime

_OWNER_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_RELATIVE = Path("evals") / "benchmarks" / "v1_golden_baseline.v1.json"


def load_manifest(project_root: Path) -> BenchmarkBaselineManifest:
    """Read and validate the registered golden baseline manifest fixture."""

    manifest_path = project_root / _MANIFEST_RELATIVE
    payload: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return BenchmarkBaselineManifest.from_mapping(payload)


async def start_run(
    *,
    owner_hash: str,
    manifest: BenchmarkBaselineManifest,
    source_revision: str,
    ordinal: int,
) -> dict[str, object]:
    settings = Settings()
    if not _OWNER_HASH_PATTERN.fullmatch(owner_hash):
        raise RuntimeError("owner hash must be a lowercase SHA-256 value")
    if ordinal not in {1, 2, 3}:
        raise RuntimeError("golden baseline ordinal must be 1, 2, or 3")
    if manifest.tier != GOLDEN_BASELINE_TIER:
        raise RuntimeError(f"golden baseline tier must be {GOLDEN_BASELINE_TIER!r}")
    if source_revision.casefold() in {"development", "unknown"}:
        raise RuntimeError("golden baseline source revision must identify the candidate build")

    # Task C: eligibility gate must PASS before any run is created.
    eligibility = evaluate_baseline_requirements(
        manifest.to_comparison_key(),
        feature_flags_known=True,
    )
    if not eligibility.eligible:
        raise RuntimeError(
            "golden baseline eligibility gate FAILED: " + ", ".join(eligibility.unmet)
        )

    # Experiment-critical flag must be observed false in the runtime config.
    runtime_flag = settings.evidence_aware_context_enabled
    if runtime_flag != manifest.evidence_aware_context_enabled:
        raise RuntimeError(
            "runtime EVIDENCE_AWARE_CONTEXT_ENABLED "
            f"({runtime_flag}) does not match manifest "
            f"({manifest.evidence_aware_context_enabled}); refusing to launch"
        )

    template_run_id = UUID(manifest.plan_template_run_id)
    runtime = ApplicationRuntime.build(settings)
    try:
        template = await runtime.research_run_service.get_run(owner_hash, template_run_id)
        model = str(template.llm_config_snapshot.get("model", "")).strip()
        if not model:
            raise RuntimeError("plan template run has no model snapshot")
        idempotency_key = f"phase15-golden:{manifest.benchmark_id}:{source_revision}:{ordinal}"
        # Stamp the Phase 15.1 Branch B declared experiment factor into the run's
        # benchmark block for provenance.  This is a *declared experiment factor*,
        # not baseline identity, so — unlike evidence_aware_context_enabled — it is
        # recorded from runtime settings without a manifest-match assertion; the
        # frozen Phase 15.0 manifest stays byte-identical.
        benchmark_block = manifest.as_dict()
        existing_flags = benchmark_block.get("feature_flags")
        flags_dict: dict[str, object] = (
            dict(existing_flags) if isinstance(existing_flags, Mapping) else {}
        )
        flags_dict["independent_source_targeting_enabled"] = (
            settings.independent_source_targeting_enabled
        )
        flags_dict["evidence_input_quality_enabled"] = (
            settings.evidence_input_quality_enabled
        )
        benchmark_block["reference_config_id"] = "v1-pre-rc-reference-1"
        benchmark_block["feature_flags"] = flags_dict
        run, created = await runtime.research_run_service.create_run(
            owner_hash,
            idempotency_key=idempotency_key,
            query=manifest.normalized_goal,
            saved_profile_version_id=template.credential_version_id,
            budget_tier=manifest.tier,
            plan_template_run_id=template_run_id,
            benchmark=benchmark_block,
        )
        return {
            "run_id": str(run.run_id),
            "created": created,
            "status": run.status.value,
            "benchmark_id": manifest.benchmark_id,
            "benchmark_version": manifest.benchmark_version,
            "metric_definition_version": manifest.metric_definition_version,
            "budget_tier": run.budget_snapshot.get("tier"),
            "source_revision": run.budget_snapshot.get("source_revision"),
            "plan_template_run_id": str(template_run_id),
            "evidence_aware_context_enabled": runtime_flag,
            "independent_source_targeting_enabled": settings.independent_source_targeting_enabled,
            "evidence_input_quality_enabled": settings.evidence_input_quality_enabled,
            "model": run.llm_config_snapshot.get("model"),
            "saved_profile_id": str(run.saved_profile_id),
            "credential_version_id": str(run.credential_version_id),
            "normalized_goal_sha256": hashlib.sha256(
                run.normalized_goal.encode("utf-8")
            ).hexdigest(),
            "idempotency_key": idempotency_key,
        }
    finally:
        await runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-hash", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--ordinal", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[4],
        help="Repository root holding evals/benchmarks (defaults to the app package root).",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.project_root)
    result = asyncio.run(
        start_run(
            owner_hash=args.owner_hash,
            manifest=manifest,
            source_revision=args.source_revision,
            ordinal=args.ordinal,
        )
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
