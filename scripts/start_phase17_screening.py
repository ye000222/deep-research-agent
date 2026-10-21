#!/usr/bin/env python3
"""Create the three Phase 17.0 Stage-1 screening runs.

This is a qualification launcher, not a production Research Loop change.  It
only calls the existing ResearchRunService.create_run path and stamps the
versioned benchmark identity onto each run for provenance.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "apps" / "api"
for path in (str(ROOT), str(API_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from app.core.config import Settings  # noqa: E402
from app.infrastructure.runtime import ApplicationRuntime  # noqa: E402

OWNER_HASH = "b26c638945a58e9fe40b3387be0a87e005f8dccda2188188258ae611a7c41c80"
PROFILE_RUN_ID = UUID("01a0c3ef-928b-7f0d-8006-40843c571b57")
SCREENING = (
    {
        "benchmark_id": "v1-gen-multimodal-manufacturing",
        "benchmark_version": "1.0.0",
        "goal": "分析多模态大模型在工业制造中的应用趋势。",
        "family": "technical_trend",
    },
    {
        "benchmark_id": "v1-gen-ai-agent-competition",
        "benchmark_version": "1.0.0",
        "goal": "分析 AI Agent 市场主要企业、产品和竞争格局。",
        "family": "industry_competition",
    },
    {
        "benchmark_id": "v1-gen-low-altitude-economy",
        "benchmark_version": "1.0.0",
        "goal": "分析低空经济未来三年的产业发展趋势。",
        "family": "industry_trend_policy",
    },
)


async def create_screening_runs(
    *,
    selected_benchmark_ids: set[str] | None = None,
    ordinal: int = 1,
    idempotency_suffix: str = "",
) -> list[dict[str, Any]]:
    settings = Settings()
    if not settings.independent_source_targeting_enabled:
        raise RuntimeError(
            "Phase 17.0 requires "
            "independent_source_targeting_enabled=true; refusing to create screening runs"
        )
    if settings.evidence_input_quality_enabled:
        raise RuntimeError(
            "Phase 17.0 requires "
            "evidence_input_quality_enabled=false; refusing to create screening runs"
        )
    if settings.evidence_aware_context_enabled:
        raise RuntimeError(
            "Phase 17.0 requires "
            "evidence_aware_context_enabled=false; refusing to create screening runs"
        )
    runtime = ApplicationRuntime.build(settings)
    try:
        profile = await runtime.research_run_service.get_run(OWNER_HASH, PROFILE_RUN_ID)
        flags = {
            "evidence_aware_context_enabled": settings.evidence_aware_context_enabled,
            "independent_source_targeting_enabled": settings.independent_source_targeting_enabled,
            "evidence_input_quality_enabled": settings.evidence_input_quality_enabled,
        }
        results: list[dict[str, Any]] = []
        selected = [
            definition
            for definition in SCREENING
            if selected_benchmark_ids is None
            or definition["benchmark_id"] in selected_benchmark_ids
        ]
        for definition in selected:
            benchmark = {
                "benchmark_id": definition["benchmark_id"],
                "benchmark_version": definition["benchmark_version"],
                "metric_definition_version": "coverage-v1",
                "tier": "standard",
                "plan_shape_policy": "dynamic_planner_recorded",
                "family": definition["family"],
                "reference_config_id": "v1-pre-rc-reference-1",
                "feature_flags": flags,
            }
            idempotency_key = (
                f"phase17-screening:{definition['benchmark_id']}"
                f":{settings.source_revision}:valid-v1:{ordinal}"
                f"{':' + idempotency_suffix if idempotency_suffix else ''}"
            )
            run, created = await runtime.research_run_service.create_run(
                OWNER_HASH,
                idempotency_key=idempotency_key,
                query=definition["goal"],
                saved_profile_version_id=profile.credential_version_id,
                budget_tier="standard",
                benchmark=benchmark,
            )
            results.append(
                {
                    "ordinal": ordinal,
                    "run_id": str(run.run_id),
                    "created": created,
                    "status": run.status.value,
                    "benchmark_id": definition["benchmark_id"],
                    "benchmark_version": definition["benchmark_version"],
                    "family": definition["family"],
                    "source_revision": settings.source_revision,
                    "reference_config_id": "v1-pre-rc-reference-1",
                    "feature_flags": flags,
                    "normalized_goal_sha256": hashlib.sha256(
                        run.normalized_goal.encode("utf-8")
                    ).hexdigest(),
                }
            )
        return results
    finally:
        await runtime.close()


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark-id",
        action="append",
        dest="benchmark_ids",
        choices=[item["benchmark_id"] for item in SCREENING],
        help="Create runs only for the selected benchmark definition; repeatable.",
    )
    parser.add_argument("--ordinal", type=int, default=1)
    parser.add_argument("--idempotency-suffix", default="")
    args = parser.parse_args()
    if args.ordinal < 1:
        parser.error("--ordinal must be >= 1")
    result = asyncio.run(
        create_screening_runs(
            selected_benchmark_ids=set(args.benchmark_ids)
            if args.benchmark_ids
            else None,
            ordinal=args.ordinal,
            idempotency_suffix=args.idempotency_suffix,
        )
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
