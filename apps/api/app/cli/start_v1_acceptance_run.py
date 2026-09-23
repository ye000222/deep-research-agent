"""Start one idempotent real V1 acceptance run from a known baseline run."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from uuid import UUID

from app.core.config import Settings
from app.infrastructure.runtime import ApplicationRuntime

_OWNER_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


async def start_run(
    *,
    owner_hash: str,
    baseline_run_id: UUID,
    source_revision: str,
    ordinal: int,
) -> dict[str, object]:
    settings = Settings()
    if settings.source_revision != source_revision:
        raise RuntimeError(
            "API SOURCE_REVISION does not match the requested acceptance revision"
        )
    if source_revision.casefold() in {"development", "unknown"}:
        raise RuntimeError("acceptance source revision must identify the candidate build")
    if not _OWNER_HASH_PATTERN.fullmatch(owner_hash):
        raise RuntimeError("owner hash must be a lowercase SHA-256 value")
    if ordinal not in {1, 2, 3}:
        raise RuntimeError("acceptance ordinal must be 1, 2, or 3")

    runtime = ApplicationRuntime.build(settings)
    try:
        baseline = await runtime.research_run_service.get_run(owner_hash, baseline_run_id)
        model = str(baseline.llm_config_snapshot.get("model", "")).strip()
        if not model:
            raise RuntimeError("baseline run has no model snapshot")
        idempotency_key = f"v1-closeout:{source_revision}:{ordinal}"
        run, created = await runtime.research_run_service.create_run(
            owner_hash,
            idempotency_key=idempotency_key,
            query=baseline.normalized_goal,
            saved_profile_version_id=baseline.credential_version_id,
            budget_tier="standard",
            plan_template_run_id=baseline_run_id,
        )
        return {
            "run_id": str(run.run_id),
            "created": created,
            "status": run.status.value,
            "budget_tier": run.budget_snapshot.get("tier"),
            "source_revision": run.budget_snapshot.get("source_revision"),
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
    parser.add_argument("--baseline-run-id", type=UUID, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--ordinal", type=int, choices=(1, 2, 3), required=True)
    args = parser.parse_args()
    result = asyncio.run(
        start_run(
            owner_hash=args.owner_hash,
            baseline_run_id=args.baseline_run_id,
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
