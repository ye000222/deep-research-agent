"""Start a real standard run for an arbitrary goal using an existing profile."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from uuid import UUID

from app.core.config import Settings
from app.infrastructure.runtime import ApplicationRuntime


async def start(
    *,
    owner_hash: str,
    profile_run_id: UUID,
    query: str,
    source_revision: str,
    run_label: str = "",
) -> dict[str, object]:
    settings = Settings()
    if settings.source_revision != source_revision:
        raise RuntimeError("API SOURCE_REVISION does not match requested revision")
    runtime = ApplicationRuntime.build(settings)
    try:
        profile_run = await runtime.research_run_service.get_run(owner_hash, profile_run_id)
        run, created = await runtime.research_run_service.create_run(
            owner_hash,
            idempotency_key=(
                f"arbitrary-v1:{profile_run_id}:"
                f"{hashlib.sha256(query.encode('utf-8')).hexdigest()[:24]}"
                f":{run_label.strip()}"
            ),
            query=query,
            saved_profile_version_id=profile_run.credential_version_id,
            budget_tier="standard",
        )
        return {"run_id": str(run.run_id), "created": created, "status": run.status.value}
    finally:
        await runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner-hash", required=True)
    parser.add_argument("--profile-run-id", type=UUID, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--run-label", default="", help="Unique label for a repeated acceptance run")
    args = parser.parse_args()
    result = asyncio.run(start(**vars(args)))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
