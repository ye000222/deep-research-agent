#!/usr/bin/env python3
"""Cancel qualification runs created with an invalid frozen configuration."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "apps" / "api"
for path in (str(ROOT), str(API_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from app.core.config import Settings  # noqa: E402
from app.infrastructure.runtime import ApplicationRuntime  # noqa: E402

OWNER_HASH = "b26c638945a58e9fe40b3387be0a87e005f8dccda2188188258ae611a7c41c80"
RUN_IDS = (
    UUID("01a0c822-fb9e-7353-9556-768c672dd692"),
    UUID("01a0c822-fbaf-7820-948e-0ac0178cc01b"),
)


async def cancel_runs() -> list[dict[str, str]]:
    runtime = ApplicationRuntime.build(Settings())
    try:
        output = []
        for run_id in RUN_IDS:
            run = await runtime.research_run_service.cancel_run(OWNER_HASH, run_id)
            output.append({"run_id": str(run.run_id), "status": run.status.value})
        return output
    finally:
        await runtime.close()


def main() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import json

    print(json.dumps(asyncio.run(cancel_runs()), indent=2))


if __name__ == "__main__":
    main()
