"""Backfill legacy research_gaps into canonical gap_requirements.

Dry-run is the default. Use ``--apply`` to insert only missing canonical rows,
or ``--rollback`` to delete only rows marked by this backfill.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import UUID

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPOSITORY_ROOT / "apps" / "api"
sys.path.insert(0, str(API_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.infrastructure.db import models as _profile_models  # noqa: E402,F401
from app.infrastructure.db import research_models as _research_models  # noqa: E402,F401
from app.infrastructure.db import run_models as _run_models  # noqa: E402,F401
from app.infrastructure.db.gap_backfill import (  # noqa: E402
    HistoricalGapBackfill,
)
from app.infrastructure.db.postgres import PostgresRuntime  # noqa: E402


async def run(*, mode: str, run_id: UUID | None) -> dict[str, object]:
    database = PostgresRuntime(get_settings().database_url)
    try:
        async with database.session_factory() as session, session.begin():
            if mode == "rollback":
                report = await HistoricalGapBackfill.rollback(session, run_id=run_id)
            elif mode == "apply":
                report = await HistoricalGapBackfill.apply(session, run_id=run_id)
            else:
                report = await HistoricalGapBackfill.dry_run(session, run_id=run_id)
            return report.as_dict()
    finally:
        await database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Insert missing canonical rows")
    mode.add_argument("--rollback", action="store_true", help="Remove this backfill's rows")
    parser.add_argument("--run-id", type=UUID, help="Limit work to one Research Run")
    args = parser.parse_args()
    selected_mode = "rollback" if args.rollback else "apply" if args.apply else "dry-run"
    print(
        json.dumps(
            asyncio.run(run(mode=selected_mode, run_id=args.run_id)),
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
