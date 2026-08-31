"""Rebuild deterministic Claim relations and Conflict records for Evidence Graph data.

Dry-run is the default. Pass ``--apply`` to persist idempotent graph relationships.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPOSITORY_ROOT / "apps" / "api"
sys.path.insert(0, str(API_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.infrastructure.db import evidence_graph_models as _evidence_graph_models  # noqa: E402
from app.infrastructure.db import models as _profile_models  # noqa: E402
from app.infrastructure.db import report_models as _report_models  # noqa: E402
from app.infrastructure.db import research_models as _research_models  # noqa: E402
from app.infrastructure.db import run_models as _run_models  # noqa: E402
from app.infrastructure.db import state_models as _state_models  # noqa: E402

_registered_model_modules = (
    _profile_models,
    _run_models,
    _research_models,
    _evidence_graph_models,
    _report_models,
    _state_models,
)

from app.infrastructure.db.evidence_graph_models import ResearchClaimRow  # noqa: E402
from app.infrastructure.db.evidence_graph_relations import (  # noqa: E402
    refresh_question_relations,
)
from app.infrastructure.db.postgres import PostgresRuntime  # noqa: E402


@dataclass(slots=True)
class RebuildStats:
    questions: int = 0
    examined_pairs: int = 0
    detected_edges: int = 0
    created_edges: int = 0
    deleted_edges: int = 0
    detected_conflicts: int = 0
    created_conflicts: int = 0
    dismissed_conflicts: int = 0
    reopened_conflicts: int = 0


async def rebuild(*, apply: bool, run_id: UUID | None) -> RebuildStats:
    database = PostgresRuntime(get_settings().database_url)
    stats = RebuildStats()
    try:
        async with database.session_factory() as session, session.begin():
            statement = (
                select(ResearchClaimRow.run_id, ResearchClaimRow.question_id)
                .distinct()
                .order_by(ResearchClaimRow.run_id, ResearchClaimRow.question_id)
            )
            if run_id is not None:
                statement = statement.where(ResearchClaimRow.run_id == run_id)
            questions = (await session.execute(statement)).tuples().all()
            stats.questions = len(questions)
            for question_run_id, question_id in questions:
                result = await refresh_question_relations(
                    session,
                    run_id=question_run_id,
                    question_id=question_id,
                    persist=apply,
                )
                stats.examined_pairs += result.examined_pairs
                stats.detected_edges += result.detected_edges
                stats.created_edges += result.created_edges
                stats.deleted_edges += result.deleted_edges
                stats.detected_conflicts += result.detected_conflicts
                stats.created_conflicts += result.created_conflicts
                stats.dismissed_conflicts += result.dismissed_conflicts
                stats.reopened_conflicts += result.reopened_conflicts
        return stats
    finally:
        await database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Persist graph relationships")
    parser.add_argument("--run-id", type=UUID, help="Limit work to one Research Run")
    args = parser.parse_args()
    stats = asyncio.run(rebuild(apply=args.apply, run_id=args.run_id))
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Evidence Graph relation rebuild [{mode}]")
    for field, value in asdict(stats).items():
        print(f"{field}: {value}")


if __name__ == "__main__":
    main()
