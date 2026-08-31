"""Verify that an Evidence Graph can be assembled from PostgreSQL."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPOSITORY_ROOT / "apps" / "api"
sys.path.insert(0, str(API_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.infrastructure.db.postgres import PostgresRuntime  # noqa: E402
from app.infrastructure.db.research_tools import ResearchToolRepository  # noqa: E402
from app.infrastructure.db.run_models import ResearchRunRow  # noqa: E402


async def verify(run_id: UUID | None) -> None:
    database = PostgresRuntime(get_settings().database_url)
    repository = ResearchToolRepository(database.session_factory)
    try:
        async with database.session_factory() as session:
            statement = select(ResearchRunRow.id, ResearchRunRow.owner_hash)
            if run_id is not None:
                statement = statement.where(ResearchRunRow.id == run_id)
            else:
                statement = statement.order_by(ResearchRunRow.created_at.desc()).limit(1)
            row = (await session.execute(statement)).first()
        if row is None:
            raise SystemExit("No matching Research Run exists")
        graph = await repository.get_evidence_graph(row.owner_hash, row.id)
        relation_types = sorted({edge.relation for edge in graph.edges})
        print(f"run_id: {graph.run_id}")
        print(f"claim_count: {graph.claim_count}")
        print(f"evidence_count: {graph.evidence_count}")
        print(f"snapshot_count: {graph.snapshot_count}")
        print(f"chunk_count: {graph.chunk_count}")
        print(f"edge_count: {graph.edge_count}")
        print(f"conflict_count: {graph.conflict_count}")
        print(f"relation_types: {','.join(relation_types) if relation_types else 'none'}")
    finally:
        await database.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=UUID, help="Research Run to verify; defaults to latest")
    args = parser.parse_args()
    asyncio.run(verify(args.run_id))


if __name__ == "__main__":
    main()