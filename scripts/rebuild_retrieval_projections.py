"""Rebuild PostgreSQL Evidence and Memory lexical projections."""

from __future__ import annotations

import argparse
import asyncio

from app.core.config import Settings
from app.infrastructure.db.postgres import PostgresRuntime
from app.retrieval.projections import rebuild_evidence, rebuild_memory


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    parser.add_argument("--owner-hash")
    args = parser.parse_args()
    db = PostgresRuntime(Settings().database_url)
    try:
        async with db.session_factory() as session, session.begin():
            run_id = __import__("uuid").UUID(args.run_id) if args.run_id else None
            evidence_count = await rebuild_evidence(session, run_id=run_id)
            memory_count = await rebuild_memory(session, owner_hash=args.owner_hash)
            print({"evidence": evidence_count, "memory": memory_count})
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
