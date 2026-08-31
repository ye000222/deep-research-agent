"""Deterministic PostgreSQL readiness, idempotency and rollback smoke checks."""

from __future__ import annotations

import asyncio
import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def main() -> None:
    url = os.environ["DATABASE_URL"]
    engine = create_async_engine(url, pool_pre_ping=True)
    try:
        async with engine.connect() as conn:
            version = (
                await conn.execute(text("select current_setting('server_version_num')"))
            ).scalar_one()
            extensions = set(
                (
                    await conn.execute(
                        text(
                            "select extname from pg_extension where extname in ("
                            + "'pg_trgm','unaccent')"
                        )
                    )
                ).scalars()
            )
            assert int(version) >= 180000, version
            assert extensions == {"pg_trgm", "unaccent"}, extensions

            accepted = (
                await conn.execute(
                    text("select count(*) from research_evidence where accepted is true")
                )
            ).scalar_one()
            projected = (
                await conn.execute(text("select count(*) from evidence_search_documents"))
            ).scalar_one()
            assert projected <= accepted

            duplicate_events = (
                await conn.execute(
                    text(
                        "select count(*) from ("
                        "select run_id, run_seq from agent_events "
                        "group by run_id, run_seq having count(*) > 1"
                        ") duplicates"
                    )
                )
            ).scalar_one()
            assert duplicate_events == 0

            await conn.execute(
                text("create temporary table recovery_probe (id integer primary key)")
            )
            await conn.commit()
            tx = await conn.begin()
            await conn.execute(text("insert into recovery_probe values (1)"))
            await tx.rollback()
            remaining = (
                await conn.execute(text("select count(*) from recovery_probe"))
            ).scalar_one()
            assert remaining == 0

            outbox_columns = (
                await conn.execute(
                    text(
                        "select count(*) from information_schema.columns "
                        "where table_name='task_dispatch_outbox' "
                        "and column_name in ('status','attempt_count','next_attempt_at')"
                    )
                )
            ).scalar_one()
            assert outbox_columns == 3

            print(
                {
                    "postgres_version_num": int(version),
                    "extensions": sorted(extensions),
                    "accepted_evidence": int(accepted),
                    "projected_evidence": int(projected),
                    "duplicate_event_keys": int(duplicate_events),
                    "rollback_probe": "passed",
                    "outbox_contract": "passed",
                }
            )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
