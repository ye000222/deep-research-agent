"""Deterministic PostgreSQL readiness, idempotency and rollback smoke checks."""

from __future__ import annotations

import asyncio
import os
import sys
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _async_url(url: str) -> str:
    """Ensure the SQLAlchemy URL selects the installed psycopg 3 async dialect."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


async def main() -> None:
    url = os.environ["DATABASE_URL"]
    engine = create_async_engine(_async_url(url), pool_pre_ping=True)
    checkpoint_url = os.getenv("CHECKPOINT_DATABASE_URI")
    checkpoint_engine = (
        create_async_engine(_async_url(checkpoint_url), pool_pre_ping=True)
        if checkpoint_url
        else None
    )
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
            duplicate_outbox_keys = (
                await conn.execute(
                    text(
                        "select count(*) from ("
                        "select dispatch_key from task_dispatch_outbox "
                        "group by dispatch_key having count(*) > 1"
                        ") duplicates"
                    )
                )
            ).scalar_one()
            assert duplicate_outbox_keys == 0
            duplicate_outbox_probe = "no_fixture"
            fixture = (
                await conn.execute(
                    text("select 1 from task_dispatch_outbox limit 1")
                )
            ).first()
            if fixture is not None:
                await conn.commit()
                outer = await conn.begin()
                nested = await conn.begin_nested()
                try:
                    await conn.execute(
                        text(
                            "insert into task_dispatch_outbox "
                            "(id, run_id, dispatch_type, dispatch_key, payload_ref, status, "
                            "attempt_count, next_attempt_at, created_at) "
                            "select :id, run_id, dispatch_type, dispatch_key, payload_ref, "
                            "'pending', 0, now(), now() from task_dispatch_outbox limit 1"
                        ),
                        {"id": str(uuid4())},
                    )
                except IntegrityError:
                    await nested.rollback()
                    duplicate_outbox_probe = "passed"
                else:
                    await nested.rollback()
                    duplicate_outbox_probe = "failed"
                await outer.commit()

            checkpoint_tables = None
            if checkpoint_engine is not None:
                async with checkpoint_engine.connect() as checkpoint_conn:
                    checkpoint_tables = (
                        await checkpoint_conn.execute(
                            text(
                                "select count(*) from information_schema.tables "
                                "where table_schema='public' and table_name in "
                                "('checkpoints','checkpoint_blobs','checkpoint_writes',"
                                "'checkpoint_migrations')"
                            )
                        )
                    ).scalar_one()
                    assert checkpoint_tables == 4, checkpoint_tables

            print(
                {
                    "postgres_version_num": int(version),
                    "extensions": sorted(extensions),
                    "accepted_evidence": int(accepted),
                    "projected_evidence": int(projected),
                    "duplicate_event_keys": int(duplicate_events),
                    "rollback_probe": "passed",
                    "outbox_contract": "passed",
                    "duplicate_outbox_keys": int(duplicate_outbox_keys),
                    "duplicate_outbox_probe": duplicate_outbox_probe,
                    "checkpoint_tables": checkpoint_tables,
                }
            )
    finally:
        await engine.dispose()
        if checkpoint_engine is not None:
            await checkpoint_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
