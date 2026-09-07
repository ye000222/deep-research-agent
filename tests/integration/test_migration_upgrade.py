"""Verify a clean V1.0-style database can upgrade through the current head."""

from __future__ import annotations

import os
import subprocess
import sys
from uuid import uuid4

import psycopg
import pytest
from sqlalchemy.engine import make_url

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="set RUN_POSTGRES_INTEGRATION=1 to run PostgreSQL integration tests",
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def test_clean_database_upgrades_to_current_head() -> None:
    """Run Alembic against an isolated database and assert the newest revision.

    The database is created on the same PostgreSQL service used by CI, but never
    reuses the business integration database.  This catches missing migration
    dependencies and verifies the V1.0-to-current upgrade path from an empty
    schema without changing a developer's existing database.
    """

    source = make_url(os.environ["TEST_DATABASE_URL"])
    database_name = f"deep_research_upgrade_{uuid4().hex[:12]}"
    admin_url = source.set(drivername="postgresql", database="postgres")
    target_url = source.set(drivername="postgresql", database=database_name)
    target_sqlalchemy_url = source.set(
        drivername="postgresql+psycopg",
        database=database_name,
    )
    admin_dsn = admin_url.render_as_string(hide_password=False)
    target_dsn = target_url.render_as_string(hide_password=False)

    with psycopg.connect(admin_dsn, autocommit=True) as admin:
        admin.execute(
            psycopg.sql.SQL("CREATE DATABASE {}").format(
                psycopg.sql.Identifier(database_name)
            )
        )

    try:
        environment = os.environ.copy()
        environment["DATABASE_URL"] = target_sqlalchemy_url.render_as_string(
            hide_password=False
        )
        completed = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert completed.returncode == 0, (
            "Alembic upgrade failed:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

        with psycopg.connect(target_dsn) as target:
            revision = target.execute("SELECT version_num FROM alembic_version").fetchone()
        assert revision is not None
        assert revision[0] == "20260906_0019"
    finally:
        with psycopg.connect(admin_dsn, autocommit=True) as admin:
            admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (database_name,),
            )
            admin.execute(
                psycopg.sql.SQL("DROP DATABASE IF EXISTS {}").format(
                    psycopg.sql.Identifier(database_name)
                )
            )
