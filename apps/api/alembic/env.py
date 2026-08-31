"""Alembic environment for the business PostgreSQL database."""

from __future__ import annotations

import asyncio
import sys
from logging.config import fileConfig

from alembic import context
from app.core.config import get_settings
from app.infrastructure.db import analysis_models as _analysis_models
from app.infrastructure.db import context_models as _context_models
from app.infrastructure.db import evaluation_models as _evaluation_models
from app.infrastructure.db import evidence_graph_models as _evidence_graph_models
from app.infrastructure.db import memory_models as _memory_models
from app.infrastructure.db import models as _profile_models
from app.infrastructure.db import report_models as _report_models
from app.infrastructure.db import research_models as _research_models
from app.infrastructure.db import run_models as _run_models
from app.infrastructure.db import state_models as _state_models
from app.infrastructure.db.base import Base
from app.retrieval import models as _retrieval_models
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

_registered_model_modules = (
    _profile_models,
    _run_models,
    _research_models,
    _evidence_graph_models,
    _evaluation_models,
    _analysis_models,
    _context_models,
    _memory_models,
    _retrieval_models,
    _report_models,
    _state_models,
)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", get_settings().database_url)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
