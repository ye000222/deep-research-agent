"""Lifecycle for the official PostgreSQL checkpoint saver."""

from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


class CheckpointRuntime:
    """Own a pool and saver without implicitly changing checkpoint schema."""

    def __init__(self, uri: str, *, min_size: int, max_size: int) -> None:
        self._uri = uri
        self._min_size = min_size
        self._max_size = max_size
        self._pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]] | None = None
        self.saver: AsyncPostgresSaver | None = None

    async def open(self) -> AsyncPostgresSaver:
        if self.saver is not None:
            return self.saver

        pool: AsyncConnectionPool[AsyncConnection[dict[str, Any]]] = AsyncConnectionPool(
            conninfo=self._uri,
            min_size=self._min_size,
            max_size=self._max_size,
            open=False,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )
        await pool.open(wait=True)
        self._pool = pool
        self.saver = AsyncPostgresSaver(pool)
        return self.saver

    async def setup_schema(self) -> None:
        """Run only from an explicit deployment/migration command."""

        saver = await self.open()
        await saver.setup()

    async def ping(self) -> None:
        saver = await self.open()
        await saver.aget_tuple({"configurable": {"thread_id": "readiness"}})

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
        self._pool = None
        self.saver = None
