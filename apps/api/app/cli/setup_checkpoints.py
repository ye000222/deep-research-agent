"""Create or migrate LangGraph checkpoint tables explicitly."""

import asyncio
import sys

from app.core.config import get_settings
from app.infrastructure.checkpoints.lifecycle import CheckpointRuntime


async def setup() -> None:
    settings = get_settings()
    runtime = CheckpointRuntime(
        settings.checkpoint_database_uri,
        min_size=settings.checkpoint_pool_min_size,
        max_size=settings.checkpoint_pool_max_size,
    )
    try:
        await runtime.setup_schema()
    finally:
        await runtime.close()


if __name__ == "__main__":
    if sys.platform == "win32":
        # psycopg async requires SelectorEventLoop; Windows defaults to ProactorEventLoop
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(setup())
