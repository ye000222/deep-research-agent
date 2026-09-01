from __future__ import annotations

import os
from uuid import uuid4

import pytest
from app.infrastructure.checkpoints.lifecycle import CheckpointRuntime
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_INTEGRATION") != "1",
    reason="set RUN_POSTGRES_INTEGRATION=1 to run PostgreSQL integration tests",
)


@pytest.mark.asyncio
async def test_async_postgres_saver_pending_writes_time_travel_and_delete() -> None:
    uri = os.environ["CHECKPOINT_DATABASE_URI"]
    runtime = CheckpointRuntime(uri, min_size=1, max_size=2)
    thread_id = f"integration-{uuid4()}"
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }
    try:
        saver = await runtime.open()
        first = empty_checkpoint()
        first["channel_values"] = {"research_state": {"version": 1}}
        first["channel_versions"] = {"research_state": "1"}
        first_config = await saver.aput(
            config,
            first,
            {"source": "input", "step": 0, "parents": {}},
            {"research_state": "1"},
        )
        await saver.aput_writes(
            first_config,
            [("evidence", {"evidence_id": "E1"})],
            task_id="extractor-1",
        )
        first_tuple = await saver.aget_tuple(first_config)
        assert first_tuple is not None
        assert first_tuple.pending_writes
        assert first_tuple.pending_writes[0][1] == "evidence"

        second = empty_checkpoint()
        second["channel_values"] = {"research_state": {"version": 2}}
        second["channel_versions"] = {"research_state": "2"}
        second_config = await saver.aput(
            first_config,
            second,
            {"source": "loop", "step": 1, "parents": {}},
            {"research_state": "2"},
        )
        history = [item async for item in saver.alist(config)]
        assert len(history) == 2
        historical = await saver.aget_tuple(first_config)
        latest = await saver.aget_tuple(second_config)
        assert historical is not None
        assert latest is not None
        assert historical.checkpoint["channel_values"]["research_state"] == {"version": 1}
        assert latest.checkpoint["channel_values"]["research_state"] == {"version": 2}

        await saver.adelete_thread(thread_id)
        assert await saver.aget_tuple(config) is None
    finally:
        await runtime.close()
