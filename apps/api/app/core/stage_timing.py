"""Bounded, secret-free process-local timing (never infer it from DB timestamps)."""

import logging
import os
import time
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

_LOG = logging.getLogger(__name__)


@contextmanager
def stage_timer(stage: str, *, run_id: object = None) -> Iterator[None]:
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    outcome = "completed"
    error_type = None
    try:
        yield
    except BaseException as exc:
        outcome, error_type = "failed", type(exc).__name__
        raise
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        _LOG.info(
            "research_timing stage=%s run_id=%s elapsed_ms=%.3f outcome=%s error_type=%s",
            stage,
            run_id,
            elapsed_ms,
            outcome,
            error_type,
            extra={
                "stage_timing": {
                    "stage": stage,
                    "run_id": str(run_id) if run_id is not None else None,
                    "process_id": os.getpid(),
                    "clock": "perf_counter",
                    "started_at": started_at,
                    "finished_at": datetime.now(UTC).isoformat(),
                    "elapsed_ms": elapsed_ms,
                    "outcome": outcome,
                    "error_type": error_type,
                }
            },
        )


async def timed_call[T](stage: str, call: Awaitable[T], *, run_id: object = None) -> T:
    with stage_timer(stage, run_id=run_id):
        return await call
