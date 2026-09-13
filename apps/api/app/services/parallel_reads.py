"""Bounded concurrent webpage reads with per-host and total concurrency caps.

Only the reading is parallelized; extraction, evaluation, and state writes
stay serial downstream. A single page failure is isolated and never cancels
sibling fetches.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlsplit

from app.domain.research_tools import ReadPage
from app.tools.errors import ToolExecutionError
from app.tools.web_reader import PublicWebReader


@dataclass(frozen=True, slots=True)
class ConcurrentReadOutcome:
    page: ReadPage | None
    error_code: str | None
    latency_ms: int


async def read_pages_concurrently(
    urls: list[str],
    reader: PublicWebReader,
    *,
    max_concurrency: int = 2,
    max_same_host: int = 1,
    deadline_at: datetime | None = None,
) -> dict[str, ReadPage | None]:
    """Read many URLs concurrently, returning url -> page (None on failure).

    Deduplicates URLs, caps total in-flight fetches with a semaphore and caps
    simultaneous requests per host, so a slow host cannot saturate the pool.
    Failures are isolated per URL.
    """

    outcomes = await read_page_attempts_concurrently(
        urls,
        reader,
        max_concurrency=max_concurrency,
        max_same_host=max_same_host,
        deadline_at=deadline_at,
    )
    return {url: outcome.page for url, outcome in outcomes.items()}


async def read_page_attempts_concurrently(
    urls: list[str],
    reader: PublicWebReader,
    *,
    max_concurrency: int = 2,
    max_same_host: int = 1,
    deadline_at: datetime | None = None,
) -> dict[str, ConcurrentReadOutcome]:
    """Return auditable outcomes for every HTTP read that was started."""

    if max_concurrency <= 0 or max_same_host <= 0:
        raise ValueError("positive concurrency caps required")
    unique_urls = list(dict.fromkeys(urls))
    if not unique_urls:
        return {}
    semaphore = asyncio.Semaphore(max_concurrency)
    host_semaphores = {
        (urlsplit(url).hostname or "unknown").lower(): asyncio.Semaphore(max_same_host)
        for url in unique_urls
    }

    async def read_one(url: str) -> tuple[str, ConcurrentReadOutcome]:
        host = (urlsplit(url).hostname or "unknown").lower()
        # Wait for the host first: queued same-host URLs must neither be dropped
        # nor occupy the global slots needed by other hosts. Context managers
        # release both slots on failure and cancellation.
        async with host_semaphores[host], semaphore:
            started = time.monotonic()
            try:
                if deadline_at is None:
                    page = await reader.read(url)
                    return url, ConcurrentReadOutcome(
                        page=page,
                        error_code=None,
                        latency_ms=int((time.monotonic() - started) * 1_000),
                    )
                deadline = deadline_at
                if deadline.tzinfo is None:
                    deadline = deadline.replace(tzinfo=UTC)
                remaining = (deadline - datetime.now(UTC)).total_seconds()
                if remaining <= 0:
                    return url, ConcurrentReadOutcome(None, "DEADLINE_EXHAUSTED", 0)
                page = await asyncio.wait_for(reader.read(url), timeout=remaining)
                return url, ConcurrentReadOutcome(
                    page=page,
                    error_code=None,
                    latency_ms=int((time.monotonic() - started) * 1_000),
                )
            except ToolExecutionError as exc:
                return url, ConcurrentReadOutcome(
                    None,
                    exc.code,
                    int((time.monotonic() - started) * 1_000),
                )
            except TimeoutError:
                return url, ConcurrentReadOutcome(
                    None,
                    "DEADLINE_EXHAUSTED",
                    int((time.monotonic() - started) * 1_000),
                )

    results: list[tuple[str, ConcurrentReadOutcome]] = await asyncio.gather(
        *(read_one(url) for url in unique_urls)
    )
    return dict(results)
