from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from app.domain.research_tools import ReadPage
from app.services.parallel_reads import read_pages_concurrently
from app.tools.errors import ToolExecutionError


class CountingReader:
    def __init__(self, fail_urls: set[str] | None = None) -> None:
        self.calls: list[str] = []
        self.fail_urls = fail_urls or set()

    async def read(self, url: str) -> ReadPage:
        self.calls.append(url)
        if url in self.fail_urls:
            raise ToolExecutionError("WEBPAGE_PROVIDER_UNAVAILABLE", retryable=False)
        return ReadPage(
            final_url=url,
            title=url,
            clean_text=f"Evidence from {url}. " * 10,
            content_hash="5" * 64,
            fetched_at=datetime.now(UTC),
        )


@pytest.mark.asyncio
async def test_read_pages_concurrently_isolates_failures() -> None:
    reader = CountingReader(fail_urls={"https://bad.example/1"})
    result = await read_pages_concurrently(
        ["https://bad.example/1", "https://ok.example/1"],
        reader,
        max_concurrency=2,
        max_same_host=4,
    )
    assert result["https://bad.example/1"] is None
    assert result["https://ok.example/1"] is not None
    assert reader.calls == ["https://bad.example/1", "https://ok.example/1"]


@pytest.mark.asyncio
async def test_read_pages_concurrently_caps_per_host() -> None:
    class SlowReader(CountingReader):
        active = 0
        peak = 0

        async def read(self, url: str) -> ReadPage:
            if "host-a" in url:
                self.active += 1
                self.peak = max(self.peak, self.active)
                try:
                    await asyncio.sleep(0.001)
                    return await super().read(url)
                finally:
                    self.active -= 1
            return await super().read(url)

    reader = SlowReader()
    urls = [f"https://host-a.example/{i}" for i in range(4)] + [
        "https://host-b.example/1"
    ]
    result = await read_pages_concurrently(
        urls,
        reader,
        max_concurrency=10,
        max_same_host=2,
    )
    read_a = [u for u in reader.calls if "host-a" in u]
    assert len(read_a) == 4  # concurrency limits do not discard queued pages
    assert reader.peak == 2
    assert sum(v is not None for v in result.values()) == 5


@pytest.mark.asyncio
async def test_same_host_failure_releases_slot_for_next_page() -> None:
    urls = ["https://a.example/bad", "https://a.example/good"]
    reader = CountingReader(fail_urls={urls[0]})
    result = await read_pages_concurrently(urls, reader, max_same_host=1)
    assert result[urls[0]] is None
    assert result[urls[1]] is not None
    assert reader.calls == urls


@pytest.mark.asyncio
async def test_read_pages_concurrently_deduplicates_and_orders_stably() -> None:
    reader = CountingReader()
    urls = ["https://a.example/1", "https://a.example/1", "https://b.example/1"]
    result = await read_pages_concurrently(
        urls,
        reader,
        max_concurrency=4,
        max_same_host=4,
    )
    assert reader.calls == ["https://a.example/1", "https://b.example/1"]
    assert set(result) == {"https://a.example/1", "https://b.example/1"}
    await asyncio.sleep(0)
