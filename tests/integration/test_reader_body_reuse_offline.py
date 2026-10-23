"""Durable run-scoped body reuse and read/page accounting; no real DB/HTTP."""

import asyncio
import hashlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from app.domain.research_tools import ReadPage
from app.infrastructure.artifacts import LocalArtifactStore
from app.infrastructure.db.research_tools import ResearchToolRepository
from app.infrastructure.db.run_models import ResearchRunRow
from app.services.run_page_reader import RunPageReader
from app.tools.errors import ToolExecutionError

from tests.integration.test_feedback_search_budget_offline import target_and_budget, usage
from tests.integration.test_v3_deterministic_e2e import (
    _sqlite_utc_datetime_compatibility,  # noqa: F401
)
from tests.unit.test_first_evidence_budget_regression import store as first_store  # noqa: F401


@pytest.fixture
def store(first_store):  # noqa: F811
    return first_store


class Reader:
    def __init__(self, final_url=None, error=None):
        self.final_url, self.error, self.calls = final_url, error, []

    async def read(self, url):
        self.calls.append(url)
        if self.error:
            raise self.error
        text = "The agency published a complete original public statement. " * 12
        return ReadPage(
            final_url=self.final_url or url,
            title="Original statement",
            clean_text=text,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            fetched_at=datetime.now(UTC),
            truncated=True,
            truncation_reason="fixture_limit",
            available_char_count=2000,
            document_type="PDF",
        )


@pytest.mark.parametrize(
    "alias",
    [
        "https://agency.gov/page",
        "https://agency.gov/page/",
        "https://agency.gov/page#date",
        "https://agency.gov/page?utm_source=fixture",
        "http://agency.gov/page",
    ],
)
async def test_cross_question_repository_restart_uses_body_without_http_or_page_charge(
    store,
    tmp_path,
    alias,
):
    tools, target = await target_and_budget(store, used=0)
    artifacts = LocalArtifactStore(tmp_path / "bodies")
    physical = Reader()
    reader = RunPageReader(physical, tools, artifacts, store[1], "offline-worker")
    assert (
        await tools.reserve_page_slots(store[1], worker_task_id="offline-worker", requested=1)
    ).granted == 1
    first = await reader.read("https://agency.gov/page")
    await tools.record_page_fetched(
        store[1],
        worker_task_id="offline-worker",
        target=target,
        url=first.final_url,
        reused=False,
        latency_ms=1,
    )
    before = usage(store)
    # A fresh repository/reader, another question, and no free page slots.
    with store[0]._sync_factory() as session, session.begin():
        run = session.get(ResearchRunRow, store[1])
        run.budget_snapshot = {**run.budget_snapshot, "max_pages_fetched": 1}
    restarted = RunPageReader(
        physical, ResearchToolRepository(store[0]), artifacts, store[1], "offline-worker"
    )
    assert (await restarted.cached(alias)).model_dump() == first.model_dump()
    second = await restarted.read(alias)
    assert not restarted.physical_fetch(alias)
    await tools.record_page_fetched(
        store[1],
        worker_task_id="offline-worker",
        target=target,
        url=second.final_url,
        reused=True,
        latency_ms=0,
        requested_url=alias,
    )
    after = usage(store)
    assert after["pages_fetched"] == before["pages_fetched"] == 1
    assert after["page_fetch_attempts"] == before["page_fetch_attempts"] == 1
    assert physical.calls == ["https://agency.gov/page"]
    assert second.model_dump() == first.model_dump()  # keep truncation/document provenance


async def test_observed_redirect_target_is_reused_before_reader(store, tmp_path):
    tools, _ = await target_and_budget(store, used=0)
    physical = Reader(final_url="https://agency.gov/canonical")
    reader = RunPageReader(
        physical, tools, LocalArtifactStore(tmp_path), store[1], "offline-worker"
    )
    await reader.read("https://agency.gov/old")
    await reader.read("https://agency.gov/canonical#section")
    assert len(physical.calls) == 1


@pytest.mark.parametrize("same_canonical", [True, False])
async def test_new_alias_requires_discovery_but_same_text_does_not_merge_sources(
    store,
    tmp_path,
    same_canonical,
):
    tools, target = await target_and_budget(store, used=0)
    physical = Reader(final_url="https://agency.gov/canonical" if same_canonical else None)
    reader = RunPageReader(
        physical, tools, LocalArtifactStore(tmp_path), store[1], "offline-worker"
    )
    for url in ["https://agency.gov/a", "https://other.gov/b"]:
        assert (
            await tools.reserve_page_slots(store[1], worker_task_id="offline-worker", requested=1)
        ).granted == 1
        page = await reader.read(url)
        await tools.record_page_fetched(
            store[1],
            worker_task_id="offline-worker",
            target=target,
            url=page.final_url,
            reused=False,
            latency_ms=1,
            requested_url=url,
        )
    assert len(physical.calls) == usage(store)["page_fetch_attempts"] == 2
    assert usage(store)["pages_fetched"] == (1 if same_canonical else 2)


@pytest.mark.parametrize("failure", ["crash", "cancel", "persist"])
async def test_unknown_read_or_failed_body_persistence_cannot_be_replayed(
    store,
    tmp_path,
    monkeypatch,
    failure,
):
    tools, _ = await target_and_budget(store, used=0)
    artifacts = LocalArtifactStore(tmp_path)
    error = (
        RuntimeError("interrupted")
        if failure == "crash"
        else (asyncio.CancelledError() if failure == "cancel" else None)
    )
    physical = Reader(error=error)
    if failure == "persist":
        monkeypatch.setattr(
            artifacts, "save_immutable_json", AsyncMock(side_effect=OSError("disk"))
        )
    reader = RunPageReader(physical, tools, artifacts, store[1], "offline-worker")
    with pytest.raises((RuntimeError, asyncio.CancelledError, OSError)):
        await reader.read("https://agency.gov/page")
    restarted = RunPageReader(physical, tools, artifacts, store[1], "offline-worker")
    with pytest.raises(RuntimeError, match="OUTCOME_UNKNOWN_NO_REPLAY"):
        await restarted.read("https://agency.gov/page")
    assert len(physical.calls) == 1


async def test_known_403_is_not_readable_and_not_refetched(store, tmp_path):
    tools, target = await target_and_budget(store, used=0)
    physical = Reader(error=ToolExecutionError("WEBPAGE_HTTP_FORBIDDEN", retryable=False))
    reader = RunPageReader(
        physical, tools, LocalArtifactStore(tmp_path), store[1], "offline-worker"
    )
    for attempt in range(2):
        await tools.reserve_page_slots(store[1], worker_task_id="offline-worker", requested=1)
        with pytest.raises(ToolExecutionError):
            await reader.read("https://agency.gov/page")
        await tools.record_page_failure(
            store[1],
            worker_task_id="offline-worker",
            target=target,
            url="https://agency.gov/page",
            error_code="WEBPAGE_HTTP_FORBIDDEN",
            reused=attempt == 1,
        )
        await tools.release_page_slots(store[1], worker_task_id="offline-worker", count=1)
    assert await reader.cached("https://agency.gov/page") is None
    assert len(physical.calls) == usage(store)["page_fetch_attempts"] == 1
    assert usage(store).get("pages_fetched", 0) == 0


async def test_cache_hash_mismatch_fails_without_network(store, tmp_path, monkeypatch):
    tools, _ = await target_and_budget(store, used=0)
    artifacts, physical = LocalArtifactStore(tmp_path), Reader()
    reader = RunPageReader(physical, tools, artifacts, store[1], "offline-worker")
    page = await reader.read("https://agency.gov/page")
    corrupted = page.model_dump(mode="json")
    corrupted["clean_text"] += "tampered"
    monkeypatch.setattr(artifacts, "read_json", AsyncMock(return_value=corrupted))
    with pytest.raises(RuntimeError, match="HASH_MISMATCH"):
        await reader.read("https://agency.gov/page")
    assert len(physical.calls) == 1


async def test_concurrent_identical_reads_cannot_bypass_durable_intent(store, tmp_path):
    tools, _ = await target_and_budget(store, used=0)
    entered, finish = asyncio.Event(), asyncio.Event()
    physical = Reader()
    original = physical.read

    async def slow(url):
        entered.set()
        await finish.wait()
        return await original(url)

    physical.read = slow
    reader = RunPageReader(
        physical, tools, LocalArtifactStore(tmp_path), store[1], "offline-worker"
    )
    first = asyncio.create_task(reader.read("https://agency.gov/page"))
    await entered.wait()
    try:
        with pytest.raises(RuntimeError, match="OUTCOME_UNKNOWN_NO_REPLAY"):
            await reader.read("https://agency.gov/page#duplicate")
    finally:
        finish.set()
        await first
    assert len(physical.calls) == 1


@pytest.mark.parametrize(
    "invalid",
    [
        "https://user:fixture@agency.gov/page",
        "https://agency.gov:1234/page",
        "file://agency.gov/page",
    ],
)
async def test_cache_does_not_bypass_existing_reader_url_validation(store, tmp_path, invalid):
    tools, _ = await target_and_budget(store, used=0)
    artifacts, physical = LocalArtifactStore(tmp_path), Reader()
    reader = RunPageReader(physical, tools, artifacts, store[1], "offline-worker")
    await reader.read("https://agency.gov/page")
    assert await reader.cached(invalid) is None
    with pytest.raises(ToolExecutionError):
        await reader.read(invalid)
    assert not reader.physical_fetch(invalid)
    assert len(physical.calls) == 1
