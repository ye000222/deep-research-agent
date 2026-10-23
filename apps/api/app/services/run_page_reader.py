"""Run-scoped body custody, separate from Reader parsing and evidence validation."""

from __future__ import annotations

import hashlib
from uuid import UUID

from app.domain.identifiers import uuid7
from app.domain.research_tools import ReadPage
from app.domain.source_policy import normalize_source_url
from app.tools.errors import ToolExecutionError
from app.tools.web_reader import _normalize_url


class RunPageReader:
    """Persist every successful body, including pages later rejected by triage.

    Only URL aliases actually observed by Reader are merged. Identical text on
    unrelated URLs never establishes a common source identity. An in-flight
    read surviving a crash remains no-replay, not an invitation to refetch.
    """

    def __init__(self, reader, repository, artifacts, run_id: UUID, worker_task_id: str):
        self.reader = reader
        self.repository = repository
        self.artifacts = artifacts
        self.run_id = run_id
        self.worker_task_id = worker_task_id
        self.observations: dict[str, bool] = {}

    async def cached(self, url: str) -> ReadPage | None:
        try:
            _normalize_url(url)
        except ToolExecutionError:
            return None
        getter = getattr(self.repository, "reader_cache_entry", None)
        if not callable(getter):
            return None
        entry = await getter(self.run_id, worker_task_id=self.worker_task_id, url=url)
        if entry and entry.get("state") == "FAILED":
            return None
        return await self._restore(entry)

    async def _restore(self, entry) -> ReadPage | None:
        if not entry:
            return None
        if entry.get("state") == "IN_FLIGHT_NO_REPLAY":
            raise RuntimeError("READER_FETCH_OUTCOME_UNKNOWN_NO_REPLAY")
        if entry.get("state") == "FAILED":
            raise ToolExecutionError(str(entry["error_code"]), retryable=False)
        if entry.get("state") != "READABLE":
            raise RuntimeError("READER_CACHE_STATE_INVALID")
        page = ReadPage.model_validate(await self.artifacts.read_json(str(entry["artifact_uri"])))
        if hashlib.sha256(page.clean_text.encode("utf-8")).hexdigest() != page.content_hash:
            raise RuntimeError("READER_CACHE_HASH_MISMATCH")
        if page.content_hash != entry["content_hash"]:
            raise RuntimeError("READER_CACHE_IDENTITY_MISMATCH")
        if normalize_source_url(page.final_url) != entry["final_url"]:
            raise RuntimeError("READER_CACHE_URL_MISMATCH")
        return page

    async def read(self, url: str) -> ReadPage:
        self.observations[url] = False
        _normalize_url(url)
        getter = getattr(self.repository, "reader_cache_entry", None)
        store = getattr(self.repository, "store_reader_cache_entry", None)
        if callable(getter) and callable(store):
            entry = await getter(
                self.run_id, worker_task_id=self.worker_task_id, url=url, reserve=True
            )
            cached = await self._restore(entry)
            if cached is not None:
                return cached
        self.observations[url] = True
        try:
            page = await self.reader.read(url)
        except ToolExecutionError as exc:
            if callable(store):
                await store(
                    self.run_id,
                    worker_task_id=self.worker_task_id,
                    requested_url=url,
                    entry={"state": "FAILED", "error_code": exc.code},
                )
            raise
        # Timeout, cancellation, unexpected errors and interrupted persistence
        # deliberately leave IN_FLIGHT_NO_REPLAY in the durable ledger.
        if callable(store):
            uri = await self.artifacts.save_immutable_json(
                f"runs/{self.run_id}/reader-cache/{uuid7()}.json", page.model_dump(mode="json")
            )
            await store(
                self.run_id,
                worker_task_id=self.worker_task_id,
                requested_url=url,
                entry={
                    "state": "READABLE",
                    "artifact_uri": uri,
                    "final_url": normalize_source_url(page.final_url),
                    "title": page.title,
                    "content_hash": page.content_hash,
                },
            )
        return page

    def physical_fetch(self, url: str) -> bool:
        return self.observations.get(url, False)

    def consume_events(self, url: str):
        if not self.physical_fetch(url):
            return []
        consumer = getattr(self.reader, "consume_events", None)
        return consumer(url) if callable(consumer) else []
