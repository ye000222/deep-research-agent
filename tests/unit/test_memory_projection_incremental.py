"""Real projection/capture SQL on disposable in-memory SQLite, never a service.

SQLite supplies deterministic to_tsvector and advisory-lock stand-ins. These
tests verify data, SQL scope, work counts and transactions, not PostgreSQL GIN
latency, lock contention or linguistic ranking. PG statements are also compiled.
"""

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from app.domain.state import ResearchState
from app.infrastructure.db.memory_models import MemoryItemRow
from app.memory.manager import ResearchMemoryManager
from app.retrieval import projections
from app.retrieval.models import MemorySearchDocumentRow, RetrievalConfigVersionRow
from app.retrieval.normalization import normalize_text
from sqlalchemy import delete, event, select, update
from sqlalchemy.dialects import postgresql

from tests.integration.test_v3_deterministic_e2e import _new_store, _seed_run

OWNER = "a" * 64
OTHER = "b" * 64


class Session:
    def __init__(self, factory):
        self.factory = factory
        self.sync = factory.sync()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        self.sync.close()

    @asynccontextmanager
    async def begin(self):
        with self.sync.begin() as transaction:

            async def commit():
                if self.factory.fail_commit is True:
                    raise RuntimeError("synthetic commit failure")
                transaction.commit()
                if self.factory.fail_commit == "after":
                    raise ConnectionError("synthetic lost commit acknowledgment")

            yield SimpleNamespace(commit=commit)

    async def execute(self, statement, params=None):
        self.factory.statements.append(statement)
        if self.factory.fail_stage and self.factory.fail_stage(statement):
            raise RuntimeError("synthetic projection failure")
        return self.sync.execute(statement, params or {})

    async def scalars(self, statement):
        return (await self.execute(statement)).scalars()

    async def scalar(self, statement):
        return (await self.execute(statement)).scalar()

    async def get(self, model, key):
        return self.sync.get(model, key)

    def add(self, item):
        self.sync.add(item)

    async def flush(self):
        self.sync.flush()


@pytest.fixture
def store(tmp_path):
    original, _ = _new_store(tmp_path)
    engine = original._sync_factory.kw["bind"]
    run_id, other_run = uuid4(), uuid4()
    _seed_run(original, run_id, "offline memory fixture")
    _seed_run(original, other_run, "offline other-owner fixture")
    from app.infrastructure.db.run_models import ResearchRunRow

    with original._sync_factory() as session, session.begin():
        session.get(ResearchRunRow, other_run).owner_hash = OTHER
        session.add(
            RetrievalConfigVersionRow(
                id=projections._CONFIG_ID,
                version="lexical-v1",
                normalizer_name="existing",
                tokenizer_name="existing",
                dictionary_hash="0" * 64,
                ranking_rule_version="existing",
                activated_at=datetime.now(UTC),
            )
        )
    with engine.connect() as connection:
        raw = connection.connection.driver_connection
        # The test-only FK targets of the seeded Run were intentionally absent;
        # from here on memory/config/projection FKs and cascade are enforced.
        raw.execute("PRAGMA foreign_keys=ON")
        raw.create_function("hashtextextended", 2, lambda *_: 0)
        raw.create_function("pg_advisory_xact_lock", 1, lambda *_: 0)
        raw.create_function("to_tsvector", 2, lambda _config, value: value or "")
    factory = SimpleNamespace(
        sync=original._sync_factory,
        statements=[],
        sql=[],
        fail_stage=None,
        fail_commit=False,
        run_id=run_id,
        other_run=other_run,
    )
    factory.sessions = lambda: Session(factory)
    event.listen(engine, "before_cursor_execute", lambda _c, _cur, sql, *_: factory.sql.append(sql))
    yield factory
    engine.dispose()


def memory(store, *, owner=OWNER, **overrides):
    now = datetime.now(UTC)
    values = dict(
        id=uuid4(),
        origin_run_id=store.run_id if owner == OWNER else store.other_run,
        owner_hash=owner,
        scope_type="run",
        scope_id=str(store.run_id),
        memory_type="semantic",
        content_summary="工业视觉 AI 2026 可追溯正文",
        source_ref_ids=[],
        keywords=[],
        confidence=0.9,
        importance=0.8,
        fingerprint=uuid4().hex * 2,
        status="active",
        access_count=0,
        utility_count=0,
        created_at=now,
        updated_at=now,
    )
    values.update(overrides)
    item = MemoryItemRow(**values)
    with store.sync() as session, session.begin():
        session.add(item)
    return item.id


async def sync(store, ids, *, owner=OWNER):
    async with store.sessions() as session, session.begin():
        return await projections.sync_memory_projection(
            session,
            owner_hash=owner,
            changed_ids=ids,
            run_id=store.run_id,
        )


async def rebuild(store, *, owner=OWNER):
    async with store.sessions() as session, session.begin():
        return await projections.rebuild_memory(session, owner_hash=owner)


def snapshot(store):
    with store.sync() as session:
        return {
            row.memory_id: {
                column.name: getattr(row, column.name)
                for column in MemorySearchDocumentRow.__table__.columns
                if column.name != "updated_at"
            }
            for row in session.scalars(select(MemorySearchDocumentRow))
        }


@pytest.mark.parametrize("size", [100, 1000])
async def test_one_added_memory_work_does_not_scale_with_owner_size(store, size):
    for _ in range(size):
        memory(store)
    await rebuild(store)
    before = snapshot(store)
    new = memory(store)
    store.sql.clear()
    store.statements.clear()
    counts = await sync(store, [new])
    assert (counts.loaded, counts.inserted, counts.updated, counts.deleted) == (1, 1, 0, 0)
    assert {key: value for key, value in snapshot(store).items() if key != new} == before
    writes = [s for s in store.sql if s.startswith(("INSERT", "UPDATE", "DELETE"))]
    assert len(writes) == 2  # one batched upsert + one bounded FTS update
    assert not any(s.startswith("DELETE") for s in writes)
    reads = [s for s in store.sql if s.startswith("SELECT") and "FROM memory_items" in s]
    assert all(
        "memory_items.id IN" in s or "memory_search_documents.memory_id IN" in s for s in reads
    )
    for statement in store.statements:
        if getattr(statement, "is_insert", False) or getattr(statement, "is_update", False):
            sql = str(statement.compile(dialect=postgresql.dialect()))
            assert "owner_hash" in sql
            assert "memory_id" in sql


async def test_content_update_only_changes_one_projection(store):
    first, second = memory(store), memory(store)
    await sync(store, [first, second])
    before = snapshot(store)
    with store.sync() as session, session.begin():
        session.execute(
            update(MemoryItemRow)
            .where(MemoryItemRow.id == first)
            .values(
                content_summary="新的中文正文 change",
                confidence=0.82,
            )
        )
    counts = await sync(store, [first])
    assert counts.updated == 1
    after = snapshot(store)
    assert after[second] == before[second]
    assert after[first]["raw_search_text"] == "新的中文正文 change"


@pytest.mark.parametrize("status", ["stale", "forgotten", "superseded", "expired", "deleted"])
async def test_inactive_expired_or_deleted_memory_loses_only_its_projection(store, status):
    first, second = memory(store), memory(store)
    await sync(store, [first, second])
    with store.sync() as session, session.begin():
        if status == "deleted":
            session.execute(delete(MemoryItemRow).where(MemoryItemRow.id == first))
        else:
            values = (
                {"status": status}
                if status != "expired"
                else {"expires_at": datetime.now(UTC) - timedelta(days=1)}
            )
            session.execute(update(MemoryItemRow).where(MemoryItemRow.id == first).values(**values))
    counts = await sync(store, [first])
    assert set(snapshot(store)) == {second}
    assert counts.deleted == (0 if status == "deleted" else 1)  # physical delete cascaded


async def test_empty_changed_set_has_no_database_work(store):
    store.statements.clear()
    assert await sync(store, []) == projections.MemoryProjectionChanges()
    assert store.statements == []


async def test_owner_filter_and_full_rebuild_never_touch_other_owner(store):
    mine = memory(store)
    other = memory(store, owner=OTHER, scope_id=str(store.run_id))
    await sync(store, [other], owner=OTHER)
    before = snapshot(store)[other]
    counts = await sync(store, [mine, other])
    assert counts.loaded == 1
    assert snapshot(store)[other] == before
    await rebuild(store)
    assert snapshot(store)[other] == before


async def test_replayed_changed_set_is_idempotent_without_fts_rewrite(store):
    item = memory(store, expires_at=datetime.now(UTC) + timedelta(days=2))
    await sync(store, [item])
    store.sql.clear()
    counts = await sync(store, [item, item])
    assert counts.requested == counts.unchanged == 1
    assert not any(s.startswith(("INSERT", "UPDATE", "DELETE")) for s in store.sql)


async def test_version_mismatch_requires_explicit_rebuild_not_implicit_owner_scan(store):
    item = memory(store)
    await sync(store, [item])
    previous = uuid4()
    with store.sync() as session, session.begin():
        session.add(
            RetrievalConfigVersionRow(
                id=previous,
                version="previous-test-version",
                normalizer_name="old",
                tokenizer_name="old",
                dictionary_hash="0" * 64,
                ranking_rule_version="old",
                activated_at=datetime.now(UTC),
            )
        )
        session.flush()
        session.execute(
            update(MemorySearchDocumentRow).values(retrieval_config_version_id=previous)
        )
    with pytest.raises(projections.MemoryProjectionVersionMismatch):
        await sync(store, [item])
    assert snapshot(store)[item]["retrieval_config_version_id"] == previous
    await rebuild(store)
    assert snapshot(store)[item]["retrieval_config_version_id"] == projections._CONFIG_ID
    assert (await sync(store, [item])).unchanged == 1


async def test_incremental_and_explicit_rebuild_have_identical_final_documents(store):
    ids = [
        memory(store, content_summary=text)
        for text in (
            "工业视觉 AI 2026",
            "日期 原始来源",
            "English traceable evidence 2026",
        )
    ]
    await sync(store, ids)
    with store.sync() as session, session.begin():
        session.execute(
            update(MemoryItemRow)
            .where(MemoryItemRow.id == ids[0])
            .values(
                content_summary="工业视觉新来源 AI 2027",
                importance=0.95,
            )
        )
        session.execute(
            update(MemoryItemRow).where(MemoryItemRow.id == ids[1]).values(status="stale")
        )
    await sync(store, ids[:2])
    incremental = snapshot(store)
    await rebuild(store)
    assert snapshot(store) == incremental
    # Independent legacy normalization oracle (not the new projection helper).
    with store.sync() as session:
        for item in session.scalars(select(MemoryItemRow).where(MemoryItemRow.status == "active")):
            expected = normalize_text(item.content_summary)
            document = incremental[item.id]
            assert (
                document["raw_search_text"],
                document["latin_text"],
                document["cjk_lexemes"],
                document["fuzzy_text"],
            ) == (
                expected.raw,
                expected.latin_text,
                expected.cjk_lexemes,
                expected.fuzzy_text,
            )
            assert document["search_vector"] == expected.latin_text + expected.cjk_lexemes


@pytest.mark.parametrize("failure", ["upsert", "fts", "commit", "interruption"])
async def test_capture_failure_rolls_back_facts_and_projection_then_safe_retry(store, failure):
    if failure == "commit":
        store.fail_commit = True
    else:

        def fail(statement):
            if failure == "interruption" and getattr(statement, "is_insert", False):
                raise KeyboardInterrupt("synthetic interruption")
            return (
                getattr(statement, "is_insert", False)
                if failure == "upsert"
                else getattr(statement, "is_update", False)
                and statement.table.name == "memory_search_documents"
            )

        store.fail_stage = fail
    manager = ResearchMemoryManager(store.sessions)
    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        await manager.capture_state(ResearchState(run_id=store.run_id), node_name="planner")
    with store.sync() as session:
        assert session.scalar(select(MemoryItemRow.id)) is None
    assert snapshot(store) == {}
    store.fail_stage = None
    store.fail_commit = False
    await manager.capture_state(ResearchState(run_id=store.run_id), node_name="planner")
    assert len(snapshot(store)) == 2


async def test_capture_uses_incremental_path_and_duplicate_state_is_noop(
    store, monkeypatch, caplog
):
    monkeypatch.setattr(
        projections, "rebuild_memory", lambda *_a, **_kw: pytest.fail("full rebuild")
    )
    manager = ResearchMemoryManager(store.sessions)
    state = ResearchState(run_id=store.run_id)
    await manager.capture_state(state, node_name="planner")
    before = snapshot(store)
    store.sql.clear()
    await manager.capture_state(state, node_name="planner")
    assert not any("memory_search_documents" in sql for sql in store.sql)
    assert snapshot(store) == before
    # A new state changes two new facts and the previous working status only.
    await manager.capture_state(state.model_copy(update={"state_version": 1}), node_name="research")
    with store.sync() as session:
        working = list(
            session.scalars(
                select(MemoryItemRow).where(
                    MemoryItemRow.memory_type == "working",
                    MemoryItemRow.status == "active",
                )
            )
        )
        assert len(working) == 1
    assert len(snapshot(store)) == 3  # active working + two episodes


async def test_lifecycle_preserves_facts_but_deletes_only_ineligible_projections(store):
    expired = memory(store, expires_at=datetime.now(UTC) + timedelta(days=1))
    forgotten = memory(store, importance=0.2)
    other = memory(store, owner=OTHER)
    await sync(store, [expired, forgotten])
    await sync(store, [other], owner=OTHER)
    before = snapshot(store)[other]
    old = datetime.now(UTC) - timedelta(days=40)
    with store.sync() as session, session.begin():
        session.execute(
            update(MemoryItemRow).where(MemoryItemRow.id == expired).values(expires_at=old)
        )
        session.execute(
            update(MemoryItemRow).where(MemoryItemRow.id == forgotten).values(updated_at=old)
        )
    result = await ResearchMemoryManager(store.sessions).apply_lifecycle()
    assert result == {"stale": 1, "forgotten": 1}
    assert snapshot(store) == {other: before}
    with store.sync() as session:
        assert session.get(MemoryItemRow, expired).status == "stale"
        assert session.get(MemoryItemRow, forgotten).status == "forgotten"


async def test_timing_uses_existing_monotonic_clock_and_never_logs_memory_text(store, caplog):
    caplog.set_level("INFO")
    await ResearchMemoryManager(store.sessions).capture_state(
        ResearchState(run_id=store.run_id),
        node_name="planner",
    )
    stages = {r.stage_timing["stage"] for r in caplog.records if hasattr(r, "stage_timing")}
    assert {
        "memory.transaction.flush",
        "memory.transaction.commit",
        "memory.projection.total",
        "memory.projection.load_memories",
        "memory.projection.determine_changes",
        "memory.projection.upsert",
        "memory.projection.fts_update",
    } <= stages
    assert all(
        r.stage_timing["clock"] == "perf_counter"
        for r in caplog.records
        if hasattr(r, "stage_timing")
    )
    assert "content_summary" not in caplog.text
    assert "memory_projection_changes" in caplog.text


async def test_access_metadata_does_not_rewrite_projection(store):
    item = memory(store)
    await sync(store, [item])
    with store.sync() as session, session.begin():
        session.execute(
            update(MemoryItemRow)
            .where(MemoryItemRow.id == item)
            .values(
                access_count=3,
                last_accessed_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )
        )
    assert (await sync(store, [item])).unchanged == 1


async def test_projection_loss_can_be_repaired_for_known_dirty_id(store):
    item = memory(store)
    await sync(store, [item])
    with store.sync() as session, session.begin():
        session.execute(
            delete(MemorySearchDocumentRow).where(
                MemorySearchDocumentRow.memory_id == item,
            )
        )
    assert (await sync(store, [item])).inserted == 1


async def test_failure_after_stale_delete_restores_old_working_and_projection(store):
    manager = ResearchMemoryManager(store.sessions)
    state = ResearchState(run_id=store.run_id)
    await manager.capture_state(state, node_name="planner")
    before = snapshot(store)
    store.fail_stage = lambda s: getattr(s, "is_insert", False)
    with pytest.raises(RuntimeError):
        await manager.capture_state(
            state.model_copy(update={"state_version": 1}),
            node_name="research",
        )
    assert snapshot(store) == before
    with store.sync() as session:
        facts = list(session.scalars(select(MemoryItemRow)))
        assert len(facts) == 2
        assert all(item.status == "active" for item in facts)


async def test_old_capture_replay_does_not_supersede_new_working_state(store):
    manager = ResearchMemoryManager(store.sessions)
    old = ResearchState(run_id=store.run_id)
    await manager.capture_state(old, node_name="planner")
    await manager.capture_state(old.model_copy(update={"state_version": 1}), node_name="research")
    before = snapshot(store)
    await manager.capture_state(old, node_name="planner")
    assert snapshot(store) == before


async def test_graph_uses_real_capture_without_full_rebuild(store, monkeypatch):
    from unittest.mock import AsyncMock

    from app.services.research_graph import ResearchGraphService
    from langgraph.checkpoint.memory import InMemorySaver

    from tests.unit.test_research_graph import (
        FakePlanner,
        FakeReportWriter,
        FakeResearchLoop,
        FakeRuns,
        FakeStates,
    )

    manager = ResearchMemoryManager(store.sessions)
    # Only initial memory retrieval's PostgreSQL @@ boundary is replaced.
    # All subsequent captures/projection SQL and transactions are real.
    monkeypatch.setattr(
        manager,
        "retrieve_for_run",
        AsyncMock(
            return_value=SimpleNamespace(items=()),
        ),
    )
    monkeypatch.setattr(
        projections,
        "rebuild_memory",
        AsyncMock(
            side_effect=AssertionError("ordinary graph must never rebuild all memory"),
        ),
    )
    service = ResearchGraphService(
        FakeRuns(),
        FakeStates(store.run_id),
        FakePlanner(),
        FakeResearchLoop(),
        FakeReportWriter(),
        memories=manager,
    )
    await service.execute(store.run_id, worker_task_id="offline", checkpointer=InMemorySaver())
    assert snapshot(store)
    projections.rebuild_memory.assert_not_awaited()


async def test_explicit_global_maintenance_rebuild_is_equivalent_for_both_owners(store):
    mine, other = memory(store), memory(store, owner=OTHER)
    await sync(store, [mine])
    await sync(store, [other], owner=OTHER)
    before = snapshot(store)
    assert await rebuild(store, owner=None) == 2
    assert snapshot(store) == before


async def test_commit_acknowledgment_loss_keeps_atomic_data_and_capture_is_idempotent(store):
    manager = ResearchMemoryManager(store.sessions)
    state = ResearchState(run_id=store.run_id)
    store.fail_commit = "after"
    with pytest.raises(ConnectionError, match="lost commit acknowledgment"):
        await manager.capture_state(state, node_name="planner")
    before = snapshot(store)
    assert len(before) == 2
    with store.sync() as session:
        assert len(list(session.scalars(select(MemoryItemRow)))) == 2
    store.fail_commit = False
    await manager.capture_state(state, node_name="planner")
    assert snapshot(store) == before
