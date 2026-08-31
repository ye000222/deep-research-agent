from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import pytest
from app.infrastructure.db.evidence_graph_models import (
    ResearchClaimEdgeRow,
    ResearchConflictRow,
)
from app.infrastructure.db.evidence_graph_relations import refresh_question_relations


class _ScalarRows:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows


class _TupleRows:
    def __init__(self, rows: list[tuple[object, object]]) -> None:
        self._rows = rows

    def tuples(self) -> "_TupleRows":
        return self

    def all(self) -> list[tuple[object, object]]:
        return self._rows


class _RelationSession:
    def __init__(self, claims: list[object], accepted_rows: list[tuple[object, object]]) -> None:
        self.claims = claims
        self.accepted_rows = accepted_rows
        self.added: list[object] = []
        self.deleted: list[object] = []
        self._scalars_call = 0

    async def scalars(self, _statement: object) -> _ScalarRows:
        self._scalars_call += 1
        if self._scalars_call == 1:
            return _ScalarRows(self.claims)
        return _ScalarRows([])

    async def execute(self, _statement: object) -> _TupleRows:
        return _TupleRows(self.accepted_rows)

    async def scalar(self, _statement: object) -> None:
        return None

    def add(self, item: object) -> None:
        self.added.append(item)

    async def delete(self, item: object) -> None:
        self.deleted.append(item)

    async def flush(self) -> None:
        return None


def _claim(identifier: int, text: str) -> SimpleNamespace:
    now = datetime(2026, 8, 28, tzinfo=UTC)
    return SimpleNamespace(
        id=UUID(int=identifier),
        atomic_claim=text,
        status="partial",
        updated_at=now,
    )


def _accepted_evidence(identifier: int, claim_id: UUID) -> SimpleNamespace:
    return SimpleNamespace(
        id=UUID(int=identifier),
        claim_id=claim_id,
        relation="supports",
    )


@pytest.mark.asyncio
async def test_distinct_sources_persist_numeric_conflict_and_dispute_claims() -> None:
    left = _claim(1, "2025年工业机器视觉市场规模为120亿美元。")
    right = _claim(2, "2025年工业机器视觉市场规模为180亿美元。")
    left_evidence = _accepted_evidence(3, left.id)
    right_evidence = _accepted_evidence(4, right.id)
    session = _RelationSession(
        [left, right],
        [
            (left_evidence, SimpleNamespace(source_owner_key="owner-a")),
            (right_evidence, SimpleNamespace(source_owner_key="owner-b")),
        ],
    )

    stats = await refresh_question_relations(
        session,  # type: ignore[arg-type]
        run_id=UUID(int=100),
        question_id="q1",
    )

    edges = [item for item in session.added if isinstance(item, ResearchClaimEdgeRow)]
    conflicts = [item for item in session.added if isinstance(item, ResearchConflictRow)]
    assert stats.created_edges == 1
    assert stats.created_conflicts == 1
    assert len(edges) == 1
    assert edges[0].relation == "contradicts"
    assert len(conflicts) == 1
    assert conflicts[0].definition_scope == "deterministic_relation_v2"
    assert left.status == "disputed"
    assert right.status == "disputed"


@pytest.mark.asyncio
async def test_same_source_owner_does_not_create_conflict() -> None:
    left = _claim(11, "2025年工业机器视觉市场规模为120亿美元。")
    right = _claim(12, "2025年工业机器视觉市场规模为180亿美元。")
    session = _RelationSession(
        [left, right],
        [
            (_accepted_evidence(13, left.id), SimpleNamespace(source_owner_key="same-owner")),
            (_accepted_evidence(14, right.id), SimpleNamespace(source_owner_key="same-owner")),
        ],
    )

    stats = await refresh_question_relations(
        session,  # type: ignore[arg-type]
        run_id=UUID(int=101),
        question_id="q1",
    )

    conflicts = [item for item in session.added if isinstance(item, ResearchConflictRow)]
    assert stats.created_edges == 1
    assert stats.created_conflicts == 0
    assert conflicts == []
    assert left.status == "partial"
    assert right.status == "partial"