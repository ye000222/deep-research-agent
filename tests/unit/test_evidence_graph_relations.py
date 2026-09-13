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


def _claim_with_source(identifier: int, text: str, owner: str) -> tuple[object, object, object]:
    claim = _claim(identifier, text)
    evidence = _accepted_evidence(identifier + 1000, claim.id)
    source = SimpleNamespace(source_owner_key=owner)
    return claim, evidence, source


@pytest.mark.asyncio
async def test_incremental_refresh_only_recomputes_touched_pairs_and_status() -> None:
    left = _claim(201, "2025年工业机器视觉市场规模为120亿美元。")
    middle = _claim(202, "2025年工业机器视觉市场规模为180亿美元。")  # conflicts with left
    other = _claim(203, "高光谱相机对果蔬表面缺陷的识别灵敏度为96%。")  # unrelated
    left_evidence = _accepted_evidence(221, left.id)
    middle_evidence = _accepted_evidence(222, middle.id)
    other_evidence = _accepted_evidence(223, other.id)
    session = _RelationSession(
        [left, middle, other],
        [
            (left_evidence, SimpleNamespace(source_owner_key="owner-a")),
            (middle_evidence, SimpleNamespace(source_owner_key="owner-b")),
            (other_evidence, SimpleNamespace(source_owner_key="owner-c")),
        ],
    )

    # Only the newly-affected claim "middle" is touched this turn.
    stats = await refresh_question_relations(
        session,  # type: ignore[arg-type]
        run_id=UUID(int=200),
        question_id="q1",
        touched_claim_ids={middle.id},
    )

    # 3 claims -> pairs (left,middle) and (middle,other) are examined; (left,other) is not.
    assert stats.examined_pairs == 2
    conflicts = [item for item in session.added if isinstance(item, ResearchConflictRow)]
    assert len(conflicts) == 1  # only the left-middle contradiction
    # The touched claim and the claim it conflicted with become disputed.
    assert middle.status == "disputed"
    assert left.status == "disputed"
    # The unaffected claim keeps its prior status.
    assert other.status == "partial"


@pytest.mark.asyncio
async def test_incremental_refresh_without_touched_is_full_refresh() -> None:
    left = _claim(301, "2025年工业机器视觉市场规模为120亿美元。")
    right = _claim(302, "2025年工业机器视觉市场规模为180亿美元。")
    session = _RelationSession(
        [left, right],
        [
            (_accepted_evidence(311, left.id), SimpleNamespace(source_owner_key="owner-a")),
            (_accepted_evidence(312, right.id), SimpleNamespace(source_owner_key="owner-b")),
        ],
    )
    stats = await refresh_question_relations(
        session,  # type: ignore[arg-type]
        run_id=UUID(int=300),
        question_id="q1",
        touched_claim_ids=None,
    )
    assert stats.examined_pairs == 1
    assert stats.created_conflicts == 1
