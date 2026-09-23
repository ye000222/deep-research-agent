from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from app.domain.gap_closure import GapClosureStatus
from app.infrastructure.db.gap_backfill import (
    HISTORICAL_MIGRATION_SOURCE,
    HistoricalGapBackfill,
    _map_legacy_gap,
)
from app.infrastructure.db.research_models import GapRequirementRow, ResearchGapRow

RUN_ID = UUID("00000000-0000-0000-0000-000000000001")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _legacy(*, status: str = "open", attempts: int = 0) -> ResearchGapRow:
    return ResearchGapRow(
        id=uuid4(),
        run_id=RUN_ID,
        plan_version=1,
        question_id="q7",
        gap_type="insufficient_diversity",
        description="legacy gap",
        acceptance_criteria="two independent sources",
        severity=1.0,
        status=status,
        resolution_attempts=attempts,
        created_at=NOW,
        updated_at=NOW,
    )


class _ScalarResult:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows


class _FakeSession:
    def __init__(self, legacy: list[ResearchGapRow], canonical: list[GapRequirementRow]) -> None:
        self.legacy = legacy
        self.canonical = canonical
        self.added: list[object] = []

    async def scalars(self, statement: object) -> _ScalarResult:
        entity = statement.column_descriptions[0]["entity"]  # type: ignore[attr-defined]
        if entity is ResearchGapRow:
            return _ScalarResult(self.legacy)
        return _ScalarResult(self.canonical)

    def add(self, row: object) -> None:
        self.added.append(row)

    async def flush(self) -> None:
        return None


def test_legacy_gap_mapping_preserves_resolved_state() -> None:
    requirement = _map_legacy_gap(_legacy(status="resolved"))

    assert requirement is not None
    assert requirement.closure_status is GapClosureStatus.CLOSED
    assert requirement.migration_source == HISTORICAL_MIGRATION_SOURCE
    assert requirement.state_version == 1


def test_legacy_gap_mapping_marks_attempted_open_gap_partial() -> None:
    requirement = _map_legacy_gap(_legacy(attempts=3))

    assert requirement is not None
    assert requirement.closure_status is GapClosureStatus.PARTIAL
    assert requirement.dimension_key == "q7:legacy"
    assert requirement.state_version == 4


def test_unsupported_legacy_status_is_reported_as_unmapped() -> None:
    assert _map_legacy_gap(_legacy(status="unknown")) is None


@pytest.mark.asyncio
async def test_apply_does_not_overwrite_existing_canonical_gap() -> None:
    legacy = _legacy()
    requirement = _map_legacy_gap(legacy)
    assert requirement is not None
    existing = GapRequirementRow(
        gap_id=requirement.gap_id,
        run_id=requirement.run_id,
        plan_version=requirement.plan_version,
        question_id=requirement.question_id,
        dimension_key=requirement.dimension_key,
        requirement_type=requirement.requirement_type.value,
        criterion="newer canonical state",
        required_evidence_count=1,
        required_independent_sources=2,
        current_evidence_count=2,
        current_independent_sources=2,
        current_coverage=1.0,
        required_coverage=1.0,
        verification_status="verified",
        closure_status="closed",
        state_version=9,
        transition_reason="runtime",
        migration_source="runtime",
        created_at=NOW,
        updated_at=NOW,
    )
    session = _FakeSession([legacy], [existing])

    report = await HistoricalGapBackfill.apply(session)

    assert report.created_count == 0
    assert report.existing_count == 1
    assert session.added == []
    assert existing.state_version == 9
    assert existing.criterion == "newer canonical state"


@pytest.mark.asyncio
async def test_dry_run_reports_candidate_without_writing() -> None:
    session = _FakeSession([_legacy()], [])

    report = await HistoricalGapBackfill.dry_run(session)

    assert report.mode == "dry-run"
    assert report.legacy_count == 1
    assert report.eligible_count == 1
    assert report.created_count == 0
    assert session.added == []
