"""Persistence adapter for canonical GapRequirement records."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.gap_closure import GapRequirement
from app.infrastructure.db.research_models import GapRequirementRow


class GapRequirementRepository:
    """Upsert versioned requirements and expose them independently of legacy gaps."""

    @staticmethod
    async def upsert_many(
        session: AsyncSession,
        requirements: Sequence[GapRequirement],
    ) -> tuple[GapRequirement, ...]:
        if not requirements:
            return ()
        ids = [requirement.gap_id for requirement in requirements]
        rows = (
            await session.scalars(
                select(GapRequirementRow).where(GapRequirementRow.gap_id.in_(ids))
            )
        ).all()
        existing_by_id = {row.gap_id: row for row in rows}
        persisted: list[GapRequirement] = []
        for requirement in requirements:
            row = existing_by_id.get(requirement.gap_id)
            now = datetime.now(UTC)
            if row is None:
                effective = requirement
                row = GapRequirementRow(
                    gap_id=effective.gap_id,
                    run_id=effective.run_id,
                    plan_version=effective.plan_version,
                    question_id=effective.question_id,
                    dimension_key=effective.dimension_key,
                    requirement_type=effective.requirement_type.value,
                    criterion=effective.criterion,
                    required_evidence_count=effective.required_evidence_count,
                    required_independent_sources=effective.required_independent_sources,
                    current_evidence_count=effective.current_evidence_count,
                    current_independent_sources=effective.current_independent_sources,
                    current_coverage=effective.current_coverage,
                    required_coverage=effective.required_coverage,
                    verification_status=effective.verification_status.value,
                    closure_status=effective.closure_status.value,
                    state_version=effective.state_version,
                    transition_reason=effective.transition_reason,
                    migration_source=effective.migration_source,
                    created_at=effective.created_at,
                    updated_at=effective.updated_at,
                )
                session.add(row)
            else:
                changed = _state_changed(row, requirement)
                effective = (
                    replace(
                        requirement,
                        state_version=max(requirement.state_version, row.state_version + 1),
                        created_at=row.created_at,
                        updated_at=now,
                    )
                    if changed
                    else replace(
                        requirement,
                        state_version=row.state_version,
                        created_at=row.created_at,
                        updated_at=row.updated_at,
                    )
                )
                _copy_to_row(row, effective)
            persisted.append(effective)
        await session.flush()
        return tuple(persisted)

    @staticmethod
    async def list_for_run(
        session: AsyncSession,
        run_id: UUID,
        *,
        plan_version: int | None = None,
    ) -> list[GapRequirementRow]:
        statement = select(GapRequirementRow).where(GapRequirementRow.run_id == run_id)
        if plan_version is not None:
            statement = statement.where(GapRequirementRow.plan_version == plan_version)
        return list(
            (
                await session.scalars(
                    statement.order_by(GapRequirementRow.dimension_key)
                )
            ).all()
        )


def _state_changed(row: GapRequirementRow, requirement: GapRequirement) -> bool:
    return any(
        (
            row.requirement_type != requirement.requirement_type.value,
            row.criterion != requirement.criterion,
            row.required_evidence_count != requirement.required_evidence_count,
            row.required_independent_sources != requirement.required_independent_sources,
            row.current_evidence_count != requirement.current_evidence_count,
            row.current_independent_sources != requirement.current_independent_sources,
            row.current_coverage != requirement.current_coverage,
            row.required_coverage != requirement.required_coverage,
            row.verification_status != requirement.verification_status.value,
            row.closure_status != requirement.closure_status.value,
            row.transition_reason != requirement.transition_reason,
            row.migration_source != requirement.migration_source,
        )
    )


def _copy_to_row(row: GapRequirementRow, requirement: GapRequirement) -> None:
    row.run_id = requirement.run_id
    row.question_id = requirement.question_id
    row.dimension_key = requirement.dimension_key
    row.requirement_type = requirement.requirement_type.value
    row.criterion = requirement.criterion
    row.required_evidence_count = requirement.required_evidence_count
    row.required_independent_sources = requirement.required_independent_sources
    row.current_evidence_count = requirement.current_evidence_count
    row.current_independent_sources = requirement.current_independent_sources
    row.current_coverage = requirement.current_coverage
    row.required_coverage = requirement.required_coverage
    row.verification_status = requirement.verification_status.value
    row.closure_status = requirement.closure_status.value
    row.state_version = requirement.state_version
    row.transition_reason = requirement.transition_reason
    row.migration_source = requirement.migration_source
    row.created_at = requirement.created_at
    row.updated_at = requirement.updated_at
