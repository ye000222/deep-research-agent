"""Safe historical ``research_gaps`` to ``gap_requirements`` backfill."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC
from typing import Any, cast
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.gap_closure import (
    GapClosureStatus,
    GapRequirement,
    GapRequirementType,
    VerificationStatus,
    gap_requirement_id,
)
from app.infrastructure.db.research_models import GapRequirementRow, ResearchGapRow

HISTORICAL_MIGRATION_SOURCE = "historical_research_gaps_v1"


@dataclass(frozen=True, slots=True)
class UnmappedGap:
    legacy_gap_id: UUID
    run_id: UUID
    question_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class GapBackfillPlan:
    legacy_gap_id: UUID
    requirement: GapRequirement


@dataclass(slots=True)
class GapBackfillReport:
    mode: str
    legacy_count: int = 0
    eligible_count: int = 0
    created_count: int = 0
    existing_count: int = 0
    duplicate_count: int = 0
    unmapped_count: int = 0
    unmapped: list[UnmappedGap] | None = None
    rolled_back_count: int = 0

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["unmapped"] = [asdict(item) for item in self.unmapped or []]
        return payload


class HistoricalGapBackfill:
    """Plan, apply, and rollback only explicitly marked historical rows."""

    @staticmethod
    async def plan(
        session: AsyncSession,
        *,
        run_id: UUID | None = None,
    ) -> tuple[list[GapBackfillPlan], list[UnmappedGap], int]:
        statement = select(ResearchGapRow).order_by(
            ResearchGapRow.run_id,
            ResearchGapRow.plan_version,
            ResearchGapRow.question_id,
        )
        if run_id is not None:
            statement = statement.where(ResearchGapRow.run_id == run_id)
        legacy_rows = (await session.scalars(statement)).all()
        plans: list[GapBackfillPlan] = []
        unmapped: list[UnmappedGap] = []
        seen_keys: set[tuple[UUID, int, str]] = set()
        for row in legacy_rows:
            key = (row.run_id, row.plan_version, row.question_id)
            if key in seen_keys:
                unmapped.append(
                    UnmappedGap(
                        legacy_gap_id=row.id,
                        run_id=row.run_id,
                        question_id=row.question_id,
                        reason="duplicate_legacy_key",
                    )
                )
                continue
            seen_keys.add(key)
            mapped = _map_legacy_gap(row)
            if mapped is None:
                unmapped.append(
                    UnmappedGap(
                        legacy_gap_id=row.id,
                        run_id=row.run_id,
                        question_id=row.question_id,
                        reason="unsupported_legacy_status_or_missing_question",
                    )
                )
                continue
            plans.append(GapBackfillPlan(legacy_gap_id=row.id, requirement=mapped))
        return plans, unmapped, len(legacy_rows)

    @classmethod
    async def dry_run(
        cls,
        session: AsyncSession,
        *,
        run_id: UUID | None = None,
    ) -> GapBackfillReport:
        plans, unmapped, legacy_count = await cls.plan(session, run_id=run_id)
        existing = await _existing_keys(session, (plan.requirement for plan in plans))
        return _report_for_plan(
            mode="dry-run",
            legacy_count=legacy_count,
            plans=plans,
            unmapped=unmapped,
            existing=existing,
        )

    @classmethod
    async def apply(
        cls,
        session: AsyncSession,
        *,
        run_id: UUID | None = None,
    ) -> GapBackfillReport:
        plans, unmapped, legacy_count = await cls.plan(session, run_id=run_id)
        existing = await _existing_keys(session, (plan.requirement for plan in plans))
        created = 0
        for plan in plans:
            requirement = plan.requirement
            key = (requirement.run_id, requirement.plan_version, requirement.dimension_key)
            if key in existing:
                continue
            session.add(_row_from_requirement(requirement))
            existing.add(key)
            created += 1
        await session.flush()
        report = _report_for_plan(
            mode="apply",
            legacy_count=legacy_count,
            plans=plans,
            unmapped=unmapped,
            existing=existing,
        )
        report.created_count = created
        report.existing_count = len(plans) - created
        return report

    @staticmethod
    async def rollback(
        session: AsyncSession,
        *,
        run_id: UUID | None = None,
    ) -> GapBackfillReport:
        statement = delete(GapRequirementRow).where(
            GapRequirementRow.migration_source == HISTORICAL_MIGRATION_SOURCE
        )
        if run_id is not None:
            statement = statement.where(GapRequirementRow.run_id == run_id)
        result = cast(Any, await session.execute(statement))
        return GapBackfillReport(
            mode="rollback",
            rolled_back_count=result.rowcount or 0,
            unmapped=[],
        )


def _map_legacy_gap(row: ResearchGapRow) -> GapRequirement | None:
    question_id = row.question_id.strip()
    status = row.status.casefold().strip()
    if not question_id or status not in {"open", "resolved"}:
        return None
    requirement_type = {
        "insufficient_diversity": GapRequirementType.INDEPENDENT_SOURCE,
        "conflict": GapRequirementType.CLAIM_VERIFICATION,
        "weak": GapRequirementType.EVIDENCE_QUALITY,
        "missing": GapRequirementType.EVIDENCE_QUALITY,
    }.get(row.gap_type.casefold(), GapRequirementType.EVIDENCE_QUALITY)
    closure_status = (
        GapClosureStatus.CLOSED
        if status == "resolved"
        else GapClosureStatus.PARTIAL
        if row.resolution_attempts > 0
        else GapClosureStatus.OPEN
    )
    verification_status = (
        VerificationStatus.VERIFIED
        if closure_status is GapClosureStatus.CLOSED
        else VerificationStatus.REQUIRED
    )
    dimension_key = f"{question_id}:legacy"
    now = (
        row.updated_at.astimezone(UTC)
        if row.updated_at.tzinfo
        else row.updated_at.replace(tzinfo=UTC)
    )
    return GapRequirement(
        gap_id=gap_requirement_id(row.run_id, row.plan_version, dimension_key),
        run_id=row.run_id,
        plan_version=row.plan_version,
        question_id=question_id,
        dimension_key=dimension_key,
        requirement_type=requirement_type,
        criterion=row.acceptance_criteria or row.description,
        required_evidence_count=1,
        required_independent_sources=(
            2 if requirement_type is GapRequirementType.INDEPENDENT_SOURCE else 1
        ),
        current_evidence_count=1 if closure_status is GapClosureStatus.CLOSED else 0,
        current_independent_sources=(
            2
            if closure_status is GapClosureStatus.CLOSED
            and requirement_type is GapRequirementType.INDEPENDENT_SOURCE
            else 1
            if closure_status is GapClosureStatus.CLOSED
            else 0
        ),
        verification_status=verification_status,
        closure_status=closure_status,
        created_at=row.created_at,
        updated_at=now,
        state_version=max(1, row.resolution_attempts + 1),
        current_coverage=1.0 if closure_status is GapClosureStatus.CLOSED else 0.0,
        required_coverage=1.0,
        transition_reason="historical_backfill_v1",
        migration_source=HISTORICAL_MIGRATION_SOURCE,
    )


async def _existing_keys(
    session: AsyncSession,
    requirements: Iterable[GapRequirement],
) -> set[tuple[UUID, int, str]]:
    keys = {(item.run_id, item.plan_version, item.dimension_key) for item in requirements}
    if not keys:
        return set()
    rows = (
        await session.scalars(
            select(GapRequirementRow).where(
                GapRequirementRow.run_id.in_({key[0] for key in keys})
            )
        )
    ).all()
    return {
        (row.run_id, row.plan_version, row.dimension_key)
        for row in rows
        if (row.run_id, row.plan_version, row.dimension_key) in keys
    }


def _row_from_requirement(requirement: GapRequirement) -> GapRequirementRow:
    return GapRequirementRow(
        gap_id=requirement.gap_id,
        run_id=requirement.run_id,
        plan_version=requirement.plan_version,
        question_id=requirement.question_id,
        dimension_key=requirement.dimension_key,
        requirement_type=requirement.requirement_type.value,
        criterion=requirement.criterion,
        required_evidence_count=requirement.required_evidence_count,
        required_independent_sources=requirement.required_independent_sources,
        current_evidence_count=requirement.current_evidence_count,
        current_independent_sources=requirement.current_independent_sources,
        current_coverage=requirement.current_coverage,
        required_coverage=requirement.required_coverage,
        verification_status=requirement.verification_status.value,
        closure_status=requirement.closure_status.value,
        state_version=requirement.state_version,
        transition_reason=requirement.transition_reason,
        migration_source=requirement.migration_source,
        created_at=requirement.created_at,
        updated_at=requirement.updated_at,
    )


def _report_for_plan(
    *,
    mode: str,
    legacy_count: int,
    plans: list[GapBackfillPlan],
    unmapped: list[UnmappedGap],
    existing: set[tuple[UUID, int, str]],
) -> GapBackfillReport:
    existing_count = sum(
        (
            plan.requirement.run_id,
            plan.requirement.plan_version,
            plan.requirement.dimension_key,
        )
        in existing
        for plan in plans
    )
    return GapBackfillReport(
        mode=mode,
        legacy_count=legacy_count,
        eligible_count=len(plans),
        existing_count=existing_count,
        duplicate_count=len(unmapped),
        unmapped_count=len(unmapped),
        unmapped=unmapped,
    )
