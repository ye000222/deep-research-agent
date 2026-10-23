"""Accepted rows are authoritative; JSON counters are transaction-local projections."""

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.db.research_models import ResearchEvidenceRow
from app.infrastructure.db.run_models import ResearchRunRow


async def sync_accepted_evidence_count(session: AsyncSession, run: ResearchRunRow) -> int:
    await session.flush()
    count = int(
        await session.scalar(
            select(func.count(ResearchEvidenceRow.id)).where(
                ResearchEvidenceRow.run_id == run.id,
                ResearchEvidenceRow.accepted.is_(True),
            )
        )
        or 0
    )
    run.usage_snapshot = {**run.usage_snapshot, "accepted_evidence": count}
    run.quality_snapshot = {**run.quality_snapshot, "accepted_evidence": count}
    return count
