"""Backfill legacy evidence into the relational Evidence Graph.

Dry-run is the default. Pass ``--apply`` to write Claim, Snapshot, Chunk references.
Rows whose original artifact is unavailable remain auditable and are reported with a
missing Chunk instead of inventing source text.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy import and_, distinct, func, or_, select

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPOSITORY_ROOT / "apps" / "api"
sys.path.insert(0, str(API_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.infrastructure.db import evidence_graph_models as _evidence_graph_models  # noqa: E402
from app.infrastructure.db import models as _profile_models  # noqa: E402
from app.infrastructure.db import report_models as _report_models  # noqa: E402
from app.infrastructure.db import research_models as _research_models  # noqa: E402
from app.infrastructure.db import run_models as _run_models  # noqa: E402
from app.infrastructure.db import state_models as _state_models  # noqa: E402

_registered_model_modules = (
    _profile_models,
    _run_models,
    _research_models,
    _evidence_graph_models,
    _report_models,
    _state_models,
)

from app.domain.evidence_graph import (  # noqa: E402
    build_evidence_chunk,
    claim_fingerprint,
    derive_claim_status,
)
from app.domain.identifiers import uuid7  # noqa: E402
from app.infrastructure.db.evidence_graph_models import (  # noqa: E402
    ResearchClaimRow,
    ResearchSourceChunkRow,
    ResearchSourceSnapshotRow,
)
from app.infrastructure.db.postgres import PostgresRuntime  # noqa: E402
from app.infrastructure.db.research_models import (  # noqa: E402
    ResearchEvidenceRow,
    ResearchSourceRow,
)


@dataclass(slots=True)
class BackfillStats:
    candidates: int = 0
    claims_created: int = 0
    snapshots_created: int = 0
    chunks_created: int = 0
    evidence_bound: int = 0
    missing_artifacts: int = 0
    unlocated_quotes: int = 0
    accepted_unlocated_quotes: int = 0


async def backfill(*, apply: bool, run_id: UUID | None) -> BackfillStats:
    settings = get_settings()
    database = PostgresRuntime(settings.database_url)
    artifact_root = settings.artifact_root.resolve()
    stats = BackfillStats()
    try:
        async with database.session_factory() as session, session.begin():
            statement = (
                select(ResearchEvidenceRow, ResearchSourceRow)
                .join(ResearchSourceRow, ResearchEvidenceRow.source_id == ResearchSourceRow.id)
                .where(
                    or_(
                        ResearchEvidenceRow.claim_id.is_(None),
                        ResearchEvidenceRow.snapshot_id.is_(None),
                        and_(
                            ResearchEvidenceRow.accepted.is_(True),
                            ResearchEvidenceRow.chunk_id.is_(None),
                        ),
                    )
                )
                .order_by(ResearchEvidenceRow.run_id, ResearchEvidenceRow.created_at)
            )
            if run_id is not None:
                statement = statement.where(ResearchEvidenceRow.run_id == run_id)
            rows = (await session.execute(statement)).tuples().all()
            stats.candidates = len(rows)
            if not apply:
                for evidence, source in rows:
                    source_text = _read_artifact(artifact_root, source.artifact_uri)
                    if source_text is None:
                        stats.missing_artifacts += 1
                    elif build_evidence_chunk(source_text, evidence.exact_quote) is None:
                        stats.unlocated_quotes += 1
                        if evidence.accepted:
                            stats.accepted_unlocated_quotes += 1
                return stats

            for evidence, source in rows:
                snapshot = await session.scalar(
                    select(ResearchSourceSnapshotRow).where(
                        ResearchSourceSnapshotRow.source_id == source.id,
                        ResearchSourceSnapshotRow.content_hash == source.content_hash,
                    )
                )
                if snapshot is None:
                    snapshot = ResearchSourceSnapshotRow(
                        id=uuid7(),
                        run_id=evidence.run_id,
                        source_id=source.id,
                        final_url=source.canonical_url,
                        fetched_at=source.fetched_at,
                        published_at=None,
                        content_hash=source.content_hash,
                        parser_version="legacy-backfill-v1",
                        artifact_uri=source.artifact_uri,
                        char_count=source.char_count,
                    )
                    session.add(snapshot)
                    await session.flush()
                    stats.snapshots_created += 1

                claim_hash = claim_fingerprint(evidence.claim)
                claim = await session.scalar(
                    select(ResearchClaimRow).where(
                        ResearchClaimRow.run_id == evidence.run_id,
                        ResearchClaimRow.question_id == evidence.question_id,
                        ResearchClaimRow.claim_hash == claim_hash,
                    )
                )
                if claim is None:
                    claim = ResearchClaimRow(
                        id=uuid7(),
                        run_id=evidence.run_id,
                        plan_version=evidence.plan_version,
                        question_id=evidence.question_id,
                        dimension_key=evidence.question_id,
                        atomic_claim=evidence.claim,
                        claim_hash=claim_hash,
                        claim_type="factual",
                        importance=0.8 if evidence.accepted else 0.5,
                        status=derive_claim_status(
                            has_accepted_evidence=evidence.accepted,
                            has_refuting_evidence=(
                                evidence.accepted and evidence.relation == "refutes"
                            ),
                            independent_source_count=1 if evidence.accepted else 0,
                        ),
                        confidence=evidence.evidence_score if evidence.accepted else 0.0,
                        created_at=evidence.created_at,
                        updated_at=evidence.created_at,
                    )
                    session.add(claim)
                    await session.flush()
                    stats.claims_created += 1
                elif evidence.accepted:
                    claim.importance = max(claim.importance, 0.8)
                    claim.confidence = max(claim.confidence, evidence.evidence_score)

                chunk = None
                source_text = _read_artifact(artifact_root, source.artifact_uri)
                if source_text is None:
                    stats.missing_artifacts += 1
                else:
                    chunk_window = build_evidence_chunk(source_text, evidence.exact_quote)
                    if chunk_window is None:
                        stats.unlocated_quotes += 1
                        if evidence.accepted:
                            stats.accepted_unlocated_quotes += 1
                    else:
                        chunk = await session.scalar(
                            select(ResearchSourceChunkRow).where(
                                ResearchSourceChunkRow.snapshot_id == snapshot.id,
                                ResearchSourceChunkRow.chunk_hash == chunk_window.chunk_hash,
                            )
                        )
                        if chunk is None:
                            chunk = ResearchSourceChunkRow(
                                id=uuid7(),
                                run_id=evidence.run_id,
                                snapshot_id=snapshot.id,
                                heading_path=None,
                                char_start=chunk_window.char_start,
                                char_end=chunk_window.char_end,
                                text=chunk_window.text,
                                token_count=chunk_window.token_count,
                                chunk_hash=chunk_window.chunk_hash,
                            )
                            session.add(chunk)
                            await session.flush()
                            stats.chunks_created += 1

                evidence.claim_id = claim.id
                evidence.snapshot_id = snapshot.id
                evidence.chunk_id = chunk.id if chunk is not None else None
                await session.flush()
                independent_source_count = await session.scalar(
                    select(func.count(distinct(ResearchSourceRow.source_owner_key)))
                    .select_from(ResearchEvidenceRow)
                    .join(
                        ResearchSourceRow,
                        ResearchEvidenceRow.source_id == ResearchSourceRow.id,
                    )
                    .where(
                        ResearchEvidenceRow.claim_id == claim.id,
                        ResearchEvidenceRow.accepted.is_(True),
                    )
                )
                refuting_evidence_count = await session.scalar(
                    select(func.count(ResearchEvidenceRow.id)).where(
                        ResearchEvidenceRow.claim_id == claim.id,
                        ResearchEvidenceRow.accepted.is_(True),
                        ResearchEvidenceRow.relation == "refutes",
                    )
                )
                claim.status = derive_claim_status(
                    has_accepted_evidence=int(independent_source_count or 0) > 0,
                    has_refuting_evidence=int(refuting_evidence_count or 0) > 0,
                    independent_source_count=int(independent_source_count or 0),
                )
                stats.evidence_bound += 1
        return stats
    finally:
        await database.close()


def _read_artifact(root: Path, artifact_uri: str) -> str | None:
    relative = Path(artifact_uri)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    resolved = (root / relative).resolve()
    if root not in resolved.parents or not resolved.is_file():
        return None
    return resolved.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Commit the backfill")
    parser.add_argument("--run-id", type=UUID, help="Limit work to one Research Run")
    args = parser.parse_args()
    stats = asyncio.run(backfill(apply=args.apply, run_id=args.run_id))
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"Evidence Graph backfill [{mode}]")
    for field, value in asdict(stats).items():
        print(f"{field}: {value}")


if __name__ == "__main__":
    main()
