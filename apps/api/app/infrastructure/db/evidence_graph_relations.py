"""Deterministic Claim relation and Conflict persistence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.evidence_graph import derive_claim_status, infer_claim_relation
from app.domain.identifiers import uuid7
from app.infrastructure.db.evidence_graph_models import (
    ResearchClaimEdgeRow,
    ResearchClaimRow,
    ResearchConflictRow,
)
from app.infrastructure.db.research_models import ResearchEvidenceRow, ResearchSourceRow


@dataclass(frozen=True, slots=True)
class RelationRefreshStats:
    examined_pairs: int = 0
    detected_edges: int = 0
    created_edges: int = 0
    deleted_edges: int = 0
    detected_conflicts: int = 0
    created_conflicts: int = 0
    dismissed_conflicts: int = 0
    reopened_conflicts: int = 0


async def refresh_question_relations(
    session: AsyncSession,
    *,
    run_id: UUID,
    question_id: str,
    persist: bool = True,
) -> RelationRefreshStats:
    """Infer conservative Claim relations and persist idempotent graph records."""

    claims = (
        await session.scalars(
            select(ResearchClaimRow)
            .where(
                ResearchClaimRow.run_id == run_id,
                ResearchClaimRow.question_id == question_id,
            )
            .order_by(ResearchClaimRow.id)
        )
    ).all()
    accepted_rows = (
        (
            await session.execute(
                select(ResearchEvidenceRow, ResearchSourceRow)
                .join(ResearchSourceRow, ResearchEvidenceRow.source_id == ResearchSourceRow.id)
                .where(
                    ResearchEvidenceRow.run_id == run_id,
                    ResearchEvidenceRow.question_id == question_id,
                    ResearchEvidenceRow.accepted.is_(True),
                    ResearchEvidenceRow.claim_id.is_not(None),
                )
                .order_by(
                    ResearchEvidenceRow.claim_id,
                    ResearchEvidenceRow.evidence_score.desc(),
                    ResearchEvidenceRow.created_at,
                )
            )
        )
        .tuples()
        .all()
    )
    best_evidence: dict[UUID, tuple[ResearchEvidenceRow, ResearchSourceRow]] = {}
    source_owners_by_claim: dict[UUID, set[str]] = {}
    refuting_claim_ids: set[UUID] = set()
    for evidence, source in accepted_rows:
        if evidence.claim_id is not None:
            best_evidence.setdefault(evidence.claim_id, (evidence, source))
            source_owners_by_claim.setdefault(evidence.claim_id, set()).add(source.source_owner_key)
            if evidence.relation == "refutes":
                refuting_claim_ids.add(evidence.claim_id)

    examined_pairs = 0
    detected_edges = 0
    created_edges = 0
    deleted_edges = 0
    detected_conflicts = 0
    created_conflicts = 0
    dismissed_conflicts = 0
    reopened_conflicts = 0
    desired_edge_keys: set[tuple[UUID, UUID, str]] = set()
    desired_conflict_pairs: set[tuple[UUID, UUID]] = set()
    disputed_claim_ids: set[UUID] = set()
    now = datetime.now(UTC)
    for left, right in combinations(claims, 2):
        examined_pairs += 1
        decision = infer_claim_relation(left.atomic_claim, right.atomic_claim)
        if decision is None:
            continue
        detected_edges += 1
        from_claim, to_claim = _canonical_pair(left, right)
        desired_edge_keys.add((from_claim.id, to_claim.id, decision.relation))
        edge = await session.scalar(
            select(ResearchClaimEdgeRow).where(
                ResearchClaimEdgeRow.from_claim_id == from_claim.id,
                ResearchClaimEdgeRow.to_claim_id == to_claim.id,
                ResearchClaimEdgeRow.relation == decision.relation,
            )
        )
        if persist and edge is None:
            session.add(
                ResearchClaimEdgeRow(
                    id=uuid7(),
                    run_id=run_id,
                    from_claim_id=from_claim.id,
                    to_claim_id=to_claim.id,
                    relation=decision.relation,
                    confidence=decision.confidence,
                    created_at=now,
                )
            )
            created_edges += 1

        if decision.relation != "contradicts":
            continue
        left_item = best_evidence.get(left.id)
        right_item = best_evidence.get(right.id)
        if left_item is None or right_item is None:
            continue
        left_evidence, left_source = left_item
        right_evidence, right_source = right_item
        if left_source.source_owner_key == right_source.source_owner_key:
            continue
        detected_conflicts += 1
        conflict_left, conflict_right = _canonical_evidence_pair(
            left_evidence,
            right_evidence,
        )
        desired_conflict_pairs.add((conflict_left.id, conflict_right.id))
        disputed_claim_ids.update((left.id, right.id))
        conflict = await session.scalar(
            select(ResearchConflictRow).where(
                ResearchConflictRow.left_evidence_id == conflict_left.id,
                ResearchConflictRow.right_evidence_id == conflict_right.id,
            )
        )
        if persist and conflict is None:
            session.add(
                ResearchConflictRow(
                    id=uuid7(),
                    run_id=run_id,
                    question_id=question_id,
                    entity=f"question:{question_id}",
                    attribute=decision.reason_code,
                    time_scope=None,
                    geo_scope=None,
                    definition_scope="deterministic_relation_v2",
                    left_evidence_id=conflict_left.id,
                    right_evidence_id=conflict_right.id,
                    severity=decision.severity,
                    status="open",
                    resolution_summary=None,
                    created_at=now,
                    updated_at=now,
                )
            )
            created_conflicts += 1
        elif persist and conflict is not None:
            if conflict.status != "open":
                conflict.status = "open"
                conflict.resolution_summary = None
                reopened_conflicts += 1
            conflict.attribute = decision.reason_code
            conflict.definition_scope = "deterministic_relation_v2"
            conflict.severity = decision.severity
            conflict.updated_at = now

    if persist:
        claim_ids = {claim.id for claim in claims}
        if claim_ids:
            existing_edges = (
                await session.scalars(
                    select(ResearchClaimEdgeRow).where(
                        ResearchClaimEdgeRow.run_id == run_id,
                        ResearchClaimEdgeRow.from_claim_id.in_(claim_ids),
                        ResearchClaimEdgeRow.to_claim_id.in_(claim_ids),
                    )
                )
            ).all()
            for edge in existing_edges:
                edge_key = (edge.from_claim_id, edge.to_claim_id, edge.relation)
                if edge_key not in desired_edge_keys:
                    await session.delete(edge)
                    deleted_edges += 1

        existing_conflicts = (
            await session.scalars(
                select(ResearchConflictRow).where(
                    ResearchConflictRow.run_id == run_id,
                    ResearchConflictRow.question_id == question_id,
                    ResearchConflictRow.definition_scope.like("deterministic_relation_v%"),
                )
            )
        ).all()
        for conflict in existing_conflicts:
            conflict_key = (conflict.left_evidence_id, conflict.right_evidence_id)
            if conflict_key not in desired_conflict_pairs and conflict.status == "open":
                conflict.status = "dismissed"
                conflict.resolution_summary = (
                    "auto-dismissed: scope mismatch under deterministic_relation_v2"
                )
                conflict.updated_at = now
                dismissed_conflicts += 1

        for claim in claims:
            expected_status = derive_claim_status(
                has_accepted_evidence=claim.id in source_owners_by_claim,
                has_refuting_evidence=claim.id in refuting_claim_ids,
                independent_source_count=len(source_owners_by_claim.get(claim.id, set())),
            )
            if claim.id in disputed_claim_ids:
                expected_status = "disputed"
            if claim.status != expected_status:
                claim.status = expected_status
                claim.updated_at = now
        await session.flush()

    return RelationRefreshStats(
        examined_pairs=examined_pairs,
        detected_edges=detected_edges,
        created_edges=created_edges,
        deleted_edges=deleted_edges,
        detected_conflicts=detected_conflicts,
        created_conflicts=created_conflicts,
        dismissed_conflicts=dismissed_conflicts,
        reopened_conflicts=reopened_conflicts,
    )


def _canonical_pair(
    left: ResearchClaimRow,
    right: ResearchClaimRow,
) -> tuple[ResearchClaimRow, ResearchClaimRow]:
    return (left, right) if str(left.id) < str(right.id) else (right, left)


def _canonical_evidence_pair(
    left: ResearchEvidenceRow,
    right: ResearchEvidenceRow,
) -> tuple[ResearchEvidenceRow, ResearchEvidenceRow]:
    return (left, right) if str(left.id) < str(right.id) else (right, left)
