"""Build observation-only Closure Evaluation input contexts."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.domain.evidence_alignment import (
    EvidenceAlignment,
    EvidenceAlignmentStatus,
)
from app.domain.gap_closure import ClosureSnapshot


@dataclass(frozen=True, slots=True)
class ClosureEvaluationContext:
    gap_id: UUID
    before_snapshot: ClosureSnapshot
    evidence_alignments: tuple[EvidenceAlignment, ...]
    new_evidence_count: int
    aligned_evidence_count: int
    partial_evidence_count: int
    independent_source_count: int

    @classmethod
    def from_alignments(
        cls,
        *,
        gap_id: UUID,
        before_snapshot: ClosureSnapshot,
        evidence_alignments: tuple[EvidenceAlignment, ...],
        independent_source_count: int,
    ) -> ClosureEvaluationContext:
        relevant = tuple(item for item in evidence_alignments if item.gap_id == gap_id)
        return cls(
            gap_id=gap_id,
            before_snapshot=before_snapshot,
            evidence_alignments=relevant,
            new_evidence_count=len(relevant),
            aligned_evidence_count=sum(
                item.alignment_status is EvidenceAlignmentStatus.ALIGNED
                for item in relevant
            ),
            partial_evidence_count=sum(
                item.alignment_status is EvidenceAlignmentStatus.PARTIAL
                for item in relevant
            ),
            independent_source_count=independent_source_count,
        )
