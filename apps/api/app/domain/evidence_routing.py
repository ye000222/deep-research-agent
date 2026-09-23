"""Route aligned search sources without invoking Reader or Evidence extraction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from uuid import NAMESPACE_URL, UUID, uuid5

from app.domain.search_result_alignment import (
    SearchResultAlignment,
    SearchResultAlignmentStatus,
)


class EvidenceRoutingStatus(StrEnum):
    ROUTE = "route"
    SKIP = "skip"
    DEFER = "defer"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class EvidenceRoutingDecision:
    """An observation-only decision about whether a source may enter Reader."""

    routing_id: UUID
    alignment_id: UUID
    source_id: str
    gap_id: UUID
    question_id: str
    decision: EvidenceRoutingStatus
    reason: str

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("routing source must not be empty")
        if not self.reason.strip():
            raise ValueError("routing reason must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "routing_id": str(self.routing_id),
            "alignment_id": str(self.alignment_id),
            "source_id": self.source_id,
            "gap_id": str(self.gap_id),
            "question_id": self.question_id,
            "decision": self.decision.value,
            "reason": self.reason,
        }


class EvidenceRoutingRouter:
    """Convert alignment observations into routing decisions only."""

    @staticmethod
    def route(
        alignment: SearchResultAlignment,
        *,
        evidence_potential: float | None = None,
    ) -> EvidenceRoutingDecision:
        if evidence_potential is not None and not 0 <= evidence_potential <= 1:
            raise ValueError("evidence potential must be between 0 and 1")

        if alignment.alignment_status is SearchResultAlignmentStatus.ALIGNED:
            return _decision(
                alignment,
                EvidenceRoutingStatus.ROUTE,
                "aligned search result is eligible for Reader",
            )
        if alignment.alignment_status is SearchResultAlignmentStatus.NOT_ALIGNED:
            return _decision(
                alignment,
                EvidenceRoutingStatus.SKIP,
                "search result does not match the target Gap",
            )
        if alignment.alignment_status is SearchResultAlignmentStatus.PARTIAL:
            if evidence_potential is not None and evidence_potential >= 0.8:
                return _decision(
                    alignment,
                    EvidenceRoutingStatus.ROUTE,
                    "partial alignment has high Evidence potential",
                )
            return _decision(
                alignment,
                EvidenceRoutingStatus.DEFER,
                "partial alignment requires more source-quality information",
            )
        return _decision(
            alignment,
            EvidenceRoutingStatus.UNKNOWN,
            "alignment status is insufficient for routing",
        )

    @staticmethod
    def route_many(
        alignments: Iterable[SearchResultAlignment],
        *,
        evidence_potentials: dict[UUID, float] | None = None,
    ) -> tuple[EvidenceRoutingDecision, ...]:
        potentials = evidence_potentials or {}
        return tuple(
            EvidenceRoutingRouter.route(
                alignment,
                evidence_potential=potentials.get(alignment.execution_id),
            )
            for alignment in alignments
        )


def _decision(
    alignment: SearchResultAlignment,
    decision: EvidenceRoutingStatus,
    reason: str,
) -> EvidenceRoutingDecision:
    return EvidenceRoutingDecision(
        routing_id=uuid5(
            NAMESPACE_URL,
            f"evidence-routing:{alignment.alignment_id}:{decision.value}",
        ),
        alignment_id=alignment.alignment_id,
        source_id=alignment.source_id,
        gap_id=alignment.gap_id,
        question_id=alignment.question_id,
        decision=decision,
        reason=reason,
    )
