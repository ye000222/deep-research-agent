"""Compatibility projection from canonical GapRequirement state.

The persisted ``research_gaps`` table predates the versioned gap model and is
still referenced by historical rows and foreign keys.  This module keeps that
shape available to old consumers while making ``GapRequirement`` the source
of truth whenever it is present.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from uuid import UUID

from app.domain.gap_closure import (
    ClosureEvaluation,
    ClosureSnapshot,
    GapClosureStatus,
    GapRequirement,
)


@dataclass(frozen=True, slots=True)
class ResearchGapProjection:
    """A legacy-compatible view of one canonical requirement."""

    gap_id: UUID
    run_id: UUID
    plan_version: int
    question_id: str
    dimension_key: str
    status: GapClosureStatus
    state_version: int
    updated_at: datetime
    transition_reason: str = "evaluator_projection"
    migration_source: str = "runtime"
    source: str = "gap_requirement"

    @classmethod
    def from_requirement(
        cls,
        requirement: GapRequirement,
        *,
        plan_version: int,
    ) -> ResearchGapProjection:
        return cls(
            gap_id=requirement.gap_id,
            run_id=requirement.run_id,
            plan_version=plan_version,
            question_id=requirement.question_id,
            dimension_key=requirement.dimension_key,
            status=requirement.closure_status,
            state_version=requirement.state_version,
            updated_at=requirement.updated_at,
            transition_reason=requirement.transition_reason,
            migration_source=requirement.migration_source,
        )

    @classmethod
    def from_legacy_record(
        cls,
        *,
        gap_id: UUID,
        run_id: UUID,
        plan_version: int,
        question_id: str,
        dimension_key: str,
        status: str,
        state_version: int = 1,
        updated_at: datetime,
    ) -> ResearchGapProjection:
        """Build an explicit fallback for pre-migration historical data."""

        canonical_status = (
            GapClosureStatus.CLOSED if status.casefold() == "resolved" else GapClosureStatus.OPEN
        )
        return cls(
            gap_id=gap_id,
            run_id=run_id,
            plan_version=plan_version,
            question_id=question_id,
            dimension_key=dimension_key,
            status=canonical_status,
            state_version=state_version,
            updated_at=updated_at,
            transition_reason="legacy_compatibility",
            migration_source="legacy_compatibility",
            source="legacy_compatibility",
        )

    @property
    def legacy_status(self) -> str:
        """Return the status expected by old consumers."""

        return "resolved" if self.status is GapClosureStatus.CLOSED else "open"

    @property
    def unresolved(self) -> bool:
        return self.status is not GapClosureStatus.CLOSED

    def as_dict(self) -> dict[str, object]:
        return {
            "gap_id": str(self.gap_id),
            "run_id": str(self.run_id),
            "plan_version": self.plan_version,
            "question_id": self.question_id,
            "dimension_key": self.dimension_key,
            "status": self.status.value,
            "legacy_status": self.legacy_status,
            "unresolved": self.unresolved,
            "state_version": self.state_version,
            "transition_reason": self.transition_reason,
            "migration_source": self.migration_source,
            "updated_at": self.updated_at.isoformat(),
            "source": self.source,
        }


def project_research_gaps(
    requirements: Iterable[GapRequirement],
    *,
    plan_version: int,
) -> tuple[ResearchGapProjection, ...]:
    """Project canonical requirements without consulting legacy gap rows."""

    return tuple(
        ResearchGapProjection.from_requirement(
            requirement,
            plan_version=plan_version,
        )
        for requirement in requirements
    )


def requirements_from_snapshot(
    raw_requirements: object,
) -> tuple[dict[str, object], ...]:
    """Return only well-formed serialized requirements from a run snapshot."""

    if not isinstance(raw_requirements, list):
        return ()
    return tuple(
        dict(item)
        for item in raw_requirements
        if isinstance(item, Mapping)
        and item.get("gap_id")
        and item.get("question_id")
        and item.get("dimension_key")
    )


def apply_closure_evaluation(
    requirement: GapRequirement,
    evaluation: ClosureEvaluation,
    *,
    now: datetime,
    transition_reason: str | None = None,
) -> GapRequirement:
    """Apply one evaluated transition to canonical requirement state."""

    if evaluation.gap_id != requirement.gap_id:
        raise ValueError("closure evaluation does not belong to requirement")
    rank = {
        GapClosureStatus.OPEN: 0,
        GapClosureStatus.PARTIAL: 1,
        GapClosureStatus.CLOSED: 2,
    }
    if rank[evaluation.closure_status] < rank[requirement.closure_status]:
        raise ValueError("gap closure state cannot move backwards")
    after = evaluation.after_snapshot
    return replace(
        requirement,
        current_coverage=after.coverage,
        current_evidence_count=after.evidence_count,
        current_independent_sources=after.independent_sources,
        verification_status=after.verification_status,
        closure_status=evaluation.closure_status,
        updated_at=now,
        state_version=requirement.state_version + 1,
        transition_reason=transition_reason or evaluation.reason,
    )


def synchronize_gap_requirements(
    previous: Iterable[GapRequirement],
    current: Iterable[GapRequirement],
    *,
    now: datetime,
) -> tuple[GapRequirement, ...]:
    """Re-evaluate current facts against prior state before persisting them."""

    previous_by_id = {item.gap_id: item for item in previous}
    synchronized: list[GapRequirement] = []
    for candidate in current:
        prior = previous_by_id.get(candidate.gap_id)
        if prior is None:
            synchronized.append(candidate)
            continue
        evaluation = ClosureEvaluation.evaluate(
            prior,
            before_snapshot=ClosureSnapshot(
                coverage=prior.current_coverage,
                evidence_count=prior.current_evidence_count,
                independent_sources=prior.current_independent_sources,
                verification_status=prior.verification_status,
            ),
            after_snapshot=ClosureSnapshot(
                coverage=candidate.current_coverage,
                evidence_count=candidate.current_evidence_count,
                independent_sources=candidate.current_independent_sources,
                verification_status=candidate.verification_status,
            ),
        )
        synchronized.append(
            apply_closure_evaluation(
                prior,
                evaluation,
                now=now,
            )
        )
    return tuple(synchronized)
