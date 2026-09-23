"""Pure Gap Requirement and Closure Evaluation prototypes.

This module is intentionally observation-only.  It does not update research
gaps, coverage, evidence acceptance, or recovery decisions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5


class GapRequirementType(StrEnum):
    DIMENSION_COVERAGE = "dimension_coverage"
    INDEPENDENT_SOURCE = "independent_source"
    CLAIM_VERIFICATION = "claim_verification"
    EVIDENCE_QUALITY = "evidence_quality"


class VerificationStatus(StrEnum):
    NOT_EVALUATED = "not_evaluated"
    REQUIRED = "required"
    VERIFIED = "verified"
    FAILED = "failed"


class GapClosureStatus(StrEnum):
    OPEN = "open"
    PARTIAL = "partial"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class ClosureSnapshot:
    """Immutable requirement facts at one evaluation point."""

    coverage: float = 0.0
    evidence_count: int = 0
    independent_sources: int = 0
    verification_status: VerificationStatus = VerificationStatus.NOT_EVALUATED

    def __post_init__(self) -> None:
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must be between 0 and 1")
        if self.evidence_count < 0:
            raise ValueError("evidence_count must be non-negative")
        if self.independent_sources < 0:
            raise ValueError("independent_sources must be non-negative")


@dataclass(frozen=True, slots=True)
class GapRequirement:
    """A versioned, question-level requirement that can be closed."""

    gap_id: UUID
    run_id: UUID
    question_id: str
    dimension_key: str
    requirement_type: GapRequirementType
    criterion: str
    required_evidence_count: int
    required_independent_sources: int
    current_evidence_count: int
    current_independent_sources: int
    verification_status: VerificationStatus
    closure_status: GapClosureStatus
    created_at: datetime
    updated_at: datetime
    state_version: int
    current_coverage: float = 0.0
    required_coverage: float | None = None
    transition_reason: str = "evaluator_projection"
    plan_version: int = 1
    migration_source: str = "runtime"

    def __post_init__(self) -> None:
        if self.required_evidence_count < 0 or self.current_evidence_count < 0:
            raise ValueError("evidence counts must be non-negative")
        if (
            self.required_independent_sources < 0
            or self.current_independent_sources < 0
        ):
            raise ValueError("independent source counts must be non-negative")
        if self.state_version < 1:
            raise ValueError("state_version must be positive")
        if self.plan_version < 1:
            raise ValueError("plan_version must be positive")
        if not 0.0 <= self.current_coverage <= 1.0:
            raise ValueError("current_coverage must be between 0 and 1")
        if self.required_coverage is not None and not 0.0 <= self.required_coverage <= 1.0:
            raise ValueError("required_coverage must be between 0 and 1")

    def apply_closure_transition(
        self,
        *,
        new_status: GapClosureStatus,
        transition_reason: str,
        updated_at: datetime | None = None,
    ) -> GapRequirement:
        """Apply one monotonic closure transition with a new state version."""

        if not transition_reason.strip():
            raise ValueError("transition_reason must not be empty")
        if _closure_rank(new_status) < _closure_rank(self.closure_status):
            raise ValueError("GapRequirement closure state cannot move backwards")
        if new_status is self.closure_status:
            return self
        return replace(
            self,
            closure_status=new_status,
            state_version=self.state_version + 1,
            transition_reason=transition_reason,
            updated_at=updated_at or datetime.now(UTC),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "gap_id": str(self.gap_id),
            "run_id": str(self.run_id),
            "question_id": self.question_id,
            "dimension_key": self.dimension_key,
            "requirement_type": self.requirement_type.value,
            "criterion": self.criterion,
            "required_evidence_count": self.required_evidence_count,
            "required_independent_sources": self.required_independent_sources,
            "current_evidence_count": self.current_evidence_count,
            "current_independent_sources": self.current_independent_sources,
            "current_coverage": self.current_coverage,
            "required_coverage": self.required_coverage,
            "verification_status": self.verification_status.value,
            "closure_status": self.closure_status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "state_version": self.state_version,
            "transition_reason": self.transition_reason,
            "plan_version": self.plan_version,
            "migration_source": self.migration_source,
        }


@dataclass(frozen=True, slots=True)
class ClosureEvaluation:
    """Pure before/after evaluation for one GapRequirement."""

    evaluation_id: UUID
    gap_id: UUID
    before_snapshot: ClosureSnapshot
    after_snapshot: ClosureSnapshot
    evidence_delta: int
    independent_source_delta: int
    requirement_satisfied: bool
    closure_status: GapClosureStatus
    reason: str

    @classmethod
    def evaluate(
        cls,
        requirement: GapRequirement,
        *,
        before_snapshot: ClosureSnapshot,
        after_snapshot: ClosureSnapshot,
    ) -> ClosureEvaluation:
        evidence_delta = after_snapshot.evidence_count - before_snapshot.evidence_count
        independent_source_delta = (
            after_snapshot.independent_sources - before_snapshot.independent_sources
        )
        requirement_satisfied = _requirement_is_satisfied(
            requirement, after_snapshot
        )
        if requirement_satisfied:
            closure_status = GapClosureStatus.CLOSED
            reason = "requirement_satisfied"
        elif _has_progress(before_snapshot, after_snapshot):
            closure_status = GapClosureStatus.PARTIAL
            reason = "evidence_or_verification_progress_without_closure"
        else:
            closure_status = GapClosureStatus.OPEN
            reason = "requirement_not_satisfied"
        return cls(
            evaluation_id=uuid4(),
            gap_id=requirement.gap_id,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            evidence_delta=evidence_delta,
            independent_source_delta=independent_source_delta,
            requirement_satisfied=requirement_satisfied,
            closure_status=closure_status,
            reason=reason,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "evaluation_id": str(self.evaluation_id),
            "gap_id": str(self.gap_id),
            "evidence_delta": self.evidence_delta,
            "independent_source_delta": self.independent_source_delta,
            "requirement_satisfied": self.requirement_satisfied,
            "closure_status": self.closure_status.value,
            "reason": self.reason,
        }


GAP_STATE_SOURCE_PRIORITY: tuple[str, ...] = (
    "closure_evaluation",
    "gap_requirement",
    "evaluator_snapshot",
    "recovery_outcome",
)


def _requirement_is_satisfied(
    requirement: GapRequirement,
    snapshot: ClosureSnapshot,
) -> bool:
    if requirement.requirement_type is GapRequirementType.DIMENSION_COVERAGE:
        required_coverage = (
            requirement.required_coverage
            if requirement.required_coverage is not None
            else 1.0
        )
        return snapshot.coverage >= required_coverage
    if requirement.requirement_type is GapRequirementType.INDEPENDENT_SOURCE:
        return (
            snapshot.independent_sources
            >= requirement.required_independent_sources
        )
    if requirement.requirement_type is GapRequirementType.CLAIM_VERIFICATION:
        # Claim verification closes from the Requirement/dimension-level numeric
        # independence fact (distinct accepted source owners >= required), which
        # is the canonical closure signal.  The persisted ``verification_status``
        # column is a legacy projection and is deliberately not treated as truth
        # here, so a Requirement is never stuck un-closed by a stale flag while
        # its dimension already satisfies the independence bar.
        return (
            snapshot.independent_sources
            >= requirement.required_independent_sources
        )
    return (
        snapshot.evidence_count >= requirement.required_evidence_count
        and snapshot.verification_status is VerificationStatus.VERIFIED
    )


def _closure_rank(status: GapClosureStatus) -> int:
    return {
        GapClosureStatus.OPEN: 0,
        GapClosureStatus.PARTIAL: 1,
        GapClosureStatus.CLOSED: 2,
    }[status]


def _has_progress(
    before_snapshot: ClosureSnapshot,
    after_snapshot: ClosureSnapshot,
) -> bool:
    return any(
        (
            after_snapshot.coverage > before_snapshot.coverage,
            after_snapshot.evidence_count > before_snapshot.evidence_count,
            after_snapshot.independent_sources
            > before_snapshot.independent_sources,
            after_snapshot.verification_status
            is not before_snapshot.verification_status,
        )
    )


def gap_requirement_id(
    run_id: UUID,
    plan_version: int,
    dimension_key: str,
) -> UUID:
    """Return a stable ID without introducing a persistence migration."""

    return uuid5(
        NAMESPACE_URL,
        f"gap-requirement:{run_id}:{plan_version}:{dimension_key}",
    )


def project_gap_requirements(
    *,
    run_id: UUID,
    plan_version: int,
    state_version: int,
    coverage_map: Sequence[Mapping[str, object]],
    claim_states: Mapping[str, Mapping[str, object]] | None = None,
    now: datetime | None = None,
) -> tuple[GapRequirement, ...]:
    """Project Evaluator requirement statuses into versioned domain records."""

    timestamp = now or datetime.now(UTC)
    known_claim_states = claim_states or {}
    projected: list[GapRequirement] = []
    for question in coverage_map:
        question_id = str(question.get("dimension_key", ""))
        statuses = question.get("requirement_statuses", ())
        if not isinstance(statuses, Sequence) or isinstance(statuses, (str, bytes)):
            continue
        for raw_status in statuses:
            if not isinstance(raw_status, Mapping):
                continue
            dimension_key = str(raw_status.get("dimension_key", ""))
            if not dimension_key:
                continue
            unresolved_claim = any(
                str(claim.get("dimension_key", "")) == dimension_key
                and bool(claim.get("unresolved", False))
                for claim in known_claim_states.values()
            )
            required_sources = _as_non_negative_int(
                raw_status.get("required_sources", 1)
            )
            requirement_type = (
                GapRequirementType.CLAIM_VERIFICATION
                if unresolved_claim
                else GapRequirementType.INDEPENDENT_SOURCE
                if required_sources > 1
                else GapRequirementType.EVIDENCE_QUALITY
            )
            coverage = _as_coverage(raw_status.get("coverage", 0.0))
            evidence_count = _as_non_negative_int(
                raw_status.get("accepted_evidence", 0)
            )
            independent_sources = _as_non_negative_int(
                raw_status.get("independent_sources", 0)
            )
            verification_status = (
                VerificationStatus.REQUIRED
                if unresolved_claim
                else VerificationStatus.VERIFIED
                if coverage >= 1.0
                else VerificationStatus.REQUIRED
            )
            requirement = GapRequirement(
                gap_id=gap_requirement_id(run_id, plan_version, dimension_key),
                run_id=run_id,
                question_id=question_id,
                dimension_key=dimension_key,
                requirement_type=requirement_type,
                criterion=str(raw_status.get("criterion", "")),
                required_evidence_count=1,
                required_independent_sources=required_sources,
                current_evidence_count=evidence_count,
                current_independent_sources=independent_sources,
                verification_status=verification_status,
                closure_status=GapClosureStatus.OPEN,
                created_at=timestamp,
                updated_at=timestamp,
                state_version=state_version,
                current_coverage=coverage,
                required_coverage=1.0,
                transition_reason="evaluator_projection",
                plan_version=plan_version,
            )
            closure_evaluation = ClosureEvaluation.evaluate(
                requirement,
                before_snapshot=ClosureSnapshot(),
                after_snapshot=ClosureSnapshot(
                    coverage=coverage,
                    evidence_count=evidence_count,
                    independent_sources=independent_sources,
                    verification_status=verification_status,
                ),
            )
            projected.append(
                replace(requirement, closure_status=closure_evaluation.closure_status)
            )
    return tuple(projected)


def _as_non_negative_int(value: object) -> int:
    if not isinstance(value, (int, float, str)):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _as_coverage(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        return 0.0
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def utc_now() -> datetime:
    """Return a timezone-aware timestamp for prototype Gap records."""

    return datetime.now(UTC)
