"""Phase 13.1 Closure Feedback Completeness framework.

The runtime closure path only emits ``ClosureFeedback`` when a ``GapRequirement``
happens to flow through a :class:`ClosureEvaluationExecution`.  A requirement can
therefore be unsatisfied (``OPEN`` / ``PARTIAL``) yet carry *no* feedback, which
breaks the ``Gap -> Feedback -> ResearchNeed -> Action`` chain and hides the gap
from every later planning round.

This module closes that hole generically.  It is a pure, observation-only
capability: it never touches Provider, Search, Budget, Planner, Evidence
Acceptance, Coverage or the Research Loop.  Every rule is derived *only* from
generic requirement/alignment state, so it is identical for any question,
dimension, gap id or benchmark case.

Guarantee (the completeness invariant)::

    for requirement in gap_requirements:
        if requirement.closure_status is not CLOSED:
            assert generate_completeness_feedback(requirement) is not None
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final
from uuid import NAMESPACE_URL, UUID, uuid5

from app.domain.closure_feedback import (
    ClosureFeedback,
    ClosureFeedbackReason,
    need_type_for_reason,
)
from app.domain.evidence_alignment import EvidenceAlignmentStatus
from app.domain.gap_closure import (
    GapClosureStatus,
    GapRequirement,
    GapRequirementType,
    VerificationStatus,
)

MISSING_FEEDBACK_REASON: Final[str] = "NO_CLOSURE_FEEDBACK"


@dataclass(frozen=True, slots=True)
class GapFeedbackCompleteness:
    """Read-only verdict on whether one GapRequirement has ClosureFeedback."""

    gap_id: str
    has_feedback: bool
    feedback_id: str | None
    missing_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "gap_id": self.gap_id,
            "has_feedback": self.has_feedback,
            "feedback_id": self.feedback_id,
            "missing_reason": self.missing_reason,
        }


def derive_feedback_reason(
    requirement: GapRequirement,
    alignment_status: EvidenceAlignmentStatus = EvidenceAlignmentStatus.UNKNOWN,
) -> ClosureFeedbackReason:
    """Map unsatisfied requirement state onto exactly one feedback reason.

    The rules are evaluated in a fixed, documented priority so the result is
    deterministic and mutually exclusive.  Only generic scalar state is read:
    evidence counts, independent-source counts, requirement type, verification
    status and the evidence-alignment verdict.
    """

    # Rule 1: not enough accepted evidence for the requirement yet.
    if requirement.current_evidence_count < requirement.required_evidence_count:
        return ClosureFeedbackReason.INSUFFICIENT_EVIDENCE
    # Rule 2: enough evidence, but too few independent sources.
    if (
        requirement.current_independent_sources
        < requirement.required_independent_sources
    ):
        return ClosureFeedbackReason.MISSING_INDEPENDENT_SOURCE
    # Rule 3: a claim-verification requirement is still unverified.
    if (
        requirement.requirement_type is GapRequirementType.CLAIM_VERIFICATION
        and requirement.verification_status is not VerificationStatus.VERIFIED
    ):
        return ClosureFeedbackReason.CLAIM_NOT_VERIFIED
    # Rule 5: evidence exists but is aligned to the wrong dimension.
    if alignment_status is EvidenceAlignmentStatus.NOT_ALIGNED:
        return ClosureFeedbackReason.DIMENSION_MISMATCH
    # Rule 4: evidence exists but is only partial / low quality.
    if alignment_status is EvidenceAlignmentStatus.PARTIAL:
        return ClosureFeedbackReason.EVIDENCE_QUALITY_LOW
    if requirement.requirement_type is GapRequirementType.EVIDENCE_QUALITY:
        return ClosureFeedbackReason.EVIDENCE_QUALITY_LOW
    # Rule 6: no specific signal explains the still-open gap.
    return ClosureFeedbackReason.NO_VALID_PATH


def generate_completeness_feedback(
    requirement: GapRequirement,
    *,
    alignment_status: EvidenceAlignmentStatus = EvidenceAlignmentStatus.UNKNOWN,
    now: datetime | None = None,
) -> ClosureFeedback | None:
    """Derive ClosureFeedback for any non-CLOSED requirement; CLOSED yields none.

    ``Requirement 3``: a ``CLOSED`` gap must not produce feedback, a research
    need, or a suggested action, so this returns ``None`` for it.
    """

    if requirement.closure_status is GapClosureStatus.CLOSED:
        return None
    reason = derive_feedback_reason(requirement, alignment_status)
    feedback_id = uuid5(
        NAMESPACE_URL,
        "closure-feedback-completeness:"
        f"{requirement.gap_id}:{requirement.state_version}:{reason.value}",
    )
    return ClosureFeedback(
        feedback_id=feedback_id,
        run_id=requirement.run_id,
        question_id=requirement.question_id,
        gap_id=requirement.gap_id,
        dimension_key=requirement.dimension_key,
        requirement_type=requirement.requirement_type.value,
        previous_status=requirement.closure_status,
        current_status=requirement.closure_status,
        closure_result=requirement.closure_status,
        failure_reason=reason,
        missing_requirement=_missing_requirement(requirement, reason),
        recommended_need_type=need_type_for_reason(reason),
        created_at=now or datetime.now(UTC),
    )


def check_feedback_completeness(
    gap_requirements: Iterable[GapRequirement],
    existing_feedbacks: Iterable[ClosureFeedback],
) -> tuple[GapFeedbackCompleteness, ...]:
    """Report, per non-CLOSED gap, whether a ClosureFeedback already exists."""

    feedback_id_by_gap = _feedback_id_by_gap(existing_feedbacks)
    records: list[GapFeedbackCompleteness] = []
    for requirement in gap_requirements:
        if requirement.closure_status is GapClosureStatus.CLOSED:
            continue
        gap_id = str(requirement.gap_id)
        known_feedback_id = feedback_id_by_gap.get(gap_id)
        if known_feedback_id is not None:
            records.append(
                GapFeedbackCompleteness(
                    gap_id=gap_id,
                    has_feedback=True,
                    feedback_id=known_feedback_id,
                    missing_reason=None,
                )
            )
        else:
            records.append(
                GapFeedbackCompleteness(
                    gap_id=gap_id,
                    has_feedback=False,
                    feedback_id=None,
                    missing_reason=MISSING_FEEDBACK_REASON,
                )
            )
    return tuple(records)


def missing_feedback_requirements(
    gap_requirements: Iterable[GapRequirement],
    existing_feedbacks: Iterable[ClosureFeedback],
) -> tuple[GapRequirement, ...]:
    """Return non-CLOSED requirements that currently have no ClosureFeedback."""

    covered = set(_feedback_id_by_gap(existing_feedbacks))
    return tuple(
        requirement
        for requirement in gap_requirements
        if requirement.closure_status is not GapClosureStatus.CLOSED
        and str(requirement.gap_id) not in covered
    )


def complete_feedback_gaps(
    gap_requirements: Iterable[GapRequirement],
    existing_feedbacks: Iterable[ClosureFeedback],
    *,
    alignment_statuses: Mapping[UUID, EvidenceAlignmentStatus] | None = None,
    now: datetime | None = None,
) -> tuple[ClosureFeedback, ...]:
    """Backfill ClosureFeedback for every gap that is missing one.

    Gaps that already carry feedback are left untouched so this is safe to run
    alongside the closure-evaluation feedback path.
    """

    status_lookup = alignment_statuses or {}
    backfilled: list[ClosureFeedback] = []
    for requirement in missing_feedback_requirements(
        gap_requirements, existing_feedbacks
    ):
        feedback = generate_completeness_feedback(
            requirement,
            alignment_status=status_lookup.get(
                requirement.gap_id, EvidenceAlignmentStatus.UNKNOWN
            ),
            now=now,
        )
        if feedback is not None:
            backfilled.append(feedback)
    return tuple(backfilled)


def feedback_completeness_violations(
    gap_requirements: Iterable[GapRequirement],
    feedbacks: Iterable[ClosureFeedback],
) -> tuple[str, ...]:
    """Return gap ids that are non-CLOSED yet have zero ClosureFeedback.

    An empty tuple means the completeness invariant holds.
    """

    covered = set(_feedback_id_by_gap(feedbacks))
    return tuple(
        str(requirement.gap_id)
        for requirement in gap_requirements
        if requirement.closure_status is not GapClosureStatus.CLOSED
        and str(requirement.gap_id) not in covered
    )


def _feedback_id_by_gap(
    feedbacks: Iterable[ClosureFeedback],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for feedback in feedbacks:
        mapping.setdefault(str(feedback.gap_id), str(feedback.feedback_id))
    return mapping


def _missing_requirement(
    requirement: GapRequirement,
    reason: ClosureFeedbackReason,
) -> str:
    if reason is ClosureFeedbackReason.MISSING_INDEPENDENT_SOURCE:
        return "independent_source"
    if reason is ClosureFeedbackReason.CLAIM_NOT_VERIFIED:
        return "claim_verification"
    return requirement.dimension_key
