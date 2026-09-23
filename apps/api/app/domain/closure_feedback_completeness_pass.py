"""Phase 13.3 Closure Feedback Completeness Pass.

Activates the Phase 13.1 completeness framework inside the formal research
feedback loop.  The pass scans a set of ``GapRequirement`` states and, for every
``OPEN`` / ``PARTIAL`` requirement that carries *no* ``ClosureFeedback`` yet,
backfills one generic feedback through ``generate_completeness_feedback``.  The
generated feedbacks are then handed to the Phase 13.2 ``ClosureFeedbackDispatcher``
by the runtime, so both feedback sources converge on the exact same
``Feedback -> ResearchNeed -> SuggestedResearchAction -> Query Pipeline`` chain.

Design boundaries:

* Pure domain logic: no database, no Search, no Budget, no Planner mutation.
* Decisions depend only on generic requirement state (``closure_status``,
  evidence counts, requirement type, verification status) and on whether a
  ``feedback_id`` already covers the gap -- never on ``question_id`` /
  ``gap_id`` / ``dimension_key`` identity or any benchmark case.
* CLOSED gaps produce no feedback (Rule 1); gaps that already carry feedback
  are never fed twice (Rule 2); uncovered gaps are backfilled (Rule 3).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from app.domain.closure_feedback import ClosureFeedback
from app.domain.closure_feedback_completeness import generate_completeness_feedback
from app.domain.evidence_alignment import EvidenceAlignmentStatus
from app.domain.gap_closure import GapClosureStatus, GapRequirement


class FeedbackCompletenessPassStatus(StrEnum):
    GENERATED = "generated"
    SKIPPED = "skipped"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class GapCompletenessVerdict:
    """Per-gap outcome of one completeness pass."""

    gap_id: UUID
    status: FeedbackCompletenessPassStatus
    reason: str
    feedback_id: UUID | None = None

    def as_event_refs(
        self, *, run_id: UUID, question_id: str
    ) -> dict[str, object]:
        """Return the persisted ``research.feedback.completeness.checked`` refs."""

        return {
            "run_id": str(run_id),
            "question_id": question_id,
            "gap_id": str(self.gap_id),
            "feedback_id": str(self.feedback_id) if self.feedback_id is not None else None,
        }


@dataclass(frozen=True, slots=True)
class FeedbackCompletenessPassResult:
    """Aggregate outcome of one Closure Feedback Completeness Pass run."""

    status: FeedbackCompletenessPassStatus
    generated_feedbacks: tuple[ClosureFeedback, ...]
    skipped_existing: tuple[UUID, ...]
    skipped_closed: tuple[UUID, ...]
    violations: tuple[UUID, ...]
    verdicts: tuple[GapCompletenessVerdict, ...]

    @property
    def generated_count(self) -> int:
        return len(self.generated_feedbacks)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped_existing) + len(self.skipped_closed)

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    def as_event_metrics(self) -> dict[str, object]:
        return {
            "pass_status": self.status.value,
            "generated_count": self.generated_count,
            "skipped_count": self.skipped_count,
            "violation_count": self.violation_count,
        }


def run_closure_feedback_completeness_pass(
    gap_requirements: Iterable[GapRequirement],
    existing_feedback_ids_by_gap: Mapping[str, str],
    *,
    alignment_statuses: Mapping[UUID, EvidenceAlignmentStatus] | None = None,
    now: datetime | None = None,
) -> FeedbackCompletenessPassResult:
    """Backfill ClosureFeedback for every non-CLOSED gap that has none.

    ``existing_feedback_ids_by_gap`` maps ``str(gap_id) -> str(feedback_id)``
    for feedbacks already produced by either the closure-evaluation path or a
    previous completeness pass; those gaps are skipped so the pass is safe to
    re-run and never duplicates feedback.
    """

    status_lookup = alignment_statuses or {}
    generated: list[ClosureFeedback] = []
    skipped_existing: list[UUID] = []
    skipped_closed: list[UUID] = []
    violations: list[UUID] = []
    verdicts: list[GapCompletenessVerdict] = []
    for requirement in gap_requirements:
        gap_id = requirement.gap_id
        # Rule 1: CLOSED gaps never produce feedback.
        if requirement.closure_status is GapClosureStatus.CLOSED:
            skipped_closed.append(gap_id)
            verdicts.append(
                GapCompletenessVerdict(
                    gap_id=gap_id,
                    status=FeedbackCompletenessPassStatus.SKIPPED,
                    reason="closed_gap",
                )
            )
            continue
        # Rule 2: a gap that already carries feedback is never fed twice.
        existing_feedback_id = existing_feedback_ids_by_gap.get(str(gap_id))
        if existing_feedback_id is not None:
            skipped_existing.append(gap_id)
            verdicts.append(
                GapCompletenessVerdict(
                    gap_id=gap_id,
                    status=FeedbackCompletenessPassStatus.SKIPPED,
                    reason="feedback_exists",
                    feedback_id=UUID(str(existing_feedback_id)),
                )
            )
            continue
        # Rule 3: an uncovered OPEN/PARTIAL gap is a completeness violation and
        # must receive exactly one generic feedback.
        violations.append(gap_id)
        feedback = generate_completeness_feedback(
            requirement,
            alignment_status=status_lookup.get(
                gap_id, EvidenceAlignmentStatus.UNKNOWN
            ),
            now=now,
        )
        if feedback is None:  # defensive: non-CLOSED always yields feedback
            verdicts.append(
                GapCompletenessVerdict(
                    gap_id=gap_id,
                    status=FeedbackCompletenessPassStatus.SKIPPED,
                    reason="no_feedback_generated",
                )
            )
            continue
        generated.append(feedback)
        verdicts.append(
            GapCompletenessVerdict(
                gap_id=gap_id,
                status=FeedbackCompletenessPassStatus.GENERATED,
                reason="completeness_feedback_generated",
                feedback_id=feedback.feedback_id,
            )
        )

    if generated:
        pass_status = FeedbackCompletenessPassStatus.GENERATED
    elif violations:
        pass_status = FeedbackCompletenessPassStatus.SKIPPED
    else:
        pass_status = FeedbackCompletenessPassStatus.COMPLETE
    return FeedbackCompletenessPassResult(
        status=pass_status,
        generated_feedbacks=tuple(generated),
        skipped_existing=tuple(skipped_existing),
        skipped_closed=tuple(skipped_closed),
        violations=tuple(violations),
        verdicts=tuple(verdicts),
    )
