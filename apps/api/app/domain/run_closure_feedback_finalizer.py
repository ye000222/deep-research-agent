"""Phase 13.4 Run-Level Closure Feedback Finalizer.

Last line of defence before a Research Run leaves the research phase and enters
report writing.  The finalizer scans *every* ``GapRequirement`` of the run's
current plan and guarantees the explainability invariant:

    every OPEN / PARTIAL gap either already carries ClosureFeedback (possibly
    produced by the closure-evaluation path, a completeness pass, or a previous
    finalization), or has an explicit terminal reason, or receives a generic
    completeness feedback here -- never a silent leftover.

Generated feedbacks are dispatched by the runtime through the Phase 13.2
``ClosureFeedbackDispatcher`` and the shared feedback chain, so no second set of
events or artifacts is introduced.

Design boundaries:

* Pure domain logic: no database, no Search, no Budget, no Planner mutation.
* Decisions depend only on ``ResearchRun`` state (the stop reason) and generic
  ``GapRequirement`` state -- never on ``question_id`` / ``gap_id`` /
  ``dimension_key`` identity or any benchmark case.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final
from uuid import UUID

from app.domain.closure_feedback import ClosureFeedback
from app.domain.closure_feedback_completeness import generate_completeness_feedback
from app.domain.evidence_alignment import EvidenceAlignmentStatus
from app.domain.gap_closure import GapClosureStatus, GapRequirement


class RunClosureFeedbackFinalizationStatus(StrEnum):
    COMPLETED = "completed"
    GENERATED = "generated"
    BLOCKED = "blocked"


class GapFinalizationAction(StrEnum):
    EXISTING_COMPLETE = "existing_complete"
    EXISTING_FEEDBACK = "existing_feedback"
    GENERATED_FEEDBACK = "generated_feedback"
    TERMINAL = "terminal"
    BLOCKED = "blocked"


# Explicit user/operator stops that are terminal even without a suffix.
_RUN_TERMINAL_STOP_REASONS: Final[frozenset[str]] = frozenset(
    {"cancelled", "user_stopped"}
)


def is_terminal_run_stop_reason(reason: str | None) -> bool:
    """Return whether a run stop reason is an explicit terminal condition.

    Terminal means research was cut short by an exhausted resource or an
    explicit stop (``*_exhausted``, ``cancelled``, ``user_stopped``); such a
    reason explains every still-open gap without needing new feedback (Rule 4).
    Non-terminal endings (e.g. ``quality_met`` or ``stagnation``) do not
    explain individually unaddressed gaps, so feedback backfill still applies.
    """

    if not reason:
        return False
    if reason in _RUN_TERMINAL_STOP_REASONS:
        return True
    return reason.endswith("_exhausted")


@dataclass(frozen=True, slots=True)
class GapFinalizationVerdict:
    """How one gap was explained when the run was finalized."""

    gap_id: UUID
    closure_status: GapClosureStatus
    feedback_exists: bool
    terminal_reason: str | None
    action: GapFinalizationAction
    feedback_id: UUID | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "gap_id": str(self.gap_id),
            "closure_status": self.closure_status.value,
            "feedback_exists": self.feedback_exists,
            "terminal_reason": self.terminal_reason,
            "action": self.action.value,
            "feedback_id": str(self.feedback_id) if self.feedback_id is not None else None,
        }


@dataclass(frozen=True, slots=True)
class RunClosureFeedbackFinalizationResult:
    """Aggregate outcome of one run-level closure feedback finalization."""

    run_id: UUID
    status: RunClosureFeedbackFinalizationStatus
    generated_feedbacks: tuple[ClosureFeedback, ...]
    existing_feedbacks: tuple[UUID, ...]
    terminal_gaps: tuple[UUID, ...]
    blocked_gaps: tuple[UUID, ...]
    violations: tuple[UUID, ...]
    verdicts: tuple[GapFinalizationVerdict, ...]

    @property
    def total_gaps(self) -> int:
        return len(self.verdicts)

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": str(self.run_id),
            "status": self.status.value,
            "total_gaps": self.total_gaps,
            "generated_feedback_ids": [
                str(feedback.feedback_id) for feedback in self.generated_feedbacks
            ],
            "generated_feedback_count": len(self.generated_feedbacks),
            "existing_feedback_count": len(self.existing_feedbacks),
            "terminal_gap_count": len(self.terminal_gaps),
            "blocked_gap_count": len(self.blocked_gaps),
            "violation_count": len(self.violations),
            "verdicts": [verdict.as_dict() for verdict in self.verdicts],
        }


def finalize_run_closure_feedback(
    *,
    run_id: UUID,
    gap_requirements: Iterable[GapRequirement],
    existing_feedback_ids_by_gap: Mapping[str, str],
    terminal_reasons: Mapping[str, str],
    alignment_statuses: Mapping[UUID, EvidenceAlignmentStatus] | None = None,
    now: datetime | None = None,
) -> RunClosureFeedbackFinalizationResult:
    """Explain every non-CLOSED gap before the run finishes.

    ``existing_feedback_ids_by_gap`` maps ``str(gap_id) -> str(feedback_id)``
    for feedback already produced by any path; ``terminal_reasons`` maps
    ``str(gap_id) -> str(reason)`` for gaps whose run gave an explicit terminal
    reason.  Gaps covered by neither receive one generic completeness
    feedback (Rule 5) so nothing is left silently unexplained.
    """

    status_lookup = alignment_statuses or {}
    generated: list[ClosureFeedback] = []
    existing_feedbacks: list[UUID] = []
    terminal_gaps: list[UUID] = []
    blocked_gaps: list[UUID] = []
    violations: list[UUID] = []
    verdicts: list[GapFinalizationVerdict] = []
    for requirement in gap_requirements:
        gap_id = requirement.gap_id
        closure_status = requirement.closure_status
        # Rule 1: CLOSED gaps are complete and need no explanation.
        if closure_status is GapClosureStatus.CLOSED:
            verdicts.append(
                GapFinalizationVerdict(
                    gap_id=gap_id,
                    closure_status=closure_status,
                    feedback_exists=False,
                    terminal_reason=None,
                    action=GapFinalizationAction.EXISTING_COMPLETE,
                )
            )
            continue
        # Rule 2: a gap that already carries feedback is never fed twice.
        existing_feedback_id = existing_feedback_ids_by_gap.get(str(gap_id))
        if existing_feedback_id is not None:
            existing_feedbacks.append(gap_id)
            verdicts.append(
                GapFinalizationVerdict(
                    gap_id=gap_id,
                    closure_status=closure_status,
                    feedback_exists=True,
                    terminal_reason=None,
                    action=GapFinalizationAction.EXISTING_FEEDBACK,
                    feedback_id=UUID(str(existing_feedback_id)),
                )
            )
            continue
        # Rule 4: an explicit terminal reason explains the gap as-is.
        terminal_reason = terminal_reasons.get(str(gap_id))
        if terminal_reason is not None:
            terminal_gaps.append(gap_id)
            verdicts.append(
                GapFinalizationVerdict(
                    gap_id=gap_id,
                    closure_status=closure_status,
                    feedback_exists=False,
                    terminal_reason=terminal_reason,
                    action=GapFinalizationAction.TERMINAL,
                )
            )
            continue
        # Rule 5: no feedback and no terminal reason -> must generate.
        violations.append(gap_id)
        feedback = generate_completeness_feedback(
            requirement,
            alignment_status=status_lookup.get(
                gap_id, EvidenceAlignmentStatus.UNKNOWN
            ),
            now=now,
        )
        if feedback is None:  # defensive: non-CLOSED always yields feedback
            blocked_gaps.append(gap_id)
            verdicts.append(
                GapFinalizationVerdict(
                    gap_id=gap_id,
                    closure_status=closure_status,
                    feedback_exists=False,
                    terminal_reason=None,
                    action=GapFinalizationAction.BLOCKED,
                )
            )
            continue
        generated.append(feedback)
        verdicts.append(
            GapFinalizationVerdict(
                gap_id=gap_id,
                closure_status=closure_status,
                feedback_exists=False,
                terminal_reason=None,
                action=GapFinalizationAction.GENERATED_FEEDBACK,
                feedback_id=feedback.feedback_id,
            )
        )

    if blocked_gaps:
        status = RunClosureFeedbackFinalizationStatus.BLOCKED
    elif generated:
        status = RunClosureFeedbackFinalizationStatus.GENERATED
    else:
        status = RunClosureFeedbackFinalizationStatus.COMPLETED
    return RunClosureFeedbackFinalizationResult(
        run_id=run_id,
        status=status,
        generated_feedbacks=tuple(generated),
        existing_feedbacks=tuple(existing_feedbacks),
        terminal_gaps=tuple(terminal_gaps),
        blocked_gaps=tuple(blocked_gaps),
        violations=tuple(violations),
        verdicts=tuple(verdicts),
    )
