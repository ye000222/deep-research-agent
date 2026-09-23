"""Phase 13.2 Closure Feedback Dispatcher.

Bridges a :class:`ClosureFeedback` back into the research pipeline by turning it
into a :class:`ResearchNeed` and a :class:`SuggestedResearchAction`.  It is the
single, generic, idempotent hand-off point between closure feedback and the
existing query pipeline.

Design boundaries:

* It never executes Search, consumes Budget, or mutates a ``GapRequirement``.
* Every decision is based only on generic requirement/feedback state, never on
  ``question_id`` / ``dimension_key`` / ``gap_id`` identity or a benchmark case.
* The downstream ``Query Pipeline`` (intent -> plan -> candidate -> execution) is
  left untouched; the dispatcher only produces the need/action artifacts and the
  dispatch verdict that the runtime emits as ``research.feedback.dispatched``.
"""

from __future__ import annotations

from collections.abc import Container, Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from app.domain.closure_feedback import ClosureFeedback
from app.domain.evidence_alignment import EvidenceAlignment
from app.domain.gap_closure import GapClosureStatus, GapRequirement
from app.domain.research_action import ResearchActionGenerator, SuggestedResearchAction
from app.domain.research_need import ResearchNeed, ResearchNeedGenerator


class FeedbackDispatchStatus(StrEnum):
    CREATED = "created"
    SKIPPED = "skipped"
    DUPLICATED = "duplicated"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class FeedbackDispatchResult:
    """Outcome of dispatching one ClosureFeedback into the research pipeline."""

    feedback_id: UUID
    gap_id: UUID
    dispatch_status: FeedbackDispatchStatus
    reason: str
    research_need_id: UUID | None = None
    action_id: UUID | None = None
    research_needs: tuple[ResearchNeed, ...] = ()
    suggested_actions: tuple[SuggestedResearchAction, ...] = ()

    @property
    def created(self) -> bool:
        return self.dispatch_status is FeedbackDispatchStatus.CREATED

    def as_event_refs(self) -> dict[str, object]:
        """Return the persisted ``research.feedback.dispatched`` reference shape."""

        return {
            "feedback_id": str(self.feedback_id),
            "gap_id": str(self.gap_id),
            "research_need_id": (
                str(self.research_need_id) if self.research_need_id is not None else None
            ),
            "dispatch_status": self.dispatch_status.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ClosureFeedbackDispatcher:
    """Dispatch feedback into needs/actions with a generic, idempotent model.

    ``already_dispatched_feedback_ids`` lets callers persist the set of feedback
    ids that have already produced a research need, so the same ``feedback_id``
    can never spawn a second need across research rounds.
    """

    already_dispatched_feedback_ids: Container[str] = frozenset()

    def dispatch(
        self,
        feedback: ClosureFeedback,
        requirement: GapRequirement,
        *,
        alignments: Iterable[EvidenceAlignment] = (),
        now: datetime | None = None,
    ) -> FeedbackDispatchResult:
        """Bridge one feedback/requirement pair into the research pipeline."""

        # Rule: a CLOSED gap must not re-enter the research pipeline.
        if requirement.closure_status is GapClosureStatus.CLOSED:
            return self._verdict(
                feedback, FeedbackDispatchStatus.SKIPPED, "closed_gap"
            )
        # Guard: feedback must describe the requirement it is dispatched against.
        if feedback.gap_id != requirement.gap_id:
            return self._verdict(
                feedback,
                FeedbackDispatchStatus.BLOCKED,
                "feedback_requirement_mismatch",
            )
        # Rule: one feedback_id produces exactly one research need.
        if str(feedback.feedback_id) in self.already_dispatched_feedback_ids:
            return self._verdict(
                feedback,
                FeedbackDispatchStatus.DUPLICATED,
                "feedback_already_dispatched",
            )

        relevant_alignments = tuple(alignments)
        needs = ResearchNeedGenerator.generate(
            requirement,
            alignments=relevant_alignments,
            feedback=feedback,
            now=now,
        )
        if not needs:
            return self._verdict(
                feedback, FeedbackDispatchStatus.BLOCKED, "no_research_need_generated"
            )
        actions = tuple(
            action
            for need in needs
            for action in ResearchActionGenerator.generate(
                need,
                requirement=requirement,
                alignments=relevant_alignments,
                feedback=feedback,
                now=now,
            )
        )
        return FeedbackDispatchResult(
            feedback_id=feedback.feedback_id,
            gap_id=feedback.gap_id,
            dispatch_status=FeedbackDispatchStatus.CREATED,
            reason="research_need_created",
            research_need_id=needs[0].need_id,
            action_id=actions[0].action_id if actions else None,
            research_needs=needs,
            suggested_actions=actions,
        )

    def _verdict(
        self,
        feedback: ClosureFeedback,
        status: FeedbackDispatchStatus,
        reason: str,
    ) -> FeedbackDispatchResult:
        return FeedbackDispatchResult(
            feedback_id=feedback.feedback_id,
            gap_id=feedback.gap_id,
            dispatch_status=status,
            reason=reason,
        )
