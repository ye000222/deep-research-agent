"""Explain why a Gap remains open or partial after closure evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final
from uuid import NAMESPACE_URL, UUID, uuid5

from app.domain.closure_evaluator import ClosureEvaluationExecution
from app.domain.gap_closure import GapClosureStatus, GapRequirement
from app.domain.research_need import ResearchNeedType


class ClosureFeedbackReason(StrEnum):
    MISSING_INDEPENDENT_SOURCE = "missing_independent_source"
    CLAIM_NOT_VERIFIED = "claim_not_verified"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    EVIDENCE_QUALITY_LOW = "evidence_quality_low"
    DIMENSION_MISMATCH = "dimension_mismatch"
    NO_VALID_PATH = "no_valid_path"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ClosureFeedback:
    """An observation-only handoff from closure evaluation to future research."""

    feedback_id: UUID
    run_id: UUID
    question_id: str
    gap_id: UUID
    dimension_key: str
    requirement_type: str
    previous_status: GapClosureStatus
    current_status: GapClosureStatus
    closure_result: GapClosureStatus
    failure_reason: ClosureFeedbackReason
    missing_requirement: str
    recommended_need_type: ResearchNeedType
    created_at: datetime
    #: Phase 14.0 explanatory payload (evidence quality / claim analysis).
    #: Defaults to empty so every pre-14.0 construction site stays compatible.
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.missing_requirement.strip():
            raise ValueError("closure feedback missing requirement must not be empty")

    def as_dict(self) -> dict[str, object]:
        return {
            "feedback_id": str(self.feedback_id),
            "run_id": str(self.run_id),
            "question_id": self.question_id,
            "gap_id": str(self.gap_id),
            "dimension_key": self.dimension_key,
            "requirement_type": self.requirement_type,
            "previous_status": self.previous_status.value,
            "current_status": self.current_status.value,
            "closure_result": self.closure_result.value,
            "failure_reason": self.failure_reason.value,
            "missing_requirement": self.missing_requirement,
            "recommended_need_type": self.recommended_need_type.value,
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


class ClosureFeedbackGenerator:
    """Generate future-research feedback without scheduling or executing work."""

    @staticmethod
    def generate(
        execution: ClosureEvaluationExecution,
        *,
        now: datetime | None = None,
    ) -> tuple[ClosureFeedback, ...]:
        result = execution.result
        if result.after_status is GapClosureStatus.CLOSED:
            return ()

        requirement = execution.requirement
        reason = _feedback_reason(result.transition_reason, result)
        need_type = _need_type_for_reason(reason)
        missing = _missing_requirement(requirement, result, reason)
        timestamp = now or datetime.now(UTC)
        feedback_id = uuid5(
            NAMESPACE_URL,
            f"closure-feedback:{requirement.gap_id}:{result.state_version}:{reason.value}",
        )
        return (
            ClosureFeedback(
                feedback_id=feedback_id,
                run_id=requirement.run_id,
                question_id=requirement.question_id,
                gap_id=requirement.gap_id,
                dimension_key=requirement.dimension_key,
                requirement_type=requirement.requirement_type.value,
                previous_status=result.before_status,
                current_status=result.after_status,
                closure_result=result.closure_status,
                failure_reason=reason,
                missing_requirement=missing,
                recommended_need_type=need_type,
                created_at=timestamp,
            ),
        )


def _feedback_reason(
    transition_reason: str,
    result: object,
) -> ClosureFeedbackReason:
    mapping = {
        "missing_independent_source": ClosureFeedbackReason.MISSING_INDEPENDENT_SOURCE,
        "claim_verification_required": ClosureFeedbackReason.CLAIM_NOT_VERIFIED,
        "dimension_mismatch": ClosureFeedbackReason.DIMENSION_MISMATCH,
        "insufficient_evidence": ClosureFeedbackReason.INSUFFICIENT_EVIDENCE,
    }
    if transition_reason in mapping:
        return mapping[transition_reason]
    if transition_reason == "requirement_not_satisfied":
        aligned = getattr(result, "aligned_evidence_count", 0)
        partial = getattr(result, "partial_evidence_count", 0)
        if aligned == 0 and partial == 0:
            return ClosureFeedbackReason.NO_VALID_PATH
        return ClosureFeedbackReason.INSUFFICIENT_EVIDENCE
    if transition_reason == "evidence_or_verification_progress_without_closure":
        return ClosureFeedbackReason.INSUFFICIENT_EVIDENCE
    return ClosureFeedbackReason.UNKNOWN


_REASON_TO_NEED_TYPE: Final[
    dict[ClosureFeedbackReason, ResearchNeedType]
] = {
    ClosureFeedbackReason.MISSING_INDEPENDENT_SOURCE: ResearchNeedType.INDEPENDENT_SOURCE,
    ClosureFeedbackReason.CLAIM_NOT_VERIFIED: ResearchNeedType.CLAIM_VERIFICATION,
    ClosureFeedbackReason.INSUFFICIENT_EVIDENCE: ResearchNeedType.EVIDENCE_QUALITY,
    ClosureFeedbackReason.EVIDENCE_QUALITY_LOW: ResearchNeedType.EVIDENCE_QUALITY,
    ClosureFeedbackReason.DIMENSION_MISMATCH: ResearchNeedType.DIMENSION_COVERAGE,
    ClosureFeedbackReason.NO_VALID_PATH: ResearchNeedType.ADDITIONAL_SOURCE,
    ClosureFeedbackReason.UNKNOWN: ResearchNeedType.UNKNOWN,
}


def need_type_for_reason(reason: ClosureFeedbackReason) -> ResearchNeedType:
    """Return the ResearchNeed type a failure reason recommends.

    Exposed publicly so the Phase 13.1 completeness framework can reuse the
    exact same reason -> need mapping as the closure-evaluation feedback path.
    """

    return _REASON_TO_NEED_TYPE[reason]


def _need_type_for_reason(reason: ClosureFeedbackReason) -> ResearchNeedType:
    return need_type_for_reason(reason)


def _missing_requirement(
    requirement: GapRequirement,
    result: object,
    reason: ClosureFeedbackReason,
) -> str:
    if reason is ClosureFeedbackReason.MISSING_INDEPENDENT_SOURCE:
        return "independent_source"
    if reason is ClosureFeedbackReason.CLAIM_NOT_VERIFIED:
        return "claim_verification"
    remaining = getattr(result, "remaining_requirements", ())
    if remaining:
        return str(remaining[0])
    return requirement.dimension_key
