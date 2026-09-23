"""Question-level research state projection.

Query-family execution is intentionally kept separate from the question's
actual research opportunity.  The latter is derived from observable search,
reader, evidence, evaluation, and resource facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FirstPassExecutionState(StrEnum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class ResearchOpportunityState(StrEnum):
    NOT_STARTED = "not_started"
    ACTIVE = "active"
    COMPLETED_WITH_GAP = "completed_with_gap"
    COMPLETED_QUALITY = "completed_quality"
    BLOCKED_TECHNICAL = "blocked_technical"
    BLOCKED_RESOURCE = "blocked_resource"


class RecoveryEligibilityState(StrEnum):
    NOT_EVALUATED = "not_evaluated"
    ELIGIBLE = "eligible"
    DENIED = "denied"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class QuestionResearchFacts:
    """Observable facts used to project one question's current state."""

    question_id: str
    search_queries_started: int = 0
    search_queries_completed: int = 0
    search_queries_failed: int = 0
    readable_sources: int = 0
    evidence_extraction_started: int = 0
    candidate_evidence: int = 0
    accepted_evidence: int = 0
    evaluation_completed: bool = False
    coverage: float = 0.0
    gap_open: bool = False
    critical_gap: bool = False
    replan_count: int = 0
    gap_resolution_attempts: int = 0
    query_family_marker_present: bool = False
    technical_blocked: bool = False
    resource_blocked: bool = False
    budget_blocked: bool = False
    budget_blocking_reason: str | None = None
    has_recovery_path: bool = True
    quality_met: bool = False


@dataclass(frozen=True, slots=True)
class QuestionResearchState:
    """Independent execution, opportunity, and recovery projections."""

    question_id: str
    first_pass_execution: FirstPassExecutionState
    research_opportunity: ResearchOpportunityState
    recovery_eligibility: RecoveryEligibilityState
    blocking_reason: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "question_id": self.question_id,
            "first_pass_execution": self.first_pass_execution.value,
            "research_opportunity": self.research_opportunity.value,
            "recovery_eligibility": self.recovery_eligibility.value,
            "blocking_reason": self.blocking_reason,
        }


def project_question_research_state(
    facts: QuestionResearchFacts,
) -> QuestionResearchState:
    """Project independent Question states from observable run facts."""

    coverage = min(1.0, max(0.0, float(facts.coverage)))
    downstream_signal = any(
        (
            facts.readable_sources > 0,
            facts.evidence_extraction_started > 0,
            facts.evaluation_completed,
            facts.replan_count > 0,
            facts.gap_resolution_attempts > 0,
        )
    )
    search_signal = facts.search_queries_started > 0 or facts.search_queries_completed > 0

    if (
        facts.query_family_marker_present
        or facts.search_queries_completed > 0
        or downstream_signal
    ):
        first_pass = FirstPassExecutionState.COMPLETED
    elif search_signal:
        first_pass = FirstPassExecutionState.RUNNING
    elif facts.technical_blocked or facts.resource_blocked:
        first_pass = FirstPassExecutionState.BLOCKED
    else:
        first_pass = FirstPassExecutionState.NOT_STARTED

    reasonable_attempts = bool(
        facts.evaluation_completed
        and (
            facts.search_queries_completed > 0
            or facts.readable_sources > 0
            or facts.evidence_extraction_started > 0
            or facts.gap_resolution_attempts > 0
        )
    )
    quality_met = facts.quality_met or (
        facts.evaluation_completed and coverage >= 1.0 and not facts.gap_open
    )
    if quality_met:
        opportunity = ResearchOpportunityState.COMPLETED_QUALITY
    elif facts.gap_open and reasonable_attempts:
        opportunity = ResearchOpportunityState.COMPLETED_WITH_GAP
    elif facts.technical_blocked:
        opportunity = ResearchOpportunityState.BLOCKED_TECHNICAL
    elif facts.resource_blocked:
        opportunity = ResearchOpportunityState.BLOCKED_RESOURCE
    elif downstream_signal or search_signal:
        opportunity = ResearchOpportunityState.ACTIVE
    else:
        opportunity = ResearchOpportunityState.NOT_STARTED

    if opportunity == ResearchOpportunityState.COMPLETED_QUALITY or not facts.gap_open:
        recovery = RecoveryEligibilityState.DENIED
    elif opportunity == ResearchOpportunityState.NOT_STARTED:
        recovery = RecoveryEligibilityState.NOT_EVALUATED
    elif facts.budget_blocked or opportunity in {
        ResearchOpportunityState.BLOCKED_TECHNICAL,
        ResearchOpportunityState.BLOCKED_RESOURCE,
    }:
        recovery = RecoveryEligibilityState.DEFERRED
    elif opportunity == ResearchOpportunityState.ACTIVE:
        recovery = RecoveryEligibilityState.NOT_EVALUATED
    elif facts.has_recovery_path:
        recovery = RecoveryEligibilityState.ELIGIBLE
    else:
        recovery = RecoveryEligibilityState.DENIED

    blocking_reason = facts.budget_blocking_reason
    if blocking_reason is None and opportunity == ResearchOpportunityState.BLOCKED_TECHNICAL:
        blocking_reason = "technical_blocked"
    if blocking_reason is None and opportunity == ResearchOpportunityState.BLOCKED_RESOURCE:
        blocking_reason = "resource_blocked"

    return QuestionResearchState(
        question_id=facts.question_id,
        first_pass_execution=first_pass,
        research_opportunity=opportunity,
        recovery_eligibility=recovery,
        blocking_reason=blocking_reason,
    )
