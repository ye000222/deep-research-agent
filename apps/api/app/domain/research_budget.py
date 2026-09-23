"""Pure budget allocation; reserved opportunities must never freeze spendable tokens."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from app.domain.question_research_state import (
    QuestionResearchState,
    RecoveryEligibilityState,
    ResearchOpportunityState,
)

ModelTokenPool = Literal["planner", "research", "verification", "writer", "safety"]
RiskLevel = Literal["low", "medium", "high", "critical"]
RiskLifecycle = Literal["open", "mitigating", "blocked", "resolved"]
VerificationState = Literal["not_required", "required", "verified", "blocked"]


@dataclass(frozen=True)
class ResourcePoolState:
    """One executable hard pool rendered from immutable limits and live usage."""

    name: str
    unit: str
    limit: int
    committed: int
    reserved: int
    uncertain: int
    remaining: int
    status: Literal["disabled", "available", "exhausted", "released"]
    limit_key: str
    usage_key: str

    def as_dict(self) -> dict[str, object]:
        return {
            "unit": self.unit,
            "limit": self.limit,
            "committed": self.committed,
            "reserved": self.reserved,
            "uncertain": self.uncertain,
            "remaining": self.remaining,
            "status": self.status,
            "limit_key": self.limit_key,
            "usage_key": self.usage_key,
        }


@dataclass(frozen=True)
class ClaimRiskState:
    """Claim-level risk, corroboration deficit, and verification lifecycle."""

    claim_id: str
    question_id: str
    dimension_key: str
    claim_status: str
    claim_type: str
    importance: float
    risk_score: float
    risk_level: RiskLevel
    lifecycle: RiskLifecycle
    verification_state: VerificationState
    unresolved: bool
    required_sources: int
    independent_sources: int
    independent_source_deficit: int
    open_conflict_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "question_id": self.question_id,
            "dimension_key": self.dimension_key,
            "claim_status": self.claim_status,
            "claim_type": self.claim_type,
            "importance": self.importance,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "lifecycle": self.lifecycle,
            "verification_state": self.verification_state,
            "unresolved": self.unresolved,
            "required_sources": self.required_sources,
            "independent_sources": self.independent_sources,
            "independent_source_deficit": self.independent_source_deficit,
            "open_conflict_ids": list(self.open_conflict_ids),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class GapRiskState:
    """Gap-level lifecycle kept even after the gap is resolved."""

    gap_id: str
    question_id: str
    gap_status: str
    gap_type: str
    severity: float
    resolution_attempts: int
    risk_score: float
    risk_level: RiskLevel
    lifecycle: RiskLifecycle
    unresolved: bool
    blocked: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "gap_id": self.gap_id,
            "question_id": self.question_id,
            "gap_status": self.gap_status,
            "gap_type": self.gap_type,
            "severity": self.severity,
            "resolution_attempts": self.resolution_attempts,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "lifecycle": self.lifecycle,
            "unresolved": self.unresolved,
            "blocked": self.blocked,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ResearchBudgetDecision:
    outcome: Literal["execute", "yield_question", "stop_run"]
    call_tokens: int
    future_reserve: int
    reason: str


def allocate_research_call(
    *,
    available: int,
    eligible_questions: int,
    other_unattempted: int,
    current_attempts: int,
    minimum_call: int,
    maximum_call: int,
) -> ResearchBudgetDecision:
    """Allocate affordable seats, rather than reserve more than the run can pay.

    The caller supplies a feasible minimum contract cost. A yield is local to
    the question; only inability to fund any contract is a run-level stop.
    """
    if minimum_call <= 0 or maximum_call < minimum_call:
        raise ValueError("invalid call bounds")
    if available < minimum_call:
        return ResearchBudgetDecision("stop_run", 0, 0, "no_affordable_call")
    if current_attempts > 0 and other_unattempted > 0:
        return ResearchBudgetDecision("yield_question", 0, 0, "unattempted_questions_first")
    seats = min(max(1, eligible_questions), available // minimum_call)
    allowance = min(maximum_call, available // seats)
    return ResearchBudgetDecision(
        "execute", allowance, (seats - 1) * minimum_call, "affordable_question_share"
    )


def question_schedule_key(
    *,
    attempts: int,
    coverage: float,
    priority: int,
    question_id: str,
) -> tuple[int, int, float, int, str]:
    """Give all questions a first pass, then bounded round-robin gap filling."""
    return (int(attempts > 0), attempts, coverage, priority, question_id)


def conservative_chars_to_tokens(value: str, *, chars_per_latin_token: int = 3) -> int:
    """Conservative lower-bound token estimate without a provider tokenizer.

    CJK characters are counted one-token-per-character (unigram Chinese
    tokenizers approach 1:1), while Latin/full-width text is counted at
    ``chars_per_latin_token`` characters per token. Tuning the Latin rate down
    makes the estimate more conservative so the run never assumes a request can
    be assembled when the budget cannot hold it.
    """

    if not value:
        return 0
    value = value or ""
    cjk = sum(1 for ch in value if "\u3400" <= ch <= "\u9fff")
    other = len(value) - cjk
    return cjk + math.ceil(other / max(1, chars_per_latin_token))


@dataclass(frozen=True)
class MinimumCallEstimate:
    """The smallest provably-assemblable cost of one evidence call."""

    fixed_tokens: int
    min_source_tokens: int
    min_output_tokens: int
    safety_margin_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class QuestionBudgetEstimate:
    """Pre-flight token envelope for one planned question.

    This is deliberately a conservative *planning* estimate, not a provider
    billing value.  It lets the run snapshot explain how the research pool was
    divided before the first search starts, while the reservation ledger still
    remains the source of truth for actual usage.
    """

    question_id: str
    minimum_tokens: int
    target_tokens: int
    priority: int
    complexity: int


@dataclass(frozen=True)
class QuestionRiskState:
    """Durable claim/gap risk summary consumed by the borrowing gate."""

    question_id: str
    priority: int
    coverage: float
    risk_score: float
    risk_level: RiskLevel
    unresolved_high_risk: bool
    gap_open: bool
    requires_independent_sources: bool
    open_dimension_keys: tuple[str, ...]
    unresolved_claim_ids: tuple[str, ...]
    high_risk_claim_ids: tuple[str, ...]
    high_risk_conflict_ids: tuple[str, ...]
    lifecycle: RiskLifecycle
    verification_state: VerificationState
    independent_source_deficit: int
    borrow_eligible: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "question_id": self.question_id,
            "priority": self.priority,
            "coverage": self.coverage,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "unresolved_high_risk": self.unresolved_high_risk,
            "gap_open": self.gap_open,
            "requires_independent_sources": self.requires_independent_sources,
            "open_dimension_keys": list(self.open_dimension_keys),
            "unresolved_claim_ids": list(self.unresolved_claim_ids),
            "high_risk_claim_ids": list(self.high_risk_claim_ids),
            "high_risk_conflict_ids": list(self.high_risk_conflict_ids),
            "lifecycle": self.lifecycle,
            "verification_state": self.verification_state,
            "independent_source_deficit": self.independent_source_deficit,
            "borrow_eligible": self.borrow_eligible,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class QuestionBorrowDecision:
    allowed: bool
    reason: str
    hard_limit: bool = False
    freeze_question: bool = False


def model_token_pool_for_node(node: str) -> ModelTokenPool:
    normalized = node.strip().casefold()
    if normalized == "planner":
        return "planner"
    if normalized in {"report_writer", "writer"}:
        return "writer"
    if normalized in {"verification", "claim_verifier", "report_verifier"}:
        return "verification"
    if normalized in {"safety", "reconciliation"}:
        return "safety"
    return "research"


def model_token_pool_limits(allocation: Mapping[str, object]) -> dict[ModelTokenPool, int]:
    return {
        "planner": max(0, _safe_int(allocation.get("planner_tokens", 0))),
        "research": max(0, _safe_int(allocation.get("research_tokens", 0))),
        "verification": max(0, _safe_int(allocation.get("verification_tokens", 0))),
        "writer": max(
            0,
            _safe_int(
                allocation.get("writer_tokens_initial", allocation.get("writer_tokens", 0))
            ),
        ),
        "safety": max(0, _safe_int(allocation.get("safety_tokens", 0))),
    }


def build_resource_pool_snapshot(
    budget: Mapping[str, object],
    usage: Mapping[str, object],
    *,
    model_token_pools: Mapping[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    """Build the complete executable pool view used by API snapshots.

    Allocation fields that have no runtime consumer do not appear here. Every
    returned pool maps to the exact hard-limit and usage keys enforced by the
    scheduler. Reservations are included for fetch/extraction concurrency so a
    dashboard cannot advertise capacity that is already held by another worker.
    """

    specs = (
        ("logical_queries", "queries", "max_logical_queries", "logical_queries", None),
        (
            "provider_requests",
            "requests",
            "max_provider_requests",
            "search_provider_requests",
            None,
        ),
        (
            "pages_fetched",
            "pages",
            "max_pages_fetched",
            "pages_fetched",
            "page_slots_reserved",
        ),
        (
            "page_fetch_attempts",
            "attempts",
            "max_page_fetch_attempts",
            "page_fetch_attempts",
            "page_slots_reserved",
        ),
        (
            "pages_extracted",
            "pages",
            "max_pages_extracted",
            "pages_extracted",
            "extraction_slots_reserved",
        ),
        (
            "extraction_calls",
            "calls",
            "max_extraction_calls",
            "extraction_calls",
            "extraction_slots_reserved",
        ),
        (
            "verification_calls",
            "calls",
            "max_verification_calls",
            "verification_calls",
            None,
        ),
        (
            "scheduler_actions",
            "actions",
            "max_scheduler_actions",
            "scheduler_actions",
            None,
        ),
        (
            "productive_iterations",
            "iterations",
            "max_iterations",
            "productive_iterations",
            None,
        ),
    )
    pools: dict[str, dict[str, object]] = {}
    for name, unit, limit_key, usage_key, reservation_key in specs:
        limit = max(0, _safe_int(budget.get(limit_key, 0)))
        # Compatibility fallbacks are read-only and never become a new pool.
        if name == "logical_queries" and limit == 0:
            limit = max(0, _safe_int(budget.get("max_searches", 0)))
        elif name in {"pages_fetched", "pages_extracted"} and limit == 0:
            limit = max(0, _safe_int(budget.get("max_pages", 0)))
        committed = max(0, _safe_int(usage.get(usage_key, 0)))
        if name == "productive_iterations" and usage_key not in usage:
            committed = max(0, _safe_int(usage.get("iterations", 0)))
        reserved = (
            max(0, _safe_int(usage.get(reservation_key, 0))) if reservation_key else 0
        )
        remaining = max(0, limit - committed - reserved)
        status: Literal["disabled", "available", "exhausted", "released"] = (
            "disabled" if limit <= 0 else "exhausted" if remaining <= 0 else "available"
        )
        pools[name] = ResourcePoolState(
            name=name,
            unit=unit,
            limit=limit,
            committed=committed,
            reserved=reserved,
            uncertain=0,
            remaining=remaining,
            status=status,
            limit_key=limit_key,
            usage_key=usage_key,
        ).as_dict()

    raw_model_pools = model_token_pools or usage.get("model_token_pools", {})
    if isinstance(raw_model_pools, Mapping):
        for pool_name in ("planner", "research", "verification", "writer", "safety"):
            raw = raw_model_pools.get(pool_name, {})
            if not isinstance(raw, Mapping):
                continue
            limit = max(0, _safe_int(raw.get("allocated_tokens", 0)))
            committed = max(0, _safe_int(raw.get("committed_tokens", 0)))
            reserved = max(0, _safe_int(raw.get("reserved_tokens", 0)))
            uncertain = max(0, _safe_int(raw.get("uncertain_tokens", 0)))
            remaining = max(0, _safe_int(raw.get("remaining_tokens", limit - committed)))
            released = (
                pool_name == "writer" and "writer_token_reserve_released" in usage
            ) or (
                pool_name == "verification"
                and "verification_token_reserve_released" in usage
            )
            status = (
                "released"
                if released
                else "disabled"
                if limit <= 0
                else "exhausted"
                if remaining <= 0
                else "available"
            )
            if released:
                remaining = 0
            pools[f"model_tokens.{pool_name}"] = ResourcePoolState(
                name=f"model_tokens.{pool_name}",
                unit="tokens",
                limit=limit,
                committed=committed,
                reserved=reserved,
                uncertain=uncertain,
                remaining=remaining,
                status=status,
                limit_key=(
                    "allocation.writer_tokens_initial"
                    if pool_name == "writer"
                    else f"allocation.{pool_name}_tokens"
                ),
                usage_key=f"model_token_pools.{pool_name}.committed_tokens",
            ).as_dict()
    return pools


def classify_claim_risk(
    *,
    claim_id: str,
    question_id: str,
    dimension_key: str,
    claim_status: str,
    claim_type: str,
    importance: float,
    independent_sources: int,
    required_sources: int,
    open_conflict_ids: Sequence[str] = (),
) -> ClaimRiskState:
    """Classify one claim from evidence independence and conflict facts."""

    status = claim_status.strip().casefold()
    claim_type = claim_type.strip().casefold() or "factual"
    importance = min(1.0, max(0.0, float(importance)))
    independent_sources = max(0, int(independent_sources))
    required_sources = max(1, int(required_sources))
    conflicts = tuple(dict.fromkeys(str(value) for value in open_conflict_ids))
    deficit = max(0, required_sources - independent_sources)
    terminal_rejection = status == "rejected"
    supported = status == "supported" and deficit == 0 and not conflicts
    unresolved = not terminal_rejection and not supported
    reasons: list[str] = []
    score = importance * 0.35
    if importance >= 0.75:
        score += 0.25
        reasons.append("high_importance")
    if claim_type in {"numeric", "forecast", "comparative", "market"}:
        score += 0.25
        reasons.append("high_risk_claim_type")
    if deficit:
        score += min(0.30, 0.15 * deficit)
        reasons.append("independent_source_deficit")
    if status == "disputed" or conflicts:
        score += 0.40
        reasons.append("open_conflict")
    if terminal_rejection:
        score = 0.0
        reasons.append("claim_rejected")
    score = round(min(1.0, score), 4)
    level = _risk_level(score)
    if supported or terminal_rejection:
        lifecycle: RiskLifecycle = "resolved"
    elif status == "disputed" or conflicts:
        lifecycle = "blocked"
    elif independent_sources > 0:
        lifecycle = "mitigating"
    else:
        lifecycle = "open"
    verification_required = (
        not terminal_rejection
        and (required_sources > 1 or level in {"high", "critical"} or bool(conflicts))
    )
    verification_state: VerificationState
    if not verification_required:
        verification_state = "not_required"
    elif supported:
        verification_state = "verified"
    elif conflicts:
        verification_state = "blocked"
    else:
        verification_state = "required"
    return ClaimRiskState(
        claim_id=str(claim_id),
        question_id=str(question_id),
        dimension_key=str(dimension_key),
        claim_status=status,
        claim_type=claim_type,
        importance=round(importance, 4),
        risk_score=score,
        risk_level=level,
        lifecycle=lifecycle,
        verification_state=verification_state,
        unresolved=unresolved,
        required_sources=required_sources,
        independent_sources=independent_sources,
        independent_source_deficit=deficit,
        open_conflict_ids=conflicts,
        reasons=tuple(reasons),
    )


def classify_gap_risk(
    *,
    gap_id: str,
    question_id: str,
    gap_status: str,
    gap_type: str,
    severity: float,
    resolution_attempts: int,
    blocked: bool = False,
) -> GapRiskState:
    """Classify one persisted gap without dropping its resolved history."""

    status = gap_status.strip().casefold()
    severity = min(1.0, max(0.0, float(severity)))
    attempts = max(0, int(resolution_attempts))
    unresolved = status != "resolved"
    reasons: list[str] = []
    score = severity * 0.70 if unresolved else 0.0
    if attempts >= 2 and unresolved:
        score += 0.15
        reasons.append("repeated_resolution_attempts")
    if blocked and unresolved:
        score += 0.25
        reasons.append("no_eligible_action")
    if unresolved:
        reasons.append("gap_open")
    else:
        reasons.append("gap_resolved")
    score = round(min(1.0, score), 4)
    if not unresolved:
        lifecycle: RiskLifecycle = "resolved"
    elif blocked:
        lifecycle = "blocked"
    elif attempts > 0:
        lifecycle = "mitigating"
    else:
        lifecycle = "open"
    return GapRiskState(
        gap_id=str(gap_id),
        question_id=str(question_id),
        gap_status=status,
        gap_type=gap_type.strip().casefold() or "missing",
        severity=round(severity, 4),
        resolution_attempts=attempts,
        risk_score=score,
        risk_level=_risk_level(score),
        lifecycle=lifecycle,
        unresolved=unresolved,
        blocked=bool(blocked and unresolved),
        reasons=tuple(reasons),
    )


def classify_question_risk(
    *,
    question_id: str,
    priority: int,
    coverage: float,
    requirements: Sequence[str],
    gap_open: bool,
    open_dimension_keys: Sequence[str] = (),
    unresolved_claim_ids: Sequence[str] = (),
    high_risk_claim_ids: Sequence[str] = (),
    high_risk_conflict_ids: Sequence[str] = (),
    independent_source_deficit: int = 0,
    blocked: bool = False,
) -> QuestionRiskState:
    """Produce a stable risk state from explicit claim, gap, and plan facts."""

    priority = min(3, max(1, int(priority)))
    coverage = min(1.0, max(0.0, float(coverage)))
    independent = any(_requirement_is_high_risk(value) for value in requirements)
    unresolved_claims = tuple(dict.fromkeys(str(value) for value in unresolved_claim_ids))
    high_risk_claims = tuple(dict.fromkeys(str(value) for value in high_risk_claim_ids))
    high_risk_conflicts = tuple(dict.fromkeys(str(value) for value in high_risk_conflict_ids))
    dimensions = tuple(dict.fromkeys(str(value) for value in open_dimension_keys))
    independent_source_deficit = max(0, int(independent_source_deficit))
    reasons: list[str] = []
    score = (1.0 - coverage) * 0.30
    if priority == 1:
        score += 0.35
        reasons.append("priority_one")
    elif priority == 2:
        score += 0.10
    if independent:
        score += 0.25
        reasons.append("independent_corroboration_required")
    if gap_open or dimensions:
        score += 0.10
        reasons.append("acceptance_gap_open")
    if high_risk_claims:
        score += 0.20
        reasons.append("high_risk_claim_unresolved")
    if high_risk_conflicts:
        score += 0.35
        reasons.append("high_risk_conflict_open")
    if independent_source_deficit:
        score += min(0.20, independent_source_deficit * 0.10)
        reasons.append("independent_source_deficit")
    if blocked and (gap_open or dimensions or unresolved_claims):
        score += 0.10
        reasons.append("no_eligible_action")
    score = round(min(1.0, score), 4)
    if score >= 0.85:
        level: RiskLevel = "critical"
    elif score >= 0.65:
        level = "high"
    elif score >= 0.35:
        level = "medium"
    else:
        level = "low"
    unresolved = bool(
        level in {"high", "critical"}
        and (
            gap_open
            or dimensions
            or high_risk_claims
            or high_risk_conflicts
            or independent_source_deficit
        )
    )
    if not gap_open and not dimensions and not unresolved_claims and not high_risk_conflicts:
        lifecycle: RiskLifecycle = "resolved"
    elif blocked or high_risk_conflicts:
        lifecycle = "blocked"
    elif coverage > 0.0:
        lifecycle = "mitigating"
    else:
        lifecycle = "open"
    verification_required = bool(
        independent
        or high_risk_claims
        or high_risk_conflicts
        or independent_source_deficit
    )
    if not verification_required:
        verification_state: VerificationState = "not_required"
    elif high_risk_conflicts:
        verification_state = "blocked"
    elif lifecycle == "resolved" and independent_source_deficit == 0:
        verification_state = "verified"
    else:
        verification_state = "required"
    borrow_eligible = bool(
        unresolved
        and lifecycle in {"open", "mitigating"}
        and verification_state != "blocked"
    )
    return QuestionRiskState(
        question_id=question_id,
        priority=priority,
        coverage=round(coverage, 4),
        risk_score=score,
        risk_level=level,
        unresolved_high_risk=unresolved,
        gap_open=gap_open,
        requires_independent_sources=independent,
        open_dimension_keys=dimensions,
        unresolved_claim_ids=unresolved_claims,
        high_risk_claim_ids=high_risk_claims,
        high_risk_conflict_ids=high_risk_conflicts,
        lifecycle=lifecycle,
        verification_state=verification_state,
        independent_source_deficit=independent_source_deficit,
        borrow_eligible=borrow_eligible,
        reasons=tuple(reasons),
    )


def decide_question_borrow(
    *,
    state: QuestionRiskState,
    all_first_passes_complete: bool,
    research_state: QuestionResearchState | None = None,
    projected_spend: int,
    target_tokens: int,
    expected_utility: float,
    low_gain_streak: int,
    has_untried_query_family: bool = False,
    utility_threshold: float = 0.005,
) -> QuestionBorrowDecision:
    """Apply the V2 first-pass, state, risk, utility, and cap contract."""

    target_tokens = max(0, int(target_tokens))
    projected_spend = max(0, int(projected_spend))
    if projected_spend <= target_tokens:
        return QuestionBorrowDecision(True, "within_question_target")
    if research_state is None:
        if not all_first_passes_complete:
            return QuestionBorrowDecision(False, "first_pass_floor_protected")
    else:
        if research_state.research_opportunity == ResearchOpportunityState.NOT_STARTED:
            return QuestionBorrowDecision(False, "first_pass_floor_protected")
        if research_state.recovery_eligibility == RecoveryEligibilityState.NOT_EVALUATED:
            return QuestionBorrowDecision(False, "first_pass_floor_protected")
        if research_state.recovery_eligibility == RecoveryEligibilityState.DEFERRED:
            return QuestionBorrowDecision(False, "first_pass_floor_protected")
        if research_state.recovery_eligibility == RecoveryEligibilityState.DENIED:
            return QuestionBorrowDecision(
                False,
                "recovery_ineligible",
                freeze_question=True,
            )
    borrow_cap = int(target_tokens * 1.5)
    # A high-risk exception is a bounded recovery allowance, not a second
    # unlimited budget.  Earlier versions let the exception bypass this cap
    # entirely; one difficult P1 could then consume the whole research pool
    # and starve unrelated questions.  Keep the normal 150% envelope, while
    # allowing at most one additional half-target recovery window for a
    # genuinely new query family.  The caller still enforces the run-level
    # research pool and reservations remain the source of truth.
    recovery_cap = int(target_tokens * 2.0)
    p1_high_risk_exception = state.priority == 1 and state.unresolved_high_risk
    # A high-risk question with an untried query family still has a bounded
    # recovery action.  Permit that one additional family to draw from the
    # global research pool instead of freezing it solely at the per-question
    # 150% cap; the global pool and provider/page limits remain hard stops.
    high_risk_recovery_exception = state.unresolved_high_risk and has_untried_query_family
    # An acceptance gap is actionable even when the current evidence has not
    # yet raised the question into the high-risk bucket.  Treating every such
    # question as ``no_unresolved_high_risk_gap`` freezes ordinary P2/P3
    # dimensions and can make the scheduler enter ``sources_exhausted`` while
    # global search/page/token capacity is still available.
    acceptance_gap_open = bool(state.gap_open or state.open_dimension_keys)
    # Classify the gap before enforcing the per-question cap.  A fetched page
    # for an open acceptance gap still needs an extractor call; rejecting it at
    # 150% used to leave valid candidate pages unprocessed.  Recovery remains
    # bounded by the low-gain guard below and by the run-level research pool.
    acceptance_recovery_exception = acceptance_gap_open and has_untried_query_family
    replanned_high_risk_exception = (
        state.unresolved_high_risk
        and acceptance_gap_open
        and low_gain_streak < 2
    )
    recovery_exception = (
        p1_high_risk_exception
        or high_risk_recovery_exception
        or acceptance_recovery_exception
        or replanned_high_risk_exception
    )
    if projected_spend > borrow_cap and not recovery_exception:
        return QuestionBorrowDecision(False, "hard_question_token_limit", hard_limit=True)
    if recovery_exception and projected_spend > recovery_cap:
        return QuestionBorrowDecision(
            False,
            "bounded_recovery_limit",
            hard_limit=True,
            freeze_question=True,
        )
    if state.verification_state == "blocked":
        return QuestionBorrowDecision(False, "verification_blocked", freeze_question=True)
    if (not state.unresolved_high_risk and not acceptance_gap_open) or (
        not state.borrow_eligible and not acceptance_gap_open
    ):
        return QuestionBorrowDecision(False, "no_unresolved_high_risk_gap", freeze_question=True)
    p1_recovery_variant = (
        state.priority == 1
        and state.unresolved_high_risk
        and has_untried_query_family
    )
    acceptance_recovery_variant = acceptance_gap_open and has_untried_query_family
    if low_gain_streak >= 2 and not (p1_recovery_variant or acceptance_recovery_variant):
        return QuestionBorrowDecision(False, "low_gain_streak", freeze_question=True)
    if expected_utility < utility_threshold:
        return QuestionBorrowDecision(
            False,
            "expected_utility_below_threshold",
            freeze_question=True,
        )
    return QuestionBorrowDecision(True, "high_risk_high_utility_borrow")


def _safe_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, bytes, bytearray)):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _risk_level(score: float) -> RiskLevel:
    if score >= 0.85:
        return "critical"
    if score >= 0.65:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def _requirement_is_high_risk(value: str) -> bool:
    normalized = value.casefold()
    markers = (
        "独立来源",
        "两个",
        "两条",
        "市场规模",
        "市场份额",
        "预测",
        "增长率",
        "比较",
        "领先",
        "independent",
        "two ",
        "market size",
        "market share",
        "forecast",
        "projection",
        "growth rate",
        "comparative",
        "leading",
        "largest",
    )
    return any(marker in normalized for marker in markers)


def estimate_question_budgets(
    questions: Sequence[Mapping[str, object]],
    *,
    max_tokens: int,
    planner_tokens: int = 0,
    writer_reserve: int = 0,
    minimum_per_question: int = 2_500,
) -> tuple[QuestionBudgetEstimate, ...]:
    """Allocate a feasible, priority-aware research envelope before execution.

    The function never increases the configured model ceiling.  High-priority
    and information-dense questions receive a larger target, but every question
    gets an explicit floor whenever the available pool can afford one.  If the
    pool is too small, targets collapse to an equal split rather than silently
    promising impossible work.
    """

    if max_tokens < 0 or planner_tokens < 0 or writer_reserve < 0:
        raise ValueError("token budgets must be non-negative")
    if not questions:
        return ()
    pool = max(0, int(max_tokens) - int(planner_tokens) - int(writer_reserve))
    floor = max(1, int(minimum_per_question))
    rows: list[tuple[str, int, int, int]] = []
    for index, question in enumerate(questions, start=1):
        question_id = str(question.get("id") or f"q{index}")
        priority = min(3, max(1, _safe_int(question.get("priority", 2), default=2)))
        requirements = question.get("evidence_requirements")
        hints = question.get("search_hints")
        requirement_count = (
            len(requirements)
            if isinstance(requirements, Sequence) and not isinstance(requirements, (str, bytes))
            else 0
        )
        hint_count = (
            len(hints)
            if isinstance(hints, Sequence) and not isinstance(hints, (str, bytes))
            else 0
        )
        complexity = 1 + requirement_count + max(0, hint_count - 1)
        # Priority 1 gets a modest uplift; requirements dominate so a broad
        # question cannot be starved simply because its priority is lower.
        weight = (4 - priority) + complexity
        rows.append((question_id, priority, complexity, weight))
    total_weight = sum(row[3] for row in rows)
    if pool < floor * len(rows):
        equal = pool // len(rows)
        return tuple(
            QuestionBudgetEstimate(qid, equal, equal, priority, complexity)
            for qid, priority, complexity, _weight in rows
        )
    remaining = pool - floor * len(rows)
    estimates: list[QuestionBudgetEstimate] = []
    assigned = 0
    remaining_pool = remaining
    for index, (qid, priority, complexity, weight) in enumerate(rows):
        share = (
            remaining_pool
            if index == len(rows) - 1
            else (remaining * weight) // total_weight
        )
        target = floor + share
        assigned += target
        remaining_pool -= share
        estimates.append(QuestionBudgetEstimate(qid, floor, target, priority, complexity))
    # Integer division can leave a few tokens unassigned; give them to q1 so
    # the persisted envelope exactly explains the available research pool.
    if estimates and assigned < pool:
        first = estimates[0]
        estimates[0] = QuestionBudgetEstimate(
            first.question_id,
            first.minimum_tokens,
            first.target_tokens + pool - assigned,
            first.priority,
            first.complexity,
        )
    return tuple(estimates)


def estimate_writer_reserve_tokens(
    *,
    question_count: int,
    evidence_count: int,
    fixed_tokens: int = 1_500,
    per_question_tokens: int = 220,
    per_evidence_tokens: int = 180,
    output_tokens: int = 2_000,
    safety_ratio: float = 0.15,
    maximum: int = 10_000,
) -> int:
    """Estimate the bounded Writer call using its assembled payload shape.

    Fixed instructions/task/schema, question metadata, evidence-card payloads,
    and a bounded completion are all reserved together. The cap preserves the
    existing hard upper bound while the lower bound scales with the report.
    """

    base = (
        max(0, int(fixed_tokens))
        + max(0, int(question_count)) * max(0, int(per_question_tokens))
        + max(0, int(evidence_count)) * max(0, int(per_evidence_tokens))
        + max(1, int(output_tokens))
    )
    margin = max(1, math.ceil(base * float(safety_ratio)))
    return min(max(1, int(maximum)), base + margin)


def estimate_minimum_call_tokens(
    *,
    fixed_tokens: int,
    min_source_tokens: int,
    min_output_tokens: int,
    safety_ratio: float = 0.15,
) -> MinimumCallEstimate:
    """Conservative minimum cost for one call, never rounded below the inputs.

    The safety margin keeps the estimate honest against provider tokenizer
    drift instead of treating an arithmetic minimum as guaranteed affordable.
    """

    fixed_tokens = max(0, int(fixed_tokens))
    min_source_tokens = max(0, int(min_source_tokens))
    min_output_tokens = max(1, int(min_output_tokens))
    base = fixed_tokens + min_source_tokens + min_output_tokens
    margin = max(1, math.ceil(base * float(safety_ratio)))
    return MinimumCallEstimate(
        fixed_tokens=fixed_tokens,
        min_source_tokens=min_source_tokens,
        min_output_tokens=min_output_tokens,
        safety_margin_tokens=margin,
        total_tokens=base + margin,
    )
