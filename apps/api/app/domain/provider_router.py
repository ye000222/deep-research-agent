"""Health-aware provider routing decisions.

Given a :class:`~app.domain.query_execution.QueryExecutionRequest` and the
current provider health, the router selects the next provider to try, prefers a
healthy provider, forbids unavailable providers, and — for feedback-driven gap
closure queries — avoids recently failed providers while preserving the
``feedback_execution_id``. The router is a pure decision function: it does not
call any provider, does not change search strategy, query ranking, coverage,
evidence acceptance, gap closure, budget, recovery policy, or the planner, and
it never triggers an unbounded provider switch (the switch budget is enforced by
the caller through ``provider_switches_used`` / ``max_provider_switches``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from app.domain.provider_health import ProviderHealthState, ProviderHealthTracker
from app.domain.provider_registry import SearchProviderRegistry
from app.domain.query_execution import QueryExecutionRequest

# Lower priority value means a more preferred provider tier.
_STATE_PRIORITY: dict[ProviderHealthState, int] = {
    ProviderHealthState.HEALTHY: 0,
    ProviderHealthState.DEGRADED: 1,
    ProviderHealthState.UNKNOWN: 2,
    ProviderHealthState.COOLDOWN: 3,
    ProviderHealthState.UNAVAILABLE: 99,
    ProviderHealthState.DISABLED_FOR_RUN: 100,
}
# Health states that must never be selected in this run.
_BLOCKED_STATES = frozenset(
    {ProviderHealthState.UNAVAILABLE, ProviderHealthState.DISABLED_FOR_RUN}
)
# Feedback queries must strongly avoid any provider that is not confirmed
# healthy so a gap-closure attempt is not spent on a struggling provider.
_FEEDBACK_NON_HEALTHY_PENALTY = 50

DEFAULT_MAX_PROVIDER_SWITCHES = 1


@dataclass(frozen=True, slots=True)
class ProviderSelectionDecision:
    """The routing outcome for a single query execution request."""

    selected_provider: str | None
    excluded_providers: tuple[str, ...]
    reason: str
    health_state: ProviderHealthState | None
    fallback_used: bool
    feedback_execution_id: UUID | None = None
    # Phase 12.4 G: routing observability fields. Pool/available counts and the
    # selection rank describe how the router navigated the provider pool; they
    # default to zero so existing construction sites remain valid.
    provider_pool_size: int = 0
    available_provider_count: int = 0
    selection_rank: int = 0

    @property
    def selected(self) -> bool:
        return self.selected_provider is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_provider": self.selected_provider,
            "excluded_providers": list(self.excluded_providers),
            "reason": self.reason,
            "health_state": (
                self.health_state.value if self.health_state is not None else None
            ),
            "fallback_used": self.fallback_used,
            "feedback_execution_id": (
                str(self.feedback_execution_id)
                if self.feedback_execution_id
                else None
            ),
            "provider_pool_size": self.provider_pool_size,
            "available_provider_count": self.available_provider_count,
            "selection_rank": self.selection_rank,
        }


class ProviderRouter:
    """Choose a provider for a request using current provider health."""

    def __init__(
        self,
        health: ProviderHealthTracker,
        *,
        candidate_providers: Sequence[str] | None = None,
        registry: SearchProviderRegistry | None = None,
        max_provider_switches: int = DEFAULT_MAX_PROVIDER_SWITCHES,
    ) -> None:
        # Phase 12.4 D: when a registry is supplied it is the source of the
        # candidate pool (already sorted by declared priority). An explicit
        # ``candidate_providers`` sequence still wins so the ~40 existing
        # call sites and unit tests keep their current behavior unchanged.
        if candidate_providers is None and registry is not None:
            candidate_providers = registry.active_providers()
        if not candidate_providers:
            raise ValueError("provider router requires at least one candidate provider")
        self._health = health
        self._candidates = tuple(dict.fromkeys(candidate_providers))
        self._max_switches = max(0, max_provider_switches)

    def select(
        self,
        request: QueryExecutionRequest,
        *,
        already_excluded: Sequence[str] = (),
        provider_switches_used: int = 0,
    ) -> ProviderSelectionDecision:
        return self.select_for(
            is_feedback=request.feedback_id is not None,
            feedback_execution_id=request.feedback_execution_id,
            already_excluded=already_excluded,
            provider_switches_used=provider_switches_used,
        )

    def select_for(
        self,
        *,
        is_feedback: bool,
        feedback_execution_id: UUID | None = None,
        already_excluded: Sequence[str] = (),
        provider_switches_used: int = 0,
    ) -> ProviderSelectionDecision:
        excluded = tuple(dict.fromkeys(already_excluded))
        primary = self._candidates[0]
        pool_size = len(self._candidates)

        ranked: list[tuple[int, int, str, ProviderHealthState]] = []
        unavailable: list[str] = []
        for order, provider in enumerate(self._candidates):
            state = self._health.state_of(provider)
            if provider in excluded or state in _BLOCKED_STATES:
                unavailable.append(provider)
                continue
            priority = _STATE_PRIORITY[state]
            if is_feedback and state is not ProviderHealthState.HEALTHY:
                priority += _FEEDBACK_NON_HEALTHY_PENALTY
            ranked.append((priority, order, provider, state))

        if not ranked:
            return ProviderSelectionDecision(
                selected_provider=None,
                excluded_providers=tuple(unavailable),
                reason="all_providers_unavailable",
                health_state=None,
                fallback_used=False,
                feedback_execution_id=feedback_execution_id,
                provider_pool_size=pool_size,
                available_provider_count=0,
                selection_rank=0,
            )

        ranked.sort(key=lambda item: (item[0], item[1]))
        _, selected_order, selected, selected_state = ranked[0]
        fallback_used = selected != primary

        if fallback_used and provider_switches_used >= self._max_switches:
            # Health routing must not bypass the query budget, recovery limit,
            # or borrow limit by endlessly switching providers.
            return ProviderSelectionDecision(
                selected_provider=None,
                excluded_providers=tuple(
                    dict.fromkeys((*unavailable, *[p for p in self._candidates if p != selected]))
                ),
                reason="provider_switch_budget_exhausted",
                health_state=selected_state,
                fallback_used=False,
                feedback_execution_id=feedback_execution_id,
                provider_pool_size=pool_size,
                available_provider_count=len(ranked),
                selection_rank=0,
            )

        return ProviderSelectionDecision(
            selected_provider=selected,
            excluded_providers=tuple(unavailable),
            reason=_selection_reason(is_feedback=is_feedback, state=selected_state),
            health_state=selected_state,
            fallback_used=fallback_used,
            feedback_execution_id=feedback_execution_id,
            provider_pool_size=pool_size,
            available_provider_count=len(ranked),
            selection_rank=selected_order + 1,
        )


def _selection_reason(*, is_feedback: bool, state: ProviderHealthState) -> str:
    if is_feedback:
        if state is ProviderHealthState.HEALTHY:
            return "feedback_query_healthy_provider_selected"
        return "feedback_query_no_healthy_provider_fallback"
    if state is ProviderHealthState.HEALTHY:
        return "healthy_provider_selected"
    if state is ProviderHealthState.DEGRADED:
        return "healthy_provider_unavailable_degraded_fallback"
    if state is ProviderHealthState.COOLDOWN:
        return "healthy_provider_unavailable_cooldown_probe"
    return "healthy_provider_unavailable_unknown_fallback"
