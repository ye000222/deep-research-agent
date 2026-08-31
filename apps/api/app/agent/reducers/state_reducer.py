from __future__ import annotations

from app.domain.state import ResearchState, StatePatch


class StateVersionConflict(RuntimeError):
    pass


def apply_patch(state: ResearchState, patch: StatePatch) -> ResearchState:
    """Apply a deterministic state patch without mutating the input state."""

    if patch.base_version != state.state_version:
        raise StateVersionConflict(
            f"patch base_version={patch.base_version} does not match "
            f"state_version={state.state_version}"
        )

    claims = {claim.claim_id: claim for claim in state.known}
    claims.update({claim.claim_id: claim for claim in patch.known_upserts})

    gaps = {gap.gap_id: gap for gap in state.gaps}
    gaps.update({gap.gap_id: gap for gap in patch.gap_upserts})

    changes: dict[str, object] = {
        "state_version": state.state_version + 1,
        "known": tuple(claims.values()),
        "gaps": tuple(gaps.values()),
    }
    for field_name in (
        "next_action",
        "budget_usage",
        "quality",
        "coverage_map",
        "phase",
        "status",
    ):
        value = getattr(patch, field_name)
        if value is not None:
            changes[field_name] = value
    if patch.clear_stop_reason:
        changes["stop_reason"] = None
    elif patch.stop_reason is not None:
        changes["stop_reason"] = patch.stop_reason

    payload = state.model_dump(mode="python")
    payload.update(changes)
    # Pydantic v2 does not validate model_copy(update=...). Rebuilding keeps
    # state invariants enforceable at every graph transition.
    return ResearchState.model_validate(payload)
