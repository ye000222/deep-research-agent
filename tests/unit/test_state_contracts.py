from uuid import uuid4

import pytest
from app.domain.state import (
    ActionType,
    GapStatus,
    GapType,
    KnowledgeStatus,
    KnownClaimRef,
    NextAction,
    ResearchGap,
    ResearchState,
)
from pydantic import ValidationError


def make_gap(*, status: GapStatus = GapStatus.OPEN) -> ResearchGap:
    return ResearchGap(
        gap_id=uuid4(),
        question_id=uuid4(),
        dimension_key="market",
        gap_type=GapType.MISSING,
        description="Missing market evidence",
        acceptance_criteria="Two independent sources",
        severity=0.8,
        status=status,
        resolved_by_claim_ids=(uuid4(),) if status is GapStatus.RESOLVED else (),
    )


def test_supported_claim_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="must reference accepted evidence"):
        KnownClaimRef(
            claim_id=uuid4(),
            question_id=uuid4(),
            dimension_key="market",
            status=KnowledgeStatus.SUPPORTED,
            confidence=0.9,
        )


def test_external_action_must_target_an_open_gap() -> None:
    gap = make_gap(status=GapStatus.RESOLVED)
    action = NextAction(
        action_id=uuid4(),
        action_type=ActionType.SEARCH_WEB,
        target_gap_ids=(gap.gap_id,),
        tool_name="web_search",
        expected_output="Find recent market data",
        public_decision_summary="Search because the market dimension is unresolved.",
    )

    with pytest.raises(ValidationError, match="open or resolving gaps"):
        ResearchState(run_id=uuid4(), gaps=(gap,), next_action=action)


def test_external_action_requires_tool_name() -> None:
    gap = make_gap()
    with pytest.raises(ValidationError, match="must name its tool"):
        NextAction(
            action_id=uuid4(),
            action_type=ActionType.SEARCH_WEB,
            target_gap_ids=(gap.gap_id,),
            expected_output="Search results",
            public_decision_summary="Search the web for missing evidence.",
        )
