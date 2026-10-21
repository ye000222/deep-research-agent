from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from app.domain.proposition_research_gap import (
    DesiredEvidenceRelation,
    GapSearchability,
    PropositionGapLifecycle,
    PropositionResearchGapType,
    RequiredEvidenceCharacteristics,
)
from app.domain.research_work import GapResearchExecutionContext
from app.infrastructure.db.research_tools import ResearchTarget
from app.services.research_loop import ResearchAttemptResult, ResearchLoopService


@pytest.mark.asyncio
async def test_execute_target_delegates_once_without_rescheduling() -> None:
    run_id = UUID("11111111-1111-4111-8111-111111111111")
    target_prop = uuid4()
    required = RequiredEvidenceCharacteristics(
        target_group_ref="a" * 64,
        target_proposition_refs=(target_prop,),
        desired_relations=(DesiredEvidenceRelation.SUPPORTS,),
    )
    gap_context = GapResearchExecutionContext(
        gap_ref="b" * 64,
        intent_ref="c" * 64,
        gap_type=PropositionResearchGapType.MISSING_SUPPORT,
        searchability=GapSearchability.SEARCHABLE,
        lifecycle_state=PropositionGapLifecycle.OPEN,
        target_group_ref="a" * 64,
        target_proposition_refs=(target_prop,),
        parent_claim_refs=(),
        required_evidence=required,
        objective_scope_ref="scope:target",
        source_semantic_snapshot_ref="runs/semantic.json",
        source_semantic_snapshot_hash="d" * 64,
        source_gap_projection_hash="e" * 64,
        attempt_index=1,
    )
    target = ResearchTarget(
        plan_version=1,
        question_id="q1",
        question="Which methods are used?",
        query="methods official source",
        gap_id=uuid4(),
        gap_attempt_index=1,
        tool_call_id=uuid4(),
        source_id_seed=uuid4(),
        gap_execution_context=gap_context,
    )
    expected = ResearchAttemptResult(0, 0, "budget_yield")
    service = ResearchLoopService(
        repository=None,  # type: ignore[arg-type]
        search=None,  # type: ignore[arg-type]
        reader=None,  # type: ignore[arg-type]
        extractor=None,  # type: ignore[arg-type]
        artifacts=None,  # type: ignore[arg-type]
        evidence_aware_context_enabled=False,
    )
    execute = AsyncMock(return_value=expected)
    service._research_target = execute  # type: ignore[method-assign]

    result = await service.execute_target(run_id, worker_task_id="worker-1", target=target)

    assert result is expected
    execute.assert_awaited_once_with(run_id, worker_task_id="worker-1", target=target)
    assert execute.await_args.kwargs["target"].gap_execution_context == gap_context
