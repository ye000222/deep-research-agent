from __future__ import annotations

from collections.abc import Coroutine
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from app.domain.adaptive_scheduler import QueryFamily
from app.domain.gap_research_intent import (
    DesiredEvidencePurpose,
    GapResearchIntent,
    deterministic_intent_ref,
)
from app.domain.proposition_research_gap import (
    DesiredEvidenceRelation,
    GapPriorityClass,
    GapPriorityInputs,
    GapSearchability,
    PropositionGapLifecycle,
    PropositionGapReasonCode,
    PropositionResearchGap,
    PropositionResearchGapType,
    RequiredEvidenceCharacteristics,
)
from app.domain.research_work import (
    GapResearchExecutionContext,
    PropositionGapResearchObjective,
    ResearchWork,
    ResearchWorkKind,
    ResearchWorkStatus,
    WorkAttempt,
    WorkAttemptState,
    proposition_gap_objective_hash,
    proposition_gap_work_identity,
)
from app.domain.run_state import GapProjectionStatus, canonical_hash
from app.domain.semantic_control_projection import (
    SemanticGapControlArtifact,
    SemanticIntentControlArtifact,
    semantic_gap_artifact_ref,
    semantic_intent_artifact_ref,
)
from app.infrastructure.db.research_tools import ResearchTarget, ResearchToolRepository
from app.services.research_loop import ResearchAttemptResult, ResearchLoopService

_RUN = UUID("50000000-0000-0000-0000-000000000001")
_CLAIM = UUID("50000000-0000-0000-0000-000000000002")
_PROP = UUID("50000000-0000-0000-0000-000000000003")
_GROUP = "a" * 64
_GAP = "b" * 64
_SNAPSHOT_HASH = "c" * 64
_STATE_HASH = "d" * 64


def _run(coro: Coroutine[Any, Any, Any]) -> Any:
    try:
        coro.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("fake async operation unexpectedly yielded")


def _authoritative_inputs(*, searchable: GapSearchability = GapSearchability.SEARCHABLE):
    required = RequiredEvidenceCharacteristics(
        target_group_ref=_GROUP,
        target_proposition_refs=(_PROP,),
        desired_relations=(DesiredEvidenceRelation.CONTRADICTS,),
        required_owner_independence=True,
        excluded_owner_keys=("owner:existing",),
        required_context_refs=("context:region", "context:time"),
        required_time_version_scope_condition_refs=("scope:version",),
        a2_conflict_pair_refs=("conflict:left-right",),
        support_side_evidence_refs=("evidence:support",),
        contradict_side_evidence_refs=("evidence:contradict",),
        conflict_group_refs=("conflict:group",),
    )
    gap = PropositionResearchGap(
        gap_ref=_GAP,
        run_ref=_RUN,
        target_group_ref=_GROUP,
        target_proposition_refs=(_PROP,),
        parent_claim_refs=(_CLAIM,),
        objective_scope_ref="scope:formal-ref",
        verification_assessment_ref="e" * 64,
        verification_assessment_hash="f" * 64,
        gap_type=PropositionResearchGapType.CONFLICT_RESOLUTION,
        searchability=searchable,
        reason_codes=(PropositionGapReasonCode.CONFLICT_SCOPE_NOT_COMPLETE,),
        required_evidence=required,
        existing_evidence_refs=("evidence:existing",),
        existing_owner_refs=("owner:existing",),
        existing_conflict_refs=("conflict:existing",),
        lifecycle_state=(
            PropositionGapLifecycle.OPEN
            if searchable is GapSearchability.SEARCHABLE
            else PropositionGapLifecycle.BLOCKED
        ),
        priority_inputs=GapPriorityInputs(
            priority_class=GapPriorityClass.ACTIVE_CONFLICT,
            parent_research_priority=2,
            attempt_count=1,
        ),
        attempt_count=1,
    )
    intent = GapResearchIntent(
        intent_ref=deterministic_intent_ref(gap_ref=gap.gap_ref, attempt_index=1),
        run_ref=_RUN,
        gap_ref=gap.gap_ref,
        target_group_ref=gap.target_group_ref,
        target_proposition_refs=gap.target_proposition_refs,
        parent_claim_refs=gap.parent_claim_refs,
        objective_scope_ref=gap.objective_scope_ref,
        gap_type=gap.gap_type,
        desired_evidence_purpose=DesiredEvidencePurpose.CONFLICT_DISAMBIGUATION,
        required_evidence=gap.required_evidence,
        existing_evidence_refs=gap.existing_evidence_refs,
        existing_owner_refs=gap.existing_owner_refs,
        existing_conflict_refs=gap.existing_conflict_refs,
        gap_reason_codes=gap.reason_codes,
        attempt_index=1,
        priority_inputs=gap.priority_inputs,
    )
    intents = (intent,)
    gap_fields = {
        "schema_version": "semantic-gap-control-projection-v1",
        "run_ref": _RUN,
        "semantic_snapshot_ref": "runs/semantic.json",
        "semantic_snapshot_hash": _SNAPSHOT_HASH,
        "projection_status": "DERIVED",
        "gaps": (gap,),
        "issues": (),
        "projection_hash": canonical_hash([gap.gap_ref]),
    }
    gap_hash = canonical_hash(
        {
            **gap_fields,
            "run_ref": str(_RUN),
            "gaps": [gap.model_dump(mode="json")],
            "issues": [],
        }
    )
    gap_artifact = SemanticGapControlArtifact(
        **gap_fields,
        artifact_hash=gap_hash,
        artifact_ref=semantic_gap_artifact_ref(_RUN, gap_hash),
    )
    intent_fields = {
        "schema_version": "semantic-intent-control-projection-v1",
        "run_ref": _RUN,
        "semantic_snapshot_ref": gap_artifact.semantic_snapshot_ref,
        "semantic_snapshot_hash": gap_artifact.semantic_snapshot_hash,
        "gap_artifact_ref": gap_artifact.artifact_ref,
        "gap_projection_hash": gap_artifact.projection_hash,
        "intents": intents,
        "issues": (),
        "projection_hash": canonical_hash([intent.intent_ref for intent in intents]),
    }
    intent_hash = canonical_hash(
        {
            **intent_fields,
            "run_ref": str(_RUN),
            "intents": [intent.model_dump(mode="json") for intent in intents],
        }
    )
    intent_artifact = SemanticIntentControlArtifact(
        **intent_fields,
        artifact_hash=intent_hash,
        artifact_ref=semantic_intent_artifact_ref(_RUN, intent_hash),
    )
    state = SimpleNamespace(
        run_ref=_RUN,
        gap_projection_status=GapProjectionStatus.DERIVED,
        semantic_state_ref=gap_artifact.semantic_snapshot_ref,
        semantic_state_hash=gap_artifact.semantic_snapshot_hash,
        gap_source_semantic_hash=gap_artifact.semantic_snapshot_hash,
        gap_projection_hash=gap_artifact.projection_hash,
        gap_refs=(gap.gap_ref,),
        intent_refs=(intents[0].intent_ref,),
        intent_source_gap_hash=gap_artifact.projection_hash,
        execution_refs=(
            ("gap_projection_artifact_ref", gap_artifact.artifact_ref),
            ("intent_projection_artifact_ref", intent_artifact.artifact_ref),
        ),
        state_ref="runs/state.json",
        state_hash=_STATE_HASH,
        state_version=2,
    )
    objective = PropositionGapResearchObjective(
        gap_ref=gap.gap_ref,
        intent_ref=intents[0].intent_ref,
        objective_scope_ref=gap.objective_scope_ref,
        target_group_ref=gap.target_group_ref,
        target_proposition_refs=gap.target_proposition_refs,
        parent_claim_refs=gap.parent_claim_refs,
        gap_type=gap.gap_type,
        required_evidence=gap.required_evidence,
        question_id="real-question-17",
        question_lineage_source="PARENT_CLAIM_QUESTION",
        source_semantic_snapshot_ref=gap_artifact.semantic_snapshot_ref,
        source_semantic_snapshot_hash=gap_artifact.semantic_snapshot_hash,
        source_gap_projection_hash=gap_artifact.projection_hash,
    )
    work = ResearchWork(
        work_ref=proposition_gap_work_identity(
            run_ref=_RUN,
            gap_ref=gap.gap_ref,
            objective_hash=proposition_gap_objective_hash(objective),
        ),
        run_ref=_RUN,
        producer_kind="SEMANTIC_GAP_PROJECTOR",
        producer_ref="gap:durable",
        work_kind=ResearchWorkKind.PROPOSITION_GAP,
        objective=objective,
        objective_hash=proposition_gap_objective_hash(objective),
        priority=1,
        source_run_state_ref=state.state_ref,
        source_run_state_hash=state.state_hash,
        source_run_state_version=state.state_version,
        status=ResearchWorkStatus.ELIGIBLE,
        execution_bounds={"max_attempts": 3},
    )
    attempt = WorkAttempt(
        attempt_ref=UUID("50000000-0000-0000-0000-000000000006"),
        work_ref=work.work_ref,
        attempt_index=1,
        state=WorkAttemptState.RESERVED,
        reservation_identity="test-reservation",
        source_run_state_ref=state.state_ref,
        source_run_state_version=state.state_version,
    )
    return gap, intents[0], gap_artifact, intent_artifact, state, work, attempt


class _LineageSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def get(self, _model: object, _key: UUID):
        return SimpleNamespace(plan_version=4)

    async def execute(self, _statement: object):
        row = SimpleNamespace(id=_CLAIM, question_id="real-question-17")
        return SimpleNamespace(all=lambda: [row])


class _MissingLineageSession(_LineageSession):
    async def execute(self, _statement: object):
        return SimpleNamespace(all=lambda: [])


class _GapRepository(ResearchToolRepository):
    def __init__(self) -> None:
        super().__init__(lambda: _LineageSession())  # type: ignore[arg-type]
        self.expected_state: tuple[str, str, int] | None = None
        self.planning_context = None

    async def _prepare_target(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        selected_question_id: str | None,
        selected_work=None,
        expected_state=None,
        gap_execution_context=None,
    ):
        assert run_id == _RUN
        assert worker_task_id == "worker"
        assert selected_question_id == "real-question-17"
        assert selected_work is None
        self.expected_state = expected_state
        self.planning_context = gap_execution_context
        return ResearchTarget(
            plan_version=4,
            question_id="real-question-17",
            question="Original durable research question",
            query="query from shared planner",
            gap_id=_PROP,
            tool_call_id=UUID("50000000-0000-0000-0000-000000000004"),
            source_id_seed=UUID("50000000-0000-0000-0000-000000000005"),
            acceptance_dimensions=(("real-question-17", "existing acceptance"),),
            priority=3,
            provider_request_allowance=2,
            baseline_pages_fetched=1,
        )


def test_production_gap_mapper_rehydrates_lineage_and_typed_execution_context() -> None:
    gap, intent, gap_artifact, intent_artifact, state, work, attempt = _authoritative_inputs()
    repository = _GapRepository()
    target = _run(
        repository.prepare_target_for_work(
            _RUN,
            worker_task_id="worker",
            work_item=work,
            attempt=attempt,
            run_state=state,
            gap_artifact=gap_artifact,
            intent_artifact=intent_artifact,
        )
    )

    assert target.question_id == "real-question-17"
    assert repository.expected_state == (state.state_ref, state.state_hash, state.state_version)
    assert repository.planning_context.required_evidence == gap.required_evidence
    assert target.gap_execution_context.target_proposition_refs == (_PROP,)
    assert target.gap_execution_context.required_evidence.a2_conflict_pair_refs == (
        "conflict:left-right",
    )
    assert target.gap_execution_context.existing_evidence_refs == ("evidence:existing",)
    assert target.priority == work.priority
    assert target.execution_work_ref == work.work_ref
    assert target.gap_attempt_index == intent.attempt_index
    assert target.search_excluded_owner_keys == ("owner:existing",)

    executed_targets: list[ResearchTarget] = []
    service = ResearchLoopService(
        repository=None,  # type: ignore[arg-type]
        search=None,  # type: ignore[arg-type]
        reader=None,  # type: ignore[arg-type]
        extractor=None,  # type: ignore[arg-type]
        artifacts=None,  # type: ignore[arg-type]
        evidence_aware_context_enabled=False,
    )

    async def passthrough(
        _run_id: UUID, *, worker_task_id: str, target: ResearchTarget
    ) -> ResearchTarget:
        assert worker_task_id == "worker"
        return target

    async def fake_research(
        _run_id: UUID, *, worker_task_id: str, target: ResearchTarget
    ) -> ResearchAttemptResult:
        assert worker_task_id == "worker"
        executed_targets.append(target)
        return ResearchAttemptResult(0, 0, "test_only")

    service._enrich_target_with_research_context = passthrough  # type: ignore[method-assign]
    service._research_target = fake_research  # type: ignore[method-assign]
    execution_result = _run(
        service.execute_target(_RUN, worker_task_id="worker", target=target)
    )
    assert execution_result.outcome == "test_only"
    assert executed_targets == [target]


def test_gap_mapper_rejects_stale_cached_question_lineage() -> None:
    gap, _intent, gap_artifact, intent_artifact, state, work, attempt = _authoritative_inputs()
    stale_objective = work.objective.model_copy(update={"question_id": "invented-question"})
    stale_work = work.model_copy(update={"objective": stale_objective})
    with pytest.raises(ValueError, match="gap_target_question_lineage_stale"):
        _run(
            _GapRepository().prepare_target_for_work(
                _RUN,
                worker_task_id="worker",
                work_item=stale_work,
                attempt=attempt,
                run_state=state,
                gap_artifact=gap_artifact,
                intent_artifact=intent_artifact,
            )
        )
    assert gap.gap_ref == work.objective.gap_ref


def test_gap_mapper_fails_closed_when_durable_question_lineage_is_missing() -> None:
    _, _, gap_artifact, intent_artifact, state, work, attempt = _authoritative_inputs()
    repository = _GapRepository()
    repository._sessions = lambda: _MissingLineageSession()  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="gap_target_question_lineage_unavailable"):
        _run(
            repository.prepare_target_for_work(
                _RUN,
                worker_task_id="worker",
                work_item=work,
                attempt=attempt,
                run_state=state,
                gap_artifact=gap_artifact,
                intent_artifact=intent_artifact,
            )
        )


def test_gap_mapper_rejects_blocked_work_before_target_construction() -> None:
    _, _, gap_artifact, intent_artifact, state, work, attempt = _authoritative_inputs()
    blocked_work = work.model_copy(update={"status": ResearchWorkStatus.BLOCKED})
    with pytest.raises(ValueError, match="gap_target_work_binding_mismatch"):
        _run(
            _GapRepository().prepare_target_for_work(
                _RUN,
                worker_task_id="worker",
                work_item=blocked_work,
                attempt=attempt,
                run_state=state,
                gap_artifact=gap_artifact,
                intent_artifact=intent_artifact,
            )
        )


@pytest.mark.parametrize(
    "searchability",
    [GapSearchability.NON_SEARCHABLE, GapSearchability.UNKNOWN_SEARCHABILITY],
)
def test_gap_mapper_rejects_non_searchable_or_unknown_gap(searchability: GapSearchability) -> None:
    _, _, gap_artifact, intent_artifact, state, work, attempt = _authoritative_inputs(
        searchable=searchability
    )
    with pytest.raises(ValueError, match="gap_target_not_searchable"):
        _run(
            _GapRepository().prepare_target_for_work(
                _RUN,
                worker_task_id="worker",
                work_item=work,
                attempt=attempt,
                run_state=state,
                gap_artifact=gap_artifact,
                intent_artifact=intent_artifact,
            )
        )


def test_cold_round_trip_mapping_is_deterministic_and_attempt_bounded() -> None:
    _, _, gap_artifact, intent_artifact, state, work, attempt = _authoritative_inputs()
    durable_work = ResearchWork.model_validate(work.model_dump(mode="json"))
    first_repository = _GapRepository()
    second_repository = _GapRepository()
    first_target = _run(
        first_repository.prepare_target_for_work(
            _RUN,
            worker_task_id="worker",
            work_item=durable_work,
            attempt=attempt,
            run_state=state,
            gap_artifact=gap_artifact,
            intent_artifact=intent_artifact,
        )
    )
    second_target = _run(
        second_repository.prepare_target_for_work(
            _RUN,
            worker_task_id="worker",
            work_item=ResearchWork.model_validate(work.model_dump(mode="json")),
            attempt=attempt,
            run_state=state,
            gap_artifact=gap_artifact,
            intent_artifact=intent_artifact,
        )
    )
    assert first_target == second_target
    assert first_target.gap_execution_context == second_target.gap_execution_context

    over_bound_attempt = attempt.model_copy(update={"attempt_index": 4})
    with pytest.raises(ValueError, match="gap_target_attempt_bound_exceeded"):
        _run(
            _GapRepository().prepare_target_for_work(
                _RUN,
                worker_task_id="worker",
                work_item=work,
                attempt=over_bound_attempt,
                run_state=state,
                gap_artifact=gap_artifact,
                intent_artifact=intent_artifact,
            )
        )


def test_gap_requirements_select_existing_planner_family_without_query_template() -> None:
    # The planner receives the authoritative typed context and only reorders
    # existing families; it does not synthesize query text or change Acceptance.
    required = RequiredEvidenceCharacteristics(
        target_group_ref=_GROUP,
        target_proposition_refs=(_PROP,),
        desired_relations=(DesiredEvidenceRelation.CONTRADICTS,),
        a2_conflict_pair_refs=("pair:formal",),
    )
    from app.infrastructure.db.research_tools import _query_family_order

    planning_context = GapResearchExecutionContext(
        gap_ref=_GAP,
        intent_ref="1" * 64,
        gap_type=PropositionResearchGapType.CONFLICT_RESOLUTION,
        searchability=GapSearchability.SEARCHABLE,
        lifecycle_state=PropositionGapLifecycle.OPEN,
        target_group_ref=_GROUP,
        target_proposition_refs=(_PROP,),
        parent_claim_refs=(_CLAIM,),
        required_evidence=required,
        objective_scope_ref="scope:formal-ref",
        source_semantic_snapshot_ref="runs/semantic.json",
        source_semantic_snapshot_hash=_SNAPSHOT_HASH,
        source_gap_projection_hash="2" * 64,
        attempt_index=0,
    )
    assert _query_family_order(prefer_authoritative=False, gap_context=planning_context) == (
        QueryFamily.CONTRADICTION,
        QueryFamily.SCOPE,
        QueryFamily.AUTHORITATIVE,
        QueryFamily.ALTERNATE,
    )
