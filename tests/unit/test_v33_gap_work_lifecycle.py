from __future__ import annotations

from collections.abc import Coroutine
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
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
from app.domain.research_runs import RunStatus
from app.domain.research_work import (
    PropositionGapResearchObjective,
    ResearchWorkKind,
    ResearchWorkStatus,
    WorkAttemptState,
    deterministic_attempt_ref,
    proposition_gap_objective_hash,
    proposition_gap_work_identity,
)
from app.domain.run_state import GapProjectionStatus, SemanticExecutionStatus, canonical_hash
from app.domain.semantic_control_projection import (
    SemanticGapControlArtifact,
    SemanticIntentControlArtifact,
    semantic_gap_artifact_ref,
    semantic_intent_artifact_ref,
)
from app.infrastructure.db.research_work import (
    ResearchWorkBindingError,
    ResearchWorkRepository,
)
from app.infrastructure.db.run_models import ResearchRunRow, ResearchWorkRow, WorkAttemptRow
from app.services.gap_research_intent import GapResearchIntentService
from app.services.semantic_feedback_recovery import SemanticFeedbackRecoveryCoordinator

_RUN = UUID("50000000-0000-0000-0000-000000000001")
_GROUP = "a" * 64
_PROP = UUID("10000000-0000-0000-0000-000000000001")
_TOOL_CALL = UUID("20000000-0000-0000-0000-000000000001")


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    try:
        coro.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("fake lifecycle coroutine unexpectedly yielded")


def _gap(
    *,
    searchable: bool = True,
    gap_ref: str = "b" * 64,
) -> PropositionResearchGap:
    required = RequiredEvidenceCharacteristics(
        target_group_ref=_GROUP,
        target_proposition_refs=(_PROP,),
        desired_relations=(DesiredEvidenceRelation.SUPPORTS,),
    )
    return PropositionResearchGap(
        gap_ref=gap_ref,
        run_ref=_RUN,
        target_group_ref=_GROUP,
        target_proposition_refs=(_PROP,),
        parent_claim_refs=(),
        objective_scope_ref="scope:group",
        verification_assessment_ref="c" * 64,
        verification_assessment_hash="d" * 64,
        gap_type=PropositionResearchGapType.MISSING_SUPPORT,
        searchability=(
            GapSearchability.SEARCHABLE if searchable else GapSearchability.NON_SEARCHABLE
        ),
        reason_codes=(PropositionGapReasonCode.NO_ELIGIBLE_SUPPORT,),
        required_evidence=required,
        lifecycle_state=(
            PropositionGapLifecycle.OPEN
            if searchable
            else PropositionGapLifecycle.BLOCKED
        ),
        priority_inputs=GapPriorityInputs(priority_class=GapPriorityClass.MISSING_SUPPORT),
    )


def _projection(
    *,
    state_version: int = 2,
    include_gap: bool = True,
    searchable: bool = True,
    derived: bool = True,
) -> tuple[SimpleNamespace, SemanticGapControlArtifact, SemanticIntentControlArtifact]:
    snapshot_ref = f"runs/semantic-{state_version}.json"
    snapshot_hash = str(state_version) * 64
    gaps = (_gap(searchable=searchable),) if include_gap else ()
    gap_hash = canonical_hash([item.gap_ref for item in gaps])
    gap_fields: dict[str, Any] = {
        "schema_version": "semantic-gap-control-projection-v1",
        "run_ref": _RUN,
        "semantic_snapshot_ref": snapshot_ref,
        "semantic_snapshot_hash": snapshot_hash,
        "projection_status": "DERIVED" if derived else "INCOMPLETE",
        "gaps": gaps,
        "issues": (),
        "projection_hash": gap_hash,
    }
    gap_artifact_hash = canonical_hash(
        {
            **gap_fields,
            "run_ref": str(_RUN),
            "gaps": [item.model_dump(mode="json") for item in gaps],
            "issues": [],
        }
    )
    gap_artifact = SemanticGapControlArtifact(
        **gap_fields,
        artifact_hash=gap_artifact_hash,
        artifact_ref=semantic_gap_artifact_ref(_RUN, gap_artifact_hash),
    )
    intents = GapResearchIntentService().derive(gaps).intents if derived else ()
    intent_fields: dict[str, Any] = {
        "schema_version": "semantic-intent-control-projection-v1",
        "run_ref": _RUN,
        "semantic_snapshot_ref": snapshot_ref,
        "semantic_snapshot_hash": snapshot_hash,
        "gap_artifact_ref": gap_artifact.artifact_ref,
        "gap_projection_hash": gap_hash,
        "intents": intents,
        "issues": (),
        "projection_hash": canonical_hash([item.intent_ref for item in intents]),
    }
    intent_hash = canonical_hash(
        {
            **intent_fields,
            "run_ref": str(_RUN),
            "intents": [item.model_dump(mode="json") for item in intents],
        }
    )
    intent_artifact = SemanticIntentControlArtifact(
        **intent_fields,
        artifact_hash=intent_hash,
        artifact_ref=semantic_intent_artifact_ref(_RUN, intent_hash),
    )
    refs = (
        ("gap_projection_artifact_ref", gap_artifact.artifact_ref),
        ("gap_projection_artifact_hash", gap_artifact.artifact_hash),
        ("intent_projection_artifact_ref", intent_artifact.artifact_ref),
        ("intent_projection_artifact_hash", intent_artifact.artifact_hash),
    )
    state = SimpleNamespace(
        run_ref=_RUN,
        state_ref="runs/state.json",
        state_hash="f" * 64,
        state_version=state_version,
        execution_refs=refs,
        gap_projection_status=(
            GapProjectionStatus.DERIVED if derived else GapProjectionStatus.INCOMPLETE
        ),
        semantic_execution_status=SemanticExecutionStatus.COMPLETE,
        semantic_state_ref=snapshot_ref,
        semantic_state_hash=snapshot_hash,
        gap_source_semantic_hash=snapshot_hash if derived else None,
        gap_projection_hash=gap_hash if derived else None,
        gap_refs=tuple(sorted(item.gap_ref for item in gaps)) if derived else (),
        intent_refs=tuple(sorted(item.intent_ref for item in intents)) if derived else (),
        intent_source_gap_hash=gap_hash if derived else None,
    )
    return state, gap_artifact, intent_artifact


def _work_row(state: SimpleNamespace, gap: PropositionResearchGap | None = None) -> SimpleNamespace:
    current_gap = gap or _gap()
    intent = GapResearchIntentService().derive((current_gap,)).intents[0]
    objective = PropositionGapResearchObjective(
        gap_ref=current_gap.gap_ref,
        intent_ref=intent.intent_ref,
        objective_scope_ref=current_gap.objective_scope_ref,
        target_group_ref=current_gap.target_group_ref,
        target_proposition_refs=current_gap.target_proposition_refs,
        parent_claim_refs=current_gap.parent_claim_refs,
        gap_type=current_gap.gap_type,
        required_evidence=current_gap.required_evidence,
        source_semantic_snapshot_ref=state.semantic_state_ref,
        source_semantic_snapshot_hash=state.semantic_state_hash,
        source_gap_projection_hash=state.gap_projection_hash or "0" * 64,
    )
    objective_hash = proposition_gap_objective_hash(objective)
    return SimpleNamespace(
        work_ref=proposition_gap_work_identity(
            run_ref=_RUN, gap_ref=objective.gap_ref, objective_hash=objective_hash
        ),
        run_id=_RUN,
        producer_kind="SEMANTIC_GAP_PROJECTOR",
        producer_ref=f"gap:{objective.gap_ref}:objective:{objective_hash}",
        work_kind=ResearchWorkKind.PROPOSITION_GAP.value,
        objective=objective.model_dump(mode="json"),
        objective_hash=objective_hash,
        priority=1,
        constraints={},
        execution_bounds={"max_attempts": 3},
        budget_attribution={"scope": "RUN_LEVEL"},
        source_run_state_ref="runs/state.json",
        source_run_state_hash="f" * 64,
        source_run_state_version=state.state_version,
        status=ResearchWorkStatus.AWAITING_REASSESSMENT.value,
        schema_version="research-work-v1",
        created_at=datetime.now(UTC),
    )


class _Result:
    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return self._rows


class _Session:
    def __init__(
        self,
        *,
        state: SimpleNamespace,
        work_rows: list[object],
        attempts: list[object] | None = None,
        active_attempt: object | None = None,
        run_pointer_current: bool = True,
        reserve_kind: ResearchWorkKind = ResearchWorkKind.LEGACY_RESEARCH,
    ) -> None:
        self.state = state
        self.work_rows = work_rows
        self.attempts = attempts or []
        self.active_attempt = active_attempt
        self.run_pointer_current = run_pointer_current
        self.reserve_kind = reserve_kind
        self.added: list[object] = []
        self._scalar_work_returned = False

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def begin(self) -> _Session:
        return self

    async def scalar(self, statement: Any) -> object | None:
        if "max(" in str(statement).lower():
            return max((row.attempt_index for row in self.attempts), default=0)
        entity = statement.column_descriptions[0].get("entity")
        if entity is ResearchRunRow:
            return SimpleNamespace(
                id=_RUN,
                orchestrator_version="V3",
                status=RunStatus.RUNNING.value,
                worker_task_id="worker",
                lease_owner="lease-1",
                lease_until=datetime.now(UTC) + timedelta(days=365),
                plan_version=1,
                latest_run_state_ref=self.state.state_ref if self.run_pointer_current else "stale",
                latest_run_state_hash=self.state.state_hash,
                latest_run_state_version=self.state.state_version,
            )
        if entity is WorkAttemptRow:
            return self.active_attempt
        if entity is ResearchWorkRow:
            if self.reserve_kind is ResearchWorkKind.LEGACY_RESEARCH:
                return None
            self._scalar_work_returned = True
            eligible = [row for row in self.work_rows if row.status == "ELIGIBLE"]
            return eligible[0] if eligible else None
        if entity is None:
            return max((row.attempt_index for row in self.attempts), default=0)
        return None

    async def scalars(self, statement: Any) -> _Result:
        entity = statement.column_descriptions[0].get("entity")
        if entity is ResearchWorkRow:
            rows = [
                row
                for row in self.work_rows
                if row.status == ResearchWorkStatus.AWAITING_REASSESSMENT.value
            ]
            return _Result(rows)
        if entity is WorkAttemptRow:
            return _Result(self.attempts)
        return _Result([])

    async def get(self, entity: type[Any], key: object) -> object | None:
        if entity is ResearchWorkRow:
            return next((row for row in self.work_rows if row.work_ref == key), None)
        return None

    def add(self, row: object) -> None:
        if isinstance(row, WorkAttemptRow):
            row.query_refs = []
            row.tool_call_refs = []
            row.produced_evidence_refs = []
            row.reused_evidence_refs = []
            row.budget_refs = []
            row.usage_summary = {}
            row.resume_metadata = {}
        self.added.append(row)

    async def flush(self) -> None:
        return None


def _repository(session: _Session) -> ResearchWorkRepository:
    return ResearchWorkRepository(lambda: session)  # type: ignore[arg-type]


def _attempt(work_ref: UUID, index: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        attempt_ref=deterministic_attempt_ref(work_ref, index),
        work_ref=work_ref,
        attempt_index=index,
        state=WorkAttemptState.EXECUTING.value,
        reservation_identity=f"gap:{index}",
        source_run_state_ref="runs/state.json",
        source_run_state_version=1,
        query_refs=[],
        tool_call_refs=[],
        produced_evidence_refs=[],
        reused_evidence_refs=[],
        budget_refs=[],
        usage_summary={},
        failure_classification=None,
        resume_metadata={
            "execution_owner_token": "owner",
            "execution_owner_worker_task_id": "worker",
            "execution_owner_run_lease_token": "lease-1",
        },
        started_at=None,
        completed_at=None,
        schema_version="work-attempt-v1",
    )


def test_gap_attempt_outcomes_wait_for_semantics_but_legacy_semantics_remain_unchanged() -> None:
    async def scenario() -> None:
        state, _gap_artifact, _intent_artifact = _projection()
        gap_work = _work_row(state)
        gap_attempt = _attempt(gap_work.work_ref)
        session = _Session(
            state=state,
            work_rows=[gap_work],
            attempts=[gap_attempt],
            active_attempt=gap_attempt,
        )
        await _repository(session).settle_attempt(
            gap_attempt.attempt_ref,
            run_ref=_RUN,
            worker_task_id="worker",
            run_lease_token="lease-1",
            execution_owner_token="owner",
            state=WorkAttemptState.COMPLETED,
            tool_call_refs=(_TOOL_CALL,),
            produced_evidence_refs=(_PROP,),
        )
        assert gap_attempt.state == WorkAttemptState.COMPLETED.value
        assert gap_work.status == ResearchWorkStatus.AWAITING_REASSESSMENT.value
        assert gap_attempt.produced_evidence_refs == [str(_PROP)]

        failed_gap_work = _work_row(state)
        failed_gap_attempt = _attempt(failed_gap_work.work_ref)
        failed_gap_session = _Session(
            state=state,
            work_rows=[failed_gap_work],
            attempts=[failed_gap_attempt],
            active_attempt=failed_gap_attempt,
        )
        await _repository(failed_gap_session).settle_attempt(
            failed_gap_attempt.attempt_ref,
            run_ref=_RUN,
            worker_task_id="worker",
            run_lease_token="lease-1",
            execution_owner_token="owner",
            state=WorkAttemptState.FAILED,
            failure_classification="EXECUTION_ERROR",
        )
        assert failed_gap_attempt.state == WorkAttemptState.FAILED.value
        assert failed_gap_work.status == ResearchWorkStatus.AWAITING_REASSESSMENT.value

        legacy = SimpleNamespace(
            work_ref=UUID("30000000-0000-0000-0000-000000000001"),
            run_id=_RUN,
            work_kind=ResearchWorkKind.LEGACY_RESEARCH.value,
            status=ResearchWorkStatus.IN_PROGRESS.value,
        )
        legacy_attempt = _attempt(legacy.work_ref)
        legacy_session = _Session(
            state=state,
            work_rows=[legacy],
            attempts=[legacy_attempt],
            active_attempt=legacy_attempt,
        )
        await _repository(legacy_session).settle_attempt(
            legacy_attempt.attempt_ref,
            run_ref=_RUN,
            worker_task_id="worker",
            run_lease_token="lease-1",
            execution_owner_token="owner",
            state=WorkAttemptState.COMPLETED,
        )
        assert legacy.status == ResearchWorkStatus.COMPLETED.value

        failed_legacy = SimpleNamespace(
            work_ref=UUID("30000000-0000-0000-0000-000000000002"),
            run_id=_RUN,
            work_kind=ResearchWorkKind.LEGACY_RESEARCH.value,
            status=ResearchWorkStatus.IN_PROGRESS.value,
            execution_bounds={"max_attempts": 3},
        )
        failed_legacy_attempt = _attempt(failed_legacy.work_ref)
        failed_legacy_session = _Session(
            state=state,
            work_rows=[failed_legacy],
            attempts=[failed_legacy_attempt],
            active_attempt=failed_legacy_attempt,
        )
        await _repository(failed_legacy_session).settle_attempt(
            failed_legacy_attempt.attempt_ref,
            run_ref=_RUN,
            worker_task_id="worker",
            run_lease_token="lease-1",
            execution_owner_token="owner",
            state=WorkAttemptState.FAILED,
        )
        assert failed_legacy.status == ResearchWorkStatus.ELIGIBLE.value

    _run(scenario())


@pytest.mark.parametrize(
    ("include_gap", "searchable", "attempt_count", "budget_exhausted", "expected"),
    [
        (False, True, 1, False, ResearchWorkStatus.COMPLETED),
        (True, True, 1, False, ResearchWorkStatus.ELIGIBLE),
        (True, True, 3, False, ResearchWorkStatus.FAILED),
        (True, True, 1, True, ResearchWorkStatus.FAILED),
        (True, False, 1, False, ResearchWorkStatus.BLOCKED),
    ],
)
def test_authoritative_reconciliation_status_matrix(
    include_gap: bool,
    searchable: bool,
    attempt_count: int,
    budget_exhausted: bool,
    expected: ResearchWorkStatus,
) -> None:
    async def scenario() -> None:
        state, gap_artifact, intent_artifact = _projection(
            include_gap=include_gap, searchable=searchable
        )
        row = _work_row(state)
        attempts = [_attempt(row.work_ref, index) for index in range(1, attempt_count + 1)]
        session = _Session(state=state, work_rows=[row], attempts=attempts)
        result = await _repository(session).reconcile_proposition_gap_work_after_semantic_update(
            state,
            gap_artifact=gap_artifact,
            intent_artifact=intent_artifact,
            budget_exhausted=budget_exhausted,
        )
        assert result[0].status is expected
        assert len(attempts) == attempt_count

    _run(scenario())


def test_gap_reconciliation_is_idempotent_and_refreshes_only_provenance() -> None:
    async def scenario() -> None:
        first_state, _first_gap, _first_intent = _projection(state_version=2)
        current_state, gap_artifact, intent_artifact = _projection(state_version=3)
        row = _work_row(first_state)
        original_hash = row.objective_hash
        original_ref = row.work_ref
        attempts = [_attempt(row.work_ref)]
        session = _Session(state=current_state, work_rows=[row], attempts=attempts)
        repository = _repository(session)
        first = await repository.reconcile_proposition_gap_work_after_semantic_update(
            current_state,
            gap_artifact=gap_artifact,
            intent_artifact=intent_artifact,
            budget_exhausted=False,
        )
        second = await repository.reconcile_proposition_gap_work_after_semantic_update(
            current_state,
            gap_artifact=gap_artifact,
            intent_artifact=intent_artifact,
            budget_exhausted=False,
        )
        assert first[0].status is ResearchWorkStatus.ELIGIBLE
        assert second == ()
        assert row.status == ResearchWorkStatus.ELIGIBLE.value
        assert row.objective_hash == original_hash
        assert row.work_ref == original_ref
        assert len(attempts) == 1

    _run(scenario())


def test_gap_next_attempt_uses_same_work_and_monotonic_attempt_index() -> None:
    async def scenario() -> None:
        state, _gap_artifact, _intent_artifact = _projection()
        row = _work_row(state)
        first_attempt = _attempt(row.work_ref, 1)
        row.status = ResearchWorkStatus.ELIGIBLE.value
        session = _Session(
            state=state,
            work_rows=[row],
            attempts=[first_attempt],
            reserve_kind=ResearchWorkKind.PROPOSITION_GAP,
        )
        reserved = await _repository(session).reserve_next(
            _RUN,
            run_state=state,  # type: ignore[arg-type]
            reservation_identity="gap-attempt-2",
            work_kind=ResearchWorkKind.PROPOSITION_GAP,
        )
        assert reserved is not None
        work, attempt = reserved
        assert work.work_ref == row.work_ref
        assert attempt.attempt_index == 2
        assert attempt.attempt_ref == deterministic_attempt_ref(row.work_ref, 2)
        assert first_attempt.state == WorkAttemptState.EXECUTING.value

    _run(scenario())


def test_incomplete_projection_stays_waiting_and_stale_pointer_fails_closed() -> None:
    async def scenario() -> None:
        state, _, _ = _projection(derived=False)
        row = _work_row(state)
        session = _Session(state=state, work_rows=[row])
        result = await _repository(session).reconcile_proposition_gap_work_after_semantic_update(
            state,
            gap_artifact=None,
            intent_artifact=None,
            budget_exhausted=False,
        )
        assert result[0].status is ResearchWorkStatus.AWAITING_REASSESSMENT

        current, gap_artifact, intent_artifact = _projection()
        stale_session = _Session(
            state=current, work_rows=[_work_row(current)], run_pointer_current=False
        )
        with pytest.raises(ResearchWorkBindingError, match="state_not_current"):
            await _repository(stale_session).reconcile_proposition_gap_work_after_semantic_update(
                current,
                gap_artifact=gap_artifact,
                intent_artifact=intent_artifact,
                budget_exhausted=False,
            )

    _run(scenario())


def test_ordinary_reservation_defaults_to_legacy_only() -> None:
    async def scenario() -> None:
        state, _, _ = _projection()
        gap = _work_row(state)
        gap.status = ResearchWorkStatus.ELIGIBLE.value
        session = _Session(state=state, work_rows=[gap])
        reserved = await _repository(session).reserve_next(
            _RUN,
            run_state=state,  # type: ignore[arg-type]
            reservation_identity="legacy-default-only",
        )
        assert reserved is None
        assert gap.status == ResearchWorkStatus.ELIGIBLE.value

    _run(scenario())


def test_cold_semantic_recovery_reconciles_waiting_gap_work_without_research() -> None:
    async def scenario() -> None:
        state, _, _ = _projection(derived=False)
        state.evidence_universe_hash = "1" * 64
        state.semantic_evidence_hash = "2" * 64
        state.semantic_execution_status = SemanticExecutionStatus.INCOMPLETE
        state.semantic_engine_version = None
        state.semantic_schema_version = None
        state.semantic_state_ref = None

        class _Runs:
            async def load_evidence_universe(self, _run_ref: UUID) -> SimpleNamespace:
                return SimpleNamespace(
                    run_ref=_RUN,
                    evidence_universe_hash="3" * 64,
                    semantic_evidence_hash="4" * 64,
                    semantic_evidence_refs=(),
                )

        class _Publisher:
            async def load_current(self, _run_ref: UUID) -> SimpleNamespace:
                return state

            async def load_control_projection(
                self, _state: object
            ) -> tuple[None, None]:
                return None, None

        class _Work:
            def __init__(self) -> None:
                self.calls = 0

            async def reconcile_proposition_gap_work_after_semantic_update(
                self, _state: object, **kwargs: object
            ) -> tuple[()]:
                assert kwargs["budget_exhausted"] is False
                self.calls += 1
                return ()

        class _Budget:
            async def run_budget_exhausted(self, _run_ref: UUID) -> bool:
                return False

        work = _Work()
        coordinator = SemanticFeedbackRecoveryCoordinator(
            sessions=None,  # type: ignore[arg-type]
            runs=_Runs(),
            publisher=_Publisher(),
            loader=None,  # type: ignore[arg-type]
            semantic_pipeline=None,
            work_repository=work,
            budget_repository=_Budget(),
        )
        assert await coordinator.recover_semantic_feedback_for_run(_RUN) == 0
        assert work.calls == 1

    _run(scenario())
