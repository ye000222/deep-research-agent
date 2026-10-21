from __future__ import annotations

from collections.abc import Coroutine
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID

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
    LegacyResearchObjective,
    PropositionGapResearchObjective,
    ResearchWork,
    ResearchWorkKind,
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
from app.infrastructure.db.research_work import ResearchWorkRepository
from app.services.gap_research_intent import GapResearchIntentService

_RUN = UUID("50000000-0000-0000-0000-000000000001")
_GROUP = "a" * 64
_PROP = UUID("10000000-0000-0000-0000-000000000001")


def _run[T](coro: Coroutine[Any, Any, T]) -> T:
    try:
        coro.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("fake repository coroutine unexpectedly yielded")


def _objective(*, snapshot_hash: str = "c" * 64) -> PropositionGapResearchObjective:
    required = RequiredEvidenceCharacteristics(
        target_group_ref=_GROUP,
        target_proposition_refs=(_PROP,),
        desired_relations=(DesiredEvidenceRelation.SUPPORTS,),
    )
    return PropositionGapResearchObjective(
        gap_ref="b" * 64,
        intent_ref="d" * 64,
        objective_scope_ref="scope:group",
        target_group_ref=_GROUP,
        target_proposition_refs=(_PROP,),
        gap_type=PropositionResearchGapType.MISSING_SUPPORT,
        required_evidence=required,
        source_semantic_snapshot_ref="runs/snapshot.json",
        source_semantic_snapshot_hash=snapshot_hash,
        source_gap_projection_hash="e" * 64,
    )


def test_gap_work_identity_is_stable_across_attempt_and_snapshot_provenance() -> None:
    first = _objective()
    later = first.model_copy(
        update={"intent_ref": "f" * 64, "source_semantic_snapshot_hash": "1" * 64}
    )
    assert proposition_gap_objective_hash(first) == proposition_gap_objective_hash(later)
    assert proposition_gap_work_identity(
        run_ref=_RUN,
        gap_ref=first.gap_ref,
        objective_hash=proposition_gap_objective_hash(first),
    ) == proposition_gap_work_identity(
        run_ref=_RUN,
        gap_ref=later.gap_ref,
        objective_hash=proposition_gap_objective_hash(later),
    )


def test_material_objective_change_changes_logical_work_identity() -> None:
    first = _objective()
    changed_required = first.required_evidence.model_copy(
        update={"desired_relations": (DesiredEvidenceRelation.CONTRADICTS,)}
    )
    changed = first.model_copy(update={"required_evidence": changed_required})
    assert proposition_gap_objective_hash(first) != proposition_gap_objective_hash(changed)


def test_research_work_objective_union_keeps_legacy_and_gap_types_disjoint() -> None:
    legacy = LegacyResearchObjective(
        question_id="q1",
        question="Question",
        rationale="Rationale",
        requirements=("evidence",),
        priority=1,
        plan_version=1,
    )
    objective = _objective()
    common = {
        "work_ref": proposition_gap_work_identity(
            run_ref=_RUN,
            gap_ref=objective.gap_ref,
            objective_hash=proposition_gap_objective_hash(objective),
        ),
        "run_ref": _RUN,
        "producer_kind": "test",
        "producer_ref": "test:gap",
        "objective_hash": proposition_gap_objective_hash(objective),
        "priority": 1,
        "source_run_state_ref": "runs/state.json",
        "source_run_state_hash": "2" * 64,
        "source_run_state_version": 1,
    }
    gap_work = ResearchWork(
        **common,
        work_kind=ResearchWorkKind.PROPOSITION_GAP,
        objective=objective,
    )
    assert isinstance(gap_work.objective, PropositionGapResearchObjective)
    legacy_work = ResearchWork(
        **{**common, "producer_ref": "test:legacy"},
        work_kind=ResearchWorkKind.LEGACY_RESEARCH,
        objective=legacy,
    )
    assert isinstance(legacy_work.objective, LegacyResearchObjective)


def test_gap_contract_fixture_is_open_only_when_searchable() -> None:
    evidence = RequiredEvidenceCharacteristics(
        target_group_ref=_GROUP,
        target_proposition_refs=(_PROP,),
        desired_relations=(DesiredEvidenceRelation.SUPPORTS,),
    )
    gap = PropositionResearchGap(
        gap_ref="b" * 64,
        run_ref=_RUN,
        target_group_ref=_GROUP,
        target_proposition_refs=(_PROP,),
        objective_scope_ref="scope:group",
        verification_assessment_ref="c" * 64,
        verification_assessment_hash="d" * 64,
        gap_type=PropositionResearchGapType.MISSING_SUPPORT,
        searchability=GapSearchability.SEARCHABLE,
        reason_codes=(PropositionGapReasonCode.NO_ELIGIBLE_SUPPORT,),
        required_evidence=evidence,
        lifecycle_state=PropositionGapLifecycle.OPEN,
        priority_inputs=GapPriorityInputs(
            priority_class=GapPriorityClass.MISSING_SUPPORT,
        ),
    )
    assert gap.searchability is GapSearchability.SEARCHABLE


def _gap() -> PropositionResearchGap:
    required = RequiredEvidenceCharacteristics(
        target_group_ref=_GROUP,
        target_proposition_refs=(_PROP,),
        desired_relations=(DesiredEvidenceRelation.SUPPORTS,),
    )
    return PropositionResearchGap(
        gap_ref="b" * 64,
        run_ref=_RUN,
        target_group_ref=_GROUP,
        target_proposition_refs=(_PROP,),
        parent_claim_refs=(),
        objective_scope_ref="scope:group",
        verification_assessment_ref="c" * 64,
        verification_assessment_hash="d" * 64,
        gap_type=PropositionResearchGapType.MISSING_SUPPORT,
        searchability=GapSearchability.SEARCHABLE,
        reason_codes=(PropositionGapReasonCode.NO_ELIGIBLE_SUPPORT,),
        required_evidence=required,
        lifecycle_state=PropositionGapLifecycle.OPEN,
        priority_inputs=GapPriorityInputs(priority_class=GapPriorityClass.MISSING_SUPPORT),
    )


class _FakeSession:
    def __init__(self, rows: list[object]) -> None:
        self.rows = rows
        self.scalar_calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def begin(self):
        return self

    async def scalar(self, _statement: object):
        self.scalar_calls += 1
        if self.scalar_calls == 1:
            return SimpleNamespace(
                id=_RUN,
                orchestrator_version="V3",
                plan_version=1,
                latest_run_state_ref="runs/state.json",
                latest_run_state_hash="2" * 64,
                latest_run_state_version=2,
            )
        return self.rows[0] if self.rows else None

    async def scalars(self, _statement: object):
        return SimpleNamespace(all=lambda: [])

    def add(self, row: object) -> None:
        row.created_at = datetime.now(UTC)
        self.rows.append(row)

    async def flush(self) -> None:
        return None


def test_gap_work_materialization_is_idempotent_and_never_resets_terminal_work() -> None:
    async def scenario() -> None:
        gap = _gap()
        gap_projection_hash = canonical_hash([gap.gap_ref])
        snapshot_ref = "runs/semantic.json"
        snapshot_hash = "1" * 64
        gap_fields = {
            "schema_version": "semantic-gap-control-projection-v1",
            "run_ref": _RUN,
            "semantic_snapshot_ref": snapshot_ref,
            "semantic_snapshot_hash": snapshot_hash,
            "projection_status": "DERIVED",
            "gaps": (gap,),
            "issues": (),
            "projection_hash": gap_projection_hash,
        }
        gap_artifact_hash = canonical_hash(
            {
                **gap_fields,
                "run_ref": str(_RUN),
                "gaps": [gap.model_dump(mode="json")],
                "issues": [],
            }
        )
        gap_artifact = SemanticGapControlArtifact(
            **gap_fields,
            artifact_hash=gap_artifact_hash,
            artifact_ref=semantic_gap_artifact_ref(_RUN, gap_artifact_hash),
        )
        intents = GapResearchIntentService().derive((gap,)).intents
        intent_fields = {
            "schema_version": "semantic-intent-control-projection-v1",
            "run_ref": _RUN,
            "semantic_snapshot_ref": snapshot_ref,
            "semantic_snapshot_hash": snapshot_hash,
            "gap_artifact_ref": gap_artifact.artifact_ref,
            "gap_projection_hash": gap_projection_hash,
            "intents": intents,
            "issues": (),
            "projection_hash": canonical_hash([item.intent_ref for item in intents]),
        }
        intent_artifact_hash = canonical_hash(
            {
                **intent_fields,
                "run_ref": str(_RUN),
                "intents": [item.model_dump(mode="json") for item in intents],
            }
        )
        intent_artifact = SemanticIntentControlArtifact(
            **intent_fields,
            artifact_hash=intent_artifact_hash,
            artifact_ref=semantic_intent_artifact_ref(_RUN, intent_artifact_hash),
        )
        state = SimpleNamespace(
            run_ref=_RUN,
            gap_projection_status=GapProjectionStatus.DERIVED,
            semantic_state_ref=snapshot_ref,
            semantic_state_hash=snapshot_hash,
            gap_source_semantic_hash=snapshot_hash,
            gap_projection_hash=gap_projection_hash,
            gap_refs=(gap.gap_ref,),
            intent_refs=tuple(sorted(item.intent_ref for item in intents)),
            intent_source_gap_hash=gap_projection_hash,
            state_ref="runs/state.json",
            state_hash="2" * 64,
            state_version=2,
        )
        rows: list[object] = []
        repository = ResearchWorkRepository(lambda: _FakeSession(rows))  # type: ignore[arg-type]
        first = await repository.materialize_proposition_gap_work(
            state, gap_artifact=gap_artifact, intent_artifact=intent_artifact
        )
        assert len(first) == 1
        assert first[0].work_kind is ResearchWorkKind.PROPOSITION_GAP
        row = rows[0]
        row.status = "COMPLETED"
        second = await repository.materialize_proposition_gap_work(
            state, gap_artifact=gap_artifact, intent_artifact=intent_artifact
        )
        assert len(second) == 1
        assert second[0].work_ref == first[0].work_ref
        assert second[0].status.value == "COMPLETED"

    _run(scenario())
