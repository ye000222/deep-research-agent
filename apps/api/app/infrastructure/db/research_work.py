"""Persistence operations for V3 ResearchWork and WorkAttempt lifecycle."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.planning import ResearchPlan
from app.domain.research_runs import RunStatus
from app.domain.research_work import (
    LegacyResearchObjective,
    PropositionGapResearchObjective,
    ResearchWork,
    ResearchWorkKind,
    ResearchWorkStatus,
    WorkAttempt,
    WorkAttemptState,
    deterministic_attempt_ref,
    legacy_objective_hash,
    legacy_work_identity,
    proposition_gap_objective_hash,
    proposition_gap_work_identity,
)
from app.domain.run_state import (
    GapProjectionStatus,
    ResearchRunState,
    SemanticExecutionStatus,
    canonical_hash,
)
from app.domain.semantic_control_projection import (
    SemanticGapControlArtifact,
    SemanticIntentControlArtifact,
)
from app.infrastructure.db.evidence_graph_models import ResearchClaimRow
from app.infrastructure.db.run_models import (
    ResearchPlanItemRow,
    ResearchRunRow,
    ResearchWorkRow,
    WorkAttemptRow,
)


class ResearchWorkBindingError(RuntimeError):
    """Work materialization/reservation disagrees with canonical run control."""


class ResearchWorkRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def materialize_legacy_plan(
        self,
        run_state: ResearchRunState,
        plan: ResearchPlan,
    ) -> tuple[ResearchWork, ...]:
        """Idempotently materialize the persisted V3 plan against its state pointer."""

        materialized: list[ResearchWork] = []
        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                select(ResearchRunRow)
                .where(ResearchRunRow.id == run_state.run_ref)
                .with_for_update()
            )
            if (
                run is None
                or run.orchestrator_version != "V3"
                or run.latest_run_state_ref != run_state.state_ref
                or run.latest_run_state_hash != run_state.state_hash
                or run.latest_run_state_version != run_state.state_version
                or run.plan_version < 1
            ):
                raise ResearchWorkBindingError("canonical_v3_plan_state_binding_mismatch")
            rows = (
                await session.scalars(
                    select(ResearchPlanItemRow).where(
                        ResearchPlanItemRow.run_id == run_state.run_ref,
                        ResearchPlanItemRow.plan_version == run.plan_version,
                    )
                )
            ).all()
            # A replan replaces the planner-owned execution set. Historical
            # completed work remains auditable, while every non-terminal old
            # item is made unselectable before current-plan materialization.
            stale_rows = (
                await session.scalars(
                    select(ResearchWorkRow).where(
                        ResearchWorkRow.run_id == run_state.run_ref,
                        ResearchWorkRow.producer_kind == "LEGACY_PLANNER",
                        ResearchWorkRow.status.notin_(
                            (
                                ResearchWorkStatus.COMPLETED.value,
                                ResearchWorkStatus.SUPERSEDED.value,
                            )
                        ),
                        ~ResearchWorkRow.producer_ref.like(
                            f"plan:{run.plan_version}:question:%"
                        ),
                    )
                )
            ).all()
            for stale in stale_rows:
                stale.status = ResearchWorkStatus.SUPERSEDED.value
            by_question = {row.question_id: row for row in rows}
            for question in plan.questions:
                row = by_question.get(question.id)
                if row is None:
                    raise ResearchWorkBindingError("planner_item_not_durable")
                objective = LegacyResearchObjective(
                    question_id=row.question_id,
                    question=row.question,
                    rationale=row.rationale,
                    requirements=tuple(row.evidence_requirements),
                    priority=row.priority,
                    search_hints=tuple(row.search_hints),
                    acceptance_context=tuple(
                        (f"{row.question_id}:d{index}", str(requirement))
                        for index, requirement in enumerate(row.evidence_requirements, start=1)
                    ),
                    plan_version=row.plan_version,
                    bounds={"max_attempts": 3},
                )
                objective_hash = legacy_objective_hash(objective)
                producer_ref = f"plan:{row.plan_version}:question:{row.question_id}"
                work_ref = legacy_work_identity(
                    run_ref=run_state.run_ref,
                    plan_version=row.plan_version,
                    question_id=row.question_id,
                    objective_hash=objective_hash,
                )
                existing = await session.scalar(
                    select(ResearchWorkRow).where(
                        ResearchWorkRow.run_id == run_state.run_ref,
                        ResearchWorkRow.producer_kind == "LEGACY_PLANNER",
                        ResearchWorkRow.producer_ref == producer_ref,
                        ResearchWorkRow.schema_version == "research-work-v1",
                    )
                )
                if existing is None:
                    existing = ResearchWorkRow(
                        work_ref=work_ref,
                        run_id=run_state.run_ref,
                        producer_kind="LEGACY_PLANNER",
                        producer_ref=producer_ref,
                        work_kind=ResearchWorkKind.LEGACY_RESEARCH.value,
                        objective=objective.model_dump(mode="json"),
                        objective_hash=objective_hash,
                        priority=row.priority,
                        constraints={},
                        execution_bounds={"max_attempts": 3},
                        budget_attribution={"plan_version": row.plan_version},
                        source_run_state_ref=run_state.state_ref,
                        source_run_state_hash=run_state.state_hash,
                        source_run_state_version=run_state.state_version,
                        status=ResearchWorkStatus.ELIGIBLE.value,
                        schema_version="research-work-v1",
                    )
                    session.add(existing)
                    await session.flush()
                elif (
                    existing.work_ref != work_ref
                    or existing.objective_hash != objective_hash
                ):
                    raise ResearchWorkBindingError("deterministic_work_identity_conflict")
                materialized.append(self._work_view(existing))
        return tuple(materialized)

    async def materialize_proposition_gap_work(
        self,
        run_state: ResearchRunState,
        *,
        gap_artifact: SemanticGapControlArtifact,
        intent_artifact: SemanticIntentControlArtifact,
    ) -> tuple[ResearchWork, ...]:
        """Idempotently persist gap work; this method does not reserve or execute it."""

        if (
            run_state.gap_projection_status.value != "DERIVED"
            or gap_artifact.projection_status != "DERIVED"
            or intent_artifact.issues
            or run_state.semantic_state_ref != gap_artifact.semantic_snapshot_ref
            or run_state.semantic_state_hash != gap_artifact.semantic_snapshot_hash
            or run_state.gap_source_semantic_hash != gap_artifact.semantic_snapshot_hash
            or run_state.gap_projection_hash != gap_artifact.projection_hash
            or run_state.gap_refs != tuple(sorted(item.gap_ref for item in gap_artifact.gaps))
            or intent_artifact.gap_artifact_ref != gap_artifact.artifact_ref
            or intent_artifact.gap_projection_hash != gap_artifact.projection_hash
            or intent_artifact.semantic_snapshot_hash != gap_artifact.semantic_snapshot_hash
            or run_state.intent_source_gap_hash != gap_artifact.projection_hash
            or run_state.intent_refs
            != tuple(sorted(item.intent_ref for item in intent_artifact.intents))
            or intent_artifact.projection_hash
            != canonical_hash(list(run_state.intent_refs))
        ):
            raise ResearchWorkBindingError("current_gap_projection_binding_mismatch")
        gaps = {item.gap_ref: item for item in gap_artifact.gaps}
        if any(item.run_ref != run_state.run_ref for item in gaps.values()):
            raise ResearchWorkBindingError("gap_projection_run_mismatch")
        materialized: list[ResearchWork] = []
        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                select(ResearchRunRow)
                .where(ResearchRunRow.id == run_state.run_ref)
                .with_for_update()
            )
            if (
                run is None
                or run.orchestrator_version != "V3"
                or run.latest_run_state_ref != run_state.state_ref
                or run.latest_run_state_hash != run_state.state_hash
                or run.latest_run_state_version != run_state.state_version
            ):
                raise ResearchWorkBindingError("canonical_gap_work_state_binding_mismatch")
            for intent in intent_artifact.intents:
                gap = gaps.get(intent.gap_ref)
                if (
                    gap is None
                    or intent.run_ref != run_state.run_ref
                    or intent.target_group_ref != gap.target_group_ref
                    or intent.target_proposition_refs != gap.target_proposition_refs
                    or intent.required_evidence != gap.required_evidence
                ):
                    raise ResearchWorkBindingError("gap_intent_semantic_binding_mismatch")
                lineage_question_id: str | None = None
                lineage_source: str | None = None
                if gap.parent_claim_refs:
                    lineage_rows = (
                        await session.execute(
                            select(ResearchClaimRow.id, ResearchClaimRow.question_id).where(
                                ResearchClaimRow.run_id == run_state.run_ref,
                                ResearchClaimRow.id.in_(gap.parent_claim_refs),
                            )
                        )
                    ).all()
                    if {row.id for row in lineage_rows} != set(gap.parent_claim_refs):
                        raise ResearchWorkBindingError("gap_parent_claim_lineage_incomplete")
                    question_ids = {row.question_id for row in lineage_rows}
                    if len(question_ids) != 1:
                        raise ResearchWorkBindingError("gap_parent_claim_question_ambiguous")
                    lineage_question_id = next(iter(question_ids))
                    lineage_source = "PARENT_CLAIM_QUESTION"
                elif run.plan_version >= 1:
                    plan_question_ids = (
                        await session.scalars(
                            select(ResearchPlanItemRow.question_id).where(
                                ResearchPlanItemRow.run_id == run_state.run_ref,
                                ResearchPlanItemRow.plan_version == run.plan_version,
                            )
                        )
                    ).all()
                    if len(plan_question_ids) == 1:
                        lineage_question_id = plan_question_ids[0]
                        lineage_source = "SINGLE_RUN_PLAN_ITEM"
                objective = PropositionGapResearchObjective(
                    gap_ref=gap.gap_ref,
                    intent_ref=intent.intent_ref,
                    objective_scope_ref=gap.objective_scope_ref,
                    target_group_ref=gap.target_group_ref,
                    target_proposition_refs=gap.target_proposition_refs,
                    parent_claim_refs=gap.parent_claim_refs,
                    gap_type=gap.gap_type,
                    required_evidence=gap.required_evidence,
                    question_id=lineage_question_id,
                    question_lineage_source=lineage_source,
                    source_semantic_snapshot_ref=gap_artifact.semantic_snapshot_ref,
                    source_semantic_snapshot_hash=gap_artifact.semantic_snapshot_hash,
                    source_gap_projection_hash=gap_artifact.projection_hash,
                )
                objective_hash = proposition_gap_objective_hash(objective)
                producer_kind = "SEMANTIC_GAP_PROJECTOR"
                producer_ref = f"gap:{gap.gap_ref}:objective:{objective_hash}"
                work_ref = proposition_gap_work_identity(
                    run_ref=run_state.run_ref,
                    gap_ref=gap.gap_ref,
                    objective_hash=objective_hash,
                )
                existing = await session.scalar(
                    select(ResearchWorkRow).where(
                        ResearchWorkRow.run_id == run_state.run_ref,
                        ResearchWorkRow.producer_kind == producer_kind,
                        ResearchWorkRow.producer_ref == producer_ref,
                        ResearchWorkRow.schema_version == "research-work-v1",
                    )
                )
                if existing is None:
                    priority = min(3, max(1, intent.priority_ordering_key[0] + 1))
                    existing = ResearchWorkRow(
                        work_ref=work_ref,
                        run_id=run_state.run_ref,
                        producer_kind=producer_kind,
                        producer_ref=producer_ref,
                        work_kind=ResearchWorkKind.PROPOSITION_GAP.value,
                        objective=objective.model_dump(mode="json"),
                        objective_hash=objective_hash,
                        priority=priority,
                        constraints={"gap_ref": gap.gap_ref},
                        execution_bounds={"max_attempts": 3},
                        budget_attribution={"scope": "RUN_LEVEL", "gap_ref": gap.gap_ref},
                        source_run_state_ref=run_state.state_ref,
                        source_run_state_hash=run_state.state_hash,
                        source_run_state_version=run_state.state_version,
                        status=ResearchWorkStatus.ELIGIBLE.value,
                        schema_version="research-work-v1",
                    )
                    session.add(existing)
                    await session.flush()
                elif existing.work_ref != work_ref or existing.objective_hash != objective_hash:
                    raise ResearchWorkBindingError("gap_work_identity_conflict")
                materialized.append(self._work_view(existing))
        return tuple(materialized)

    async def reserve_next(
        self,
        run_ref: UUID,
        *,
        run_state: ResearchRunState,
        reservation_identity: str,
        work_kind: ResearchWorkKind = ResearchWorkKind.LEGACY_RESEARCH,
    ) -> tuple[ResearchWork, WorkAttempt] | None:
        """Atomically reserve one work and its next attempt before dispatch."""

        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                select(ResearchRunRow).where(ResearchRunRow.id == run_ref).with_for_update()
            )
            if (
                run is None
                or run.orchestrator_version != "V3"
                or run.latest_run_state_ref != run_state.state_ref
                or run.latest_run_state_hash != run_state.state_hash
                or run.latest_run_state_version != run_state.state_version
            ):
                raise ResearchWorkBindingError("canonical_v3_state_binding_mismatch")
            active_attempt = await session.scalar(
                select(WorkAttemptRow)
                .join(ResearchWorkRow, ResearchWorkRow.work_ref == WorkAttemptRow.work_ref)
                .where(
                    ResearchWorkRow.run_id == run_ref,
                    ResearchWorkRow.work_kind == work_kind.value,
                    *(
                        (ResearchWorkRow.producer_ref.like(
                            f"plan:{run.plan_version}:question:%"
                        ),)
                        if work_kind is ResearchWorkKind.LEGACY_RESEARCH
                        else ()
                    ),
                    ResearchWorkRow.status == ResearchWorkStatus.IN_PROGRESS.value,
                    WorkAttemptRow.state.in_(
                        (WorkAttemptState.RESERVED.value, WorkAttemptState.EXECUTING.value)
                    ),
                )
                .order_by(WorkAttemptRow.attempt_index)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if active_attempt is not None:
                active_work = await session.get(ResearchWorkRow, active_attempt.work_ref)
                if active_work is None:
                    raise ResearchWorkBindingError("active_attempt_work_not_found")
                return self._work_view(active_work), self._attempt_view(active_attempt)

            work_row = await session.scalar(
                select(ResearchWorkRow)
                .where(
                    ResearchWorkRow.run_id == run_ref,
                    ResearchWorkRow.status == ResearchWorkStatus.ELIGIBLE.value,
                    ResearchWorkRow.work_kind == work_kind.value,
                    *(
                        (ResearchWorkRow.producer_ref.like(
                            f"plan:{run.plan_version}:question:%"
                        ),)
                        if work_kind is ResearchWorkKind.LEGACY_RESEARCH
                        else ()
                    ),
                )
                .order_by(ResearchWorkRow.priority, ResearchWorkRow.producer_ref)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if work_row is None:
                return None
            latest = await session.scalar(
                select(func.max(WorkAttemptRow.attempt_index)).where(
                    WorkAttemptRow.work_ref == work_row.work_ref
                )
            )
            attempt_index = int(latest or 0) + 1
            attempt_ref = deterministic_attempt_ref(work_row.work_ref, attempt_index)
            attempt = WorkAttemptRow(
                attempt_ref=attempt_ref,
                work_ref=work_row.work_ref,
                attempt_index=attempt_index,
                state=WorkAttemptState.RESERVED.value,
                reservation_identity=reservation_identity,
                source_run_state_ref=run_state.state_ref,
                source_run_state_version=run_state.state_version,
                schema_version="work-attempt-v1",
            )
            session.add(attempt)
            work_row.status = ResearchWorkStatus.IN_PROGRESS.value
            await session.flush()
            return self._work_view(work_row), self._attempt_view(attempt)

    async def claim_attempt_execution_ownership(
        self,
        run_ref: UUID,
        attempt_ref: UUID,
        *,
        worker_task_id: str,
        run_lease_token: str,
        execution_owner_token: str,
    ) -> bool:
        """Atomically grant one caller ownership under the current Run lease.

        The Run lease is the durable expiry/takeover authority.  The existing
        WorkAttempt JSONB resume metadata records the per-invocation fencing
        token, so no second lease table or schema migration is needed.
        """

        if not worker_task_id or not run_lease_token or not execution_owner_token:
            raise ValueError("execution_owner_identity_required")
        async with self._sessions() as session, session.begin():
            run = await self._lock_current_execution_run(
                session, run_ref, worker_task_id=worker_task_id
            )
            if run.lease_owner != run_lease_token:
                raise ResearchWorkBindingError("execution_run_lease_token_mismatch")
            attempt = await session.scalar(
                select(WorkAttemptRow)
                .where(WorkAttemptRow.attempt_ref == attempt_ref)
                .with_for_update()
            )
            if attempt is None:
                raise ResearchWorkBindingError("attempt_not_found")
            work = await session.get(ResearchWorkRow, attempt.work_ref)
            if work is None or work.run_id != run.id:
                raise ResearchWorkBindingError("attempt_run_binding_mismatch")
            if (
                work.status != ResearchWorkStatus.IN_PROGRESS.value
                or attempt.state not in {
                    WorkAttemptState.RESERVED.value,
                    WorkAttemptState.EXECUTING.value,
                }
            ):
                raise ResearchWorkBindingError("attempt_not_executable")

            metadata = dict(attempt.resume_metadata or {})
            current_token = metadata.get("execution_owner_token")
            current_worker = metadata.get("execution_owner_worker_task_id")
            current_lease = metadata.get("execution_owner_run_lease_token")
            if (
                current_token == execution_owner_token
                and current_worker == worker_task_id
                and current_lease == run_lease_token
            ):
                return True
            if current_token is not None and current_lease == run_lease_token:
                # Same live Run lease, different invocation: only its original
                # token may execute.  A later invocation cannot steal it.
                return False

            # A changed Run lease token proves the previous owner lost its
            # durable lease; this is the stale-owner takeover boundary.
            metadata["execution_owner_token"] = execution_owner_token
            metadata["execution_owner_worker_task_id"] = worker_task_id
            metadata["execution_owner_run_lease_token"] = run_lease_token
            metadata["execution_owner_claimed_at"] = datetime.now(UTC).isoformat()
            attempt.resume_metadata = metadata
            await session.flush()
            return True

    async def assert_attempt_execution_ownership(
        self,
        run_ref: UUID,
        attempt_ref: UUID,
        *,
        worker_task_id: str,
        run_lease_token: str,
        execution_owner_token: str,
    ) -> bool:
        """Revalidate the current Run lease and Attempt token before dispatch."""

        async with self._sessions() as session, session.begin():
            run = await self._lock_current_execution_run(
                session, run_ref, worker_task_id=worker_task_id
            )
            if run.lease_owner != run_lease_token:
                return False
            attempt = await session.scalar(
                select(WorkAttemptRow)
                .where(WorkAttemptRow.attempt_ref == attempt_ref)
                .with_for_update()
            )
            if attempt is None:
                raise ResearchWorkBindingError("attempt_not_found")
            work = await session.get(ResearchWorkRow, attempt.work_ref)
            if work is None or work.run_id != run.id:
                raise ResearchWorkBindingError("attempt_run_binding_mismatch")
            metadata = dict(attempt.resume_metadata or {})
            return (
                work.status == ResearchWorkStatus.IN_PROGRESS.value
                and attempt.state in {
                    WorkAttemptState.RESERVED.value,
                    WorkAttemptState.EXECUTING.value,
                }
                and metadata.get("execution_owner_token") == execution_owner_token
                and metadata.get("execution_owner_worker_task_id") == worker_task_id
                and metadata.get("execution_owner_run_lease_token") == run_lease_token
            )

    @staticmethod
    async def _lock_current_execution_run(
        session: AsyncSession,
        run_ref: UUID,
        *,
        worker_task_id: str,
    ) -> ResearchRunRow:
        run = await session.scalar(
            select(ResearchRunRow)
            .where(ResearchRunRow.id == run_ref)
            .with_for_update()
        )
        now = datetime.now(UTC)
        if (
            run is None
            or run.orchestrator_version != "V3"
            or run.status != RunStatus.RUNNING.value
            or run.worker_task_id != worker_task_id
            or not run.lease_owner
            or run.lease_until is None
            or run.lease_until <= now
        ):
            raise ResearchWorkBindingError("execution_run_lease_not_owned")
        return run

    async def reconcile_proposition_gap_work_after_semantic_update(
        self,
        run_state: ResearchRunState,
        *,
        gap_artifact: SemanticGapControlArtifact | None,
        intent_artifact: SemanticIntentControlArtifact | None,
        budget_exhausted: bool,
    ) -> tuple[ResearchWork, ...]:
        """Reconcile only waiting Gap work against the current formal projection.

        Incomplete projections are deliberately a no-op: they cannot close a
        Work or make it reservable.  The caller must pass artifacts loaded from
        the authoritative RunState pointer.
        """

        if run_state.gap_projection_status is GapProjectionStatus.DERIVED:
            self._validate_current_gap_projection(
                run_state, gap_artifact=gap_artifact, intent_artifact=intent_artifact
            )
        elif gap_artifact is not None or intent_artifact is not None:
            self._validate_projection_artifact_bindings(
                run_state, gap_artifact=gap_artifact, intent_artifact=intent_artifact
            )

        async with self._sessions() as session, session.begin():
            run = await session.scalar(
                select(ResearchRunRow)
                .where(ResearchRunRow.id == run_state.run_ref)
                .with_for_update()
            )
            if (
                run is None
                or run.orchestrator_version != "V3"
                or run.latest_run_state_ref != run_state.state_ref
                or run.latest_run_state_hash != run_state.state_hash
                or run.latest_run_state_version != run_state.state_version
            ):
                raise ResearchWorkBindingError("gap_reconciliation_state_not_current")

            waiting_rows = (
                await session.scalars(
                    select(ResearchWorkRow)
                    .where(
                        ResearchWorkRow.run_id == run_state.run_ref,
                        ResearchWorkRow.work_kind == ResearchWorkKind.PROPOSITION_GAP.value,
                        ResearchWorkRow.status
                        == ResearchWorkStatus.AWAITING_REASSESSMENT.value,
                    )
                    .order_by(ResearchWorkRow.priority, ResearchWorkRow.producer_ref)
                    .with_for_update()
                )
            ).all()
            if (
                not waiting_rows
                or run_state.gap_projection_status is not GapProjectionStatus.DERIVED
            ):
                return tuple(self._work_view(row) for row in waiting_rows)
            if (
                run_state.semantic_execution_status is not SemanticExecutionStatus.COMPLETE
                or gap_artifact is None
                or intent_artifact is None
                or intent_artifact.issues
            ):
                return tuple(self._work_view(row) for row in waiting_rows)

            gaps = {gap.gap_ref: gap for gap in gap_artifact.gaps}
            intents_by_gap = {intent.gap_ref: intent for intent in intent_artifact.intents}
            reconciled: list[ResearchWork] = []
            for row in waiting_rows:
                work = self._work_view(row)
                objective = work.objective
                if not isinstance(objective, PropositionGapResearchObjective):
                    raise ResearchWorkBindingError("gap_reconciliation_objective_invalid")
                gap = gaps.get(objective.gap_ref)
                if gap is None:
                    row.status = ResearchWorkStatus.COMPLETED.value
                    reconciled.append(self._work_view(row))
                    continue

                intent = intents_by_gap.get(objective.gap_ref)
                if (
                    gap.target_group_ref != objective.target_group_ref
                    or gap.target_proposition_refs != objective.target_proposition_refs
                    or gap.parent_claim_refs != objective.parent_claim_refs
                    or gap.gap_type != objective.gap_type
                    or gap.required_evidence != objective.required_evidence
                    or gap.objective_scope_ref != objective.objective_scope_ref
                ):
                    raise ResearchWorkBindingError("gap_reconciliation_objective_changed")
                if (
                    gap.searchability.value != "SEARCHABLE"
                    or gap.lifecycle_state.value != "OPEN"
                    or intent is None
                    or intent.target_group_ref != gap.target_group_ref
                    or intent.target_proposition_refs != gap.target_proposition_refs
                    or intent.required_evidence != gap.required_evidence
                ):
                    row.status = ResearchWorkStatus.BLOCKED.value
                    reconciled.append(self._work_view(row))
                    continue

                attempts = (
                    await session.scalars(
                        select(WorkAttemptRow)
                        .where(WorkAttemptRow.work_ref == row.work_ref)
                        .order_by(WorkAttemptRow.attempt_index)
                    )
                ).all()
                objective = objective.model_copy(
                    update={
                        "intent_ref": intent.intent_ref,
                        "source_semantic_snapshot_ref": gap_artifact.semantic_snapshot_ref,
                        "source_semantic_snapshot_hash": gap_artifact.semantic_snapshot_hash,
                        "source_gap_projection_hash": gap_artifact.projection_hash,
                    }
                )
                row.objective = objective.model_dump(mode="json")
                try:
                    maximum_attempts = max(
                        1, min(3, int(row.execution_bounds.get("max_attempts", 3)))
                    )
                except (TypeError, ValueError):
                    maximum_attempts = 3
                row.status = (
                    ResearchWorkStatus.FAILED.value
                    if budget_exhausted or len(attempts) >= maximum_attempts
                    else ResearchWorkStatus.ELIGIBLE.value
                )
                reconciled.append(self._work_view(row))
            await session.flush()
            return tuple(reconciled)

    @staticmethod
    def _validate_projection_artifact_bindings(
        run_state: ResearchRunState,
        *,
        gap_artifact: SemanticGapControlArtifact | None,
        intent_artifact: SemanticIntentControlArtifact | None,
    ) -> None:
        refs = dict(run_state.execution_refs)
        if gap_artifact is None and intent_artifact is None:
            if run_state.gap_projection_status is GapProjectionStatus.DERIVED:
                raise ResearchWorkBindingError("gap_reconciliation_projection_missing")
            return
        if gap_artifact is None or intent_artifact is None:
            raise ResearchWorkBindingError("gap_reconciliation_projection_incomplete")
        if (
            refs.get("gap_projection_artifact_ref") != gap_artifact.artifact_ref
            or refs.get("intent_projection_artifact_ref") != intent_artifact.artifact_ref
            or refs.get("gap_projection_artifact_hash") != gap_artifact.artifact_hash
            or refs.get("intent_projection_artifact_hash") != intent_artifact.artifact_hash
            or gap_artifact.run_ref != run_state.run_ref
            or intent_artifact.run_ref != run_state.run_ref
            or gap_artifact.semantic_snapshot_ref != run_state.semantic_state_ref
            or gap_artifact.semantic_snapshot_hash != run_state.semantic_state_hash
            or intent_artifact.semantic_snapshot_ref != gap_artifact.semantic_snapshot_ref
            or intent_artifact.semantic_snapshot_hash != gap_artifact.semantic_snapshot_hash
            or intent_artifact.gap_artifact_ref != gap_artifact.artifact_ref
            or intent_artifact.gap_projection_hash != gap_artifact.projection_hash
        ):
            raise ResearchWorkBindingError("gap_reconciliation_projection_binding_mismatch")

    @classmethod
    def _validate_current_gap_projection(
        cls,
        run_state: ResearchRunState,
        *,
        gap_artifact: SemanticGapControlArtifact | None,
        intent_artifact: SemanticIntentControlArtifact | None,
    ) -> None:
        cls._validate_projection_artifact_bindings(
            run_state, gap_artifact=gap_artifact, intent_artifact=intent_artifact
        )
        assert gap_artifact is not None and intent_artifact is not None
        if (
            run_state.semantic_execution_status is not SemanticExecutionStatus.COMPLETE
            or gap_artifact.projection_status != "DERIVED"
            or run_state.gap_source_semantic_hash != run_state.semantic_state_hash
            or run_state.gap_projection_hash != gap_artifact.projection_hash
            or run_state.gap_refs
            != tuple(sorted(gap.gap_ref for gap in gap_artifact.gaps))
            or run_state.intent_source_gap_hash != gap_artifact.projection_hash
            or run_state.intent_refs
            != tuple(sorted(intent.intent_ref for intent in intent_artifact.intents))
            or intent_artifact.projection_hash != canonical_hash(list(run_state.intent_refs))
        ):
            raise ResearchWorkBindingError("gap_reconciliation_projection_not_current")

    async def checkpoint_attempt(
        self,
        attempt_ref: UUID,
        *,
        checkpoint: str,
        refs: dict[str, str] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> WorkAttempt:
        """Persist progress using the existing attempt JSON/ref columns."""

        if not checkpoint or len(checkpoint) > 80:
            raise ValueError("invalid_attempt_checkpoint")
        async with self._sessions() as session, session.begin():
            attempt = await session.scalar(
                select(WorkAttemptRow)
                .where(WorkAttemptRow.attempt_ref == attempt_ref)
                .with_for_update()
            )
            if attempt is None:
                raise ResearchWorkBindingError("attempt_not_found")
            if attempt.state not in {
                WorkAttemptState.RESERVED.value,
                WorkAttemptState.EXECUTING.value,
            }:
                raise ResearchWorkBindingError("attempt_not_active")
            progress = dict(attempt.resume_metadata)
            progress["last_durable_checkpoint"] = checkpoint
            progress.update(metadata or {})
            attempt.resume_metadata = progress
            for ref_kind, value in (refs or {}).items():
                if ref_kind == "query":
                    attempt.query_refs = list(dict.fromkeys([*attempt.query_refs, value]))
                elif ref_kind == "tool_call":
                    attempt.tool_call_refs = list(dict.fromkeys([*attempt.tool_call_refs, value]))
                elif ref_kind == "produced_evidence":
                    attempt.produced_evidence_refs = list(
                        dict.fromkeys([*attempt.produced_evidence_refs, value])
                    )
                elif ref_kind == "reused_evidence":
                    attempt.reused_evidence_refs = list(
                        dict.fromkeys([*attempt.reused_evidence_refs, value])
                    )
                else:
                    raise ValueError("unsupported_attempt_checkpoint_ref")
            await session.flush()
            return self._attempt_view(attempt)

    async def mark_attempt_executing(
        self,
        attempt_ref: UUID,
        *,
        run_ref: UUID,
        worker_task_id: str,
        run_lease_token: str,
        execution_owner_token: str,
    ) -> WorkAttempt:
        async with self._sessions() as session, session.begin():
            run = await self._lock_current_execution_run(
                session, run_ref, worker_task_id=worker_task_id
            )
            if run.lease_owner != run_lease_token:
                raise ResearchWorkBindingError("execution_run_lease_token_mismatch")
            attempt = await session.scalar(
                select(WorkAttemptRow)
                .where(WorkAttemptRow.attempt_ref == attempt_ref)
                .with_for_update()
            )
            if attempt is None:
                raise ResearchWorkBindingError("attempt_not_found")
            work = await session.get(ResearchWorkRow, attempt.work_ref)
            metadata = dict(attempt.resume_metadata or {})
            if (
                work is None
                or work.run_id != run.id
                or metadata.get("execution_owner_token") != execution_owner_token
                or metadata.get("execution_owner_worker_task_id") != worker_task_id
                or metadata.get("execution_owner_run_lease_token") != run_lease_token
            ):
                raise ResearchWorkBindingError("attempt_execution_owner_mismatch")
            if attempt.state != WorkAttemptState.RESERVED.value:
                raise ResearchWorkBindingError("attempt_not_reserved")
            attempt.state = WorkAttemptState.EXECUTING.value
            attempt.started_at = datetime.now(UTC)
            await session.flush()
            return self._attempt_view(attempt)

    async def settle_attempt(
        self,
        attempt_ref: UUID,
        *,
        run_ref: UUID,
        worker_task_id: str,
        run_lease_token: str,
        execution_owner_token: str,
        state: WorkAttemptState,
        query_refs: tuple[UUID, ...] = (),
        tool_call_refs: tuple[UUID, ...] = (),
        produced_evidence_refs: tuple[UUID, ...] = (),
        reused_evidence_refs: tuple[UUID, ...] = (),
        budget_refs: tuple[str, ...] = (),
        failure_classification: str | None = None,
        usage_summary: dict[str, object] | None = None,
    ) -> WorkAttempt:
        if state not in {
            WorkAttemptState.COMPLETED,
            WorkAttemptState.FAILED,
            WorkAttemptState.BLOCKED,
        }:
            raise ValueError("attempt_settlement_state_not_terminal")
        async with self._sessions() as session, session.begin():
            run = await self._lock_current_execution_run(
                session, run_ref, worker_task_id=worker_task_id
            )
            if run.lease_owner != run_lease_token:
                raise ResearchWorkBindingError("execution_run_lease_token_mismatch")
            attempt = await session.scalar(
                select(WorkAttemptRow)
                .where(WorkAttemptRow.attempt_ref == attempt_ref)
                .with_for_update()
            )
            if attempt is None:
                raise ResearchWorkBindingError("attempt_not_found")
            work = await session.get(ResearchWorkRow, attempt.work_ref)
            metadata = dict(attempt.resume_metadata or {})
            if (
                work is None
                or work.run_id != run.id
                or metadata.get("execution_owner_token") != execution_owner_token
                or metadata.get("execution_owner_worker_task_id") != worker_task_id
                or metadata.get("execution_owner_run_lease_token") != run_lease_token
            ):
                raise ResearchWorkBindingError("attempt_execution_owner_mismatch")
            if attempt.state not in {
                WorkAttemptState.RESERVED.value,
                WorkAttemptState.EXECUTING.value,
            }:
                if attempt.state == state.value:
                    return self._attempt_view(attempt)
                raise ResearchWorkBindingError("attempt_already_settled")
            attempt.state = state.value
            attempt.query_refs = list(
                dict.fromkeys([*attempt.query_refs, *(str(ref) for ref in query_refs)])
            )
            attempt.tool_call_refs = list(
                dict.fromkeys([*attempt.tool_call_refs, *(str(ref) for ref in tool_call_refs)])
            )
            attempt.produced_evidence_refs = list(
                dict.fromkeys(
                    [*attempt.produced_evidence_refs, *(str(ref) for ref in produced_evidence_refs)]
                )
            )
            attempt.reused_evidence_refs = list(
                dict.fromkeys(
                    [*attempt.reused_evidence_refs, *(str(ref) for ref in reused_evidence_refs)]
                )
            )
            attempt.budget_refs = list(dict.fromkeys([*attempt.budget_refs, *budget_refs]))
            attempt.failure_classification = failure_classification
            attempt.usage_summary = dict(usage_summary or {})
            attempt.completed_at = datetime.now(UTC)
            if state in {WorkAttemptState.COMPLETED, WorkAttemptState.FAILED} and (
                work.work_kind == ResearchWorkKind.PROPOSITION_GAP.value
            ):
                # A Gap Attempt outcome (including transport/execution failure)
                # is not a semantic verdict.  Reconcile the same logical Work
                # against the newest complete Gap/Intent projection first.
                work.status = ResearchWorkStatus.AWAITING_REASSESSMENT.value
            elif state is WorkAttemptState.COMPLETED:
                work.status = (
                    ResearchWorkStatus.COMPLETED.value
                )
            elif state is WorkAttemptState.BLOCKED:
                work.status = ResearchWorkStatus.BLOCKED.value
            else:
                max_attempts = work.execution_bounds.get("max_attempts", 3)
                try:
                    bounded_max = max(1, min(3, int(max_attempts)))
                except (TypeError, ValueError):
                    bounded_max = 3
                work.status = (
                    ResearchWorkStatus.ELIGIBLE.value
                    if attempt.attempt_index < bounded_max
                    else ResearchWorkStatus.FAILED.value
                )
            await session.flush()
            return self._attempt_view(attempt)

    @staticmethod
    def _work_view(row: ResearchWorkRow) -> ResearchWork:
        objective_data = row.objective
        objective = (
            PropositionGapResearchObjective.model_validate(objective_data)
            if objective_data.get("objective_kind") == "PROPOSITION_GAP"
            else LegacyResearchObjective.model_validate(objective_data)
        )
        return ResearchWork(
            work_ref=row.work_ref,
            run_ref=row.run_id,
            producer_kind=row.producer_kind,
            producer_ref=row.producer_ref,
            work_kind=ResearchWorkKind(row.work_kind),
            objective=objective,
            objective_hash=row.objective_hash,
            priority=row.priority,
            constraints=dict(row.constraints),
            execution_bounds=dict(row.execution_bounds),
            budget_attribution=dict(row.budget_attribution),
            source_run_state_ref=row.source_run_state_ref,
            source_run_state_hash=row.source_run_state_hash,
            source_run_state_version=row.source_run_state_version,
            status=ResearchWorkStatus(row.status),
            schema_version=row.schema_version,
            created_at=row.created_at,
        )

    @staticmethod
    def _attempt_view(row: WorkAttemptRow) -> WorkAttempt:
        return WorkAttempt(
            attempt_ref=row.attempt_ref,
            work_ref=row.work_ref,
            attempt_index=row.attempt_index,
            state=WorkAttemptState(row.state),
            reservation_identity=row.reservation_identity,
            source_run_state_ref=row.source_run_state_ref,
            source_run_state_version=row.source_run_state_version,
            query_refs=tuple(UUID(value) for value in row.query_refs),
            tool_call_refs=tuple(UUID(value) for value in row.tool_call_refs),
            produced_evidence_refs=tuple(UUID(value) for value in row.produced_evidence_refs),
            reused_evidence_refs=tuple(UUID(value) for value in row.reused_evidence_refs),
            budget_refs=tuple(row.budget_refs),
            usage_summary=dict(row.usage_summary),
            failure_classification=row.failure_classification,
            resume_metadata=dict(row.resume_metadata),
            started_at=row.started_at,
            completed_at=row.completed_at,
            schema_version=row.schema_version,
        )
