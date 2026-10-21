"""Bounded pull recovery for semantic feedback over durable Evidence."""

from __future__ import annotations

from collections import defaultdict
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.run_state import (
    SEMANTIC_STATE_SCHEMA,
    ResearchRunState,
    SemanticExecutionStatus,
)
from app.infrastructure.db.run_models import AgentEventRow, ResearchWorkRow, WorkAttemptRow
from app.services.durable_semantic_input import (
    DurableSemanticInputError,
    DurableSemanticInputLoader,
)
from app.services.semantic_engine import SEMANTIC_ENGINE_V23


class SemanticRecoveryError(RuntimeError):
    """Semantic recovery could not prove a bounded durable replay candidate."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class SemanticFeedbackRecoveryCoordinator:
    """Discover and replay only page-scoped durable Evidence for one run."""

    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        runs: Any,
        publisher: Any,
        loader: DurableSemanticInputLoader,
        semantic_pipeline: Any,
        work_repository: Any | None = None,
        budget_repository: Any | None = None,
    ) -> None:
        self._sessions = sessions
        self._runs = runs
        self._publisher = publisher
        self._loader = loader
        self._semantic_pipeline = semantic_pipeline
        self._work_repository = work_repository
        self._budget_repository = budget_repository

    async def recover_semantic_feedback_for_run(self, run_ref: UUID) -> int:
        """Return the number of page inputs replayed; synced runs are no-ops."""

        previous = await self._publisher.load_current(run_ref)
        if previous is None:
            raise SemanticRecoveryError("AUTHORITATIVE_RUN_STATE_MISSING")
        universe = await self._runs.load_evidence_universe(run_ref)
        if universe.run_ref != run_ref:
            raise SemanticRecoveryError("EVIDENCE_UNIVERSE_RUN_MISMATCH")
        if await self._state_is_current(previous, universe):
            await self._reconcile_gap_work(previous)
            return 0
        if not universe.semantic_evidence_refs:
            # No semantic-eligible input exists; do not fabricate an empty recovery.
            await self._reconcile_gap_work(previous)
            return 0

        pages = await self._discover_pages(run_ref, set(universe.semantic_evidence_refs))
        if not pages:
            raise SemanticRecoveryError("SEMANTIC_LAG_WITHOUT_DURABLE_PAGE_REFS")

        replayed = 0
        for page_ref, evidence_refs in sorted(pages.items()):
            try:
                persisted = await self._loader.load_page(
                    run_ref,
                    page_ref,
                    tuple(sorted(evidence_refs, key=str)),
                )
            except DurableSemanticInputError as exc:
                raise SemanticRecoveryError(exc.reason_code) from exc
            if persisted.receipt_status != "AVAILABLE":
                raise SemanticRecoveryError("DURABLE_PAGE_RECEIPT_UNAVAILABLE")
            await self._semantic_pipeline.process_persisted_page(run_ref, persisted)
            replayed += 1

        latest = await self._publisher.load_current(run_ref)
        current_universe = await self._runs.load_evidence_universe(run_ref)
        if latest is None or not await self._state_is_current(latest, current_universe):
            raise SemanticRecoveryError("SEMANTIC_RECOVERY_DID_NOT_CONVERGE")
        await self._reconcile_gap_work(latest)
        return replayed

    async def _reconcile_gap_work(self, state: ResearchRunState) -> None:
        if self._work_repository is None or self._budget_repository is None:
            return
        gap_artifact, intent_artifact = await self._publisher.load_control_projection(state)
        budget_exhausted = await self._budget_repository.run_budget_exhausted(state.run_ref)
        await self._work_repository.reconcile_proposition_gap_work_after_semantic_update(
            state,
            gap_artifact=gap_artifact,
            intent_artifact=intent_artifact,
            budget_exhausted=budget_exhausted,
        )

    async def _state_is_current(self, state: ResearchRunState, universe: Any) -> bool:
        if (
            state.evidence_universe_hash != universe.evidence_universe_hash
            or state.semantic_evidence_hash != universe.semantic_evidence_hash
            or state.semantic_execution_status is not SemanticExecutionStatus.COMPLETE
            or state.semantic_engine_version != SEMANTIC_ENGINE_V23
            or state.semantic_schema_version != SEMANTIC_STATE_SCHEMA
            or state.semantic_state_ref is None
        ):
            return False
        snapshot = await self._publisher.load_semantic_snapshot(state)
        return bool(
            snapshot is not None
            and snapshot.semantic_evidence_refs == universe.semantic_evidence_refs
            and snapshot.evidence_universe_hash == universe.evidence_universe_hash
            and snapshot.semantic_evidence_hash == universe.semantic_evidence_hash
        )

    async def recover_known_orphan_candidate(self, artifact_ref: str) -> bool:
        """Use the existing pointer-CAS seam when an exact artifact ref is known."""

        if not artifact_ref:
            raise SemanticRecoveryError("KNOWN_ORPHAN_ARTIFACT_REF_REQUIRED")
        return bool(await self._publisher.recover_orphan_candidate(artifact_ref))

    async def _discover_pages(
        self, run_ref: UUID, semantic_evidence_refs: set[UUID]
    ) -> dict[str, set[UUID]]:
        async with self._sessions() as session:
            attempt_rows = (
                await session.execute(
                    select(WorkAttemptRow.produced_evidence_refs)
                    .join(ResearchWorkRow, ResearchWorkRow.work_ref == WorkAttemptRow.work_ref)
                    .where(ResearchWorkRow.run_id == run_ref)
                )
            ).all()
            attempted_refs: set[UUID] = set()
            for (raw_refs,) in attempt_rows:
                for raw_ref in raw_refs or ():
                    try:
                        attempted_refs.add(UUID(str(raw_ref)))
                    except (TypeError, ValueError) as exc:
                        raise SemanticRecoveryError("ATTEMPT_EVIDENCE_REF_INVALID") from exc

            event_rows = (
                await session.scalars(
                    select(AgentEventRow)
                    .where(
                        AgentEventRow.run_id == run_ref,
                        AgentEventRow.event_type == "evidence.formation.diagnostic",
                    )
                    .order_by(AgentEventRow.run_seq)
                )
            ).all()

        pages: dict[str, set[UUID]] = defaultdict(set)
        event_refs: dict[UUID, str] = {}
        for event in event_rows:
            refs = event.refs
            if (
                refs.get("diagnostic_version") != "evidence-formation-diagnostic.v1"
                or refs.get("stage") != "PERSISTENCE"
                or refs.get("outcome") != "SUCCEEDED"
            ):
                continue
            page_ref = refs.get("page_ref")
            raw_evidence_ref = refs.get("evidence_ref")
            if not isinstance(page_ref, str) or not isinstance(raw_evidence_ref, str):
                continue
            try:
                evidence_ref = UUID(raw_evidence_ref)
            except ValueError as exc:
                raise SemanticRecoveryError("FORMATION_EVIDENCE_REF_INVALID") from exc
            if evidence_ref not in semantic_evidence_refs:
                continue
            if evidence_ref in event_refs and event_refs[evidence_ref] != page_ref:
                raise SemanticRecoveryError("EVIDENCE_MULTIPLE_PAGE_BINDINGS")
            event_refs[evidence_ref] = page_ref

        # Durable Attempt refs are a required corroborating source, not an
        # alternative authority to the current Evidence universe/events.
        missing_event_refs = semantic_evidence_refs - set(event_refs)
        if missing_event_refs:
            raise SemanticRecoveryError("SEMANTIC_EVIDENCE_PAGE_EVENT_MISSING")
        missing = semantic_evidence_refs - attempted_refs
        if missing:
            raise SemanticRecoveryError("ATTEMPT_PAGE_EVIDENCE_BINDING_INCOMPLETE")
        for evidence_ref in semantic_evidence_refs.intersection(event_refs):
            pages[event_refs[evidence_ref]].add(evidence_ref)
        return dict(pages)
