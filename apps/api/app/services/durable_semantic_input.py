"""Read-only reconstruction of semantic inputs from durable Evidence facts."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.document_evidence_context import (
    CanonicalEvidenceSpan,
    SourceLocator,
    SourceLocatorType,
    _identity_hash,
    _sha256,
)
from app.domain.evidence_graph import claim_fingerprint
from app.infrastructure.artifacts import LocalArtifactStore
from app.infrastructure.db.evidence_graph_models import (
    ResearchClaimRow,
    ResearchSourceChunkRow,
    ResearchSourceSnapshotRow,
)
from app.infrastructure.db.research_models import (
    ResearchEvidenceRow,
    ResearchSourceRow,
)
from app.infrastructure.db.research_tools import (
    ClaimPersistenceReceipt,
    EvidencePersistenceReceipt,
    PagePersistenceResult,
    ResearchToolRepository,
)
from app.infrastructure.db.run_models import AgentEventRow


class DurableSemanticInputError(RuntimeError):
    """A durable page receipt could not be proven complete and consistent."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


class DurableSemanticInputLoader:
    """Rehydrate one page-scoped semantic input without using process memory."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        artifacts: LocalArtifactStore,
    ) -> None:
        self._sessions = sessions
        self._artifacts = artifacts

    async def load_page(
        self,
        run_ref: UUID,
        page_ref: str,
        evidence_refs: tuple[UUID, ...] | None = None,
    ) -> PagePersistenceResult:
        """Load and validate the committed receipt belonging to one page ref."""

        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(AgentEventRow)
                    .where(
                        AgentEventRow.run_id == run_ref,
                        AgentEventRow.event_type == "evidence.formation.diagnostic",
                    )
                    .order_by(AgentEventRow.run_seq)
                )
            ).all()
            events = [
                row
                for row in rows
                if row.refs.get("page_ref") == page_ref
                and row.refs.get("diagnostic_version") == "evidence-formation-diagnostic.v1"
            ]
            persistence = [
                row
                for row in events
                if row.refs.get("stage") == "PERSISTENCE"
                and row.refs.get("outcome") == "SUCCEEDED"
                and row.refs.get("evidence_ref")
            ]
            if not persistence:
                raise DurableSemanticInputError("DURABLE_PAGE_PERSISTENCE_EVENT_MISSING")

            persistence_by_ref: dict[UUID, Any] = {}
            for event in persistence:
                try:
                    evidence_id = UUID(str(event.refs["evidence_ref"]))
                except (KeyError, ValueError, TypeError) as exc:
                    raise DurableSemanticInputError("DURABLE_EVIDENCE_REF_INVALID") from exc
                previous = persistence_by_ref.get(evidence_id)
                if previous is not None and previous.global_id != event.global_id:
                    raise DurableSemanticInputError("DUPLICATE_PAGE_EVIDENCE_EVENT")
                persistence_by_ref[evidence_id] = event

            requested = set(evidence_refs or ())
            if requested and not requested.issubset(persistence_by_ref):
                raise DurableSemanticInputError("EVIDENCE_PAGE_BINDING_MISMATCH")
            selected_ids = requested or set(persistence_by_ref)
            if not selected_ids:
                raise DurableSemanticInputError("DURABLE_PAGE_EVIDENCE_EMPTY")

            evidence_rows = (
                await session.scalars(
                    select(ResearchEvidenceRow).where(
                        ResearchEvidenceRow.run_id == run_ref,
                        ResearchEvidenceRow.id.in_(selected_ids),
                    )
                )
            ).all()
            evidence_by_id = {row.id: row for row in evidence_rows}
            if set(evidence_by_id) != selected_ids:
                raise DurableSemanticInputError("DURABLE_EVIDENCE_ROW_MISSING")

            localization_by_evidence: dict[UUID, Any] = {}
            for event in events:
                if event.refs.get("stage") != "QUOTE_LOCALIZATION":
                    continue
                localized_evidence_id = _event_evidence_ref(event.refs, persistence_by_ref)
                if localized_evidence_id is not None and localized_evidence_id in selected_ids:
                    localization_by_evidence[localized_evidence_id] = event

            joined: dict[UUID, tuple[Any, Any, Any, Any | None]] = {}
            for evidence in evidence_rows:
                if not evidence.accepted:
                    continue
                if evidence.claim_id is None or evidence.snapshot_id is None:
                    raise DurableSemanticInputError("ACCEPTED_EVIDENCE_BINDING_INCOMPLETE")
                claim = await session.get(ResearchClaimRow, evidence.claim_id)
                source = await session.get(ResearchSourceRow, evidence.source_id)
                snapshot = await session.get(ResearchSourceSnapshotRow, evidence.snapshot_id)
                chunk = (
                    await session.get(ResearchSourceChunkRow, evidence.chunk_id)
                    if evidence.chunk_id is not None
                    else None
                )
                if claim is None or source is None or snapshot is None:
                    raise DurableSemanticInputError("DURABLE_LINEAGE_ROW_MISSING")
                if (
                    claim.run_id != run_ref
                    or source.run_id != run_ref
                    or snapshot.run_id != run_ref
                    or snapshot.source_id != source.id
                    or claim.question_id != evidence.question_id
                    or evidence.claim != claim.atomic_claim
                ):
                    raise DurableSemanticInputError("DURABLE_LINEAGE_BINDING_MISMATCH")
                if chunk is not None and (
                    chunk.run_id != run_ref or chunk.snapshot_id != snapshot.id
                ):
                    raise DurableSemanticInputError("DURABLE_CHUNK_BINDING_MISMATCH")
                joined[evidence.id] = (claim, source, snapshot, chunk)

            grouped: dict[UUID, list[EvidencePersistenceReceipt]] = defaultdict(list)
            claim_values: dict[UUID, tuple[Any, ...]] = {}
            ordered_ids = sorted(
                selected_ids,
                key=lambda evidence_id: persistence_by_ref[evidence_id].global_id,
            )
            for evidence_id in ordered_ids:
                evidence = evidence_by_id[evidence_id]
                if not evidence.accepted:
                    continue
                claim, source, snapshot, _chunk = joined[evidence_id]
                persisted_event = persistence_by_ref[evidence_id]
                persisted_facts = (
                    persisted_event.metrics.get("facts")
                    if isinstance(persisted_event.metrics, dict)
                    else None
                )
                if (
                    not isinstance(persisted_facts, dict)
                    or persisted_facts.get("row_accepted") is not True
                    or persisted_event.refs.get("page_ref") != page_ref
                    or persisted_event.refs.get("source_ref") != str(source.id)
                    or persisted_event.refs.get("snapshot_ref") != str(snapshot.id)
                ):
                    raise DurableSemanticInputError("ACCEPTED_EVIDENCE_EVENT_MISMATCH")
                if claim.claim_hash != claim_fingerprint(claim.atomic_claim):
                    raise DurableSemanticInputError("CLAIM_CONTENT_HASH_MISMATCH")
                localization_event = localization_by_evidence.get(evidence_id)
                span = await self._canonical_span(
                    evidence=evidence,
                    source=source,
                    snapshot=snapshot,
                    event=localization_event,
                )
                terminal_source_id = await self._resolve_terminal_source_id(session, source)
                owner_key = (
                    source.source_owner_key
                    if source.source_owner_key.casefold() != "unknown"
                    else None
                )
                grouped[claim.id].append(
                    EvidencePersistenceReceipt(
                        claim_id=claim.id,
                        evidence_id=evidence.id,
                        source_id=source.id,
                        snapshot_id=snapshot.id,
                        snapshot_content_hash=snapshot.content_hash,
                        question_id=evidence.question_id,
                        dimension_key=claim.dimension_key,
                        source_owner_key=owner_key,
                        canonical_span=span,
                        accepted_and_persisted=True,
                    )
                )
                if claim.id not in claim_values:
                    claim_values[claim.id] = (
                        claim,
                        terminal_source_id,
                        owner_key,
                    )

            claim_receipts: list[ClaimPersistenceReceipt] = []
            for claim_id in sorted(grouped, key=str):
                receipts = tuple(grouped[claim_id])
                claim, terminal_source_id, owner_key = claim_values[claim_id]
                evidence_ids = tuple(item.evidence_id for item in receipts)
                source_ids = tuple(dict.fromkeys(item.source_id for item in receipts))
                snapshot_ids = tuple(dict.fromkeys(item.snapshot_id for item in receipts))
                provenance_refs = [f"claim:{claim_id}"]
                for item in receipts:
                    receipt_source = await session.get(ResearchSourceRow, item.source_id)
                    if receipt_source is None:
                        raise DurableSemanticInputError("DURABLE_SOURCE_ROW_MISSING")
                    terminal_id = await self._resolve_terminal_source_id(session, receipt_source)
                    for ref in (
                        f"evidence:{item.evidence_id}",
                        f"source:{item.source_id}",
                        f"source_terminal:{terminal_id}",
                        f"snapshot:{item.snapshot_id}",
                    ):
                        if ref not in provenance_refs:
                            provenance_refs.append(ref)
                claim_receipts.append(
                    ClaimPersistenceReceipt(
                        claim_id=claim_id,
                        evidence_ids=evidence_ids,
                        source_ids=source_ids,
                        snapshot_ids=snapshot_ids,
                        source_provenance_refs=tuple(provenance_refs),
                        source_diversity_key=hashlib.sha256(
                            str(terminal_source_id).encode("ascii")
                        ).hexdigest(),
                        source_owner_key=owner_key,
                        claim_content_hash=claim.claim_hash,
                        claim_text=claim.atomic_claim,
                        evidence_receipts=receipts,
                        created_at=claim.created_at,
                        accepted_and_persisted=True,
                    )
                )

            return PagePersistenceResult(
                inserted_count=len(selected_ids),
                accepted_count=sum(row.accepted for row in evidence_rows),
                claim_receipts=tuple(claim_receipts),
            )

    async def _canonical_span(
        self,
        *,
        evidence: Any,
        source: Any,
        snapshot: Any,
        event: Any | None,
    ) -> CanonicalEvidenceSpan:
        if event is None:
            raise DurableSemanticInputError("CANONICAL_SPAN_EVENT_MISSING")
        facts = event.metrics.get("facts") if isinstance(event.metrics, dict) else None
        if not isinstance(facts, dict) or facts.get("localized") is not True:
            raise DurableSemanticInputError("CANONICAL_SPAN_FACTS_INCOMPLETE")
        try:
            text = await self._artifacts.read_text(snapshot.artifact_uri)
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            start = int(facts["char_start"])
            end = int(facts["char_end"])
            locator_type = SourceLocatorType(str(facts["source_locator_type"]))
            locator = SourceLocator(
                locator_type=locator_type,
                page_start=facts.get("source_page_start"),
                page_end=facts.get("source_page_end"),
                block_ordinal=facts.get("source_block_ordinal"),
            )
            span_ref = str(facts["span_ref"])
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise DurableSemanticInputError("CANONICAL_SPAN_FACTS_INVALID") from exc
        if content_hash != snapshot.content_hash or content_hash != source.content_hash:
            raise DurableSemanticInputError("SOURCE_ARTIFACT_HASH_MISMATCH")
        if end <= start or start < 0 or end > len(text):
            raise DurableSemanticInputError("CANONICAL_SPAN_OFFSET_INVALID")
        exact_text = text[start:end]
        if exact_text != evidence.exact_quote:
            raise DurableSemanticInputError("CANONICAL_SPAN_QUOTE_MISMATCH")
        if locator_type is SourceLocatorType.UNKNOWN:
            raise DurableSemanticInputError("CANONICAL_SPAN_LOCATOR_UNKNOWN")
        if locator_type is SourceLocatorType.PDF_PAGE and (
            locator.page_start is None or locator.page_end is None
        ):
            raise DurableSemanticInputError("CANONICAL_SPAN_LOCATOR_INCOMPLETE")
        if locator_type in {SourceLocatorType.HTML_SECTION, SourceLocatorType.DOCUMENT_BLOCK} and (
            locator.block_ordinal is None
        ):
            raise DurableSemanticInputError("CANONICAL_SPAN_LOCATOR_INCOMPLETE")

        snapshot_ref = _sha256(snapshot.content_hash)
        document_ref = _identity_hash(
            "document", {"source_ref": str(source.id), "snapshot_ref": snapshot_ref}
        )
        content_sha256 = hashlib.sha256(exact_text.encode("utf-8")).hexdigest()
        expected_span_ref = _identity_hash(
            "canonical-evidence-span",
            {
                "document_ref": document_ref,
                "start": start,
                "end": end,
                "content_sha256": content_sha256,
            },
        )
        if span_ref != expected_span_ref:
            raise DurableSemanticInputError("CANONICAL_SPAN_REF_MISMATCH")
        return CanonicalEvidenceSpan(
            span_ref=span_ref,
            document_ref=document_ref,
            source_ref=str(source.id),
            snapshot_ref=snapshot_ref,
            char_start=start,
            char_end=end,
            exact_text=exact_text,
            content_sha256=content_sha256,
            source_locator=locator,
        )

    @staticmethod
    async def _resolve_terminal_source_id(session: AsyncSession, source: Any) -> UUID:
        try:
            return await ResearchToolRepository._terminal_source_id(session, source)
        except ValueError as exc:
            raise DurableSemanticInputError("SOURCE_PROVENANCE_CHAIN_INVALID") from exc


def _event_evidence_ref(refs: dict[str, Any], persistence_by_ref: dict[UUID, Any]) -> UUID | None:
    raw = refs.get("evidence_ref")
    if raw is None:
        # Localization events precede the Evidence insert; their candidate ref
        # binds to the persistence event for this page and snapshot.
        candidate_ref = refs.get("candidate_ref")
        page_ref = refs.get("page_ref")
        snapshot_ref = refs.get("snapshot_ref")
        matches = [
            evidence_id
            for evidence_id, event in persistence_by_ref.items()
            if event.refs.get("candidate_ref") == candidate_ref
            and event.refs.get("page_ref") == page_ref
            and event.refs.get("snapshot_ref") == snapshot_ref
        ]
        return matches[0] if len(matches) == 1 else None
    try:
        return UUID(str(raw))
    except (ValueError, TypeError):
        return None
