"""PostgreSQL Evidence Store retrieval tool with auditable idempotency."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.domain.controlled_tools import (
    EvidenceSearchCard,
    EvidenceSearchInput,
    EvidenceSearchResult,
)
from app.domain.identifiers import uuid7
from app.infrastructure.db.research_models import (
    ResearchEvidenceRow,
    ResearchSourceRow,
    ResearchToolCallRow,
)
from app.retrieval.models import EvidenceSearchDocumentRow
from app.retrieval.normalization import normalize_text
from app.tools.errors import ToolExecutionError

_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._%+-]*|[\u3400-\u9fff]")


class SearchEvidenceTool:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def execute(self, request: EvidenceSearchInput) -> EvidenceSearchResult:
        duplicate_key = hashlib.sha256(
            (
                f"{request.question_id}:{' '.join(request.query.lower().split())}:"
                f"{request.min_score}:{request.top_k}"
            ).encode()
        ).hexdigest()
        now = datetime.now(UTC)
        async with self._sessions() as session, session.begin():
            call = await session.scalar(
                select(ResearchToolCallRow).where(
                    ResearchToolCallRow.run_id == request.run_id,
                    ResearchToolCallRow.tool_name == "search_evidence",
                    ResearchToolCallRow.duplicate_key == duplicate_key,
                )
            )
            if call is None:
                call = ResearchToolCallRow(
                    id=uuid7(),
                    run_id=request.run_id,
                    question_id=request.question_id,
                    gap_id=request.target_gap_ids[0],
                    action_id=request.action_id,
                    tool_name="search_evidence",
                    duplicate_key=duplicate_key,
                    status="running",
                    arguments={
                        "query": request.query,
                        "min_score": request.min_score,
                        "top_k": request.top_k,
                    },
                    result_refs={},
                    started_at=now,
                )
                session.add(call)
                await session.flush()
            else:
                if call.status == "succeeded":
                    raw_ids = call.result_refs.get("evidence_ids", [])
                    replay_ids = tuple(UUID(str(item)) for item in raw_ids)
                    if not replay_ids:
                        return EvidenceSearchResult(
                            call_id=call.id,
                            status="success",
                            items=(),
                            result_refs=(),
                        )
                    replay_rows = (
                        await session.execute(
                            select(ResearchEvidenceRow, ResearchSourceRow)
                            .join(
                                ResearchSourceRow,
                                ResearchEvidenceRow.source_id == ResearchSourceRow.id,
                            )
                            .where(ResearchEvidenceRow.id.in_(replay_ids))
                        )
                    ).tuples()
                    by_id = {evidence.id: (evidence, source) for evidence, source in replay_rows}
                    cards = tuple(
                        EvidenceSearchCard(
                            evidence_id=evidence.id,
                            claim_id=evidence.claim_id,
                            snapshot_id=evidence.snapshot_id,
                            question_id=evidence.question_id,
                            claim=evidence.claim,
                            exact_quote=evidence.exact_quote,
                            relation=evidence.relation,
                            source_title=source.title,
                            source_url=source.canonical_url,
                            source_owner_key=source.source_owner_key,
                            evidence_score=evidence.evidence_score,
                            retrieval_score=evidence.evidence_score,
                        )
                        for evidence_id in replay_ids
                        if (pair := by_id.get(evidence_id)) is not None
                        for evidence, source in (pair,)
                    )
                    return EvidenceSearchResult(
                        call_id=call.id,
                        status="success",
                        items=cards,
                        result_refs=tuple(item.evidence_id for item in cards),
                    )
                raise ToolExecutionError("TOOL_IN_PROGRESS", retryable=True)
            normalized = normalize_text(request.query)
            query_text = normalized.cjk_lexemes or normalized.latin_text or normalized.raw
            ts_query = func.plainto_tsquery("simple", query_text)
            lexical_score = func.ts_rank_cd(EvidenceSearchDocumentRow.search_vector, ts_query)
            fuzzy_score = func.similarity(
                EvidenceSearchDocumentRow.fuzzy_text, normalized.fuzzy_text
            )
            rows = (
                await session.execute(
                    select(
                        ResearchEvidenceRow,
                        ResearchSourceRow,
                        lexical_score.label("lexical_score"),
                        fuzzy_score.label("fuzzy_score"),
                    )
                    .join(
                        ResearchSourceRow,
                        ResearchEvidenceRow.source_id == ResearchSourceRow.id,
                    )
                    .join(
                        EvidenceSearchDocumentRow,
                        EvidenceSearchDocumentRow.evidence_id == ResearchEvidenceRow.id,
                    )
                    .where(
                        ResearchEvidenceRow.run_id == request.run_id,
                        ResearchEvidenceRow.accepted.is_(True),
                        or_(
                            EvidenceSearchDocumentRow.search_vector.op("@@")(ts_query),
                            fuzzy_score >= 0.05,
                        ),
                    )
                    .order_by(
                        (0.65 * lexical_score + 0.20 * fuzzy_score).desc(),
                        ResearchEvidenceRow.evidence_score.desc(),
                        ResearchEvidenceRow.created_at,
                    )
                    .limit(200)
                )
            ).all()
            query_terms = _terms(request.query)
            scored: list[tuple[ResearchEvidenceRow, ResearchSourceRow, float]] = []
            for evidence, source, db_lexical, db_fuzzy in rows:
                lexical = max(
                    float(db_lexical or 0.0),
                    _overlap(query_terms, _terms(f"{evidence.claim} {evidence.exact_quote}")),
                )
                fuzzy = float(db_fuzzy or 0.0)
                question_bonus = 0.25 if evidence.question_id == request.question_id else 0.0
                retrieval_score = min(
                    1.0,
                    0.65 * min(1.0, lexical)
                    + 0.20 * fuzzy
                    + 0.15 * evidence.evidence_score
                    + question_bonus,
                )
                if retrieval_score >= request.min_score:
                    scored.append((evidence, source, retrieval_score))
            scored.sort(key=lambda item: (-item[2], -item[0].evidence_score, str(item[0].id)))
            selected = scored[: request.top_k]
            cards = tuple(
                EvidenceSearchCard(
                    evidence_id=evidence.id,
                    claim_id=evidence.claim_id,
                    snapshot_id=evidence.snapshot_id,
                    question_id=evidence.question_id,
                    claim=evidence.claim,
                    exact_quote=evidence.exact_quote,
                    relation=evidence.relation,
                    source_title=source.title,
                    source_url=source.canonical_url,
                    source_owner_key=source.source_owner_key,
                    evidence_score=evidence.evidence_score,
                    retrieval_score=round(score, 6),
                )
                for evidence, source, score in selected
            )
            call.status = "succeeded"
            call.result_refs = {"evidence_ids": [str(item.evidence_id) for item in cards]}
            call.error_code = None
            call.retryable = None
            call.finished_at = now
            return EvidenceSearchResult(
                call_id=call.id,
                status="success",
                items=cards,
                result_refs=tuple(item.evidence_id for item in cards),
            )


def _terms(value: str) -> set[str]:
    chars = [match.group(0).lower() for match in _TERM_RE.finditer(value)]
    terms = set(chars)
    cjk = [item for item in chars if len(item) == 1 and "\u3400" <= item <= "\u9fff"]
    terms.update("".join(cjk[index : index + 2]) for index in range(max(0, len(cjk) - 1)))
    return terms


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)
