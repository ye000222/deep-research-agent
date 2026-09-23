from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from app.domain.evidence_input_quality import (
    QuoteSupportAudit,
    SourceRoleAudit,
    Suitability,
    assess_source_suitability,
    classify_quote_support,
    select_supporting_quote,
)
from app.domain.research_tools import EvidenceBatch, ReadPage
from app.infrastructure.db.research_tools import ResearchToolRepository
from app.services.evidence_extractor import EvidenceExtractorService


class _EventSession:
    def __init__(self) -> None:
        self.rows: list[Any] = []

    def add(self, row: Any) -> None:
        self.rows.append(row)

    async def flush(self) -> None:
        return None


def test_role_audit_distinguishes_media_role_from_business_role() -> None:
    suitability, audit = assess_source_suitability(
        claim_type="factual", observed_role="webpage"
    )
    assert suitability is Suitability.UNKNOWN
    assert audit is SourceRoleAudit.ROLE_CLASSIFICATION_TOO_COARSE


def test_suitable_and_true_role_mismatch_are_distinct() -> None:
    suitable, _ = assess_source_suitability(
        claim_type="market_statistic", observed_role="government"
    )
    unsuitable, audit = assess_source_suitability(
        claim_type="market_statistic", observed_role="manufacturer"
    )
    assert suitable is Suitability.SUITABLE
    assert unsuitable is Suitability.UNSUITABLE
    assert audit is SourceRoleAudit.TRUE_SOURCE_ROLE_MISMATCH


def test_missing_role_metadata_is_unknown() -> None:
    suitability, audit = assess_source_suitability(
        claim_type="factual", observed_role=None, metadata_complete=False
    )
    assert suitability is Suitability.UNKNOWN
    assert audit is SourceRoleAudit.ROLE_METADATA_MISSING


def test_claim_aware_locator_can_recover_wrong_quote() -> None:
    claim = "The market grew 20% in 2024."
    source = "Background. The market grew 20% in 2024. Other context."
    assert classify_quote_support(
        claim=claim, quote="Background context", source_text=source
    ) is QuoteSupportAudit.SUPPORT_EXISTS_WRONG_QUOTE
    assert select_supporting_quote(claim=claim, source_text=source) == (
        "The market grew 20% in 2024."
    )


def test_unrelated_quote_is_not_reclassified_as_supported() -> None:
    assert classify_quote_support(
        claim="The market grew 20% in 2024.",
        quote="A company released a product.",
        source_text="A company released a product.",
    ) is QuoteSupportAudit.SOURCE_DOES_NOT_SUPPORT_CLAIM


def test_input_quality_flag_is_opt_in_and_does_not_change_acceptance_threshold() -> None:
    page = ReadPage(
        final_url="https://example.gov/report",
        title="Report",
        clean_text=(
            "The market grew 20% in 2024. Other context. "
            "This sentence provides additional report context for the reader."
        ),
        content_hash="a" * 64,
        fetched_at=datetime.now(UTC),
    )
    batch = EvidenceBatch(
        items=[
            {
                "claim": "The market grew 20% in 2024.",
                "exact_quote": "Other context.",
                "dimension_key": "d1",
                "relevance": 0.9,
                "confidence": 0.9,
            }
        ]
    )
    baseline = EvidenceExtractorService._score_batch(
        batch,
        page=page,
        dimension_criteria={"d1": "market statistic"},
        reliability=0.9,
        input_quality_enabled=False,
    )
    candidate = EvidenceExtractorService._score_batch(
        batch,
        page=page,
        dimension_criteria={"d1": "market statistic"},
        reliability=0.9,
        input_quality_enabled=True,
    )
    assert baseline[0].input_quality is None
    assert candidate[0].input_quality is not None
    assert candidate[0].candidate.exact_quote == "The market grew 20% in 2024."
    assert baseline[0].accepted is False
    assert candidate[0].accepted is True
    assert candidate[0].input_quality["evaluation_executed"] == "true"


def test_real_scoring_result_is_persisted_as_evaluation_event_for_improved_quote() -> None:
    page = ReadPage(
        final_url="https://example.gov/report",
        title="Report",
        clean_text=(
            "The market grew 20% in 2024. Other context. "
            "This report provides additional context for the industrial market."
        ),
        content_hash="b" * 64,
        fetched_at=datetime.now(UTC),
    )
    scored = EvidenceExtractorService._score_batch(
        EvidenceBatch(
            items=[
                {
                    "claim": "The market grew 20% in 2024.",
                    "exact_quote": "Other context.",
                    "dimension_key": "d1",
                    "relevance": 0.9,
                    "confidence": 0.9,
                }
            ]
        ),
        page=page,
        dimension_criteria={"d1": "market statistic"},
        reliability=0.9,
        input_quality_enabled=True,
    )
    assert scored[0].input_quality is not None
    assert scored[0].input_quality["input_quality_decision"] == "quote_improved"

    session = _EventSession()
    run = SimpleNamespace(
        id=uuid4(),
        next_event_seq=1,
        phase="researching",
        usage_snapshot={},
    )
    asyncio.run(
        ResearchToolRepository._append_event(
            session,
            run,
            event_type="evidence.input_quality.evaluated",
            public_summary="input quality evaluated",
            refs={
                "evidence_id": str(uuid4()),
                "source_id": str(uuid4()),
                "question_id": "q1",
                "claim_id": str(uuid4()),
                "requirement_id": str(uuid4()),
                "dimension_key": "d1",
                "source": page.final_url,
                "original_quote_state": scored[0].input_quality[
                    "original_quote_state"
                ],
                "input_quality_decision": scored[0].input_quality[
                    "input_quality_decision"
                ],
                "final_quote_state": scored[0].input_quality["final_quote_state"],
            },
            metrics=scored[0].input_quality,
        )
    )
    event = session.rows[0]
    assert event.event_type == "evidence.input_quality.evaluated"
    assert event.refs["source"] == page.final_url
    assert event.metrics["evaluation_executed"] == "true"


def test_retained_quote_also_produces_input_quality_evaluation_metadata() -> None:
    page = ReadPage(
        final_url="https://example.gov/report",
        title="Report",
        clean_text=(
            "The market grew 20% in 2024. "
            "This report provides additional context for the industrial market. "
            "It also documents the regional demand outlook and research methodology."
        ),
        content_hash="c" * 64,
        fetched_at=datetime.now(UTC),
    )
    scored = EvidenceExtractorService._score_batch(
        EvidenceBatch(
            items=[
                {
                    "claim": "The market grew 20% in 2024.",
                    "exact_quote": "The market grew 20% in 2024.",
                    "dimension_key": "d1",
                    "relevance": 0.9,
                    "confidence": 0.9,
                }
            ]
        ),
        page=page,
        dimension_criteria={"d1": "market statistic"},
        reliability=0.9,
        input_quality_enabled=True,
    )
    assert scored[0].input_quality is not None
    assert scored[0].input_quality["input_quality_decision"] == "quote_retained"
    assert scored[0].input_quality["evaluation_executed"] == "true"
