from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

from app.infrastructure.db.reports import (
    ReportContext,
    ReportEvidenceCard,
    ReportQuestion,
)
from app.services.report_writer import (
    DraftParagraph,
    DraftSection,
    ReportDraft,
    assemble_report,
    select_writer_evidence,
)


def _card(number: int, question_id: str, score: float = 0.8) -> ReportEvidenceCard:
    return ReportEvidenceCard(
        evidence_id=UUID(int=number),
        claim_id=UUID(int=1_000 + number),
        snapshot_id=UUID(int=2_000 + number),
        chunk_id=UUID(int=3_000 + number),
        question_id=question_id,
        claim=f"可验证结论 {number}",
        exact_quote=f"原文证据 {number}",
        evidence_score=score,
        source_title=f"来源 {number}",
        source_url=f"https://example{number}.com/source",
        source_domain=f"example{number}.com",
        source_content_hash=f"{number:064x}",
        fetched_at=datetime(2026, 8, 26, tzinfo=UTC),
    )


def _context(*cards: ReportEvidenceCard) -> ReportContext:
    return ReportContext(
        run_id=UUID(int=100),
        goal="研究工业视觉缺陷检测",
        stop_reason="research_budget_exhausted",
        budget_snapshot={"max_tokens": 100_000},
        usage_snapshot={"evidence_total_tokens": 95_000},
        quality_snapshot={"coverage": 1.0, "cross_validation": 0.5},
        questions=(
            ReportQuestion("q1", "技术路线是什么?", 1),
            ReportQuestion("q2", "主要厂商有哪些?", 2),
        ),
        evidence=tuple(cards),
    )


def test_selection_keeps_question_coverage_and_is_bounded() -> None:
    cards = [_card(index, "q1" if index < 22 else "q2", index / 100) for index in range(1, 26)]

    selected = select_writer_evidence(_context(*cards))

    assert len(selected) == 20
    assert {item.question_id for item in selected} == {"q1", "q2"}


def test_assembler_maps_only_known_evidence_to_stable_citations() -> None:
    artifact_id = UUID(int=9_001)
    first = replace(
        _card(1, "q1"),
        analysis_artifact_id=artifact_id,
        analysis_operation="cagr",
        analysis_formula="(end/start)^(1/years)-1",
        analysis_result={"value": 0.249},
    )
    second = _card(2, "q2")
    draft = ReportDraft(
        title="工业视觉缺陷检测研究报告",
        executive_summary=[
            DraftParagraph(
                text="研究显示存在两条主要观察。",
                evidence_ids=[str(first.evidence_id), str(second.evidence_id)],
            )
        ],
        sections=[
            DraftSection(
                question_id="q1",
                title="技术路线",
                paragraphs=[
                    DraftParagraph(
                        text="技术路线结论 [99]。",
                        evidence_ids=[str(first.evidence_id), str(UUID(int=999))],
                    )
                ],
            ),
            DraftSection(
                question_id="q2",
                title="主要厂商",
                paragraphs=[
                    DraftParagraph(
                        text="厂商结论。",
                        evidence_ids=[str(second.evidence_id)],
                    )
                ],
            ),
        ],
        limitations=[],
    )

    report = assemble_report(
        _context(first, second),
        [first, second],
        draft=draft,
        fallback_reason=None,
    )

    assert [item.citation_number for item in report.citations] == [1, 2]
    assert [item.evidence_id for item in report.citations] == [
        first.evidence_id,
        second.evidence_id,
    ]
    assert [item.claim_id for item in report.citations] == [
        first.claim_id,
        second.claim_id,
    ]
    assert [item.snapshot_id for item in report.citations] == [
        first.snapshot_id,
        second.snapshot_id,
    ]
    assert [item.chunk_id for item in report.citations] == [
        first.chunk_id,
        second.chunk_id,
    ]
    assert report.citations[0].analysis_artifact_id == artifact_id
    assert report.verification_result["analysis_artifact_citations"] == 1
    assert "[99]" not in report.final_markdown
    assert report.verification_result["citation_completeness"] == 1.0
    assert report.verification_result["numeric_citation_rate"] == 1.0
    assert report.verification_result["verified"] is True
    assert any("预算已耗尽" in item for item in report.limitations)


def test_deterministic_fallback_never_creates_uncited_claims() -> None:
    first = _card(1, "q1")
    second = _card(2, "q2")

    report = assemble_report(
        _context(first, second),
        [first, second],
        draft=None,
        fallback_reason="MODEL_TIMEOUT",
    )

    assert report.citations
    assert report.verification_result["verified"] is True
    assert "Writer 使用证据模板降级生成" in report.final_markdown
    assert all(f"[{item.citation_number}]" in report.final_markdown for item in report.citations)

