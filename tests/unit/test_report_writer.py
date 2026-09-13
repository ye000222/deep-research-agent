from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID

import pytest
from app.context.manager import ContextBudgetInsufficientError
from app.domain.context import (
    ContextBudgetAllocation,
    ContextCandidate,
    ContextEnvelope,
    ContextItemType,
)
from app.domain.providers import TokenUsage, UsageAccuracy
from app.infrastructure.db.reports import (
    ReportContext,
    ReportEvidenceCard,
    ReportQuestion,
)
from app.services.report_writer import (
    DraftParagraph,
    DraftSection,
    ReportDraft,
    ReportWriterService,
    _draft_rejection_reason,
    _research_completion_has_limitations,
    _selected_writer_evidence_ids,
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


def test_writer_rejects_false_no_evidence_refusal() -> None:
    card = _card(1, "q1")
    draft = ReportDraft(
        title="无法生成研究报告: 未提供 Evidence Cards",
        executive_summary=[
            DraftParagraph(text="存在可验证结论。", evidence_ids=[str(card.evidence_id)])
        ],
        sections=[
            DraftSection(
                question_id="q1",
                title="技术路线",
                paragraphs=[
                    DraftParagraph(text="可验证结论。", evidence_ids=[str(card.evidence_id)])
                ],
            )
        ],
        limitations=[],
    )

    assert (
        _draft_rejection_reason(draft, [card])
        == "WRITER_DRAFT_FALSE_NO_EVIDENCE_REFUSAL"
    )


def test_writer_does_not_send_empty_cards_after_context_pruning() -> None:
    envelope = ContextEnvelope(
        manifest_id=UUID(int=9_999),
        allocation=ContextBudgetAllocation(
            context_window=4_096,
            input_budget=512,
            output_reserve=3_072,
            safety_margin=512,
        ),
        selected=(),
        rejected=(
            ContextCandidate(
                item_type=ContextItemType.EVIDENCE_CARD,
                content="accepted card",
                rank_score=0.9,
                source_ref_type="evidence",
                source_ref_id="evidence-1",
            ),
        ),
        token_before=10,
        token_after=0,
        rendered_prompt_hash="a" * 64,
    )

    with pytest.raises(ContextBudgetInsufficientError, match="REPORT_EVIDENCE_CARDS_PRUNED"):
        _selected_writer_evidence_ids(envelope)


def test_writer_rejects_draft_without_known_evidence_references() -> None:
    card = _card(1, "q1")
    unknown_id = UUID(int=999)
    draft = ReportDraft(
        title="工业视觉缺陷检测研究报告",
        executive_summary=[
            DraftParagraph(text="无法核验的结论。", evidence_ids=[str(unknown_id)])
        ],
        sections=[
            DraftSection(
                question_id="q1",
                title="技术路线",
                paragraphs=[
                    DraftParagraph(text="无法核验的结论。", evidence_ids=[str(unknown_id)])
                ],
            )
        ],
        limitations=[],
    )

    assert (
        _draft_rejection_reason(draft, [card])
        == "WRITER_DRAFT_NO_VALID_EVIDENCE_REFERENCES"
    )


def test_writer_fallback_does_not_downgrade_quality_met_run() -> None:
    context = replace(
        _context(_card(1, "q1"), _card(2, "q2")),
        stop_reason="quality_met",
        quality_snapshot={
            "coverage": 0.90,
            "priority_one_coverage": 0.90,
            "source_quality": 0.80,
            "cross_validation": 0.80,
            "critical_gaps": 0,
        },
    )

    assert _research_completion_has_limitations(context) is False


def test_unknown_or_non_quality_stop_reason_remains_limited() -> None:
    context = replace(_context(_card(1, "q1")), stop_reason="sources_exhausted")

    assert _research_completion_has_limitations(context) is True


def test_draft_context_hash_changes_with_evidence_or_questions() -> None:
    from app.services.report_writer import _draft_context_hash

    context = _context(_card(1, "q1"), _card(2, "q2"))
    selected = select_writer_evidence(context)
    base = _draft_context_hash(context, selected)

    assert base == _draft_context_hash(context, selected)  # deterministic
    # A new evidence card changes the assembled context hash (invalidation).
    assert base != _draft_context_hash(context, [_card(3, "q2"), *selected])
    # A question change invalidates the hash too.
    other = replace(
        context,
        questions=(
            ReportQuestion("q1", "不同的问题?", 1),
            ReportQuestion("q2", "主要厂商有哪些?", 2),
        ),
    )
    assert base != _draft_context_hash(other, selected)


@pytest.mark.asyncio
async def test_writer_reserves_and_settles_its_token_pool_before_model_call() -> None:
    card = _card(1, "q1")
    context = replace(
        _context(card),
        stop_reason="quality_met",
        budget_snapshot={
            "max_tokens": 30_000,
            "allocation": {
                "writer_tokens_initial": 4_500,
                "verification_tokens": 4_500,
                "safety_tokens": 3_000,
            },
        },
        usage_snapshot={"evidence_total_tokens": 10_000},
        quality_snapshot={
            "coverage": 1.0,
            "priority_one_coverage": 1.0,
            "source_quality": 0.9,
            "cross_validation": 1.0,
            "critical_gaps": 0,
        },
    )
    draft = ReportDraft(
        title="研究报告",
        executive_summary=[
            DraftParagraph(text="摘要结论。", evidence_ids=[str(card.evidence_id)])
        ],
        sections=[
            DraftSection(
                question_id="q1",
                title="技术路线",
                paragraphs=[
                    DraftParagraph(text="可验证结论。", evidence_ids=[str(card.evidence_id)])
                ],
            )
        ],
    )

    class Repository:
        async def load_for_writing(self, *args: object, **kwargs: object) -> ReportContext:
            return context

        async def save(self, *args: object, **kwargs: object) -> object:
            return SimpleNamespace(status="completed", citations=kwargs["citations"])

    class Bindings:
        async def get(self, *args: object, **kwargs: object) -> object:
            return SimpleNamespace(
                encrypted_secret=b"secret",
                credential_id=UUID(int=77),
                credential_version=1,
                adapter_type=SimpleNamespace(value="openai_compatible_chat"),
                base_url="https://model.example/v1",
                model="writer-model",
                max_output_tokens=5_000,
                context_window=32_000,
            )

    class Cipher:
        def decrypt(self, *args: object, **kwargs: object) -> str:
            return "key"

    class Gateway:
        def __init__(self) -> None:
            self.max_output_tokens = 0

        async def generate_structured(self, *args: object, **kwargs: object) -> object:
            request = kwargs["request"]
            self.max_output_tokens = request.max_output_tokens
            return SimpleNamespace(
                parsed_object=draft.model_dump(mode="json"),
                usage=TokenUsage(
                    input_tokens=1_000,
                    output_tokens=500,
                    total_tokens=1_500,
                    accuracy=UsageAccuracy.EXACT,
                ),
            )

    class Budget:
        def __init__(self) -> None:
            self.reserved: list[dict[str, object]] = []
            self.settled: list[dict[str, object]] = []

        async def reserve_model_tokens(self, *args: object, **kwargs: object) -> object:
            self.reserved.append(kwargs)
            return SimpleNamespace(granted=True)

        async def settle_model_reservation(self, *args: object, **kwargs: object) -> bool:
            self.settled.append(kwargs)
            return True

    gateway = Gateway()
    budget = Budget()
    service = ReportWriterService(  # type: ignore[arg-type]
        Repository(),
        Bindings(),
        Cipher(),
        gateway,
        budget_repository=budget,
    )

    result = await service.write(context.run_id, worker_task_id="worker-1")

    assert result.startswith("report_completed:completed")
    assert len(budget.reserved) == 1
    assert budget.reserved[0]["node"] == "report_writer"
    assert int(budget.reserved[0]["estimated_input"]) + int(
        budget.reserved[0]["max_output"]
    ) == 4_500
    assert gateway.max_output_tokens <= int(budget.reserved[0]["max_output"])
    assert budget.settled[0]["actual_total"] == 1_500


@pytest.mark.asyncio
async def test_writer_budget_denial_skips_model_and_uses_deterministic_fallback() -> None:
    card = _card(1, "q1")
    context = replace(
        _context(card),
        stop_reason="quality_met",
        budget_snapshot={
            "max_tokens": 30_000,
            "allocation": {
                "writer_tokens_initial": 4_500,
                "verification_tokens": 4_500,
                "safety_tokens": 3_000,
            },
        },
        usage_snapshot={"evidence_total_tokens": 10_000},
        quality_snapshot={
            "coverage": 1.0,
            "priority_one_coverage": 1.0,
            "source_quality": 0.9,
            "cross_validation": 1.0,
            "critical_gaps": 0,
        },
    )

    class Repository:
        def __init__(self) -> None:
            self.saved: dict[str, object] = {}

        async def load_for_writing(self, *args: object, **kwargs: object) -> ReportContext:
            return context

        async def save(self, *args: object, **kwargs: object) -> object:
            self.saved = kwargs
            return SimpleNamespace(status="completed", citations=kwargs["citations"])

    class NeverCalled:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"model dependency unexpectedly used: {name}")

    class Budget:
        def __init__(self) -> None:
            self.calls = 0

        async def reserve_model_tokens(self, *args: object, **kwargs: object) -> object:
            self.calls += 1
            return SimpleNamespace(granted=False)

    repository = Repository()
    budget = Budget()
    service = ReportWriterService(  # type: ignore[arg-type]
        repository,
        NeverCalled(),
        NeverCalled(),
        NeverCalled(),
        budget_repository=budget,
    )

    result = await service.write(context.run_id, worker_task_id="worker-1")

    assert result.startswith("report_completed:completed")
    assert budget.calls == 1
    assert repository.saved["writer_mode"] == "deterministic_fallback"
    assert repository.saved["usage"] is None
