from dataclasses import replace
from uuid import UUID

import pytest
from app.context.manager import ContextBudgetInsufficientError
from app.services.report_writer import (
    WRITER_INSTRUCTIONS,
    DraftParagraph,
    DraftSection,
    ReportDraft,
    _draft_rejection_reason,
)
from app.services.writer_context import pack_writer_context

from tests.unit.test_event_coverage import GOAL, QUOTES, cards_for
from tests.unit.test_report_writer import _card, _context


def pack(context, cards, budget=2488):
    return pack_writer_context(
        context,
        cards,
        instructions=WRITER_INSTRUCTIONS,
        schema=ReportDraft.model_json_schema(),
        input_budget=budget,
    )


def test_duplicate_claims_and_sources_are_packed_once_with_intact_quotes():
    first = _card(1, "q1")
    duplicate = replace(first, evidence_id=UUID(int=2))
    other = replace(_card(3, "q2"), source_url=first.source_url, source_domain=first.source_domain)
    payload = pack(_context(first, duplicate, other), [first, duplicate, other])
    assert len(payload["sources"]) == 1 and len(payload["evidence_cards"]) == 2
    assert {c["exact_quote"] for c in payload["evidence_cards"]} == {
        first.exact_quote,
        other.exact_quote,
    }


def test_an_oversize_quote_is_omitted_whole_never_truncated():
    long = replace(_card(1, "q1"), exact_quote="完整不可裁剪原文" * 3000)
    short = _card(2, "q2")
    payload = pack(_context(long, short), [long, short])
    assert [c["evidence_id"] for c in payload["evidence_cards"]] == [str(short.evidence_id)]
    assert len(long.exact_quote) == 24000


def test_required_gaps_are_not_silently_pruned_to_make_room_for_cards():
    card = _card(1, "q1")
    context = replace(
        _context(card),
        quality_snapshot={
            "gap_requirements": [
                {"gap_id": "critical", "criterion": "必须保留缺口" * 4000, "closure_status": "open"}
            ]
        },
    )
    with pytest.raises(ContextBudgetInsufficientError, match="REQUIRED_CONTEXT"):
        pack(context, [card])


def test_schema_valid_opening_paragraph_in_completion_section_is_rejected():
    cards = cards_for([2], two_sources=True)
    paragraph = DraftParagraph(text=QUOTES[2], evidence_ids=[str(cards[0].evidence_id)])
    draft = ReportDraft(
        title="事件研究",
        executive_summary=[paragraph],
        sections=[DraftSection(question_id="COMPLETION", title="建成", paragraphs=[paragraph])],
    )
    assert (
        _draft_rejection_reason(draft, cards, context=replace(_context(*cards), goal=GOAL))
        == "WRITER_EVENT_BOUNDARY_MISMATCH"
    )
