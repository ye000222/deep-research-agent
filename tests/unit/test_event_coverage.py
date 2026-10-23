from dataclasses import replace

import pytest
from app.domain.event_coverage import (
    EVENTS,
    MISSING_EVENT_TEXT,
    ensure_event_plan,
    project_event_coverage,
    reconcile_coverage_map,
    temporal_support,
)
from app.domain.planning import ResearchPlan, ResearchQuestion
from app.services.report_writer import assemble_report

from tests.unit.test_report_writer import _card, _context

GOAL = "Atlas系统何时开始建设、建成、正式开通服务和投入运营? 请区分四个事件。"
QUOTES = [
    "Atlas系统于2011年1月2日开始建设。",
    "Atlas系统于2018年2月3日建成。",
    "Atlas系统于2019年3月4日正式开通服务。",
    "Atlas系统于2021年4月5日投入运营。",
]


def cards_for(indices, *, two_sources=False):
    return [
        replace(
            _card(10 * i + owner + 1, f"q{i + 1}"),
            claim=QUOTES[i],
            exact_quote=QUOTES[i],
            source_domain=f"publisher{owner}.gov",
            source_url=f"https://publisher{owner}.gov/{i}",
        )
        for i in indices
        for owner in range(2 if two_sources else 1)
    ]


@pytest.mark.parametrize("index", range(4))
def test_literal_event_cannot_cover_other_three_events(index):
    projection = project_event_coverage(GOAL, cards_for([index], two_sources=True))
    assert [t["coverage_status"] for t in projection["targets"]].count("SUPPORTED") == 1
    assert projection["targets"][index]["coverage_status"] == "SUPPORTED"
    assert projection["critical_unresolved_count"] == 3


def test_all_four_events_need_independent_sources_and_remain_traceable():
    projection = project_event_coverage(GOAL, cards_for(range(4), two_sources=True))
    assert projection["complete"] and projection["critical_unresolved_count"] == 0
    assert all(len(t["supporting_accepted_evidence_ids"]) == 2 for t in projection["targets"])
    one_source = project_event_coverage(GOAL, cards_for(range(4)))
    assert not one_source["complete"]
    assert all(t["coverage_status"] == "PARTIALLY_SUPPORTED" for t in one_source["targets"])


@pytest.mark.parametrize(
    "quote",
    [
        "Atlas系统于2019年3月4日进入全球服务新时代。",
        "Atlas系统于2019年3月4日正式开通, 不代表建成或投入运营。",
        "Atlas系统计划于2019年3月4日投入运营。",
        "Atlas系统尚未建成。",
    ],
)
def test_era_metaphor_negative_and_planned_events_do_not_close_operation(quote):
    card = replace(_card(1, "q1"), claim=quote, exact_quote=quote)
    projection = project_event_coverage(GOAL, [card], minimum_sources=1)
    assert projection["targets"][-1]["coverage_status"] == "UNRESOLVED"


@pytest.mark.parametrize(
    "quote,complete",
    [
        ("Atlas系统6月23日正式开通服务。", False),
        ("Atlas系统2020年6月23日正式开通服务。", True),
        ("Atlas系统于2020年建成。", True),
        ("Atlas system opened on June 23.", False),
        ("In 2020, Atlas system opened on June 23.", True),
    ],
)
def test_year_must_be_in_supported_text(quote, complete):
    assert temporal_support(quote) is complete


def test_yearless_event_remains_partial_even_with_two_publishers():
    cards = [
        replace(
            c, claim="Atlas系统6月23日正式开通服务。", exact_quote="Atlas系统6月23日正式开通服务。"
        )
        for c in cards_for([2], two_sources=True)
    ]
    target = project_event_coverage(GOAL, cards)["targets"][2]
    assert target["coverage_status"] == "PARTIALLY_SUPPORTED"
    assert target["temporal_support"] == "PARTIAL"


def test_disagreeing_dates_are_conflicting_not_more_support():
    cards = cards_for([2], two_sources=True)
    cards[1] = replace(cards[1], exact_quote=QUOTES[2].replace("4日", "5日"))
    assert project_event_coverage(GOAL, cards)["targets"][2]["coverage_status"] == "CONFLICTING"


def test_same_owner_or_rejected_candidates_do_not_create_source_diversity():
    cards = cards_for([2], two_sources=True)
    cards[1] = replace(cards[1], source_url="https://publisher0.gov/reprint")
    target = project_event_coverage(GOAL, cards)["targets"][2]
    assert target["source_diversity"] == 1 and target["coverage_status"] != "SUPPORTED"
    candidate = {
        "id": "rejected",
        "accepted": False,
        "exact_quote": QUOTES[2],
        "source_url": "https://other.gov",
    }
    assert (
        project_event_coverage(GOAL, [candidate])["targets"][2]["coverage_status"] == "UNRESOLVED"
    )


def test_equivalent_date_formats_are_not_a_conflict():
    cards = cards_for([2], two_sources=True)
    cards[1] = replace(cards[1], exact_quote=QUOTES[2].replace("2019年3月4日", "2019-03-04"))
    assert project_event_coverage(GOAL, cards)["targets"][2]["coverage_status"] == "SUPPORTED"


def test_explicit_contradicting_evidence_does_not_become_support_in_report():
    cards = cards_for([2], two_sources=True)
    cards[1] = replace(cards[1], relation="contradicts")
    projection = project_event_coverage(GOAL, cards)
    assert projection["targets"][2]["coverage_status"] == "CONFLICTING"
    assert (
        str(cards[1].evidence_id)
        not in projection["targets"][2]["supporting_accepted_evidence_ids"]
    )


def test_missing_required_event_replaces_optional_elaboration_without_more_questions():
    plan = ResearchPlan(
        goal=GOAL,
        scope_summary=GOAL,
        completion_criteria=["四节点独立核实", "引用可追溯"],
        questions=[
            ResearchQuestion(
                id=f"q{i + 1}",
                question=f"Atlas系统{label}时间是什么?",
                priority=1,
                rationale="核实时间节点和来源。",
                evidence_requirements=[f"{label}时间原文"],
                search_hints=[],
            )
            for i, label in enumerate(["正式开通服务", "建成", "投入运营", "最后卫星发射"])
        ],
    )
    fixed = ensure_event_plan(plan)
    assert len(fixed.questions) == 4
    assert any("开始建设" in q.question for q in fixed.questions)
    assert not any("最后卫星" in q.question for q in fixed.questions)
    assert fixed.goal == GOAL
    assert fixed == ensure_event_plan(fixed)


def test_explicit_gap_dimensions_survive_an_old_incomplete_plan():
    dimensions = [
        {
            "dimension_key": "q1",
            "question": "Atlas系统何时正式开通服务?",
            "priority": 1,
            "requirement_statuses": [{"criterion": "原始报道", "coverage": 1.0}],
        }
    ]
    projection = project_event_coverage(GOAL, cards_for([2], two_sources=True))
    reconcile_coverage_map(dimensions, projection)
    statuses = dimensions[0]["requirement_statuses"]
    assert len([s for s in statuses if ":event:" in s.get("dimension_key", "")]) == 4
    assert any("开始建设" in reason for reason in dimensions[0]["missing_reasons"])
    assert all(GOAL in t["gap_query"] for t in projection["targets"])


def test_short_outline_keeps_execution_slots_and_tracks_missing_events_as_gaps():
    plan = ResearchPlan(
        goal=GOAL,
        scope_summary=GOAL,
        completion_criteria=["核实节点", "引用可追溯"],
        questions=[
            ResearchQuestion(
                id="q1",
                question="Atlas系统何时正式开通服务?",
                priority=1,
                rationale="首先核实核心日期。",
                evidence_requirements=["官方开通公告"],
                search_hints=[],
            )
        ],
    )
    assert ensure_event_plan(plan) == plan
    assert len(project_event_coverage(plan.goal, [])["targets"]) == 4


@pytest.mark.parametrize("all_events", [False, True])
def test_report_event_sections_expose_gaps_instead_of_borrowing_other_events(all_events):
    cards = cards_for(range(4) if all_events else [2], two_sources=True)
    context = replace(
        _context(*cards),
        goal=GOAL,
        stop_reason="quality_met",
        quality_snapshot={
            "coverage": 1.0,
            "priority_one_coverage": 1.0,
            "source_quality": 0.9,
            "cross_validation": 1.0,
            "critical_gaps": 0,
        },
    )
    report = assemble_report(context, cards, draft=None, fallback_reason="WRITER_PROVIDER_ERROR")
    assert report.verification_result["verified"]
    assert report.verification_result["research_complete"] is all_events
    sections = {s.section_key: s for s in report.sections}
    assert set(EVENTS) <= set(sections)
    if not all_events:
        for target in set(EVENTS) - {"OFFICIAL_SERVICE_OPENING"}:
            assert sections[target].markdown == MISSING_EVENT_TEXT
    assert all(c.evidence_id in {card.evidence_id for card in cards} for c in report.citations)


def test_yearless_historical_fact_is_not_output_as_complete_fact():
    card = replace(
        _card(1, "q1"),
        claim="Atlas系统6月23日正式开通服务。",
        exact_quote="Atlas系统6月23日正式开通服务。",
    )
    report = assemble_report(
        replace(_context(card), goal=GOAL), [card], draft=None, fallback_reason=None
    )
    assert "6月23日" not in report.final_markdown
    assert any("年份" in reason for reason in report.limitations)


def test_model_cannot_supply_a_year_missing_from_its_cited_quote():
    from app.services.report_writer import (
        DraftParagraph,
        DraftSection,
        ReportDraft,
        _draft_rejection_reason,
    )

    card = replace(
        _card(1, "q1"),
        claim="Atlas系统2020年6月23日正式开通服务。",
        exact_quote="Atlas系统6月23日正式开通服务。",
    )
    paragraph = DraftParagraph(text=card.claim, evidence_ids=[str(card.evidence_id)])
    draft = ReportDraft(
        title="研究结果",
        executive_summary=[paragraph],
        sections=[DraftSection(question_id="q1", title="开通", paragraphs=[paragraph])],
    )
    assert (
        _draft_rejection_reason(draft, [card], context=replace(_context(card), goal=GOAL))
        == "WRITER_PARTIAL_TEMPORAL_SUPPORT"
    )
