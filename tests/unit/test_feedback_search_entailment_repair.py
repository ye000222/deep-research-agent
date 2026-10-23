"""Quote-local facts and deterministic closure query anchors; no service calls."""
# ruff: noqa: RUF001 -- preserve natural Chinese fixtures

from dataclasses import replace

import pytest
from app.domain.adaptive_scheduler import claim_quote_comparison, claim_quote_entails
from app.domain.feedback_query import bind_feedback_query
from app.domain.query_candidate import QueryCandidateGenerator
from app.domain.query_plan_validation import QueryPlanValidator
from app.domain.research_context import ResearchContext

from tests.unit.test_feedback_execution import _fixture


@pytest.mark.parametrize(
    ("claim", "quote"),
    [
        ("6月", "June"),
        ("June", "6月"),
        ("6月23日", "June 23"),
        ("6月23日", "23 June"),
        ("6月23日", "June 23rd"),
        ("2020年6月23日", "June 23, 2020"),
        ("2020年6月23日", "23 June 2020"),
        ("第55颗卫星", "55th satellite"),
        ("第1", "1st"),
        ("第2", "2nd"),
        ("第3", "3rd"),
        ("第11", "11th"),
        ("9时43分", "9:43 AM"),
        ("09时43分", "09:43"),
        ("9时43分", "9:43AM"),
        ("9时43分", "9:43 a.m."),
        ("21时43分", "9:43 p.m."),
        ("0时43分", "12:43 AM"),
        (
            "2020年6月23日9时43分，第55颗北斗卫星成功发射。",
            "On June 23, 2020, the 55th BeiDou satellite was successfully launched at 9:43 a.m.",
        ),
    ],
)
def test_explicit_bilingual_facts_are_equivalent(claim, quote):
    assert claim_quote_entails(claim, quote), claim_quote_comparison(claim, quote)


@pytest.mark.parametrize(
    ("number", "names"),
    list(
        enumerate(
            [
                ("January", "Jan"),
                ("February", "Feb"),
                ("March", "Mar"),
                ("April", "Apr"),
                ("May",),
                ("June", "Jun"),
                ("July", "Jul"),
                ("August", "Aug"),
                ("September", "Sep", "Sept"),
                ("October", "Oct"),
                ("November", "Nov"),
                ("December", "Dec"),
            ],
            1,
        )
    ),
)
def test_month_names_and_abbreviations(number, names):
    for name in names:
        assert claim_quote_entails(f"{number}月23日", f"{name} 23")


@pytest.mark.parametrize(
    ("claim", "quote"),
    [
        ("2020年6月23日", "June 23"),
        ("2020年6月23日", "June 23, 2021"),
        ("6月23日", "July 23"),
        ("6月23日", "June 24 and July 23"),
        ("9时43分", "9:43 PM"),
        ("第55颗卫星", "55 satellites"),
        ("第55颗卫星", "54th satellite"),
        ("北斗卫星6月成功发射。", "The satellite is planned for June."),
        ("北斗卫星实际发射。", "The BeiDou satellite has a scheduled launch."),
        ("北斗三号正式开通服务。", "BeiDou completed constellation deployment."),
        (
            "The satellite was successfully launched.",
            "The satellite was not successfully launched.",
        ),
        ("2020年2月30日", "February 30, 2020"),
        ("24时43分", "24:43"),
        ("2020美元", "2020元"),
        ("Revenue was 2020元.", "Revenue was 2020美元."),
    ],
)
def test_normalization_never_adds_or_changes_facts(claim, quote):
    assert not claim_quote_entails(claim, quote)


def test_year_stays_missing_and_diagnostics_contain_no_body():
    claim = "2020年6月23日9时43分，第55颗北斗卫星成功发射。"
    quote = "At 9:43AM on June 23, the 55th BDS satellite was successfully launched."
    passed, diagnostics = claim_quote_comparison(claim, quote)
    assert not passed
    assert "year:2020" in diagnostics["unmatched_required_facts"]
    assert "ordinal:55" in diagnostics["normalized_quote_signals"]
    assert "time:09:43" in diagnostics["normalized_quote_signals"]
    assert claim not in repr(diagnostics) and quote not in repr(diagnostics)


@pytest.mark.parametrize("dimension", ["q3:d2", "q3:d3", "q4:d2"])
@pytest.mark.parametrize(
    "question",
    [
        "北斗三号正式开通服务日期是什么？",
        "最后一颗北斗三号组网卫星何时成功发射？",
        "When was the final BeiDou-3 satellite successfully launched?",
        "When was BeiDou-3 service officially opened?",
    ],
)
def test_generic_feedback_is_anchored_to_current_question(dimension, question):
    result = bind_feedback_query(
        f"{dimension} primary source official report authoritative evidence",
        current_question=question,
        original_query="Unrelated original topic must not replace the current question",
        requirement="需要官方来源",
    )
    assert result.allowed
    assert question in result.query
    assert dimension not in result.query
    assert "Unrelated" not in result.query
    assert bind_feedback_query(result.query, current_question=question).query == result.query


def test_generic_only_feedback_without_identity_is_not_dispatchable():
    result = bind_feedback_query(
        "primary source official report authoritative evidence",
        current_question="需要官方来源",
        original_query="verify reliable evidence",
    )
    assert not result.allowed


def test_original_question_is_only_a_fallback_anchor():
    result = bind_feedback_query(
        "q1:d3 official source",
        current_question="需要证据",
        original_query="北斗三号正式开通服务日期是什么？请提供证据。",
    )
    assert result.allowed and "北斗三号" in result.query and "开通" in result.query


def test_event_without_subject_uses_original_topic():
    result = bind_feedback_query(
        "official evidence",
        current_question="何时正式开通？",
        original_query="北斗三号何时正式开通服务？",
    )
    assert result.allowed and "北斗三号" in result.query


def test_context_survives_plan_candidate_pipeline():
    requirement, plan, _ = _fixture()
    context = ResearchContext(
        current_question="最后一颗北斗三号组网卫星何时成功发射？",
        original_query="研究北斗时间线",
        missing_requirement="需要独立官方来源",
    )
    plan = replace(plan, research_context=context)
    validation = QueryPlanValidator.validate(plan, requirement=requirement)[0]
    candidate = QueryCandidateGenerator.generate(
        plan, requirement=requirement, validation=validation
    )[0]
    assert "北斗三号" in candidate.query_text and "发射" in candidate.query_text
    assert ResearchContext.from_mapping(context.as_dict()) == context


def test_low_relevance_ambiguous_csgpc_quote_remains_rejected():
    import hashlib
    from datetime import UTC, datetime

    from app.domain.document_evidence_context import (
        LocalizationStatus,
        MappedTextWindow,
        build_readable_document,
    )
    from app.domain.research_tools import EvidenceBatch, EvidenceCandidate, ReadPage
    from app.services.evidence_extractor import EvidenceExtractorService, _attach_canonical_spans

    quote = (
        "7月31日，中共中央总书记、国家主席、中央军委主席习近平向全世界庄严宣布"
        "“北斗三号全球卫星导航系统正式开通”。历时26年，历经北斗一号、二号、三号的艰难建设过程，"
        "我国独立自主运行控制的国家时空基础设施正式建成，并投入运行。"
    )
    text = quote + "\n\n" + quote
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    page = ReadPage(
        final_url="https://www.csgpc.org/detail/6549.html",
        title="北斗系统开通",
        clean_text=text,
        content_hash=content_hash,
        fetched_at=datetime.now(UTC),
    )
    batch = EvidenceBatch(
        items=[
            EvidenceCandidate(
                claim="网页宣布北斗三号全球卫星导航系统正式开通，并表述为正式建成，并投入运行。",
                exact_quote=quote,
                relation="context",
                relevance=0.55,
                confidence=0.8,
            )
        ]
    )
    scored = EvidenceExtractorService._score_batch(
        batch, page=page, dimension_criteria={}, reliability=0.7805
    )
    assert not scored[0].accepted
    assert scored[0].rejection_reason == "evidence_relevance_below_threshold"
    document = build_readable_document(
        source_ref="csgpc-fixture",
        snapshot_content_hash=content_hash,
        canonical_text=text,
        available_text_length=len(text),
        document_type="HTML",
        was_truncated=False,
    )
    located, _ = _attach_canonical_spans(
        scored,
        document=document,
        windows=(
            MappedTextWindow(
                text=text,
                canonical_start=0,
                canonical_end=len(text),
                chunk_ref="a" * 64,
                selection_reason="fixture",
            ),
        ),
    )
    assert not located[0].accepted
    assert located[0].localization_status is LocalizationStatus.AMBIGUOUS
    assert located[0].canonical_span is None
