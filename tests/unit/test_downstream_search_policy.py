"""Intent and identity regressions; no I/O."""
# ruff: noqa: RUF001

import pytest
from app.domain.adaptive_scheduler import QueryFamily, build_family_query
from app.domain.planning import ResearchQuestion, build_gap_resolution_hints
from app.domain.query_expansion import intent_gap_hints, query_intent
from app.domain.source_policy import normalize_source_url
from app.tools.web_search import _ACADEMIC_ENGINES, _strategies_for_query


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("北斗三号何时正式开通服务？", "factual_date_event"),
        ("When was the Atlas bridge officially opened?", "factual_date_event"),
        ("Orion plant construction start date", "factual_date_event"),
        ("该系统目前是否投入运营？", "entity_status"),
        ("How is the term geoid defined? definition", "definition"),
        ("Explain satellite signal processing algorithms", "technical_background"),
        ("视觉检测应用案例", "application_example"),
        ("欧洲卫星导航法规正式文本", "policy_regulation"),
        ("Model accuracy performance benchmark", "performance_benchmark"),
    ],
)
def test_intent_specific_retrieval_vocabulary(question, expected):
    assert query_intent(question) == expected
    for hint in intent_gap_hints(question):
        assert hint.startswith(question)
        if expected != "application_example":
            assert not any(term in hint for term in ["catalog", "case study", "目录", "落地"])


@pytest.mark.parametrize("rotation", range(6))
@pytest.mark.parametrize("event", ["正式开通服务", "开始建设", "建成", "投入运营", "成功发射"])
def test_date_replan_does_not_change_event_or_add_business_terms(rotation, event):
    text = f"Orion 项目何时{event}？"
    question = ResearchQuestion(
        id="q1",
        question=text,
        priority=1,
        rationale="核实事实日期",
        evidence_requirements=["官方公告原文"],
        search_hints=["Orion"],
    )
    hints = build_gap_resolution_hints(question, ("缺少第二个独立来源原文",), rotation=rotation)
    assert len(hints) == 3
    assert all(text in h and "independent source" in h for h in hints)
    assert not any(
        term in " ".join(hints).lower()
        for term in ["catalog", "application example", "case study", "目录", "落地"]
    )


@pytest.mark.parametrize("family", list(QueryFamily))
def test_recovery_families_keep_event_and_temporal_anchor(family):
    query = build_family_query(
        question="Orion 系统何时正式开通服务？",
        criterion="官方公告原文",
        hints=("Orion", "primary source official report"),
        family=family,
        prefer_alternate_hint=True,
    )
    assert "Orion" in query and "正式开通" in query and "何时" in query
    assert "catalog" not in query and "论文" not in query


@pytest.mark.parametrize("question", ["Orion 开通日期", "系统何时投入运营", "法规正式文本"])
def test_date_status_policy_do_not_spend_only_http_slot_on_academic_lane(question):
    assert _strategies_for_query(question, question)[0][0].get("engines") != _ACADEMIC_ENGINES


@pytest.mark.parametrize(
    "url",
    [
        "https://agency.gov/page/",
        "https://agency.gov/page#date",
        "https://agency.gov/page?utm_source=x",
        "http://agency.gov/page",
        "https://AGENCY.GOV/page",
        "https://agency.gov:443/page",
    ],
)
def test_existing_identity_equivalence_is_preserved(url):
    assert normalize_source_url(url) == "https://agency.gov/page"


@pytest.mark.parametrize(
    "different",
    [
        "https://agency.gov/page2",
        "https://agency.gov/page?id=2",
        "https://agency.gov/Page",
        "https://agency.gov:80/page",
        "https://other.gov/page",
    ],
)
def test_distinct_real_pages_are_not_merged(different):
    assert normalize_source_url(different) != "https://agency.gov/page"
