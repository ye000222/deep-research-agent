from app.domain.adaptive_scheduler import QueryFamily, build_family_query, numeric_scope_consistent
from app.domain.source_policy import source_owner_key


def test_source_owner_collapses_subdomains_to_one_publisher() -> None:
    assert source_owner_key("https://docs.example.com/spec") == "example.com"
    assert source_owner_key("https://www.example.co.uk/report") == "example.co.uk"


def test_numeric_scope_requires_unit_and_geography() -> None:
    criterion = "2025年中国市场规模为10亿美元\uff0c平均增长率为20%"
    matching = "2025年中国市场规模为10亿美元\uff0c平均增长率为20%"
    wrong_unit = "2025年中国市场规模为10亿元\uff0c平均增长率为20%"
    wrong_region = "2025年美国市场规模为10亿美元\uff0c平均增长率为20%"

    assert numeric_scope_consistent(criterion=criterion, claim=matching, quote=matching)
    assert not numeric_scope_consistent(criterion=criterion, claim=wrong_unit, quote=wrong_unit)
    assert not numeric_scope_consistent(
        criterion=criterion, claim=wrong_region, quote=wrong_region
    )


def test_family_query_avoids_repeating_the_full_question() -> None:
    question = (
        "工业视觉缺陷检测中传统图像处理方法与深度学习算法各自的技术原理、"
        "适用场景与局限是什么\uff1f"
    )
    query = build_family_query(
        question=question,
        criterion="原文对传统图像处理方法的原理与适用场景的描述",
        hints=("工业视觉缺陷检测 传统图像处理 深度学习 技术路线",),
        family=QueryFamily.SCOPE,
    )
    assert question not in query
    assert len(query) < len(question) + 100
    assert "工业视觉缺陷检测" in query
    assert "原文对传统图像处理方法" in query
