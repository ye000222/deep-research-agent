from app.domain.adaptive_scheduler import numeric_scope_consistent
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
