from app.infrastructure.db.research_models import SearchQueryRow
from app.infrastructure.db.research_tools import (
    normalize_search_query,
    search_query_duplicate_key,
)


def test_search_query_normalization_deduplicates_case_punctuation_and_width() -> None:
    assert normalize_search_query("工业视觉\uff1a缺陷检测\uff01") == "工业视觉 缺陷检测"
    assert normalize_search_query("Industrial-Vision  DEFECT") == "industrial vision defect"
    assert normalize_search_query("\uff21\uff22\uff23  defect") == "abc defect"


def test_search_query_idempotency_key_is_plan_version_independent() -> None:
    first = search_query_duplicate_key("q1", "Industrial-Vision DEFECT")
    second = search_query_duplicate_key("q1", "industrial vision defect")

    assert first == second
    assert first != search_query_duplicate_key("q2", "industrial vision defect")


def test_search_query_database_dedupe_is_question_scoped() -> None:
    index = next(
        item
        for item in SearchQueryRow.__table__.indexes
        if item.name == "uq_research_search_query_question_hash"
    )
    assert [column.name for column in index.columns] == [
        "run_id",
        "question_id",
        "normalized_hash",
    ]
