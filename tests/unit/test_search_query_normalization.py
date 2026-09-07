from app.infrastructure.db.research_tools import normalize_search_query


def test_search_query_normalization_deduplicates_case_punctuation_and_width() -> None:
    assert normalize_search_query("工业视觉\uff1a缺陷检测\uff01") == "工业视觉 缺陷检测"
    assert normalize_search_query("Industrial-Vision  DEFECT") == "industrial vision defect"
    assert normalize_search_query("\uff21\uff22\uff23  defect") == "abc defect"
