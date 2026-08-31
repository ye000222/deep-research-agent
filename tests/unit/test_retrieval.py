from app.retrieval.normalization import normalize_text, reciprocal_rank_fusion


def test_normalize_text_preserves_latin_cjk_and_bigrams():
    value = normalize_text("工业视觉 AI 2026")
    assert "ai" in value.latin_text
    assert "2026" in value.latin_text
    assert "工 业" in value.cjk_lexemes
    assert "工业" in value.cjk_lexemes
    assert value.fuzzy_text == "工业视觉 ai 2026"


def test_reciprocal_rank_fusion_is_deterministic():
    scores = reciprocal_rank_fusion(
        ("a", "b", "c"),
        ("b", "a", "d"),
    )
    assert scores["a"] == scores["b"]
    assert scores["c"] == scores["d"]
    assert reciprocal_rank_fusion(("a", "b")) == reciprocal_rank_fusion(("a", "b"))
