from app.context.source_selector import rank_source_windows, select_relevant_blocks


def test_tail_evidence_is_selected_without_rewriting_source() -> None:
    text = ("Navigation and advertising.\n\n" * 1000) + (
        "Structured light measures surface defects. Not suitable for transparent glass."
    )
    windows = rank_source_windows(text, ("structured light transparent glass",), limit=2)
    assert "Not suitable" in windows[0].content
    for w in windows:
        assert w.content == text[w.start : w.end]


def test_chinese_query_and_repeatability() -> None:
    text = "无关内容。" * 1500 + "\n\n工业视觉检测采用结构光; 精度不是固定值。"
    a = rank_source_windows(text, ("结构光检测精度",), limit=2)
    assert "不是固定值" in a[0].content
    assert a == rank_source_windows(text, ("结构光检测精度",), limit=2)


def test_block_selector_preserves_offsets_and_original_order() -> None:
    text = "导航与广告。\n\n高光谱相机成像正确。\n\n单目测距精度受基线影响。"
    windows = select_relevant_blocks(text, ("高光谱相机成像",), limit=8)
    assert windows
    for w in windows:
        assert w.content == text[w.start : w.end]
        assert w.char_start == w.start and w.char_end == w.end
    # Original-text order is preserved across selected blocks.
    assert [w.start for w in windows] == sorted(w.start for w in windows)


def test_block_selector_expands_neighbors_for_cross_block_answers() -> None:
    text = (
        "开篇不相关内容。\n\n"
        "结构光方案存在局限。\n\n"
        "精度受到表面反光影响。\n\n"
        "结尾另一个话题叙述若干。"
    )
    # MMR picks the strongest single block; the cross-block answer is absent.
    narrow = select_relevant_blocks(text, ("结构光精度",), limit=1, expand_neighbors=0)
    narrow_joined = "".join(w.content for w in narrow)
    assert len(narrow) <= 1
    assert "反光" not in narrow_joined

    # Expanding one neighbor pulls the cross-block answer in, without reaching
    # the unrelated topic two blocks away.
    expanded = select_relevant_blocks(text, ("结构光精度",), limit=1, expand_neighbors=1)
    joined = "".join(w.content for w in expanded)
    assert "反光" in joined
    assert "另一个话题" not in joined


def test_block_selector_attaches_table_heading() -> None:
    text = (
        "导航与广告介绍。\n\n"
        "出货量市场规模对比\n"
        "中国、美国、欧洲 2024 年出货量单位为万台。\n"
        "中国 1200\n"
        "美国 800\n"
    )
    windows = select_relevant_blocks(text, ("市场 出货量 中国",), limit=8)
    assert any("出货量" in (w.heading or "") for w in windows)
    # Every content block carries original offsets for exact-quote location.
    for w in windows:
        assert text[w.start : w.end] == w.content