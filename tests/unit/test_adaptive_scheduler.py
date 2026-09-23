from app.domain.adaptive_scheduler import (
    QueryFamily,
    build_family_query,
    cheap_triage,
    claim_quote_entails,
    classify_source_role,
    infer_claim_type,
    numeric_scope_consistent,
    source_role_fits_claim,
)
from app.domain.source_policy import source_owner_key


def test_source_owner_collapses_subdomains_to_one_publisher() -> None:
    assert source_owner_key("https://docs.example.com/spec") == "example.com"
    assert source_owner_key("https://www.example.co.uk/report") == "example.co.uk"


def test_vendor_product_claim_requires_a_manufacturer_source() -> None:
    claim_type = infer_claim_type("原文列出厂商名称与其缺陷检测产品名称")

    assert claim_type == "vendor_product"
    assert source_role_fits_claim(claim_type=claim_type, source_role="manufacturer")
    assert not source_role_fits_claim(claim_type=claim_type, source_role="publisher")


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


def test_non_numeric_small_sample_criterion_is_not_treated_as_numeric_scope() -> None:
    assert numeric_scope_consistent(
        criterion="原文给出的数据增强、合成或小样本方法",
        claim="该方法从有限支持特征中学习正常概念。",
        quote="The method learns normal concepts from limited support features.",
    )


def test_alphanumeric_technology_tokens_are_not_numeric_scope_requirements() -> None:
    criterion = "原文列出深度学习、3D视觉等具体技术路线"
    claim = "该方案采用深度学习与3D视觉进行缺陷检测。"

    assert numeric_scope_consistent(criterion=criterion, claim=claim, quote=claim)
    assert infer_claim_type("该方案采用3D视觉进行缺陷检测") == "factual"


def test_institutional_repository_pdf_is_classified_as_academic() -> None:
    assert (
        classify_source_role(
            "https://vbn.aau.dk/ws/portalfiles/portal/123/paper.pdf",
            text="Abstract. A peer-reviewed study. Keywords: inspection. DOI 10.1/example",
        )
        == "academic"
    )


def test_market_report_on_publisher_domain_is_research_source() -> None:
    assert (
        classify_source_role(
            "https://www.example.com/market/vision",
            text="Market research report: global market size and CAGR forecast.",
        )
        == "independent_research"
    )


def test_claim_quote_entailment_handles_translation_but_keeps_numbers_strict() -> None:
    quote = "The dataset contains 1,000 non-defective images and 87 defective images."

    assert claim_quote_entails("数据集包含1000张无缺陷图像和87张缺陷图像。", quote)
    assert not claim_quote_entails("数据集包含1000张无缺陷图像和99张缺陷图像。", quote)


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
    assert "原文" not in query
    assert query == "工业视觉缺陷检测 传统图像处理 深度学习 技术路线"


def test_family_query_can_select_english_hint_for_provider_routing() -> None:
    query = build_family_query(
        question="工业视觉缺陷检测有哪些方法?",
        criterion="原文描述代表方法的表述",
        hints=(
            "工业视觉缺陷检测 方法",
            "industrial visual defect inspection methods",
        ),
        family=QueryFamily.SCOPE,
        prefer_alternate_hint=True,
    )

    assert query == "industrial visual defect inspection methods"
    assert not any("\u4e00" <= char <= "\u9fff" for char in query)


def test_alternate_family_keeps_english_hint_language_consistent() -> None:
    query = build_family_query(
        question="工业视觉缺陷检测有哪些方法?",
        criterion="代表方法",
        hints=(
            "工业视觉缺陷检测 方法",
            "industrial visual defect inspection methods",
        ),
        family=QueryFamily.ALTERNATE,
    )

    assert query.startswith("industrial visual defect inspection methods")
    assert "synonyms alternate terminology international" in query
    assert not any("\u4e00" <= char <= "\u9fff" for char in query)


def test_family_query_compacts_replanned_audit_hint() -> None:
    query = build_family_query(
        question="工业视觉缺陷检测有哪些方法?",
        criterion="原文描述代表方法的表述",
        hints=(
            "工业视觉缺陷检测有哪些方法\uff1f 原文列出代表方法\uff1b"
            "official report benchmark independent source",
        ),
        family=QueryFamily.SCOPE,
    )

    assert query == "工业视觉缺陷检测有哪些方法"
    assert "official" not in query


def test_family_query_preserves_rotated_replan_search_angle() -> None:
    query = build_family_query(
        question="2024—2026年全球工业视觉缺陷检测市场规模、增长率和厂商是什么?",
        criterion="至少两个独立来源给出市场规模",
        hints=(
            "全球工业视觉缺陷检测市场规模 independent market report methodology",
        ),
        family=QueryFamily.SCOPE,
    )

    assert "market report methodology" in query


def test_cheap_triage_uses_bilingual_query_anchor_for_english_source() -> None:
    text = (
        "Industrial defect detection uses few-shot learning and synthetic data "
        "augmentation for manufacturing quality inspection. "
        "The study reports experiments, datasets, and comparative results. "
    ) * 4

    without_anchor = cheap_triage(
        question="工业缺陷检测的小样本与类别不均衡如何解决?",
        criteria=("原文给出数据增强或小样本学习方案",),
        text=text,
        url="https://arxiv.org/abs/2406.00501",
    )
    with_anchor = cheap_triage(
        question="工业缺陷检测的小样本与类别不均衡如何解决?",
        criteria=("原文给出数据增强或小样本学习方案",),
        query_hints=(
            "industrial defect detection few-shot learning data augmentation",
        ),
        text=text,
        url="https://arxiv.org/abs/2406.00501",
    )

    assert without_anchor.accepted is False
    assert with_anchor.accepted is True
