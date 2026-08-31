from app.domain.evidence_graph import (
    build_evidence_chunk,
    claim_fingerprint,
    derive_claim_status,
    infer_claim_relation,
    locate_quote,
    normalize_graph_text,
)


def test_infer_claim_relation_detects_same_scope_numeric_conflict() -> None:
    decision = infer_claim_relation(
        "2025年工业机器视觉市场规模为120亿美元。",
        "2025年工业机器视觉市场规模为180亿美元。",
    )

    assert decision is not None
    assert decision.relation == "contradicts"
    assert decision.reason_code == "same_scope_numeric_difference"
    assert decision.severity >= 0.4


def test_infer_claim_relation_recognizes_numeric_agreement() -> None:
    decision = infer_claim_relation(
        "2025年工业机器视觉市场规模为120亿美元。",
        "工业机器视觉市场在2025年的规模为120亿美元。",
    )

    assert decision is not None
    assert decision.relation == "supports"


def test_infer_claim_relation_detects_opposite_polarity() -> None:
    decision = infer_claim_relation(
        "多模态模型已经进入工业视觉检测应用。",
        "多模态模型尚未进入工业视觉检测应用。",
    )

    assert decision is not None
    assert decision.relation == "contradicts"
    assert decision.reason_code == "opposite_polarity"


def test_infer_claim_relation_ignores_unrelated_claims() -> None:
    assert (
        infer_claim_relation(
            "Cognex提供机器视觉硬件。",
            "纺织行业需要实时质量控制。",
        )
        is None
    )


def test_claim_status_requires_independent_corroboration() -> None:
    assert (
        derive_claim_status(
            has_accepted_evidence=True,
            has_refuting_evidence=False,
            independent_source_count=1,
        )
        == "partial"
    )
    assert (
        derive_claim_status(
            has_accepted_evidence=True,
            has_refuting_evidence=False,
            independent_source_count=2,
        )
        == "supported"
    )


def test_claim_status_surfaces_refuting_evidence() -> None:
    assert (
        derive_claim_status(
            has_accepted_evidence=True,
            has_refuting_evidence=True,
            independent_source_count=2,
        )
        == "disputed"
    )


def test_claim_fingerprint_is_stable_across_case_and_whitespace() -> None:
    first = claim_fingerprint("Cognex  provides machine vision systems.")
    second = claim_fingerprint("  cognex\nprovides machine vision systems. ")

    assert first == second


def test_locate_quote_preserves_original_offsets_with_whitespace_changes() -> None:
    source = "Header\nCognex provides\n\nmachine vision systems for factories.\nFooter"
    quote = "Cognex provides machine vision systems for factories."

    location = locate_quote(source, quote)

    assert location is not None
    start, end = location
    assert normalize_graph_text(source[start:end]) == quote


def test_build_evidence_chunk_is_bounded_and_reproducible() -> None:
    source = "A" * 2_000 + " exact evidence quote " + "B" * 2_000

    first = build_evidence_chunk(source, "exact evidence quote", context_chars=100)
    second = build_evidence_chunk(source, "exact evidence quote", context_chars=100)

    assert first is not None
    assert second == first
    assert first.char_end - first.char_start < len(source)
    assert "exact evidence quote" in first.text
    assert len(first.chunk_hash) == 64
    assert first.token_count > 0


def test_build_evidence_chunk_rejects_unverifiable_quote() -> None:
    assert build_evidence_chunk("source text", "invented quote") is None


def test_infer_claim_relation_does_not_conflict_different_market_scopes() -> None:
    decision = infer_claim_relation(
        "中国汽车制造机器视觉产品市场规模为31.1亿元。",
        "中国工业机器视觉市场规模为268.3亿元。",
    )

    assert decision is None or decision.relation != "contradicts"


def test_infer_claim_relation_does_not_compare_different_metrics() -> None:
    decision = infer_claim_relation(
        "3D相机销售额年均复合增长率为24.9%。",
        "未来三年(2026-2028)中国机器视觉行业规模将从503.0亿元增长至784.0亿元,"
        "年均复合增长率为24.8%。",
    )

    assert decision is None or decision.relation != "contradicts"


def test_infer_claim_relation_ignores_identifier_number_changes() -> None:
    decision = infer_claim_relation(
        "国家标准GB/T 46608-2025规范钢管视觉检测。",
        "国家标准GB/T 40659-2021规范机器视觉在线检测系统。",
    )

    assert decision is None or decision.relation != "contradicts"
