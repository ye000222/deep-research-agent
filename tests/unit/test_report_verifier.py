from app.evaluation.report_verifier import verify_evidence_support


def test_semantic_support_accepts_claim_aligned_line() -> None:
    result = verify_evidence_support(
        ["工业视觉平台覆盖汽车和电子制造场景。[1]"],
        {1: ("工业视觉平台覆盖汽车和电子制造场景", "平台覆盖汽车和电子制造")},
    )

    assert result["semantic_support_rate"] == 1.0
    assert result["semantic_unsupported_citations"] == []


def test_semantic_support_rejects_unrelated_citation() -> None:
    result = verify_evidence_support(
        ["市场规模将在三年内翻倍。[1]"],
        {1: ("公司提供机器视觉相机", "产品用于缺陷检测")},
    )

    assert result["semantic_support_rate"] == 0.0
    assert result["semantic_unsupported_citations"] == [1]
