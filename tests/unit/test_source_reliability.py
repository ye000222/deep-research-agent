from app.services.evidence_extractor import source_reliability


def test_source_reliability_distinguishes_primary_sources_from_mirrors() -> None:
    assert source_reliability("https://www.cognex.com/products/vision") >= 0.85
    assert source_reliability("https://www.cognex.cn/zh-cn/products") >= 0.85
    assert source_reliability("https://example.com/article") == 0.68
    assert source_reliability("https://example.org/article") == 0.72
    assert source_reliability("https://blog.csdn.net/example") < 0.6


def test_source_reliability_keeps_academic_and_government_sources_strong() -> None:
    assert source_reliability("https://example.edu/paper") == 0.9
    assert source_reliability("https://data.example.gov/report") == 0.92
    assert source_reliability("https://wap.cnki.net/article/123") == 0.9
    assert source_reliability("https://cdmd.cnki.com.cn/article/123") == 0.9
    assert source_reliability("https://www.aas.net.cn/article/123") == 0.9
    assert source_reliability("https://www.askci.com/report/123") == 0.8


def test_industrial_publishers_are_not_misclassified_as_unknown_com() -> None:
    assert source_reliability("https://www.deepvai.com/solutions/defect") >= 0.85
    assert source_reliability("https://www.shuangyi-tech.com/article/defect") >= 0.85
    assert source_reliability("https://www.leadingir.com/report/vision") == 0.8
    assert source_reliability("https://www.marketresearchfuture.com/reports/defect") == 0.8
