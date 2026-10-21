import socket
from pathlib import Path

import httpx
import pytest
import respx
from app.tools.errors import ToolExecutionError
from app.tools.web_reader import (
    PublicWebReader,
    _extract_pdf_document,
    _html_text_segments,
    _public_fallback_url,
)


def test_public_fallback_url_uses_known_public_document_endpoints() -> None:
    assert _public_fallback_url("https://www.mdpi.com/2076-3417/16/14/7096") == (
        "https://www.mdpi.com/2076-3417/16/14/7096/pdf"
    )
    assert _public_fallback_url("https://www.ssrn.com/abstract=5938793") == (
        "https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=5938793"
    )
    assert _public_fallback_url("https://example.com/report") is None


def public_dns(*args: object, **kwargs: object) -> list[tuple[object, ...]]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]


@pytest.mark.asyncio
@respx.mock
async def test_reader_extracts_public_html_and_hashes_clean_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", public_dns)
    paragraph = "Industrial inspection systems verify product quality with traceable results. " * 5
    respx.get("https://example.com/report").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=(
                "<html><head><title>Inspection Report</title></head>"
                f"<body><p>{paragraph}</p></body></html>"
            ),
        )
    )
    async with httpx.AsyncClient() as client:
        page = await PublicWebReader(client).read("https://example.com/report#section")

    assert page.title == "Inspection Report"
    assert "traceable results" in page.clean_text
    assert len(page.content_hash) == 64
    assert page.final_url == "https://example.com/report"


@pytest.mark.asyncio
@respx.mock
async def test_reader_extracts_bounded_public_pdf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", public_dns)
    monkeypatch.setattr(
        "app.tools.web_reader._extract_pdf_document",
        lambda body: (
            "Industrial defect detection benchmark evidence. " * 5,
            (),
            False,
        ),
    )
    respx.get("https://example.com/paper.pdf").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "application/pdf"},
            content=b"%PDF-1.7 test fixture",
        )
    )

    async with httpx.AsyncClient() as client:
        page = await PublicWebReader(client).read("https://example.com/paper.pdf")

    assert page.title == "paper.pdf"
    assert "benchmark evidence" in page.clean_text
    assert page.final_url == "https://example.com/paper.pdf"
    assert page.document_type == "PDF"
    assert page.truncated is False


@pytest.mark.asyncio
async def test_reader_preserves_full_cleaned_html_over_legacy_30000_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", public_dns)
    paragraph = "Long report evidence line. " * 1_500
    extracted = paragraph.strip()
    monkeypatch.setattr("app.tools.web_reader._extract_html_text", lambda _html: extracted)
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<html><head><title>Long report</title></head><body>text</body></html>",
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        page = await PublicWebReader(client).read("https://example.com/long-report")

    assert len(page.clean_text) > 30_000
    assert page.available_char_count == len(page.clean_text)
    assert page.truncated is False
    assert page.truncation_reason is None
    assert page.document_type == "HTML"


def test_html_text_segments_preserve_only_real_block_offsets() -> None:
    text = "A first paragraph.\n\nA second paragraph."
    segments = _html_text_segments(text)

    assert [(item.canonical_start, item.canonical_end) for item in segments] == [
        (0, len("A first paragraph.")),
        (len("A first paragraph.\n\n"), len(text)),
    ]
    assert [item.locator.block_ordinal for item in segments] == [0, 1]
    assert all(item.locator.locator_type.value == "DOCUMENT_BLOCK" for item in segments)


def test_pdf_reader_keeps_page_numbers_before_text_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakePage:
        def __init__(self, text: str) -> None:
            self.text = text

        def extract_text(self) -> str:
            return self.text

    class FakePdfReader:
        def __init__(self, _stream: object, *, strict: bool) -> None:
            assert strict is False
            self.is_encrypted = False
            self.pages = [FakePage("Page one"), FakePage("Page two")]

    monkeypatch.setattr("app.tools.web_reader.PdfReader", FakePdfReader)
    text, segments, truncated = _extract_pdf_document(b"fake")

    assert text == "Page one\n\nPage two"
    assert truncated is False
    assert [(segment.locator.page_start, segment.locator.page_end) for segment in segments] == [
        (1, 1),
        (2, 2),
    ]
    assert segments[1].canonical_start == len("Page one\n\n")


def test_pdf_reader_marks_page_cap_without_inventing_missing_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePage:
        def extract_text(self) -> str:
            return "page text"

    class FakePdfReader:
        def __init__(self, _stream: object, *, strict: bool) -> None:
            assert strict is False
            self.is_encrypted = False
            self.pages = [FakePage() for _ in range(81)]

    monkeypatch.setattr("app.tools.web_reader.PdfReader", FakePdfReader)
    text, segments, truncated = _extract_pdf_document(b"fake")

    assert len(segments) == 80
    assert text.count("page text") == 80
    assert truncated is True


@pytest.mark.asyncio
@respx.mock
async def test_reader_uses_standard_metadata_when_javascript_shell_has_no_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", public_dns)
    abstract = (
        "This peer reviewed study compares traditional image processing with "
        "deep learning for industrial surface defect inspection, describes the "
        "benchmark dataset, and reports traceable evaluation results. " * 3
    )
    respx.get("https://publisher.example/paper").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=(
                "<html><head>"
                '<meta name="citation_title" content="Industrial inspection study">'
                f'<meta name="citation_abstract" content="{abstract}">'
                '<meta name="citation_keywords" content="ignored field">'
                "</head><body><script>renderApp()</script></body></html>"
            ),
        )
    )

    async with httpx.AsyncClient() as client:
        page = await PublicWebReader(client).read("https://publisher.example/paper")

    assert page.title == "Industrial inspection study"
    assert "traditional image processing" in page.clean_text
    assert len(page.clean_text) >= 300


@pytest.mark.asyncio
@respx.mock
async def test_reader_rejects_weak_generic_metadata_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", public_dns)
    monkeypatch.setattr("app.tools.web_reader._extract_html_text", lambda _html: None)
    keywords = ", ".join(f"通用关键词{i}" for i in range(30))
    description = "这是页面的通用简介.不包含可验证的主体正文." * 5
    html = (
        "<html><head><title>动态页面</title>"
        f'<meta name="description" content="{description}">'
        f'<meta name="keywords" content="{keywords}">'
        "</head><body><div id='app'></div><script>renderApp()</script></body></html>"
    )
    respx.get("https://example.com/weak-shell").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=html,
        )
    )
    reader = PublicWebReader(httpx.AsyncClient())

    with pytest.raises(ToolExecutionError) as caught:
        await reader.read("https://example.com/weak-shell")

    assert caught.value.code == "WEBPAGE_EXTRACTION_EMPTY"
    event_types = [event["event_type"] for event in reader.events]
    assert "source.content_quality_passed" not in event_types
    assert "source.readable" not in event_types
    assert reader.events[-1]["event_type"] in {
        "source.parse_failed",
        "source.content_quality_rejected",
    }
    if reader.events[-1]["event_type"] == "source.content_quality_rejected":
        assert reader.events[-1]["reason"] == "body_too_short"


@pytest.mark.asyncio
@respx.mock
async def test_reader_body_quality_is_independent_of_auxiliary_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", public_dns)
    paragraph = "这是一段真实的主体内容.包含明确的背景.过程和结果信息." * 8
    html = (
        "<html><head><title>研究报告</title>"
        '<meta name="description" content="简短摘要">'
        '<meta name="keywords" content="词条一,词条二,词条三">'
        "</head><body>"
        f"<article><p>{paragraph}</p></article>"
        "</body></html>"
    )
    respx.get("https://example.com/body-with-metadata").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=html,
        )
    )
    reader = PublicWebReader(httpx.AsyncClient())

    page = await reader.read("https://example.com/body-with-metadata")

    assert paragraph[:30] in page.clean_text
    assert page.title == "研究报告"
    assert "词条一" in page.clean_text
    assert reader.events[-1]["event_type"] == "source.readable"


@pytest.mark.asyncio
@respx.mock
async def test_reader_extracts_chinese_long_article_with_real_trafilatura(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", public_dns)
    fixture_path = Path(__file__).parents[1] / "fixtures" / "web_reader_chinese_long_article.html"
    html = fixture_path.read_text(encoding="utf-8")
    before_article, article_marker, article_suffix = html.partition("<article>")
    article_body, closing_marker, after_article = article_suffix.partition("</article>")
    assert article_marker and closing_marker
    expanded_articles = "\n".join(
        "<article><h1>第{index}组观测记录</h1>{body}</article>".format(
            index=index,
            body=article_body.replace("<p>", f"<p>第{index}组复核记录。"),
        )
        for index in range(1, 51)
    )
    html = before_article + expanded_articles + after_article
    respx.get("https://example.com/chinese-long-article").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=html,
        )
    )
    reader = PublicWebReader(httpx.AsyncClient())

    page = await reader.read("https://example.com/chinese-long-article")

    assert "本报告通过连续观测记录了样本采集与校验流程" in page.clean_text
    assert "第二阶段的独立复核确认原始记录可以逐项追溯" in page.clean_text
    assert len(page.clean_text) > 30_000
    assert "导航菜单不应进入正文" not in page.clean_text
    assert "页脚版权信息不应进入正文" not in page.clean_text
    assert page.truncated is False
    assert page.truncation_reason is None
    assert page.available_char_count == len(page.clean_text)
    assert page.source_segments
    assert page.source_segments[0].canonical_start == 0
    assert page.source_segments[-1].canonical_end == len(page.clean_text)
    assert all(
        segment.locator.locator_type.value == "DOCUMENT_BLOCK"
        for segment in page.source_segments
    )


@pytest.mark.asyncio
async def test_reader_rejects_private_destination_before_network() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(ToolExecutionError) as caught:
            await PublicWebReader(client).read("http://127.0.0.1/admin")

    assert caught.value.code == "WEBPAGE_PRIVATE_DESTINATION"


@pytest.mark.asyncio
async def test_reader_rejects_public_hostname_resolving_to_private_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DNS rebinding/SSRF protection must validate resolved addresses, not only host text."""

    def private_dns(*args: object, **kwargs: object) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.7", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", private_dns)
    async with httpx.AsyncClient() as client:
        with pytest.raises(ToolExecutionError) as caught:
            await PublicWebReader(client).read("https://public-looking.example/report")

    assert caught.value.code == "WEBPAGE_PRIVATE_DESTINATION"


@pytest.mark.asyncio
@respx.mock
async def test_reader_rechecks_redirect_destination_and_does_not_forward_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", public_dns)
    first = respx.get("https://example.com/start").mock(
        return_value=httpx.Response(302, headers={"location": "https://other.example/final"})
    )
    second = respx.get("https://other.example/final").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=(
                "<html><title>Safe</title><body>"
                + "Public content is available for independent verification. " * 8
                + "</body></html>"
            ),
        )
    )

    async with httpx.AsyncClient() as client:
        page = await PublicWebReader(client).read("https://example.com/start")

    assert page.final_url == "https://other.example/final"
    assert first.calls[0].request.headers.get("authorization") is None
    assert second.calls[0].request.headers.get("authorization") is None
