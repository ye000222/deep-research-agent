import socket

import httpx
import pytest
import respx
from app.tools.errors import ToolExecutionError
from app.tools.web_reader import PublicWebReader, _public_fallback_url


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
        "app.tools.web_reader._extract_pdf_text",
        lambda body: "Industrial defect detection benchmark evidence. " * 5,
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
