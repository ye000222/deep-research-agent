import socket

import httpx
import pytest
import respx
from app.tools.errors import ToolExecutionError
from app.tools.web_reader import PublicWebReader


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
async def test_reader_rejects_private_destination_before_network() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(ToolExecutionError) as caught:
            await PublicWebReader(client).read("http://127.0.0.1/admin")

    assert caught.value.code == "WEBPAGE_PRIVATE_DESTINATION"
