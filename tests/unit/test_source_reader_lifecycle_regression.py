"""Fail-first regression tests for the Source Reader lifecycle contract.

The current web reader returns a ``ReadPage`` or raises a ``ToolExecutionError``
but does not expose source lifecycle events.  These tests intentionally define
the phase 5.2 telemetry contract and are expected to fail until Reader and
Evidence lifecycles are separated.
"""

import socket
from typing import cast

import httpx
import pytest
import respx
from app.tools.errors import ToolExecutionError
from app.tools.web_reader import PublicWebReader


def _public_dns(*args: object, **kwargs: object) -> list[tuple[object, ...]]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]


def _events(reader: PublicWebReader) -> list[dict[str, object]]:
    raw = getattr(reader, "events", [])
    if not isinstance(raw, list):
        return []
    return [cast(dict[str, object], event) for event in raw if isinstance(event, dict)]


def _event_types(reader: PublicWebReader) -> list[str]:
    return [str(event.get("event_type")) for event in _events(reader)]


async def _read_capturing_error(
    reader: PublicWebReader,
    url: str,
) -> Exception | None:
    try:
        await reader.read(url)
    except Exception as exc:  # The failure event is asserted separately.
        return exc
    return None


@pytest.mark.asyncio
@respx.mock
async def test_normal_page_has_reader_lifecycle_without_evidence_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    paragraph = "Industrial inspection systems provide traceable defect evidence. " * 8
    respx.get("https://example.com/normal").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=f"<html><body><p>{paragraph}</p></body></html>",
        )
    )
    reader = PublicWebReader(httpx.AsyncClient())

    error = await _read_capturing_error(reader, "https://example.com/normal")

    assert error is None
    assert _event_types(reader) == [
        "source.fetch_started",
        "source.fetch_success",
        "source.parse_started",
        "source.parse_success",
        "source.content_extracted",
        "source.content_quality_passed",
        "source.readable",
    ]
    assert not any(event_type.startswith("evidence.") for event_type in _event_types(reader))


@pytest.mark.asyncio
@respx.mock
async def test_empty_content_is_reader_quality_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    respx.get("https://example.com/empty").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<html><body></body></html>",
        )
    )
    reader = PublicWebReader(httpx.AsyncClient())

    await _read_capturing_error(reader, "https://example.com/empty")

    assert _event_types(reader) == [
        "source.fetch_started",
        "source.fetch_success",
        "source.parse_started",
        "source.parse_success",
        "source.content_extracted",
        "source.content_quality_rejected",
    ]
    rejection = _events(reader)[-1]
    assert rejection.get("reason") == "empty_content"


@pytest.mark.asyncio
@respx.mock
async def test_short_body_is_distinct_from_empty_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    respx.get("https://example.com/short").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<html><body><p>Short page text.</p></body></html>",
        )
    )
    reader = PublicWebReader(httpx.AsyncClient())

    await _read_capturing_error(reader, "https://example.com/short")

    assert _event_types(reader)[-1] == "source.content_quality_rejected"
    assert _events(reader)[-1].get("reason") == "body_too_short"


@pytest.mark.asyncio
@respx.mock
async def test_http_failure_preserves_failure_type_and_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    respx.get("https://example.com/blocked").mock(
        return_value=httpx.Response(403, headers={"content-type": "text/html"})
    )
    reader = PublicWebReader(httpx.AsyncClient())

    error = await _read_capturing_error(reader, "https://example.com/blocked")

    assert isinstance(error, ToolExecutionError)
    failed = _events(reader)[-1]
    assert failed.get("event_type") == "source.fetch_failed"
    assert failed.get("reason") == "http_error"
    assert failed.get("http_status") == 403


@pytest.mark.asyncio
@respx.mock
async def test_parser_failure_has_parse_failed_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    monkeypatch.setattr(
        "app.tools.web_reader._extract_html_text",
        lambda html: (_ for _ in ()).throw(ValueError("parser failure")),
    )
    respx.get("https://example.com/parser-failure").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<html><body><p>Parser failure fixture.</p></body></html>",
        )
    )
    reader = PublicWebReader(httpx.AsyncClient())

    await _read_capturing_error(reader, "https://example.com/parser-failure")

    assert _event_types(reader)[-1] == "source.parse_failed"
    assert _events(reader)[-1].get("reason") == "parser_error"


@pytest.mark.asyncio
@respx.mock
async def test_script_shell_has_dynamic_page_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    respx.get("https://example.com/dynamic").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text="<html><body><script>renderApplication()</script></body></html>",
        )
    )
    reader = PublicWebReader(httpx.AsyncClient())

    await _read_capturing_error(reader, "https://example.com/dynamic")

    assert _event_types(reader)[-1] == "source.parse_failed"
    assert _events(reader)[-1].get("reason") == "dynamic_page"
