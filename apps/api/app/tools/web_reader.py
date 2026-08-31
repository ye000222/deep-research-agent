"""Public-HTML reader with SSRF, redirect, size, and content-type guards."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import socket
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
import trafilatura

from app.domain.research_tools import ReadPage
from app.tools.errors import ToolExecutionError

_MAX_DOWNLOAD_BYTES = 2_000_000
_MAX_CLEAN_CHARS = 30_000
_MAX_REDIRECTS = 3


class PublicWebReader:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def read(self, url: str) -> ReadPage:
        current = _normalize_url(url)
        for redirect_count in range(_MAX_REDIRECTS + 1):
            await _require_public_destination(current)
            try:
                async with self._client.stream(
                    "GET",
                    current,
                    headers={
                        "Accept": "text/html,application/xhtml+xml",
                        "User-Agent": "DeepResearchAgent/0.1 (+public-research-reader)",
                    },
                    timeout=httpx.Timeout(25.0, connect=8.0),
                    follow_redirects=False,
                ) as response:
                    if 300 <= response.status_code < 400:
                        location = response.headers.get("location")
                        if not location or redirect_count >= _MAX_REDIRECTS:
                            raise ToolExecutionError("WEBPAGE_REDIRECT_REJECTED", retryable=False)
                        current = _normalize_url(urljoin(current, location))
                        continue
                    if response.status_code >= 500:
                        raise ToolExecutionError("WEBPAGE_PROVIDER_UNAVAILABLE", retryable=True)
                    if response.status_code >= 400:
                        raise ToolExecutionError("WEBPAGE_REQUEST_REJECTED", retryable=False)
                    content_type = response.headers.get("content-type", "").lower()
                    if (
                        "text/html" not in content_type
                        and "application/xhtml+xml" not in content_type
                    ):
                        raise ToolExecutionError(
                            "WEBPAGE_CONTENT_TYPE_UNSUPPORTED", retryable=False
                        )
                    body = await _bounded_body(response)
                    encoding = response.encoding or "utf-8"
            except ToolExecutionError:
                raise
            except httpx.TimeoutException as exc:
                raise ToolExecutionError("WEBPAGE_TIMEOUT", retryable=True) from exc
            except httpx.RequestError as exc:
                raise ToolExecutionError("WEBPAGE_NETWORK_ERROR", retryable=True) from exc

            html = body.decode(encoding, errors="replace")
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_links=False,
                favor_precision=True,
            )
            if not extracted or len(extracted.strip()) < 100:
                raise ToolExecutionError("WEBPAGE_EXTRACTION_EMPTY", retryable=False)
            clean = extracted.strip()
            truncated = len(clean) > _MAX_CLEAN_CHARS
            clean = clean[:_MAX_CLEAN_CHARS]
            title_parser = _TitleParser()
            title_parser.feed(html[:100_000])
            title = title_parser.title or urlsplit(current).hostname or "Untitled source"
            return ReadPage(
                final_url=current,
                title=title[:1000],
                clean_text=clean,
                content_hash=hashlib.sha256(clean.encode("utf-8")).hexdigest(),
                fetched_at=datetime.now(UTC),
                truncated=truncated,
            )
        raise ToolExecutionError("WEBPAGE_REDIRECT_REJECTED", retryable=False)


async def _bounded_body(response: httpx.Response) -> bytes:
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > _MAX_DOWNLOAD_BYTES:
        raise ToolExecutionError("WEBPAGE_TOO_LARGE", retryable=False)
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > _MAX_DOWNLOAD_BYTES:
            raise ToolExecutionError("WEBPAGE_TOO_LARGE", retryable=False)
        chunks.append(chunk)
    return b"".join(chunks)


def _normalize_url(url: str) -> str:
    parsed = urlsplit(url.strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ToolExecutionError("WEBPAGE_URL_REJECTED", retryable=False)
    if not parsed.hostname or parsed.username or parsed.password:
        raise ToolExecutionError("WEBPAGE_URL_REJECTED", retryable=False)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ToolExecutionError("WEBPAGE_URL_REJECTED", retryable=False) from exc
    if port not in {None, 80, 443}:
        raise ToolExecutionError("WEBPAGE_PORT_REJECTED", retryable=False)
    netloc = parsed.hostname.lower()
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, ""))


async def _require_public_destination(url: str) -> None:
    host = urlsplit(url).hostname
    if host is None or host.lower() == "localhost":
        raise ToolExecutionError("WEBPAGE_PRIVATE_DESTINATION", retryable=False)
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if not literal.is_global:
            raise ToolExecutionError("WEBPAGE_PRIVATE_DESTINATION", retryable=False)
        return
    try:
        records = await asyncio.to_thread(socket.getaddrinfo, host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ToolExecutionError("WEBPAGE_DNS_FAILED", retryable=True) from exc
    addresses = {record[4][0] for record in records}
    if not addresses or any(not ipaddress.ip_address(address).is_global for address in addresses):
        raise ToolExecutionError("WEBPAGE_PRIVATE_DESTINATION", retryable=False)


class _TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._inside_title = False
        self._parts: list[str] = []

    @property
    def title(self) -> str:
        return " ".join("".join(self._parts).split())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self._inside_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._inside_title = False

    def handle_data(self, data: str) -> None:
        if self._inside_title:
            self._parts.append(data)
