"""Public-HTML reader with SSRF, redirect, size, and content-type guards."""

from __future__ import annotations

import asyncio
import hashlib
import html as html_module
import io
import ipaddress
import re
import socket
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
import trafilatura
from pypdf import PdfReader

from app.domain.research_tools import ReadPage
from app.tools.errors import ToolExecutionError

_MAX_DOWNLOAD_BYTES = 2_000_000
_MAX_PDF_DOWNLOAD_BYTES = 12_000_000
_MAX_PDF_PAGES = 80
_MAX_CLEAN_CHARS = 30_000
_MAX_REDIRECTS = 3
_HTML_PARSE_TIMEOUT_SECONDS = 20.0


class PublicWebReader:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client
        self.events: list[dict[str, object]] = []
        self._events_by_url: dict[str, list[dict[str, object]]] = {}

    def consume_events(self, requested_url: str) -> list[dict[str, object]]:
        """Return and remove the lifecycle events for one read attempt."""
        return self._events_by_url.pop(requested_url, [])

    def _finish_events(
        self,
        requested_url: str,
        events: list[dict[str, object]],
    ) -> None:
        self.events = list(events)
        self._events_by_url[requested_url] = list(events)

    @staticmethod
    def _event(
        event_type: str,
        *,
        requested_url: str,
        current_url: str,
        stage: str,
        metrics: dict[str, object] | None = None,
        reason: str | None = None,
    ) -> dict[str, object]:
        parsed = urlsplit(current_url)
        event: dict[str, object] = {
            "event_type": event_type,
            "candidate_id": None,
            "source_id": None,
            "requested_url": requested_url[:1000],
            "final_url": current_url[:1000],
            "normalized_url": current_url[:1000],
            "domain": (parsed.hostname or "unknown")[:255],
            "stage": stage,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        if reason is not None:
            event["reason"] = reason
            event["failure_reason"] = reason
        if metrics:
            event["metrics"] = dict(metrics)
            event.update(metrics)
        return event

    @classmethod
    def _failure_event(
        cls,
        *,
        requested_url: str,
        current_url: str,
        reason: str,
        metrics: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return cls._event(
            "source.fetch_failed",
            requested_url=requested_url,
            current_url=current_url,
            stage="fetch",
            reason=reason,
            metrics=metrics,
        )

    async def read(self, url: str) -> ReadPage:
        requested_url = url
        events: list[dict[str, object]] = [
            self._event(
                "source.fetch_started",
                requested_url=requested_url,
                current_url=requested_url,
                stage="fetch",
            )
        ]
        try:
            current = _normalize_url(url)
        except ToolExecutionError as exc:
            events.append(
                self._failure_event(
                    requested_url=requested_url,
                    current_url=requested_url,
                    reason="http_error",
                    metrics={"error_code": exc.code},
                )
            )
            self._finish_events(requested_url, events)
            raise
        events[0].update(
            {
                "final_url": current[:1000],
                "normalized_url": current[:1000],
                "domain": (urlsplit(current).hostname or "unknown")[:255],
            }
        )

        for redirect_count in range(_MAX_REDIRECTS + 1):
            try:
                await _require_public_destination(current)
                async with self._client.stream(
                    "GET",
                    current,
                    headers={
                        "Accept": "text/html,application/xhtml+xml,application/pdf",
                        "User-Agent": "DeepResearchAgent/0.1 (+public-research-reader)",
                    },
                    timeout=httpx.Timeout(25.0, connect=8.0),
                    follow_redirects=False,
                ) as response:
                    if 300 <= response.status_code < 400:
                        location = response.headers.get("location")
                        if not location or redirect_count >= _MAX_REDIRECTS:
                            raise ToolExecutionError(
                                "WEBPAGE_REDIRECT_REJECTED",
                                retryable=False,
                                details={"http_status": response.status_code},
                            )
                        current = _normalize_url(urljoin(current, location))
                        continue
                    if response.status_code >= 500:
                        raise ToolExecutionError(
                            "WEBPAGE_PROVIDER_UNAVAILABLE",
                            retryable=True,
                            details={"http_status": response.status_code},
                        )
                    if response.status_code >= 400:
                        fallback_url = _public_fallback_url(current)
                        if fallback_url is not None and redirect_count < _MAX_REDIRECTS:
                            # A number of scholarly publishers reject their
                            # abstract HTML while exposing the same public
                            # document through a stable PDF/download endpoint.
                            # Treat that endpoint like a normal redirect so all
                            # SSRF, size, content-type, and extraction guards
                            # remain in force.
                            current = fallback_url
                            continue
                        raise ToolExecutionError(
                            "WEBPAGE_REQUEST_REJECTED",
                            retryable=False,
                            details={"http_status": response.status_code},
                        )
                    content_type = response.headers.get("content-type", "").lower()
                    is_pdf = "application/pdf" in content_type
                    if not is_pdf and (
                        "text/html" not in content_type
                        and "application/xhtml+xml" not in content_type
                    ):
                        raise ToolExecutionError(
                            "WEBPAGE_CONTENT_TYPE_UNSUPPORTED", retryable=False
                        )
                    body = await _bounded_body(
                        response,
                        max_bytes=(
                            _MAX_PDF_DOWNLOAD_BYTES if is_pdf else _MAX_DOWNLOAD_BYTES
                        ),
                    )
                    encoding = response.encoding or "utf-8"
                    events.append(
                        self._event(
                            "source.fetch_success",
                            requested_url=requested_url,
                            current_url=current,
                            stage="fetch",
                            metrics={
                                "redirect_count": redirect_count,
                                "content_type": content_type,
                                "response_bytes": len(body),
                            },
                        )
                    )
            except ToolExecutionError as exc:
                details = dict(exc.details)
                reason = "http_error" if "http_status" in details else exc.code.lower()
                events.append(
                    self._failure_event(
                        requested_url=requested_url,
                        current_url=current,
                        reason=reason,
                        metrics={
                            "redirect_count": redirect_count,
                            "error_code": exc.code,
                            **details,
                        },
                    )
                )
                self._finish_events(requested_url, events)
                raise
            except httpx.TimeoutException as exc:
                events.append(
                    self._failure_event(
                        requested_url=requested_url,
                        current_url=current,
                        reason="timeout",
                        metrics={"redirect_count": redirect_count},
                    )
                )
                self._finish_events(requested_url, events)
                raise ToolExecutionError("WEBPAGE_TIMEOUT", retryable=True) from exc
            except httpx.RequestError as exc:
                events.append(
                    self._failure_event(
                        requested_url=requested_url,
                        current_url=current,
                        reason="network_error",
                        metrics={"redirect_count": redirect_count},
                    )
                )
                self._finish_events(requested_url, events)
                raise ToolExecutionError("WEBPAGE_NETWORK_ERROR", retryable=True) from exc

            html = ""
            extracted: str | None
            events.append(
                self._event(
                    "source.parse_started",
                    requested_url=requested_url,
                    current_url=current,
                    stage="parse",
                    metrics={"content_type": content_type},
                )
            )
            try:
                if is_pdf:
                    extracted = await asyncio.to_thread(_extract_pdf_text, body)
                else:
                    html = body.decode(encoding, errors="replace")
                    metadata_parser = _MetadataParser()
                    metadata_parser.feed(html[:200_000])
                    try:
                        # Trafilatura is synchronous and can spend minutes on a
                        # malformed or script-heavy publisher page.  Keep it off
                        # the event loop so lease heartbeats and sibling reads
                        # continue, and bound the damage to this one page.
                        extracted = await asyncio.wait_for(
                            asyncio.to_thread(_extract_html_text, html),
                            timeout=_HTML_PARSE_TIMEOUT_SECONDS,
                        )
                    except TimeoutError as exc:
                        events.append(
                            self._event(
                                "source.parse_failed",
                                requested_url=requested_url,
                                current_url=current,
                                stage="parse",
                                reason="parser_error",
                                metrics={"error_code": "WEBPAGE_EXTRACTION_TIMEOUT"},
                            )
                        )
                        self._finish_events(requested_url, events)
                        raise ToolExecutionError(
                            "WEBPAGE_EXTRACTION_TIMEOUT", retryable=False
                        ) from exc
                    extracted = _merge_extracted_text(extracted, metadata_parser.metadata_text)
            except ToolExecutionError:
                raise
            except Exception:
                events.append(
                    self._event(
                        "source.parse_failed",
                        requested_url=requested_url,
                        current_url=current,
                        stage="parse",
                        reason="parser_error",
                    )
                )
                self._finish_events(requested_url, events)
                raise

            clean_extracted = extracted.strip() if extracted else ""
            if not clean_extracted and not is_pdf and _looks_like_dynamic_page(html):
                events.append(
                    self._event(
                        "source.parse_failed",
                        requested_url=requested_url,
                        current_url=current,
                        stage="parse",
                        reason="dynamic_page",
                    )
                )
                self._finish_events(requested_url, events)
                raise ToolExecutionError("WEBPAGE_EXTRACTION_EMPTY", retryable=False)
            events.append(
                self._event(
                    "source.parse_success",
                    requested_url=requested_url,
                    current_url=current,
                    stage="parse",
                )
            )
            events.append(
                self._event(
                    "source.content_extracted",
                    requested_url=requested_url,
                    current_url=current,
                    stage="content",
                    metrics={
                        "extracted": bool(clean_extracted),
                        "clean_chars": len(clean_extracted),
                    },
                )
            )
            if not clean_extracted:
                events.append(
                    self._event(
                        "source.content_quality_rejected",
                        requested_url=requested_url,
                        current_url=current,
                        stage="content",
                        reason="empty_content",
                    )
                )
                self._finish_events(requested_url, events)
                raise ToolExecutionError("WEBPAGE_EXTRACTION_EMPTY", retryable=False)
            if len(clean_extracted) < 100:
                events.append(
                    self._event(
                        "source.content_quality_rejected",
                        requested_url=requested_url,
                        current_url=current,
                        stage="content",
                        reason="body_too_short",
                        metrics={"clean_chars": len(clean_extracted)},
                    )
                )
                self._finish_events(requested_url, events)
                raise ToolExecutionError("WEBPAGE_EXTRACTION_EMPTY", retryable=False)
            events.append(
                self._event(
                    "source.content_quality_passed",
                    requested_url=requested_url,
                    current_url=current,
                    stage="content",
                    metrics={"clean_chars": len(clean_extracted)},
                )
            )
            clean = extracted.strip()
            truncated = len(clean) > _MAX_CLEAN_CHARS
            clean = clean[:_MAX_CLEAN_CHARS]
            title_parser = _MetadataParser()
            if html:
                title_parser.feed(html[:200_000])
            title = (
                title_parser.title
                or urlsplit(current).path.rsplit("/", 1)[-1]
                or urlsplit(current).hostname
                or "Untitled source"
            )
            events.append(
                self._event(
                    "source.readable",
                    requested_url=requested_url,
                    current_url=current,
                    stage="reader",
                    metrics={"clean_chars": len(clean), "truncated": truncated},
                )
            )
            self._finish_events(requested_url, events)
            return ReadPage(
                final_url=current,
                title=title[:1000],
                clean_text=clean,
                content_hash=hashlib.sha256(clean.encode("utf-8")).hexdigest(),
                fetched_at=datetime.now(UTC),
                published_at=_extract_published_at(html),
                truncated=truncated,
            )
        self._finish_events(requested_url, events)
        raise ToolExecutionError("WEBPAGE_REDIRECT_REJECTED", retryable=False)


def _public_fallback_url(url: str) -> str | None:
    """Return a deterministic public document endpoint for common publishers.

    These are URL-shape transformations only; no credentials, private hosts,
    or arbitrary proxy endpoints are introduced. Returning ``None`` keeps the
    original rejection semantics for all unknown sites.
    """

    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path
    if host in {"www.ssrn.com", "ssrn.com"}:
        match = re.search(r"abstract(?:id)?[=/](\d+)", f"{path}?{parsed.query}")
        if match:
            return f"https://papers.ssrn.com/sol3/Delivery.cfm?abstractid={match.group(1)}"
    if host == "www.mdpi.com" and not path.endswith("/pdf"):
        return urlunsplit(
            (parsed.scheme, parsed.netloc, path.rstrip("/") + "/pdf", parsed.query, "")
        )
    if host in {"ieeexplore.ieee.org", "www.ieeexplore.ieee.org"}:
        match = re.search(r"/document/(\d+)", path)
        if match:
            return f"https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber={match.group(1)}"
    if host == "dl.acm.org" and "/doi/abs/" in path:
        return urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                path.replace("/doi/abs/", "/doi/pdf/"),
                parsed.query,
                "",
            )
        )
    if host == "academic.oup.com" and "/article/" in path and not parsed.query:
        return urlunsplit((parsed.scheme, parsed.netloc, path, "download=1", ""))
    if host in {
        "www.spiedigitallibrary.org",
        "proceedings.spiedigitallibrary.org",
    } and path.endswith(".aspx"):
        return urlunsplit((parsed.scheme, parsed.netloc, path[:-5] + ".full", parsed.query, ""))
    return None


_PUBLICATION_META_RE = re.compile(
    r"<meta[^>]+(?:property|name)=[\"'](?:article:published_time|date|pubdate|datePublished)"
    r"[\"'][^>]+content=[\"']([^\"']+)[\"']|"
    r"<meta[^>]+content=[\"']([^\"']+)[\"'][^>]+(?:property|name)=[\"']"
    r"(?:article:published_time|date|pubdate|datePublished)[\"']",
    re.IGNORECASE,
)


def _extract_published_at(html: str) -> datetime | None:
    """Extract a conservative publication timestamp from common page metadata."""

    for match in _PUBLICATION_META_RE.finditer(html[:200_000]):
        raw = next((value for value in match.groups() if value), "").strip()
        if not raw:
            continue
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    return None


def _extract_pdf_text(body: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(body), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            return ""
        parts = [page.extract_text() or "" for page in reader.pages[:_MAX_PDF_PAGES]]
    except Exception:
        return ""
    return "\n\n".join(part.strip() for part in parts if part.strip())


def _extract_html_text(html: str) -> str | None:
    return trafilatura.extract(
        html,
        include_comments=False,
        include_links=False,
        favor_precision=True,
    )


def _looks_like_dynamic_page(html: str) -> bool:
    """Identify an empty script shell without changing extraction behavior."""
    return bool(re.search(r"<script\b", html, flags=re.IGNORECASE))


async def _bounded_body(response: httpx.Response, *, max_bytes: int) -> bytes:
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > max_bytes:
        raise ToolExecutionError("WEBPAGE_TOO_LARGE", retryable=False)
    chunks: list[bytes] = []
    size = 0
    async for chunk in response.aiter_bytes():
        size += len(chunk)
        if size > max_bytes:
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


_CONTENT_META_KEYS = frozenset(
    {
        "citation_abstract",
        "citation_title",
        "dc.description",
        "description",
        "keywords",
        "og:description",
        "og:title",
        "twitter:description",
        "twitter:title",
    }
)


def _clean_metadata_value(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", html_module.unescape(value))
    return " ".join(without_tags.split())


def _merge_extracted_text(extracted: str | None, metadata_text: str) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for value in (extracted or "", metadata_text):
        clean = value.strip()
        identity = clean.casefold()
        if clean and identity not in seen:
            parts.append(clean)
            seen.add(identity)
    return "\n\n".join(parts)


class _MetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._inside_title = False
        self._parts: list[str] = []
        self._metadata: list[tuple[str, str]] = []

    @property
    def title(self) -> str:
        title = " ".join("".join(self._parts).split())
        if title:
            return title
        for key, value in self._metadata:
            if key in {"citation_title", "og:title", "twitter:title"}:
                return value
        return ""

    @property
    def metadata_text(self) -> str:
        values: list[str] = []
        seen: set[str] = set()
        for _key, value in self._metadata:
            identity = value.casefold()
            if identity not in seen:
                values.append(value)
                seen.add(identity)
        return "\n\n".join(values)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self._inside_title = True
            return
        if tag.lower() != "meta":
            return
        values = {key.casefold(): value for key, value in attrs if value is not None}
        key = (
            values.get("name")
            or values.get("property")
            or values.get("itemprop")
            or ""
        ).casefold()
        content = _clean_metadata_value(values.get("content") or "")
        if key in _CONTENT_META_KEYS and content:
            self._metadata.append((key, content))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._inside_title = False

    def handle_data(self, data: str) -> None:
        if self._inside_title:
            self._parts.append(data)
