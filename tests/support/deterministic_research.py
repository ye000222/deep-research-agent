"""Deterministic external-boundary adapters for the L2 integration smoke."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from app.domain.providers import (
    CanonicalModelRequest,
    CanonicalModelResult,
    TokenUsage,
    UsageAccuracy,
)
from app.domain.research_tools import SearchResult

SOURCE_A_URL = "https://cognex.com/__test__/example-research-labs-a"
SOURCE_B_URL = "https://keyence.com/__test__/example-research-labs-b"
SOURCE_A_TITLE = "Example Research Labs profile — source A"
SOURCE_B_TITLE = "Example Research Labs profile — source B"
FOUNDING_QUOTE = "Example Research Labs was founded in 2018."
HEADQUARTERS_QUOTE = "Example Research Labs is headquartered in Berlin, Germany."


class DeterministicFixtureSearchProvider:
    """Search abstraction fake with stable ordering and URL exclusions."""

    provider_name = "DeterministicFixture"
    last_request_count = 1
    last_timeout_count = 0
    last_fallback_count = 0
    last_healthy_response_count = 1
    last_unresponsive_response_count = 0
    last_productive_response_count = 1

    def __init__(self, *, empty: bool = False, calls: list[str] | None = None) -> None:
        self.empty = empty
        self.calls = calls if calls is not None else []

    async def search(
        self,
        query: str,
        *,
        limit: int = 8,
        max_provider_requests: int | None = None,
        alternate_query: str | None = None,
        exclude_domains: tuple[str, ...] = (),
        exclude_urls: tuple[str, ...] = (),
        single_provider_only: bool = False,
    ) -> list[SearchResult]:
        del max_provider_requests, alternate_query, single_provider_only
        self.calls.append(query)
        if self.empty:
            return []
        excluded_domains = {domain.casefold() for domain in exclude_domains}
        excluded_urls_folded = {url.casefold() for url in exclude_urls}
        fixed = [
            SearchResult(
                title=SOURCE_A_TITLE,
                url=SOURCE_A_URL,
                snippet="Independent profile with founding and headquarters facts.",
                rank=1,
            ),
            SearchResult(
                title=SOURCE_B_TITLE,
                url=SOURCE_B_URL,
                snippet="Independent corroborating profile with company facts.",
                rank=2,
            ),
        ]
        return [
            result
            for result in fixed
            if result.url.casefold() not in excluded_urls_folded
            and result.url.split("/", 3)[2].casefold() not in excluded_domains
        ][:limit]

    async def bing_only_search(
        self,
        query: str,
        *,
        limit: int = 8,
        exclude_domains: tuple[str, ...] = (),
        exclude_urls: tuple[str, ...] = (),
    ) -> list[SearchResult]:
        return await self.search(
            query,
            limit=limit,
            exclude_domains=exclude_domains,
            exclude_urls=exclude_urls,
        )


class DeterministicFixtureLLMGateway:
    """Returns schema-valid fixtures at the model boundary, not app services."""

    def __init__(
        self,
        *_: object,
        empty_evidence: bool = False,
        calls: list[str] | None = None,
        **__: object,
    ) -> None:
        self.empty_evidence = empty_evidence
        self.calls = calls if calls is not None else []

    async def generate_structured(
        self,
        *,
        request: CanonicalModelRequest,
        **_: object,
    ) -> CanonicalModelResult:
        self.calls.append(request.task_kind)
        if request.task_kind in {"research_planning", "research_planning_compact"}:
            parsed = _plan_payload()
        elif request.task_kind in {"evidence_extraction", "evidence_extraction_compact"}:
            parsed = _evidence_payload(request, empty=self.empty_evidence)
        elif request.task_kind == "report_writing":
            parsed = _report_payload(request)
        else:
            raise AssertionError(f"unexpected model task in deterministic L2: {request.task_kind}")
        return CanonicalModelResult(
            parsed_object=parsed,
            usage=TokenUsage(
                input_tokens=12,
                output_tokens=8,
                total_tokens=20,
                accuracy=UsageAccuracy.EXACT,
            ),
            capability_strategy={"structured_output": "test_fixture"},
        )


def _plan_payload() -> dict[str, Any]:
    return {
        "scope_summary": "Verify a small set of company facts using supplied sources.",
        "questions": [
            {
                "question": "What year was Example Research Labs founded?",
                "priority": 1,
                "rationale": "Establish the founding year from source text.",
                "evidence_requirements": ["A directly quoted source statement"],
                "search_hints": ["Example Research Labs founding year"],
            },
            {
                "question": "Where is Example Research Labs headquartered?",
                "priority": 1,
                "rationale": "Establish the headquarters location.",
                "evidence_requirements": ["A directly quoted source statement"],
                "search_hints": ["Example Research Labs headquarters"],
            },
            {
                "question": "Which sources corroborate the company profile?",
                "priority": 2,
                "rationale": "Check that independent sources are available.",
                "evidence_requirements": ["Independent source corroboration"],
                "search_hints": ["Example Research Labs independent profile"],
            },
            {
                "question": "What limitations apply to these source records?",
                "priority": 3,
                "rationale": "Record limits of the supplied source material.",
                "evidence_requirements": ["Source limitations"],
                "search_hints": ["Example Research Labs source profile"],
            },
            {
                "question": "Are the company facts consistent across sources?",
                "priority": 2,
                "rationale": "Check consistency between source statements.",
                "evidence_requirements": ["Cross-source consistency"],
                "search_hints": ["Example Research Labs company facts"],
            },
        ],
        "completion_criteria": [
            "Directly supported facts are cited",
            "Source limits are disclosed",
        ],
    }


def _evidence_payload(request: CanonicalModelRequest, *, empty: bool) -> dict[str, Any]:
    if empty:
        return {"items": []}
    content = "\n".join(
        part.value if isinstance(part.value, str) else json.dumps(part.value, sort_keys=True)
        for part in request.content_parts
    )
    if FOUNDING_QUOTE in content:
        quote = FOUNDING_QUOTE
        claim = FOUNDING_QUOTE
    elif HEADQUARTERS_QUOTE in content:
        quote = HEADQUARTERS_QUOTE
        claim = HEADQUARTERS_QUOTE
    else:
        return {"items": []}
    return {
        "items": [
            {
                "claim": claim,
                "exact_quote": quote,
                "dimension_key": None,
                "relation": "supports",
                "relevance": 0.99,
                "confidence": 0.99,
            }
        ]
    }


def _report_payload(request: CanonicalModelRequest) -> dict[str, Any]:
    raw = request.content_parts[0].value
    payload = json.loads(raw) if isinstance(raw, str) else raw
    cards = payload.get("evidence_cards", []) if isinstance(payload, dict) else []
    if not cards:
        raise AssertionError("report fixture must receive persisted accepted evidence cards")
    card = cards[0]
    evidence_id = str(card["evidence_id"])
    claim = str(card["claim"])
    question_id = str(card["question_id"])
    return {
        "title": "Example Research Labs — deterministic integration report",
        "executive_summary": [{"text": claim, "evidence_ids": [evidence_id]}],
        "sections": [
            {
                "question_id": question_id,
                "title": "Supported fact",
                "paragraphs": [{"text": claim, "evidence_ids": [evidence_id]}],
            }
        ],
        "limitations": ["This deterministic fixture validates pipeline plumbing only."],
    }


def deterministic_result_order() -> tuple[str, ...]:
    """A small public hook for tests that prove fixture repeatability."""
    return (SOURCE_A_URL, SOURCE_B_URL)


def install_fixture_http_transport(
    monkeypatch: Any,
    *,
    html_by_url: dict[str, str],
    request_log: list[str],
) -> None:
    """Force all worker AsyncClient traffic through an allow-listed in-memory transport."""
    import httpx

    original_init: Callable[..., None] = httpx.AsyncClient.__init__

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        request_log.append(url)
        if request.method != "GET" or url not in html_by_url:
            raise AssertionError(f"L2 attempted unapproved HTTP access: {request.method} {url}")
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            text=html_by_url[url],
            request=request,
        )

    def guarded_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        kwargs["transport"] = httpx.MockTransport(handler)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", guarded_init)


__all__ = [
    "FOUNDING_QUOTE",
    "HEADQUARTERS_QUOTE",
    "SOURCE_A_TITLE",
    "SOURCE_A_URL",
    "SOURCE_B_TITLE",
    "SOURCE_B_URL",
    "DeterministicFixtureLLMGateway",
    "DeterministicFixtureSearchProvider",
    "deterministic_result_order",
    "install_fixture_http_transport",
]
