"""Phase 14.2: evidence-aware Research Loop integration tests.

Covers the spec cases:

1. no ResearchContext -> the query stays byte-identical;
2. hints present -> the SearchTarget query is enhanced by appending them;
3. enrichment is idempotent for the same context;
4. identity independence (q1 / q100 / random_gap get the same treatment);
5. domain independence (medical / finance / software / industrial share hints);
6. a real ``ResearchTarget`` reaching the provider dispatch carries the
   enriched query, plus the additive ``research.query.context.enriched``
   handoff and the event-free legacy path.
"""

from __future__ import annotations

import json
from dataclasses import replace
from uuid import uuid4

import httpx
import pytest
from app.domain.provider_health import ProviderHealthState
from app.domain.provider_router import ProviderSelectionDecision
from app.domain.research_context import EMPTY_RESEARCH_CONTEXT, ResearchContext
from app.domain.research_context_enricher import (
    ResearchContextEnricher,
    ResearchContextEnrichmentResult,
)
from app.domain.research_context_resolver import (
    SNAPSHOT_KEY,
    ResearchContextResolver,
)
from app.domain.research_tools import SearchResult
from app.infrastructure.db.research_tools import ResearchTarget
from app.services.research_loop import ResearchLoopService
from app.tools.web_search import SearXNGSearchProvider
from test_research_loop import FakeArtifacts, FakeExtractor, FakeReader, FakeRepository

ENRICHER = ResearchContextEnricher()


def _context() -> ResearchContext:
    return ResearchContext(
        evidence_failure_reason="claim_unverified",
        missing_evidence_type="CLAIM_VERIFICATION",
        refined_need_type="CLAIM_VALIDATION",
        query_hints=("benchmark", "evaluation", "paper"),
    )


# ---------------------------------------------------------------- Case 1


def test_enrich_without_context_keeps_query_byte_identical() -> None:
    result = ENRICHER.enrich("abc", None)

    assert isinstance(result, ResearchContextEnrichmentResult)
    assert result.original_query == "abc"
    assert result.enriched_query == "abc"
    assert result.applied_hints == ()
    assert result.changed is False


def test_enrich_with_empty_or_hintless_context_keeps_query_byte_identical() -> None:
    for context in (EMPTY_RESEARCH_CONTEXT, ResearchContext(missing_evidence_type="X")):
        result = ENRICHER.enrich("abc", context)
        assert result.enriched_query == "abc"
        assert result.changed is False


# ---------------------------------------------------------------- Case 2


def test_enrich_appends_missing_evidence_hints() -> None:
    result = ENRICHER.enrich("model performance verification", _context())

    assert result.enriched_query == (
        "model performance verification benchmark evaluation paper"
    )
    assert result.applied_hints == ("benchmark", "evaluation", "paper")
    assert result.changed is True
    # The base expression is only ever appended to, never rewritten.
    assert result.enriched_query.startswith(result.original_query)


def test_enrich_skips_hints_already_present_and_stays_byte_identical() -> None:
    context = ResearchContext(query_hints=("benchmark", "Benchmark "))
    result = ENRICHER.enrich("vision defect detection benchmark", context)

    assert result.enriched_query == "vision defect detection benchmark"
    assert result.applied_hints == ()
    assert result.changed is False


def test_enrichment_result_as_dict_is_json_safe() -> None:
    payload = ENRICHER.enrich("abc", _context()).as_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["changed"] is True


# ---------------------------------------------------------------- Case 3


def test_enrich_is_idempotent_for_the_same_context() -> None:
    context = _context()
    first = ENRICHER.enrich("model performance verification", context)
    second = ENRICHER.enrich(first.enriched_query, context)
    third = ENRICHER.enrich(second.enriched_query, context)

    assert second.enriched_query == first.enriched_query
    assert third.enriched_query == first.enriched_query
    # Repeated feedback never grows the query without bound.
    assert second.applied_hints == ()
    assert second.changed is False
    assert len(first.enriched_query.split()) == 6


# ---------------------------------------------------------------- Case 4


def test_enrichment_is_identity_agnostic() -> None:
    context = _context()
    suffixes = set()
    for question_identity in ("q1", "q100", "random_gap"):
        result = ENRICHER.enrich(f"{question_identity} survey answers", context)
        assert result.changed is True
        suffixes.add(result.enriched_query[len(result.original_query):])

    assert suffixes == {" benchmark evaluation paper"}


# ---------------------------------------------------------------- Case 5


def test_same_failure_yields_same_hints_across_domains() -> None:
    metadata = {
        "evidence_failure_reason": "no_independent_validation",
        "missing_evidence_type": "INDEPENDENT_SOURCE",
        "refined_need_type": "INDEPENDENT_SOURCE_DISCOVERY",
        "query_hints": ["market report", "official statistics"],
    }
    context = ResearchContext.from_mapping(metadata)
    suffixes = set()
    for domain in ("medical", "finance", "software", "industrial"):
        base = f"{domain} equipment adoption rate 2025"
        result = ENRICHER.enrich(base, context)
        assert result.changed is True
        suffixes.add(result.enriched_query[len(base):])

    assert suffixes == {" market report official statistics"}


# ---------------------------------------------------------------- resolver


def test_resolver_roundtrip_keeps_one_entry_per_question_latest_wins() -> None:
    need_id = uuid4()
    snapshot = ResearchContextResolver.record(
        {},
        question_id="q7",
        context=_context(),
        research_need_id=need_id,
        query_plan_id="plan-1",
        query_candidate_id="candidate-1",
    )
    resolved = ResearchContextResolver.resolve(snapshot, question_id="q7")

    assert resolved is not None
    assert resolved.context.query_hints == ("benchmark", "evaluation", "paper")
    assert resolved.research_need_id == str(need_id)
    assert resolved.query_plan_id == "plan-1"
    assert resolved.query_candidate_id == "candidate-1"
    assert json.loads(json.dumps(snapshot[SNAPSHOT_KEY])) == snapshot[SNAPSHOT_KEY]

    refined = ResearchContextResolver.record(
        snapshot,
        question_id="q7",
        context=ResearchContext(missing_evidence_type="EVIDENCE_QUALITY"),
        research_need_id=uuid4(),
    )
    entries = refined[SNAPSHOT_KEY]
    assert isinstance(entries, dict)
    assert list(entries) == ["q7"]
    again = ResearchContextResolver.resolve(refined, question_id="q7")
    assert again is not None
    assert again.context.missing_evidence_type == "EVIDENCE_QUALITY"
    assert again.query_candidate_id is None


def test_resolver_returns_none_for_absent_or_malformed_entries() -> None:
    assert ResearchContextResolver.resolve(None, question_id="q1") is None
    assert ResearchContextResolver.resolve({}, question_id="q1") is None
    assert (
        ResearchContextResolver.resolve(
            {SNAPSHOT_KEY: "not-a-mapping"}, question_id="q1"
        )
        is None
    )
    assert (
        ResearchContextResolver.resolve(
            {SNAPSHOT_KEY: {"q1": {"query_hints": "not-a-list"}}}, question_id="q1"
        )
        is None
    )
    assert (
        ResearchContextResolver.resolve(
            {SNAPSHOT_KEY: {"q2": {"query_hints": ["x"]}}},
            question_id="q1",
        )
        is None
    )


# ------------------------------------------------------------- Case 6


class QuerySpyAdapter:
    """Provider adapter double that records the query handed to the provider."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(
        self,
        query: str,
        *,
        limit: int = 8,
        exclude_domains: tuple[str, ...] = (),
        exclude_urls: tuple[str, ...] = (),
        request_context: object = None,
    ) -> list[SearchResult]:
        del limit, exclude_domains, exclude_urls, request_context
        self.queries.append(query)
        return [
            SearchResult(title=f"Source {rank}", url=f"https://example.com/{rank}", rank=rank)
            for rank in (1, 2, 3)
        ]


class ContextRepository(FakeRepository):
    """Fake repository wired through the explicit resolver/recorder interface."""

    def __init__(self, *, context: ResearchContext | None) -> None:
        super().__init__()
        snapshot: dict[str, object] = {}
        if context is not None:
            snapshot = ResearchContextResolver.record(
                snapshot,
                question_id="q1",
                context=context,
                research_need_id=uuid4(),
                query_plan_id=uuid4(),
                query_candidate_id=uuid4(),
            )
        self.snapshot = snapshot
        self.enrichment_events: list[dict[str, object]] = []
        self.resolver_accesses = 0
        self.target = replace(
            self.target,
            query="model performance verification",
            question="How is agent model performance verified?",
        )

    async def research_context_for_question(
        self, run_id: object, *, worker_task_id: str, question_id: str
    ) -> object:
        del run_id, worker_task_id
        # Phase 14.5: the disabled switch must gate before any resolver access
        # so the baseline arm keeps zero main-path side effects.
        self.resolver_accesses += 1
        return ResearchContextResolver.resolve(self.snapshot, question_id=question_id)

    async def record_query_context_enriched(
        self,
        run_id: object,
        *,
        worker_task_id: str,
        target: ResearchTarget,
        resolved: object,
        enrichment: ResearchContextEnrichmentResult,
    ) -> None:
        del run_id, worker_task_id
        assert resolved is not None
        self.enrichment_events.append(
            {
                "refs": {
                    "research_need_id": resolved.research_need_id,
                    "query_plan_id": resolved.query_plan_id,
                    "query_candidate_id": resolved.query_candidate_id,
                },
                "metrics": {
                    "hint_count": len(resolved.context.query_hints),
                    "query_changed": enrichment.changed,
                },
            }
        )

    async def select_provider_for_target(
        self, *args: object, **kwargs: object
    ) -> ProviderSelectionDecision:
        return ProviderSelectionDecision(
            selected_provider="QuerySpy",
            excluded_providers=(),
            reason="healthy_provider_selected",
            health_state=ProviderHealthState.HEALTHY,
            fallback_used=False,
        )


def _service(
    repository: ContextRepository,
    *,
    evidence_aware_context_enabled: bool = False,
) -> tuple[ResearchLoopService, QuerySpyAdapter]:
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json={"results": []}))
    provider = SearXNGSearchProvider(httpx.AsyncClient(transport=transport), "http://searxng.test")
    spy = QuerySpyAdapter()
    service = ResearchLoopService(
        repository,  # type: ignore[arg-type]
        provider,  # type: ignore[arg-type]
        FakeReader(),
        FakeExtractor(),
        FakeArtifacts(),
        search_adapters={"QuerySpy": spy},
        evidence_aware_context_enabled=evidence_aware_context_enabled,
    )
    return service, spy


@pytest.mark.asyncio
async def test_research_context_enriches_the_real_search_target_query() -> None:
    repository = ContextRepository(context=_context())
    # Phase 14.5 made the flag default-off; the enabled arm is exercised here
    # with the flag explicitly on (Task B: the old behavior is preserved).
    service, spy = _service(repository, evidence_aware_context_enabled=True)

    await service.run_one_iteration(uuid4(), worker_task_id="enrich-1")

    # ClosureFeedback -> ResearchContext -> resolver -> enricher -> provider.
    assert spy.queries == ["model performance verification benchmark evaluation paper"]
    assert repository.resolver_accesses == 1
    assert len(repository.enrichment_events) == 1
    event = repository.enrichment_events[0]
    assert event["metrics"] == {"hint_count": 3, "query_changed": True}
    refs = event["refs"]
    assert isinstance(refs, dict)
    assert all(isinstance(value, str) and value for value in refs.values())


@pytest.mark.asyncio
async def test_switch_disabled_keeps_baseline_query_and_emits_no_event() -> None:
    # Phase 14.3 A/B baseline arm, Phase 14.5 V1 default: even with a
    # ResearchContext present, the disabled switch bypasses the enrichment
    # layer before any resolver access so the provider query and the event
    # stream stay identical to Phase 12.4.
    repository = ContextRepository(context=_context())
    service, spy = _service(repository, evidence_aware_context_enabled=False)

    await service.run_one_iteration(uuid4(), worker_task_id="baseline-arm-1")

    assert spy.queries == ["model performance verification"]
    assert repository.enrichment_events == []
    assert repository.resolver_accesses == 0


@pytest.mark.asyncio
async def test_service_default_construction_is_enrichment_disabled() -> None:
    # Phase 14.5 closeout: constructing the service without the flag (the
    # worker passes settings.evidence_aware_context_enabled, whose V1 default
    # is false) must already be the byte-identical baseline path.
    repository = ContextRepository(context=_context())
    service, spy = _service(repository)

    await service.run_one_iteration(uuid4(), worker_task_id="default-off-1")

    assert spy.queries == [repository.target.query]
    assert repository.enrichment_events == []
    assert repository.resolver_accesses == 0


@pytest.mark.asyncio
async def test_absent_research_context_keeps_provider_query_and_emits_no_event() -> None:
    repository = ContextRepository(context=None)
    service, spy = _service(repository)

    await service.run_one_iteration(uuid4(), worker_task_id="legacy-1")

    # Byte-identical legacy behavior and no enrichment event at all.
    assert spy.queries == ["model performance verification"]
    assert repository.enrichment_events == []


@pytest.mark.asyncio
async def test_repository_without_resolver_interface_keeps_legacy_path() -> None:
    repository = FakeRepository()
    spy = QuerySpyAdapter()
    provider = _searxng_provider()
    service = ResearchLoopService(
        repository,  # type: ignore[arg-type]
        provider,  # type: ignore[arg-type]
        FakeReader(),
        FakeExtractor(),
        FakeArtifacts(),
        search_adapters={"SearXNG": spy, "Bing": spy},
    )

    await service.run_one_iteration(uuid4(), worker_task_id="legacy-2")

    assert spy.queries == [repository.target.query]


def _searxng_provider() -> SearXNGSearchProvider:
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json={"results": []}))
    return SearXNGSearchProvider(httpx.AsyncClient(transport=transport), "http://searxng.test")
