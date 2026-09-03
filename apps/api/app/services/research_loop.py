"""Budgeted, evidence-driven autonomous research loop."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit
from uuid import UUID

from app.domain.controlled_tools import ControlledToolName, EvidenceSearchInput, ToolDecisionRequest
from app.domain.identifiers import uuid7
from app.domain.research_tools import SearchResult
from app.infrastructure.artifacts import LocalArtifactStore
from app.infrastructure.db.research_tools import ResearchTarget, ResearchToolRepository
from app.llm.adapters import ModelGatewayError
from app.services.evidence_extractor import EvidenceExtractorService
from app.tools.errors import ToolExecutionError
from app.tools.gateway import ControlledToolGateway
from app.tools.web_reader import PublicWebReader
from app.tools.web_search import SearXNGSearchProvider

_MAX_SEARCH_RESULTS = 10
_MAX_PAGE_ATTEMPTS = 8
_MAX_PAGES_READ = 3
_ISOLATED_EXTRACTION_ERRORS = {
    "EVIDENCE_OUTPUT_SCHEMA_INVALID",
    "MODEL_OUTPUT_INVALID",
    "MODEL_OUTPUT_TRUNCATED",
    "MODEL_RESPONSE_INVALID",
    "MODEL_TIMEOUT",
    "MODEL_NETWORK_ERROR",
    "MODEL_RATE_LIMITED",
    "MODEL_PROVIDER_UNAVAILABLE",
}
_STRUCTURED_EXTRACTION_ERRORS = {
    "EVIDENCE_OUTPUT_SCHEMA_INVALID",
    "MODEL_OUTPUT_INVALID",
    "MODEL_OUTPUT_TRUNCATED",
    "MODEL_RESPONSE_INVALID",
}
_STRUCTURED_EXTRACTION_FAILURE_LIMIT = 2
_PREFERRED_READABLE_DOMAINS = {
    "arxiv.org",
    "openaccess.thecvf.com",
    "pmc.ncbi.nlm.nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
}
_RESTRICTED_SOURCE_DOMAINS = {
    "doi.org",
    "ieeexplore.ieee.org",
    "link.springer.com",
    "researchgate.net",
    "sciencedirect.com",
    "tandfonline.com",
}


@dataclass(frozen=True, slots=True)
class ResearchIterationResult:
    outcome: str
    continue_research: bool
    decision: str
    pages_read: int
    accepted_evidence: int
    coverage: float
    information_gain: float
    low_information_gain_streak: int


@dataclass(frozen=True, slots=True)
class ResearchAttemptResult:
    pages_read: int
    accepted_evidence: int
    outcome: str


class ResearchLoopService:
    def __init__(
        self,
        repository: ResearchToolRepository,
        search: SearXNGSearchProvider,
        reader: PublicWebReader,
        extractor: EvidenceExtractorService,
        artifacts: LocalArtifactStore,
        controlled_tools: ControlledToolGateway | None = None,
    ) -> None:
        self._repository = repository
        self._search = search
        self._reader = reader
        self._extractor = extractor
        self._artifacts = artifacts
        self._controlled_tools = controlled_tools

    async def run_one_iteration(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
    ) -> ResearchIterationResult:
        """Execute one bounded research iteration for one LangGraph super-step."""

        target = await self._repository.prepare_target(
            run_id,
            worker_task_id=worker_task_id,
        )
        if target is None:
            return ResearchIterationResult(
                outcome=self._outcome("no_pending_question", 0, 0, 0),
                continue_research=False,
                decision="no_pending_question",
                pages_read=0,
                accepted_evidence=0,
                coverage=0.0,
                information_gain=0.0,
                low_information_gain_streak=0,
            )

        attempt = await self._research_target(
            run_id,
            worker_task_id=worker_task_id,
            target=target,
        )
        evaluation = await self._repository.finish_iteration(
            run_id,
            worker_task_id=worker_task_id,
            target=target,
            attempt_outcome=attempt.outcome,
        )
        return ResearchIterationResult(
            outcome=self._outcome(
                evaluation.decision,
                0 if attempt.outcome == "provider_error" else 1,
                attempt.pages_read,
                attempt.accepted_evidence,
            ),
            continue_research=evaluation.continue_research,
            decision=evaluation.decision,
            pages_read=attempt.pages_read,
            accepted_evidence=attempt.accepted_evidence,
            coverage=evaluation.coverage,
            information_gain=evaluation.information_gain,
            low_information_gain_streak=evaluation.low_information_gain_streak,
        )

    async def run_iteration(self, run_id: UUID, *, worker_task_id: str) -> str:
        """Compatibility wrapper; the production Graph calls one iteration per node."""

        completed_iterations = 0
        total_pages = 0
        total_accepted = 0
        last_decision = "no_pending_question"
        while True:
            result = await self.run_one_iteration(
                run_id,
                worker_task_id=worker_task_id,
            )
            completed_iterations += 1 if result.decision != "no_pending_question" else 0
            total_pages += result.pages_read
            total_accepted += result.accepted_evidence
            last_decision = result.decision
            if not result.continue_research:
                return self._outcome(
                    last_decision,
                    completed_iterations,
                    total_pages,
                    total_accepted,
                )

    async def _research_target(
        self,
        run_id: UUID,
        *,
        worker_task_id: str,
        target: ResearchTarget,
    ) -> ResearchAttemptResult:
        if self._controlled_tools is not None:
            await self._controlled_tools.search_evidence(
                EvidenceSearchInput(
                    run_id=run_id,
                    action_id=target.tool_call_id,
                    target_gap_ids=(target.gap_id,),
                    question_id=target.question_id,
                    query=target.query,
                )
            )
            self._controlled_tools.authorize_web_search(
                ToolDecisionRequest(
                    action_id=target.tool_call_id,
                    tool_name=ControlledToolName.WEB_SEARCH,
                    target_gap_ids=(target.gap_id,),
                    duplicate_key=f"web-search:{target.tool_call_id}",
                    evidence_checked=True,
                )
            )
        try:
            results = await self._search.search(target.query, limit=_MAX_SEARCH_RESULTS)
        except ToolExecutionError as exc:
            await self._repository.record_tool_failure(
                run_id,
                worker_task_id=worker_task_id,
                target=target,
                error_code=exc.code,
                retryable=exc.retryable,
            )
            return ResearchAttemptResult(0, 0, "provider_error")

        await self._repository.record_search_results(
            run_id,
            worker_task_id=worker_task_id,
            target=target,
            results=results,
        )
        pages_read = 0
        accepted = 0
        structured_extraction_failures = 0
        for result in _prioritize_search_results(
            results,
            query=target.query,
            used_owner_keys=set(target.used_source_owner_keys),
        )[:_MAX_PAGE_ATTEMPTS]:
            if pages_read >= _MAX_PAGES_READ:
                break
            try:
                page = await self._reader.read(result.url)
            except ToolExecutionError as exc:
                await self._repository.record_page_failure(
                    run_id,
                    worker_task_id=worker_task_id,
                    target=target,
                    url=result.url,
                    error_code=exc.code,
                )
                continue

            source_id = uuid7()
            artifact_uri = await self._artifacts.save_page(run_id, source_id, page.clean_text)
            await self._repository.record_extraction_started(
                run_id,
                worker_task_id=worker_task_id,
                target=target,
                source_id=source_id,
            )
            try:
                evidence, usage, manifest = await self._extractor.extract(
                    run_id,
                    question=target.question,
                    acceptance_dimensions=target.acceptance_dimensions,
                    page=page,
                    source_id=source_id,
                )
            except ModelGatewayError as exc:
                if exc.code not in _ISOLATED_EXTRACTION_ERRORS:
                    raise
                await self._repository.record_extraction_failure(
                    run_id,
                    worker_task_id=worker_task_id,
                    target=target,
                    source_id=source_id,
                    page=page,
                    artifact_uri=artifact_uri,
                    error_code=exc.code,
                    detail_code=exc.detail_code,
                )
                pages_read += 1
                if exc.code in _STRUCTURED_EXTRACTION_ERRORS:
                    structured_extraction_failures += 1
                    if (
                        structured_extraction_failures
                        >= _STRUCTURED_EXTRACTION_FAILURE_LIMIT
                        and accepted == 0
                    ):
                        raise ModelGatewayError(
                            "MODEL_CAPABILITY_INSUFFICIENT",
                            retryable=False,
                            detail_code=(
                                "EVIDENCE_EXTRACTOR_CIRCUIT_OPEN_AFTER_2_SOURCES"
                            ),
                        ) from exc
                else:
                    structured_extraction_failures = 0
                continue

            _, page_accepted = await self._repository.record_page(
                run_id,
                worker_task_id=worker_task_id,
                target=target,
                source_id=source_id,
                page=page,
                artifact_uri=artifact_uri,
                evidence=evidence,
                usage=usage,
                context_manifest=manifest,
            )
            pages_read += 1
            accepted += page_accepted
            structured_extraction_failures = 0

        if not results:
            outcome = "zero_results"
        elif pages_read == 0:
            outcome = "unreadable"
        elif accepted == 0:
            outcome = "no_evidence"
        else:
            outcome = "evidence_gained"
        return ResearchAttemptResult(pages_read, accepted, outcome)

    @staticmethod
    def _outcome(decision: str, iterations: int, pages: int, accepted: int) -> str:
        return (
            f"research_stopped:{decision}:iterations={iterations}:pages={pages}:accepted={accepted}"
        )


def _prioritize_search_results(
    results: list[SearchResult],
    *,
    query: str = "",
    used_owner_keys: set[str] | None = None,
) -> list[SearchResult]:
    """Rank readable, relevant and owner-diverse public sources first."""

    used_owners = used_owner_keys or set()
    query_tokens = _search_tokens(query)

    def sort_key(result: SearchResult) -> tuple[float, int]:
        parsed = urlsplit(result.url)
        hostname = (parsed.hostname or "").lower()
        path = parsed.path.lower().rstrip("/")
        penalty = 0.0
        if _matches_domain(hostname, _PREFERRED_READABLE_DOMAINS):
            penalty -= 20
        if _matches_domain(hostname, _RESTRICTED_SOURCE_DOMAINS):
            penalty += 20
        if path.endswith(".pdf") or "/pdf" in path:
            penalty += 30
        if _source_owner_key(result.url) in used_owners:
            penalty += 18
        candidate_tokens = _search_tokens(f"{result.title} {result.snippet}")
        overlap = len(query_tokens & candidate_tokens) / max(len(query_tokens), 1)
        penalty -= overlap * 12
        return penalty, result.rank

    return sorted(results, key=sort_key)


def _matches_domain(hostname: str, domains: set[str]) -> bool:
    return any(hostname == domain or hostname.endswith(f".{domain}") for domain in domains)


def _source_owner_key(url: str) -> str:
    hostname = (urlsplit(url).hostname or "unknown").lower().rstrip(".")
    for prefix in ("www.", "m.", "en.", "zh.", "docs."):
        if hostname.startswith(prefix):
            hostname = hostname[len(prefix) :]
    return hostname[:255]


def _search_tokens(value: str) -> set[str]:
    normalized = "".join(char.casefold() if char.isalnum() else " " for char in value)
    words = {word for word in normalized.split() if len(word) > 1}
    cjk = {
        normalized[index : index + 2]
        for index in range(max(0, len(normalized) - 1))
        if all("\u4e00" <= char <= "\u9fff" for char in normalized[index : index + 2])
    }
    return words | cjk
