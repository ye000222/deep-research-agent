"""Budgeted, evidence-driven autonomous research loop."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from app.domain.controlled_tools import ControlledToolName, EvidenceSearchInput, ToolDecisionRequest
from app.domain.identifiers import uuid7
from app.infrastructure.artifacts import LocalArtifactStore
from app.infrastructure.db.research_tools import ResearchTarget, ResearchToolRepository
from app.llm.adapters import ModelGatewayError
from app.services.evidence_extractor import EvidenceExtractorService
from app.tools.errors import ToolExecutionError
from app.tools.gateway import ControlledToolGateway
from app.tools.web_reader import PublicWebReader
from app.tools.web_search import SearXNGSearchProvider

_MAX_SEARCH_RESULTS = 10
_MAX_PAGE_ATTEMPTS = 6
_MAX_PAGES_READ = 2
_ISOLATED_EXTRACTION_ERRORS = {
    "EVIDENCE_OUTPUT_SCHEMA_INVALID",
    "MODEL_OUTPUT_INVALID",
    "MODEL_RESPONSE_INVALID",
    "MODEL_TIMEOUT",
    "MODEL_NETWORK_ERROR",
    "MODEL_RATE_LIMITED",
    "MODEL_PROVIDER_UNAVAILABLE",
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

        pages_read, accepted = await self._research_target(
            run_id,
            worker_task_id=worker_task_id,
            target=target,
        )
        evaluation = await self._repository.finish_iteration(
            run_id,
            worker_task_id=worker_task_id,
            target=target,
        )
        return ResearchIterationResult(
            outcome=self._outcome(evaluation.decision, 1, pages_read, accepted),
            continue_research=evaluation.continue_research,
            decision=evaluation.decision,
            pages_read=pages_read,
            accepted_evidence=accepted,
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
    ) -> tuple[int, int]:
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
            return 0, 0

        await self._repository.record_search_results(
            run_id,
            worker_task_id=worker_task_id,
            target=target,
            results=results,
        )
        pages_read = 0
        accepted = 0
        for result in results[:_MAX_PAGE_ATTEMPTS]:
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
                    page=page,
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
                )
                pages_read += 1
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

        return pages_read, accepted

    @staticmethod
    def _outcome(decision: str, iterations: int, pages: int, accepted: int) -> str:
        return (
            f"research_stopped:{decision}:iterations={iterations}:pages={pages}:accepted={accepted}"
        )
