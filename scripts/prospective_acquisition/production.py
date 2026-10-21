"""Production component binding for the frozen bounded acquisition profile.

This is a fixed-depth adapter over the same Search, Reader, EvidenceExtractor,
artifact-store, and ResearchToolRepository abstractions used by the worker.
All HTTP clients are supplied by the profile so dispatch budgets sit below the
production adapters at the HTTP transport boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit
from uuid import UUID

from app.context.manager import ContextBudgetManager
from app.domain.provider_adapter import SearchRequestContext
from app.domain.providers import TokenUsage
from app.domain.research_tools import ReadPage, ScoredEvidence
from app.infrastructure.artifacts import LocalArtifactStore
from app.infrastructure.db.extraction_cache import ExtractionCacheRepository
from app.infrastructure.db.research_tools import (
    PagePersistenceResult,
    ResearchTarget,
    ResearchToolRepository,
)
from app.infrastructure.db.run_providers import RunProviderBindingRepository
from app.llm.adapters import LLMGateway
from app.security.secrets import SecretCipher
from app.services.evidence_extractor import EvidenceExtractorService
from app.tools.provider_adapters import SearXNGAdapter
from app.tools.web_reader import PublicWebReader
from app.tools.web_search import SearXNGSearchProvider

from scripts.prospective_acquisition.control import ClaimRecord
from scripts.prospective_acquisition.execution_identity import (
    FrozenProviderBinding,
    ProspectiveAcquisitionExecutionContext,
)
from scripts.prospective_acquisition.profile import (
    AcquisitionComponents,
    AcquisitionHTTPClients,
    AcquisitionSource,
    AcquisitionTask,
)


@dataclass(frozen=True)
class ExtractedEvidenceBatch:
    evidence: tuple[ScoredEvidence, ...]
    usage: TokenUsage
    context_manifest: dict[str, int | bool]


class ProductionSearchComponent:
    def __init__(self, provider: SearXNGAdapter) -> None:
        self._provider = provider

    async def search(self, query: str, *, limit: int) -> Sequence[AcquisitionSource]:
        results = await self._provider.search(
            query,
            limit=min(limit, 1),
            request_context=SearchRequestContext(provider_request_allowance=1),
        )
        return tuple(AcquisitionSource(url=item.url, title=item.title) for item in results[:1])


class ProductionReaderComponent:
    def __init__(self, reader: PublicWebReader) -> None:
        self._reader = reader

    async def read(self, url: str) -> ReadPage:
        return await self._reader.read(url)


class ProductionExtractorComponent:
    def __init__(
        self,
        extractor: EvidenceExtractorService,
        *,
        run_id: UUID,
        target: ResearchTarget,
        bindings: RunProviderBindingRepository,
        frozen_binding: FrozenProviderBinding,
    ) -> None:
        self._extractor = extractor
        self._run_id = run_id
        self._target = target
        self._bindings = bindings
        self._frozen_binding = frozen_binding

    async def extract(self, query: str, page: object) -> ExtractedEvidenceBatch:
        if not isinstance(page, ReadPage):
            raise TypeError("production Reader must return ReadPage")
        if query != self._target.query:
            raise ValueError("task query does not match the bound ResearchTarget")
        run_binding = await self._bindings.get(self._run_id)
        expected = self._frozen_binding
        if (
            run_binding.credential_id != UUID(expected.credential_version_id)
            or run_binding.credential_version != expected.credential_version
            or run_binding.adapter_type.value != expected.adapter
            or run_binding.model != expected.model
            or _endpoint_family(run_binding.base_url) != expected.endpoint_family
        ):
            raise RuntimeError("EXECUTION_PROVIDER_BINDING_CHANGED")
        evidence, usage, context = await self._extractor.extract(
            self._run_id,
            question=self._target.question,
            acceptance_dimensions=self._target.acceptance_dimensions,
            page=page,
            source_id=self._target.source_id_seed,
        )
        return ExtractedEvidenceBatch(tuple(evidence), usage, context)


class ProductionClaimFormationComponent:
    def __init__(
        self,
        repository: ResearchToolRepository,
        artifact_store: LocalArtifactStore,
        *,
        task: AcquisitionTask,
        target: ResearchTarget,
        run_id: UUID,
        worker_task_id: str,
    ) -> None:
        self._repository = repository
        self._artifact_store = artifact_store
        self._task = task
        self._target = target
        self._run_id = run_id
        self._worker_task_id = worker_task_id

    async def record_page(
        self,
        task: AcquisitionTask,
        page: object,
        evidence: object,
    ) -> Sequence[ClaimRecord]:
        if task != self._task or not isinstance(page, ReadPage):
            raise ValueError("task/page does not match production acquisition binding")
        if not isinstance(evidence, ExtractedEvidenceBatch):
            raise RuntimeError("missing production extraction usage/context metadata")
        if any(not isinstance(item, ScoredEvidence) for item in evidence.evidence):
            raise TypeError("production EvidenceExtractor must return ScoredEvidence")
        artifact_uri = await self._artifact_store.save_page(
            self._run_id, self._target.source_id_seed, page.clean_text
        )
        result = await self._repository.record_page_with_receipts(
            self._run_id,
            worker_task_id=self._worker_task_id,
            target=self._target,
            source_id=self._target.source_id_seed,
            page=page,
            artifact_uri=artifact_uri,
            evidence=list(evidence.evidence),
            usage=evidence.usage,
            context_manifest=evidence.context_manifest,
        )
        if not isinstance(result, PagePersistenceResult):
            raise TypeError("repository did not return authoritative persistence receipts")
        return tuple(
            ClaimRecord(
                claim_id=str(receipt.claim_id),
                created_at=receipt.created_at.isoformat(),
                source_refs=receipt.source_provenance_refs,
                source_key=receipt.source_diversity_key,
                owner_key=receipt.source_owner_key,
                content_hash=receipt.claim_content_hash,
                provenance_complete=bool(
                    receipt.evidence_ids
                    and receipt.source_ids
                    and receipt.snapshot_ids
                    and receipt.source_provenance_refs
                    and receipt.source_diversity_key
                ),
                identity_isolation_attested=True,
                evidence_ids=tuple(str(item) for item in receipt.evidence_ids),
                source_ids=tuple(str(item) for item in receipt.source_ids),
                snapshot_ids=tuple(str(item) for item in receipt.snapshot_ids),
            )
            for receipt in result.claim_receipts
            if receipt.accepted_and_persisted
        )


class ProductionAcquisitionComponentFactory:
    """Build all production components around the profile's guarded clients."""

    def __init__(
        self,
        *,
        search_base_url: str,
        bindings: RunProviderBindingRepository,
        cipher: SecretCipher,
        repository: ResearchToolRepository,
        artifact_store: LocalArtifactStore,
        contexts: ContextBudgetManager | None = None,
        extraction_cache: ExtractionCacheRepository | None = None,
    ) -> None:
        self._search_base_url = search_base_url
        self._bindings = bindings
        self._cipher = cipher
        self._repository = repository
        self._artifact_store = artifact_store
        self._contexts = contexts
        self._extraction_cache = extraction_cache

    def __call__(
        self,
        task: AcquisitionTask,
        context: ProspectiveAcquisitionExecutionContext,
        binding: FrozenProviderBinding,
        clients: AcquisitionHTTPClients,
    ) -> AcquisitionComponents:
        del context
        if task.research_target is None or not task.worker_task_id:
            raise ValueError("task lacks its authenticated run target/worker lease binding")
        run_id = UUID(task.run_id)
        if task.research_target.query != task.task_text:
            raise ValueError("task query differs from frozen ResearchTarget query")

        search_provider = SearXNGSearchProvider(
            clients.search,
            self._search_base_url,
            fallback_client=clients.search,
        )
        search = ProductionSearchComponent(SearXNGAdapter(search_provider))
        reader = ProductionReaderComponent(PublicWebReader(clients.reader))
        extractor_service = EvidenceExtractorService(
            self._bindings,
            self._cipher,
            LLMGateway(clients.provider),
            contexts=self._contexts,
            extraction_cache=self._extraction_cache,
        )
        extractor = ProductionExtractorComponent(
            extractor_service,
            run_id=run_id,
            target=task.research_target,
            bindings=self._bindings,
            frozen_binding=binding,
        )
        claim_formation = ProductionClaimFormationComponent(
            self._repository,
            self._artifact_store,
            task=task,
            target=task.research_target,
            run_id=run_id,
            worker_task_id=task.worker_task_id,
        )
        return AcquisitionComponents(search, reader, extractor, claim_formation)


def _endpoint_family(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if not parsed.scheme or not parsed.hostname:
        return ""
    return f"{parsed.scheme.lower()}://{parsed.hostname.lower()}"
