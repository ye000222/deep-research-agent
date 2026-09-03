"""Provider-aware context budgeting, MMR selection, compression, and audit manifests."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import DataError, IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.context.compressor import CompressionResult, compress_candidate
from app.context.policy import ContextPolicyRegistry
from app.context.ranker import mmr_order
from app.domain.context import (
    CompressionArtifactMetric,
    ContextBudgetAllocation,
    ContextCandidate,
    ContextEnvelope,
    ContextItemMetric,
    ContextManifestView,
)
from app.domain.identifiers import uuid7
from app.infrastructure.db.context_models import (
    CompressionArtifactRow,
    ContextItemRow,
    ContextManifestRow,
)

DEFAULT_CONTEXT_WINDOW = 16_000
MIN_INPUT_BUDGET = 512
MAX_INPUT_FRACTION = 0.70


class ContextBudgetInsufficientError(ValueError):
    """Raised when protected context cannot fit without unsafe truncation."""


class ContextManifestPersistenceError(RuntimeError):
    """Raised when a context manifest cannot be persisted safely."""

    code = "CONTEXT_MANIFEST_PERSISTENCE_FAILED"


def estimate_tokens(content: str) -> int:
    """Use a deterministic conservative estimate when provider tokenizers are unavailable."""
    if not content:
        return 0
    return max(1, math.ceil(len(content) / 3))


def allocate_budget(
    *,
    context_window: int | None,
    requested_output_tokens: int,
    provider_max_output_tokens: int | None = None,
) -> ContextBudgetAllocation:
    window = context_window or DEFAULT_CONTEXT_WINDOW
    if window < 1024:
        raise ContextBudgetInsufficientError("context window is too small")
    output_cap = provider_max_output_tokens or requested_output_tokens
    output_reserve = min(requested_output_tokens, output_cap)
    safety_margin = max(512, math.ceil(window * 0.10))
    available = window - output_reserve - safety_margin
    input_budget = min(available, math.floor(window * MAX_INPUT_FRACTION))
    if input_budget < MIN_INPUT_BUDGET:
        raise ContextBudgetInsufficientError("CONTEXT_BUDGET_INSUFFICIENT")
    return ContextBudgetAllocation(
        context_window=window,
        input_budget=input_budget,
        output_reserve=output_reserve,
        safety_margin=safety_margin,
    )


class ContextBudgetManager:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        policies: ContextPolicyRegistry | None = None,
    ) -> None:
        self._sessions = sessions
        self._policies = policies or ContextPolicyRegistry()

    async def build(
        self,
        *,
        run_id: UUID,
        node_name: str,
        provider_adapter: str,
        model: str,
        candidates: Sequence[ContextCandidate],
        requested_output_tokens: int,
        context_window: int | None,
        provider_max_output_tokens: int | None,
        prompt_template_version: str,
    ) -> ContextEnvelope:
        allocation = allocate_budget(
            context_window=context_window,
            requested_output_tokens=requested_output_tokens,
            provider_max_output_tokens=provider_max_output_tokens,
        )
        original = tuple(candidates)
        measured = [(candidate, estimate_tokens(candidate.content)) for candidate in original]
        protected_tokens = sum(tokens for candidate, tokens in measured if candidate.protected)
        if protected_tokens > allocation.input_budget:
            raise ContextBudgetInsufficientError("protected context exceeds input budget")

        policy = self._policies.resolve(node_name)
        ordered_indices, policy_pruned = mmr_order(original, policy)
        selected_by_index: dict[int, ContextCandidate] = {}
        token_by_index: dict[int, int] = {}
        compression_by_index: dict[int, CompressionResult] = {}
        used = 0
        for index in ordered_indices:
            if index in policy_pruned:
                continue
            candidate, tokens = measured[index]
            if used + tokens <= allocation.input_budget:
                selected_by_index[index] = candidate
                token_by_index[index] = tokens
                used += tokens
                continue
            if candidate.protected or candidate.item_type not in policy.compressible_types:
                continue
            compressed = compress_candidate(
                candidate,
                target_tokens=allocation.input_budget - used,
            )
            if compressed is None:
                continue
            selected_by_index[index] = compressed.candidate
            token_by_index[index] = compressed.token_after
            compression_by_index[index] = compressed
            used += compressed.token_after

        selected = tuple(
            selected_by_index[index] for index in range(len(original)) if index in selected_by_index
        )
        rejected = tuple(
            original[index] for index in range(len(original)) if index not in selected_by_index
        )
        rendered = "\n\n".join(candidate.content for candidate in selected)
        rendered_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        manifest_id = uuid7()
        token_before = sum(tokens for _, tokens in measured)

        try:
            async with self._sessions() as session, session.begin():
                session.add(
                    ContextManifestRow(
                        id=manifest_id,
                        run_id=run_id,
                        node_name=node_name,
                        provider_adapter=provider_adapter,
                        model=model,
                        context_window=allocation.context_window,
                        input_budget=allocation.input_budget,
                        output_reserve=allocation.output_reserve,
                        safety_margin=allocation.safety_margin,
                        selected_count=len(selected),
                        rejected_count=len(rejected),
                        protected_count=sum(1 for candidate in selected if candidate.protected),
                        compressed_count=len(compression_by_index),
                        token_before=token_before,
                        token_after=used,
                        truncated=bool(rejected) or bool(compression_by_index),
                        rendered_prompt_hash=rendered_hash,
                        prompt_template_version=prompt_template_version,
                    )
                )
                compression_ids: dict[int, UUID] = {}
                for index, result in compression_by_index.items():
                    artifact_id = uuid7()
                    compression_ids[index] = artifact_id
                    session.add(
                        CompressionArtifactRow(
                            id=artifact_id,
                            context_manifest_id=manifest_id,
                            input_hash=result.input_hash,
                            output_hash=result.output_hash,
                            compression_level=result.candidate.compression_level.value,
                            token_before=result.token_before,
                            token_after=result.token_after,
                            validation_status=result.validation_status,
                            provenance_refs=list(result.candidate.provenance_refs),
                        )
                    )
                for ordinal, (candidate, original_tokens) in enumerate(measured):
                    effective = selected_by_index.get(ordinal, candidate)
                    is_selected = ordinal in selected_by_index
                    if is_selected:
                        reason = effective.selected_reason_code
                    elif ordinal in policy_pruned:
                        reason = "context_policy_pruned"
                    else:
                        reason = "context_budget_pruned"
                    session.add(
                        ContextItemRow(
                            id=uuid7(),
                            context_manifest_id=manifest_id,
                            ordinal=ordinal,
                            item_type=candidate.item_type.value,
                            source_ref_type=candidate.source_ref_type,
                            source_ref_id=candidate.source_ref_id,
                            rank_score=candidate.rank_score,
                            token_count=token_by_index.get(ordinal, original_tokens),
                            compression_level=effective.compression_level.value,
                            content_hash=hashlib.sha256(effective.content.encode("utf-8")).hexdigest(),
                            selected=is_selected,
                            protected=candidate.protected,
                            selected_reason_code=reason,
                            compression_artifact_id=compression_ids.get(ordinal),
                        )
                    )
        except (DataError, IntegrityError) as exc:
            raise ContextManifestPersistenceError from exc

        return ContextEnvelope(
            manifest_id=manifest_id,
            allocation=allocation,
            selected=selected,
            rejected=rejected,
            token_before=token_before,
            token_after=used,
            rendered_prompt_hash=rendered_hash,
        )

    async def list_metrics(self, run_id: UUID) -> list[ContextManifestView]:
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(ContextManifestRow)
                    .options(
                        selectinload(ContextManifestRow.items),
                        selectinload(ContextManifestRow.compression_artifacts),
                    )
                    .where(ContextManifestRow.run_id == run_id)
                    .order_by(ContextManifestRow.created_at, ContextManifestRow.id)
                )
            ).all()
        return [self._view(row) for row in rows]

    @staticmethod
    def _view(row: ContextManifestRow) -> ContextManifestView:
        ratio = 1.0 if row.token_before == 0 else row.token_after / row.token_before
        return ContextManifestView(
            manifest_id=row.id,
            run_id=row.run_id,
            node_name=row.node_name,
            provider_adapter=row.provider_adapter,
            model=row.model,
            context_window=row.context_window,
            input_budget=row.input_budget,
            output_reserve=row.output_reserve,
            safety_margin=row.safety_margin,
            selected_count=row.selected_count,
            rejected_count=row.rejected_count,
            protected_count=row.protected_count,
            compressed_count=row.compressed_count,
            token_before=row.token_before,
            token_after=row.token_after,
            compression_ratio=round(ratio, 4),
            truncated=row.truncated,
            rendered_prompt_hash=row.rendered_prompt_hash,
            prompt_template_version=row.prompt_template_version,
            created_at=row.created_at,
            items=[
                ContextItemMetric(
                    item_type=item.item_type,
                    source_ref_type=item.source_ref_type,
                    source_ref_id=item.source_ref_id,
                    rank_score=item.rank_score,
                    token_count=item.token_count,
                    compression_level=item.compression_level,
                    selected=item.selected,
                    protected=item.protected,
                    selected_reason_code=item.selected_reason_code,
                    content_hash=item.content_hash,
                    compression_artifact_id=item.compression_artifact_id,
                )
                for item in row.items
            ],
            compression_artifacts=[
                CompressionArtifactMetric(
                    artifact_id=artifact.id,
                    input_hash=artifact.input_hash,
                    output_hash=artifact.output_hash,
                    compression_level=artifact.compression_level,
                    token_before=artifact.token_before,
                    token_after=artifact.token_after,
                    validation_status=artifact.validation_status,
                    provenance_refs=tuple(artifact.provenance_refs),
                )
                for artifact in row.compression_artifacts
            ],
        )
