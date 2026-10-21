"""Small shadow-only facade; intentionally absent from production call paths."""

from __future__ import annotations

from collections.abc import Mapping

from app.domain.atomic_proposition import ATOMIC_PROPOSITION_SCHEMA_VERSION
from app.domain.atomic_proposition_contracts import AtomicPropositionExecutionMode
from app.services.atomic_proposition_decomposer import (
    AtomicPropositionDecomposer,
    ShadowDecompositionResult,
)


class AtomicPropositionShadow:
    def __init__(self, decomposer: AtomicPropositionDecomposer) -> None:
        self._decomposer = decomposer

    async def evaluate(
        self,
        claim_text: str,
        context_metadata: Mapping[str, object] | None = None,
        *,
        claim_id: str | None = None,
        schema_version: str = ATOMIC_PROPOSITION_SCHEMA_VERSION,
        prompt_version: str | None = None,
    ) -> ShadowDecompositionResult:
        return await self._decomposer.decompose_claim(
            claim_text,
            context_metadata,
            claim_id=claim_id,
            schema_version=schema_version,
            prompt_version=prompt_version,
            execution_mode=AtomicPropositionExecutionMode.SHADOW,
        )
