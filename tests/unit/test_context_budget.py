from uuid import uuid4

import pytest
from app.context.compressor import compress_candidate
from app.context.manager import (
    ContextBudgetInsufficientError,
    allocate_budget,
    estimate_tokens,
)
from app.context.policy import ContextPolicyRegistry
from app.context.ranker import mmr_order
from app.domain.context import ContextCandidate, ContextItemType


def test_budget_reserves_output_and_never_exceeds_seventy_percent() -> None:
    budget = allocate_budget(
        context_window=10_000,
        requested_output_tokens=2_000,
        provider_max_output_tokens=4_000,
    )
    assert budget.input_budget == 7_000
    assert budget.output_reserve == 2_000
    assert budget.safety_margin == 1_000


def test_budget_rejects_window_that_cannot_protect_minimum_context() -> None:
    with pytest.raises(ContextBudgetInsufficientError):
        allocate_budget(
            context_window=1_024,
            requested_output_tokens=600,
            provider_max_output_tokens=600,
        )


def test_token_estimate_is_deterministic_and_nonzero() -> None:
    assert estimate_tokens("工业视觉 defect 2026") == estimate_tokens("工业视觉 defect 2026")
    assert estimate_tokens("x") == 1
    assert estimate_tokens("") == 0


def test_context_candidate_contract_keeps_exact_content() -> None:
    evidence_id = uuid4()
    quote = "Exact quote 24.9% must not be rewritten."
    candidate = ContextCandidate(
        item_type=ContextItemType.EVIDENCE_CARD,
        content=quote,
        rank_score=0.9,
        source_ref_type="evidence",
        source_ref_id=str(evidence_id),
    )
    assert candidate.content == quote
    assert candidate.source_ref_id == str(evidence_id)


def test_mmr_prunes_near_duplicate_context() -> None:
    candidates = (
        ContextCandidate(ContextItemType.EVIDENCE_CARD, "Cognex revenue was 100 in 2025.", 0.9),
        ContextCandidate(ContextItemType.EVIDENCE_CARD, "Cognex revenue was 100 in 2025.", 0.89),
        ContextCandidate(
            ContextItemType.EVIDENCE_CARD, "Keyence launched a new vision sensor.", 0.8
        ),
    )
    order, pruned = mmr_order(candidates, ContextPolicyRegistry().resolve("report_writer"))
    assert 0 in order
    assert 1 in pruned
    assert 2 in order


def test_compressor_preserves_numeric_sentence_and_provenance() -> None:
    candidate = ContextCandidate(
        ContextItemType.SOURCE_CHUNK,
        "General introduction. Revenue grew 24.9% in 2025. " * 20,
        0.8,
        source_ref_type="source_chunk",
        source_ref_id="chunk-1",
    )
    result = compress_candidate(candidate, target_tokens=30)
    assert result is not None
    assert result.token_after <= 30
    assert "24.9%" in result.candidate.content
    assert "chunk-1" in result.candidate.provenance_refs
    assert result.validation_status == "provenance_preserved"
