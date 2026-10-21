from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID, uuid4

from app.domain.claim_architecture import CanonicalClaim, CanonicalClaimType, ClaimRelationType
from app.domain.providers import (
    AdapterType,
    CanonicalModelResult,
    TokenUsage,
    UsageAccuracy,
)
from app.domain.semantic_claim_equivalence import SemanticMatchReason
from app.services.semantic_claim_fallback import (
    SEMANTIC_FALLBACK_PROMPT_VERSION,
    SemanticClaimFallback,
)
from pydantic import SecretStr


def _claim(text: str, *, qualifiers: dict[str, str] | None = None) -> CanonicalClaim:
    return CanonicalClaim.from_text(
        claim_id=uuid4(),
        run_id=UUID("00000000-0000-0000-0000-000000000001"),
        question_id="q1",
        dimension_key="q1:d1",
        original_text=text,
        claim_type=CanonicalClaimType.UNKNOWN,
        qualifiers=qualifiers or {},
    )


class _FakeGateway:
    def __init__(self, response: dict[str, Any] | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def generate_structured(self, **kwargs: Any) -> CanonicalModelResult:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return CanonicalModelResult(
            parsed_object=self.response,
            usage=TokenUsage(accuracy=UsageAccuracy.ESTIMATED),
        )


def _fallback(gateway: _FakeGateway, *, max_calls: int = 2) -> SemanticClaimFallback:
    return SemanticClaimFallback(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://model.example/v1",
        api_key=SecretStr("test-only"),
        model="test-model",
        max_calls=max_calls,
    )


def test_generic_fallback_can_classify_raw_text_paraphrase_without_domain_rules() -> None:
    gateway = _FakeGateway(
        {
            "relation": "same",
            "confidence": 0.97,
            "rationale": "The claims express the same founding event and year.",
            "reason_codes": [SemanticMatchReason.SEMANTIC_PARAPHRASE.value],
        }
    )
    fallback = _fallback(gateway)

    result = asyncio.run(
        fallback.compare(
            _claim("The organization was established in 2018."),
            _claim("The organization was founded in 2018."),
        )
    )

    assert result.relation is ClaimRelationType.SAME
    assert result.method == f"llm_fallback:{SEMANTIC_FALLBACK_PROMPT_VERSION}"
    assert fallback.calls == 1
    assert gateway.calls[0]["allow_regeneration"] is False
    request = gateway.calls[0]["request"]
    assert request.generation_parameters["temperature"] == 0.0
    assert request.response_contract is not None
    assert fallback.input_tokens == 0
    assert fallback.output_tokens == 0


def test_known_time_conflict_never_reaches_semantic_fallback() -> None:
    gateway = _FakeGateway(
        {
            "relation": "same",
            "confidence": 1.0,
            "rationale": "unsafe",
            "reason_codes": [SemanticMatchReason.SEMANTIC_PARAPHRASE.value],
        }
    )
    fallback = _fallback(gateway)

    result = asyncio.run(
        fallback.compare(
            _claim("Revenue was $20B.", qualifiers={"time": "2024"}),
            _claim("Revenue was $20B.", qualifiers={"time": "2025"}),
        )
    )

    assert result.relation is not ClaimRelationType.SAME
    assert gateway.calls == []
    assert fallback.calls == 0


def test_approximate_vs_exact_numeric_conflict_blocks_same_fallback() -> None:
    gateway = _FakeGateway(
        {
            "relation": "same",
            "confidence": 0.99,
            "rationale": "unsafe approximation collapse",
            "reason_codes": [SemanticMatchReason.SEMANTIC_PARAPHRASE.value],
        }
    )
    fallback = _fallback(gateway)
    left = CanonicalClaim.from_text(
        claim_id=uuid4(), run_id=UUID("00000000-0000-0000-0000-000000000001"),
        question_id="q1", dimension_key="q1:d1", original_text="About 20 billion.",
        claim_type=CanonicalClaimType.QUANTITATIVE, subject="revenue", predicate="amount",
        object_or_value="about 20 billion", qualifiers={"unit": "USD"},
    )
    right = CanonicalClaim.from_text(
        claim_id=uuid4(), run_id=left.run_id, question_id="q1", dimension_key="q1:d1",
        original_text="20 billion.", claim_type=CanonicalClaimType.QUANTITATIVE,
        subject="revenue", predicate="amount", object_or_value="20 billion",
        qualifiers={"unit": "USD"},
    )

    result = asyncio.run(fallback.compare(left, right))

    assert result.relation is ClaimRelationType.UNKNOWN
    assert gateway.calls == []


def test_invalid_structured_output_fails_closed_to_unknown() -> None:
    gateway = _FakeGateway({"relation": "not-a-relation"})
    fallback = _fallback(gateway)

    result = asyncio.run(
        fallback.compare(_claim("A claim with no structure."), _claim("A different claim."))
    )

    assert result.relation is ClaimRelationType.UNKNOWN
    assert result.reason_codes == (SemanticMatchReason.SEMANTIC_MODEL_UNCERTAIN,)
    assert fallback.calls == 1


def test_gateway_failure_fails_closed_to_unknown() -> None:
    fallback = _fallback(_FakeGateway(error=TimeoutError("test timeout")))

    result = asyncio.run(
        fallback.compare(_claim("Claim A."), _claim("Claim B."))
    )

    assert result.relation is ClaimRelationType.UNKNOWN
    assert result.method == f"fallback_unknown:{SEMANTIC_FALLBACK_PROMPT_VERSION}"


def test_fallback_call_budget_is_bounded() -> None:
    gateway = _FakeGateway(
        {
            "relation": "unknown",
            "confidence": 0.2,
            "rationale": "Not enough context.",
            "reason_codes": [SemanticMatchReason.INSUFFICIENT_INFORMATION.value],
        }
    )
    fallback = _fallback(gateway, max_calls=1)
    left = _claim("Claim A.")
    right = _claim("Claim B.")

    asyncio.run(fallback.compare(left, right))
    asyncio.run(fallback.compare(left, right))

    assert fallback.calls == 1
    assert len(gateway.calls) == 1
