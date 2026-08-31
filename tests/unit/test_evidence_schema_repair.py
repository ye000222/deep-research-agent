from datetime import UTC, datetime
from uuid import uuid4

import pytest
from app.domain.providers import (
    AdapterType,
    CanonicalModelResult,
    TokenUsage,
    UsageAccuracy,
)
from app.domain.research_tools import ReadPage
from app.infrastructure.db.run_providers import RunProviderBinding
from app.security.secrets import SecretCipher
from app.services.evidence_extractor import EvidenceExtractorService
from pydantic import SecretStr


def result(payload: dict[str, object], tokens: int) -> CanonicalModelResult:
    return CanonicalModelResult(
        parsed_object=payload,
        usage=TokenUsage(
            input_tokens=tokens,
            output_tokens=tokens,
            total_tokens=tokens * 2,
            accuracy=UsageAccuracy.EXACT,
        ),
    )


class FakeBindingRepository:
    def __init__(self, binding: RunProviderBinding) -> None:
        self.binding = binding

    async def get(self, run_id: object) -> RunProviderBinding:
        return self.binding


class RepairGateway:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def generate_structured(self, **kwargs: object) -> CanonicalModelResult:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return result({"items": [{"claim": "too short"}]}, 10)
        return result(
            {
                "items": [
                    {
                        "claim": "The platform supports electronics inspection.",
                        "exact_quote": (
                            "The platform supports electronics inspection on production lines."
                        ),
                        "relation": "supports",
                        "relevance": 0.95,
                        "confidence": 0.9,
                    }
                ]
            },
            20,
        )


@pytest.mark.asyncio
async def test_extractor_repairs_schema_once_and_combines_usage() -> None:
    run_id = uuid4()
    credential_id = uuid4()
    cipher = SecretCipher(b"x" * 32)
    encrypted = cipher.encrypt(
        SecretStr("test-provider-key"),
        credential_id=credential_id,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT.value,
        credential_version=1,
    )
    binding = RunProviderBinding(
        run_id=run_id,
        goal="Research industrial inspection",
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://api.example.com",
        model="test-model",
        credential_id=credential_id,
        credential_version=1,
        encrypted_secret=encrypted,
    )
    page = ReadPage(
        final_url="https://vendor.example.com/product",
        title="Product",
        clean_text=(
            "The platform supports electronics inspection on production lines. "
            "This documentation describes deployment and traceability in detail."
        ),
        content_hash="a" * 64,
        fetched_at=datetime.now(UTC),
    )
    gateway = RepairGateway()
    service = EvidenceExtractorService(  # type: ignore[arg-type]
        FakeBindingRepository(binding), cipher, gateway
    )

    evidence, usage, _ = await service.extract(run_id, question="What is supported?", page=page)

    assert len(gateway.calls) == 2
    assert gateway.calls[1]["allow_regeneration"] is False
    repair_request = gateway.calls[1]["request"]
    assert repair_request.metadata["schema_repair"] == "1"  # type: ignore[union-attr]
    assert evidence[0].accepted is True
    assert usage.total_tokens == 60
