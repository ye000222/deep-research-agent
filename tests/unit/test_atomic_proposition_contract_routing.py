from __future__ import annotations

import asyncio
from typing import Any

from app.domain.atomic_proposition import ATOMIC_PROPOSITION_SCHEMA_VERSION
from app.domain.atomic_proposition_contracts import (
    ATOMIC_DECOMPOSITION_PROMPT_V2,
    ATOMIC_DECOMPOSITION_PROMPT_V3,
    ATOMIC_PROPOSITION_SCHEMA_V3,
    AtomicPropositionExecutionMode,
    resolve_atomic_proposition_contract,
)
from app.domain.providers import AdapterType


def test_omitted_contract_resolves_to_v2_production_default() -> None:
    resolution = resolve_atomic_proposition_contract()

    assert resolution.status == "RESOLVED"
    assert resolution.contract is not None
    assert resolution.contract.schema_version == ATOMIC_PROPOSITION_SCHEMA_VERSION
    assert resolution.contract.prompt_version == ATOMIC_DECOMPOSITION_PROMPT_V2
    assert resolution.execution_mode is AtomicPropositionExecutionMode.PRODUCTION


def test_explicit_v2_shadow_pair_is_supported() -> None:
    resolution = resolve_atomic_proposition_contract(
        schema_version=ATOMIC_PROPOSITION_SCHEMA_VERSION,
        prompt_version=ATOMIC_DECOMPOSITION_PROMPT_V2,
        execution_mode=AtomicPropositionExecutionMode.SHADOW,
    )

    assert resolution.status == "RESOLVED"
    assert resolution.contract is not None
    assert resolution.contract.schema_version == ATOMIC_PROPOSITION_SCHEMA_VERSION


def test_explicit_v3_shadow_pair_is_supported() -> None:
    resolution = resolve_atomic_proposition_contract(
        schema_version=ATOMIC_PROPOSITION_SCHEMA_V3,
        prompt_version=ATOMIC_DECOMPOSITION_PROMPT_V3,
        execution_mode=AtomicPropositionExecutionMode.SHADOW,
    )

    assert resolution.status == "RESOLVED"
    assert resolution.contract is not None
    assert resolution.contract.schema_version == ATOMIC_PROPOSITION_SCHEMA_V3
    assert resolution.contract.prompt_version == ATOMIC_DECOMPOSITION_PROMPT_V3
    assert resolution.contract.production_allowed is False


def test_v3_production_mode_is_blocked() -> None:
    resolution = resolve_atomic_proposition_contract(
        schema_version=ATOMIC_PROPOSITION_SCHEMA_V3,
        execution_mode=AtomicPropositionExecutionMode.PRODUCTION,
    )

    assert resolution.status == "BLOCKED"
    assert resolution.reason == "execution_mode_not_allowed"


def test_v2_schema_with_v3_prompt_is_blocked() -> None:
    resolution = resolve_atomic_proposition_contract(
        schema_version=ATOMIC_PROPOSITION_SCHEMA_VERSION,
        prompt_version=ATOMIC_DECOMPOSITION_PROMPT_V3,
        execution_mode=AtomicPropositionExecutionMode.SHADOW,
    )

    assert resolution.status == "BLOCKED"
    assert resolution.reason == "schema_prompt_pair_mismatch"


def test_v3_schema_with_v2_prompt_is_blocked() -> None:
    resolution = resolve_atomic_proposition_contract(
        schema_version=ATOMIC_PROPOSITION_SCHEMA_V3,
        prompt_version=ATOMIC_DECOMPOSITION_PROMPT_V2,
        execution_mode=AtomicPropositionExecutionMode.SHADOW,
    )

    assert resolution.status == "BLOCKED"
    assert resolution.reason == "schema_prompt_pair_mismatch"


def test_unknown_schema_is_blocked_without_fallback() -> None:
    resolution = resolve_atomic_proposition_contract(
        schema_version="atomic-proposition-schema-v999",
        execution_mode=AtomicPropositionExecutionMode.SHADOW,
    )

    assert resolution.status == "BLOCKED"
    assert resolution.reason == "unsupported_schema_version"
    assert resolution.contract is None


def test_unknown_execution_mode_is_blocked() -> None:
    resolution = resolve_atomic_proposition_contract(
        schema_version=ATOMIC_PROPOSITION_SCHEMA_V3,
        execution_mode="CANARYISH",
    )

    assert resolution.status == "BLOCKED"
    assert resolution.reason == "unsupported_execution_mode"
    assert resolution.safe_dict()["execution_mode"] == "CANARYISH"


def test_unknown_prompt_is_blocked() -> None:
    resolution = resolve_atomic_proposition_contract(
        schema_version=ATOMIC_PROPOSITION_SCHEMA_VERSION,
        prompt_version="atomic-decomposition.generic.v999",
        execution_mode=AtomicPropositionExecutionMode.SHADOW,
    )

    assert resolution.status == "BLOCKED"
    assert resolution.reason == "unsupported_prompt_version"
    assert resolution.contract is None


def test_omitted_versions_remain_v2_for_historical_production_callers() -> None:
    default_resolution = resolve_atomic_proposition_contract(
        execution_mode=AtomicPropositionExecutionMode.PRODUCTION,
    )
    legacy_resolution = resolve_atomic_proposition_contract(
        schema_version=ATOMIC_PROPOSITION_SCHEMA_VERSION,
        execution_mode=AtomicPropositionExecutionMode.PRODUCTION,
    )

    assert default_resolution.contract == legacy_resolution.contract
    assert default_resolution.contract is not None
    assert default_resolution.contract.schema_version == "atomic-proposition-schema-v2"


def test_decomposer_rejects_v3_in_default_production_mode_without_gateway_call() -> None:
    from app.services.atomic_proposition_decomposer import AtomicPropositionDecomposer
    from pydantic import SecretStr

    class NoCallGateway:
        calls: list[dict[str, Any]]

        def __init__(self) -> None:
            self.calls = []

        async def generate_structured(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            raise AssertionError("blocked contract must not reach the Gateway")

    gateway = NoCallGateway()
    decomposer = AtomicPropositionDecomposer(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://not-contacted.invalid/v1",
        api_key=SecretStr("synthetic-only"),
        model="synthetic-model",
    )
    result = asyncio.run(
        decomposer.decompose_claim(
            "Synthetic assertion.", schema_version=ATOMIC_PROPOSITION_SCHEMA_V3
        )
    )

    assert result.status.value == "SCHEMA_INVALID"
    assert result.validation_issues[0].code == "execution_mode_not_allowed"
    assert result.model_call_count == 0
    assert gateway.calls == []
