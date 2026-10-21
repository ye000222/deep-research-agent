"""Version and execution-mode contracts for Atomic Proposition decomposition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from app.domain.atomic_proposition import ATOMIC_PROPOSITION_SCHEMA_VERSION

ATOMIC_DECOMPOSITION_PROMPT_V2 = "atomic-decomposition.generic.v2"
ATOMIC_PROPOSITION_SCHEMA_V3 = "atomic-proposition-schema-v3"
ATOMIC_DECOMPOSITION_PROMPT_V3 = "atomic-decomposition.generic.v3"


class AtomicPropositionExecutionMode(StrEnum):
    PRODUCTION = "PRODUCTION"
    SHADOW = "SHADOW"


@dataclass(frozen=True)
class AtomicPropositionContract:
    schema_version: str
    prompt_version: str
    production_allowed: bool
    shadow_allowed: bool


@dataclass(frozen=True)
class ContractResolution:
    requested_schema_version: str | None
    requested_prompt_version: str | None
    execution_mode: AtomicPropositionExecutionMode | str
    status: str
    reason: str | None
    contract: AtomicPropositionContract | None

    def safe_dict(self) -> dict[str, str | bool | None]:
        return {
            "requested_schema_version": self.requested_schema_version,
            "requested_prompt_version": self.requested_prompt_version,
            "resolved_schema_version": self.contract.schema_version if self.contract else None,
            "resolved_prompt_version": self.contract.prompt_version if self.contract else None,
            "execution_mode": (
                self.execution_mode.value
                if isinstance(self.execution_mode, AtomicPropositionExecutionMode)
                else self.execution_mode
            ),
            "contract_resolution_status": self.status,
            "contract_resolution_reason": self.reason,
            "production_allowed": self.contract.production_allowed if self.contract else None,
            "shadow_allowed": self.contract.shadow_allowed if self.contract else None,
        }


ATOMIC_PROPOSITION_CONTRACTS: Mapping[str, AtomicPropositionContract] = MappingProxyType(
    {
        ATOMIC_PROPOSITION_SCHEMA_VERSION: AtomicPropositionContract(
            schema_version=ATOMIC_PROPOSITION_SCHEMA_VERSION,
            prompt_version=ATOMIC_DECOMPOSITION_PROMPT_V2,
            production_allowed=True,
            shadow_allowed=True,
        ),
        ATOMIC_PROPOSITION_SCHEMA_V3: AtomicPropositionContract(
            schema_version=ATOMIC_PROPOSITION_SCHEMA_V3,
            prompt_version=ATOMIC_DECOMPOSITION_PROMPT_V3,
            production_allowed=False,
            shadow_allowed=True,
        ),
    }
)


def resolve_atomic_proposition_contract(
    *,
    schema_version: str | None = None,
    prompt_version: str | None = None,
    execution_mode: AtomicPropositionExecutionMode | str = (
        AtomicPropositionExecutionMode.PRODUCTION
    ),
) -> ContractResolution:
    """Resolve one supported schema/prompt pair, failing closed on ambiguity."""

    requested_schema = schema_version
    requested_prompt = prompt_version
    try:
        resolved_mode = AtomicPropositionExecutionMode(execution_mode)
    except (TypeError, ValueError):
        return ContractResolution(
            requested_schema,
            requested_prompt,
            str(execution_mode),
            "BLOCKED",
            "unsupported_execution_mode",
            None,
        )
    resolved_schema = schema_version or ATOMIC_PROPOSITION_SCHEMA_VERSION
    contract = ATOMIC_PROPOSITION_CONTRACTS.get(resolved_schema)
    if contract is None:
        return ContractResolution(
            requested_schema,
            requested_prompt,
            resolved_mode,
            "BLOCKED",
            "unsupported_schema_version",
            None,
        )
    if prompt_version is not None and prompt_version not in {
        item.prompt_version for item in ATOMIC_PROPOSITION_CONTRACTS.values()
    }:
        return ContractResolution(
            requested_schema,
            requested_prompt,
            resolved_mode,
            "BLOCKED",
            "unsupported_prompt_version",
            None,
        )
    if prompt_version is not None and prompt_version != contract.prompt_version:
        return ContractResolution(
            requested_schema,
            requested_prompt,
            resolved_mode,
            "BLOCKED",
            "schema_prompt_pair_mismatch",
            None,
        )
    mode_allowed = (
        contract.production_allowed
        if resolved_mode is AtomicPropositionExecutionMode.PRODUCTION
        else contract.shadow_allowed
    )
    if not mode_allowed:
        return ContractResolution(
            requested_schema,
            requested_prompt,
            resolved_mode,
            "BLOCKED",
            "execution_mode_not_allowed",
            contract,
        )
    return ContractResolution(
        requested_schema,
        requested_prompt,
        resolved_mode,
        "RESOLVED",
        None,
        contract,
    )
