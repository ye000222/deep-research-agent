"""Resolve the frozen PATH-C seed amendment without performing generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

SeedCapability = Literal[
    "SEED_SUPPORTED",
    "SEED_SUPPORT_NOT_ESTABLISHED",
    "SEED_UNSUPPORTED",
]

_ORIGINAL_SEED_BEHAVIOR = "BLOCK_BEFORE_PROVIDER_REQUEST"


class SeedAmendmentIntegrityError(ValueError):
    """Raised when the frozen base or amendment does not match its contract."""


@dataclass(frozen=True)
class SeedResolution:
    effective_contract: dict[str, object]
    capability: SeedCapability
    send_seed: bool
    seed_value: str | None
    record_seed_sent_on_dispatch: bool
    deterministic_generation_claim: bool
    provider_dispatch_authorized: Literal[False] = False


def _read_object(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SeedAmendmentIntegrityError("contract_artifact_unreadable") from exc
    if not isinstance(value, dict):
        raise SeedAmendmentIntegrityError("contract_artifact_not_object")
    return cast(dict[str, object], value)


def resolve_seed_amendment(
    base_contract_path: Path,
    amendment_path: Path,
    capability: SeedCapability,
) -> SeedResolution:
    """Apply only the seed-capability clause after verifying the frozen base hash.

    This resolver is a pure local contract operation. It has no Provider,
    network, persistence, task-generation, or acquisition dependency.
    """
    if capability not in {
        "SEED_SUPPORTED",
        "SEED_SUPPORT_NOT_ESTABLISHED",
        "SEED_UNSUPPORTED",
    }:
        raise SeedAmendmentIntegrityError("unknown_seed_capability")

    amendment = _read_object(amendment_path)
    base_bytes = base_contract_path.read_bytes()
    base_hash = hashlib.sha256(base_bytes).hexdigest()
    base = _read_object(base_contract_path)

    manifest = _read_object(amendment_path.parent / "hash_manifest.json")
    manifest_files = manifest.get("files")
    if not isinstance(manifest_files, dict):
        raise SeedAmendmentIntegrityError("amendment_hash_manifest_invalid")
    expected_amendment_hash = manifest_files.get(amendment_path.name)
    if expected_amendment_hash != hashlib.sha256(amendment_path.read_bytes()).hexdigest():
        raise SeedAmendmentIntegrityError("amendment_artifact_hash_mismatch")

    base_reference = amendment.get("base_contract")
    supersedes = amendment.get("supersedes")
    if not isinstance(base_reference, dict) or not isinstance(supersedes, dict):
        raise SeedAmendmentIntegrityError("amendment_contract_shape_invalid")
    if base_reference.get("sha256") != base_hash:
        raise SeedAmendmentIntegrityError("base_contract_hash_mismatch")
    if manifest.get("base_contract_sha256") != base_hash:
        raise SeedAmendmentIntegrityError("base_contract_hash_mismatch")
    if supersedes.get("clause") != "unsupported_seed_behavior":
        raise SeedAmendmentIntegrityError("unexpected_superseded_clause")
    if supersedes.get("original_value") != _ORIGINAL_SEED_BEHAVIOR:
        raise SeedAmendmentIntegrityError("base_seed_behavior_mismatch")
    if base.get("unsupported_seed_behavior") != _ORIGINAL_SEED_BEHAVIOR:
        raise SeedAmendmentIntegrityError("base_seed_behavior_mismatch")
    if amendment.get("provider_dispatch_authorized") is not False:
        raise SeedAmendmentIntegrityError("amendment_must_not_authorize_dispatch")

    seed_value = amendment.get("seed_value")
    if not isinstance(seed_value, str) or not seed_value:
        raise SeedAmendmentIntegrityError("amendment_seed_value_missing")
    if not isinstance(base.get("seed"), str) or seed_value not in cast(str, base["seed"]):
        raise SeedAmendmentIntegrityError("amendment_seed_value_mismatch")

    policy = amendment.get("seed_capability_policy")
    if not isinstance(policy, dict) or not isinstance(policy.get(capability), dict):
        raise SeedAmendmentIntegrityError("amendment_seed_policy_missing")
    capability_policy = cast(dict[str, object], policy[capability])
    send_seed = capability_policy.get("send_seed")
    deterministic_claim = capability_policy.get("deterministic_generation_claim")
    if not isinstance(send_seed, bool) or deterministic_claim is not False:
        raise SeedAmendmentIntegrityError("amendment_seed_policy_invalid")
    if send_seed != (capability == "SEED_SUPPORTED"):
        raise SeedAmendmentIntegrityError("amendment_seed_policy_inconsistent")

    effective = dict(base)
    effective["unsupported_seed_behavior"] = {
        key: dict(cast(dict[str, object], policy[key]))
        for key in (
            "SEED_SUPPORTED",
            "SEED_SUPPORT_NOT_ESTABLISHED",
            "SEED_UNSUPPORTED",
        )
    }
    return SeedResolution(
        effective_contract=effective,
        capability=capability,
        send_seed=send_seed,
        seed_value=seed_value if send_seed else None,
        record_seed_sent_on_dispatch=send_seed,
        deterministic_generation_claim=False,
    )
