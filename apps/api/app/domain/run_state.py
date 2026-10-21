"""Canonical, reference-only V3 run and semantic state contracts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

RUN_STATE_SCHEMA = "research-run-state-v1"
SEMANTIC_STATE_SCHEMA = "semantic-state-snapshot-v1"
ORCHESTRATOR_V3 = "V3"


def canonical_hash(value: object) -> str:
    """Hash JSON-compatible semantic content with stable ordering."""

    def normalize(item: Any) -> Any:
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        if isinstance(item, UUID):
            return str(item)
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, datetime):
            return item.isoformat()
        if isinstance(item, dict):
            return {str(key): normalize(nested) for key, nested in item.items()}
        if isinstance(item, (tuple, list)):
            return [normalize(nested) for nested in item]
        return item

    payload = json.dumps(
        normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _with_model_defaults(model: type[BaseModel], values: dict[str, Any]) -> dict[str, Any]:
    """Include model defaults in the hashed representation before validation."""

    complete = dict(values)
    for name, field in model.model_fields.items():
        if name in complete or field.is_required():
            continue
        complete[name] = field.get_default(
            call_default_factory=True, validated_data=complete
        )
    return complete


class SemanticExecutionStatus(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    FAILED = "FAILED"


class GapProjectionStatus(StrEnum):
    DERIVED = "DERIVED"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    INCOMPLETE = "INCOMPLETE"
    CAPABILITY_BLOCKED = "CAPABILITY_BLOCKED"


class EvidenceUniverseEntry(BaseModel):
    """Stable identity of one persisted evidence item, without its quote/body."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: UUID
    claim_ref: UUID | None = None
    source_ref: UUID
    source_snapshot_ref: UUID | None = None
    source_content_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    accepted: bool
    semantic_eligible: bool


class EvidenceUniverse(BaseModel):
    """Exact full DB universe and its separately identified semantic subset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_ref: UUID
    entries: tuple[EvidenceUniverseEntry, ...]
    evidence_universe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_evidence_refs: tuple[UUID, ...]
    semantic_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = "evidence-universe-v1"

    @model_validator(mode="after")
    def validate_identity(self) -> EvidenceUniverse:
        entries = tuple(sorted(self.entries, key=lambda item: str(item.evidence_ref)))
        if entries != self.entries or len({item.evidence_ref for item in entries}) != len(entries):
            raise ValueError("evidence_universe_entries_not_canonical")
        expected_full = canonical_hash(
            [item.model_dump(mode="json") for item in entries]
        )
        expected_semantic_refs = tuple(
            item.evidence_ref for item in entries if item.semantic_eligible
        )
        expected_semantic = canonical_hash(
            [item.model_dump(mode="json") for item in entries if item.semantic_eligible]
        )
        if self.evidence_universe_hash != expected_full:
            raise ValueError("evidence_universe_hash_mismatch")
        if self.semantic_evidence_refs != expected_semantic_refs:
            raise ValueError("semantic_evidence_refs_mismatch")
        if self.semantic_evidence_hash != expected_semantic:
            raise ValueError("semantic_evidence_hash_mismatch")
        return self

    @classmethod
    def build(
        cls, run_ref: UUID, entries: tuple[EvidenceUniverseEntry, ...]
    ) -> EvidenceUniverse:
        unique: dict[UUID, EvidenceUniverseEntry] = {}
        for entry in entries:
            existing = unique.get(entry.evidence_ref)
            if existing is not None and existing != entry:
                raise ValueError("duplicate_evidence_identity_conflict")
            unique[entry.evidence_ref] = entry
        ordered = tuple(sorted(unique.values(), key=lambda item: str(item.evidence_ref)))
        semantic_refs = tuple(item.evidence_ref for item in ordered if item.semantic_eligible)
        return cls(
            run_ref=run_ref,
            entries=ordered,
            evidence_universe_hash=canonical_hash(
                [item.model_dump(mode="json") for item in ordered]
            ),
            semantic_evidence_refs=semantic_refs,
            semantic_evidence_hash=canonical_hash(
                [item.model_dump(mode="json") for item in ordered if item.semantic_eligible]
            ),
        )


class SemanticProductBinding(BaseModel):
    """Proof metadata that one immutable semantic product belongs to this run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    product_kind: str = Field(min_length=1, max_length=80)
    product_ref: str = Field(min_length=1, max_length=1024)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    product_version: str = Field(min_length=1, max_length=100)
    run_ref: UUID
    evidence_universe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_execution_ref: str = Field(min_length=1, max_length=256)
    semantic_engine_version: str = Field(min_length=1, max_length=100)


class SemanticStateSnapshot(BaseModel):
    """Immutable output envelope around formal V2 semantic products."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_ref: UUID
    # COMPLETE is scoped to the persisted universe at this boundary, not run lifecycle completion.
    state_scope: str = "RUN_CURRENT_EVIDENCE_UNIVERSE"
    semantic_execution_ref: str = Field(min_length=1, max_length=256)
    semantic_engine_version: str = Field(min_length=1, max_length=100)
    evidence_universe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_output_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget_state_hash: str = Field(
        default_factory=lambda: canonical_hash({}), pattern=r"^[0-9a-f]{64}$"
    )
    diagnostics_hash: str = Field(
        default_factory=lambda: canonical_hash([]), pattern=r"^[0-9a-f]{64}$"
    )
    diagnostic_codes: tuple[str, ...] = ()
    semantic_evidence_refs: tuple[UUID, ...]
    required_product_versions: tuple[tuple[str, str], ...]
    required_product_refs: tuple[tuple[str, str, str], ...]
    output_product_refs: tuple[tuple[str, str, str], ...]
    output_product_bindings: tuple[SemanticProductBinding, ...] = ()
    semantic_statuses: tuple[tuple[str, str], ...] = ()
    known_limitations: tuple[str, ...] = ()
    capability_states: tuple[str, ...] = ()
    deferred_states: tuple[str, ...] = ()
    provenance_refs: tuple[str, ...] = ()
    execution_status: SemanticExecutionStatus
    failure_codes: tuple[str, ...] = ()
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_ref: str = Field(min_length=1, max_length=1024)
    schema_version: str = SEMANTIC_STATE_SCHEMA

    @model_validator(mode="after")
    def validate_snapshot(self) -> SemanticStateSnapshot:
        if self.schema_version != SEMANTIC_STATE_SCHEMA:
            raise ValueError("unknown_semantic_state_schema")
        if self.state_scope != "RUN_CURRENT_EVIDENCE_UNIVERSE":
            raise ValueError("unknown_semantic_state_scope")
        expected_execution_ref = self.deterministic_execution_ref(
            run_ref=self.run_ref,
            semantic_engine_version=self.semantic_engine_version,
            evidence_universe_hash=self.evidence_universe_hash,
            semantic_evidence_hash=self.semantic_evidence_hash,
            required_product_versions=self.required_product_versions,
        )
        if self.semantic_execution_ref != expected_execution_ref:
            raise ValueError("semantic_execution_identity_not_canonical")
        if self.semantic_evidence_refs != tuple(
            sorted(set(self.semantic_evidence_refs), key=str)
        ):
            raise ValueError("semantic_evidence_refs_not_canonical")
        for field_name in (
            "required_product_versions",
            "required_product_refs",
            "output_product_refs",
            "semantic_statuses",
            "provenance_refs",
            "diagnostic_codes",
        ):
            value = getattr(self, field_name)
            if value != tuple(sorted(set(value))):
                raise ValueError(f"{field_name}_not_canonical")
        binding_keys = tuple(
            (item.product_kind, item.product_ref, item.content_hash)
            for item in self.output_product_bindings
        )
        if binding_keys != tuple(sorted(set(binding_keys))):
            raise ValueError("output_product_bindings_not_canonical")
        if self.known_limitations != tuple(sorted(set(self.known_limitations))):
            raise ValueError("known_limitations_not_canonical")
        if self.capability_states != tuple(sorted(set(self.capability_states))):
            raise ValueError("capability_states_not_canonical")
        if self.deferred_states != tuple(sorted(set(self.deferred_states))):
            raise ValueError("deferred_states_not_canonical")
        if self.failure_codes != tuple(sorted(set(self.failure_codes))):
            raise ValueError("failure_codes_not_canonical")
        expected_outputs = {
            (kind, ref, digest) for kind, ref, digest in self.output_product_refs
        }
        expected_required = {
            (kind, ref, digest) for kind, ref, digest in self.required_product_refs
        }
        versions = dict(self.required_product_versions)
        if len(versions) != len(self.required_product_versions):
            raise ValueError("duplicate_required_product_kind")
        required_ref_kinds = {kind for kind, _, _ in self.required_product_refs}
        if required_ref_kinds != set(versions):
            raise ValueError("required_product_version_reference_mismatch")
        for binding in self.output_product_bindings:
            if (
                binding.run_ref != self.run_ref
                or binding.evidence_universe_hash != self.evidence_universe_hash
                or binding.semantic_execution_ref != self.semantic_execution_ref
                or binding.semantic_engine_version != self.semantic_engine_version
                or versions.get(binding.product_kind) != binding.product_version
            ):
                raise ValueError("semantic_product_binding_mismatch")
        bound_outputs = {
            (item.product_kind, item.product_ref, item.content_hash)
            for item in self.output_product_bindings
        }
        if expected_outputs != bound_outputs:
            raise ValueError("semantic_output_product_binding_incomplete")
        if self.semantic_output_hash != canonical_hash(
            [item.model_dump(mode="json") for item in self.output_product_bindings]
        ):
            raise ValueError("semantic_output_hash_mismatch")
        if self.execution_status is SemanticExecutionStatus.COMPLETE:
            if expected_required - expected_outputs:
                raise ValueError("required_semantic_product_missing")
            if self.failure_codes:
                raise ValueError("complete_semantic_snapshot_has_failures")
        if self.execution_status is SemanticExecutionStatus.FAILED and not self.failure_codes:
            raise ValueError("failed_semantic_snapshot_requires_failure_code")
        if self.execution_status is SemanticExecutionStatus.INCOMPLETE and not (
            expected_required - expected_outputs or self.deferred_states or self.known_limitations
        ):
            raise ValueError("incomplete_semantic_snapshot_requires_limitation")
        if self.snapshot_hash != self.compute_hash(self.model_dump(mode="json")):
            raise ValueError("semantic_snapshot_hash_mismatch")
        return self

    @staticmethod
    def compute_hash(payload: dict[str, Any]) -> str:
        excluded = {"snapshot_hash", "artifact_ref"}
        return canonical_hash({key: value for key, value in payload.items() if key not in excluded})

    @classmethod
    def build(cls, **values: Any) -> SemanticStateSnapshot:
        values = _with_model_defaults(cls, values)
        values.setdefault("schema_version", SEMANTIC_STATE_SCHEMA)
        values.setdefault(
            "semantic_execution_ref",
            cls.deterministic_execution_ref(
                run_ref=values["run_ref"],
                semantic_engine_version=values["semantic_engine_version"],
                evidence_universe_hash=values["evidence_universe_hash"],
                semantic_evidence_hash=values["semantic_evidence_hash"],
                required_product_versions=values["required_product_versions"],
            ),
        )
        values["snapshot_hash"] = cls.compute_hash({**values, "snapshot_hash": "0" * 64})
        values["artifact_ref"] = semantic_state_artifact_ref(
            values["run_ref"], values["snapshot_hash"]
        )
        return cls.model_validate(values)

    @staticmethod
    def deterministic_execution_ref(
        *,
        run_ref: UUID,
        semantic_engine_version: str,
        evidence_universe_hash: str,
        semantic_evidence_hash: str,
        required_product_versions: tuple[tuple[str, str], ...],
    ) -> str:
        identity = canonical_hash(
            {
                "run_ref": str(run_ref),
                "semantic_engine_version": semantic_engine_version,
                "evidence_universe_hash": evidence_universe_hash,
                "semantic_evidence_hash": semantic_evidence_hash,
                "required_product_versions": sorted(required_product_versions),
            }
        )
        return f"semantic-execution:{identity}"


class ResearchRunState(BaseModel):
    """Canonical run-level projection; contains references, never duplicated truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_ref: UUID
    orchestrator_version: str = ORCHESTRATOR_V3
    state_version: int = Field(ge=1)
    state_ref: str = Field(min_length=1, max_length=1024)
    parent_state_ref: str | None = None
    parent_state_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expected_parent_version: int | None = Field(default=None, ge=1)
    lifecycle_status: str = Field(min_length=1, max_length=40)
    phase: str = Field(min_length=1, max_length=40)
    evidence_universe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    semantic_state_ref: str | None = None
    semantic_state_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    semantic_engine_version: str | None = None
    semantic_schema_version: str | None = None
    semantic_execution_status: SemanticExecutionStatus
    gap_projection_status: GapProjectionStatus
    gap_refs: tuple[str, ...] = ()
    gap_source_semantic_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    gap_projection_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    intent_refs: tuple[str, ...] = ()
    intent_source_gap_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    budget_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    usage_state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    known_limitations: tuple[str, ...] = ()
    capability_states: tuple[str, ...] = ()
    deferred_states: tuple[str, ...] = ()
    budget_projection: tuple[tuple[str, int | str | bool | None], ...] = ()
    termination_reason: str | None = None
    execution_refs: tuple[tuple[str, str], ...] = ()
    state_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    schema_version: str = RUN_STATE_SCHEMA

    @model_validator(mode="after")
    def validate_state(self) -> ResearchRunState:
        if self.schema_version != RUN_STATE_SCHEMA or self.orchestrator_version != ORCHESTRATOR_V3:
            raise ValueError("unknown_research_run_state_version")
        if (self.parent_state_ref is None) != (self.parent_state_hash is None):
            raise ValueError("parent_state_ref_hash_mismatch")
        if self.parent_state_ref is None:
            if self.expected_parent_version is not None or self.state_version != 1:
                raise ValueError("invalid_genesis_state_lineage")
        elif self.expected_parent_version != self.state_version - 1:
            raise ValueError("state_parent_version_mismatch")
        if (self.semantic_state_ref is None) != (self.semantic_state_hash is None):
            raise ValueError("semantic_state_ref_hash_mismatch")
        if self.semantic_execution_status is SemanticExecutionStatus.COMPLETE:
            if self.semantic_state_ref is None:
                raise ValueError("complete_state_requires_semantic_snapshot")
            if self.semantic_engine_version is None or self.semantic_schema_version is None:
                raise ValueError("complete_state_requires_semantic_contract_identity")
        elif self.semantic_state_ref is not None:
            raise ValueError("non_complete_semantic_state_cannot_be_authoritative")
        for name in (
            "gap_refs",
            "intent_refs",
            "known_limitations",
            "capability_states",
            "deferred_states",
        ):
            value = getattr(self, name)
            if value != tuple(sorted(set(value))):
                raise ValueError(f"{name}_not_canonical")
        if self.budget_projection != tuple(sorted(set(self.budget_projection))):
            raise ValueError("budget_projection_not_canonical")
        if self.execution_refs != tuple(sorted(set(self.execution_refs))):
            raise ValueError("execution_refs_not_canonical")
        if self.gap_projection_status is not GapProjectionStatus.DERIVED and self.gap_refs:
            raise ValueError("unavailable_gap_projection_must_not_claim_empty_or_stale_refs")
        if (
            self.gap_projection_status is GapProjectionStatus.DERIVED
            and not self.gap_source_semantic_hash
        ):
            raise ValueError("derived_gap_projection_requires_semantic_binding")
        if self.gap_projection_status is GapProjectionStatus.DERIVED:
            if self.gap_projection_hash != canonical_hash(list(self.gap_refs)):
                raise ValueError("gap_projection_hash_mismatch")
        elif self.gap_projection_hash is not None:
            raise ValueError("unavailable_gap_projection_has_hash")
        if self.intent_refs and (
            self.intent_source_gap_hash is None
            or self.intent_source_gap_hash != self.gap_projection_hash
        ):
            raise ValueError("derived_intents_require_current_gap_binding")
        if self.state_ref != research_run_state_artifact_ref(self.run_ref, self.state_hash):
            raise ValueError("research_run_state_ref_mismatch")
        if self.state_hash != self.compute_hash(self.model_dump(mode="json")):
            raise ValueError("research_run_state_hash_mismatch")
        return self

    @staticmethod
    def compute_hash(payload: dict[str, Any]) -> str:
        excluded = {"state_hash", "state_ref", "created_at"}
        return canonical_hash({key: value for key, value in payload.items() if key not in excluded})


def build_research_run_state(**values: Any) -> ResearchRunState:
    """Construct a state whose identity excludes creation time and random IDs."""

    values = _with_model_defaults(ResearchRunState, values)
    values.setdefault("created_at", datetime.now(UTC))
    values.setdefault("schema_version", RUN_STATE_SCHEMA)
    values.setdefault("orchestrator_version", ORCHESTRATOR_V3)
    values["state_hash"] = ResearchRunState.compute_hash(
        {**values, "state_hash": "0" * 64}
    )
    values["state_ref"] = research_run_state_artifact_ref(values["run_ref"], values["state_hash"])
    return ResearchRunState.model_validate(values)


def research_run_state_artifact_ref(run_ref: UUID, state_hash: str) -> str:
    return f"runs/{run_ref}/run-state/{state_hash}.json"


def semantic_state_artifact_ref(run_ref: UUID, snapshot_hash: str) -> str:
    return f"runs/{run_ref}/semantic-state/{snapshot_hash}.json"
