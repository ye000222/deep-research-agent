"""Analysis-only canonical normalization for frozen D3.1 replay artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

EXPECTED_SOURCE_SCHEMA = "source-context-extraction-schema-v2"
NORMALIZATION_CONTRACT_VERSION = "grounding-requirement-d3-normalized-binding.v1"


@dataclass(frozen=True)
class NormalizedReplayBindingView:
    claim_id: str
    binding_ref: str
    semantic_role: str | None
    target_kind: str | None
    target_ref: str | None
    node_ref: str | None
    evidence_ref: str | None
    textual_grounding: str | None
    structural_grounding: str | None
    hybrid_support: str | None
    normalization_status: str
    support_resolution_status: str
    source_artifact: str | None
    source_ref: str | None
    schema_version: str | None
    normalization_contract_version: str
    provenance_steps: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OmissionRoleResolution:
    status: str
    matching_binding_refs: tuple[str, ...]
    unresolved_binding_refs: tuple[str, ...]


def normalized_view_from_dict(payload: dict[str, Any]) -> NormalizedReplayBindingView:
    """Rehydrate the current view or fail closed for older/incompatible rows."""
    expected_fields = set(NormalizedReplayBindingView.__dataclass_fields__)
    if set(payload) != expected_fields:
        claim_id = payload.get("claim_id")
        binding_ref = payload.get("binding_ref")
        if not isinstance(claim_id, str) or not isinstance(binding_ref, str):
            raise ValueError("normalized_binding_view_identity_unresolved")
        return NormalizedReplayBindingView(
            claim_id=claim_id,
            binding_ref=binding_ref,
            semantic_role=None,
            target_kind=None,
            target_ref=None,
            node_ref=None,
            evidence_ref=None,
            textual_grounding=None,
            structural_grounding=None,
            hybrid_support=None,
            normalization_status="ARTIFACT_SCHEMA_UNSUPPORTED",
            support_resolution_status="NOT_RESOLVED",
            source_artifact=(
                payload.get("source_artifact")
                if isinstance(payload.get("source_artifact"), str)
                else None
            ),
            source_ref=binding_ref,
            schema_version=(
                payload.get("schema_version")
                if isinstance(payload.get("schema_version"), str)
                else None
            ),
            normalization_contract_version=NORMALIZATION_CONTRACT_VERSION,
            provenance_steps=("normalized_view_schema_rejected",),
        )
    values = dict(payload)
    values["provenance_steps"] = tuple(values["provenance_steps"])
    return NormalizedReplayBindingView(**values)


def normalize_replay_binding(
    *,
    claim_id: str,
    binding_ref: str,
    canonical_bindings: list[dict[str, Any]],
    phase_c_rows: list[dict[str, Any]],
    source_artifact: str | None,
    schema_version: str | None,
) -> NormalizedReplayBindingView:
    """Resolve fields only through the frozen source binding's stable identity."""
    base: dict[str, Any] = {
        "claim_id": claim_id,
        "binding_ref": binding_ref,
        "source_artifact": source_artifact,
        "source_ref": binding_ref or None,
        "schema_version": schema_version,
        "normalization_contract_version": NORMALIZATION_CONTRACT_VERSION,
    }
    if not source_artifact:
        return NormalizedReplayBindingView(
            **base,
            semantic_role=None,
            target_kind=None,
            target_ref=None,
            node_ref=None,
            evidence_ref=None,
            textual_grounding=None,
            structural_grounding=None,
            hybrid_support=None,
            normalization_status="PROVENANCE_LOOKUP_FAILED",
            support_resolution_status="PROVENANCE_LOOKUP_FAILED",
            provenance_steps=("source_artifact_missing",),
        )
    if schema_version != EXPECTED_SOURCE_SCHEMA:
        return NormalizedReplayBindingView(
            **base,
            semantic_role=None,
            target_kind=None,
            target_ref=None,
            node_ref=None,
            evidence_ref=None,
            textual_grounding=None,
            structural_grounding=None,
            hybrid_support=None,
            normalization_status="ARTIFACT_SCHEMA_UNSUPPORTED",
            support_resolution_status="NOT_RESOLVED",
            provenance_steps=("schema_version_rejected",),
        )
    if not binding_ref:
        return NormalizedReplayBindingView(
            **base,
            semantic_role=None,
            target_kind=None,
            target_ref=None,
            node_ref=None,
            evidence_ref=None,
            textual_grounding=None,
            structural_grounding=None,
            hybrid_support=None,
            normalization_status="BINDING_REF_NOT_FOUND",
            support_resolution_status="NOT_RESOLVED",
            provenance_steps=("empty_binding_ref",),
        )

    matches = [row for row in canonical_bindings if str(row.get("binding_ref", "")) == binding_ref]
    if len(matches) != 1:
        status = "BINDING_REF_NOT_FOUND" if not matches else "PROVENANCE_LOOKUP_FAILED"
        return NormalizedReplayBindingView(
            **base,
            semantic_role=None,
            target_kind=None,
            target_ref=None,
            node_ref=None,
            evidence_ref=None,
            textual_grounding=None,
            structural_grounding=None,
            hybrid_support=None,
            normalization_status=status,
            support_resolution_status="NOT_RESOLVED",
            provenance_steps=("canonical_binding_ref_lookup",),
        )

    source = matches[0]
    role = source.get("semantic_role")
    if not isinstance(role, str) or not role:
        return NormalizedReplayBindingView(
            **base,
            semantic_role=None,
            target_kind=str(source["target_kind"]) if source.get("target_kind") else None,
            target_ref=str(source["target_ref"]) if source.get("target_ref") else None,
            node_ref=str(source["target_ref"]) if source.get("target_ref") else None,
            evidence_ref=str(source["evidence_ref"]) if source.get("evidence_ref") else None,
            textual_grounding=None,
            structural_grounding=None,
            hybrid_support=None,
            normalization_status="LEGACY_FIELD_UNRESOLVED",
            support_resolution_status="NOT_RESOLVED",
            provenance_steps=("canonical_binding_ref_lookup", "semantic_role_missing_in_source"),
        )

    phase_matches = [
        row
        for row in phase_c_rows
        if str(row.get("claim_id", "")) == claim_id
        and str(row.get("binding_ref", "")) == binding_ref
    ]
    dimensions = source.get("dimensions", {})
    phase = phase_matches[0] if len(phase_matches) == 1 else None
    support_value_present = bool(
        phase and phase.get("final_oracle_simulated_semantic_support") is not None
    )
    support_status = (
        "PROVENANCE_LOOKUP_FAILED"
        if phase is None
        else "RESOLVED"
        if support_value_present
        else "LEGACY_FIELD_UNRESOLVED"
    )
    return NormalizedReplayBindingView(
        **base,
        semantic_role=role,
        target_kind=str(source["target_kind"]) if source.get("target_kind") else None,
        target_ref=str(source["target_ref"]) if source.get("target_ref") else None,
        node_ref=str(source["target_ref"]) if source.get("target_ref") else None,
        evidence_ref=str(source["evidence_ref"]) if source.get("evidence_ref") else None,
        textual_grounding=(
            str(dimensions["textual_grounding"])
            if dimensions.get("textual_grounding") is not None
            else None
        ),
        structural_grounding=(
            str(dimensions["structural_grounding"])
            if dimensions.get("structural_grounding") is not None
            else None
        ),
        hybrid_support=(
            str(phase["final_oracle_simulated_semantic_support"])
            if phase and phase.get("final_oracle_simulated_semantic_support") is not None
            else None
        ),
        normalization_status="RESOLVED",
        support_resolution_status=support_status,
        provenance_steps=(
            "claim_and_binding_ref_lookup",
            "semantic_role_from_frozen_source_binding",
            "grounding_dimensions_from_frozen_source_binding",
            "hybrid_support_from_phase_c_claim_and_binding_ref",
        ),
    )


def resolve_omission_role(
    *,
    claim_id: str,
    node_ref: str,
    required_role: str,
    normalized_bindings: list[NormalizedReplayBindingView],
) -> OmissionRoleResolution:
    """Compare a known omission with normalized bindings; never infer from notes."""
    scoped = [
        row for row in normalized_bindings if row.claim_id == claim_id and row.node_ref == node_ref
    ]
    unresolved = tuple(
        sorted(row.binding_ref for row in scoped if row.normalization_status != "RESOLVED")
    )
    if unresolved:
        return OmissionRoleResolution(
            status="LEGACY_FIELD_UNRESOLVED",
            matching_binding_refs=(),
            unresolved_binding_refs=unresolved,
        )
    matched = tuple(sorted(row.binding_ref for row in scoped if row.semantic_role == required_role))
    return OmissionRoleResolution(
        status="ROLE_MATCHED" if matched else "ROLE_NOT_REPRESENTED",
        matching_binding_refs=matched,
        unresolved_binding_refs=(),
    )
