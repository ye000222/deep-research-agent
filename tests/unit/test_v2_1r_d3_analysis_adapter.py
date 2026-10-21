from __future__ import annotations

from scripts.v2_1r_d3_adapter import (
    NORMALIZATION_CONTRACT_VERSION,
    NormalizedReplayBindingView,
    normalize_replay_binding,
    normalized_view_from_dict,
    resolve_omission_role,
)


def _source_binding(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "binding_ref": "B-1",
        "evidence_ref": "E-1",
        "semantic_role": "EPISTEMIC_AGENT",
        "target_kind": "CONTEXT",
        "target_ref": "C-1",
        "dimensions": {"textual_grounding": "VALID", "structural_grounding": "VALID"},
    }
    row.update(updates)
    return row


def _phase_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "claim_id": "Q-1",
        "binding_ref": "B-1",
        "final_oracle_simulated_semantic_support": "VALID",
    }
    row.update(updates)
    return row


def _normalize(
    source: list[dict[str, object]] | None = None,
    phase: list[dict[str, object]] | None = None,
    *,
    ref: str = "B-1",
    schema: str = "source-context-extraction-schema-v2",
    artifact: str | None = "canary-result.json",
) -> NormalizedReplayBindingView:
    return normalize_replay_binding(
        claim_id="Q-1",
        binding_ref=ref,
        canonical_bindings=source if source is not None else [_source_binding()],
        phase_c_rows=phase if phase is not None else [_phase_row()],
        source_artifact=artifact,
        schema_version=schema,
    )


def test_semantic_role_resolves_from_canonical_binding_record() -> None:
    view = _normalize()
    assert view.normalization_status == "RESOLVED"
    assert view.semantic_role == "EPISTEMIC_AGENT"
    assert view.source_ref == "B-1"
    assert view.provenance_steps[1] == "semantic_role_from_frozen_source_binding"


def test_missing_binding_ref_fails_closed() -> None:
    view = _normalize(ref="MISSING")
    assert view.normalization_status == "BINDING_REF_NOT_FOUND"
    assert view.semantic_role is None


def test_legacy_provenance_row_resolves_binding_role_but_not_missing_support_value() -> None:
    # Phase-C/provenance records need only carry the stable reference.
    view = _normalize(phase=[{"claim_id": "Q-1", "binding_ref": "B-1"}])
    assert view.normalization_status == "RESOLVED"
    assert view.semantic_role == "EPISTEMIC_AGENT"
    assert view.support_resolution_status == "LEGACY_FIELD_UNRESOLVED"
    assert view.hybrid_support is None


def test_older_normalized_row_without_role_returns_explicit_schema_unresolved() -> None:
    view = normalized_view_from_dict(
        {"claim_id": "Q-1", "binding_ref": "B-1", "source_artifact": "legacy.json"}
    )
    assert view.semantic_role is None
    assert view.normalization_status == "ARTIFACT_SCHEMA_UNSUPPORTED"
    assert view.support_resolution_status == "NOT_RESOLVED"
    assert view.provenance_steps == ("normalized_view_schema_rejected",)


def test_valid_indirect_resolution_preserves_canonical_target() -> None:
    view = _normalize()
    assert view.target_kind == "CONTEXT"
    assert view.target_ref == view.node_ref == "C-1"


def test_unknown_schema_version_fails_closed() -> None:
    view = _normalize(schema="unknown-schema-v99")
    assert view.normalization_status == "ARTIFACT_SCHEMA_UNSUPPORTED"
    assert view.semantic_role is None


def test_missing_provenance_source_fails_closed() -> None:
    view = _normalize(artifact=None)
    assert view.normalization_status == "PROVENANCE_LOOKUP_FAILED"
    assert view.semantic_role is None


def test_omission_comparison_resolves_role_from_normalized_view() -> None:
    view = _normalize()
    result = resolve_omission_role(
        claim_id="Q-1",
        node_ref="C-1",
        required_role="EPISTEMIC_AGENT",
        normalized_bindings=[view],
    )
    assert result.status == "ROLE_MATCHED"
    assert result.matching_binding_refs == ("B-1",)


def test_omission_comparison_fails_closed_for_unresolved_role() -> None:
    view = _normalize(source=[_source_binding(semantic_role=None)])
    result = resolve_omission_role(
        claim_id="Q-1",
        node_ref="C-1",
        required_role="EPISTEMIC_AGENT",
        normalized_bindings=[view],
    )
    assert result.status == "LEGACY_FIELD_UNRESOLVED"
    assert result.matching_binding_refs == ()


def test_human_notes_do_not_synthesize_semantic_role() -> None:
    view = _normalize(source=[_source_binding(semantic_role=None)])
    result = resolve_omission_role(
        claim_id="Q-1",
        node_ref="C-1",
        required_role="EPISTEMIC_AGENT",
        normalized_bindings=[view],
    )
    assert view.normalization_contract_version == NORMALIZATION_CONTRACT_VERSION
    assert view.semantic_role is None
    assert result.status == "LEGACY_FIELD_UNRESOLVED"
