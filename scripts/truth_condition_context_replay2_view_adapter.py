"""Fail-closed analyzer adapter for historical/current assertion-view payloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

CURRENT_VIEW_SCHEMA = "truth-condition-context-shadow-v1"


@dataclass(frozen=True)
class NormalizedVerificationView:
    proposition_ref: str | None
    composition_eligibility: str
    source_context_completeness: str
    preservation_completeness: str | None
    material_context_unknown: bool | None
    detector_limited_families: tuple[str, ...] | None
    resolved_material_contexts: tuple[dict[str, Any], ...] | None
    epistemic_contexts: tuple[dict[str, Any], ...] | None
    epistemic_context_signatures: tuple[str, ...] | None
    epistemic_completeness: str | None
    view_schema_version: str
    compatibility_diagnostics: tuple[str, ...]

    def to_summary_payload(self) -> dict[str, Any]:
        """Provide the explicitly named legacy summary key only when resolved."""
        payload = asdict(self)
        if self.preservation_completeness is not None:
            payload["context_completeness"] = self.preservation_completeness
        return payload


def normalize_verification_view(
    raw: object,
    *,
    replay_preservation_completeness: str | None = None,
) -> NormalizedVerificationView:
    diagnostics: list[str] = []
    if not isinstance(raw, dict):
        return NormalizedVerificationView(
            proposition_ref=None,
            composition_eligibility="UNKNOWN",
            source_context_completeness="UNRESOLVED",
            preservation_completeness=None,
            material_context_unknown=None,
            detector_limited_families=None,
            resolved_material_contexts=None,
            epistemic_contexts=None,
            epistemic_context_signatures=None,
            epistemic_completeness=None,
            view_schema_version="UNKNOWN",
            compatibility_diagnostics=("VIEW_SCHEMA_VERSION_UNKNOWN", "VIEW_FIELD_MISSING"),
        )

    nested_base = raw.get("base_verification_assertion_view")
    if isinstance(nested_base, dict):
        raw = {**nested_base, **raw}
        diagnostics.append("LEGACY_VIEW_ADAPTED")
    declared_schema = raw.get("schema_version")
    if declared_schema is not None and declared_schema != CURRENT_VIEW_SCHEMA:
        diagnostics.append("VIEW_SCHEMA_VERSION_UNKNOWN")

    proposition_ref = raw.get("proposition_ref")
    if not isinstance(proposition_ref, str) or not proposition_ref:
        diagnostics.append("VIEW_FIELD_MISSING:proposition_ref")
        proposition_ref = None

    eligibility = raw.get("composition_eligibility")
    if not isinstance(eligibility, str) or not eligibility:
        diagnostics.append("VIEW_FIELD_MISSING:composition_eligibility")
        eligibility = "UNKNOWN"

    current_source_completeness = raw.get("source_context_completeness")
    legacy_completeness = raw.get("context_completeness")
    if isinstance(current_source_completeness, str) and current_source_completeness:
        source_completeness = current_source_completeness
        if replay_preservation_completeness is not None:
            preservation_completeness = replay_preservation_completeness
        elif isinstance(legacy_completeness, str) and legacy_completeness:
            preservation_completeness = legacy_completeness
        else:
            preservation_completeness = None
            diagnostics.append("COMPLETENESS_FIELD_UNRESOLVED:preservation_completeness")
        schema_version = str(raw.get("schema_version") or CURRENT_VIEW_SCHEMA)
        diagnostics.append("CURRENT_VIEW_NATIVE")
    elif isinstance(legacy_completeness, str) and legacy_completeness:
        # This historical field describes replay preservation/unknown state, not source
        # detector coverage. Never alias it into source_context_completeness.
        source_completeness = "UNRESOLVED"
        preservation_completeness = legacy_completeness
        schema_version = "legacy-phase-c-supplemental-view"
        if "LEGACY_VIEW_ADAPTED" not in diagnostics:
            diagnostics.append("LEGACY_VIEW_ADAPTED")
        diagnostics.append("COMPLETENESS_FIELD_UNRESOLVED:source_context_completeness")
    elif replay_preservation_completeness is not None:
        source_completeness = "UNRESOLVED"
        preservation_completeness = replay_preservation_completeness
        schema_version = "legacy-phase-c-supplemental-view"
        diagnostics.extend(
            ("LEGACY_VIEW_ADAPTED", "COMPLETENESS_FIELD_UNRESOLVED:source_context_completeness")
        )
    else:
        source_completeness = "UNRESOLVED"
        preservation_completeness = None
        schema_version = "UNKNOWN"
        diagnostics.extend(
            (
                "VIEW_SCHEMA_VERSION_UNKNOWN",
                "VIEW_FIELD_MISSING:source_context_completeness",
                "VIEW_FIELD_MISSING:context_completeness",
            )
        )

    if raw.get("schema_version") is None and schema_version == "UNKNOWN":
        diagnostics.append("VIEW_SCHEMA_VERSION_UNKNOWN")

    material_unknown = raw.get("material_context_unknown")
    if not isinstance(material_unknown, bool):
        material_unknown = None
        diagnostics.append("VIEW_FIELD_MISSING:material_context_unknown")

    limited = raw.get("detector_limited_context_families")
    if isinstance(limited, list | tuple) and all(isinstance(item, str) for item in limited):
        limited_families: tuple[str, ...] | None = tuple(limited)
    else:
        limited_families = None
        diagnostics.append("VIEW_FIELD_MISSING:detector_limited_context_families")

    resolved = raw.get("resolved_material_contexts")
    resolved_contexts = (
        tuple(item for item in resolved if isinstance(item, dict))
        if isinstance(resolved, list | tuple)
        else None
    )
    if resolved_contexts is None:
        diagnostics.append("VIEW_FIELD_MISSING:resolved_material_contexts")

    epistemic = raw.get("epistemic_contexts")
    epistemic_contexts = (
        tuple(item for item in epistemic if isinstance(item, dict))
        if isinstance(epistemic, list | tuple)
        else None
    )
    if epistemic_contexts is None:
        diagnostics.append("VIEW_FIELD_MISSING:epistemic_contexts")

    signatures = raw.get("epistemic_context_signatures")
    if isinstance(signatures, list | tuple) and all(isinstance(item, str) for item in signatures):
        epistemic_signatures: tuple[str, ...] | None = tuple(signatures)
    else:
        epistemic_signatures = None

    epistemic_completeness = raw.get("epistemic_completeness")
    if epistemic_completeness is not None and not isinstance(epistemic_completeness, str):
        epistemic_completeness = None
        diagnostics.append("VIEW_FIELD_INVALID:epistemic_completeness")

    return NormalizedVerificationView(
        proposition_ref=proposition_ref,
        composition_eligibility=eligibility,
        source_context_completeness=source_completeness,
        preservation_completeness=preservation_completeness,
        material_context_unknown=material_unknown,
        detector_limited_families=limited_families,
        resolved_material_contexts=resolved_contexts,
        epistemic_contexts=epistemic_contexts,
        epistemic_context_signatures=epistemic_signatures,
        epistemic_completeness=epistemic_completeness,
        view_schema_version=schema_version,
        compatibility_diagnostics=tuple(dict.fromkeys(diagnostics)),
    )
