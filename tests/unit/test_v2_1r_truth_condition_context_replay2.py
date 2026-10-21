from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import replay_v2_1r_truth_condition_context_phase_c_replay2 as replay2  # noqa: E402  # type: ignore[import-not-found]
from truth_condition_context_replay2_view_adapter import (  # noqa: E402
    normalize_verification_view,
)


def test_replay2_view_retains_eligibility_and_material_context_fields() -> None:
    row = {
        "propositions": [
            {
                "local_ref": "p0",
                "source_span": "In 2025 Atlas reached 8 units",
                "subject": "Atlas",
                "predicate": "reached",
                "typed_value": {"kind": "quantity", "value": 8, "unit": "units"},
                "polarity": "POSITIVE",
                "material_qualifiers": [],
                "eligibility": "INDEPENDENTLY_ASSERTED",
            }
        ]
    }
    source = replay2.tcc.project_source_context_graph(
        "In 2025, Atlas reached 8 units.",
        [{"local_ref": "p0", "source_span": "Atlas reached 8 units"}],
    )
    output = replay2.tcc.project_output_context_graph(row["propositions"])
    claim = {
        "source_context_graph": source.model_dump(mode="json"),
        "output_context_graph": output.model_dump(mode="json"),
    }

    views = replay2.build_view_rows(claim, row)

    assert len(views) == 1
    assert views[0]["proposition_ref"] == "p0"
    assert views[0]["composition_eligibility"] == "INDEPENDENTLY_ASSERTED"
    assert "normalized" not in views[0]
    assert "detector_limit_impact" in views[0]
    assert "resolved_material_contexts" in views[0]


def test_view_impact_reports_cardinality_per_assertion_not_per_claim() -> None:
    claims = [
        {
            "cohort": "bounded_shadow",
            "claim_id": "claim-1",
            "verification_assertion_views": [
                {
                    "source_context_completeness": "SUPPORTED_FOR_CONTRACT",
                    "composition_eligibility": "INDEPENDENTLY_ASSERTED",
                    "detector_limit_impact": {
                        "limited_family_cardinality": "NONE",
                        "affirmative_future_semantic_comparison_safe": True,
                    },
                },
                {
                    "source_context_completeness": "DETECTOR_LIMIT",
                    "composition_eligibility": "INDEPENDENTLY_ASSERTED",
                    "detector_limit_impact": {
                        "limited_family_cardinality": "MULTIPLE_FAMILIES",
                        "affirmative_future_semantic_comparison_safe": False,
                    },
                },
            ],
        }
    ]

    result = replay2._view_impact(claims)

    assert result["views_total"] == 2
    assert result["views_no_known_material_detector_limit"] == 1
    assert result["views_multiple_limited_families"] == 1
    assert result["views_affirmative_comparison_unsafe_due_to_detector_limits_or_unknown"] == 1


def _complete_current_view(**updates: object) -> dict[str, object]:
    result: dict[str, object] = {
        "proposition_ref": "p0",
        "composition_eligibility": "INDEPENDENTLY_ASSERTED",
        "source_context_completeness": "DETECTOR_LIMIT",
        "schema_version": "truth-condition-context-shadow-v1",
        "material_context_unknown": True,
        "detector_limited_context_families": ["EPISTEMIC", "CONDITION"],
        "resolved_material_contexts": [{"context_ref": "oc0", "context_type": "TEMPORAL"}],
        "epistemic_contexts": [],
        "epistemic_completeness": "DETECTOR_LIMIT",
    }
    result.update(updates)
    return result


def test_current_source_completeness_is_distinct_from_preservation_completeness() -> None:
    view = normalize_verification_view(
        _complete_current_view(context_completeness="MATERIAL_UNKNOWN")
    )
    assert view.source_context_completeness == "DETECTOR_LIMIT"
    assert view.preservation_completeness == "MATERIAL_UNKNOWN"
    assert view.view_schema_version == "truth-condition-context-shadow-v1"


def test_historical_context_completeness_does_not_alias_source_coverage() -> None:
    view = normalize_verification_view(
        {
            "proposition_ref": "p0",
            "composition_eligibility": "INDEPENDENTLY_ASSERTED",
            "context_completeness": "DETECTOR_LIMIT",
            "material_context_unknown": True,
            "resolved_material_contexts": [],
            "epistemic_context_signatures": [],
        }
    )
    assert view.preservation_completeness == "DETECTOR_LIMIT"
    assert view.source_context_completeness == "UNRESOLVED"
    assert "LEGACY_VIEW_ADAPTED" in view.compatibility_diagnostics
    assert "COMPLETENESS_FIELD_UNRESOLVED:source_context_completeness" in (
        view.compatibility_diagnostics
    )


def test_historical_completeness_is_not_a_pure_rename_of_current_source_coverage() -> None:
    adapted = normalize_verification_view(
        {
            "proposition_ref": "p0",
            "composition_eligibility": "INDEPENDENTLY_ASSERTED",
            "context_completeness": "COMPLETE",
        }
    )
    native = normalize_verification_view(
        _complete_current_view(source_context_completeness="SUPPORTED_FOR_CONTRACT")
    )
    assert adapted.source_context_completeness == "UNRESOLVED"
    assert native.source_context_completeness == "SUPPORTED_FOR_CONTRACT"
    assert adapted.preservation_completeness == "COMPLETE"


def test_missing_completeness_fails_closed_without_assuming_success() -> None:
    view = normalize_verification_view({"proposition_ref": "p0"})
    assert view.source_context_completeness == "UNRESOLVED"
    assert view.preservation_completeness is None
    assert view.composition_eligibility == "UNKNOWN"
    assert "VIEW_FIELD_MISSING:context_completeness" in view.compatibility_diagnostics


def test_compatibility_failure_does_not_rewrite_known_eligibility() -> None:
    view = normalize_verification_view(
        {"proposition_ref": "p0", "composition_eligibility": "NOT_INDEPENDENTLY_ASSERTED"}
    )
    assert view.composition_eligibility == "NOT_INDEPENDENTLY_ASSERTED"
    assert view.source_context_completeness == "UNRESOLVED"


def test_detector_limited_families_are_preserved_exactly() -> None:
    view = normalize_verification_view(_complete_current_view())
    assert view.detector_limited_families == ("EPISTEMIC", "CONDITION")


def test_material_context_unknown_is_preserved() -> None:
    view = normalize_verification_view(_complete_current_view(material_context_unknown=True))
    assert view.material_context_unknown is True


def test_nested_legacy_artifact_layout_is_adapted_without_inference() -> None:
    view = normalize_verification_view(
        {
            "base_verification_assertion_view": {
                "proposition_ref": "p7",
                "composition_eligibility": "INDEPENDENTLY_ASSERTED",
                "schema_version": "truth-condition-context-shadow-v1",
            },
            "context_completeness": "MATERIAL_MISSING",
            "material_context_unknown": True,
            "resolved_material_contexts": [],
            "epistemic_context_signatures": ["sig-a"],
        }
    )
    assert view.proposition_ref == "p7"
    assert view.preservation_completeness == "MATERIAL_MISSING"
    assert view.source_context_completeness == "UNRESOLVED"
    assert view.epistemic_context_signatures == ("sig-a",)


def test_unknown_view_schema_is_diagnosed() -> None:
    view = normalize_verification_view(
        _complete_current_view(schema_version="truth-condition-context-shadow-v99")
    )
    assert "VIEW_SCHEMA_VERSION_UNKNOWN" in view.compatibility_diagnostics


def test_shared_context_rows_are_enriched_with_claim_identity() -> None:
    rows = replay2._shared_context_occurrences(
        [
            {
                "cohort": "fresh_validation",
                "claim_id": "claim-42",
                "shared_contexts": [{"context_ref": "sc0", "outcome": "PRESERVED"}],
            }
        ]
    )
    assert rows == [
        {
            "cohort": "fresh_validation",
            "claim_id": "claim-42",
            "context_ref": "sc0",
            "outcome": "PRESERVED",
        }
    ]
