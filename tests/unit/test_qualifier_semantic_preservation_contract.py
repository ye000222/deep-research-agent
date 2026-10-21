from __future__ import annotations

import json
from pathlib import Path

from app.domain.qualifier_preservation import PreservationEvidenceStrength, SourceApplicability
from app.services.atomic_proposition_validation import (
    QualifierCheckStatus,
    inspect_qualifier_fields,
)

from scripts.replay_v2_1r_qualifier_obligations_100 import replay


def _checks(claim: str, *propositions: dict[str, object]) -> dict[str, object]:
    return {item.category: item for item in inspect_qualifier_fields(claim, list(propositions))}


def test_semantics_can_move_between_valid_structured_carriers() -> None:
    cases = (
        (
            "On 2042-03-11, device A became active.",
            {
                "proposition_text": "Device A became active on 2042-03-11.",
                "subject": "device A",
                "predicate": "became active",
                "polarity": "asserted",
                "material_qualifiers": [
                    {"category": "time", "value": "2042-03-11", "source_span": "2042-03-11"}
                ],
            },
            "date_time",
        ),
        (
            "Version 3.2.1 is supported.",
            {
                "proposition_text": "Version 3.2.1 is supported.",
                "subject": "software",
                "predicate": "is supported",
                "polarity": "asserted",
                "material_qualifiers": [
                    {"category": "version", "value": "3.2.1", "source_span": "Version 3.2.1"}
                ],
            },
            "version",
        ),
        (
            "The device moved 100 km.",
            {
                "proposition_text": "The device moved 100 km.",
                "subject": "device",
                "predicate": "moved",
                "polarity": "asserted",
                "typed_value": {
                    "kind": "quantity",
                    "numeric": {"operator": "exact", "value": 100, "unit": "kilometers"},
                },
            },
            "unit",
        ),
        (
            "System may operate.",
            {
                "proposition_text": "System may operate.",
                "subject": "system",
                "predicate": "operate",
                "polarity": "asserted",
                "modality": "may",
            },
            "modality",
        ),
        (
            "The study finds a result.",
            {
                "proposition_text": "The study finds a result.",
                "subject": "study",
                "predicate": "finds",
                "polarity": "asserted",
            },
            "attribution",
        ),
        (
            "Device A ranks higher than Device B.",
            {
                "proposition_text": "Device A ranks higher than Device B.",
                "subject": "Device A",
                "predicate": "ranks higher",
                "polarity": "asserted",
                "typed_value": {
                    "kind": "comparison",
                    "comparison_target": "Device B",
                    "comparison_metric": "rank",
                    "comparison_metric_status": "explicit",
                },
            },
            "comparison",
        ),
        (
            "Under configuration B, the device runs.",
            {
                "proposition_text": "Under configuration B, the device runs.",
                "subject": "device",
                "predicate": "runs",
                "polarity": "asserted",
                "material_qualifiers": [
                    {
                        "category": "condition",
                        "value": "configuration B",
                        "source_span": "Under configuration B",
                    }
                ],
            },
            "condition",
        ),
        (
            "The estimate used four samples.",
            {
                "proposition_text": "The estimate used four samples.",
                "subject": "estimate",
                "predicate": "used",
                "polarity": "asserted",
                "typed_value": {"kind": "text", "text": "four samples"},
                "material_qualifiers": [
                    {
                        "category": "sample_count",
                        "value": "four samples",
                        "source_span": "four samples",
                    }
                ],
            },
            "sample_count",
        ),
    )
    for claim, proposition, category in cases:
        result = _checks(claim, proposition)[category]
        assert result.status is QualifierCheckStatus.PRESERVED, (category, result)
        assert result.applicability is SourceApplicability.SUPPORTED_SIGNAL
        assert result.evidence_strength in {
            PreservationEvidenceStrength.STRONG,
            PreservationEvidenceStrength.MEDIUM,
        }


def test_wrong_value_or_wrong_attachment_is_not_preserved() -> None:
    cases = (
        (
            "On 2042-03-11, device A became active.",
            {
                "proposition_text": "Device A became active on 2043-03-11.",
                "subject": "device A",
                "predicate": "became active",
                "polarity": "asserted",
                "material_qualifiers": [
                    {"category": "time", "value": "2043-03-11", "source_span": "2042-03-11"}
                ],
            },
            "date_time",
        ),
        (
            "The device moved 100 km.",
            {
                "proposition_text": "The device moved 100 miles.",
                "subject": "device",
                "predicate": "moved",
                "polarity": "asserted",
                "typed_value": {
                    "kind": "quantity",
                    "numeric": {"operator": "exact", "value": 100, "unit": "miles"},
                },
            },
            "unit",
        ),
        (
            "Device A ranks higher than Device B.",
            {
                "proposition_text": "Device A ranks higher than Device B.",
                "subject": "Device A",
                "predicate": "ranks higher",
                "polarity": "asserted",
                "typed_value": {
                    "kind": "comparison",
                    "comparison_target": "Device C",
                    "comparison_metric": "rank",
                    "comparison_metric_status": "explicit",
                },
            },
            "comparison",
        ),
        (
            "The study finds a result.",
            {
                "proposition_text": "The study finds a result.",
                "subject": "study",
                "predicate": "finds",
                "attribution": "Vendor X",
                "polarity": "asserted",
            },
            "attribution",
        ),
        (
            "The device uses method Alpha to inspect parts.",
            {
                "proposition_text": "The device uses method Alpha to inspect parts.",
                "subject": "device",
                "predicate": "uses",
                "typed_value": {"kind": "text", "text": "method Beta"},
                "material_qualifiers": [
                    {
                        "category": "method",
                        "value": "method Beta",
                        "source_span": "uses method Alpha",
                    }
                ],
                "polarity": "asserted",
            },
            "method",
        ),
        (
            "Only devices in region North qualify.",
            {
                "proposition_text": "Only devices in region North qualify.",
                "subject": "devices",
                "predicate": "qualify",
                "material_qualifiers": [
                    {
                        "category": "scope",
                        "value": "region South",
                        "source_span": "Only devices in region North",
                    }
                ],
                "polarity": "asserted",
            },
            "scope",
        ),
    )
    for claim, proposition, category in cases:
        result = _checks(claim, proposition)[category]
        assert result.status is QualifierCheckStatus.POTENTIAL_LOSS, (category, result)


def test_equal_token_on_unrelated_proposition_does_not_establish_attachment() -> None:
    claim = "In 2042, device A became active; device B remained idle."
    propositions = (
        {
            "proposition_text": "In 2042, device A became active.",
            "subject": "device A",
            "predicate": "became active",
            "polarity": "asserted",
            "source_span": "In 2042, device A became active",
            "material_qualifiers": [
                {
                    "category": "time",
                    "value": "2042",
                    "source_span": "In 2042, device A became active",
                }
            ],
        },
        {
            "proposition_text": "Device B remained idle in 2043.",
            "subject": "device B",
            "predicate": "remained idle",
            "polarity": "asserted",
            "source_span": "device B remained idle",
            "material_qualifiers": [
                {"category": "time", "value": "2043", "source_span": "device B remained idle"}
            ],
        },
    )
    assert _checks(claim, *propositions)["date_time"].status is QualifierCheckStatus.PRESERVED


def test_ambiguous_attachment_and_true_drop_fail_closed() -> None:
    ambiguous = _checks(
        "Under setting B, device A runs and device C sleeps.",
        {
            "proposition_text": "Device A runs under setting B.",
            "subject": "device A",
            "predicate": "runs",
            "material_qualifiers": [{"category": "condition", "value": "setting B"}],
        },
        {
            "proposition_text": "Device C sleeps under setting B.",
            "subject": "device C",
            "predicate": "sleeps",
            "material_qualifiers": [{"category": "condition", "value": "setting B"}],
        },
    )
    dropped = _checks(
        "The device moved 100 km.",
        {
            "proposition_text": "The device moved.",
            "subject": "device",
            "predicate": "moved",
            "polarity": "asserted",
        },
    )
    assert ambiguous["condition"].status is QualifierCheckStatus.AMBIGUOUS
    assert dropped["unit"].status is QualifierCheckStatus.POTENTIAL_LOSS


def test_no_detected_marker_is_not_claimed_as_semantic_absence() -> None:
    result = _checks(
        "The device is active.",
        {"proposition_text": "The device is active.", "subject": "device", "predicate": "active"},
    )["method"]
    assert result.status is QualifierCheckStatus.NOT_APPLICABLE
    assert result.applicability is SourceApplicability.NO_SUPPORTED_SIGNAL
    assert result.reason_codes == ("NO_SUPPORTED_SOURCE_SIGNAL",)


def test_unsupported_source_script_is_reported_as_detector_coverage_limit() -> None:
    checks = inspect_qualifier_fields("Устройство работает без ошибок.", [])
    assert checks
    assert all(item.status is QualifierCheckStatus.NOT_APPLICABLE for item in checks)
    assert all(item.applicability is SourceApplicability.DETECTOR_COVERAGE_LIMIT for item in checks)
    assert all(item.reason_codes == ("SOURCE_SCRIPT_OUTSIDE_DETECTOR_COVERAGE",) for item in checks)


def test_frozen_100_replay_is_offline_and_migrates_all_audited_warnings() -> None:
    result = replay(
        Path("artifacts/v2_1r_bounded_shadow_100/canary_diagnostics"),
        Path("artifacts/v2_1r_bounded_shadow_100_report.json"),
    )
    assert result["claims_replayed"] == 100
    assert result["warning_events_migrated"] == 36
    assert result["persisted_provider_calls_in_inputs"] == 100
    assert result["provider_calls_during_replay"] == 0
    # One previously "resolved" item was a human-reviewed true attachment
    # ambiguity; Revision 3 intentionally retains that conservative blocker.
    assert result["previously_31_qualifier_blocked_claims_resolved_by_replay"] == 28
    revision2 = json.loads(
        Path("artifacts/v2_1r_qualifier_semantic_preservation_revision2_replay.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(revision2["eligible_claim_ids"]).issubset(set(result["eligible_claim_ids"]))
    assert result["not_evaluated_invalid_or_abstention_counts"] == {
        "UNRESOLVED_DECOMPOSITION": 3,
        "COMPOSITION_SEMANTIC_INVALID": 4,
        "COMPOSITION_REFERENCE_INVALID": 1,
        "SCHEMA_INVALID": 1,
    }
