from __future__ import annotations

import json
from pathlib import Path

from app.services.atomic_proposition_validation import (
    QualifierCheckStatus,
    inspect_qualifier_fields,
)


def _checks(claim: str, *propositions: dict[str, object]) -> dict[str, object]:
    return {item.category: item for item in inspect_qualifier_fields(claim, list(propositions))}


def test_date_range_endpoint_is_equivalent_but_wrong_event_date_is_not() -> None:
    range_result = _checks(
        "The plan covers 2026—2030年.",
        {
            "proposition_text": "The plan covers 2026 to 2030.",
            "subject": "plan",
            "predicate": "covers",
            "source_span": "The plan covers 2026—2030年",
            "material_qualifiers": [
                {
                    "category": "period",
                    "value": "2026 to 2030",
                    "source_span": "2026—2030年",
                }
            ],
        },
    )["date_time"]
    wrong_event = _checks(
        "Device A launched in 2026; Device B launched in 2030.",
        {
            "proposition_text": "Device A launched in 2030.",
            "subject": "Device A",
            "predicate": "launched",
            "source_span": "Device A launched in 2026",
            "material_qualifiers": [
                {"category": "date", "value": "2030", "source_span": "2026"}
            ],
        },
        {
            "proposition_text": "Device B launched in 2030.",
            "subject": "Device B",
            "predicate": "launched",
            "source_span": "Device B launched in 2030",
            "material_qualifiers": [
                {"category": "date", "value": "2030", "source_span": "2030"}
            ],
        },
    )["date_time"]
    assert range_result.status is QualifierCheckStatus.PRESERVED
    assert wrong_event.status is QualifierCheckStatus.POTENTIAL_LOSS


def test_percent_metric_suffix_requires_same_number_and_source_metric_context() -> None:
    claim = "System reached 99.3% AP on the image-level task."
    correct = _checks(
        claim,
        {
            "proposition_text": "System reached 99.3% AP on the image-level task.",
            "subject": "System",
            "predicate": "reached",
            "source_span": claim,
            "typed_value": {
                "kind": "quantity",
                "numeric": {"operator": "exact", "value": 99.3, "unit": "% AP"},
            },
        },
    )["unit"]
    wrong_number = _checks(
        claim,
        {
            "proposition_text": "System reached 98.3% AP on the image-level task.",
            "subject": "System",
            "predicate": "reached",
            "source_span": claim,
            "typed_value": {
                "kind": "quantity",
                "numeric": {"operator": "exact", "value": 98.3, "unit": "% AP"},
            },
        },
    )["unit"]
    wrong_metric = _checks(
        claim,
        {
            "proposition_text": "System reached 99.3% F1 on the image-level task.",
            "subject": "System",
            "predicate": "reached",
            "source_span": claim,
            "typed_value": {
                "kind": "quantity",
                "numeric": {"operator": "exact", "value": 99.3, "unit": "% F1"},
            },
        },
    )["unit"]
    assert correct.status is QualifierCheckStatus.PRESERVED
    assert wrong_number.status is QualifierCheckStatus.POTENTIAL_LOSS
    assert wrong_metric.status is QualifierCheckStatus.POTENTIAL_LOSS


def test_method_relation_object_may_be_carried_by_subject_or_method_field() -> None:
    claim = "The system is based on model Alpha for inspection."
    preserved = _checks(
        claim,
        {
            "proposition_text": "The system is based on model Alpha for inspection.",
            "subject": "system based on model Alpha",
            "predicate": "is used for inspection",
            "source_span": claim,
            "material_qualifiers": [
                {
                    "category": "method",
                    "value": "based on model Alpha",
                    "source_span": "based on model Alpha",
                }
            ],
        },
    )["method"]
    rejected = _checks(
        "The system uses method Alpha for inspection.",
        {
            "proposition_text": "The system uses method Beta for inspection.",
            "subject": "system",
            "predicate": "uses",
            "source_span": "The system uses method Alpha for inspection",
            "material_qualifiers": [
                {"category": "method", "value": "method Beta", "source_span": "uses method Alpha"}
            ],
        },
    )["method"]
    assert preserved.status is QualifierCheckStatus.PRESERVED
    assert rejected.status is QualifierCheckStatus.POTENTIAL_LOSS


def test_comparison_target_without_group_wide_source_attachment_stays_ambiguous() -> None:
    claim = "Compared with baseline system A reduced latency and memory use."
    shared = {"kind": "comparison", "comparison_target": "baseline"}
    result = _checks(
        claim,
        {
            "proposition_text": "System A reduced latency compared with baseline.",
            "subject": "system A",
            "predicate": "reduced latency",
            "source_span": "Compared with baseline system A reduced latency",
            "typed_value": shared,
        },
        {
            "proposition_text": "System A reduced memory use compared with baseline.",
            "subject": "system A",
            "predicate": "reduced memory use",
            "source_span": "memory use",
            "material_qualifiers": [
                {
                    "category": "comparison",
                    "value": "baseline",
                    "source_span": "Compared with baseline",
                }
            ],
            "typed_value": shared,
        },
    )["comparison"]
    # The source-side cue context ends before the second predicate. Do not infer
    # that the comparison target scopes over both propositions from output alone.
    assert result.status is QualifierCheckStatus.AMBIGUOUS


def test_source_attachment_uses_occurrence_position_for_repeated_marker() -> None:
    claim = "The study uses method Alpha. Later it uses method Beta."
    result = _checks(
        claim,
        {
            "proposition_text": "The study uses method Alpha.",
            "subject": "study",
            "predicate": "uses",
            "source_span": "The study uses method Alpha",
            "material_qualifiers": [
                {"category": "method", "value": "method Alpha", "source_span": "uses method Alpha"}
            ],
        },
        {
            "proposition_text": "Later it uses method Beta.",
            "subject": "study",
            "predicate": "uses",
            "source_span": "Later it uses method Beta",
            "material_qualifiers": [
                {"category": "method", "value": "method Beta", "source_span": "uses method Beta"}
            ],
        },
    )["method"]
    assert result.status is QualifierCheckStatus.PRESERVED


def test_embedded_cjk_efficiency_and_probability_are_not_bare_rate_obligations() -> None:
    efficiency = _checks(
        "The goal is to analyze data efficiency.",
        {
            "proposition_text": "The goal is to analyze data efficiency.",
            "subject": "goal",
            "predicate": "analyze data efficiency",
            "source_span": "The goal is to analyze data efficiency",
        },
    )["measurement"]
    probability = _checks(
        "Training uses equal probability to select a real caption or an embedding.",
        {
            "proposition_text": (
                "Training uses equal probability to select a real caption or an embedding."
            ),
            "subject": "training",
            "predicate": "uses equal probability to select",
            "source_span": (
                "Training uses equal probability to select a real caption or an embedding"
            ),
        },
    )["measurement"]
    assert efficiency.status is QualifierCheckStatus.NOT_APPLICABLE
    assert probability.status is QualifierCheckStatus.NOT_APPLICABLE


def test_comparison_target_noun_does_not_create_method_obligation() -> None:
    result = _checks(
        "The model outperforms baseline methods.",
        {
            "proposition_text": "The model outperforms baseline methods.",
            "subject": "model",
            "predicate": "outperforms",
            "typed_value": {"kind": "comparison", "comparison_target": "baseline methods"},
        },
    )["method"]
    assert result.status is QualifierCheckStatus.NOT_APPLICABLE


def test_revision2_prechange_warning_ledger_is_complete_and_offline() -> None:
    ledger = json.loads(
        Path("artifacts/v2_1r_qualifier_revision2_warning_ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["warning_count"] == 22
    assert ledger["warning_claim_count"] == 19
    assert ledger["cohort_warning_event_counts"] == {
        "PERSISTENTLY_BLOCKED": 18,
        "NEWLY_BLOCKED": 4,
    }
    assert ledger["provider_calls_during_audit"] == 0
