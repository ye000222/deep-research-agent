"""Unit tests for the historical baseline regression audit analyzer (read-only)."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.analyze_historical_baseline_regression import (
    build_config_diff,
    build_funnel_comparison,
    build_funnel_section,
    build_unit_comparison,
    classify_config_difference,
    decide_primary_case,
    era_aggregate,
    normalize_revision,
    replay_coverage,
    stage_counts,
    stored_map_required_summary,
    summarize_definition_diff,
    validate_run_sets,
)


def _plan_items() -> list[dict[str, Any]]:
    return [
        {
            "question_id": "q1",
            "priority": 1,
            "evidence_requirements": ["技术现状概述", "两个独立来源的对比数据"],
            "plan_version": 1,
        },
        {
            "question_id": "q2",
            "priority": 2,
            "evidence_requirements": ["市场规模估算"],
            "plan_version": 1,
        },
    ]


def _claims() -> list[dict[str, Any]]:
    return [
        {
            "id": "c1",
            "question_id": "q1",
            "dimension_key": "q1:d1",
            "claim_type": "fact",
            "importance": 0.5,
            "status": "supported",
        },
        {
            "id": "c2",
            "question_id": "q1",
            "dimension_key": "q1:d2",
            "claim_type": "comparative",
            "importance": 0.6,
            "status": "supported",
        },
        {
            "id": "c3",
            "question_id": "q2",
            "dimension_key": "q2:d1",
            "claim_type": "numeric",
            "importance": 0.9,
            "status": "supported",
        },
    ]


def _evidence() -> list[dict[str, Any]]:
    return [
        {"id": "e1", "claim_id": "c1", "question_id": "q1", "accepted": True, "source_id": "s1"},
        {"id": "e2", "claim_id": "c1", "question_id": "q1", "accepted": True, "source_id": "s2"},
        {"id": "e3", "claim_id": "c2", "question_id": "q1", "accepted": True, "source_id": "s3"},
        {"id": "e4", "claim_id": "c3", "question_id": "q2", "accepted": True, "source_id": "s4"},
        {"id": "e5", "claim_id": "c3", "question_id": "q2", "accepted": True, "source_id": "s5"},
        {"id": "e6", "claim_id": "c1", "question_id": "q1", "accepted": False, "source_id": "s6"},
    ]


def _sources() -> list[dict[str, Any]]:
    return [
        {"id": "s1", "source_owner_key": "owner-a"},
        {"id": "s2", "source_owner_key": "owner-a"},
        {"id": "s3", "source_owner_key": "owner-b"},
        {"id": "s4", "source_owner_key": "owner-c"},
        {"id": "s5", "source_owner_key": "owner-d"},
        {"id": "s6", "source_owner_key": "owner-e"},
    ]


def _arm_run(**overrides: int | None) -> dict[str, Any]:
    stages: dict[str, int | None] = {
        "plan_items": 10,
        "plan_dimensions": 20,
        "logical_queries": 12,
        "search_completed": 10,
        "candidate_urls": 300,
        "unique_urls": 200,
        "readable_attempts": 60,
        "unique_readable": 40,
        "reader_completed": 40,
        "chunked_snapshots": 38,
        "chunks_total": 150,
        "extraction_calls": 40,
        "candidate_evidence": 120,
        "accepted_evidence": 60,
        "alignment_completed": 60,
        "gap_transitions": 5,
    }
    for key, value in overrides.items():
        stages[key] = value
    return {"stages": stages, "found": True}


def test_validate_run_sets_separates_arms() -> None:
    assert validate_run_sets("a", ["b"], ["c", "d"]) == []
    overlap = validate_run_sets("a", ["b"], ["a"])
    assert any("both arms" in problem for problem in overlap)
    duplicates = validate_run_sets("a", ["b", "b"], ["c"])
    assert any("duplicate" in problem for problem in duplicates)
    incomplete = validate_run_sets("", [], [])
    assert len(incomplete) == 2


def test_replay_coverage_matches_current_definition() -> None:
    result = replay_coverage(_plan_items(), _claims(), _evidence(), _sources(), 1)
    assert result["coverage"] == 0.85
    assert result["accepted_evidence"] == 5
    assert result["candidate_evidence"] == 6
    assert result["dimensions"] == 3
    assert result["dimensions_requiring_two_sources"] == 2
    by_question = {entry["question_id"]: entry for entry in result["coverage_map"]}
    assert by_question["q1"]["coverage"] == 0.75
    assert by_question["q2"]["coverage"] == 1.0
    first = by_question["q1"]["requirements"][0]
    assert first["independent_sources"] == 1  # duplicate owner key collapsed
    assert first["required_sources"] == 1
    second = by_question["q1"]["requirements"][1]
    assert second["required_via_marker"] is True
    assert second["coverage"] == 0.5


def test_replay_coverage_filters_by_plan_version() -> None:
    items = [
        *_plan_items(),
        {
            "question_id": "q3",
            "priority": 1,
            "evidence_requirements": ["later replan dimension"],
            "plan_version": 2,
        },
    ]
    result = replay_coverage(items, _claims(), _evidence(), _sources(), 1)
    assert result["plan_items"] == 2
    assert result["dimensions"] == 3
    assert result["coverage"] == 0.85


def test_replay_coverage_high_risk_claim_forces_two_sources() -> None:
    plan = [
        {
            "question_id": "q9",
            "priority": 1,
            "evidence_requirements": ["neutral description"],
            "plan_version": 1,
        }
    ]
    claims = [
        {
            "id": "c9",
            "question_id": "q9",
            "dimension_key": "q9:d1",
            "claim_type": "fact",
            "importance": 0.9,
            "status": "supported",
        }
    ]
    evidence = [
        {"id": "e1", "claim_id": "c9", "question_id": "q9", "accepted": True, "source_id": "s1"}
    ]
    sources = [{"id": "s1", "source_owner_key": "o1"}]
    result = replay_coverage(plan, claims, evidence, sources, 1)
    detail = result["coverage_map"][0]["requirements"][0]
    assert detail["required_via_marker"] is False
    assert detail["required_via_high_risk_claim"] is True
    assert detail["required_sources"] == 2
    assert detail["coverage"] == 0.5
    assert result["coverage"] == 0.5


def test_stored_map_missing_required_field_is_counted_not_guessed() -> None:
    stored_map = [
        {
            "question_id": "q1",
            "requirement_statuses": [
                {"dimension_key": "q1:d1", "coverage": 1.0},
                {"dimension_key": "q1:d2", "coverage": 1.0, "required_sources": 1},
            ],
        }
    ]
    summary = stored_map_required_summary(stored_map)
    assert summary["dimensions"] == 2
    assert summary["field_present"] == 1
    assert summary["required_value_counts"] == {"1": 1}
    assert summary["coverage_value_counts"] == {"1.0": 2}


def test_stage_counts_missing_data_stays_none() -> None:
    counts = stage_counts({"usage": {}, "event_counts": {}, "replay": {}, "storage": {}})
    assert counts["readable_attempts"] is None
    assert counts["unique_readable"] is None
    assert counts["search_completed"] is None
    assert counts["accepted_evidence"] is None
    assert counts["plan_items"] is None


def test_stage_counts_derives_unique_readable_from_duplicate_events() -> None:
    counts = stage_counts(
        {
            "usage": {},
            "event_counts": {"source.readable": 10, "source.duplicate_skipped": 4},
            "replay": {},
            "storage": {},
        }
    )
    assert counts["readable_attempts"] == 10
    assert counts["unique_readable"] == 6


def test_funnel_conversion_and_first_divergence_kind_behaviour() -> None:
    historical = [_arm_run(), _arm_run()]
    current = [_arm_run(unique_readable=30), _arm_run(unique_readable=30)]
    section = build_funnel_section(historical, current)
    rows = {row["stage"]: row for row in section["stages"]}
    assert rows["plan_dimensions"]["conversion_historical"] == 2.0
    assert rows["plan_dimensions"]["conversion_current"] == 2.0
    assert section["first_divergence"] is not None
    assert section["first_divergence"]["stage"] == "unique_readable"
    assert section["first_divergence"]["kind"] == "behaviour"
    assert rows["unique_readable"]["relative_delta"] == -0.25


def test_funnel_first_divergence_task_shape_is_config() -> None:
    section = build_funnel_section([_arm_run()], [_arm_run(plan_items=5, plan_dimensions=10)])
    assert section["first_divergence"] is not None
    assert section["first_divergence"]["stage"] == "plan_items"
    assert section["first_divergence"]["kind"] == "config"


def test_funnel_comparison_skips_non_comparable_stages() -> None:
    rows = build_funnel_comparison({"plan_items": None}, {"plan_items": 5})
    first = rows[0]
    assert first["stage"] == "plan_items"
    assert first["comparable"] is False
    assert first["divergent"] is False
    assert first["delta"] is None
    assert first["relative_delta"] is None


def test_unit_comparison_zero_denominator_is_none() -> None:
    rows = build_unit_comparison(
        {"plan_dimensions": 5, "logical_queries": 5, "search_completed": 0, "unique_urls": 3},
        {"plan_dimensions": 5, "logical_queries": 5, "search_completed": 0, "unique_urls": 3},
    )
    metrics = {row["metric"]: row for row in rows}
    assert metrics["unique_urls_per_search"]["historical"] is None
    assert metrics["searches_per_dimension"]["historical"] == 1.0


def test_definition_diff_detects_required_sources_drift() -> None:
    replay = replay_coverage(_plan_items(), _claims(), _evidence(), _sources(), 1)
    run = {
        "run_id": "hist-1",
        "card": {
            "stored_coverage": 0.9,
            "stored_coverage_map": [
                {
                    "question_id": "q1",
                    "requirement_statuses": [
                        {"dimension_key": "q1:d1", "coverage": 1.0, "required_sources": 1},
                        {"dimension_key": "q1:d2", "coverage": 1.0},
                    ],
                },
                {
                    "question_id": "q2",
                    "requirement_statuses": [
                        {"dimension_key": "q2:d1", "coverage": 1.0, "required_sources": 1}
                    ],
                },
            ],
        },
        "replay": replay,
        "coverage_match": False,
    }
    summary = summarize_definition_diff(run)
    assert summary["stored_dimensions"] == 3
    assert summary["stored_required_field_absent"] == 1
    assert summary["required_sources_drift_dimensions"] == 1
    assert summary["examples"][0]["dimension_key"] == "q2:d1"
    assert summary["examples"][0]["stored_required_sources"] == 1
    assert summary["examples"][0]["current_required_sources"] == 2


def test_config_diff_classifies_and_counts_semantic_differences() -> None:
    historical = [
        {
            "run_id": "h1",
            "card": {"normalized_goal": "goal-A", "scoring_rule_version": "rules-2"},
        }
    ]
    current = [
        {
            "run_id": "c1",
            "card": {"normalized_goal": "goal-B", "scoring_rule_version": "rules-2"},
        }
    ]
    diff = build_config_diff(historical, current, {})
    fields = {row["field"]: row for row in diff["rows"]}
    assert fields["normalized_goal"]["equal"] is False
    assert fields["normalized_goal"]["classification"] == "research_semantic"
    assert fields["scoring_rule_version"]["equal"] is True
    assert fields["source_revision"]["equal"] is True
    assert fields["profile"]["classification"] == "unknown"
    assert fields["profile"]["historical"] is None
    assert diff["research_semantic_difference_count"] == 1


def test_classify_config_difference_rejects_unknown_classification() -> None:
    with pytest.raises(ValueError):
        classify_config_difference("f", 1, 2, "made-up")


def test_decide_primary_case_insufficient_data() -> None:
    diagnosis = decide_primary_case({})
    assert diagnosis["case"] is None
    assert diagnosis["status"] == "insufficient_data"


def test_decide_primary_case_a_metric_definition_drift() -> None:
    diagnosis = decide_primary_case({"hist_replay_mean": 0.46, "cur_replay_mean": 0.45})
    assert diagnosis["case"] == "A"
    assert diagnosis["status"] == "decided"


def test_decide_primary_case_a_with_recorded_mismatch() -> None:
    diagnosis = decide_primary_case(
        {
            "hist_replay_mean": 0.50,
            "cur_replay_mean": 0.45,
            "formula_match_all": False,
            "formula_mismatch_runs": ["old-run"],
        }
    )
    assert diagnosis["case"] == "A"


def test_decide_primary_case_b_same_task_degradation() -> None:
    diagnosis = decide_primary_case(
        {
            "hist_replay_mean": 0.85,
            "cur_replay_mean": 0.45,
            "behaviour_divergence_count": 3,
            "config_semantic_diff_count": 1,
            "same_task": {"hist_mean": 0.85, "cur_mean": 0.45, "hist_n": 6},
        }
    )
    assert diagnosis["case"] == "B"


def test_decide_primary_case_d_config_with_flat_same_task() -> None:
    diagnosis = decide_primary_case(
        {
            "hist_replay_mean": 0.85,
            "cur_replay_mean": 0.45,
            "behaviour_divergence_count": 0,
            "config_semantic_diff_count": 3,
            "same_task": {"hist_mean": 0.48, "cur_mean": 0.45, "hist_n": 8},
        }
    )
    assert diagnosis["case"] == "D"


def test_decide_primary_case_c_provider_environment() -> None:
    diagnosis = decide_primary_case(
        {
            "hist_replay_mean": 0.85,
            "cur_replay_mean": 0.45,
            "provider_environment_divergence": True,
        }
    )
    assert diagnosis["case"] == "C"


def test_era_aggregate_flags_insufficient_observations() -> None:
    rows = [
        {
            "era": "thin",
            "created_at": "2026-09-11T01:00:00",
            "stored_coverage": 0.5,
            "replayed_coverage": 0.4,
            "accepted_evidence": 10,
        },
        {
            "era": "dense",
            "created_at": "2026-09-11T01:00:00",
            "stored_coverage": 0.6,
            "replayed_coverage": 0.5,
            "accepted_evidence": 12,
        },
        {
            "era": "dense",
            "created_at": "2026-09-11T02:00:00",
            "stored_coverage": 0.7,
            "replayed_coverage": 0.6,
            "accepted_evidence": 14,
        },
    ]
    aggregates = {entry["era"]: entry for entry in era_aggregate(rows)}
    assert aggregates["thin"]["insufficient_observations"] is True
    assert aggregates["dense"]["insufficient_observations"] is False
    assert aggregates["dense"]["mean_replayed_coverage"] == 0.55
    assert aggregates["dense"]["mean_accepted_evidence"] == 13.0


def test_normalize_revision_buckets_eras() -> None:
    assert normalize_revision(None) == "none"
    assert normalize_revision("") == "none"
    assert normalize_revision("ea3f0635ce84abcd") == "ea3f0635ce84"
    assert normalize_revision("c1c009540b09-worktree-2025") == "c1c009540b09-worktree-*"
    assert normalize_revision("v28-search-fallback-proxy-and-bing-parser").startswith("v28-")
    assert normalize_revision("deadbeefcafe1234567890") == "deadbeefcafe"
