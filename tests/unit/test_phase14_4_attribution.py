"""Unit tests for the Phase 14.4 enrichment-attribution analyzer (read-only)."""

from scripts.analyze_phase14_4_enrichment_attribution import (
    CANDIDATE_SKIPPED_EVENT,
    FEEDBACK_BLOCKED_EVENT,
    FUNNEL_STAGES,
    PROVIDER_FAILED_EVENT,
    SEARCH_REUSED_EVENT,
    SEARCH_STARTED_EVENT,
    SOURCE_SPACE_EXHAUSTED_EVENT,
    build_funnel,
    classify_suppression,
    compute_context_stickiness,
    compute_cost_efficiency,
    compute_enrichment_overlap,
    compute_failure_migration,
    compute_incremental_value,
    decide_cases,
    detect_deterministic_bottleneck,
    funnel_first_loss,
    hint_present,
    pair_query_outcomes,
    validate_arms,
)


def test_hint_present_requires_whole_token_for_ascii_hints() -> None:
    assert hint_present("paper", "we read a paper about benchmarks")
    assert not hint_present("paper", "newspaper coverage")
    assert hint_present("论文", "数据集 论文 性能结果")


def _stage_counts(**overrides: int) -> dict[str, int]:
    counts = {label: 0 for label, _source in FUNNEL_STAGES}
    counts.update(overrides)
    return counts


def test_funnel_conversion_is_none_for_zero_denominator() -> None:
    counts = _stage_counts(closure_feedback=4, research_need_refined=4)
    funnel = build_funnel({"baseline": counts})
    rows = {row["stage"]: row for row in funnel["baseline"]}
    assert rows["closure_feedback"]["conversion_rate"] is None
    assert rows["research_need_refined"]["conversion_rate"] == 1.0
    # context recorded dropped to zero -> 0.0 conversion, then the next stage
    # has a zero denominator and must stay None (not a fake 0.0).
    assert rows["research_context_recorded"]["conversion_rate"] == 0.0
    assert rows["research_context_recorded"]["dropoff_rate"] == 1.0
    assert rows["query_enrichment_attempt"]["conversion_rate"] is None


def test_funnel_first_loss_skips_undefined_conversions() -> None:
    baseline = _stage_counts(
        closure_feedback=10,
        research_need_refined=10,
        research_context_recorded=10,
        query_enrichment_attempt=10,
        query_actually_changed=10,
        search_query_started=10,
        provider_execution=10,
        search_completed=10,
        candidate_urls=10,
        readable_source=10,
        evidence_extraction=10,
        candidate_evidence=10,
        accepted_evidence=10,
        evidence_alignment=10,
        closure_evaluation=10,
        gap_transition=10,
    )
    candidate = dict(baseline)
    candidate["search_completed"] = 4
    funnel = build_funnel({"baseline": baseline, "candidate": candidate})
    loss = funnel_first_loss(funnel)
    assert loss["stage"] == "search_completed"
    assert loss["conversion_delta"] == -0.6


def test_enrichment_overlap_flags_no_op_and_full_overlap() -> None:
    enrichments = [
        {
            "question_id": "q2",
            "research_need_id": "need-1",
            "original_query": "defect detection benchmark evaluation paper experiment",
            "enriched_query": "defect detection benchmark evaluation paper experiment",
            "applied_hints": [],
            "hint_count": 4,
            "applied_hint_count": 0,
            "query_changed": False,
        },
        {
            "question_id": "q2",
            "research_need_id": "need-1",
            "original_query": "defect detection 精度结果",
            "enriched_query": "defect detection 精度结果 benchmark evaluation paper experiment",
            "applied_hints": [],
            "hint_count": 4,
            "applied_hint_count": 4,
            "query_changed": True,
        },
    ]
    candidate_queries = [
        {
            "question_id": "q2",
            "query_text": "q2:d1 primary source benchmark evaluation paper experiment",
            "context_hints": ["benchmark", "evaluation", "paper", "experiment"],
        }
    ]
    hints_by_need = {"need-1": ["benchmark", "evaluation", "paper", "experiment"]}
    result = compute_enrichment_overlap(enrichments, candidate_queries, hints_by_need)
    assert result["total_enrichment_attempts"] == 2
    assert result["no_op_count"] == 1
    assert result["query_changed_count"] == 1
    assert result["exact_no_op_rate"] == 0.5
    assert result["avg_added_hint_count"] == 2.0
    assert result["median_added_hint_count"] == 2.0
    assert result["mean_overlap_ratio"] == 1.0
    assert result["full_overlap_rate"] == 1.0
    assert result["fully_novel_enrichment_rate"] == 0.0
    assert result["unique_hint_contribution"] == []
    assert result["derived_hint_events"] == 2
    assert result["context_missing_events"] == 0


def test_enrichment_overlap_zero_attempts_is_safe() -> None:
    result = compute_enrichment_overlap([], [], {})
    assert result["total_enrichment_attempts"] == 0
    assert result["exact_no_op_rate"] == 0.0
    assert result["partial_overlap_rate"] == 0.0
    assert result["fully_novel_enrichment_rate"] == 0.0
    assert result["mean_overlap_ratio"] == 0.0
    assert result["vocabulary_size"] == 0


def test_enrichment_overlap_tolerates_missing_context_history() -> None:
    enrichments = [
        {
            "question_id": "q9",
            "research_need_id": "missing-need",
            "original_query": "x",
            "enriched_query": "x benchmark",
            "applied_hints": [],
            "hint_count": 2,
            "applied_hint_count": 1,
            "query_changed": True,
        }
    ]
    result = compute_enrichment_overlap(enrichments, [], {})
    assert result["context_missing_events"] == 1
    assert result["derived_hint_events"] == 0
    assert result["exact_no_op_rate"] == 0.0


def test_failure_migration_detects_bucket_shift() -> None:
    baseline_rows = [
        {"accepted": True, "rejection_reason": ""},
        {"accepted": False, "rejection_reason": "source_role_mismatch"},
        {"accepted": False, "rejection_reason": "source_role_mismatch"},
        {"accepted": False, "rejection_reason": "source_role_mismatch"},
        {"accepted": False, "rejection_reason": "evidence_relevance_below_threshold"},
    ]
    candidate_rows = [
        {"accepted": True, "rejection_reason": ""},
        {"accepted": False, "rejection_reason": "source_role_mismatch"},
        {"accepted": False, "rejection_reason": "evidence_relevance_below_threshold"},
        {"accepted": False, "rejection_reason": "evidence_relevance_below_threshold"},
        {"accepted": False, "rejection_reason": "evidence_relevance_below_threshold"},
    ]
    result = compute_failure_migration(baseline_rows, candidate_rows, [], [])
    assert result["migration_detected"] is True
    directions = {row["bucket"]: row["direction"] for row in result["migrations"]}
    assert directions["DIMENSION_MISMATCH"] == "increased"
    assert directions["SOURCE_QUALITY_LOW"] == "decreased"
    assert result["baseline_evidence"]["acceptance_rate"] == 0.2
    # feedback history missing -> zero denominators stay at 0.0
    assert result["baseline_feedback"]["events"] == 0
    assert result["candidate_feedback"]["buckets"]["CLAIM_NOT_SUPPORTED"]["share"] == 0.0
    # no enriched rows supplied -> the enrichment chain stays absent
    assert "candidate_enriched_evidence" not in result


def test_failure_migration_reports_enriched_chain() -> None:
    baseline_rows = [{"accepted": True, "rejection_reason": ""}]
    candidate_rows = [{"accepted": False, "rejection_reason": "source_role_mismatch"}]
    enriched_rows = [
        {"accepted": False, "rejection_reason": "evidence_relevance_below_threshold"},
        {"accepted": True, "rejection_reason": ""},
    ]
    result = compute_failure_migration(
        baseline_rows, candidate_rows, [], [], enriched_rows
    )
    chain = result["candidate_enriched_evidence"]
    assert chain["rows"] == 2
    assert chain["accepted"] == 1
    assert chain["acceptance_rate"] == 0.5
    assert chain["rejection_buckets"]["DIMENSION_MISMATCH"]["count"] == 1


def test_pair_query_outcomes_weighted_avg_delta() -> None:
    core = "cnn transformer vision language surface"
    baseline_queries = [
        {
            "question_id": "q1",
            "query": f"{core} adaptation benchmark",
            "result_count": 10,
            "unique_urls": 10,
            "reused": False,
        },
        {
            "question_id": "q1",
            "query": f"{core} adaptation benchmark",
            "result_count": 20,
            "unique_urls": 20,
            "reused": False,
        },
    ]
    candidate_queries = [
        {
            "question_id": "q1",
            "query": f"{core} adaptation benchmark",
            "result_count": 12,
            "unique_urls": 12,
            "reused": False,
        }
    ]
    result = pair_query_outcomes(baseline_queries, candidate_queries)
    assert result["paired_core_count"] == 1
    assert result["paired_baseline_avg_results"] == 15.0
    assert result["paired_candidate_avg_results"] == 12.0
    assert result["paired_avg_result_delta"] == -3.0


def test_pair_query_outcomes_zero_pairs_is_safe() -> None:
    result = pair_query_outcomes([], [])
    assert result["paired_core_count"] == 0
    assert result["paired_avg_result_delta"] == 0.0
    assert result["paired_result_delta"] == 0


def test_classify_suppression_uses_event_order_and_reuse_flags() -> None:
    events = [
        {
            "run_seq": 1,
            "event_type": SEARCH_STARTED_EVENT,
            "refs": {"question_id": "q1"},
            "metrics": {"reused": False},
        },
        {
            "run_seq": 2,
            "event_type": SEARCH_STARTED_EVENT,
            "refs": {"question_id": "q2"},
            "metrics": {"reused": True},
        },
        {
            "run_seq": 3,
            "event_type": SEARCH_REUSED_EVENT,
            "refs": {"question_id": "q2"},
            "metrics": {},
        },
        {
            "run_seq": 4,
            "event_type": SOURCE_SPACE_EXHAUSTED_EVENT,
            "refs": {"question_id": "q1"},
            "metrics": {},
        },
        {
            "run_seq": 5,
            "event_type": FEEDBACK_BLOCKED_EVENT,
            "refs": {"question_id": "q1", "stop_reason": "feedback_recovery_limit"},
            "metrics": {"feedback_execution_count": 2},
        },
        {
            "run_seq": 6,
            "event_type": CANDIDATE_SKIPPED_EVENT,
            "refs": {"reason": "duplicate"},
            "metrics": {},
        },
        {
            "run_seq": 7,
            "event_type": PROVIDER_FAILED_EVENT,
            "refs": {},
            "metrics": {"error_code": "timeout"},
        },
    ]
    usage = {
        "source_space_exhausted_by_question": {"q1": True, "q4": False},
        "query_strategy_exhausted_by_question": {"q1": True},
        "research_stop_reason": "deadline_exhausted",
    }
    result = classify_suppression(events, usage)
    assert result["started_new_searches"] == 1
    assert result["started_reused_searches"] == 1
    assert result["buckets"]["reuse_only"] == 1
    assert result["buckets"]["query_already_executed"] == 1
    assert result["buckets"]["source_space_exhausted"] == 1
    assert result["buckets"]["feedback_execution_limit"] == 1
    assert result["buckets"]["duplicate_query"] == 1
    assert result["buckets"]["provider_degraded"] == 1
    assert result["buckets"]["logical_query_budget_exhausted"] == 1
    assert result["buckets"]["search_budget_exhausted"] == 1
    assert result["evidence"]["feedback_execution_limit"][0].startswith("seq=5")
    assert result["exhaustion_run_seqs"] == [4]


def test_classify_suppression_empty_events_is_all_zero() -> None:
    result = classify_suppression([], {})
    assert all(value == 0 for value in result["buckets"].values())
    assert result["started_new_searches"] == 0
    assert result["stop_reason"] == ""


def test_context_stickiness_flags_usage_after_newer_need() -> None:
    enrichments = [
        {"run_seq": 100, "question_id": "q2", "research_need_id": "old-need"},
        {"run_seq": 500, "question_id": "q2", "research_need_id": "old-need"},
    ]
    context_records = [{"run_seq": 90, "question_id": "q2", "research_need_id": "old-need"}]
    need_events = [
        {"run_seq": 80, "question_id": "q2", "need_id": "old-need"},
        {"run_seq": 300, "question_id": "q2", "need_id": "new-need"},
    ]
    result = compute_context_stickiness(enrichments, context_records, need_events)
    assert result["stale_context_usage_count"] == 1
    assert result["stale_examples"][0]["newer_need_run_seq"] == 300
    assert result["stale_examples"][0]["used_need_id"] == "old-need"
    assert result["avg_uses_per_need"] == 2.0
    assert result["context_records"] == 1


def test_context_stickiness_without_newer_needs_is_clean() -> None:
    enrichments = [{"run_seq": 100, "question_id": "q3", "research_need_id": "need-a"}]
    context_records = [{"run_seq": 90, "question_id": "q3", "research_need_id": "need-a"}]
    need_events = [{"run_seq": 80, "question_id": "q3", "need_id": "need-a"}]
    result = compute_context_stickiness(enrichments, context_records, need_events)
    assert result["stale_context_usage_count"] == 0
    assert result["stale_examples"] == []


def test_deterministic_bottleneck_pins_only_identical_scalars() -> None:
    def make_run(searches: int, feedback: int, accepted: int) -> dict[str, object]:
        return {
            "usage": {
                "searches": searches,
                "feedback_execution_count": feedback,
                "source_space_exhausted_by_question": {"q1": True},
            },
            "quality": {"coverage": 0.3929, "accepted_evidence": accepted, "critical_gaps": 3},
        }

    runs = [make_run(20, 2, 32), make_run(20, 2, 32), make_run(20, 3, 30)]
    result = detect_deterministic_bottleneck(runs)
    pinned = {row["field"] for row in result["pinned_fields"]}
    assert "searches" in pinned
    assert "source_space_exhausted_by_question" in pinned
    assert "feedback_execution_count" not in pinned
    assert "accepted_evidence" not in pinned
    assert result["coverage_sd_zero"] is True
    assert result["coverage_values"] == [0.3929, 0.3929, 0.3929]
    assert any(
        "source_space_exhausted_by_question" in item
        for item in result["bottleneck_candidates"]
    )


def test_deterministic_bottleneck_requires_two_runs() -> None:
    result = detect_deterministic_bottleneck([{"usage": {}, "quality": {}}])
    assert result["pinned_fields"] == []
    assert result["coverage_sd_zero"] is False
    assert result["bottleneck_candidates"] == []


def test_incremental_value_not_estimable_without_attempts() -> None:
    result = compute_incremental_value({"total_enrichment_attempts": 0}, {}, {})
    assert result["estimable"] is False
    assert result["14_1_only_adaptation_coverage"] is None


def test_incremental_value_reports_full_14_1_coverage_without_unique_hints() -> None:
    overlap = {
        "total_enrichment_attempts": 2,
        "no_op_count": 1,
        "unique_hint_contribution": [],
        "vocabulary_size": 4,
    }
    result = compute_incremental_value(
        overlap, {"candidate": {}, "baseline": {}}, {"paired_result_delta": -3}
    )
    assert result["estimable"] is True
    assert result["14_1_only_adaptation_coverage"] == 1.0
    assert result["14_2_additional_adaptation_coverage"] == 0.0
    assert result["14_2_no_op_rate"] == 0.5


def test_cost_efficiency_zero_denominator_is_safe() -> None:
    empty_run: dict[str, object] = {"usage": {}, "quality": {}, "counters": {}}
    result = compute_cost_efficiency([empty_run], [empty_run])
    assert result["candidate_per_unit"]["accepted_evidence_per_search"] == 0.0
    assert result["candidate_per_unit"]["runtime_per_accepted_evidence"] == 0.0
    assert result["verdict"] == "no cost advantage and no efficiency gain"


def test_cost_efficiency_reads_stage_counts_for_funnel_metrics() -> None:
    run: dict[str, object] = {
        "usage": {"searches": 10, "model_tokens": 1000},
        "quality": {"accepted_evidence": 5, "coverage": 0.5},
        "stage_counts": {
            "search_completed": 10,
            "readable_source": 6,
            "evidence_extraction": 4,
            "candidate_evidence": 4,
        },
    }
    result = compute_cost_efficiency([run], [run])
    totals = result["baseline_totals"]
    assert totals["search_completed"] == 10
    assert totals["readable_sources"] == 6
    assert totals["evidence_extracted"] == 4
    assert totals["candidate_evidence"] == 4
    assert result["baseline_per_unit"]["readable_per_search"] == 0.6


def test_decide_cases_prefers_redundancy_then_gating() -> None:
    overlap = {
        "exact_no_op_rate": 0.1333,
        "unique_hint_contribution": [],
        "mean_overlap_ratio": 1.0,
        "context_recorded_hint_sets": [
            ("benchmark", "evaluation", "paper", "experiment")
        ],
    }
    suppression = {
        "candidate": {"source_space_exhausted": 11, "reuse_only": 42},
        "baseline": {"source_space_exhausted": 4, "reuse_only": 43},
    }
    migration = {
        "migration_detected": True,
        "candidate_evidence": {
            "rejection_buckets": {
                "DIMENSION_MISMATCH": {"count": 5},
                "SOURCE_QUALITY_LOW": {"count": 9},
            }
        },
    }
    specificity = {"paired_result_delta": -12}
    cost = {
        "baseline_totals": {"accepted_evidence": 43},
        "candidate_totals": {"accepted_evidence": 32},
    }
    result = decide_cases(
        overlap, suppression, migration, {"bottleneck_candidates": []}, specificity, cost
    )
    assert result["primary_case"] == 1
    assert result["secondary_case"] == 4
    assert [signal["case"] for signal in result["all_signals"]] == [1, 4, 2]


def test_decide_cases_defaults_to_case_one_without_signals() -> None:
    result = decide_cases(
        {
            "exact_no_op_rate": 0.0,
            "unique_hint_contribution": ["sensor"],
            "mean_overlap_ratio": 0.0,
        },
        {"candidate": {}, "baseline": {}},
        {"migration_detected": False},
        {"bottleneck_candidates": []},
        {"paired_result_delta": 5},
        {"baseline_totals": {}, "candidate_totals": {}},
    )
    assert result["primary_case"] == 1
    assert "no decisive attribution signal" in result["primary_rationale"]


def test_validate_arms_reports_size_mismatch_and_cross_arm_ids() -> None:
    problems = validate_arms(["a", "b", "c"], ["c", "d"])
    assert any("mismatch" in problem for problem in problems)
    assert any("both arms" in problem for problem in problems)
    assert any("duplicate" in problem for problem in problems)
    assert validate_arms(["a"], ["b"]) == []


def test_validate_arms_rejects_empty_arms() -> None:
    problems = validate_arms([], ["b"])
    assert "baseline arm has no run ids" in problems
    problems = validate_arms(["a"], [])
    assert "candidate arm has no run ids" in problems
