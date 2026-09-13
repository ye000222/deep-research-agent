from dataclasses import replace

import pytest

from evals.graders.performance_grader import PerformanceSample, compare_samples


def test_fast_failure_cannot_pass_as_optimization() -> None:
    old = PerformanceSample("a", 0, "same-config", 1000, 100, 0.9, 1, True, True)
    new = replace(old, elapsed_ms=1, total_tokens=1, technical_success=False, quality_success=False)
    report = compare_samples([old], [new])
    assert report["observed_quality_nonregression"] is False
    assert report["release_approved"] is False


def test_missing_failure_and_changed_configuration_are_rejected() -> None:
    old = PerformanceSample("a", 0, "same-config", 1000, 100, 0.9, 1, True, True)
    with pytest.raises(ValueError):
        compare_samples([old], [replace(old, case_id="b")])
    with pytest.raises(ValueError):
        compare_samples([old], [replace(old, config_hash="other-model")])


def test_duplicate_pair_is_rejected() -> None:
    old = PerformanceSample("a", 0, "same-config", 1000, 100, 0.9, 1, True, True)
    with pytest.raises(ValueError):
        compare_samples([old, old], [old, old])


@pytest.mark.parametrize("change", [
    {"technical_success": "false"}, {"total_tokens": True},
    {"repetition": 0.5}, {"elapsed_ms": "12"}, {"coverage": float("nan")},
    {"technical_success": False}, {"case_id": "  "},
])
def test_invalid_measurements_are_rejected(change: dict) -> None:
    old = PerformanceSample("a", 0, "same-config", 1000, 100, 0.9, 1, True, True)
    with pytest.raises(ValueError):
        replace(old, **change)


def test_repeated_question_is_one_cluster_and_result_is_reproducible() -> None:
    old = PerformanceSample("a", 0, "same-config", 1000, 100, 0.5, 1, True, True)
    baseline = [replace(old, repetition=i) for i in range(30)]
    candidate = [replace(s, coverage=0.8) for s in baseline]
    report = compare_samples(baseline, candidate)
    assert report["case_count"] == 1
    assert report["minimum_case_count_met"] is False
    assert report["clustered_intervals"]["coverage"]["lower_95"] is None
    assert report == compare_samples(baseline, candidate)


def test_case_weighting_does_not_overweight_many_repeats() -> None:
    old = PerformanceSample("a", 0, "same-config", 1000, 100, 0.5, 1, True, True)
    baseline = [replace(old, repetition=i) for i in range(20)] + [replace(old, case_id="b")]
    candidate = [replace(s, coverage=1 if s.case_id == "a" else 0) for s in baseline]
    report = compare_samples(baseline, candidate)
    interval = report["clustered_intervals"]["coverage"]
    assert interval["mean_paired_delta"] == 0
    assert interval["lower_95"] < 0 < interval["upper_95"]
    assert report["release_approved"] is False
