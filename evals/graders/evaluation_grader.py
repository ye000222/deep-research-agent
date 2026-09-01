"""Deterministic Golden Eval grader for V1 termination decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _as_int(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def grade_case(case: Mapping[str, object]) -> dict[str, object]:
    claim_items = _sequence(case.get("claims", []))
    dimensions = [item for item in claim_items if isinstance(item, Mapping)]
    covered = sum(
        1
        for item in dimensions
        if _as_int(item.get("accepted_evidence", 0)) >= 1
        and _as_int(item.get("independent_sources", 0)) >= 2
    )
    coverage = covered / len(dimensions) if dimensions else 0.0
    conflicts = _as_int(case.get("unresolved_conflicts", 0))
    blocked_flags = (
        "stale_claims",
        "scope_mismatches",
        "search_failures",
        "repeated_queries",
        "low_information_gain_rounds",
        "unsupported_citations",
        "analysis_errors",
    )
    blocked = conflicts > 0 or any(_as_int(case.get(flag, 0)) > 0 for flag in blocked_flags)
    budget_exhausted = bool(case.get("budget_exhausted", False))
    no_evidence = bool(case.get("no_verifiable_evidence", False))
    if no_evidence:
        verdict = "fail"
    elif coverage >= 0.85 and not blocked:
        verdict = "write_limited" if budget_exhausted else "write"
    else:
        verdict = "continue"
    expected = str(case.get("expected_verdict", ""))
    predicted_gaps = {
        str(item.get("dimension", ""))
        for item in dimensions
        if _as_int(item.get("accepted_evidence", 0)) < 1
        or _as_int(item.get("independent_sources", 0)) < 2
    }
    if blocked:
        predicted_gaps.update(str(item.get("dimension", "")) for item in dimensions)
    predicted_gaps.discard("")
    expected_gaps = {
        str(item) for item in _sequence(case.get("expected_gap_keys", ())) if str(item)
    }
    return {
        "case_id": str(case.get("id", "")),
        "coverage": coverage,
        "verdict": verdict,
        "expected_verdict": expected,
        "passed": verdict == expected,
        "predicted_gap_keys": sorted(predicted_gaps),
        "expected_gap_keys": sorted(expected_gaps),
        "gap_true_positives": len(predicted_gaps & expected_gaps),
        "gap_false_positives": len(predicted_gaps - expected_gaps),
        "gap_false_negatives": len(expected_gaps - predicted_gaps),
        "early_stop_error": (
            verdict in {"write", "write_limited"}
            and expected not in {"write", "write_limited"}
        ),
        "support_checks": [
            {
                "expected": bool(item.get("expected", False)),
                "predicted": bool(item.get("predicted", False)),
            }
            for item in _sequence(case.get("evidence_support_cases", ()))
            if isinstance(item, Mapping)
        ],
    }


def grade_dataset(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    results = [grade_case(case) for case in cases]
    gap_tp = sum(_as_int(item["gap_true_positives"]) for item in results)
    gap_fp = sum(_as_int(item["gap_false_positives"]) for item in results)
    gap_fn = sum(_as_int(item["gap_false_negatives"]) for item in results)
    support_checks: list[Mapping[str, object]] = []
    for item in results:
        support_checks.extend(
            check
            for check in _sequence(item.get("support_checks", ()))
            if isinstance(check, Mapping)
        )
    support_tp = sum(
        bool(item.get("expected")) and bool(item.get("predicted")) for item in support_checks
    )
    support_fp = sum(
        not bool(item.get("expected")) and bool(item.get("predicted")) for item in support_checks
    )
    support_fn = sum(
        bool(item.get("expected")) and not bool(item.get("predicted")) for item in support_checks
    )
    accuracy = (
        sum(bool(item["passed"]) for item in results) / len(results) if results else 0.0
    )
    gap_recall = gap_tp / (gap_tp + gap_fn) if gap_tp + gap_fn else 1.0
    gap_precision = gap_tp / (gap_tp + gap_fp) if gap_tp + gap_fp else 1.0
    support_precision = (
        support_tp / (support_tp + support_fp) if support_tp + support_fp else 1.0
    )
    support_recall = (
        support_tp / (support_tp + support_fn) if support_tp + support_fn else 1.0
    )
    early_stop_error_rate = (
        sum(bool(item["early_stop_error"]) for item in results) / len(results)
        if results
        else 0.0
    )
    return {
        "total": len(results),
        "passed": sum(bool(item["passed"]) for item in results),
        "accuracy": accuracy,
        "stop_decision_accuracy": accuracy,
        "gap_recall": gap_recall,
        "gap_precision": gap_precision,
        "evidence_support_precision": support_precision,
        "evidence_support_recall": support_recall,
        "early_stop_error_rate": early_stop_error_rate,
        "release_gate_passed": (
            accuracy >= 0.90
            and gap_recall >= 0.85
            and support_precision >= 0.95
            and early_stop_error_rate <= 0.05
        ),
        "results": results,
    }
