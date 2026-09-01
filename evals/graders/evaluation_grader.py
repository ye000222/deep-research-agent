"""Deterministic Golden Eval grader for V1 termination decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def grade_case(case: Mapping[str, object]) -> dict[str, object]:
    claims = case.get("claims", [])
    claim_items = claims if isinstance(claims, Sequence) else []
    dimensions = [item for item in claim_items if isinstance(item, Mapping)]
    covered = sum(
        1
        for item in dimensions
        if int(item.get("accepted_evidence", 0)) >= 1
        and int(item.get("independent_sources", 0)) >= 2
    )
    coverage = covered / len(dimensions) if dimensions else 0.0
    conflicts = int(case.get("unresolved_conflicts", 0) or 0)
    blocked_flags = (
        "stale_claims",
        "scope_mismatches",
        "search_failures",
        "repeated_queries",
        "low_information_gain_rounds",
        "unsupported_citations",
        "analysis_errors",
    )
    blocked = conflicts > 0 or any(int(case.get(flag, 0) or 0) > 0 for flag in blocked_flags)
    budget_exhausted = bool(case.get("budget_exhausted", False))
    no_evidence = bool(case.get("no_verifiable_evidence", False))
    if no_evidence:
        verdict = "fail"
    elif coverage >= 0.85 and not blocked:
        verdict = "write_limited" if budget_exhausted else "write"
    else:
        verdict = "continue"
    expected = str(case.get("expected_verdict", ""))
    return {
        "case_id": str(case.get("id", "")),
        "coverage": coverage,
        "verdict": verdict,
        "expected_verdict": expected,
        "passed": verdict == expected,
    }


def grade_dataset(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    results = [grade_case(case) for case in cases]
    return {
        "total": len(results),
        "passed": sum(bool(item["passed"]) for item in results),
        "accuracy": (
            sum(bool(item["passed"]) for item in results) / len(results) if results else 0.0
        ),
        "results": results,
    }
