"""Phase 15.0 Golden Baseline + Evidence-to-Coverage funnel tests (spec §26).

27 cases, one per numbered requirement in the Phase 15.0 spec §26.  Everything
here is pure: the domain gate is exercised directly, and the analyzer's
arithmetic is exercised on synthetic run data so no database is touched.  The
last case is the Generality Guard proof (spec §25): renaming every
question/dimension/domain/owner literal must leave every name-independent
aggregate byte-for-byte identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.domain.benchmark_comparability import (
    METRIC_DEFINITION_VERSION,
    BaselineRequirement,
    BenchmarkComparisonKey,
    ComparisonResult,
    compare_keys,
    evaluate_baseline_requirements,
)
from app.domain.golden_baseline import (
    GOLDEN_BASELINE_TIER,
    BenchmarkBaselineManifest,
    GoldenBaselineStatus,
    comparison_key_from_run,
    evaluate_golden_baseline,
)

from scripts.analyze_phase15_conversion_funnel import (
    analyze_claim_conversion,
    analyze_coverage_loss,
    analyze_dimension_matrix,
    analyze_evidence_waste,
    analyze_gap_requirements,
    analyze_rejections,
    analyze_source_roles,
    build_funnel,
    build_run_metrics,
    classify_primary_bottleneck,
    estimate_coverage_unlock,
    gap_blocking_causes,
    gap_effective_state,
    ratio,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _PROJECT_ROOT / "evals" / "benchmarks" / "v1_golden_baseline.v1.json"


# ---------------------------------------------------------------------------
# manifest + comparison-key fixtures
# ---------------------------------------------------------------------------


def _manifest_dict(**over: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "benchmark_id": "v1-dev-model-comparison",
        "benchmark_version": "1.0.0",
        "normalized_goal": (
            "Compare CNN, Transformer and Vision-Language Models for "
            "industrial surface defect detection."
        ),
        "tier": "standard",
        "plan_shape_policy": "frozen_reference_plan:questions=5,priority_one=4,priority_two=1",
        "plan_template_run_id": "01a08bcc-1034-76ba-92d4-12fa1a8badcb",
        "budget_profile": "standard",
        "metric_definition_version": "coverage-v1",
        "feature_flags": {"evidence_aware_context_enabled": False},
    }
    data.update(over)
    return data


def _manifest(**over: Any) -> BenchmarkBaselineManifest:
    return BenchmarkBaselineManifest.from_mapping(_manifest_dict(**over))


def _golden_key(**over: Any) -> BenchmarkComparisonKey:
    base: dict[str, Any] = {
        "benchmark_id": "v1-dev-model-comparison",
        "benchmark_version": "1.0.0",
        "normalized_goal": "compare defect detection models",
        "tier": "standard",
        "plan_template_run_id": "tpl-1",
        "plan_shape_policy": "frozen_reference_plan",
        "metric_definition_version": "coverage-v1",
        "evidence_aware_context_enabled": "false",
    }
    base.update(over)
    return BenchmarkComparisonKey(**base)


# ---------------------------------------------------------------------------
# synthetic run fixtures for the analyzer (spec §26 funnel / waste / etc.)
# ---------------------------------------------------------------------------

_SOURCES: list[dict[str, Any]] = [
    {"id": "S1", "source_type": "webpage", "source_owner_key": "ownerA", "domain": "a.com",
     "reliability": 0.8},
    {"id": "S2", "source_type": "webpage", "source_owner_key": "ownerB", "domain": "b.com",
     "reliability": 0.7},
    {"id": "S3", "source_type": "news", "source_owner_key": "ownerC", "domain": "c.com",
     "reliability": 0.5},
    {"id": "S4", "source_type": "academic", "source_owner_key": "ownerD", "domain": "d.edu",
     "reliability": 0.9},
]

_CLAIMS: list[dict[str, Any]] = [
    {"id": "C1", "question_id": "q1", "dimension_key": "dimX", "claim_type": "factual",
     "importance": 1.0, "status": "partial"},
    {"id": "C2", "question_id": "q2", "dimension_key": "dimY", "claim_type": "factual",
     "importance": 1.0, "status": "rejected"},
    {"id": "C3", "question_id": "q3", "dimension_key": "dimZ", "claim_type": "factual",
     "importance": 0.5, "status": "partial"},
    {"id": "C4", "question_id": "q4", "dimension_key": "dimW", "claim_type": "factual",
     "importance": 1.0, "status": "partial"},
]


def _evidence(
    eid: str,
    claim_id: str | None,
    source_id: str | None,
    accepted: bool,
    reason: str | None = None,
    question_id: str | None = None,
) -> dict[str, Any]:
    return {
        "id": eid,
        "claim_id": claim_id,
        "question_id": question_id,
        "accepted": accepted,
        "source_id": source_id,
        "rejection_reason": reason,
        "relation": "supports" if accepted else None,
    }


_EVIDENCE: list[dict[str, Any]] = [
    _evidence("E1", "C1", "S1", True, question_id="q1"),
    _evidence("E2", "C1", "S2", True, question_id="q1"),
    _evidence("E3", None, "S1", True),  # accepted, no claim bound
    _evidence("E4", "C2", "S3", False, "source_role_mismatch", question_id="q2"),
    _evidence("E5", "C2", "S3", False, "evidence_relevance_below_threshold", question_id="q2"),
    _evidence("E6", "C3", "S1", True, question_id="q3"),  # single owner -> unverified
    _evidence("E7", "C4", "S1", True, question_id="q4"),
    _evidence("E8", "C4", "S2", True, question_id="q4"),  # 2 owners -> verified, dimW partial
]


def _gap(
    gap_id: str,
    question_id: str,
    dim: str,
    req_type: str,
    req_ev: int,
    req_src: int,
    cur_ev: int,
    cur_src: int,
    cur_cov: float,
    req_cov: float | None,
) -> dict[str, Any]:
    return {
        "gap_id": gap_id,
        "question_id": question_id,
        "dimension_key": dim,
        "requirement_type": req_type,
        "required_evidence_count": req_ev,
        "required_independent_sources": req_src,
        "current_evidence_count": cur_ev,
        "current_independent_sources": cur_src,
        "current_coverage": cur_cov,
        "required_coverage": req_cov,
        "verification_status": "required",  # the stuck/dead enum, deliberately present
        "closure_status": "partial",  # the stuck/dead enum, deliberately present
    }


_GAP_REQS: list[dict[str, Any]] = [
    _gap("G1", "q1", "dimX", "independent_source", 2, 2, 2, 2, 1.0, 1.0),  # closed
    _gap("G2", "q2", "dimY", "claim_verification", 2, 1, 0, 0, 0.0, 0.5),  # open, multi-cause
    _gap("G3", "q3", "dimZ", "evidence_quality", 2, 2, 1, 1, 0.3, 0.5),  # partial, multi-cause
    _gap("G4", "q4", "dimW", "dimension_coverage", 2, 2, 2, 2, 0.2, 0.8),  # partial, 1 cause
]

_EVENTS: dict[str, int] = {
    "search.query.started": 3,
    "research.search.completed": 3,
}

_STORAGE: dict[str, int] = {
    "reader_completed": 2,
    "chunked_snapshots": 2,
    "unique_urls": 10,
    "readable_sources": 4,
}


def _card(**over: Any) -> dict[str, Any]:
    card: dict[str, Any] = {
        "run_id": "run-1",
        "found": True,
        "status": "completed",
        "normalized_goal": "compare defect detection models",
        "runtime_seconds": 600.0,
        "usage": {
            "searches": 3,
            "candidate_urls": 12,
            "extraction_calls": 2,
            "model_tokens": 1000,
            "evidence_total_tokens": 500,
            "logical_queries": 3,
            "search_provider_requests": 5,
        },
        "quality": {"coverage": 0.5},
        "budget": {"tier": "standard", "plan_template_run_id": "tpl-1"},
        "benchmark": {
            "benchmark_id": "v1-dev-model-comparison",
            "benchmark_version": "1.0.0",
            "metric_definition_version": "coverage-v1",
        },
        "stored_coverage": 0.5,
        "stored_priority_one_coverage": 0.6,
        "stored_critical_gaps": 1,
    }
    card.update(over)
    return card


def _metrics(**over: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "card": _card(),
        "claims": _CLAIMS,
        "evidence": _EVIDENCE,
        "sources": _SOURCES,
        "gap_reqs": _GAP_REQS,
        "events": _EVENTS,
        "storage": _STORAGE,
    }
    kwargs.update(over)
    return build_run_metrics(**kwargs)


def _edge_run(
    readable: float,
    cand: float,
    acc: float,
    sup: float,
    ver: float,
    gap_sat: float,
    gap_total: float,
) -> dict[str, Any]:
    """Minimal per-run metrics shaped for _edge_conversion_rates."""

    return {
        "research_work": {"readable_sources": readable},
        "funnel": {
            "candidate_evidence": cand,
            "accepted_evidence": acc,
            "supported_claims": sup,
            "verified_claims": ver,
            "gap_satisfied": gap_sat,
        },
        "quality": {"gap_requirement_total": gap_total},
    }


# ---------------------------------------------------------------------------
# §26 items 1-6: registered benchmark + eligibility gate (domain)
# ---------------------------------------------------------------------------


def test_01_registered_benchmark_metadata() -> None:
    """Committed manifest loads and projects a fully-known comparison key."""

    data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest = BenchmarkBaselineManifest.from_mapping(data)
    assert manifest.benchmark_id == "v1-dev-model-comparison"
    assert manifest.metric_definition_version == METRIC_DEFINITION_VERSION
    assert manifest.tier == GOLDEN_BASELINE_TIER
    assert manifest.evidence_aware_context_enabled is False
    key = manifest.to_comparison_key()
    for field in ("benchmark_id", "normalized_goal", "plan_shape_policy", "tier"):
        assert key.is_known(field)


def test_02_golden_baseline_eligibility_pass() -> None:
    key = _manifest().to_comparison_key()
    verdict = evaluate_baseline_requirements(key, feature_flags_known=True)
    assert verdict.eligible is True
    assert verdict.unmet == ()


def test_03_missing_benchmark_id_is_invalid() -> None:
    # A run created without the budget_snapshot.benchmark block (every historical
    # run) yields unknown identity and fails the eligibility gate outright.
    key = comparison_key_from_run(
        {"normalized_goal": "x", "budget_snapshot": {"tier": "standard"}}
    )
    verdict = evaluate_baseline_requirements(key, feature_flags_known=False)
    assert verdict.eligible is False
    assert BaselineRequirement.REGISTERED_BENCHMARK.value in verdict.unmet

    # A manifest that itself has no benchmark_id cannot be a golden baseline.
    unregistered = BenchmarkBaselineManifest(
        benchmark_id="",
        benchmark_version="1.0.0",
        normalized_goal="g",
        tier="standard",
        plan_shape_policy="frozen_reference_plan",
        plan_template_run_id="tpl-1",
        budget_profile="standard",
        metric_definition_version="coverage-v1",
        evidence_aware_context_enabled=False,
    )
    evaluation = evaluate_golden_baseline(
        unregistered,
        [unregistered.to_comparison_key()],
        feature_flags_confirmed=True,
        valid_run_count=1,
    )
    assert evaluation.status is GoldenBaselineStatus.INVALID


def test_04_mixed_benchmark_versions() -> None:
    # benchmark_version is a soft field: mismatch is only PARTIALLY_COMPARABLE,
    # so evaluate_golden_baseline refuses to count the odd run as a golden member.
    a = _golden_key()
    b = _golden_key(benchmark_version="2.0.0")
    assert compare_keys(a, b).result is ComparisonResult.PARTIALLY_COMPARABLE
    evaluation = evaluate_golden_baseline(
        _manifest(),
        [_golden_key(), _golden_key(), b],
        feature_flags_confirmed=True,
    )
    assert evaluation.status is not GoldenBaselineStatus.VALID


def test_05_mixed_metric_versions() -> None:
    # metric_definition_version is a hard field -> NOT_COMPARABLE.
    a = _golden_key()
    b = _golden_key(metric_definition_version="coverage-v0")
    assert compare_keys(a, b).result is ComparisonResult.NOT_COMPARABLE


def test_06_mixed_feature_flags() -> None:
    # evidence_aware_context_enabled is a hard field -> NOT_COMPARABLE.
    a = _golden_key()
    b = _golden_key(evidence_aware_context_enabled="true")
    assert compare_keys(a, b).result is ComparisonResult.NOT_COMPARABLE


# ---------------------------------------------------------------------------
# §26 items 7, 20: funnel stage counting + conversion rate
# ---------------------------------------------------------------------------


def test_07_funnel_stage_counting() -> None:
    funnel = {row["stage"]: row for row in build_funnel([_metrics()])}
    assert funnel["candidate_evidence"]["aggregate_count"] == 8
    assert funnel["accepted_evidence"]["aggregate_count"] == 6
    assert funnel["supported_claims"]["aggregate_count"] == 3
    assert funnel["verified_claims"]["aggregate_count"] == 2
    assert funnel["gap_satisfied"]["aggregate_count"] == 1


def test_20_conversion_rate_between_stages() -> None:
    funnel = {row["stage"]: row for row in build_funnel([_metrics()])}
    # accepted -> supported = 3/6 = 0.5 ; supported -> verified = 2/3.
    assert funnel["supported_claims"]["conversion_rate_from_previous"] == 0.5
    assert funnel["verified_claims"]["conversion_rate_from_previous"] == round(2 / 3, 6)


# ---------------------------------------------------------------------------
# §26 items 8, 9: rejection attribution + unknown reason
# ---------------------------------------------------------------------------


def test_08_rejection_reason_attribution() -> None:
    rejections = analyze_rejections(_metrics())
    reasons = {item["reason"]: item for item in rejections["reasons"]}
    assert rejections["rejected"] == 2
    assert reasons["source_role_mismatch"]["situation"] == "A_upstream_input_quality"
    relevant = reasons["evidence_relevance_below_threshold"]["situation"]
    assert relevant == "threshold_candidate_for_review"
    assert reasons["source_role_mismatch"]["share_of_rejected"] == 0.5


def test_09_unknown_rejection_reason() -> None:
    evidence = [*_EVIDENCE, _evidence("E9", "C2", "S3", False, "brand_new_reason")]
    rejections = analyze_rejections(_metrics(evidence=evidence))
    reasons = {item["reason"]: item for item in rejections["reasons"]}
    assert reasons["brand_new_reason"]["situation"] == "other_or_unknown"

    # A rejected row with a null reason is surfaced as "(unknown)", not dropped.
    evidence2 = [*_EVIDENCE, _evidence("E10", "C2", "S3", False, None)]
    reasons2 = {i["reason"]: i for i in analyze_rejections(_metrics(evidence=evidence2))["reasons"]}
    assert "(unknown)" in reasons2


# ---------------------------------------------------------------------------
# §26 items 10-12: claim association / support / verification
# ---------------------------------------------------------------------------


def test_10_accepted_evidence_without_claim() -> None:
    conversion = analyze_claim_conversion(_metrics())
    assert conversion["accepted_with_claim"] == 5
    assert conversion["accepted_without_claim"] == 1  # E3


def test_11_supported_claim() -> None:
    metrics = _metrics()
    # C1, C3, C4 carry accepted evidence; C2's evidence is all rejected.
    assert metrics["funnel"]["supported_claims"] == 3


def test_12_verified_claim_uses_max_independent_bar() -> None:
    metrics = _metrics()
    # dimX requires 2 owners (C1 has 2 -> verified); dimZ requires 2 (C3 has 1 -> not);
    # dimW requires 2 (C4 has 2 -> verified).
    assert metrics["funnel"]["verified_claims"] == 2
    assert metrics["quality"]["verified_claims"] == 2


# ---------------------------------------------------------------------------
# §26 items 13, 14: gap aggregation + open/partial/closed
# ---------------------------------------------------------------------------


def test_13_gap_aggregation_by_requirement_type() -> None:
    gaps = analyze_gap_requirements(_metrics())
    by_type = gaps["by_requirement_type"]
    assert set(by_type) == {
        "independent_source",
        "claim_verification",
        "evidence_quality",
        "dimension_coverage",
    }
    assert by_type["independent_source"]["closed"] == 1
    assert by_type["claim_verification"]["open"] == 1


def test_14_open_partial_closed_states() -> None:
    metrics = _metrics()
    states = [gap_effective_state(r) for r in metrics["_gap_reqs"]]
    assert states == ["closed", "open", "partial", "partial"]
    assert metrics["quality"]["gap_closed"] == 1
    assert metrics["quality"]["gap_open"] == 1
    assert metrics["quality"]["gap_partial"] == 2


# ---------------------------------------------------------------------------
# §26 item 15: multi-cause coverage loss (bounded)
# ---------------------------------------------------------------------------


def test_15_multi_cause_coverage_loss() -> None:
    dimY = _GAP_REQS[1]
    causes = set(gap_blocking_causes(dimY))
    assert causes == {
        "missing_candidate_evidence",
        "missing_independent_source",
        "coverage_below_required",
    }
    loss = analyze_coverage_loss(_metrics())
    # Coverage impact is normalised per dimension so no single cause can exceed
    # the [0, 1] coverage range (the pre-fix double count reached 6.58).
    for cause in loss["causes"]:
        assert 0.0 <= cause["estimated_coverage_impact"] <= 1.0
    assert loss["aggregate_coverage_shortfall"] <= 1.0


# ---------------------------------------------------------------------------
# §26 items 16-19: evidence waste types 1-5
# ---------------------------------------------------------------------------


def test_16_evidence_waste_type1_readable_no_evidence() -> None:
    waste = analyze_evidence_waste(_metrics())
    # S4 (academic) is a source with no evidence rows.
    assert waste["type1_readable_no_evidence"]["count"] == 1


def test_17_evidence_waste_type2_extracted_all_rejected() -> None:
    waste = analyze_evidence_waste(_metrics())
    # S3 is the only source whose extracted evidence is entirely rejected.
    assert waste["type2_extracted_all_rejected"]["count"] == 1


def test_18_evidence_waste_type3_accepted_no_claim() -> None:
    waste = analyze_evidence_waste(_metrics())
    assert waste["type3_accepted_no_claim"]["count"] == 1


def test_19_evidence_waste_type4_and_type5() -> None:
    waste = analyze_evidence_waste(_metrics())
    # C3 supported but unverified -> type4; C4 verified but dimW not closed -> type5.
    assert waste["type4_supported_no_verification"]["count"] == 1
    assert waste["type5_verified_no_gap_transition"]["count"] == 1


# ---------------------------------------------------------------------------
# §26 items 21-24: zero denominator / missing events / incomplete evidence /
# source role
# ---------------------------------------------------------------------------


def test_21_zero_denominator_is_none() -> None:
    assert ratio(5, 0) is None
    assert ratio(0, 0) is None
    assert ratio(6, 3) == 2.0


def test_22_missing_historical_event() -> None:
    metrics = _metrics(events={})  # an event type never recorded
    assert metrics["research_work"]["search_started"] == 0
    assert metrics["research_work"]["evidence_extractions"] == 2  # from usage, not events


def test_23_incomplete_evidence_rows() -> None:
    # An accepted row missing both claim and source must not crash and must not
    # be counted as bound to a claim.
    evidence = [*_EVIDENCE, _evidence("E99", None, None, True)]
    metrics = _metrics(evidence=evidence)
    assert metrics["quality"]["accepted_evidence"] == 7
    assert metrics["accepted_without_claim"] == 2


def test_24_source_role_analysis() -> None:
    roles = analyze_source_roles(_metrics())
    assert roles["webpage"]["candidate_evidence"] == 6
    assert roles["webpage"]["accepted"] == 6
    assert roles["news"]["candidate_evidence"] == 2
    assert roles["news"]["accepted"] == 0
    assert roles["academic"]["sources"] == 1


# ---------------------------------------------------------------------------
# §26 item 25: Generality Guard — question/dimension/domain identity independence
# ---------------------------------------------------------------------------


def _rename(obj: dict[str, Any], maps: dict[str, dict[str, str]]) -> dict[str, Any]:
    out = dict(obj)
    for field, mapping in maps.items():
        if out.get(field) in mapping:
            out[field] = mapping[out[field]]
    return out


def test_25_question_dimension_domain_identity_independence() -> None:
    dim_map = {"dimX": "alpha", "dimY": "beta", "dimZ": "gamma", "dimW": "delta"}
    q_map = {"q1": "zz1", "q2": "zz2", "q3": "zz3", "q4": "zz4"}
    owner_map = {"ownerA": "oP", "ownerB": "oQ", "ownerC": "oR", "ownerD": "oS"}
    domain_map = {"a.com": "x.io", "b.com": "y.io", "c.com": "z.io", "d.edu": "w.edu"}

    base = _metrics()
    renamed = _metrics(
        claims=[_rename(c, {"dimension_key": dim_map, "question_id": q_map}) for c in _CLAIMS],
        evidence=[_rename(e, {"question_id": q_map}) for e in _EVIDENCE],
        sources=[
            _rename(s, {"source_owner_key": owner_map, "domain": domain_map}) for s in _SOURCES
        ],
        gap_reqs=[
            _rename(g, {"dimension_key": dim_map, "question_id": q_map}) for g in _GAP_REQS
        ],
    )

    def fingerprint(metrics: dict[str, Any]) -> dict[str, Any]:
        loss = analyze_coverage_loss(metrics)
        gaps = analyze_gap_requirements(metrics)
        matrix = analyze_dimension_matrix(metrics)
        return {
            "funnel": metrics["funnel"],
            "gap_states": sorted(
                gap_effective_state(r) for r in metrics["_gap_reqs"]
            ),
            "rejection_reasons": sorted(
                (r["reason"], r["count"]) for r in analyze_rejections(metrics)["reasons"]
            ),
            "coverage_causes": sorted((c["cause"], c["count"]) for c in loss["causes"]),
            "transition_types": gaps["transition_types"],
            "per_type_totals": {k: v["total"] for k, v in gaps["by_requirement_type"].items()},
            "verified": metrics["funnel"]["verified_claims"],
            "matrix_sorted": sorted(
                (row["candidate_evidence"], row["accepted"], row["supported"], row["verified"])
                for row in matrix
            ),
        }

    assert fingerprint(base) == fingerprint(renamed)


# ---------------------------------------------------------------------------
# §26 items 26-27: primary bottleneck + theoretical coverage unlock
# ---------------------------------------------------------------------------


def test_26_primary_bottleneck_classification() -> None:
    # Candidate -> Accepted (0.2) is the lowest edge on this shape.
    case_b = classify_primary_bottleneck([_edge_run(10, 10, 2, 2, 2, 2, 2)])
    assert case_b["primary"] == "Case_B"
    assert case_b["secondary"] in ("Case_A", "Case_C", "Case_D")

    # Case A/B/C/D compete; the canonical funnel loses hardest at Case_D (0.25).
    canonical = classify_primary_bottleneck([_metrics()])
    assert canonical["primary"] == "Case_D"
    assert set(canonical["edge_rates"]) == {"Case_A", "Case_B", "Case_C", "Case_D"}

    # Two edges within 0.05 of each other -> honest Mixed (Case_E).
    tie = classify_primary_bottleneck([_edge_run(5, 10, 6, 5, 3, 5, 5)])
    assert tie["primary"] == "Case_E"
    assert tie["near_tie"] is True


def test_27_theoretical_coverage_unlock_estimate() -> None:
    bottleneck = {"primary": "Case_B"}
    coverage_loss = {
        "causes": [
            {"cause": "missing_candidate_evidence", "estimated_coverage_impact": 0.10},
            {"cause": "coverage_below_required", "estimated_coverage_impact": 0.15},
        ]
    }
    unlock = estimate_coverage_unlock(0.5, bottleneck, coverage_loss)
    assert unlock["current_coverage"] == 0.5
    assert unlock["estimated_blocked_coverage"] == 0.25
    assert unlock["theoretical_upper_bound"] == 0.75
    assert unlock["estimated"] is True
    assert unlock["counterfactual"] is True
    assert unlock["experimentally_validated"] is False

    # Insufficient data is reported honestly rather than fabricated.
    assert estimate_coverage_unlock(None, bottleneck, coverage_loss)["status"] == (
        "insufficient-data"
    )
