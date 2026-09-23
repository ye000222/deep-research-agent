#!/usr/bin/env python3
"""Phase 15.0 Evidence-to-Coverage conversion funnel analyzer (read-only).

This analyzer takes the three registered Golden Baseline run ids and produces a
purely diagnostic view of where research work fails to convert into coverage.
It never writes to the database (Task F): every query is a ``SELECT``.

Design discipline enforced throughout the module (Generality Guard, spec §25):

* No question-id, dimension-name, domain or benchmark special-casing.  All
  classification is driven by *generic* state that already exists in the
  production schema: evidence acceptance flags and their real
  ``rejection_reason`` values, claim ``status`` values, ``requirement_type``
  taxonomy, ``source_type`` / domain / reliability metadata, event counters and
  the persisted snapshots.
* Every division is routed through :func:`ratio`, which returns ``None`` on a
  zero denominator instead of raising (Task O).
* The persisted ``gap_requirements.closure_status`` / ``verification_status``
  columns are observed to be *stuck* at ``partial`` / ``required`` even when the
  numeric ``current_*`` fields fully satisfy ``required_*``.  The analyzer
  therefore derives effective gap state from the numeric satisfaction fields and
  reports the raw persisted distribution separately as a diagnosis, rather than
  trusting the dead enum.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

try:  # psycopg is only needed for the live DB read; tests import pure helpers.
    import psycopg
except ImportError:  # pragma: no cover - exercised only when the driver is absent
    psycopg = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# taxonomy (generic, derived from the production schema — not invented)
# ---------------------------------------------------------------------------

#: The four requirement-type buckets the spec asks about.  Any *other*
#: requirement_type observed in data is grouped under ``other`` rather than
#: being special-cased away, so new types degrade safely.
SPEC_REQUIREMENT_TYPES: Final[tuple[str, ...]] = (
    "dimension_coverage",
    "independent_source",
    "claim_verification",
    "evidence_quality",
)

#: Real rejection reasons that indicate the evidence was genuinely unfit for the
#: requirement (an *upstream* signal: search / reader / extraction produced the
#: wrong material).  Membership is by reason family, never by question/domain.
UPSTREAM_REJECTION_REASONS: Final[frozenset[str]] = frozenset(
    {
        "source_role_mismatch",
        "dimension_mismatch",
        "claim_not_supported",
        "claim_quote_entailment_failed",
        "quote_not_found_in_source",
        "quote_not_located_for_graph",
        "scope_mismatch",
        "numeric_scope_inconsistent",
        "prompt_injection_detected",
    }
)

#: Reasons driven by a numeric acceptance threshold; borderline cases in this
#: family are the only legitimate candidates for a *possible* false rejection,
#: and even then only after per-item score review (never asserted here).
THRESHOLD_REJECTION_REASONS: Final[frozenset[str]] = frozenset(
    {
        "evidence_relevance_below_threshold",
        "evidence_score_below_threshold",
        "source_reliability_below_threshold",
        "evidence_confidence_below_threshold",
    }
)

#: Ordered funnel stages (count, label).  ``conversion_rate_from_previous`` is
#: computed for each stage relative to the preceding non-``None`` stage.
FUNNEL_STAGES: Final[tuple[tuple[str, str], ...]] = (
    ("search_completed", "Search executions completed"),
    ("candidate_urls", "Candidate URLs discovered"),
    ("unique_urls", "Unique result URLs"),
    ("readable_sources", "Readable sources (unique readable)"),
    ("reader_completed", "Reader completions (snapshots)"),
    ("evidence_extractions", "Evidence extraction calls"),
    ("candidate_evidence", "Candidate evidence rows"),
    ("accepted_evidence", "Accepted evidence rows"),
    ("supported_claims", "Claims with accepted-evidence support"),
    ("verified_claims", "Claims meeting independent-source verification"),
    ("gap_satisfied", "Gap requirements numerically satisfied"),
    ("coverage", "Coverage (priority-weighted mean)"),
)


# ---------------------------------------------------------------------------
# generic numeric helpers (zero-denominator safe)
# ---------------------------------------------------------------------------


def ratio(numerator: float, denominator: float) -> float | None:
    """Safe division: ``None`` when the denominator is zero (Task O)."""

    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def rate_or_zero(numerator: float, denominator: float) -> float:
    """Safe rate that reports ``0.0`` when there is nothing to divide."""

    return 0.0 if denominator == 0 else round(numerator / denominator, 6)


def _as_int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _database_uri(value: str) -> str:
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    """mean / SD / min / max over the observed values (Task E)."""

    present = [v for v in values if v is not None]
    if not present:
        return {"mean": None, "sd": None, "min": None, "max": None, "n": 0}
    return {
        "mean": round(statistics.fmean(present), 6),
        "sd": round(statistics.pstdev(present), 6) if len(present) > 1 else 0.0,
        "min": round(min(present), 6),
        "max": round(max(present), 6),
        "n": len(present),
    }


# ---------------------------------------------------------------------------
# DB read layer (SELECT only)
# ---------------------------------------------------------------------------


def _fetch_run_card(connection: Any, run_id: str) -> dict[str, Any]:
    row = connection.execute(
        """
        SELECT id, status, termination_reason, normalized_goal, plan_version,
               created_at, finished_at, quality_snapshot, usage_snapshot, budget_snapshot
        FROM research_runs WHERE id = %s
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return {"run_id": run_id, "found": False}
    (
        rid,
        status,
        termination_reason,
        goal,
        plan_version,
        created_at,
        finished_at,
        quality,
        usage,
        budget,
    ) = row
    quality = quality if isinstance(quality, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    budget = budget if isinstance(budget, dict) else {}
    runtime_seconds: float | None = None
    if created_at is not None and finished_at is not None:
        runtime_seconds = round((finished_at - created_at).total_seconds(), 1)
    benchmark = budget.get("benchmark") if isinstance(budget.get("benchmark"), dict) else {}
    return {
        "run_id": str(rid),
        "found": True,
        "status": status,
        "termination_reason": termination_reason,
        "normalized_goal": goal,
        "plan_version": int(plan_version) if plan_version is not None else None,
        "runtime_seconds": runtime_seconds,
        "quality": quality,
        "usage": usage,
        "budget": budget,
        "benchmark": benchmark,
        "tier": budget.get("tier"),
        "stored_coverage": _as_float(quality.get("coverage")),
        "stored_candidate_evidence": _as_int(quality.get("candidate_evidence")),
        "stored_accepted_evidence": _as_int(quality.get("accepted_evidence")),
        "stored_claim_count": _as_int(quality.get("claim_count")),
        "stored_priority_one_coverage": _as_float(quality.get("priority_one_coverage")),
        "stored_critical_gaps": _as_int(quality.get("critical_gaps")),
    }


def _fetch_claims(connection: Any, run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT id, question_id, dimension_key, claim_type, importance, status
        FROM research_claims WHERE run_id = %s
        """,
        (run_id,),
    ).fetchall()
    return [
        {
            "id": str(claim_id),
            "question_id": str(question_id) if question_id is not None else None,
            "dimension_key": str(dimension_key) if dimension_key is not None else None,
            "claim_type": claim_type,
            "importance": float(importance or 0.0),
            "status": str(status),
        }
        for claim_id, question_id, dimension_key, claim_type, importance, status in rows
    ]


def _fetch_evidence(connection: Any, run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT id, claim_id, question_id, accepted, source_id, rejection_reason, relation
        FROM research_evidence WHERE run_id = %s
        """,
        (run_id,),
    ).fetchall()
    return [
        {
            "id": str(evidence_id),
            "claim_id": str(claim_id) if claim_id is not None else None,
            "question_id": str(question_id) if question_id is not None else None,
            "accepted": bool(accepted),
            "source_id": str(source_id) if source_id is not None else None,
            "rejection_reason": rejection_reason,
            "relation": relation,
        }
        for (
            evidence_id,
            claim_id,
            question_id,
            accepted,
            source_id,
            rejection_reason,
            relation,
        ) in rows
    ]


def _fetch_sources(connection: Any, run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT id, source_type, source_owner_key, domain, reliability
        FROM research_sources WHERE run_id = %s
        """,
        (run_id,),
    ).fetchall()
    return [
        {
            "id": str(source_id),
            "source_type": source_type,
            "source_owner_key": str(owner or source_id),
            "domain": domain,
            "reliability": float(reliability or 0.0),
        }
        for source_id, source_type, owner, domain, reliability in rows
    ]


def _fetch_gap_requirements(connection: Any, run_id: str) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT gap_id, question_id, dimension_key, requirement_type,
               required_evidence_count, required_independent_sources,
               current_evidence_count, current_independent_sources,
               current_coverage, required_coverage,
               verification_status, closure_status
        FROM gap_requirements WHERE run_id = %s
        """,
        (run_id,),
    ).fetchall()
    return [
        {
            "gap_id": str(gap_id),
            "question_id": str(question_id) if question_id is not None else None,
            "dimension_key": str(dimension_key) if dimension_key is not None else None,
            "requirement_type": str(requirement_type),
            "required_evidence_count": _as_int(required_evidence),
            "required_independent_sources": _as_int(required_sources),
            "current_evidence_count": _as_int(current_evidence),
            "current_independent_sources": _as_int(current_sources),
            "current_coverage": _as_float(current_coverage) or 0.0,
            "required_coverage": _as_float(required_coverage),
            "verification_status": str(verification_status),
            "closure_status": str(closure_status),
        }
        for (
            gap_id,
            question_id,
            dimension_key,
            requirement_type,
            required_evidence,
            required_sources,
            current_evidence,
            current_sources,
            current_coverage,
            required_coverage,
            verification_status,
            closure_status,
        ) in rows
    ]


def _fetch_event_counts(connection: Any, run_id: str) -> dict[str, int]:
    rows = connection.execute(
        "SELECT event_type, count(*) FROM agent_events WHERE run_id = %s GROUP BY event_type",
        (run_id,),
    ).fetchall()
    return {str(event_type): int(count) for event_type, count in rows}


def _fetch_storage_stats(connection: Any, run_id: str) -> dict[str, int]:
    snapshots = connection.execute(
        "SELECT count(*) FROM research_source_snapshots WHERE run_id = %s",
        (run_id,),
    ).fetchone()
    chunked = connection.execute(
        """
        SELECT count(*) FROM research_source_snapshots s
        WHERE s.run_id = %s
          AND EXISTS (SELECT 1 FROM research_source_chunks c WHERE c.snapshot_id = s.id)
        """,
        (run_id,),
    ).fetchone()
    unique_urls = connection.execute(
        """
        SELECT count(DISTINCT lower(r.url))
        FROM research_search_results r
        JOIN research_search_queries q ON r.search_query_id = q.id
        WHERE q.run_id = %s
        """,
        (run_id,),
    ).fetchone()
    readable = connection.execute(
        """
        SELECT count(DISTINCT lower(s.canonical_url))
        FROM research_sources s
        WHERE s.run_id = %s
          AND EXISTS (SELECT 1 FROM research_source_snapshots ss WHERE ss.source_id = s.id)
        """,
        (run_id,),
    ).fetchone()
    return {
        "reader_completed": _as_int(snapshots[0] if snapshots else 0),
        "chunked_snapshots": _as_int(chunked[0] if chunked else 0),
        "unique_urls": _as_int(unique_urls[0] if unique_urls else 0),
        "readable_sources": _as_int(readable[0] if readable else 0),
    }


# ---------------------------------------------------------------------------
# gap / claim effective-state derivation (numeric, not the stuck enum)
# ---------------------------------------------------------------------------


def gap_effective_state(req: dict[str, Any]) -> str:
    """Derive real closure from the numeric satisfaction fields.

    ``closed``  - every quantitative criterion is met;
    ``open``    - nothing at all has been gathered;
    ``partial`` - some but not all criteria met.

    The persisted ``closure_status`` column is deliberately ignored here because
    it is observed stuck at ``partial``; the raw value is still reported
    separately elsewhere as a diagnosis.
    """

    required_coverage = req.get("required_coverage")
    coverage_full = required_coverage is None or req["current_coverage"] >= required_coverage - 1e-9
    evidence_ok = req["current_evidence_count"] >= req["required_evidence_count"]
    sources_ok = req["current_independent_sources"] >= req["required_independent_sources"]
    if coverage_full and evidence_ok and sources_ok:
        return "closed"
    if req["current_coverage"] <= 1e-9 and req["current_evidence_count"] == 0:
        return "open"
    return "partial"


def gap_blocking_causes(req: dict[str, Any]) -> list[str]:
    """Every unmet quantitative criterion for one requirement (MULTI_CAUSE ok)."""

    causes: list[str] = []
    if req["current_evidence_count"] < req["required_evidence_count"]:
        causes.append("missing_candidate_evidence")
    if req["current_independent_sources"] < req["required_independent_sources"]:
        causes.append("missing_independent_source")
    required_coverage = req.get("required_coverage")
    if required_coverage is not None and req["current_coverage"] < required_coverage - 1e-9:
        causes.append("coverage_below_required")
    return causes


def build_run_metrics(
    card: dict[str, Any],
    claims: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    gap_reqs: list[dict[str, Any]],
    events: dict[str, int],
    storage: dict[str, int],
) -> dict[str, Any]:
    """Compose every per-run number the spec asks for (Tasks E + funnel input)."""

    usage = card.get("usage") or {}
    source_by_id = {s["id"]: s for s in sources}

    accepted = [e for e in evidence if e["accepted"]]
    rejected = [e for e in evidence if not e["accepted"]]

    # claim -> independent owners over its accepted evidence
    claim_owners: dict[str, set[str]] = defaultdict(set)
    claim_accepted_count: dict[str, int] = defaultdict(int)
    for e in accepted:
        cid = e.get("claim_id")
        if cid is None:
            continue
        src = source_by_id.get(e["source_id"] or "")
        owner = src["source_owner_key"] if src else (e.get("source_id") or "")
        claim_owners[cid].add(owner)
        claim_accepted_count[cid] += 1

    # dimension -> max required independent sources (verification bar)
    dim_required_sources: dict[str, int] = defaultdict(int)
    for req in gap_reqs:
        dim = req.get("dimension_key")
        if dim is not None:
            dim_required_sources[dim] = max(
                dim_required_sources[dim], req["required_independent_sources"]
            )

    supported_claims = [c for c in claims if claim_accepted_count.get(c["id"], 0) > 0]
    verified_claims = [
        c
        for c in supported_claims
        if len(claim_owners.get(c["id"], set()))
        >= (dim_required_sources.get(c.get("dimension_key") or "", 1) or 1)
    ]

    single_source = [c for c in supported_claims if len(claim_owners.get(c["id"], set())) == 1]
    multi_source = [c for c in supported_claims if len(claim_owners.get(c["id"], set())) >= 2]

    gap_states = [gap_effective_state(r) for r in gap_reqs]
    gap_satisfied = sum(1 for s in gap_states if s == "closed")

    accepted_with_claim = sum(1 for e in accepted if e.get("claim_id"))
    funnel_counts = {
        "search_completed": _as_int(usage.get("searches")),
        "candidate_urls": _as_int(usage.get("candidate_urls")),
        "unique_urls": _as_int(storage.get("unique_urls")),
        "readable_sources": _as_int(storage.get("readable_sources")),
        "reader_completed": _as_int(storage.get("reader_completed")),
        "evidence_extractions": _as_int(usage.get("extraction_calls")),
        "candidate_evidence": len(evidence),
        "accepted_evidence": len(accepted),
        "supported_claims": len(supported_claims),
        "verified_claims": len(verified_claims),
        "gap_satisfied": gap_satisfied,
        "coverage": card.get("stored_coverage"),
    }

    tokens = _as_int(usage.get("model_tokens")) + _as_int(usage.get("evidence_total_tokens"))

    return {
        "run_id": card["run_id"],
        "status": card.get("status"),
        "benchmark_id": (card.get("benchmark") or {}).get("benchmark_id"),
        "benchmark_version": (card.get("benchmark") or {}).get("benchmark_version"),
        "metric_definition_version": (card.get("benchmark") or {}).get("metric_definition_version"),
        "funnel": funnel_counts,
        "quality": {
            "coverage": card.get("stored_coverage"),
            "priority_one_coverage": card.get("stored_priority_one_coverage"),
            "critical_gap_count": card.get("stored_critical_gaps"),
            "candidate_evidence": len(evidence),
            "accepted_evidence": len(accepted),
            "acceptance_rate": ratio(len(accepted), len(evidence)),
            "supported_claims": len(supported_claims),
            "verified_claims": len(verified_claims),
            "unverified_claims": len(supported_claims) - len(verified_claims),
            "gap_requirement_total": len(gap_reqs),
            "gap_open": gap_states.count("open"),
            "gap_partial": gap_states.count("partial"),
            "gap_closed": gap_states.count("closed"),
            "gap_closure_rate": ratio(gap_satisfied, len(gap_reqs)),
        },
        "research_work": {
            "logical_queries": _as_int(usage.get("logical_queries")),
            "search_started": _as_int(events.get("search.query.started")),
            "search_completed": _as_int(usage.get("searches")),
            "provider_requests": _as_int(usage.get("search_provider_requests")),
            "candidate_urls": _as_int(usage.get("candidate_urls")),
            "unique_urls": _as_int(storage.get("unique_urls")),
            "readable_sources": _as_int(storage.get("readable_sources")),
            "reader_completed": _as_int(storage.get("reader_completed")),
            "evidence_extractions": _as_int(usage.get("extraction_calls")),
        },
        "cost": {
            "runtime_seconds": card.get("runtime_seconds"),
            "tokens": tokens,
        },
        "accepted_with_claim": accepted_with_claim,
        "accepted_without_claim": len(accepted) - accepted_with_claim,
        "single_source_claims": len(single_source),
        "multi_source_claims": len(multi_source),
        # raw rows retained for the detailed analyses below
        "_claims": claims,
        "_evidence": evidence,
        "_sources": sources,
        "_gap_reqs": gap_reqs,
        "_accepted": accepted,
        "_rejected": rejected,
        "_claim_owners": {k: sorted(v) for k, v in claim_owners.items()},
        "_dim_required_sources": dict(dim_required_sources),
    }


def _rejection_situation(reason: str) -> str:
    """Map a real reason to Situation A (upstream) / threshold / other.

    This is a reason-family classification, never a question/domain judgement,
    so it satisfies the Generality Guard.
    """

    if reason in UPSTREAM_REJECTION_REASONS:
        return "A_upstream_input_quality"
    if reason in THRESHOLD_REJECTION_REASONS:
        return "threshold_candidate_for_review"
    return "other_or_unknown"


def analyze_rejections(run: dict[str, Any]) -> dict[str, Any]:
    """Candidate -> Accepted loss attribution using the real rejection reasons."""

    evidence = run["_evidence"]
    rejected = run["_rejected"]
    source_by_id = {s["id"]: s for s in run["_sources"]}
    claim_by_id = {c["id"]: c for c in run["_claims"]}

    reason_counts: dict[str, int] = defaultdict(int)
    situation_counts: dict[str, int] = defaultdict(int)
    by_question: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_dimension: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_source_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    total_candidate = len(evidence)
    for e in rejected:
        reason = str(e.get("rejection_reason") or "(unknown)")
        reason_counts[reason] += 1
        situation_counts[_rejection_situation(reason)] += 1
        q = str(e.get("question_id") or "(none)")
        by_question[q][reason] += 1
        claim = claim_by_id.get(e.get("claim_id") or "")
        dim = str((claim or {}).get("dimension_key") or "(none)")
        by_dimension[dim][reason] += 1
        src = source_by_id.get(e.get("source_id") or "")
        stype = str((src or {}).get("source_type") or "(none)")
        by_source_type[stype][reason] += 1

    total_rejected = len(rejected)
    ranked = sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "candidate_evidence": total_candidate,
        "rejected": total_rejected,
        "accepted": total_candidate - total_rejected,
        "candidate_to_accepted_rate": ratio(total_candidate - total_rejected, total_candidate),
        "reasons": [
            {
                "reason": reason,
                "count": count,
                "share_of_rejected": ratio(count, total_rejected),
                "share_of_candidate": ratio(count, total_candidate),
                "situation": _rejection_situation(reason),
            }
            for reason, count in ranked
        ],
        "situation_totals": dict(situation_counts),
        "by_question": {q: dict(r) for q, r in by_question.items()},
        "by_dimension": {d: dict(r) for d, r in by_dimension.items()},
        "by_source_type": {s: dict(r) for s, r in by_source_type.items()},
    }


# ---------------------------------------------------------------------------
# claim conversion (Tasks H + I)
# ---------------------------------------------------------------------------


def analyze_claim_conversion(run: dict[str, Any]) -> dict[str, Any]:
    accepted = run["_accepted"]
    relation_counts: dict[str, int] = defaultdict(int)
    for e in accepted:
        relation_counts[str(e.get("relation") or "(unknown)")] += 1
    supported = run["funnel"]["supported_claims"]
    verified = run["funnel"]["verified_claims"]
    accepted_total = len(accepted)
    return {
        "accepted_evidence_total": accepted_total,
        "accepted_with_claim": run["accepted_with_claim"],
        "accepted_without_claim": run["accepted_without_claim"],
        "support_relation_mix": dict(relation_counts),
        "supported_claims": supported,
        "single_source_claims": run["single_source_claims"],
        "multi_source_claims": run["multi_source_claims"],
        "verified_claims": verified,
        "unverified_claims": supported - verified,
        "verification_unknown": sum(
            1 for e in accepted if str(e.get("relation")) == "(unknown)"
        ),
        "accepted_to_supported_rate": ratio(supported, accepted_total),
        "supported_to_verified_rate": ratio(verified, supported),
        "verified_of_accepted": ratio(verified, accepted_total),
    }


# ---------------------------------------------------------------------------
# gap-requirement type analysis + Claim -> GapRequirement conversion (Task J)
# ---------------------------------------------------------------------------


def analyze_gap_requirements(run: dict[str, Any]) -> dict[str, Any]:
    gap_reqs = run["_gap_reqs"]
    # dimension-level evidence presence for the type 1-4 classification
    dim_accepted: dict[str, int] = defaultdict(int)
    for e in run["_accepted"]:
        claim = next((c for c in run["_claims"] if c["id"] == e.get("claim_id")), None)
        if claim and claim.get("dimension_key"):
            dim_accepted[claim["dimension_key"]] += 1

    per_type: dict[str, dict[str, Any]] = {}
    for req_type in sorted({r["requirement_type"] for r in gap_reqs}):
        group = [r for r in gap_reqs if r["requirement_type"] == req_type]
        states = [gap_effective_state(r) for r in group]
        per_type[req_type] = {
            "total": len(group),
            "open": states.count("open"),
            "partial": states.count("partial"),
            "closed": states.count("closed"),
            "closure_rate": ratio(states.count("closed"), len(group)),
        }
    # Claim -> GapRequirement transition classification (type 1-4)
    type_buckets: dict[str, int] = defaultdict(int)
    for req in gap_reqs:
        dim = req.get("dimension_key") or ""
        if gap_effective_state(req) == "closed":
            type_buckets["satisfied"] += 1
        elif req["current_evidence_count"] == 0 and dim_accepted.get(dim, 0) == 0:
            type_buckets["type1_no_evidence"] += 1
        elif req["current_evidence_count"] == 0:
            type_buckets["type2_candidate_not_accepted"] += 1
        elif req["current_independent_sources"] < req["required_independent_sources"]:
            type_buckets["type3_accepted_but_unverified"] += 1
        else:
            type_buckets["type4_verified_but_gap_open"] += 1
    persisted_closure: dict[str, int] = defaultdict(int)
    persisted_verify: dict[str, int] = defaultdict(int)
    for req in gap_reqs:
        persisted_closure[req["closure_status"]] += 1
        persisted_verify[req["verification_status"]] += 1
    return {
        "by_requirement_type": per_type,
        "transition_types": dict(type_buckets),
        "persisted_state_note": {
            "closure_status": dict(persisted_closure),
            "verification_status": dict(persisted_verify),
        },
    }


# ---------------------------------------------------------------------------
# Coverage loss decomposition (Task K) - MULTI_CAUSE allowed
# ---------------------------------------------------------------------------


def analyze_coverage_loss(run: dict[str, Any]) -> dict[str, Any]:
    gap_reqs = run["_gap_reqs"]
    cause_counts: dict[str, int] = defaultdict(int)
    uncovered = 0
    # A dimension may repeat across plan_versions; summing every row's coverage
    # shortfall would multiply the *same* coverage gap (the historical run reached
    # an impossible 6.58 > 1).  Track the worst shortfall per dimension instead.
    dim_shortfall: dict[str, float] = defaultdict(float)
    dim_causes: dict[str, set[str]] = {}
    all_dims: set[str] = set()
    for req in gap_reqs:
        dim = req.get("dimension_key") or "(none)"
        all_dims.add(dim)
        causes = gap_blocking_causes(req)
        if not causes:
            continue
        uncovered += 1
        for cause in causes:
            cause_counts[cause] += 1
        required_coverage = req.get("required_coverage")
        shortfall = (required_coverage - req["current_coverage"]) if required_coverage else 0.0
        dim_shortfall[dim] = max(dim_shortfall[dim], max(shortfall, 0.0))
        dim_causes.setdefault(dim, set()).update(causes)
    # Coverage is an equal-weight mean over dimensions, so one dimension's
    # shortfall moves the aggregate by shortfall / n_dims (bounded [0, 1]).
    n_dims = max(len(all_dims), 1)
    cause_coverage_impact: dict[str, float] = defaultdict(float)
    total_gap = 0.0
    for dim, blocking in dim_causes.items():
        points = dim_shortfall[dim] / n_dims
        total_gap += points
        share = 1.0 / len(blocking)
        for cause in blocking:
            cause_coverage_impact[cause] += points * share
    return {
        "uncovered_requirements": uncovered,
        "total_requirements": len(gap_reqs),
        "share_uncovered": ratio(uncovered, len(gap_reqs)),
        "aggregate_coverage_shortfall": round(total_gap, 6),
        "causes": [
            {
                "cause": cause,
                "count": count,
                "estimated_coverage_impact": round(cause_coverage_impact[cause], 6),
                "share_of_uncovered": ratio(count, uncovered),
                "multi_cause": True,
            }
            for cause, count in sorted(cause_counts.items(), key=lambda kv: kv[1], reverse=True)
        ],
    }


# ---------------------------------------------------------------------------
# Dimension conversion matrix (Task L)
# ---------------------------------------------------------------------------


def analyze_dimension_matrix(run: dict[str, Any]) -> list[dict[str, Any]]:
    dims: dict[str, dict[str, Any]] = {}
    claim_by_id = {c["id"]: c for c in run["_claims"]}
    source_by_id = {s["id"]: s for s in run["_sources"]}

    def ensure(dim: str) -> dict[str, Any]:
        return dims.setdefault(
            dim,
            {
                "dimension_key": dim,
                "candidate": 0,
                "accepted": 0,
                "supported": 0,
                "verified": 0,
                "gap_states": [],
                "coverage": 0.0,
                "loss_causes": defaultdict(int),
            },
        )

    for e in run["_evidence"]:
        claim = claim_by_id.get(e.get("claim_id") or "")
        dim = (claim or {}).get("dimension_key") or "(none)"
        rec = ensure(dim)
        rec["candidate"] += 1
        if e["accepted"]:
            rec["accepted"] += 1
    dim_required = run["_dim_required_sources"]
    for c in run["_claims"]:
        dim = c.get("dimension_key") or "(none)"
        owners = run["_claim_owners"].get(c["id"], [])
        if owners:
            ensure(dim)["supported"] += 1
        # use the same MAX verification bar across the dimension as the funnel
        req_bar = dim_required.get(dim, 1) or 1
        if len(owners) >= req_bar and owners:
            ensure(dim)["verified"] += 1
        _ = source_by_id  # owners already resolved; keep mapping referenced for clarity
    for req in run["_gap_reqs"]:
        dim = req.get("dimension_key") or "(none)"
        rec = ensure(dim)
        rec["gap_states"].append(gap_effective_state(req))
        rec["coverage"] = max(rec["coverage"], req["current_coverage"])
        for cause in gap_blocking_causes(req):
            rec["loss_causes"][cause] += 1

    rows: list[dict[str, Any]] = []
    for rec in dims.values():
        states = rec["gap_states"] or ["n/a"]
        gap_state = "closed" if all(s == "closed" for s in states) else (
            "open" if all(s == "open" for s in states) else "partial"
        )
        main_loss = (
            max(rec["loss_causes"].items(), key=lambda kv: kv[1])[0]
            if rec["loss_causes"]
            else "none"
        )
        rows.append(
            {
                "dimension_key": rec["dimension_key"],
                "candidate_evidence": rec["candidate"],
                "accepted": rec["accepted"],
                "supported": rec["supported"],
                "verified": rec["verified"],
                "gap_state": gap_state,
                "coverage": round(rec["coverage"], 4),
                "main_loss": main_loss,
            }
        )
    return sorted(rows, key=lambda r: (r["coverage"], -r["candidate_evidence"]))


# ---------------------------------------------------------------------------
# Source role analysis (Task M)
# ---------------------------------------------------------------------------


def analyze_source_roles(run: dict[str, Any]) -> dict[str, Any]:
    source_by_id = {s["id"]: s for s in run["_sources"]}
    roles: dict[str, dict[str, Any]] = {}

    def ensure(key: str) -> dict[str, Any]:
        return roles.setdefault(
            key,
            {
                "source": 0,
                "candidate": 0,
                "accepted": 0,
                "rejected": 0,
                "reasons": defaultdict(int),
            },
        )

    for s in run["_sources"]:
        ensure(str(s.get("source_type") or "(none)"))["source"] += 1
    for e in run["_evidence"]:
        src = source_by_id.get(e.get("source_id") or "")
        key = str((src or {}).get("source_type") or "(none)")
        rec = ensure(key)
        rec["candidate"] += 1
        if e["accepted"]:
            rec["accepted"] += 1
        else:
            rec["rejected"] += 1
            rec["reasons"][str(e.get("rejection_reason") or "(unknown)")] += 1
    out = {}
    for key, rec in roles.items():
        out[key] = {
            "sources": rec["source"],
            "candidate_evidence": rec["candidate"],
            "accepted": rec["accepted"],
            "rejected": rec["rejected"],
            "acceptance_rate": ratio(rec["accepted"], rec["candidate"]),
            "top_rejection": max(rec["reasons"].items(), key=lambda kv: kv[1])[0]
            if rec["reasons"]
            else None,
        }
    return out


# ---------------------------------------------------------------------------
# Evidence waste analysis (Task N) - five types
# ---------------------------------------------------------------------------


def analyze_evidence_waste(run: dict[str, Any]) -> dict[str, Any]:
    evidence_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in run["_evidence"]:
        if e.get("source_id"):
            evidence_by_source[e["source_id"]].append(e)
    readable = run["research_work"]["readable_sources"]
    sources_total = len(run["_sources"])
    w1 = sum(1 for s in run["_sources"] if not evidence_by_source.get(s["id"]))
    w2 = sum(
        1
        for sid, evs in evidence_by_source.items()
        if evs and all(not e["accepted"] for e in evs)
    )
    w3 = run["accepted_without_claim"]
    supported = run["funnel"]["supported_claims"]
    verified = run["funnel"]["verified_claims"]
    w4 = supported - verified
    # verified claims whose dimension gap is still not closed
    closed_dims = {
        r.get("dimension_key")
        for r in run["_gap_reqs"]
        if gap_effective_state(r) == "closed"
    }
    claim_by_id = {c["id"]: c for c in run["_claims"]}
    dim_required = run["_dim_required_sources"]
    w5 = 0
    for c in run["_claims"]:
        owners = run["_claim_owners"].get(c["id"], [])
        # same MAX verification bar as the funnel's verified_claims
        bar = dim_required.get(c.get("dimension_key") or "", 1) or 1
        if owners and len(owners) >= bar and c.get("dimension_key") not in closed_dims:
            w5 += 1
    _ = claim_by_id
    return {
        "readable_sources": readable,
        "sources_total": sources_total,
        "type1_readable_no_evidence": {"count": w1, "rate": ratio(w1, readable)},
        "type2_extracted_all_rejected": {"count": w2, "rate": ratio(w2, len(evidence_by_source))},
        "type3_accepted_no_claim": {"count": w3, "rate": ratio(w3, len(run["_accepted"]))},
        "type4_supported_no_verification": {"count": w4, "rate": ratio(w4, supported)},
        "type5_verified_no_gap_transition": {"count": w5, "rate": ratio(w5, verified)},
    }


# ---------------------------------------------------------------------------
# funnel aggregation (spec §10) + efficiency (Task O)
# ---------------------------------------------------------------------------


def build_funnel(per_run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, float] = defaultdict(float)
    for run in per_run:
        for stage, value in run["funnel"].items():
            if value is not None:
                totals[stage] += value
    rows: list[dict[str, Any]] = []
    prev_key: str | None = None
    mean_runs = max(len(per_run), 1)
    for key, label in FUNNEL_STAGES:
        count = totals.get(key, 0.0)
        coverage = key == "coverage"
        displayed: float | int = (
            round(count / mean_runs, 4) if coverage else round(count / mean_runs)
        )
        prev_count = totals.get(prev_key) if prev_key else None
        dropoff_count = (
            round(prev_count - count) if prev_count is not None and not coverage else None
        )
        rows.append(
            {
                "stage": key,
                "label": label,
                "count": displayed,
                "aggregate_count": round(count, 4) if coverage else int(count),
                "conversion_rate_from_previous": (
                    None if coverage else (ratio(count, prev_count) if prev_count else None)
                ),
                "dropoff_count": dropoff_count,
                "dropoff_rate": (
                    ratio(prev_count - count, prev_count)
                    if prev_count is not None and prev_count > 0 and not coverage
                    else None
                ),
            }
        )
        prev_key = key
    return rows


def compute_efficiency(per_run: list[dict[str, Any]]) -> dict[str, float | None]:
    tot: dict[str, float] = defaultdict(float)
    for run in per_run:
        f = run["funnel"]
        rw = run["research_work"]
        cost = run["cost"]
        tot["searches"] += rw["search_completed"]
        tot["candidate"] += f["candidate_evidence"]
        tot["accepted"] += f["accepted_evidence"]
        tot["supported"] += f["supported_claims"]
        tot["verified"] += f["verified_claims"]
        tot["gap_closed"] += f["gap_satisfied"]
        tot["coverage"] += f["coverage"] or 0.0
        tot["tokens"] += cost["tokens"]
        tot["runtime"] += cost["runtime_seconds"] or 0.0
    n = max(len(per_run), 1)
    coverage_points = tot["coverage"] / n if n else 0.0
    return {
        "candidate_to_accepted_rate": ratio(tot["accepted"], tot["candidate"]),
        "accepted_to_supported_claim_rate": ratio(tot["supported"], tot["accepted"]),
        "supported_to_verified_claim_rate": ratio(tot["verified"], tot["supported"]),
        "verified_to_closed_gap_rate": ratio(tot["gap_closed"], tot["verified"]),
        "candidate_evidence_per_coverage_point": ratio(tot["candidate"], coverage_points),
        "accepted_evidence_per_coverage_point": ratio(tot["accepted"], coverage_points),
        "search_per_candidate_evidence": ratio(tot["searches"], tot["candidate"]),
        "search_per_accepted_evidence": ratio(tot["searches"], tot["accepted"]),
        "search_per_verified_claim": ratio(tot["searches"], tot["verified"]),
        "search_per_closed_gap": ratio(tot["searches"], tot["gap_closed"]),
        "tokens_per_accepted_evidence": ratio(tot["tokens"], tot["accepted"]),
        "tokens_per_verified_claim": ratio(tot["tokens"], tot["verified"]),
        "tokens_per_closed_gap": ratio(tot["tokens"], tot["gap_closed"]),
        "runtime_per_accepted_evidence": ratio(tot["runtime"], tot["accepted"]),
        "runtime_per_closed_gap": ratio(tot["runtime"], tot["gap_closed"]),
    }


# ---------------------------------------------------------------------------
# First major conversion loss (Task P) + primary bottleneck (§22)
# ---------------------------------------------------------------------------


def _edge_conversion_rates(per_run: list[dict[str, Any]]) -> dict[str, float | None]:
    """Bounded [0,1] survival rate for each substantive conversion edge.

    Uses aggregate totals (so expansion stages cannot distort the ratio).  Each
    rate answers "of what reached the previous node, how much survived into this
    one"; 1.0 means the edge is healthy, a low value is where work is lost.
    """

    tot: dict[str, float] = defaultdict(float)
    for run in per_run:
        f = run["funnel"]
        rw = run["research_work"]
        tot["readable"] += rw["readable_sources"]
        tot["candidate"] += f["candidate_evidence"]
        tot["accepted"] += f["accepted_evidence"]
        tot["supported"] += f["supported_claims"]
        tot["verified"] += f["verified_claims"]
        tot["gap_satisfied"] += f["gap_satisfied"]
        tot["gap_total"] += run["quality"]["gap_requirement_total"]

    def bounded(num: float, den: float) -> float | None:
        if den <= 0:
            return None
        return max(0.0, min(1.0, num / den))

    return {
        # Search / reader / extraction -> usable candidate evidence (Case A)
        "Case_A": bounded(tot["candidate"], tot["readable"])
        if tot["readable"]
        else None,
        # Candidate -> Accepted (Case B)
        "Case_B": bounded(tot["accepted"], tot["candidate"]),
        # Accepted/Supported claim -> Verified (Case C)
        "Case_C": bounded(tot["verified"], tot["supported"]),
        # Verified -> Gap requirement satisfied (Case D)
        "Case_D": bounded(tot["gap_satisfied"], tot["gap_total"]),
    }


def classify_primary_bottleneck(per_run: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the first, highest-value conversion loss across the funnel.

    The four substantive edges compete on their survival rate; the lowest rate
    is the primary bottleneck (business significance follows the funnel order,
    so the *first* low edge wins ties).  A near tie across several edges becomes
    ``Case_E`` (Mixed) only when the data genuinely cannot separate them.
    """

    edges = _edge_conversion_rates(per_run)
    order = ("Case_A", "Case_B", "Case_C", "Case_D")
    scored: dict[str, float] = {}
    for case in order:
        rate = edges.get(case)
        if rate is not None:
            scored[case] = rate
    if not scored:
        return {
            "primary": "Case_E",
            "secondary": None,
            "near_tie": False,
            "edge_rates": edges,
            "ranking": [],
        }
    # largest loss first; preserve funnel order for equal rates
    ranked = sorted(scored.items(), key=lambda kv: (kv[1], order.index(kv[0])))
    best_case, best_rate = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    near_tie = bool(
        runner_up is not None and abs(best_rate - runner_up[1]) <= 0.05
    )
    primary = "Case_E" if near_tie and len(ranked) > 1 else best_case
    return {
        "primary": primary,
        "secondary": (runner_up[0] if runner_up else None),
        "near_tie": near_tie,
        "edge_rates": {k: (round(v, 4) if v is not None else None) for k, v in edges.items()},
        "ranking": [{"case": c, "conversion": round(r, 4)} for c, r in ranked],
    }


_CASE_LABELS: Final[dict[str, str]] = {
    "Case_A": "Search -> Candidate Evidence (source targeting / reader / extraction)",
    "Case_B": "Candidate -> Accepted (dimension / role / relevance / entailment unfit)",
    "Case_C": "Accepted -> Claim Verified (association / independent source / verification)",
    "Case_D": "Verified Claim -> Gap Closed (closure / requirement satisfaction)",
    "Case_E": "Mixed (multiple nodes lose a comparable amount)",
}


def estimate_coverage_unlock(
    mean_coverage: float | None,
    bottleneck: dict[str, Any],
    coverage_loss: dict[str, Any],
) -> dict[str, Any]:
    """Analytical upper bound if the primary bottleneck were fully removed."""

    if mean_coverage is None:
        return {"current_coverage": None, "status": "insufficient-data"}
    case = bottleneck.get("primary")
    impact = 0.0
    cause_map = {
        c["cause"]: c["estimated_coverage_impact"] for c in coverage_loss.get("causes", [])
    }
    per_run_impact = cause_map.get("missing_independent_source", 0.0) + cause_map.get(
        "coverage_below_required", 0.0
    )
    if case == "Case_B":
        impact = sum(
            cause_map.get(k, 0.0)
            for k in ("missing_candidate_evidence", "coverage_below_required")
        )
    elif case == "Case_C":
        impact = cause_map.get("missing_independent_source", 0.0)
    elif case == "Case_D":
        impact = cause_map.get("coverage_below_required", 0.0)
    elif case == "Case_A":
        impact = cause_map.get("missing_candidate_evidence", 0.0)
    else:
        impact = per_run_impact
    # normalise the aggregate shortfall (summed over runs) to a per-run value
    ceiling = min(1.0, mean_coverage + impact)
    return {
        "current_coverage": round(mean_coverage, 4),
        "primary_case": case,
        "primary_label": _CASE_LABELS.get(str(case), str(case)),
        "estimated_blocked_coverage": round(impact, 4),
        "theoretical_upper_bound": round(ceiling, 4),
        "estimated": True,
        "counterfactual": True,
        "experimentally_validated": False,
        "note": (
            "Analytical upper-bound approximation from the coverage shortfall of "
            "requirements attributable to the primary bottleneck; not a promise and "
            "not experimentally validated."
        ),
    }


# ---------------------------------------------------------------------------
# cross-run aggregation (Task E: mean / SD / min / max)
# ---------------------------------------------------------------------------


def aggregate_runs(per_run: list[dict[str, Any]]) -> dict[str, Any]:
    def collect(section: str, field: str) -> list[float]:
        values: list[float] = []
        for run in per_run:
            raw = run.get(section, {}).get(field)
            val = _as_float(raw)
            if val is not None:
                values.append(val)
        return values

    return {
        "run_count": len(per_run),
        "quality": {
            field: _summary(collect("quality", field))
            for field in (
                "coverage",
                "priority_one_coverage",
                "critical_gap_count",
                "candidate_evidence",
                "accepted_evidence",
                "acceptance_rate",
                "supported_claims",
                "verified_claims",
                "unverified_claims",
                "gap_requirement_total",
                "gap_open",
                "gap_partial",
                "gap_closed",
                "gap_closure_rate",
            )
        },
        "research_work": {
            field: _summary(collect("research_work", field))
            for field in (
                "logical_queries",
                "search_started",
                "search_completed",
                "provider_requests",
                "candidate_urls",
                "unique_urls",
                "readable_sources",
                "reader_completed",
                "evidence_extractions",
            )
        },
        "cost": {
            field: _summary(collect("cost", field))
            for field in ("runtime_seconds", "tokens")
        },
    }


def analyze_run(card: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    """Full per-run analysis bundle."""

    metrics = build_run_metrics(
        card,
        raw["claims"],
        raw["evidence"],
        raw["sources"],
        raw["gap_reqs"],
        raw["events"],
        raw["storage"],
    )
    return {
        "metrics": metrics,
        "rejections": analyze_rejections(metrics),
        "claim_conversion": analyze_claim_conversion(metrics),
        "gap_requirements": analyze_gap_requirements(metrics),
        "coverage_loss": analyze_coverage_loss(metrics),
        "dimension_matrix": analyze_dimension_matrix(metrics),
        "source_roles": analyze_source_roles(metrics),
        "evidence_waste": analyze_evidence_waste(metrics),
    }


def _aggregate_rejections(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: dict[str, int] = defaultdict(int)
    situations: dict[str, int] = defaultdict(int)
    candidate = accepted = 0
    for b in bundles:
        r = b["rejections"]
        candidate += r["candidate_evidence"]
        accepted += r["accepted"]
        for item in r["reasons"]:
            reasons[item["reason"]] += item["count"]
        for k, v in r["situation_totals"].items():
            situations[k] += v
    rejected = candidate - accepted
    ranked = sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "candidate_evidence": candidate,
        "accepted": accepted,
        "rejected": rejected,
        "candidate_to_accepted_rate": ratio(accepted, candidate),
        "reasons": [
            {
                "reason": reason,
                "count": count,
                "share_of_rejected": ratio(count, rejected),
                "share_of_candidate": ratio(count, candidate),
                "situation": _rejection_situation(reason),
            }
            for reason, count in ranked
        ],
        "situation_totals": dict(situations),
    }


def _aggregate_coverage_loss(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    impacts: dict[str, float] = defaultdict(float)
    uncovered = total = 0
    for b in bundles:
        c = b["coverage_loss"]
        uncovered += c["uncovered_requirements"]
        total += c["total_requirements"]
        for cause in c["causes"]:
            counts[cause["cause"]] += cause["count"]
            impacts[cause["cause"]] += cause["estimated_coverage_impact"]
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "uncovered_requirements": uncovered,
        "total_requirements": total,
        "share_uncovered": ratio(uncovered, total),
        "causes": [
            {
                "cause": cause,
                "count": count,
                "estimated_coverage_impact": round(impacts[cause] / max(len(bundles), 1), 4),
                "share_of_uncovered": ratio(count, uncovered),
                "multi_cause": True,
            }
            for cause, count in ranked
        ],
    }


def _aggregate_waste(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    types = [
        "type1_readable_no_evidence",
        "type2_extracted_all_rejected",
        "type3_accepted_no_claim",
        "type4_supported_no_verification",
        "type5_verified_no_gap_transition",
    ]
    out: dict[str, Any] = {}
    for t in types:
        counts = [b["evidence_waste"][t]["count"] for b in bundles]
        rates = [_as_float(b["evidence_waste"][t]["rate"]) for b in bundles]
        rates_present = [r for r in rates if r is not None]
        out[t] = {
            "mean_count": round(statistics.fmean(counts), 2) if counts else 0,
            "mean_rate": round(statistics.fmean(rates_present), 4) if rates_present else None,
        }
    return out


def _aggregate_source_roles(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    roles: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for b in bundles:
        for stype, rec in b["source_roles"].items():
            roles[stype]["sources"] += rec["sources"]
            roles[stype]["candidate"] += rec["candidate_evidence"]
            roles[stype]["accepted"] += rec["accepted"]
            roles[stype]["rejected"] += rec["rejected"]
    return {
        stype: {
            **dict(vals),
            "acceptance_rate": ratio(vals["accepted"], vals["candidate"]),
        }
        for stype, vals in sorted(roles.items())
    }


def _aggregate_claim_conversion(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    sums = {
        "accepted_evidence_total": 0,
        "accepted_with_claim": 0,
        "accepted_without_claim": 0,
        "supported_claims": 0,
        "verified_claims": 0,
        "unverified_claims": 0,
        "single_source_claims": 0,
        "multi_source_claims": 0,
    }
    for b in bundles:
        c = b["claim_conversion"]
        for key in sums:
            sums[key] += c[key]
    return {
        **sums,
        "accepted_to_supported_rate": ratio(
            sums["supported_claims"], sums["accepted_evidence_total"]
        ),
        "supported_to_verified_rate": ratio(sums["verified_claims"], sums["supported_claims"]),
    }


def _first_major_loss(funnel: list[dict[str, Any]]) -> dict[str, Any] | None:
    loss_stages = {"candidate_evidence", "accepted_evidence", "verified_claims", "gap_satisfied"}
    candidates = [
        row
        for row in funnel
        if row["conversion_rate_from_previous"] is not None and row["stage"] in loss_stages
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r["conversion_rate_from_previous"])


def build_report(
    run_ids: list[str],
    cards: list[dict[str, Any]],
    bundles: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = [b["metrics"] for b in bundles]
    funnel = build_funnel(metrics)
    bottleneck = classify_primary_bottleneck(metrics)
    coverage_loss = _aggregate_coverage_loss(bundles)
    agg = aggregate_runs(metrics)
    mean_cov = agg["quality"]["coverage"]["mean"]
    unlock = estimate_coverage_unlock(mean_cov, bottleneck, coverage_loss)
    identities = [
        {
            "run_id": m["run_id"],
            "status": m["status"],
            "benchmark_id": m["benchmark_id"],
            "benchmark_version": m.get("benchmark_version"),
            "metric_definition_version": m.get("metric_definition_version"),
        }
        for m in metrics
    ]
    return {
        "schema_version": "phase15-conversion-funnel.v1",
        "run_ids": run_ids,
        "identities": identities,
        "golden_baseline_summary": agg,
        "funnel": funnel,
        "first_major_loss": _first_major_loss(funnel),
        "rejections": _aggregate_rejections(bundles),
        "claim_conversion": _aggregate_claim_conversion(bundles),
        "gap_requirements": [b["gap_requirements"] for b in bundles],
        "coverage_loss": coverage_loss,
        "dimension_matrix": [b["dimension_matrix"] for b in bundles],
        "source_roles": _aggregate_source_roles(bundles),
        "evidence_waste": _aggregate_waste(bundles),
        "efficiency": compute_efficiency(metrics),
        "primary_bottleneck": {
            **bottleneck,
            "primary_label": _CASE_LABELS.get(str(bottleneck.get("primary")), ""),
        },
        "estimated_coverage_unlock": unlock,
        "per_run": [
            {
                "run_id": m["run_id"],
                "quality": m["quality"],
                "research_work": m["research_work"],
                "cost": m["cost"],
            }
            for m in metrics
        ],
    }


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".") if abs(value) < 1 else f"{value:.2f}"
    return str(value)


def _fmt_summary(block: dict[str, Any]) -> str:
    mean = block.get("mean")
    sd = block.get("sd")
    if mean is None:
        return "n/a"
    return f"{mean:.4g} ± {sd:.4g}"


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = ["# Phase 15.0 — Evidence-to-Coverage Conversion Funnel", ""]
    lines.append(f"Runs analysed: {', '.join(report['run_ids'])}")
    lines.append("")
    lines.append("## Golden Baseline Summary (mean ± SD across runs)")
    q = report["golden_baseline_summary"]["quality"]
    quality_fields = (
        "coverage",
        "acceptance_rate",
        "verified_claims",
        "gap_closed",
        "gap_closure_rate",
    )
    for field in quality_fields:
        lines.append(f"- **{field}**: {_fmt_summary(q[field])}")
    rw = report["golden_baseline_summary"]["research_work"]
    work_fields = (
        "logical_queries",
        "search_completed",
        "readable_sources",
        "evidence_extractions",
    )
    for field in work_fields:
        lines.append(f"- **{field}**: {_fmt_summary(rw[field])}")
    cost = report["golden_baseline_summary"]["cost"]
    lines.append(f"- **runtime_seconds**: {_fmt_summary(cost['runtime_seconds'])}")
    lines.append(f"- **tokens**: {_fmt_summary(cost['tokens'])}")
    lines.append("")
    lines.append("## Conversion Funnel")
    lines.append("| Stage | Count (mean/run) | Conversion | Dropoff | Dropoff rate |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in report["funnel"]:
        lines.append(
            f"| {row['label']} | {_fmt(row['count'])} "
            f"| {_fmt(row['conversion_rate_from_previous'])} "
            f"| {_fmt(row['dropoff_count'])} | {_fmt(row['dropoff_rate'])} |"
        )
    fm = report.get("first_major_loss")
    if fm:
        lines.append("")
        rate = _fmt(fm["conversion_rate_from_previous"])
        lines.append(f"**First major conversion loss**: {fm['label']} (rate {rate})")
    lines.append("")
    lines.append("## Candidate Evidence Rejection Breakdown")
    lines.append("| Reason | Count | Share of rejected | Situation |")
    lines.append("| --- | --- | --- | --- |")
    for item in report["rejections"]["reasons"]:
        lines.append(
            f"| {item['reason']} | {item['count']} "
            f"| {_fmt(item['share_of_rejected'])} | {item['situation']} |"
        )
    lines.append("")
    lines.append("## Claim Conversion")
    cc = report["claim_conversion"]
    lines.append(
        f"- accepted_with_claim: {cc['accepted_with_claim']} "
        f"/ accepted_without_claim: {cc['accepted_without_claim']}"
    )
    lines.append(
        f"- supported: {cc['supported_claims']} "
        f"(single {cc['single_source_claims']} / multi {cc['multi_source_claims']})"
    )
    lines.append(f"- verified: {cc['verified_claims']} / unverified: {cc['unverified_claims']}")
    lines.append(
        f"- accepted→supported {_fmt(cc['accepted_to_supported_rate'])}, "
        f"supported→verified {_fmt(cc['supported_to_verified_rate'])}"
    )
    lines.append("")
    lines.append("## Coverage Loss Decomposition")
    lines.append("| Cause | Count | Est. coverage impact | Share of uncovered |")
    lines.append("| --- | --- | --- | --- |")
    for cause in report["coverage_loss"]["causes"]:
        lines.append(
            f"| {cause['cause']} | {cause['count']} "
            f"| {_fmt(cause['estimated_coverage_impact'])} "
            f"| {_fmt(cause['share_of_uncovered'])} |"
        )
    lines.append("")
    lines.append("## Evidence Waste")
    for t, rec in report["evidence_waste"].items():
        lines.append(f"- {t}: count {_fmt(rec['mean_count'])}, rate {_fmt(rec['mean_rate'])}")
    lines.append("")
    lines.append("## Efficiency Metrics")
    eff = report["efficiency"]
    for key, value in eff.items():
        lines.append(f"- {key}: {_fmt(value)}")
    lines.append("")
    pb = report["primary_bottleneck"]
    lines.append("## Primary Bottleneck")
    lines.append(f"- **Primary**: {pb.get('primary')} — {pb.get('primary_label')}")
    lines.append(f"- **Secondary**: {pb.get('secondary')}")
    lines.append(f"- Edge rates: {json.dumps(pb.get('edge_rates', {}))}")
    unlock = report["estimated_coverage_unlock"]
    lines.append("")
    lines.append("## Estimated Coverage Unlock (counterfactual, not validated)")
    lines.append(f"- current coverage: {_fmt(unlock.get('current_coverage'))}")
    lines.append(
        "- estimated blocked by primary: "
        f"{_fmt(unlock.get('estimated_blocked_coverage'))}"
    )
    lines.append(f"- theoretical upper bound: {_fmt(unlock.get('theoretical_upper_bound'))}")
    lines.append(
        f"- flags: estimated={unlock.get('estimated')}, "
        f"counterfactual={unlock.get('counterfactual')}, "
        f"experimentally_validated={unlock.get('experimentally_validated')}"
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# entry point (read-only)
# ---------------------------------------------------------------------------


def _split_run_ids(raw: str) -> list[str]:
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


def analyze(run_ids: list[str], database_url: str) -> dict[str, Any]:
    if psycopg is None:  # pragma: no cover
        raise RuntimeError("psycopg is required for the live funnel analysis")
    cards: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    with psycopg.connect(_database_uri(database_url)) as connection:
        for run_id in run_ids:
            card = _fetch_run_card(connection, run_id)
            if not card.get("found"):
                raise RuntimeError(f"run {run_id} not found")
            raw = {
                "claims": _fetch_claims(connection, run_id),
                "evidence": _fetch_evidence(connection, run_id),
                "sources": _fetch_sources(connection, run_id),
                "gap_reqs": _fetch_gap_requirements(connection, run_id),
                "events": _fetch_event_counts(connection, run_id),
                "storage": _fetch_storage_stats(connection, run_id),
            }
            cards.append(card)
            bundles.append(analyze_run(card, raw))
    return build_report(run_ids, cards, bundles)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ids", required=True, help="comma-separated Golden Baseline run ids")
    parser.add_argument(
        "--database-url",
        default="postgresql+psycopg://deep_research:deep_research@127.0.0.1:5432/deep_research",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("artifacts/phase15_conversion_funnel"),
        help="writes <prefix>.json and <prefix>.md",
    )
    args = parser.parse_args()
    run_ids = _split_run_ids(args.run_ids)
    if len(run_ids) < 1:
        raise SystemExit("at least one run id is required")
    report = analyze(run_ids, args.database_url)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.out_prefix.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    args.out_prefix.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(
        f"wrote {args.out_prefix.with_suffix('.json')} and "
        f"{args.out_prefix.with_suffix('.md')} for {len(run_ids)} run(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
