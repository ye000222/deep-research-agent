#!/usr/bin/env python3
"""Phase 15.1 Task A — Source Independence Ground-Truth Audit (read-only).

Phase 15.0 established that the V1 Golden Baseline verifies *zero* claims even
though 100% of accepted evidence becomes claim support.  The single most
important discipline of this phase (spec §2 / §32) is to **separate Source Role
from Source Identity before deciding what to fix**: a corpus that is "100%
webpage" must not be mistaken for "only one independent source".

This analyzer answers exactly one question with real data: is ``verified = 0``
caused by an *identity-modeling* error (multiple genuinely distinct sources
being collapsed to one by verification) or by a *genuine independent-source
shortage* (each claim really only has one source)?

Design discipline (Generality Guard, spec §28):

* Every query is a ``SELECT`` — nothing is written to the database.
* No question-id, dimension-name, domain or benchmark special-casing.  All
  classification is driven by generic, persisted identity fields already in the
  schema (``research_sources.domain`` / ``source_owner_key`` / ``source_type``),
  the real ``research_evidence.accepted`` flag, and the numeric
  ``gap_requirements.required_independent_sources`` bar.
* Independence is measured by *owner / registrable domain*, never by the
  ``source_type`` role label.  The role is only ever reported as a diagnostic
  cross-check so we can prove whether a role-based check would mislead.

Outputs (per spec §4):
``artifacts/phase15_1_source_independence_audit.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Final

try:  # psycopg is only needed for the live read; tests import the pure helpers.
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPOSITORY_ROOT / "apps" / "api"
for _path in (str(REPOSITORY_ROOT), str(API_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from app.domain.source_policy import source_owner_key  # noqa: E402

# ---------------------------------------------------------------------------
# taxonomy (generic; no domain / question / benchmark literals)
# ---------------------------------------------------------------------------

#: Ground-truth classification buckets for a *supported* claim (spec §11 Task H).
IDENTITY_COLLAPSE: Final[str] = "IDENTITY_COLLAPSE"
GENUINE_SINGLE_SOURCE: Final[str] = "GENUINE_SINGLE_SOURCE"
MISSING_SOURCE_METADATA: Final[str] = "MISSING_SOURCE_METADATA"
MIXED: Final[str] = "MIXED"
UNKNOWN: Final[str] = "UNKNOWN"
ALREADY_VERIFIED: Final[str] = "ALREADY_VERIFIED"

#: Owner/domain placeholder values that mean "we could not attribute a real
#: publisher" — conservative, not counted as an independent source.
UNKNOWN_OWNER_KEYS: Final[frozenset[str]] = frozenset({"", "unknown", "none", "null"})

#: Branch-decision thresholds (spec §12 Task I).  A branch is chosen purely from
#: the *share* of supported-but-unverified claims falling into each class, never
#: from a specific claim, question or domain.
COLLAPSE_DOMINANT_SHARE: Final[float] = 0.5
COLLAPSE_MEANINGFUL_SHARE: Final[float] = 0.2
GENUINE_DOMINANT_SHARE: Final[float] = 0.5


def _database_uri(value: str) -> str:
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def _split_run_ids(raw: str) -> list[str]:
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


# ---------------------------------------------------------------------------
# pure helpers (fully unit-testable without a database)
# ---------------------------------------------------------------------------


def is_unknown_owner(owner_key: str | None) -> bool:
    """Whether a persisted owner key carries no usable attribution."""

    return (owner_key or "").strip().casefold() in UNKNOWN_OWNER_KEYS


def identity_collapse(observed_domains: int, observed_owners: int) -> bool:
    """True when two or more distinct domains were merged into a single owner.

    This is the exact signature of an identity-modeling bug: the corpus clearly
    holds more than one registrable domain, yet independence evaluation counts
    only one.  Sub-domain folding alone (many hosts, one owner) is *correct*
    publisher normalization and therefore only counts as collapse when it drops
    the distinct-owner count below the distinct-domain count *to one*.
    """

    return observed_domains >= 2 and observed_owners <= 1


def classify_support(
    *,
    accepted: int,
    distinct_owners: int,
    distinct_domains: int,
    required: int | None,
    unknown_metadata: bool,
) -> str:
    """Classify one *supported* claim into a Task-H bucket.

    The decision uses the numeric independent-source bar and the persisted
    identity fields only; it never inspects the claim/question/domain text.
    """

    if accepted < 1:
        return UNKNOWN
    if unknown_metadata:
        return MISSING_SOURCE_METADATA
    collapse = identity_collapse(distinct_domains, distinct_owners)
    meets_bar = required is not None and distinct_owners >= required
    if meets_bar and collapse:
        # Would be verified only because domains were (wrongly) counted
        # separately while collapsing to one owner — contradictory signal.
        return MIXED
    if meets_bar:
        return ALREADY_VERIFIED
    if collapse and distinct_owners >= 1 and required is not None and distinct_owners < required:
        # Genuine multi-domain material that verification under-counted.
        return IDENTITY_COLLAPSE
    if distinct_owners <= 1:
        return GENUINE_SINGLE_SOURCE
    # More than one owner but still below the bar, with no collapse: a real (if
    # partial) shortage.  Report MIXED so it is not silently treated as verified.
    return MIXED


def decide_branch(classification_counts: dict[str, int]) -> dict[str, Any]:
    """Pick implementation Branch A / B / C from the audit shares (Task I)."""

    unverified = {
        IDENTITY_COLLAPSE: classification_counts.get(IDENTITY_COLLAPSE, 0),
        GENUINE_SINGLE_SOURCE: classification_counts.get(GENUINE_SINGLE_SOURCE, 0),
        MIXED: classification_counts.get(MIXED, 0),
        MISSING_SOURCE_METADATA: classification_counts.get(MISSING_SOURCE_METADATA, 0),
        UNKNOWN: classification_counts.get(UNKNOWN, 0),
    }
    total_unverified = sum(unverified.values())
    if total_unverified == 0:
        return {"branch": "NONE", "shares": {}, "rationale": "no supported-but-unverified claims"}
    collapse_share = (
        unverified[IDENTITY_COLLAPSE] + 0.5 * unverified[MIXED]
    ) / total_unverified
    genuine_share = unverified[GENUINE_SINGLE_SOURCE] / total_unverified
    shares = {
        "identity_collapse_share": round(collapse_share, 4),
        "genuine_shortage_share": round(genuine_share, 4),
        "total_unverified": total_unverified,
    }
    if collapse_share >= COLLAPSE_DOMINANT_SHARE:
        branch = "A"
        rationale = "identity collapse dominates; fix independence evaluation first"
    elif collapse_share >= COLLAPSE_MEANINGFUL_SHARE and genuine_share >= GENUINE_DOMINANT_SHARE:
        branch = "C"
        rationale = "both present; fix identity modeling, then target remaining genuine shortage"
    elif genuine_share >= GENUINE_DOMINANT_SHARE and collapse_share < COLLAPSE_MEANINGFUL_SHARE:
        branch = "B"
        rationale = (
            "independent-source shortage dominates with no identity collapse; "
            "add independent-source targeting rather than re-scoring identity"
        )
    else:
        branch = "UNKNOWN"
        rationale = "shares are inconclusive; escalate for manual review"
    return {"branch": branch, "shares": shares, "rationale": rationale}


# ---------------------------------------------------------------------------
# DB read layer (SELECT only)
# ---------------------------------------------------------------------------


def _fetch_identity(connection: Any, run_id: str) -> dict[str, Any]:
    row = connection.execute(
        "SELECT budget_snapshot FROM research_runs WHERE id = %s", (run_id,)
    ).fetchone()
    budget: dict[str, Any] = row[0] if row and isinstance(row[0], dict) else {}
    raw_benchmark = budget.get("benchmark")
    benchmark: dict[str, Any] = raw_benchmark if isinstance(raw_benchmark, dict) else {}
    return {
        "benchmark_id": benchmark.get("benchmark_id"),
        "benchmark_version": benchmark.get("benchmark_version"),
        "metric_definition_version": benchmark.get("metric_definition_version"),
        "tier": budget.get("tier"),
    }


def _fetch_required_bars(connection: Any, run_id: str) -> dict[str, int]:
    """dimension_key -> MAX(required_independent_sources) across plan versions."""

    bars: dict[str, int] = {}
    rows = connection.execute(
        """
        SELECT dimension_key, required_independent_sources
        FROM gap_requirements
        WHERE run_id = %s
        """,
        (run_id,),
    ).fetchall()
    for dimension_key, required in rows:
        key = str(dimension_key)
        value = int(required or 0)
        bars[key] = max(bars.get(key, 0), value)
    return bars


def _fetch_sources(connection: Any, run_id: str) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    rows = connection.execute(
        """
        SELECT id, canonical_url, domain, source_owner_key, source_type
        FROM research_sources
        WHERE run_id = %s
        """,
        (run_id,),
    ).fetchall()
    for sid, canonical_url, domain, owner_key, source_type in rows:
        sources[str(sid)] = {
            "canonical_url": canonical_url,
            "domain": domain,
            "source_owner_key": owner_key,
            "source_type": source_type,
        }
    return sources


def _fetch_claim_evidence(
    connection: Any, run_id: str, sources: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Per-claim accepted / rejected identity rollup driven by owner key."""

    claims: dict[str, dict[str, Any]] = {}
    rows = connection.execute(
        """
        SELECT c.id, c.dimension_key, c.status, e.id, e.accepted, e.source_id
        FROM research_claims c
        JOIN research_evidence e ON e.claim_id = c.id
        WHERE c.run_id = %s
        """,
        (run_id,),
    ).fetchall()
    for claim_id, dimension_key, status, evidence_id, accepted, source_id in rows:
        claim = claims.setdefault(
            str(claim_id),
            {
                "claim_id": str(claim_id),
                "dimension_key": str(dimension_key),
                "status": status,
                "accepted_evidence": [],
                "rejected_evidence": 0,
                "accepted_owners": set(),
                "accepted_domains": set(),
                "accepted_roles": set(),
                "all_owners": set(),
                "unknown_metadata": False,
            },
        )
        source = sources.get(str(source_id), {})
        owner_key = source.get("source_owner_key")
        domain = source.get("domain")
        role = source.get("source_type")
        if bool(accepted):
            claim["accepted_evidence"].append(str(evidence_id))
            if is_unknown_owner(owner_key) or is_unknown_owner(domain):
                claim["unknown_metadata"] = True
            else:
                claim["accepted_owners"].add(str(owner_key))
                claim["accepted_domains"].add(str(domain))
            if role:
                claim["accepted_roles"].add(str(role))
        else:
            claim["rejected_evidence"] += 1
        if owner_key and not is_unknown_owner(owner_key):
            claim["all_owners"].add(str(owner_key))
    return claims


def audit_run(
    connection: Any, run_id: str, *, fallback_required: int
) -> dict[str, Any]:
    identity = _fetch_identity(connection, run_id)
    bars = _fetch_required_bars(connection, run_id)
    sources = _fetch_sources(connection, run_id)
    claims = _fetch_claim_evidence(connection, run_id, sources)

    # Corpus-level role vs identity cross-check: prove the role label is not
    # what independence is (or should be) keyed on.
    owners_by_role: dict[str, set[str]] = defaultdict(set)
    owner_rederivation_mismatches = 0
    for source in sources.values():
        owner = source.get("source_owner_key")
        role = source.get("source_type") or "(none)"
        if owner and not is_unknown_owner(owner):
            owners_by_role[role].add(str(owner))
        # Task E/G: the persisted owner key must equal a fresh registrable-domain
        # derivation from the same URL.  A mismatch would itself be an identity
        # bug (persistence drifting from the policy), independent of role.
        canonical = source.get("canonical_url")
        if canonical and owner and source_owner_key(str(canonical)) != str(owner):
            owner_rederivation_mismatches += 1
    corpus_distinct_owners = {
        str(owner) for s in sources.values() if (owner := s.get("source_owner_key"))
    }

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = defaultdict(int)
    role_would_collapse = 0
    for claim in claims.values():
        accepted = len(claim["accepted_evidence"])
        if accepted < 1:
            continue  # only *supported* claims are audited for verification
        distinct_owners = len(claim["accepted_owners"])
        distinct_domains = len(claim["accepted_domains"])
        distinct_roles = len(claim["accepted_roles"])
        required = bars.get(claim["dimension_key"], fallback_required) or fallback_required
        classification = classify_support(
            accepted=accepted,
            distinct_owners=distinct_owners,
            distinct_domains=distinct_domains,
            required=required,
            unknown_metadata=claim["unknown_metadata"],
        )
        counts[classification] += 1
        # Diagnostic: a role-based dedup would see one role here while the
        # owner identity actually spans multiple publishers.
        if distinct_roles <= 1 and distinct_owners >= 2:
            role_would_collapse += 1
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "dimension_key": claim["dimension_key"],
                "accepted_evidence": accepted,
                "rejected_evidence": claim["rejected_evidence"],
                "distinct_domains": distinct_domains,
                "distinct_owners": distinct_owners,
                "distinct_roles": distinct_roles,
                "required_independent_sources": required,
                "verified": distinct_owners >= required,
                "identity_collapse": identity_collapse(distinct_domains, distinct_owners),
                "classification": classification,
            }
        )

    rows.sort(key=lambda r: (r["dimension_key"], r["claim_id"]))
    supported = len(rows)
    unverified = sum(
        1 for r in rows if r["classification"] != ALREADY_VERIFIED
    )
    return {
        "run_id": run_id,
        "identity": identity,
        "corpus": {
            "sources": len(sources),
            "distinct_owners": len(corpus_distinct_owners),
            "distinct_roles": len(owners_by_role),
            "owners_per_role": {role: len(owners) for role, owners in owners_by_role.items()},
            "owner_rederivation_mismatches": owner_rederivation_mismatches,
        },
        "supported_claims": supported,
        "verified_claims": sum(1 for r in rows if r["verified"]),
        "supported_but_unverified": unverified,
        "classification_counts": dict(counts),
        "role_would_collapse_if_used_for_verification": role_would_collapse,
        "claims": rows,
    }


def _aggregate(per_run: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    supported = verified = 0
    role_collapse = 0
    for run in per_run:
        supported += run["supported_claims"]
        verified += run["verified_claims"]
        role_collapse += run["role_would_collapse_if_used_for_verification"]
        for key, value in run["classification_counts"].items():
            counts[key] += value
    total = max(supported, 1)
    shares = {key: round(value / total, 4) for key, value in counts.items()}
    identity_collapse_total = counts.get(IDENTITY_COLLAPSE, 0)
    identity_modeling_correct = identity_collapse_total == 0
    branch = decide_branch(dict(counts))
    return {
        "supported_claims_total": supported,
        "verified_claims_total": verified,
        "role_would_collapse_total": role_collapse,
        "classification_counts": dict(counts),
        "classification_shares": shares,
        "identity_collapse_detected": identity_collapse_total > 0,
        "identity_modeling_appears_correct": identity_modeling_correct,
        "branch_decision": branch,
    }


def analyze(run_ids: list[str], database_url: str) -> dict[str, Any]:
    if psycopg is None:  # pragma: no cover
        raise RuntimeError("psycopg is required to run the source-independence audit")
    per_run: list[dict[str, Any]] = []
    with psycopg.connect(_database_uri(database_url)) as connection:
        # A single run-level fallback bar, taken from the data (MAX persisted
        # requirement) so dimensions without their own row are not hardcoded.
        fallback_required = _run_level_fallback(connection, run_ids)
        for run_id in run_ids:
            per_run.append(audit_run(connection, run_id, fallback_required=fallback_required))
    return {
        "schema_version": "phase15.1-source-independence-audit.v1",
        "run_ids": run_ids,
        "fallback_required_independent_sources": fallback_required,
        "per_run": per_run,
        "aggregate": _aggregate(per_run),
    }


def _run_level_fallback(connection: Any, run_ids: list[str]) -> int:
    values: list[int] = []
    for run_id in run_ids:
        rows = connection.execute(
            """
            SELECT required_independent_sources FROM gap_requirements
            WHERE run_id = %s AND requirement_type IN ('independent_source', 'claim_verification')
            """,
            (run_id,),
        ).fetchall()
        values.extend(int(row[0]) for row in rows if row[0] is not None)
    return max(values) if values else 2


# ---------------------------------------------------------------------------
# markdown rendering (Task H table + branch conclusion)
# ---------------------------------------------------------------------------


def render_markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    branch = aggregate["branch_decision"]
    lines: list[str] = ["# Phase 15.1 — Source Independence Ground-Truth Audit", ""]
    lines.append(f"Runs audited: {', '.join(report['run_ids'])}")
    lines.append("")
    lines.append("## Aggregate Classification (Task H)")
    lines.append("| Classification | Count | Share of supported |")
    lines.append("| --- | --- | --- |")
    order = (
        ALREADY_VERIFIED,
        IDENTITY_COLLAPSE,
        GENUINE_SINGLE_SOURCE,
        MISSING_SOURCE_METADATA,
        MIXED,
        UNKNOWN,
    )
    counts = aggregate["classification_counts"]
    shares = aggregate["classification_shares"]
    for key in order:
        if key in counts:
            lines.append(f"| {key} | {counts[key]} | {shares.get(key, 0.0)} |")
    lines.append("")
    lines.append("## Root-Cause Gate")
    lines.append(
        f"- Supported claims: {aggregate['supported_claims_total']}, "
        f"verified: {aggregate['verified_claims_total']}"
    )
    lines.append(
        f"- Identity-collapse claims: {counts.get(IDENTITY_COLLAPSE, 0)} "
        f"(role-based dedup would have collapsed "
        f"{aggregate['role_would_collapse_total']} claims that actually carry "
        f"distinct owners)"
    )
    verdict = (
        "Identity Modeling appears correct (no IDENTITY_COLLAPSE detected)"
        if aggregate["identity_modeling_appears_correct"]
        else "Identity Collapse detected"
    )
    lines.append(f"- Verdict: **{verdict}**")
    lines.append(f"- Branch (Task I): **{branch['branch']}** — {branch['rationale']}")
    if branch.get("shares"):
        lines.append(f"- Branch shares: {branch['shares']}")
    lines.append("")
    lines.append("## Per-Run Corpus (role vs identity)")
    for run in report["per_run"]:
        corpus = run["corpus"]
        lines.append(
            f"- `{run['run_id']}`: {corpus['sources']} sources, "
            f"{corpus['distinct_owners']} distinct owners, {corpus['distinct_roles']} distinct "
            f"role(s) -> owners_per_role={corpus['owners_per_role']}; "
            f"supported={run['supported_claims']} verified={run['verified_claims']}"
        )
    lines.append("")
    lines.append("## Claim Table (first 40 rows)")
    lines.append("| Claim | Dimension | Acc | Domains | Owners | Roles | Required | Class |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    shown = 0
    for run in report["per_run"]:
        for row in run["claims"]:
            if shown >= 40:
                break
            lines.append(
                f"| `{row['claim_id'][:8]}` | {row['dimension_key']} | "
                f"{row['accepted_evidence']} | {row['distinct_domains']} | "
                f"{row['distinct_owners']} | {row['distinct_roles']} | "
                f"{row['required_independent_sources']} | {row['classification']} |"
            )
            shown += 1
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-ids", required=True, help="comma-separated golden run ids")
    parser.add_argument(
        "--database-url",
        default="postgresql+psycopg://deep_research:deep_research@127.0.0.1:5432/deep_research",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "phase15_1_source_independence_audit",
        help="writes <prefix>.json and <prefix>.md",
    )
    args = parser.parse_args()
    run_ids = _split_run_ids(args.run_ids)
    if not run_ids:
        raise SystemExit("at least one run id is required")
    report = analyze(run_ids, args.database_url)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.out_prefix.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    args.out_prefix.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    branch = report["aggregate"]["branch_decision"]
    print(
        f"wrote {args.out_prefix.with_suffix('.json')} and "
        f"{args.out_prefix.with_suffix('.md')} "
        f"(branch={branch['branch']}, identity_collapse="
        f"{report['aggregate']['classification_counts'].get(IDENTITY_COLLAPSE, 0)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
