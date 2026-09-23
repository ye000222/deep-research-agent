"""Phase 15.0 formal V1 Golden Baseline governance (pure domain).

This module turns the Phase 14.5 baseline *rules* into a concrete, reusable
registration object without touching any research behaviour.  It composes the
Phase 14.5 comparability primitives (:mod:`app.domain.benchmark_comparability`)
rather than reinventing them:

* :class:`BenchmarkBaselineManifest` freezes one registered benchmark's
  identity (benchmark id/version, normalized goal, tier, plan-shape policy,
  plan template, budget profile, metric definition version and the
  experiment-critical feature flags).  It is the single source of truth for
  benchmark identity and can project itself to a
  :class:`~app.domain.benchmark_comparability.BenchmarkComparisonKey`.
* :func:`comparison_key_from_run` builds the same key from a run's persisted
  ``budget_snapshot`` so a run can be checked against the manifest it claims
  to belong to.  ``benchmark_id``/``benchmark_version``/``metric_definition_version``
  are read from the run's own ``budget_snapshot['benchmark']`` block: a run
  that never registered (the Phase 14.3 baselines) therefore yields an
  *unregistered* key that fails eligibility — exactly the behaviour the closeout
  required, with no back-filling of historical runs.
* :class:`GoldenBaselineStatus` + :func:`evaluate_golden_baseline` combine the
  manifest eligibility (all seven Phase 14.5 rules), the *count of valid,
  mutually-COMPARABLE runs*, and confirmation that the experiment-critical
  feature flags held at runtime, into the three-state verdict the spec demands:
  ``VALID`` only when registered + >= 3 comparable runs, ``PROVISIONAL`` when
  registered but fewer than three usable runs, and ``INVALID`` when the identity
  itself is not a legitimate baseline.

Pure domain logic: no database, no provider, no research behaviour.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from app.domain.benchmark_comparability import (
    METRIC_DEFINITION_VERSION,
    BenchmarkComparisonKey,
    ComparisonResult,
    compare_keys,
    evaluate_baseline_requirements,
)

#: The budget profile / tier the V1 golden baseline is fixed to.  Repeated
#: golden ordinals must all record this value to stay comparable.
GOLDEN_BASELINE_TIER: Final[str] = "standard"

#: Minimum number of mutually-comparable valid runs for a VALID baseline.
GOLDEN_BASELINE_MIN_RUNS: Final[int] = 3


class GoldenBaselineStatus(StrEnum):
    """Three-state verdict on whether a formal golden baseline is established."""

    VALID = "VALID"
    PROVISIONAL = "PROVISIONAL"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class BenchmarkBaselineManifest:
    """One registered benchmark's fixed configuration (Task A/B).

    Instantiated from ``evals/benchmarks/v1_golden_baseline.v1.json``.  The
    identity fields here are what every golden run must reproduce; transient
    values (run_id, timestamp, provider health) are intentionally absent.
    """

    benchmark_id: str
    benchmark_version: str
    normalized_goal: str
    tier: str
    plan_shape_policy: str
    plan_template_run_id: str
    budget_profile: str
    metric_definition_version: str
    evidence_aware_context_enabled: bool

    def to_comparison_key(self) -> BenchmarkComparisonKey:
        """Project the manifest identity onto the Phase 14.5 comparability key."""

        return BenchmarkComparisonKey(
            benchmark_id=self.benchmark_id,
            benchmark_version=self.benchmark_version,
            normalized_goal=self.normalized_goal,
            tier=self.tier,
            plan_template_run_id=self.plan_template_run_id,
            plan_shape_policy=self.plan_shape_policy,
            metric_definition_version=self.metric_definition_version,
            evidence_aware_context_enabled=_flag_value(self.evidence_aware_context_enabled),
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> BenchmarkBaselineManifest:
        """Build a manifest from the JSON fixture (validating presence)."""

        flags = data.get("feature_flags") or {}
        required = (
            "benchmark_id",
            "benchmark_version",
            "normalized_goal",
            "tier",
            "plan_shape_policy",
            "plan_template_run_id",
            "budget_profile",
            "metric_definition_version",
        )
        missing = [name for name in required if not str(data.get(name) or "").strip()]
        if missing:
            raise ValueError(
                "golden baseline manifest missing required fields: " + ", ".join(missing)
            )
        if "evidence_aware_context_enabled" not in flags:
            raise ValueError("golden baseline manifest missing feature_flags."
                             "evidence_aware_context_enabled")
        return cls(
            benchmark_id=str(data["benchmark_id"]).strip(),
            benchmark_version=str(data["benchmark_version"]).strip(),
            normalized_goal=" ".join(str(data["normalized_goal"]).split()),
            tier=str(data["tier"]).strip(),
            plan_shape_policy=str(data["plan_shape_policy"]).strip(),
            plan_template_run_id=str(data["plan_template_run_id"]).strip(),
            budget_profile=str(data["budget_profile"]).strip(),
            metric_definition_version=str(data["metric_definition_version"]).strip(),
            evidence_aware_context_enabled=_as_bool(flags["evidence_aware_context_enabled"]),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "normalized_goal": self.normalized_goal,
            "tier": self.tier,
            "plan_shape_policy": self.plan_shape_policy,
            "plan_template_run_id": self.plan_template_run_id,
            "budget_profile": self.budget_profile,
            "metric_definition_version": self.metric_definition_version,
            "feature_flags": {
                "evidence_aware_context_enabled": self.evidence_aware_context_enabled,
            },
        }


def comparison_key_from_run(run: Mapping[str, Any]) -> BenchmarkComparisonKey:
    """Build the comparability key from one persisted run's metadata.

    ``run`` is a plain mapping with the run's ``normalized_goal`` plus its
    ``budget_snapshot`` (JSONB).  The ``benchmark`` block written at creation
    carries the registered identity; a run created without it (every historical
    run, including the Phase 14.3 baselines) produces unknown identity fields
    and therefore fails eligibility.  Nothing here guesses or back-fills.
    """

    budget = run.get("budget_snapshot") or run.get("budget") or {}
    if not isinstance(budget, Mapping):
        budget = {}
    block = budget.get("benchmark") or {}
    if not isinstance(block, Mapping):
        block = {}
    flags = block.get("feature_flags") or {}
    flag_value: str | None = None
    if isinstance(flags, Mapping) and "evidence_aware_context_enabled" in flags:
        flag_value = _flag_value(_as_bool(flags["evidence_aware_context_enabled"]))
    else:
        raw = block.get("evidence_aware_context_enabled")
        flag_value = str(raw).strip().lower() if raw is not None and str(raw).strip() else None
    # Phase 15.1 Branch B declared experiment factor: read the *stamped* value
    # when present and leave it ``None`` when the run predates the feature.  The
    # None-vs-value distinction is what lets an A/B report record that a frozen
    # historical arm was interpreted via inferred_legacy_default rather than a
    # silently back-filled value; comparability resolves the legacy default itself.
    targeting_raw: object | None = None
    if isinstance(flags, Mapping) and "independent_source_targeting_enabled" in flags:
        targeting_raw = flags["independent_source_targeting_enabled"]
    else:
        targeting_raw = block.get("independent_source_targeting_enabled")
    targeting_value = (
        _flag_value(_as_bool(targeting_raw))
        if targeting_raw is not None and str(targeting_raw).strip()
        else None
    )
    return BenchmarkComparisonKey(
        benchmark_id=_text(block.get("benchmark_id")),
        benchmark_version=_text(block.get("benchmark_version")),
        normalized_goal=_text(run.get("normalized_goal")),
        tier=_text(budget.get("tier")),
        plan_template_run_id=_text(budget.get("plan_template_run_id")),
        plan_shape_policy=_text(block.get("plan_shape_policy")),
        metric_definition_version=_text(block.get("metric_definition_version")),
        evidence_aware_context_enabled=flag_value,
        independent_source_targeting_enabled=targeting_value,
    )


@dataclass(frozen=True, slots=True)
class GoldenBaselineEvaluation:
    """Verdict plus the reasons behind it (Task C / Q)."""

    status: GoldenBaselineStatus
    eligible: bool
    unmet_requirements: tuple[str, ...]
    valid_run_count: int
    comparable_run_pairs: int
    non_comparable_pairs: tuple[str, ...]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "eligible": self.eligible,
            "unmet_requirements": list(self.unmet_requirements),
            "valid_run_count": self.valid_run_count,
            "comparable_run_pairs": self.comparable_run_pairs,
            "non_comparable_pairs": list(self.non_comparable_pairs),
            "reasons": list(self.reasons),
        }


def evaluate_golden_baseline(
    manifest: BenchmarkBaselineManifest,
    run_keys: Sequence[BenchmarkComparisonKey],
    *,
    feature_flags_confirmed: bool,
    valid_run_count: int | None = None,
) -> GoldenBaselineEvaluation:
    """Decide VALID / PROVISIONAL / INVALID for a set of golden runs.

    ``run_keys`` are :func:`comparison_key_from_run` projections of the candidate
    golden runs.  A run is only counted when it (a) matches the manifest's
    identity and (b) is fully ``COMPARABLE`` with every other counted run.  The
    verdict then follows the spec:

    * not registered / metric version not fixed / flags unknown -> ``INVALID``;
    * registered but fewer than three usable comparable runs -> ``PROVISIONAL``;
    * registered and at least three mutually-comparable runs -> ``VALID``.
    """

    manifest_eligibility = evaluate_baseline_requirements(
        manifest.to_comparison_key(),
        feature_flags_known=feature_flags_confirmed,
    )
    reasons: list[str] = []
    if not manifest_eligibility.eligible:
        reasons.extend(
            f"manifest requirement unmet: {name}" for name in manifest_eligibility.unmet
        )

    # A run only counts if it is itself baseline-eligible and COMPARABLE with
    # the manifest identity.
    accepted: list[BenchmarkComparisonKey] = []
    for key in run_keys:
        run_eligibility = evaluate_baseline_requirements(
            key, feature_flags_known=feature_flags_confirmed
        )
        decision = compare_keys(manifest.to_comparison_key(), key)
        if run_eligibility.eligible and decision.result is ComparisonResult.COMPARABLE:
            accepted.append(key)
        else:
            reasons.append(
                "run excluded: "
                + (
                    ", ".join(decision.reasons)
                    if decision.result is not ComparisonResult.COMPARABLE
                    else "run not baseline-eligible"
                )
            )

    # All surviving runs must be mutually COMPARABLE, not just each match the
    # manifest; otherwise a silently different tier/plan could slip through.
    non_comparable: list[str] = []
    for i in range(len(accepted)):
        for j in range(i + 1, len(accepted)):
            pair = compare_keys(accepted[i], accepted[j])
            if pair.result is not ComparisonResult.COMPARABLE:
                non_comparable.append(", ".join(pair.reasons) or pair.result.value)
    comparable_run_pairs = len(accepted) * (len(accepted) - 1) // 2
    usable = accepted if not non_comparable else []

    count = valid_run_count if valid_run_count is not None else len(accepted)
    usable_count = len(usable) if not non_comparable else 0
    effective_count = min(count, usable_count) if non_comparable else count

    if not manifest_eligibility.eligible or non_comparable:
        status = GoldenBaselineStatus.INVALID
        if non_comparable:
            reasons.extend(
                f"golden runs not mutually comparable: {reason}" for reason in non_comparable
            )
    elif effective_count < GOLDEN_BASELINE_MIN_RUNS:
        status = GoldenBaselineStatus.PROVISIONAL
        reasons.append(
            f"registered and comparable but only {effective_count} usable run(s); "
            f"golden baseline VALID requires >= {GOLDEN_BASELINE_MIN_RUNS}"
        )
    else:
        status = GoldenBaselineStatus.VALID

    return GoldenBaselineEvaluation(
        status=status,
        eligible=manifest_eligibility.eligible and not non_comparable,
        unmet_requirements=manifest_eligibility.unmet,
        valid_run_count=effective_count,
        comparable_run_pairs=comparable_run_pairs,
        non_comparable_pairs=tuple(non_comparable),
        reasons=tuple(reasons),
    )


def expected_metric_version() -> str:
    """The metric definition version the current code produces."""

    return METRIC_DEFINITION_VERSION


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _flag_value(enabled: bool) -> str:
    return "true" if enabled else "false"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
