"""Phase 14.5 benchmark comparability contract (closeout / baseline governance).

The Historical Baseline Regression Audit proved that the largest source of
misdiagnosis in this project was comparing coverage numbers between runs that
do not share a benchmark identity, goal, plan shape, or metric definition
(e.g. a 0.85 survey-shaped Chinese task vs a 0.45 comparative English task,
and 9-10 era stored scores of 1.0/0.6429 that replay to 0.5714/0.4286 under
today's rules).  This module freezes the lessons as a pure, reusable contract:

* :class:`BenchmarkComparisonKey` — the identity fields that make two runs
  comparable.  Deliberately excludes run_id, timestamps, transient provider
  health and request-scoped randomness, none of which change what a score
  means.
* :func:`compare_keys` — classifies any pair as ``COMPARABLE`` /
  ``PARTIALLY_COMPARABLE`` / ``NOT_COMPARABLE`` / ``UNKNOWN`` with stable,
  human-readable reasons.  Missing identity is ``UNKNOWN``; it is never
  guessed into a stronger verdict.
* :data:`FORBIDDEN_LONGITUDINAL_TERMS` / :func:`language_policy` — when a
  comparison is not ``COMPARABLE`` a report must not use causal/longitudinal
  words ("regression", "improvement", ...); the neutral alternatives are
  spelled out here so every analyzer renders the same vocabulary.
* :data:`METRIC_DEFINITION_VERSION` — the stable identifier for the coverage
  / required_sources / evidence-acceptance semantics currently in force
  ("coverage-v1").  Runs persisted before versioning existed carry
  :data:`METRIC_VERSION_UNVERSIONED` and must be treated as unknown semantics.
* :func:`replay_policy` — replay-first rule: whenever the stored metric
  versions differ or either side is unknown, direct stored-value comparison
  is forbidden and the historical run must be re-derived with the current
  definition instead.
* :class:`BaselineRequirement` / :func:`evaluate_baseline_requirements` — the
  checklist a run must satisfy before it may be adopted as the formal V1
  regression baseline (Task I rule; the evaluation itself lives here so CLI /
  web / report callers share one verdict).

Pure domain logic: no database, no provider, no research behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

# ---------------------------------------------------------------------------
# Metric definition versioning (Task G)
# ---------------------------------------------------------------------------

#: Stable identifier for the metric semantics in force since the Phase 14.4
#: supplemental audit.  It bundles three coupled rule sets that changed
#: together during 2026-09: (a) coverage = priority-weighted mean over plan
#: dimensions; (b) the independent-owner rule (two distinct ``source_owner_key``
#: owners when the criterion matches an independent-source marker or the
#: dimension carries a high-risk claim); (c) the current evidence-acceptance
#: vocabulary (role / relevance / entailment / quote-location checks).
#: Bump this revision whenever any of those semantics change; historical runs
#: replayed under the new rules must not silently claim the old version.
METRIC_DEFINITION_VERSION: Final[str] = "coverage-v1"

#: Sentinel for runs persisted before metric versioning existed.  Their stored
#: scores carry unknown semantics (the audit measured up to +0.2501 coverage
#: inflation on 9-10 era runs), so "unversioned" never equals any version.
METRIC_VERSION_UNVERSIONED: Final[str] = "unversioned-legacy"

#: Value any key field may hold when the run did not persist it.
UNKNOWN_VALUE: Final[str] = "unknown"


class ComparisonResult(StrEnum):
    """Verdict for a pair of :class:`BenchmarkComparisonKey` values."""

    COMPARABLE = "COMPARABLE"
    #: All baseline-identity fields match and the *only* difference is a
    #: declared experiment factor (Phase 15.1 Branch B).  The pair is valid for
    #: a controlled A/B but the two arms must never be described as "identically
    #: configured"; the differing factor is surfaced as the experimental delta.
    COMPARABLE_FOR_EXPERIMENT = "COMPARABLE_FOR_EXPERIMENT"
    PARTIALLY_COMPARABLE = "PARTIALLY_COMPARABLE"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    UNKNOWN = "UNKNOWN"


class BaselineRequirement(StrEnum):
    """Conditions a run must satisfy to become the formal V1 baseline (Task I)."""

    REGISTERED_BENCHMARK = "registered_benchmark"
    FIXED_BENCHMARK_VERSION = "fixed_benchmark_version"
    FIXED_GOAL = "fixed_goal"
    FIXED_TIER = "fixed_tier"
    FIXED_PLAN_POLICY = "fixed_plan_policy"
    FIXED_METRIC_DEFINITION = "fixed_metric_definition"
    KNOWN_FEATURE_FLAGS = "known_feature_flags"


#: Comparison fields whose difference changes what the task *is* or how its
#: score is produced.  Any hard difference is NOT_COMPARABLE.
_HARD_FIELDS: Final[tuple[str, ...]] = (
    "benchmark_id",
    "normalized_goal",
    "metric_definition_version",
    "plan_shape_policy",
    "evidence_aware_context_enabled",
)

#: Comparison fields that keep the score interpretable but weaken the
#: like-for-like claim (volume / provenance differences).
_SOFT_FIELDS: Final[tuple[str, ...]] = (
    "benchmark_version",
    "tier",
    "plan_template_run_id",
)

#: Phase 15.1 Branch B — declared experiment variables.  These are carried for
#: provenance and A/B eligibility but are deliberately *not* baseline-identity
#: fields: their difference is the reason a controlled experiment exists, so a
#: lone difference here must not make two runs NOT_COMPARABLE, and the mere
#: absence of the factor on a frozen historical run must not retroactively
#: invalidate that baseline (a historical run predates the feature).  They are
#: kept out of :data:`_HARD_FIELDS` / :data:`_SOFT_FIELDS` so the Phase 15.0
#: golden-baseline VALID verdict is computed exactly as before.
EXPERIMENT_FACTOR_FIELDS: Final[tuple[str, ...]] = (
    "independent_source_targeting_enabled",
)

#: The behaviour a frozen historical run exhibited for an experiment factor that
#: did not yet exist when the run was created.  Absence is read as this value in
#: an A/B comparison, and the inference is recorded as provenance rather than
#: silently back-filled into the historical run's metadata.
INFERRED_LEGACY_DEFAULTS: Final[Mapping[str, str]] = {
    "independent_source_targeting_enabled": "false",
}


@dataclass(frozen=True, slots=True)
class BenchmarkComparisonKey:
    """Identity of one run for comparability purposes.

    Every field is an optional string; ``None`` and the empty string both
    mean "not persisted" and normalize to :data:`UNKNOWN_VALUE`.  run_id,
    timestamps and transient provider health are intentionally absent.
    """

    benchmark_id: str | None = None
    benchmark_version: str | None = None
    normalized_goal: str | None = None
    tier: str | None = None
    plan_template_run_id: str | None = None
    plan_shape_policy: str | None = None
    metric_definition_version: str | None = None
    evidence_aware_context_enabled: str | None = None
    #: Phase 15.1 Branch B declared experiment factor.  ``None`` means the run
    #: never recorded it (a pre-feature or legacy run); comparison treats that as
    #: :data:`INFERRED_LEGACY_DEFAULTS` while recording the inference as provenance.
    independent_source_targeting_enabled: str | None = None

    def normalized(self) -> dict[str, str]:
        """Field -> comparable value with unknowns collapsed to ``UNKNOWN``."""

        return {
            name: _normalize_field(getattr(self, name))
            for name in (*_HARD_FIELDS, *_SOFT_FIELDS)
        }

    def experiment_factors(self) -> dict[str, str]:
        """Declared experiment factors with legacy-absence resolved for comparison.

        Absent factors collapse to their :data:`INFERRED_LEGACY_DEFAULTS` value so
        an A/B can recognise a genuine baseline-vs-candidate delta, while callers
        that need to show provenance read the raw field to learn the value was
        inferred rather than stamped.
        """

        resolved: dict[str, str] = {}
        for name in EXPERIMENT_FACTOR_FIELDS:
            raw = getattr(self, name)
            text = raw.strip().lower() if raw is not None and raw.strip() else ""
            resolved[name] = text or INFERRED_LEGACY_DEFAULTS.get(name, UNKNOWN_VALUE)
        return resolved

    def is_known(self, name: str) -> bool:
        """Whether one identity field was actually persisted (not unknown)."""

        return self.normalized()[name] != UNKNOWN_VALUE

    def as_dict(self) -> dict[str, str]:
        """JSON-safe projection for reports and event payloads."""

        return self.normalized()


@dataclass(frozen=True, slots=True)
class ComparisonDecision:
    """Verdict plus the stable reasons behind it."""

    result: ComparisonResult
    reasons: tuple[str, ...] = ()
    #: Names of identity fields where at least one side is unknown.
    unknown_fields: tuple[str, ...] = ()
    #: Declared experiment factors (Phase 15.1 Branch B) that differ between the
    #: two arms, as ``name: baseline_value->candidate_value``.  Empty unless the
    #: verdict is :attr:`ComparisonResult.COMPARABLE_FOR_EXPERIMENT`.
    declared_experiment_delta: tuple[str, ...] = ()

    def allows_longitudinal_language(self) -> bool:
        """Whether the pair may speak of regression/improvement.

        True only for a fully identical-config :attr:`ComparisonResult.COMPARABLE`
        pair or a :attr:`ComparisonResult.COMPARABLE_FOR_EXPERIMENT` controlled A/B
        (where the sole difference is a declared experiment factor).
        """

        return self.result in (
            ComparisonResult.COMPARABLE,
            ComparisonResult.COMPARABLE_FOR_EXPERIMENT,
        )

    def is_controlled_experiment(self) -> bool:
        """Whether the arms differ by exactly one declared experiment factor."""

        return self.result is ComparisonResult.COMPARABLE_FOR_EXPERIMENT

    def as_dict(self) -> dict[str, object]:
        return {
            "result": self.result.value,
            "reasons": list(self.reasons),
            "unknown_fields": list(self.unknown_fields),
            "declared_experiment_delta": list(self.declared_experiment_delta),
            "allows_longitudinal_language": self.allows_longitudinal_language(),
            "is_controlled_experiment": self.is_controlled_experiment(),
        }


def compare_keys(
    left: BenchmarkComparisonKey,
    right: BenchmarkComparisonKey,
) -> ComparisonDecision:
    """Classify one pair of runs without ever blocking inspection."""

    left_values = left.normalized()
    right_values = right.normalized()
    unknown_fields = tuple(
        name
        for name in (*_HARD_FIELDS, *_SOFT_FIELDS)
        if left_values[name] == UNKNOWN_VALUE or right_values[name] == UNKNOWN_VALUE
    )
    hard_diffs = tuple(
        name for name in _HARD_FIELDS if _differs(left_values, right_values, name)
    )
    soft_diffs = tuple(
        name for name in _SOFT_FIELDS if _differs(left_values, right_values, name)
    )

    reasons: list[str] = []
    if hard_diffs:
        # A known hard difference outranks unknowns: the pair is settled.
        result = ComparisonResult.NOT_COMPARABLE
        reasons.extend(f"{name} differs" for name in hard_diffs)
    elif unknown_fields:
        # Missing identity must not be guessed into COMPARABLE or a
        # definite NOT_COMPARABLE verdict (audit discipline: UNKNOWN).
        result = ComparisonResult.UNKNOWN
        reasons.extend(f"{name} unknown on at least one side" for name in unknown_fields)
        if soft_diffs:
            reasons.extend(f"{name} differs" for name in soft_diffs)
    elif soft_diffs:
        result = ComparisonResult.PARTIALLY_COMPARABLE
        reasons.extend(f"{name} differs" for name in soft_diffs)
    else:
        result = ComparisonResult.COMPARABLE
        reasons.append("all identity fields match")

    # Phase 15.1 Branch B: only a baseline-identical pair (result COMPARABLE)
    # may be re-read as a controlled experiment.  Any hard / soft / unknown
    # difference outranks the experiment factor — an A/B whose arms differ in
    # more than the declared variable is NOT rescue-comparable.
    experiment_delta: tuple[str, ...] = ()
    if result is ComparisonResult.COMPARABLE:
        experiment_delta = _experiment_deltas(left, right)
        if len(experiment_delta) == 1:
            result = ComparisonResult.COMPARABLE_FOR_EXPERIMENT
            reasons = [f"controlled experimental delta: {experiment_delta[0]}"]
        elif len(experiment_delta) > 1:
            reasons.extend(
                f"declared experiment factor {delta} differs" for delta in experiment_delta
            )

    return ComparisonDecision(
        result=result,
        reasons=tuple(reasons),
        unknown_fields=unknown_fields,
        declared_experiment_delta=experiment_delta,
    )


def _experiment_deltas(
    left: BenchmarkComparisonKey,
    right: BenchmarkComparisonKey,
) -> tuple[str, ...]:
    """``name: left_value->right_value`` for each differing experiment factor."""

    left_factors = left.experiment_factors()
    right_factors = right.experiment_factors()
    return tuple(
        f"{name}: {left_factors[name]}->{right_factors[name]}"
        for name in EXPERIMENT_FACTOR_FIELDS
        if left_factors[name] != right_factors[name]
    )


# ---------------------------------------------------------------------------
# Report language policy (Task F)
# ---------------------------------------------------------------------------

#: Words that assert a causal/longitudinal claim; forbidden unless the
#: comparison is COMPARABLE.
FORBIDDEN_LONGITUDINAL_TERMS: Final[tuple[str, ...]] = (
    "regression",
    "improvement",
    "quality increased",
    "quality decreased",
)

#: Neutral replacements for non-comparable comparisons.
NEUTRAL_COMPARISON_TERMS: Final[tuple[str, ...]] = (
    "cross-run difference",
    "exploratory comparison",
    "not regression-safe",
)


def language_policy(decision: ComparisonDecision) -> dict[str, object]:
    """Machine-readable guidance for rendering one comparison."""

    if decision.allows_longitudinal_language():
        return {
            "longitudinal_language_allowed": True,
            "forbidden_terms": [],
            "preferred_terms": [],
        }
    return {
        "longitudinal_language_allowed": False,
        "forbidden_terms": list(FORBIDDEN_LONGITUDINAL_TERMS),
        "preferred_terms": list(NEUTRAL_COMPARISON_TERMS),
    }


# ---------------------------------------------------------------------------
# Replay-first historical policy (Task H)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplayPolicy:
    """How a report must treat a stored historical score."""

    direct_stored_comparison_allowed: bool
    replay_required: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "direct_stored_comparison_allowed": self.direct_stored_comparison_allowed,
            "replay_required": self.replay_required,
            "reason": self.reason,
        }


def normalize_metric_version(value: str | None) -> str:
    """Collapse absent / pre-versioning stored versions to one sentinel."""

    if value is None or not value.strip():
        return METRIC_VERSION_UNVERSIONED
    return value.strip()


def replay_policy(
    stored_metric_definition_version: str | None,
    *,
    current_metric_definition_version: str = METRIC_DEFINITION_VERSION,
) -> ReplayPolicy:
    """Replay-first rule for historical vs current scores.

    Direct stored-value comparison is allowed only when the run demonstrably
    recorded the *current* metric semantics.  Anything else (a different
    version, or no version at all) must be re-derived with the current
    definition before any number is shown; the audit measured old-era stored
    values up to 0.25 coverage points high, so silence is never an option.
    """

    stored = normalize_metric_version(stored_metric_definition_version)
    if stored == current_metric_definition_version:
        return ReplayPolicy(
            direct_stored_comparison_allowed=True,
            replay_required=False,
            reason=f"stored scores already carry {current_metric_definition_version} semantics",
        )
    if stored == METRIC_VERSION_UNVERSIONED:
        return ReplayPolicy(
            direct_stored_comparison_allowed=False,
            replay_required=True,
            reason="stored metric definition version is unknown; compare only after "
            "replaying the run with the current definition",
        )
    return ReplayPolicy(
        direct_stored_comparison_allowed=False,
        replay_required=True,
        reason=f"stored metric definition version {stored!r} != current "
        f"{current_metric_definition_version!r}; show stored and current-definition "
        "replay values side by side",
    )


# ---------------------------------------------------------------------------
# Formal V1 baseline rules (Task I)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BaselineEligibility:
    """Verdict on adopting one run as the formal V1 regression baseline."""

    eligible: bool
    unmet: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, object]:
        return {"eligible": self.eligible, "unmet": list(self.unmet)}


def evaluate_baseline_requirements(
    key: BenchmarkComparisonKey,
    *,
    feature_flags_known: bool,
) -> BaselineEligibility:
    """Check the Phase 14.5 rules without running anything new.

    A baseline must belong to a registered benchmark with a fixed version, a
    fixed goal, tier and plan-shape policy, a known metric definition, and
    known experiment-critical feature flags.  The three Phase 14.3 baseline
    runs fail ``registered_benchmark`` (empty benchmark id), which is exactly
    why they stay a temporary baseline instead of becoming the golden one.
    """

    unmet: list[str] = []
    if not key.is_known("benchmark_id"):
        unmet.append(BaselineRequirement.REGISTERED_BENCHMARK.value)
    if not key.is_known("benchmark_version"):
        unmet.append(BaselineRequirement.FIXED_BENCHMARK_VERSION.value)
    if not key.is_known("normalized_goal"):
        unmet.append(BaselineRequirement.FIXED_GOAL.value)
    if not key.is_known("tier"):
        unmet.append(BaselineRequirement.FIXED_TIER.value)
    if not key.is_known("plan_shape_policy"):
        unmet.append(BaselineRequirement.FIXED_PLAN_POLICY.value)
    if normalize_metric_version(key.metric_definition_version) == METRIC_VERSION_UNVERSIONED:
        unmet.append(BaselineRequirement.FIXED_METRIC_DEFINITION.value)
    if not feature_flags_known or not key.is_known("evidence_aware_context_enabled"):
        unmet.append(BaselineRequirement.KNOWN_FEATURE_FLAGS.value)
    return BaselineEligibility(eligible=not unmet, unmet=tuple(unmet))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _normalize_field(value: str | None) -> str:
    if value is None:
        return UNKNOWN_VALUE
    text = value.strip()
    return text if text else UNKNOWN_VALUE


def _differs(
    left: Mapping[str, str],
    right: Mapping[str, str],
    name: str,
) -> bool:
    """Known-and-different check; unknown-vs-known is not a difference."""

    a, b = left[name], right[name]
    if a == UNKNOWN_VALUE or b == UNKNOWN_VALUE:
        return False
    return a != b
