"""Phase 14.5 closeout tests: default-off flag + benchmark comparability contract.

Covers the spec cases:

1. default disabled  - Settings default is off and every deployment surface
                       (config / docker-compose / .env.example) agrees;
2. default-off path  - service-level behavior is covered in
                       test_research_context_enricher.py (byte-identical
                       provider query, zero enrichment events, no resolver
                       access), here the static wiring is asserted;
5. comparability     - COMPARABLE for identical identity keys;
6. different task    - NOT_COMPARABLE with a stable reason;
7. different tier    - PARTIALLY_COMPARABLE with a stable reason;
8. metric drift      - version mismatch / unversioned legacy require replay;
9. missing metadata  - unknown identity yields UNKNOWN, never guessed.

Plus: the report language policy (Task F) and the formal V1 baseline
eligibility rules (Task I).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from app.core.config import Settings
from app.domain.benchmark_comparability import (
    FORBIDDEN_LONGITUDINAL_TERMS,
    METRIC_DEFINITION_VERSION,
    METRIC_VERSION_UNVERSIONED,
    NEUTRAL_COMPARISON_TERMS,
    BaselineRequirement,
    BenchmarkComparisonKey,
    ComparisonResult,
    compare_keys,
    evaluate_baseline_requirements,
    language_policy,
    replay_policy,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# A fully-known identity key; per-test variants change one field at a time.
_FULL_KEY = BenchmarkComparisonKey(
    benchmark_id="v1-dev-industrial-vision-defect",
    benchmark_version="1.0.0",
    normalized_goal="研究工业视觉缺陷检测领域的发展情况。",
    tier="standard",
    plan_template_run_id="01a0b2c4-5da2-722e-8cdb-e179fd4a2d04",
    plan_shape_policy="dimensions-per-question=2",
    metric_definition_version=METRIC_DEFINITION_VERSION,
    evidence_aware_context_enabled="false",
)


def _replace_key(**overrides: str) -> BenchmarkComparisonKey:
    values = {
        "benchmark_id": _FULL_KEY.benchmark_id,
        "benchmark_version": _FULL_KEY.benchmark_version,
        "normalized_goal": _FULL_KEY.normalized_goal,
        "tier": _FULL_KEY.tier,
        "plan_template_run_id": _FULL_KEY.plan_template_run_id,
        "plan_shape_policy": _FULL_KEY.plan_shape_policy,
        "metric_definition_version": _FULL_KEY.metric_definition_version,
        "evidence_aware_context_enabled": _FULL_KEY.evidence_aware_context_enabled,
    }
    values.update(overrides)
    return BenchmarkComparisonKey(**values)


# ---------------------------------------------------------------- Case 1
# Default disabled: the V1 configuration default.


def test_settings_default_disables_main_path_enrichment(monkeypatch: pytest.MonkeyPatch) -> None:
    # No EVIDENCE_AWARE_CONTEXT_ENABLED in the environment at all.
    monkeypatch.delenv("EVIDENCE_AWARE_CONTEXT_ENABLED", raising=False)

    assert Settings(app_env="test").evidence_aware_context_enabled is False


def test_settings_explicit_true_still_enables_the_experiment_arm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Task B: the flag remains available for deliberate experiments.
    monkeypatch.setenv("EVIDENCE_AWARE_CONTEXT_ENABLED", "true")

    assert Settings(app_env="test").evidence_aware_context_enabled is True


def test_every_deployment_surface_defaults_to_false() -> None:
    compose = (REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    env_example = (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8")

    assert "EVIDENCE_AWARE_CONTEXT_ENABLED: ${EVIDENCE_AWARE_CONTEXT_ENABLED:-false}" in compose
    # Both the API and the worker service receive the same default.
    assert compose.count("EVIDENCE_AWARE_CONTEXT_ENABLED:-false}") == 2
    assert "EVIDENCE_AWARE_CONTEXT_ENABLED=false" in env_example
    assert "EVIDENCE_AWARE_CONTEXT_ENABLED=true" not in env_example


# ---------------------------------------------------------------- Case 4
# The preserved architecture: nothing in the diagnostic/feedback infrastructure
# was removed by the closeout (Task B / Task C boundaries).


def test_preserved_architecture_modules_stay_importable() -> None:
    from app.domain import (
        benchmark_comparability,
        research_context,
        research_context_enricher,
        research_context_resolver,
    )

    assert benchmark_comparability is not None
    assert research_context is not None
    assert research_context_enricher is not None
    assert research_context_resolver is not None


def test_closeout_did_not_touch_planner_or_coverage_modules() -> None:
    # Sanity: the flag is read in exactly one place in the research loop and
    # nowhere in planner/ranking/budget/provider modules.
    services = REPOSITORY_ROOT / "apps" / "api" / "app" / "services"
    hits = sorted(
        path.name
        for path in services.glob("*.py")
        if "_evidence_aware_context_enabled" in path.read_text(encoding="utf-8")
    )
    assert hits == ["research_loop.py"]


# ---------------------------------------------------------------- Case 5
# Identical identity -> COMPARABLE.


def test_identical_identity_is_comparable() -> None:
    decision = compare_keys(_FULL_KEY, _replace_key())

    assert decision.result is ComparisonResult.COMPARABLE
    assert decision.reasons == ("all identity fields match",)
    assert decision.allows_longitudinal_language()
    payload = decision.as_dict()
    assert payload["result"] == "COMPARABLE"


def test_identity_key_excludes_run_scoped_noise() -> None:
    # run_id / timestamp / transient health are not fields on the key at all.
    fields = set(BenchmarkComparisonKey.__dataclass_fields__)

    assert "run_id" not in fields
    assert "created_at" not in fields
    assert "provider_health" not in fields


# ---------------------------------------------------------------- Case 6
# Different benchmark/goal -> NOT_COMPARABLE with stable reasons.


def test_different_benchmark_id_is_not_comparable() -> None:
    decision = compare_keys(_FULL_KEY, _replace_key(benchmark_id="other-benchmark"))

    assert decision.result is ComparisonResult.NOT_COMPARABLE
    assert "benchmark_id differs" in decision.reasons
    assert not decision.allows_longitudinal_language()


def test_different_goal_and_metric_version_are_not_comparable() -> None:
    # Metric definition drift is a hard difference: scores mean different
    # things and must never be read as a longitudinal change.
    other_metric = _replace_key(metric_definition_version="coverage-v0")
    decision = compare_keys(_FULL_KEY, other_metric)
    assert decision.result is ComparisonResult.NOT_COMPARABLE
    assert "metric_definition_version differs" in decision.reasons

    other_goal = _replace_key(normalized_goal="Compare CNN, Transformer and VLMs.")
    decision = compare_keys(_FULL_KEY, other_goal)
    assert decision.result is ComparisonResult.NOT_COMPARABLE
    assert "normalized_goal differs" in decision.reasons


# ---------------------------------------------------------------- Case 7
# Soft differences -> PARTIALLY_COMPARABLE with a reason.


def test_different_tier_is_partially_comparable() -> None:
    decision = compare_keys(_FULL_KEY, _replace_key(tier="quick"))

    assert decision.result is ComparisonResult.PARTIALLY_COMPARABLE
    assert decision.reasons == ("tier differs",)
    assert not decision.allows_longitudinal_language()


def test_different_plan_template_is_partially_comparable() -> None:
    decision = compare_keys(
        _FULL_KEY,
        _replace_key(plan_template_run_id="01a08bcc-1034-76ba-92d4-12fa1a8badcb"),
    )

    assert decision.result is ComparisonResult.PARTIALLY_COMPARABLE
    assert "plan_template_run_id differs" in decision.reasons


# ---------------------------------------------------------------- Case 9
# Missing historical metadata -> UNKNOWN, never guessed.


def test_missing_benchmark_version_yields_unknown() -> None:
    legacy = _replace_key(benchmark_version="")
    decision = compare_keys(_FULL_KEY, legacy)

    assert decision.result is ComparisonResult.UNKNOWN
    assert "benchmark_version unknown on at least one side" in decision.reasons
    assert "benchmark_version" in decision.unknown_fields


def test_missing_identity_never_overrides_a_known_hard_difference() -> None:
    # A proven hard difference still decides NOT_COMPARABLE even when another
    # field is missing on one side (no speculation in either direction).
    legacy = _replace_key(benchmark_id="", metric_definition_version="coverage-v0")
    decision = compare_keys(_FULL_KEY, legacy)

    assert decision.result is ComparisonResult.NOT_COMPARABLE
    assert "metric_definition_version differs" in decision.reasons


# ---------------------------------------------------------------- Case 8
# Metric version drift -> replay-first.


def test_current_metric_version_allows_direct_stored_comparison() -> None:
    policy = replay_policy(METRIC_DEFINITION_VERSION)

    assert policy.direct_stored_comparison_allowed is True
    assert policy.replay_required is False


def test_version_mismatch_requires_replay() -> None:
    policy = replay_policy("coverage-v0")

    assert policy.direct_stored_comparison_allowed is False
    assert policy.replay_required is True
    assert "coverage-v0" in policy.reason


def test_unversioned_legacy_requires_replay_and_is_never_equal_to_current() -> None:
    for missing in (None, "", "   "):
        policy = replay_policy(missing)
        assert policy.replay_required is True
        assert policy.direct_stored_comparison_allowed is False
    assert METRIC_VERSION_UNVERSIONED != METRIC_DEFINITION_VERSION


# ---------------------------------------------------------------- Task F
# Language policy.


def test_non_comparable_reports_forbid_longitudinal_language() -> None:
    decision = compare_keys(_FULL_KEY, _replace_key(tier="quick"))
    policy = language_policy(decision)

    assert policy["longitudinal_language_allowed"] is False
    assert policy["forbidden_terms"] == list(FORBIDDEN_LONGITUDINAL_TERMS)
    assert policy["preferred_terms"] == list(NEUTRAL_COMPARISON_TERMS)


def test_comparable_reports_keep_longitudinal_language_available() -> None:
    policy = language_policy(compare_keys(_FULL_KEY, _replace_key()))

    assert policy["longitudinal_language_allowed"] is True
    assert policy["forbidden_terms"] == []


# ---------------------------------------------------------------- Task I
# Formal V1 baseline eligibility rules.


def test_unregistered_baseline_runs_are_not_eligible() -> None:
    # Exactly the three Phase 14.3 baseline runs: empty benchmark identity.
    provisional = BenchmarkComparisonKey(
        tier="standard",
        plan_shape_policy="dimensions-per-question=1",
        evidence_aware_context_enabled="false",
    )
    verdict = evaluate_baseline_requirements(provisional, feature_flags_known=True)

    assert verdict.eligible is False
    assert BaselineRequirement.REGISTERED_BENCHMARK.value in verdict.unmet
    assert BaselineRequirement.FIXED_BENCHMARK_VERSION.value in verdict.unmet
    assert BaselineRequirement.FIXED_GOAL.value in verdict.unmet
    assert BaselineRequirement.FIXED_METRIC_DEFINITION.value in verdict.unmet


def test_fully_registered_key_is_eligible() -> None:
    verdict = evaluate_baseline_requirements(_FULL_KEY, feature_flags_known=True)

    assert verdict.eligible is True
    assert verdict.unmet == ()


def test_unknown_feature_flags_block_eligibility() -> None:
    verdict = evaluate_baseline_requirements(_FULL_KEY, feature_flags_known=False)

    assert verdict.eligible is False
    assert BaselineRequirement.KNOWN_FEATURE_FLAGS.value in verdict.unmet
