"""Phase 15.1 Branch B tests: independent-source targeting + declared experiment factor.

Branch B only fires when, *after* Branch A's identity-correct closure, a claim
verification / independent-source requirement is STILL short on distinct source
owners.  These tests lock the two invariants the spec demands:

1. **Targeting metadata is generic, gated, and additive** — it carries the
   already-counted canonical source *owners* (never a role, never stitched into
   query text, never a per-question / per-domain special case) so the follow-up
   need targets a genuinely new publisher.  When the run-level experiment factor
   is off the closure-feedback payload is byte-identical to Phase 15.0.

2. **The experiment factor is a *declared experiment variable*, not baseline
   identity** — its sole difference yields ``COMPARABLE_FOR_EXPERIMENT`` (a
   controlled A/B), while any hard/soft identity difference still outranks it,
   and a frozen pre-feature run (absent factor) never becomes non-comparable, so
   the Phase 15.0 Golden Baseline ``VALID`` verdict is preserved.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

from app.domain.benchmark_comparability import (
    BenchmarkComparisonKey,
    ComparisonResult,
    compare_keys,
)
from app.domain.closure_feedback import ClosureFeedback, ClosureFeedbackReason
from app.domain.closure_feedback_dispatcher import ClosureFeedbackDispatcher
from app.domain.gap_closure import (
    GapClosureStatus,
    GapRequirement,
    GapRequirementType,
    VerificationStatus,
)
from app.domain.golden_baseline import comparison_key_from_run
from app.domain.research_context import (
    EXISTING_DOMAINS_KEY,
    EXISTING_SOURCE_OWNERS_KEY,
    MISSING_SOURCE_COUNT_KEY,
    REQUIRED_SOURCE_COUNT_KEY,
    ResearchContext,
    independent_source_eligibility,
    meaningful_context,
)
from app.domain.research_need import ResearchNeedType
from app.infrastructure.db.research_tools import (
    ResearchToolRepository,
    _independent_source_targeting_enabled,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000b15")
NOW = datetime(2026, 9, 21, tzinfo=UTC)


def _requirement(
    *,
    requirement_type: GapRequirementType = GapRequirementType.CLAIM_VERIFICATION,
    required_independent_sources: int = 2,
    closure_status: GapClosureStatus = GapClosureStatus.PARTIAL,
) -> GapRequirement:
    return GapRequirement(
        gap_id=uuid4(),
        run_id=RUN_ID,
        question_id="alpha",
        dimension_key="alpha:dim-core",
        requirement_type=requirement_type,
        criterion="generic criterion",
        required_evidence_count=2,
        required_independent_sources=required_independent_sources,
        current_evidence_count=3,
        current_independent_sources=1,
        verification_status=VerificationStatus.REQUIRED,
        closure_status=closure_status,
        created_at=NOW,
        updated_at=NOW,
        state_version=3,
    )


def _run(targeting: object) -> SimpleNamespace:
    """One run whose benchmark block records the declared experiment factor."""

    feature_flags: dict[str, object] = {"evidence_aware_context_enabled": False}
    if targeting is not None:
        feature_flags["independent_source_targeting_enabled"] = targeting
    return SimpleNamespace(
        budget_snapshot={"benchmark": {"feature_flags": feature_flags}}
    )


# ---------------------------------------------------------------------------
# Run-level experiment-factor reader
# ---------------------------------------------------------------------------


def test_targeting_flag_reads_stamped_benchmark_feature_flag() -> None:
    assert _independent_source_targeting_enabled(_run(True)) is True
    assert _independent_source_targeting_enabled(_run("true")) is True
    assert _independent_source_targeting_enabled(_run(False)) is False
    assert _independent_source_targeting_enabled(_run(None)) is False


def test_targeting_flag_absent_is_legacy_false() -> None:
    # A frozen Phase 15.0 golden run recorded no such key -> off, byte-identical.
    plain = SimpleNamespace(budget_snapshot={"benchmark": {}})
    assert _independent_source_targeting_enabled(plain) is False
    # A non-benchmark production run has no benchmark block at all.
    assert _independent_source_targeting_enabled(SimpleNamespace(budget_snapshot={})) is False
    assert _independent_source_targeting_enabled(SimpleNamespace()) is False


# ---------------------------------------------------------------------------
# _independent_source_targeting_metadata: gated, generic, shortage-only
# ---------------------------------------------------------------------------


def test_metadata_is_none_when_flag_off_even_if_short() -> None:
    metadata = ResearchToolRepository._independent_source_targeting_metadata(
        _run(None),
        requirement=_requirement(),
        observed_source_owners=("nist.gov",),
    )
    # Flag off -> Phase 15.0 payload unchanged (byte-identical).
    assert metadata is None


def test_metadata_records_owner_exclusion_for_real_shortage() -> None:
    metadata = ResearchToolRepository._independent_source_targeting_metadata(
        _run(True),
        requirement=_requirement(required_independent_sources=2),
        # Owners arriving here are already canonical source_owner_key values
        # (registrable domains, normalised upstream by source_policy); the helper
        # only de-dupes them -- it never re-derives identity from a role or text.
        observed_source_owners=("nist.gov", "nist.gov"),
    )
    assert metadata is not None
    assert metadata[EXISTING_SOURCE_OWNERS_KEY] == ["nist.gov"]
    assert metadata[REQUIRED_SOURCE_COUNT_KEY] == 2
    assert metadata[MISSING_SOURCE_COUNT_KEY] == 1


def test_metadata_skips_when_independence_already_satisfied() -> None:
    metadata = ResearchToolRepository._independent_source_targeting_metadata(
        _run(True),
        requirement=_requirement(required_independent_sources=2),
        observed_source_owners=("nist.gov", "reuters.com"),
    )
    assert metadata is None


def test_metadata_skips_closed_and_non_independence_requirements() -> None:
    closed = ResearchToolRepository._independent_source_targeting_metadata(
        _run(True),
        requirement=_requirement(closure_status=GapClosureStatus.CLOSED),
        observed_source_owners=("nist.gov",),
    )
    assert closed is None
    coverage = ResearchToolRepository._independent_source_targeting_metadata(
        _run(True),
        requirement=_requirement(
            requirement_type=GapRequirementType.DIMENSION_COVERAGE,
            required_independent_sources=0,
        ),
        observed_source_owners=(),
    )
    assert coverage is None


def test_metadata_applies_for_independent_source_requirement_type() -> None:
    metadata = ResearchToolRepository._independent_source_targeting_metadata(
        _run(True),
        requirement=_requirement(
            requirement_type=GapRequirementType.INDEPENDENT_SOURCE,
            required_independent_sources=2,
        ),
        observed_source_owners=("acme.com",),
    )
    assert metadata is not None
    assert metadata[MISSING_SOURCE_COUNT_KEY] == 1


# ---------------------------------------------------------------------------
# Task O requirement-scoped eligibility (never discards, no role/domain logic)
# ---------------------------------------------------------------------------


def test_independent_source_eligibility_is_requirement_scoped() -> None:
    existing = ("nist.gov", "reuters.com")
    # A genuinely new publisher advances *this* requirement.
    assert independent_source_eligibility("microsoft.com", existing) is True
    # A same-owner duplicate is "not independent for this requirement" only.
    assert independent_source_eligibility("reuters.com", existing) is False
    # Comparison is owner-key based and case-insensitive; unknown never counts.
    assert independent_source_eligibility("REUTERS.COM", existing) is False
    assert independent_source_eligibility("unknown", existing) is False
    assert independent_source_eligibility("", existing) is False


# ---------------------------------------------------------------------------
# Carrier: feedback targeting metadata reaches the ResearchNeed context
# ---------------------------------------------------------------------------


def _feedback(requirement: GapRequirement, metadata: dict[str, object]) -> ClosureFeedback:
    return ClosureFeedback(
        feedback_id=uuid4(),
        run_id=requirement.run_id,
        question_id=requirement.question_id,
        gap_id=requirement.gap_id,
        dimension_key=requirement.dimension_key,
        requirement_type=requirement.requirement_type.value,
        previous_status=GapClosureStatus.OPEN,
        current_status=GapClosureStatus.PARTIAL,
        closure_result=GapClosureStatus.PARTIAL,
        failure_reason=ClosureFeedbackReason.MISSING_INDEPENDENT_SOURCE,
        missing_requirement="independent_source",
        recommended_need_type=ResearchNeedType.INDEPENDENT_SOURCE,
        created_at=NOW,
        metadata=dict(metadata),
    )


def test_need_carries_owner_exclusion_context_from_feedback_metadata() -> None:
    requirement = _requirement(
        requirement_type=GapRequirementType.INDEPENDENT_SOURCE,
        required_independent_sources=2,
    )
    feedback = _feedback(
        requirement,
        {
            EXISTING_SOURCE_OWNERS_KEY: ["nist.gov"],
            EXISTING_DOMAINS_KEY: ["nist.gov"],
            REQUIRED_SOURCE_COUNT_KEY: 2,
            MISSING_SOURCE_COUNT_KEY: 1,
        },
    )
    need = ClosureFeedbackDispatcher().dispatch(feedback, requirement).research_needs[0]
    context = need.research_context
    assert context.existing_source_owners == ("nist.gov",)
    assert context.required_source_count == 2
    assert context.missing_source_count == 1
    assert context.independent_source_shortage is True
    # Round-trips through the persisted snapshot projection unchanged.
    assert ResearchContext.from_mapping(context.as_dict()).existing_source_owners == (
        "nist.gov",
    )


def test_need_without_targeting_metadata_stays_byte_identical() -> None:
    requirement = _requirement(
        requirement_type=GapRequirementType.INDEPENDENT_SOURCE,
    )
    need = (
        ClosureFeedbackDispatcher()
        .dispatch(_feedback(requirement, {}), requirement)
        .research_needs[0]
    )
    context = need.research_context
    assert context.is_empty is True
    assert context.independent_source_shortage is False
    assert meaningful_context(context) is None


# ---------------------------------------------------------------------------
# Declared experiment factor in comparability (A/B faithfulness + baseline safety)
# ---------------------------------------------------------------------------


def _golden_key(**overrides: str) -> BenchmarkComparisonKey:
    base: dict[str, str] = {
        "benchmark_id": "v1-dev-model-comparison",
        "benchmark_version": "1.0.0",
        "normalized_goal": "Compare models for defect detection.",
        "tier": "standard",
        "plan_template_run_id": "01a08bcc-1034-76ba-92d4-12fa1a8badcb",
        "plan_shape_policy": "frozen_reference_plan:questions=5,priority_one=4,priority_two=1",
        "metric_definition_version": "coverage-v1",
        "evidence_aware_context_enabled": "false",
    }
    base.update(overrides)
    return BenchmarkComparisonKey(**base)


def test_both_arms_legacy_absent_stays_fully_comparable() -> None:
    # Two frozen golden runs (neither stamped the factor) must stay COMPARABLE so
    # the Phase 15.0 VALID verdict is computed exactly as before.
    decision = compare_keys(_golden_key(), _golden_key())
    assert decision.result is ComparisonResult.COMPARABLE
    assert decision.declared_experiment_delta == ()


def test_single_experiment_factor_difference_is_comparable_for_experiment() -> None:
    baseline = _golden_key()  # no factor -> legacy false
    candidate = _golden_key(independent_source_targeting_enabled="true")
    decision = compare_keys(baseline, candidate)
    assert decision.result is ComparisonResult.COMPARABLE_FOR_EXPERIMENT
    assert decision.is_controlled_experiment() is True
    assert decision.declared_experiment_delta == (
        "independent_source_targeting_enabled: false->true",
    )
    # A controlled experiment legitimately supports a causal A/B delta but the
    # arms are NOT described as identically configured.
    assert decision.allows_longitudinal_language() is True
    assert "all identity fields match" not in decision.reasons


def test_experiment_factor_cannot_rescue_a_hard_identity_difference() -> None:
    baseline = _golden_key()
    candidate = _golden_key(
        independent_source_targeting_enabled="true",
        metric_definition_version="coverage-v2",  # a second, disqualifying change
    )
    decision = compare_keys(baseline, candidate)
    assert decision.result is ComparisonResult.NOT_COMPARABLE


def test_experiment_factor_does_not_override_soft_difference() -> None:
    baseline = _golden_key()
    candidate = _golden_key(
        independent_source_targeting_enabled="true",
        plan_template_run_id="different-template",
    )
    decision = compare_keys(baseline, candidate)
    assert decision.result is ComparisonResult.PARTIALLY_COMPARABLE


def test_comparison_key_from_run_reads_only_stamped_factor() -> None:
    stamped = {
        "normalized_goal": "g",
        "budget_snapshot": {
            "tier": "standard",
            "benchmark": {
                "benchmark_id": "v1-dev-model-comparison",
                "feature_flags": {"independent_source_targeting_enabled": True},
            },
        },
    }
    legacy = {
        "normalized_goal": "g",
        "budget_snapshot": {"tier": "standard", "benchmark": {"benchmark_id": "x"}},
    }
    assert comparison_key_from_run(stamped).independent_source_targeting_enabled == "true"
    # Absent stays None (raw) so provenance can record the inference; comparison
    # resolves the legacy default to "false" itself.
    assert comparison_key_from_run(legacy).independent_source_targeting_enabled is None
    assert comparison_key_from_run(legacy).experiment_factors()[
        "independent_source_targeting_enabled"
    ] == "false"
