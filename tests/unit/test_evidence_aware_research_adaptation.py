"""Phase 14.1 tests: evidence-aware research adaptation down the query pipeline.

The chain under test is the production one:

ClosureFeedback(+metadata) -> ResearchNeed -> SuggestedResearchAction
    -> QueryIntent -> QueryPlan -> QueryCandidate

Every assertion is driven by generic requirement/evidence state; the same
state must produce the same adaptation regardless of question identity or
research domain, and a feedback without metadata must keep the pre-14.1
behavior byte-for-byte.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from app.domain.closure_feedback import ClosureFeedback, ClosureFeedbackReason
from app.domain.closure_feedback_dispatcher import ClosureFeedbackDispatcher
from app.domain.evidence_quality_analyzer import build_feedback_analysis
from app.domain.gap_closure import (
    GapClosureStatus,
    GapRequirement,
    GapRequirementType,
    VerificationStatus,
)
from app.domain.query_candidate import QueryCandidate, QueryCandidateGenerator
from app.domain.query_candidate_validation import QueryCandidateValidator
from app.domain.query_plan_validation import QueryPlanValidator
from app.domain.research_action import SuggestedResearchAction
from app.domain.research_context import ResearchContext, apply_query_hints
from app.domain.research_need import ResearchNeed, ResearchNeedType
from app.domain.research_query_intent import (
    ResearchQueryIntent,
    ResearchQueryIntentGenerator,
)
from app.domain.research_query_plan import (
    ResearchQueryPlan,
    ResearchQueryPlanGenerator,
)

RUN_ID = UUID("00000000-0000-0000-0000-000000000707")
NOW = datetime(2026, 5, 1, tzinfo=UTC)


def _requirement(
    *,
    question_id: str = "alpha",
    dimension_key: str = "dim-core",
    requirement_type: GapRequirementType = GapRequirementType.CLAIM_VERIFICATION,
    gap_id: UUID | None = None,
) -> GapRequirement:
    return GapRequirement(
        gap_id=gap_id or uuid4(),
        run_id=RUN_ID,
        question_id=question_id,
        dimension_key=dimension_key,
        requirement_type=requirement_type,
        criterion="generic criterion",
        required_evidence_count=2,
        required_independent_sources=1,
        current_evidence_count=2,
        current_independent_sources=1,
        verification_status=VerificationStatus.REQUIRED,
        closure_status=GapClosureStatus.PARTIAL,
        created_at=NOW,
        updated_at=NOW,
        state_version=3,
    )


def _feedback(
    requirement: GapRequirement,
    metadata: dict[str, object] | None = None,
) -> ClosureFeedback:
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
        failure_reason=ClosureFeedbackReason.CLAIM_NOT_VERIFIED,
        missing_requirement="claim_verification",
        recommended_need_type=ResearchNeedType.CLAIM_VERIFICATION,
        created_at=NOW,
        metadata=dict(metadata or {}),
    )


def _chain(
    requirement: GapRequirement,
    feedback: ClosureFeedback,
) -> tuple[
    ResearchNeed,
    SuggestedResearchAction,
    ResearchQueryIntent,
    ResearchQueryPlan,
    QueryCandidate,
]:
    """Rebuild the production need -> action -> intent -> plan -> candidate chain."""

    dispatch = ClosureFeedbackDispatcher().dispatch(feedback, requirement)
    assert dispatch.created
    need = dispatch.research_needs[0]
    action = dispatch.suggested_actions[0]
    intent = ResearchQueryIntentGenerator.generate(
        action,
        requirement=requirement,
        research_context=need.research_context,
    )[0]
    plan = ResearchQueryPlanGenerator.generate(intent, requirement=requirement)[0]
    validation = QueryPlanValidator.validate(plan, requirement=requirement)[0]
    candidate = QueryCandidateGenerator.generate(
        plan,
        requirement=requirement,
        validation=validation,
        feedback=feedback,
    )[0]
    return need, action, intent, plan, candidate


# --------------------------------------------------------------------------
# 1. Evidence failure analysis -> ResearchNeed metadata
# --------------------------------------------------------------------------


def test_feedback_analysis_metadata_lands_on_the_research_need() -> None:
    requirement = _requirement()
    metadata = build_feedback_analysis(requirement)
    need = _chain(requirement, _feedback(requirement, metadata))[0]

    assert need.evidence_failure_reason == "claim_not_supported"
    assert need.missing_evidence_type == "verification_support"
    assert need.refined_need_type == "claim_verification"
    assert need.query_hints == ("benchmark", "evaluation", "paper", "experiment")
    assert need.need_type is ResearchNeedType.CLAIM_VERIFICATION
    payload = need.as_dict()
    assert payload["query_hints"] == ["benchmark", "evaluation", "paper", "experiment"]


def test_need_identity_is_stable_with_and_without_metadata() -> None:
    requirement = _requirement()
    plain = _chain(requirement, _feedback(requirement))[0]
    refined = _chain(requirement, _feedback(requirement, build_feedback_analysis(requirement)))[0]

    assert plain.need_id == refined.need_id
    assert plain.research_context.is_empty
    assert not refined.research_context.is_empty


# --------------------------------------------------------------------------
# 2 + 3. Metadata propagation: need -> intent -> plan -> candidate
# --------------------------------------------------------------------------


def test_research_context_propagates_through_the_pipeline() -> None:
    requirement = _requirement()
    metadata = build_feedback_analysis(requirement)
    need, _action, intent, plan, candidate = _chain(
        requirement, _feedback(requirement, metadata)
    )

    assert intent.research_context == need.research_context
    assert plan.research_context == need.research_context
    assert candidate.research_context == need.research_context
    assert candidate.research_context is not None
    assert candidate.research_context.query_hints == (
        "benchmark",
        "evaluation",
        "paper",
        "experiment",
    )


def test_every_stage_exports_research_context_in_as_dict() -> None:
    requirement = _requirement()
    _need, _action, intent, plan, candidate = _chain(
        requirement, _feedback(requirement, build_feedback_analysis(requirement))
    )

    for artifact in (intent, plan, candidate):
        context = artifact.as_dict()["research_context"]
        assert isinstance(context, dict)
        assert context["evidence_failure_reason"] == "claim_not_supported"
        assert context["missing_evidence_type"] == "verification_support"
        assert context["refined_need_type"] == "claim_verification"
        assert context["query_hints"] == ["benchmark", "evaluation", "paper", "experiment"]
        assert json.loads(json.dumps(artifact.as_dict()))  # JSON-safe


# --------------------------------------------------------------------------
# 4. Query adaptation + backward compatibility with the Phase 13.x pipeline
# --------------------------------------------------------------------------


def test_query_hints_are_appended_without_rewriting_the_base_query() -> None:
    requirement = _requirement(dimension_key="dim-claims")
    base_text = (
        "dim-claims claim official documentation technical report verification"
    )
    _need, _action, _intent, _plan, candidate = _chain(
        requirement, _feedback(requirement, build_feedback_analysis(requirement))
    )

    assert candidate.query_text.startswith(base_text)
    assert candidate.query_text == f"{base_text} benchmark evaluation paper experiment"


def test_hint_injection_keeps_the_candidate_validation_valid() -> None:
    requirement = _requirement()
    _need, _action, _intent, plan, candidate = _chain(
        requirement, _feedback(requirement, build_feedback_analysis(requirement))
    )
    validation = QueryPlanValidator.validate(plan, requirement=requirement)[0]

    checks = QueryCandidateValidator.validate(
        candidate,
        plan=plan,
        requirement=requirement,
    )

    assert validation.validation_status.value == "valid"
    assert checks[0].validation_status.value == "valid"


def test_feedback_without_metadata_behaves_exactly_like_phase_13_x() -> None:
    requirement = _requirement(dimension_key="dim-legacy")
    legacy = _feedback(requirement)
    need, _action, intent, plan, candidate = _chain(requirement, legacy)

    assert need.query_hints == ()
    assert (need.evidence_failure_reason, need.missing_evidence_type) == (None, None)
    assert need.research_context.is_empty
    assert intent.research_context is None
    assert plan.research_context is None
    assert candidate.research_context is None
    assert (
        candidate.query_text
        == "dim-legacy claim official documentation technical report verification"
    )
    for artifact in (intent, plan, candidate):
        assert artifact.as_dict()["research_context"] is None


def test_apply_query_hints_is_idempotent_and_empty_safe() -> None:
    base = "dim claim official documentation"
    hints = ("benchmark", "official", "", "  evaluation   ")

    adapted = apply_query_hints(base, hints)
    assert adapted == "dim claim official documentation benchmark evaluation"
    assert apply_query_hints(adapted, hints) == adapted
    assert apply_query_hints(base, ()) == base
    assert apply_query_hints(f"  {base}  ", ()) == f"  {base}  "


def test_research_context_ignores_malformed_metadata() -> None:
    context = ResearchContext.from_mapping(
        {"query_hints": "benchmark", "missing_evidence_type": 7}
    )

    assert context.is_empty
    assert ResearchContext.from_mapping({}).is_empty
    assert ResearchContext.from_mapping(None).is_empty


# --------------------------------------------------------------------------
# 5. Identity independence
# --------------------------------------------------------------------------


@pytest.mark.parametrize("question_id", ["q1", "q5", "q999"])
def test_same_evidence_failure_yields_the_same_adaptation(question_id: str) -> None:
    requirement = _requirement(question_id=question_id, dimension_key="dim-shared")
    need, _action, _intent, _plan, candidate = _chain(
        requirement, _feedback(requirement, build_feedback_analysis(requirement))
    )

    assert need.query_hints == ("benchmark", "evaluation", "paper", "experiment")
    assert candidate.query_text == (
        "dim-shared claim official documentation technical report verification "
        "benchmark evaluation paper experiment"
    )


# --------------------------------------------------------------------------
# 6. Domain independence
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dimension_key",
    [
        "industrial-quality",
        "medical-outcome",
        "finance-risk",
        "software-reliability",
    ],
)
def test_failure_hints_are_domain_independent(dimension_key: str) -> None:
    requirement = _requirement(dimension_key=dimension_key)
    feedback = _feedback(requirement, build_feedback_analysis(requirement))
    need, _action, _intent, _plan, candidate = _chain(requirement, feedback)
    context = need.research_context

    assert context.query_hints == ("benchmark", "evaluation", "paper", "experiment")
    assert context.refined_need_type == "claim_verification"
    assert candidate.query_text.endswith("benchmark evaluation paper experiment")
    assert candidate.query_text.startswith(f"{dimension_key} claim")


def test_independent_source_failure_refines_toward_third_party_material() -> None:
    requirement = _requirement(
        requirement_type=GapRequirementType.INDEPENDENT_SOURCE,
    )
    requirement = replace(
        requirement,
        verification_status=VerificationStatus.VERIFIED,
        required_independent_sources=3,
        current_independent_sources=1,
    )
    feedback = replace(
        _feedback(requirement, build_feedback_analysis(requirement)),
        recommended_need_type=ResearchNeedType.INDEPENDENT_SOURCE,
    )
    need, _action, _intent, _plan, candidate = _chain(requirement, feedback)

    assert need.evidence_failure_reason == "no_independent_validation"
    assert need.refined_need_type == "independent_validation"
    assert need.query_hints == (
        "official documentation",
        "research paper",
        "third party analysis",
    )
    assert candidate.query_text == (
        "dim-core market report official statistics third party analysis "
        "official documentation research paper"
    )


# --------------------------------------------------------------------------
# 7. No benchmark / identity coupling in the adapted pipeline
# --------------------------------------------------------------------------


def test_pipeline_has_no_benchmark_or_identity_coupling() -> None:
    domain_root = Path(__file__).resolve().parents[2] / "apps" / "api" / "app" / "domain"
    targets = (
        "research_context.py",
        "research_need.py",
        "research_query_intent.py",
        "research_query_plan.py",
        "query_candidate.py",
    )
    forbidden_patterns = (
        re.compile(r"\bq\d+\b", re.IGNORECASE),
        re.compile(r"industrial|vision[- ]?defect", re.IGNORECASE),
        re.compile(r"v1_benchmark|benchmark_suite|golden\.json", re.IGNORECASE),
    )
    for name in targets:
        source = (domain_root / name).read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            assert not pattern.search(source), f"{name} matches {pattern.pattern}"
