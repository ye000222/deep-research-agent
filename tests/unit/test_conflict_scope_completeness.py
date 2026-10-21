from __future__ import annotations

from uuid import UUID

import pytest
from app.domain.atomic_proposition import (
    EvidencePropositionRelation,
    PropositionEvidenceLink,
    PropositionRelation,
)
from app.domain.conflict_scope_completeness import (
    ConflictScopeCompletenessEvaluator,
    PlannedPropositionPair,
)
from app.domain.contradiction_detection import detect_proposition_conflicts
from app.domain.corroboration_aggregation import (
    CorroborationEvidenceEligibilityInput,
    aggregate_independent_corroboration,
)
from app.domain.proposition_support_group import (
    PropositionAssertionBinding,
    PropositionEvidenceMembership,
    PropositionGroupingStatus,
    PropositionSupportGroup,
)
from app.domain.proposition_verification import (
    ConflictScopeCompleteness,
    ConflictState,
    PropositionVerificationStatus,
    aggregate_proposition_verification,
)
from app.domain.verification_assertion import (
    ComparabilityStatus,
    ComparisonReasonCode,
    PropositionPairAssessment,
)

RUN = UUID("30000000-0000-0000-0000-000000000001")
PROP_A = UUID("10000000-0000-0000-0000-000000000001")
PROP_B = UUID("10000000-0000-0000-0000-000000000002")
OWNER_A = "owner-a"
OWNER_B = "owner-b"


def _group(ref: str, proposition: UUID, *, evidence_n: int = 0) -> PropositionSupportGroup:
    memberships = tuple(
        PropositionEvidenceMembership(
            evidence_ref=UUID(int=index),
            provenance_links=(
                PropositionEvidenceLink(
                    proposition_id=proposition,
                    evidence_id=UUID(int=index),
                    claim_id=UUID(int=9000 + index),
                    relation=EvidencePropositionRelation.SUPPORTS,
                    source_id=UUID(int=1000 + index),
                    source_owner_key=OWNER_A if index % 2 else OWNER_B,
                    quote_reference=f"quote-{index}",
                    span_start=0,
                    span_end=5,
                    method="fixture",
                    run_id=RUN,
                    question_id="q1",
                    dimension_key="d1",
                ),
            ),
        )
        for index in range(1, evidence_n + 1)
    )
    return PropositionSupportGroup(
        group_ref=ref,
        proposition_refs=(proposition,),
        evidence_memberships=memberships,
        grouping_status=PropositionGroupingStatus.SINGLETON,
    )


def _fixture(
    *,
    relation: PropositionRelation = PropositionRelation.RELATED,
    comparability: ComparabilityStatus = ComparabilityStatus.COMPARABLE,
    evidence: bool = False,
):
    groups = (
        _group("a" * 64, PROP_A, evidence_n=1 if evidence else 0),
        _group("b" * 64, PROP_B, evidence_n=1 if evidence else 0),
    )
    bindings = (
        PropositionAssertionBinding(proposition_ref=PROP_A, assertion_ref="assertion-a"),
        PropositionAssertionBinding(proposition_ref=PROP_B, assertion_ref="assertion-b"),
    )
    reasons = (
        (ComparisonReasonCode.EXPLICIT_SEMANTIC_ASSOCIATION,)
        if relation is PropositionRelation.RELATED
        else ()
    )
    pair = PropositionPairAssessment(
        left_assertion_ref="assertion-a",
        right_assertion_ref="assertion-b",
        comparability=comparability,
        relation=relation,
        relation_eligible=True if relation is not PropositionRelation.UNKNOWN else None,
        reason_codes=reasons,
    )
    c1 = detect_proposition_conflicts(groups, (pair,), bindings)
    planned = (
        PlannedPropositionPair(
            pair_ref="c" * 64,
            left_proposition_ref=PROP_A,
            right_proposition_ref=PROP_B,
        ),
    )
    states = {"c" * 64: "COMPLETED"}
    result = ConflictScopeCompletenessEvaluator().evaluate(
        run_ref=RUN,
        groups=groups,
        assertion_bindings=bindings,
        coverage=c1,
        planned_pairs=planned,
        pair_execution_states=states,
        pair_assessments=(pair,),
    )
    return groups, bindings, pair, c1, planned, states, result


def _corroboration(group: PropositionSupportGroup):
    return aggregate_independent_corroboration(
        group,
        tuple(
            CorroborationEvidenceEligibilityInput(
                group_ref=group.group_ref,
                evidence_ref=membership.evidence_ref,
                accepted=True,
            )
            for membership in group.evidence_memberships
        ),
    )


def test_complete_scope_and_c2_receives_producer_result() -> None:
    groups, _bindings, _pair, c1, _planned, _states, result = _fixture()
    assert result.completeness is ConflictScopeCompleteness.COMPLETE
    assert result.for_target(groups[0].group_ref).candidate_group_refs == tuple(
        sorted(group.group_ref for group in groups)
    )
    c2 = aggregate_proposition_verification(
        groups[0],
        None,
        c1,
        conflict_scope=result.for_target(groups[0].group_ref),
    )
    assert c2.verification_scope.completeness is ConflictScopeCompleteness.COMPLETE
    assert c2.verification_scope.scope_ref == result.scope_ref


def test_formally_assessed_unknown_pair_is_covered_but_not_clear() -> None:
    groups, _bindings, _pair, c1, _planned, _states, result = _fixture(
        relation=PropositionRelation.UNKNOWN,
        comparability=ComparabilityStatus.UNKNOWN,
    )
    assert result.completeness is ConflictScopeCompleteness.COMPLETE
    c2 = aggregate_proposition_verification(
        groups[0], None, c1, conflict_scope=result.for_target(groups[0].group_ref)
    )
    assert c2.conflict_state is ConflictState.UNRESOLVED
    assert c2.verification_status is PropositionVerificationStatus.UNKNOWN


def test_formally_not_comparable_pair_counts_as_assessed() -> None:
    _groups, _bindings, _pair, _c1, _planned, _states, result = _fixture(
        relation=PropositionRelation.UNKNOWN,
        comparability=ComparabilityStatus.NOT_COMPARABLE,
    )
    assert result.completeness is ConflictScopeCompleteness.COMPLETE


@pytest.mark.parametrize(
    "state",
    [
        "NOT_PROCESSED_BUDGET",
        "NOT_PROCESSED_REQUEST_SIZE",
        "FAILED_ISOLATED",
        "RESERVED",
    ],
)
def test_noncompleted_required_execution_is_incomplete(state: str) -> None:
    groups, bindings, _pair, c1, planned, states, _ = _fixture()
    states["c" * 64] = state
    result = ConflictScopeCompletenessEvaluator().evaluate(
        run_ref=RUN,
        groups=groups,
        assertion_bindings=bindings,
        coverage=c1,
        planned_pairs=planned,
        pair_execution_states=states,
        pair_assessments=(),
    )
    assert result.completeness is ConflictScopeCompleteness.INCOMPLETE


def test_missing_pair_and_candidate_plan_truncation_are_incomplete() -> None:
    groups, bindings, _pair, c1, _planned, states, _ = _fixture()
    result = ConflictScopeCompletenessEvaluator().evaluate(
        run_ref=RUN,
        groups=groups,
        assertion_bindings=bindings,
        coverage=c1,
        planned_pairs=(),
        pair_execution_states=states,
        pair_assessments=(),
    )
    assert result.completeness is ConflictScopeCompleteness.INCOMPLETE


def test_group_universe_and_coverage_mismatch_fail_closed_unknown() -> None:
    groups, bindings, _pair, c1, planned, states, _ = _fixture()
    result = ConflictScopeCompletenessEvaluator().evaluate(
        run_ref=RUN,
        groups=groups[:1],
        assertion_bindings=bindings[:1],
        coverage=c1,
        planned_pairs=planned,
        pair_execution_states=states,
        pair_assessments=(),
    )
    assert result.completeness is ConflictScopeCompleteness.UNKNOWN


def test_scope_identity_is_order_independent_and_hash_mismatch_fails_closed() -> None:
    groups, bindings, _pair, c1, planned, states, first = _fixture()
    second = ConflictScopeCompletenessEvaluator().evaluate(
        run_ref=RUN,
        groups=tuple(reversed(groups)),
        assertion_bindings=tuple(reversed(bindings)),
        coverage=c1,
        planned_pairs=tuple(reversed(planned)),
        pair_execution_states=states,
        pair_assessments=(),
    )
    assert first.scope_ref == second.scope_ref
    assert first.universe_hash == second.universe_hash
    mismatch = ConflictScopeCompletenessEvaluator().evaluate(
        run_ref=RUN,
        groups=groups,
        assertion_bindings=bindings,
        coverage=c1,
        planned_pairs=planned,
        pair_execution_states=states,
        pair_assessments=(_fixture()[2],),
        expected_coverage_hash="0" * 64,
    )
    assert mismatch.completeness is ConflictScopeCompleteness.UNKNOWN


def test_active_conflict_stays_conflicted_with_complete_scope() -> None:
    groups, bindings, pair, _c1, planned, states, _ = _fixture(
        relation=PropositionRelation.CONTRADICTS,
        evidence=True,
    )
    c1 = detect_proposition_conflicts(groups, (pair,), bindings)
    result = ConflictScopeCompletenessEvaluator().evaluate(
        run_ref=RUN,
        groups=groups,
        assertion_bindings=bindings,
        coverage=c1,
        planned_pairs=planned,
        pair_execution_states=states,
        pair_assessments=(pair,),
    )
    c2 = aggregate_proposition_verification(
        groups[0],
        _corroboration(groups[0]),
        c1,
        (_corroboration(groups[1]),),
        result.for_target(groups[0].group_ref),
    )
    assert result.completeness is ConflictScopeCompleteness.COMPLETE
    assert c2.conflict_state is ConflictState.ACTIVE
    assert c2.verification_status is PropositionVerificationStatus.CONFLICTED


def test_resolved_conflict_is_clear_only_with_complete_scope() -> None:
    groups, bindings, pair, _c1, planned, states, _ = _fixture(
        relation=PropositionRelation.RELATED,
        evidence=True,
    )
    c1 = detect_proposition_conflicts(groups, (pair,), bindings)
    result = ConflictScopeCompletenessEvaluator().evaluate(
        run_ref=RUN,
        groups=groups,
        assertion_bindings=bindings,
        coverage=c1,
        planned_pairs=planned,
        pair_execution_states=states,
        pair_assessments=(pair,),
    )
    c2 = aggregate_proposition_verification(
        groups[0],
        _corroboration(groups[0]),
        c1,
        (_corroboration(groups[1]),),
        result.for_target(groups[0].group_ref),
    )
    assert result.completeness is ConflictScopeCompleteness.COMPLETE
    assert c2.conflict_state is ConflictState.CLEAR
    assert c2.verification_status is PropositionVerificationStatus.SUPPORTED


def test_incomplete_scope_does_not_turn_resolved_pair_into_clear() -> None:
    groups, bindings, pair, c1, planned, states, _ = _fixture(
        relation=PropositionRelation.RELATED,
        evidence=True,
    )
    states.clear()
    result = ConflictScopeCompletenessEvaluator().evaluate(
        run_ref=RUN,
        groups=groups,
        assertion_bindings=bindings,
        coverage=c1,
        planned_pairs=planned,
        pair_execution_states=states,
        pair_assessments=(pair,),
    )
    c2 = aggregate_proposition_verification(
        groups[0],
        _corroboration(groups[0]),
        c1,
        (_corroboration(groups[1]),),
        result.for_target(groups[0].group_ref),
    )
    assert result.completeness is ConflictScopeCompleteness.INCOMPLETE
    assert c2.conflict_state is ConflictState.UNRESOLVED
