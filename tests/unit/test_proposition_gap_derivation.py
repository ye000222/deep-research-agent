from __future__ import annotations

from uuid import UUID

import pytest
from app.domain.atomic_proposition import (
    ClaimPropositionLink,
    DecompositionStatus,
    EvidencePropositionRelation,
    PropositionEvidenceLink,
    PropositionRelation,
)
from app.domain.contradiction_detection import (
    CounterEvidenceEligibilityInput,
    detect_proposition_conflicts,
)
from app.domain.corroboration_aggregation import (
    CorroborationEvidenceEligibilityInput,
    IndependentCorroborationAssessment,
    OwnerCorroborationStatus,
    aggregate_independent_corroboration,
)
from app.domain.proposition_evidence_direction import (
    DirectionStatus,
    PropositionEvidenceDirectionAssessment,
)
from app.domain.proposition_research_gap import (
    GapDiagnostic,
    GapDiagnosticCode,
    GapScopePolicy,
    GapSearchability,
    PropositionGapLifecycle,
    PropositionGapReasonCode,
    PropositionResearchGapType,
    ScopeExpansionPolicy,
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
    PropositionVerificationAssessment,
    PropositionVerificationStatus,
    VerificationConflictScope,
    aggregate_proposition_verification,
)
from app.domain.semantic_runtime_budget import SemanticRuntimeBudgetState
from app.domain.source_reliability_assessment import (
    ClaimInputKind,
    ClaimSourceAuthorityFit,
    ClaimSourceFitAssessment,
    ClaimTypeSignalStatus,
    SourceReliabilityAssessment,
)
from app.domain.verification_assertion import (
    ComparabilityStatus,
    ComparisonReasonCode,
    PropositionPairAssessment,
)
from app.services.proposition_gap_derivation import (
    PropositionGapDerivationService,
    PropositionGapDerivationTarget,
)

RUN = UUID("50000000-0000-0000-0000-000000000001")
CLAIM = UUID("20000000-0000-0000-0000-000000000001")
PROP = UUID("10000000-0000-0000-0000-000000000001")
OPPOSING_PROP = UUID("10000000-0000-0000-0000-000000000002")
GROUP = "a" * 64
OPPOSING_GROUP = "b" * 64


def _group(
    group_ref: str = GROUP,
    proposition_ref: UUID = PROP,
    evidence: tuple[tuple[int, str | None, EvidencePropositionRelation], ...] = (),
) -> PropositionSupportGroup:
    memberships = tuple(
        PropositionEvidenceMembership(
            evidence_ref=UUID(int=ordinal),
            provenance_links=(
                PropositionEvidenceLink(
                    proposition_id=proposition_ref,
                    evidence_id=UUID(int=ordinal),
                    claim_id=CLAIM,
                    relation=relation,
                    source_id=UUID(int=ordinal + 1000),
                    source_owner_key=owner,
                    quote_reference="not-copied-to-gap",
                    span_start=0,
                    span_end=8,
                    method="fixture",
                    run_id=RUN,
                    question_id="q1",
                    dimension_key="d1",
                ),
            ),
        )
        for ordinal, owner, relation in evidence
    )
    return PropositionSupportGroup(
        group_ref=group_ref,
        proposition_refs=(proposition_ref,),
        evidence_memberships=tuple(sorted(memberships, key=lambda item: str(item.evidence_ref))),
        grouping_status=PropositionGroupingStatus.SINGLETON,
    )


def _corroboration(group: PropositionSupportGroup) -> IndependentCorroborationAssessment:
    inputs = tuple(
        CorroborationEvidenceEligibilityInput(
            group_ref=group.group_ref,
            evidence_ref=membership.evidence_ref,
            accepted=True,
        )
        for membership in group.evidence_memberships
        if membership.provenance_links[0].relation is EvidencePropositionRelation.SUPPORTS
    )
    return aggregate_independent_corroboration(group, inputs)


def _conflicts(
    groups: tuple[PropositionSupportGroup, ...],
    *,
    contradictory: bool = False,
    not_comparable: bool = False,
):
    bindings = tuple(
        PropositionAssertionBinding(
            proposition_ref=group.proposition_refs[0],
            assertion_ref=f"assertion-{index}",
        )
        for index, group in enumerate(groups, 1)
    )
    pairs = tuple(
        PropositionPairAssessment(
            left_assertion_ref=f"assertion-{left_index}",
            right_assertion_ref=f"assertion-{right_index}",
            comparability=(
                ComparabilityStatus.NOT_COMPARABLE
                if not_comparable
                else ComparabilityStatus.COMPARABLE
            ),
            relation=(
                PropositionRelation.CONTRADICTS
                if contradictory and left_index == 1
                else PropositionRelation.UNKNOWN
                if not_comparable
                else PropositionRelation.RELATED
            ),
            relation_eligible=not not_comparable,
            reason_codes=(
                ComparisonReasonCode.DIRECT_POLARITY_CONFLICT
                if contradictory and left_index == 1
                else ComparisonReasonCode.RELATION_NOT_EVALUATED
                if not_comparable
                else ComparisonReasonCode.RELATION_SEMANTICS_INSUFFICIENT,
            ),
        )
        for left_index in range(1, len(groups) + 1)
        for right_index in range(left_index + 1, len(groups) + 1)
    )
    counter_inputs = tuple(
        CounterEvidenceEligibilityInput(
            group_ref=groups[0].group_ref,
            evidence_ref=membership.evidence_ref,
            accepted=True,
        )
        for membership in groups[0].evidence_memberships
        if membership.provenance_links[0].relation is EvidencePropositionRelation.CONTRADICTS
    )
    return detect_proposition_conflicts(
        groups, pairs, bindings, counterevidence_eligibility=counter_inputs
    )


def _assessment(
    group: PropositionSupportGroup,
    corroboration: IndependentCorroborationAssessment,
    *,
    scope_complete: bool = True,
    opposing: tuple[PropositionSupportGroup, ...] = (),
    conflicts=None,
) -> PropositionVerificationAssessment:
    groups = (group, *opposing)
    scope = VerificationConflictScope(
        target_group_ref=group.group_ref,
        completeness=(
            ConflictScopeCompleteness.COMPLETE
            if scope_complete
            else ConflictScopeCompleteness.INCOMPLETE
        ),
        scope_ref="scope-fixture" if scope_complete else None,
        candidate_group_refs=tuple(sorted(item.group_ref for item in groups)),
    )
    return aggregate_proposition_verification(
        group,
        corroboration,
        conflicts if conflicts is not None else _conflicts(groups),
        tuple(_corroboration(item) for item in opposing),
        scope,
    )


def _target(
    group: PropositionSupportGroup,
    assessment: PropositionVerificationAssessment,
    corroboration: IndependentCorroborationAssessment,
    *,
    opposing: tuple[PropositionSupportGroup, ...] = (),
    conflicts=None,
    scope_policy: ScopeExpansionPolicy = ScopeExpansionPolicy.ALLOWED,
    diagnostics: tuple[GapDiagnostic, ...] = (),
    directions: tuple[PropositionEvidenceDirectionAssessment, ...] = (),
    budget: SemanticRuntimeBudgetState | None = None,
    search_budget: bool | None = True,
    attempt_count: int = 0,
    source_reliability: tuple[SourceReliabilityAssessment, ...] = (),
) -> PropositionGapDerivationTarget:
    return PropositionGapDerivationTarget(
        run_ref=RUN,
        objective_scope_ref="research-objective-v1",
        target_group=group,
        support_groups=(group, *opposing),
        verification_assessment=assessment,
        corroboration_assessments=(corroboration, *(_corroboration(item) for item in opposing)),
        conflict_assessment=conflicts if conflicts is not None else _conflicts((group, *opposing)),
        claim_proposition_links=tuple(
            ClaimPropositionLink(
                claim_id=CLAIM,
                proposition_id=prop,
                proposition_index=index,
                decomposition_status=DecompositionStatus.SINGLE_PROPOSITION,
            )
            for index, prop in enumerate(group.proposition_refs)
        ),
        proposition_evidence_links=tuple(
            link for member in group.evidence_memberships for link in member.provenance_links
        ),
        direction_assessments=directions,
        diagnostics=diagnostics,
        budget_state=budget,
        scope_policy=GapScopePolicy(expansion=scope_policy),
        source_reliability_assessments=source_reliability,
        parent_research_priority=2,
        attempt_count=attempt_count,
        research_search_budget_available=search_budget,
    )


def _derive(target: PropositionGapDerivationTarget):
    return PropositionGapDerivationService().derive((target,))


def test_verified_has_no_same_objective_gap() -> None:
    group = _group(
        evidence=(
            (1, "publisher-a", EvidencePropositionRelation.SUPPORTS),
            (2, "publisher-b", EvidencePropositionRelation.SUPPORTS),
        )
    )
    corroboration = _corroboration(group)
    assessment = _assessment(group, corroboration)
    assert assessment.verification_status is PropositionVerificationStatus.VERIFIED
    result = _derive(_target(group, assessment, corroboration))
    assert result.gaps == ()


def test_verified_is_not_reopened_by_capability_or_budget_diagnostics() -> None:
    group = _group(
        evidence=(
            (1, "publisher-a", EvidencePropositionRelation.SUPPORTS),
            (2, "publisher-b", EvidencePropositionRelation.SUPPORTS),
        )
    )
    corroboration = _corroboration(group)
    assessment = _assessment(group, corroboration)
    diagnostics = (
        GapDiagnostic(
            code=GapDiagnosticCode.FORMATION_CAPABILITY_LIMITATION.value,
            target_group_refs=(GROUP,),
        ),
        GapDiagnostic(
            code=GapDiagnosticCode.NOT_PROCESSED_BUDGET.value,
            target_group_refs=(GROUP,),
        ),
    )
    result = _derive(
        _target(
            group,
            assessment,
            corroboration,
            diagnostics=diagnostics,
        )
    )
    assert assessment.verification_status is PropositionVerificationStatus.VERIFIED
    assert result.gaps == ()


def test_supported_single_owner_creates_searchable_independent_gap() -> None:
    group = _group(evidence=((1, "publisher-a", EvidencePropositionRelation.SUPPORTS),))
    corroboration = _corroboration(group)
    assessment = _assessment(group, corroboration)
    result = _derive(_target(group, assessment, corroboration))
    (gap,) = result.gaps
    assert assessment.verification_status is PropositionVerificationStatus.SUPPORTED
    assert gap.gap_type is PropositionResearchGapType.MISSING_INDEPENDENT_CORROBORATION
    assert gap.searchability is GapSearchability.SEARCHABLE
    assert gap.required_evidence.required_owner_independence
    assert gap.required_evidence.excluded_owner_keys == ("publisher-a",)
    assert gap.required_evidence.desired_relations
    assert "publisher-a" in gap.existing_owner_refs


@pytest.mark.parametrize(
    ("policy", "searchability", "lifecycle"),
    [
        (ScopeExpansionPolicy.ALLOWED, GapSearchability.SEARCHABLE, PropositionGapLifecycle.OPEN),
        (
            ScopeExpansionPolicy.DISALLOWED,
            GapSearchability.NON_SEARCHABLE,
            PropositionGapLifecycle.BLOCKED,
        ),
        (
            ScopeExpansionPolicy.UNKNOWN,
            GapSearchability.UNKNOWN_SEARCHABILITY,
            PropositionGapLifecycle.BLOCKED,
        ),
    ],
)
def test_supported_multi_owner_incomplete_scope_obeys_scope_policy(
    policy, searchability, lifecycle
) -> None:
    group = _group(
        evidence=(
            (1, "publisher-a", EvidencePropositionRelation.SUPPORTS),
            (2, "publisher-b", EvidencePropositionRelation.SUPPORTS),
        )
    )
    corroboration = _corroboration(group)
    assessment = _assessment(group, corroboration, scope_complete=False)
    result = _derive(_target(group, assessment, corroboration, scope_policy=policy))
    (gap,) = result.gaps
    assert gap.gap_type is PropositionResearchGapType.SCOPE_COMPLETION
    assert gap.searchability is searchability
    assert gap.lifecycle_state is lifecycle


def test_unsupported_requires_formal_no_eligible_support_and_is_not_false() -> None:
    group = _group()
    corroboration = _corroboration(group)
    assessment = _assessment(group, corroboration)
    assert assessment.verification_status is PropositionVerificationStatus.UNSUPPORTED
    result = _derive(_target(group, assessment, corroboration))
    (gap,) = result.gaps
    assert gap.gap_type is PropositionResearchGapType.MISSING_SUPPORT
    assert gap.reason_codes == (PropositionGapReasonCode.NO_ELIGIBLE_SUPPORT,)
    assert not hasattr(gap, "truth")


def test_not_processed_budget_is_non_searchable_and_not_missing_support() -> None:
    group = _group()
    corroboration = _corroboration(group)
    assessment = _assessment(group, corroboration)
    direction = PropositionEvidenceDirectionAssessment(
        assessment_ref="c" * 64,
        claim_ref=CLAIM,
        proposition_ref=PROP,
        source_assertion_ref="d" * 64,
        evidence_ref=UUID(int=90),
        source_ref="source-ref",
        snapshot_ref="e" * 64,
        grounding_usability="UNKNOWN",
        composition_eligibility="UNKNOWN",
        measurement_eligibility="UNKNOWN",
        direction_status=DirectionStatus.NOT_PROCESSED_BUDGET,
    )
    result = _derive(
        _target(
            group,
            assessment,
            corroboration,
            directions=(direction,),
            budget=SemanticRuntimeBudgetState(),
        )
    )
    assert len(result.gaps) == 1
    (gap,) = result.gaps
    assert gap.gap_type is PropositionResearchGapType.SEMANTIC_WORK_NOT_PROCESSED
    assert gap.searchability is GapSearchability.NON_SEARCHABLE
    assert gap.lifecycle_state is PropositionGapLifecycle.BLOCKED
    assert PropositionResearchGapType.MISSING_SUPPORT not in {item.gap_type for item in result.gaps}


@pytest.mark.parametrize(
    "code",
    [
        GapDiagnosticCode.UNSUPPORTED_VALUE_PROFILE,
        GapDiagnosticCode.GROUNDING_CAPABILITY_LIMITATION,
        GapDiagnosticCode.IDENTITY_CAPABILITY_LIMITATION,
    ],
)
def test_explicit_capability_limitation_is_blocked(code) -> None:
    group = _group()
    corroboration = _corroboration(group)
    assessment = _assessment(group, corroboration)
    diagnostic = GapDiagnostic(code=code.value, target_group_refs=(GROUP,))
    result = _derive(_target(group, assessment, corroboration, diagnostics=(diagnostic,)))
    (gap,) = result.gaps
    assert gap.gap_type is PropositionResearchGapType.SEMANTIC_CAPABILITY_BLOCKED
    assert gap.searchability is GapSearchability.NON_SEARCHABLE


def test_unknown_reason_or_unknown_diagnostic_version_never_creates_searchable_gap() -> None:
    group = _group(evidence=((1, "publisher-a", EvidencePropositionRelation.SUPPORTS),))
    corroboration = _corroboration(group)
    assessment = _assessment(group, corroboration)
    for diagnostic in (
        GapDiagnostic(code="FUTURE_UNRECOGNIZED_CODE"),
        GapDiagnostic(schema_version="future-diagnostic-v2", code="FORMATION_UNRESOLVED"),
    ):
        result = _derive(_target(group, assessment, corroboration, diagnostics=(diagnostic,)))
        assert result.gaps == ()
        assert result.issues
        assert all(
            item.searchability is GapSearchability.UNKNOWN_SEARCHABILITY for item in result.issues
        )


def test_unknown_verification_reason_fails_closed() -> None:
    group = _group(evidence=((1, "publisher-a", EvidencePropositionRelation.SUPPORTS),))
    corroboration = _corroboration(group)
    assessment = _assessment(group, corroboration).model_copy(
        update={"verification_status": PropositionVerificationStatus.UNKNOWN}
    )
    result = _derive(_target(group, assessment, corroboration))
    assert result.gaps == ()
    assert result.issues


def test_conflicted_generates_conflict_gap_with_both_evidence_sides() -> None:
    group = _group(
        evidence=(
            (1, "publisher-a", EvidencePropositionRelation.SUPPORTS),
            (2, "publisher-b", EvidencePropositionRelation.CONTRADICTS),
        )
    )
    corroboration = _corroboration(group)
    conflicts = _conflicts((group,))
    counter_ref = str(UUID(int=2))
    assessment = _assessment(group, corroboration, conflicts=conflicts).model_copy(
        update={
            "verification_status": PropositionVerificationStatus.CONFLICTED,
            "conflict_state": ConflictState.ACTIVE,
            "active_conflict_refs": (f"EVIDENCE:{counter_ref}",),
            "active_counterevidence_refs": (counter_ref,),
        }
    )
    result = _derive(_target(group, assessment, corroboration, conflicts=conflicts))
    (gap,) = result.gaps
    assert gap.gap_type is PropositionResearchGapType.CONFLICT_RESOLUTION
    assert gap.required_evidence.support_side_evidence_refs == (str(UUID(int=1)),)
    assert gap.required_evidence.contradict_side_evidence_refs == (counter_ref,)


def test_not_comparable_pair_creates_context_gap_not_fact_conflict() -> None:
    target = _group(GROUP, PROP, ((1, "owner-a", EvidencePropositionRelation.SUPPORTS),))
    opposing = _group(
        OPPOSING_GROUP,
        OPPOSING_PROP,
        ((2, "owner-b", EvidencePropositionRelation.SUPPORTS),),
    )
    corroboration = _corroboration(target)
    conflicts = _conflicts((target, opposing), not_comparable=True)
    assessment = _assessment(
        target, corroboration, opposing=(opposing,), conflicts=conflicts
    ).model_copy(
        update={
            "verification_status": PropositionVerificationStatus.CONFLICTED,
            "conflict_state": ConflictState.ACTIVE,
            "active_conflict_refs": (f"PROPOSITION_GROUP:{OPPOSING_GROUP}",),
            "active_conflicting_group_refs": (OPPOSING_GROUP,),
        }
    )
    result = _derive(
        _target(target, assessment, corroboration, opposing=(opposing,), conflicts=conflicts)
    )
    (gap,) = result.gaps
    assert gap.gap_type is PropositionResearchGapType.CONTEXT_DISAMBIGUATION
    assert gap.required_evidence.a2_conflict_pair_refs


def test_two_source_diversity_paths_with_same_owner_do_not_count_as_independent() -> None:
    group = _group(
        evidence=(
            (1, "same-owner", EvidencePropositionRelation.SUPPORTS),
            (2, "same-owner", EvidencePropositionRelation.SUPPORTS),
        )
    )
    corroboration = _corroboration(group)
    assert corroboration.owner_corroboration_status is OwnerCorroborationStatus.SINGLE_KNOWN_OWNER
    assessment = _assessment(group, corroboration)
    (gap,) = _derive(_target(group, assessment, corroboration)).gaps
    assert gap.gap_type is PropositionResearchGapType.MISSING_INDEPENDENT_CORROBORATION
    assert gap.required_evidence.excluded_owner_keys == ("same-owner",)


def test_searchability_is_fail_closed_when_research_budget_is_not_available() -> None:
    group = _group(evidence=((1, "publisher-a", EvidencePropositionRelation.SUPPORTS),))
    corroboration = _corroboration(group)
    assessment = _assessment(group, corroboration)
    for available, expected in (
        (False, GapSearchability.NON_SEARCHABLE),
        (None, GapSearchability.UNKNOWN_SEARCHABILITY),
    ):
        (gap,) = _derive(_target(group, assessment, corroboration, search_budget=available)).gaps
        assert gap.searchability is expected
        assert gap.lifecycle_state is PropositionGapLifecycle.BLOCKED


def test_gap_refs_deduplicate_and_ignore_assessment_hash_and_attempt() -> None:
    group = _group(evidence=((1, "publisher-a", EvidencePropositionRelation.SUPPORTS),))
    corroboration = _corroboration(group)
    assessment = _assessment(group, corroboration)
    first = _target(group, assessment, corroboration, attempt_count=0)
    second_assessment = assessment.model_copy(update={"known_limitations": ("NEW_LIMITATION",)})
    second = _target(group, second_assessment, corroboration, attempt_count=3)
    result = PropositionGapDerivationService().derive((first, first, second))
    assert len(result.gaps) == 1
    assert result.gaps[0].gap_ref == _derive(first).gaps[0].gap_ref
    assert result.gaps[0].attempt_count == 0


def test_unknown_versions_are_isolated_and_do_not_block_other_targets() -> None:
    good = _group(evidence=((1, "publisher-a", EvidencePropositionRelation.SUPPORTS),))
    bad = _group(
        "c" * 64,
        UUID("10000000-0000-0000-0000-000000000003"),
        ((3, "publisher-c", EvidencePropositionRelation.SUPPORTS),),
    )
    good_coro = _corroboration(good)
    bad_coro = _corroboration(bad)
    good_assessment = _assessment(good, good_coro)
    bad_assessment = _assessment(bad, bad_coro).model_copy(
        update={"schema_version": "future-verification-v2"}
    )
    result = PropositionGapDerivationService().derive(
        (
            _target(good, good_assessment, good_coro),
            _target(bad, bad_assessment, bad_coro),
        )
    )
    assert len(result.gaps) == 1
    assert len(result.issues) == 1
    assert result.gaps[0].target_group_ref == good.group_ref


def test_source_reliability_is_reference_only_and_does_not_change_gap_classification() -> None:
    group = _group(evidence=((1, "publisher-a", EvidencePropositionRelation.SUPPORTS),))
    corroboration = _corroboration(group)
    assessment = _assessment(group, corroboration)
    fit = ClaimSourceFitAssessment(
        claim_ref=str(CLAIM),
        input_kind=ClaimInputKind.PROPOSITION,
        assessment=ClaimSourceAuthorityFit.UNRESOLVED,
        claim_type_status=ClaimTypeSignalStatus.CLAIM_TYPE_UNKNOWN,
        basis_refs=(),
        reason_codes=("AUTHORITY_UNRESOLVED",),
    )
    source = SourceReliabilityAssessment(
        schema_version="source-reliability-assessment.v1",
        source_ref="source-ref-only",
        snapshot_ref=None,
        claim_source_fit=fit,
        dimensions=(),
        reason_codes=("LINEAGE_UNKNOWN",),
    )
    baseline = _derive(_target(group, assessment, corroboration)).gaps[0]
    enriched = _derive(
        _target(group, assessment, corroboration, source_reliability=(source,))
    ).gaps[0]
    assert enriched.gap_type is baseline.gap_type
    assert enriched.searchability is baseline.searchability
    assert enriched.source_reliability_refs
    assert "LINEAGE_UNKNOWN" in enriched.source_reliability_reason_codes
    assert not hasattr(enriched, "reliability_score")


def test_legacy_dimension_gap_can_coexist_without_question_dimension_fields() -> None:
    group = _group(evidence=((1, "publisher-a", EvidencePropositionRelation.SUPPORTS),))
    corroboration = _corroboration(group)
    assessment = _assessment(group, corroboration)
    (gap,) = _derive(_target(group, assessment, corroboration)).gaps
    assert gap.target_group_ref == GROUP
    assert not hasattr(gap, "question_id")
    assert not hasattr(gap, "dimension_key")


def test_semantic_budget_is_a_hashed_input_reference_not_an_evidence_gap() -> None:
    group = _group(evidence=((1, "publisher-a", EvidencePropositionRelation.SUPPORTS),))
    corroboration = _corroboration(group)
    assessment = _assessment(group, corroboration)
    (gap,) = _derive(
        _target(group, assessment, corroboration, budget=SemanticRuntimeBudgetState())
    ).gaps
    assert gap.budget_state_ref is not None
    assert gap.budget_state == SemanticRuntimeBudgetState().model_dump(mode="json")
