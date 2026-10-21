from __future__ import annotations

from uuid import UUID

import app.domain.proposition_evidence_direction as direction
import pytest
from app.domain.atomic_proposition import (
    AtomicProposition,
    ClaimPropositionLink,
    DecompositionMetadata,
    DecompositionStatus,
    EvidencePropositionRelation,
    PropositionPolarity,
    PropositionValue,
    PropositionValueKind,
)
from app.domain.claim_architecture import (
    CanonicalClaim,
    CanonicalClaimType,
    ClaimEvidenceLink,
    ClaimEvidenceRelation,
)
from app.domain.comparability_evaluator import ComparabilityEvaluator
from app.domain.proposition_evidence_direction import (
    DirectionReason,
    DirectionStatus,
    EvidenceDirectionScope,
    MeasurementAdmissibilityBinding,
    PropositionDirectionTarget,
    PropositionMentionBasis,
    SourceAssertionEvidenceBinding,
    evaluate_proposition_evidence_directions,
)
from app.domain.proposition_support_group import (
    PropositionAssertionBinding,
    group_propositions,
)
from app.domain.relation_evaluator import ExplicitSemanticAssociation, RelationEvaluator
from app.domain.semantic_identity_alignment import (
    SemanticAlignmentStatus,
    SemanticIdentityAlignmentFact,
    SemanticIdentityAlignmentReasonCode,
    SemanticIdentityFieldRole,
)
from app.domain.semantic_runtime_budget import SemanticRuntimeBudget
from app.domain.source_atomic_assertion import (
    SourceAssertionEligibility,
    SourceAssertionField,
    SourceAtomicAssertion,
    SourceFieldGrounding,
    SourceGroundingState,
    SourceSemanticTerm,
)
from app.domain.source_context_extraction_v3_3 import (
    MeasurementAdmissibilityV33,
    MeasurementIdentityTypeV33,
    MeasurementSemanticsV33,
    SourceContextV33,
)
from app.domain.truth_condition_context import (
    ContextOrigin,
    ContextScope,
    ContextScopeKind,
    ContextType,
    TruthConditionContext,
)
from app.domain.verification_assertion import (
    AssertionGroundingUsability,
    ComparisonEligibility,
    SemanticIdentity,
    SemanticIdentityBasis,
    SemanticIdentityStatus,
    VerificationAssertion,
)

CLAIM_ID = UUID("10000000-0000-0000-0000-000000000001")
EVIDENCE_ID = UUID("20000000-0000-0000-0000-000000000001")
SOURCE_ID = UUID("30000000-0000-0000-0000-000000000001")
RUN_ID = UUID("40000000-0000-0000-0000-000000000001")
PROP_ID = UUID("50000000-0000-0000-0000-000000000001")
SNAPSHOT = "a" * 64
SPAN = "b" * 64


def _budget(**changes: int) -> SemanticRuntimeBudget:
    values = {
        "max_ap_decomposition_calls": 0,
        "max_source_semantic_extraction_calls": 0,
        "max_identity_resolution_calls": 0,
        "max_evidence_inputs": 10,
        "max_propositions_per_evidence": 10,
        "max_source_assertions_per_evidence": 10,
        "max_candidate_pairs": 100,
        "max_direction_assessments": 100,
    }
    values.update(changes)
    return SemanticRuntimeBudget(**values)


def _identity(key: str = "entity:x", *, resolved: bool = True) -> SemanticIdentity:
    if not resolved:
        return SemanticIdentity.unresolved()
    return SemanticIdentity(
        status=SemanticIdentityStatus.RESOLVED,
        key=key,
        namespace="test-namespace",
        normalization_version="test-normalization-v1",
        basis=SemanticIdentityBasis.HUMAN_CURATED,
        provenance_ref="test-fixture",
    )


def _source_assertion(
    ref: str = "1" * 64,
    *,
    value: bool = True,
    eligibility: SourceAssertionEligibility = SourceAssertionEligibility.INDEPENDENTLY_ASSERTED,
    subject_resolved: bool = True,
    grounding: SourceGroundingState = SourceGroundingState.VALID,
    composition_ref: str | None = None,
    context_refs: tuple[str, ...] = (),
) -> SourceAtomicAssertion:
    refs = ("evidence-span-1",)
    field_names = [
        SourceAssertionField.SUBJECT,
        SourceAssertionField.PREDICATE,
        SourceAssertionField.VALUE,
        SourceAssertionField.POLARITY,
    ]
    if context_refs:
        field_names.append(SourceAssertionField.CONTEXT)
    fields = tuple(
        SourceFieldGrounding(
            field=field,
            evidence_refs=refs,
            textual_grounding=grounding,
            semantic_support=grounding,
            structural_grounding=grounding,
        )
        for field in sorted(
            field_names,
            key=lambda item: item.value,
        )
    )
    return SourceAtomicAssertion(
        assertion_ref=ref,
        source_ref="source://fixture",
        snapshot_ref=SNAPSHOT,
        canonical_evidence_span_ref=SPAN,
        source_assertion_anchor_refs=("anchor-1",),
        evidence_span_refs=refs,
        source_evidence_refs=refs,
        subject=SourceSemanticTerm(
            source_representation="The system",
            semantic_identity=_identity(resolved=subject_resolved),
        ),
        predicate=SourceSemanticTerm(
            source_representation="is active", semantic_identity=_identity("predicate:active")
        ),
        value=PropositionValue(kind=PropositionValueKind.BOOLEAN, boolean=value),
        polarity=PropositionPolarity.ASSERTED,
        truth_condition_context_refs=context_refs,
        composition_ref=composition_ref,
        verification_eligibility=eligibility,
        field_grounding=fields,
    )


def _scope(
    assertions: tuple[SourceAtomicAssertion, ...] | None = None,
    *,
    target_ids: tuple[UUID, ...] = (PROP_ID,),
    target_grounding: AssertionGroundingUsability = AssertionGroundingUsability.SUFFICIENT,
    target_composition: ComparisonEligibility = ComparisonEligibility.ELIGIBLE,
    mention: bool = False,
    explicit_association: bool = False,
    target_value: PropositionValue | None = None,
    claim_id: UUID = CLAIM_ID,
    evidence_id: UUID = EVIDENCE_ID,
    truth_condition_contexts: tuple[TruthConditionContext, ...] = (),
    source_contexts: tuple[SourceContextV33, ...] = (),
    measurement_admissibility: tuple[MeasurementAdmissibilityBinding, ...] = (),
) -> EvidenceDirectionScope:
    claim = CanonicalClaim.from_text(
        claim_id=claim_id,
        original_text="The system is active.",
        run_id=RUN_ID,
        question_id="q1",
        dimension_key="d1",
        claim_type=CanonicalClaimType.FACTUAL,
    )
    claim_evidence = ClaimEvidenceLink(
        claim_id=claim_id,
        evidence_id=evidence_id,
        relation=ClaimEvidenceRelation.SUPPORTS,
        source_id=SOURCE_ID,
        source_owner_key="owner-1",
        quote_reference="quote-ref-1",
    )
    targets = []
    for index, proposition_id in enumerate(target_ids):
        typed_target_value = target_value or PropositionValue(
            kind=PropositionValueKind.BOOLEAN, boolean=True
        )
        proposition = AtomicProposition(
            proposition_id=proposition_id,
            schema_version="atomic-proposition-schema-v3",
            original_text="The system is active.",
            subject="The system",
            predicate="is active",
            value=typed_target_value,
            polarity=PropositionPolarity.ASSERTED,
            decomposition=DecompositionMetadata(
                status=DecompositionStatus.SINGLE_PROPOSITION,
                schema_version="atomic-proposition-schema-v3",
                proposition_index=index,
            ),
        )
        target_assertion = VerificationAssertion(
            assertion_ref=f"target-{proposition_id}",
            source_proposition_id=proposition_id,
            subject="The system",
            subject_identity=_identity(),
            predicate="is active",
            predicate_identity=_identity("predicate:active"),
            typed_value=proposition.value,
            polarity=proposition.polarity,
            composition_eligibility=target_composition,
            grounding_usability=target_grounding,
        )
        targets.append(
            PropositionDirectionTarget(
                membership=ClaimPropositionLink(
                    claim_id=claim_id,
                    proposition_id=proposition_id,
                    proposition_index=index,
                    decomposition_status=DecompositionStatus.SINGLE_PROPOSITION,
                ),
                proposition=proposition,
                assertion=target_assertion,
            )
        )
    source_assertions = assertions or (_source_assertion(),)
    bindings = tuple(
        SourceAssertionEvidenceBinding(
            source_assertion_ref=item.assertion_ref,
            evidence_id=evidence_id,
            source_id=SOURCE_ID,
            source_ref=item.source_ref,
            snapshot_ref=item.snapshot_ref,
            canonical_evidence_span_ref=item.canonical_evidence_span_ref,
            source_owner_key="owner-1",
        )
        for item in source_assertions
    )
    mention_associations = ()
    explicit_semantic_associations = ()
    if mention or explicit_association:
        item = source_assertions[0]
        association = ExplicitSemanticAssociation(
            left_assertion_ref=item.assertion_ref,
            right_assertion_ref=f"target-{target_ids[0]}",
            established=True,
            provenance_ref="formal-mention-provenance",
        )
        explicit_semantic_associations = (association,)
    if mention:
        item = source_assertions[0]
        mention_associations = (
            PropositionMentionBasis(
                proposition_ref=target_ids[0],
                evidence_ref=evidence_id,
                source_assertion_ref=item.assertion_ref,
                basis_ref="mention-basis-1",
                association=association,
            ),
        )
    return EvidenceDirectionScope(
        claim=claim,
        claim_evidence=claim_evidence,
        targets=tuple(targets),
        source_assertions=source_assertions,
        source_bindings=bindings,
        truth_condition_contexts=truth_condition_contexts,
        source_contexts=source_contexts,
        measurement_admissibility=measurement_admissibility,
        explicit_semantic_associations=explicit_semantic_associations,
        mention_associations=mention_associations,
    )


def _measurement_scope(
    admissibility: MeasurementAdmissibilityV33,
) -> EvidenceDirectionScope:
    source = _source_assertion(context_refs=("measurement-context-1",))
    context = TruthConditionContext(
        context_ref="measurement-context-1",
        semantic_signature="measurement-signature-v1",
        context_type=ContextType.MEASUREMENT,
        semantic_value={},
        source_span="measurement-span-1",
        materiality="MATERIAL",
        scope=ContextScope(
            kind=ContextScopeKind.PROPOSITION,
            target_refs=(source.assertion_ref,),
        ),
        origin=ContextOrigin.SOURCE,
    )
    source_context = SourceContextV33(
        context_ref="measurement-context-1",
        context_type=ContextType.MEASUREMENT,
        semantic_value=MeasurementSemanticsV33(
            measurement_identity={
                "name": "latency",
                "identity_type": MeasurementIdentityTypeV33.NAMED_METRIC,
                "source_span": {"evidence_ref": "evidence-span-1"},
            }
        ),
    )
    return _scope(
        (source,),
        truth_condition_contexts=(context,),
        source_contexts=(source_context,),
        measurement_admissibility=(
            MeasurementAdmissibilityBinding(
                context_ref="measurement-context-1",
                admissibility=admissibility,
            ),
        ),
    )


def test_same_relation_uses_frozen_a2_a3_and_creates_existing_support_link(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a2_calls: list[tuple[str, str]] = []
    a3_calls: list[tuple[str, str]] = []
    original_a2 = ComparabilityEvaluator.evaluate
    original_a3 = RelationEvaluator.evaluate

    def a2(left: VerificationAssertion, right: VerificationAssertion):
        a2_calls.append((left.assertion_ref, right.assertion_ref))
        return original_a2(left, right)

    def a3(
        left: VerificationAssertion,
        right: VerificationAssertion,
        comparability,
        *,
        association=None,
    ):
        a3_calls.append((left.assertion_ref, right.assertion_ref))
        return original_a3(left, right, comparability, association=association)

    monkeypatch.setattr(direction.ComparabilityEvaluator, "evaluate", a2)
    monkeypatch.setattr(direction.RelationEvaluator, "evaluate", a3)
    result = evaluate_proposition_evidence_directions((_scope(),), budget=_budget())
    assert len(a2_calls) == len(a3_calls) == 1
    assert result.assessments[0].direction_status is DirectionStatus.RESOLVED_SUPPORT
    assert result.links[0].relation is EvidencePropositionRelation.SUPPORTS
    assert result.links[0].evidence_id == EVIDENCE_ID


def test_unresolved_identity_is_not_repaired_from_equal_surface_text() -> None:
    result = evaluate_proposition_evidence_directions(
        (_scope((_source_assertion(subject_resolved=False),)),), budget=_budget()
    )
    assert result.assessments[0].direction_status is DirectionStatus.UNRESOLVED
    assert DirectionReason.IDENTITY_UNRESOLVED in result.assessments[0].reason_codes
    assert result.links == ()


def test_grounding_composition_and_target_eligibility_are_required() -> None:
    ungrounded = evaluate_proposition_evidence_directions(
        (_scope((_source_assertion(grounding=SourceGroundingState.UNKNOWN),)),), budget=_budget()
    )
    ineligible = evaluate_proposition_evidence_directions(
        (
            _scope(
                (
                    _source_assertion(
                        eligibility=SourceAssertionEligibility.NOT_INDEPENDENTLY_ASSERTED
                    ),
                )
            ),
        ),
        budget=_budget(),
    )
    target_blocked = evaluate_proposition_evidence_directions(
        (_scope(target_composition=ComparisonEligibility.UNKNOWN),), budget=_budget()
    )
    assert ungrounded.links == ineligible.links == target_blocked.links == ()
    assert DirectionReason.GROUNDING_UNUSABLE in ungrounded.assessments[0].reason_codes
    assert DirectionReason.COMPOSITION_INELIGIBLE in ineligible.assessments[0].reason_codes


@pytest.mark.parametrize(
    "admissibility",
    [MeasurementAdmissibilityV33.NOT_MEASUREMENT, MeasurementAdmissibilityV33.UNKNOWN],
)
def test_measurement_admissibility_blocks_affirmative_direction(
    admissibility: MeasurementAdmissibilityV33,
) -> None:
    result = evaluate_proposition_evidence_directions(
        (_measurement_scope(admissibility),), budget=_budget()
    )
    assert result.assessments[0].measurement_eligibility is admissibility
    assert result.assessments[0].direction_status is DirectionStatus.UNRESOLVED
    assert result.links == ()
    assert DirectionReason.MEASUREMENT_INADMISSIBLE in result.assessments[0].reason_codes


def test_measurement_semantics_are_projected_without_re_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[object] = []
    original = ComparabilityEvaluator.evaluate

    def capture(left: VerificationAssertion, right: VerificationAssertion):
        observed.extend(item.measurement_semantics for item in left.material_contexts)
        return original(left, right)

    monkeypatch.setattr(direction.ComparabilityEvaluator, "evaluate", capture)
    result = evaluate_proposition_evidence_directions(
        (_measurement_scope(MeasurementAdmissibilityV33.ADMISSIBLE),),
        budget=_budget(),
    )
    assert result.assessments[0].measurement_eligibility is MeasurementAdmissibilityV33.ADMISSIBLE
    assert observed and observed[0] is not None


def test_contradiction_direction_is_not_a_provider_or_legacy_relation_shortcut() -> None:
    result = evaluate_proposition_evidence_directions(
        (_scope((_source_assertion(value=False),)),), budget=_budget()
    )
    assert result.assessments[0].direction_status is DirectionStatus.RESOLVED_CONTRADICTION
    assert result.links[0].relation is EvidencePropositionRelation.CONTRADICTS


def test_related_requires_proposition_and_evidence_relative_mention_basis() -> None:
    # An explicit association is necessary for A3 RELATED; the bridge also binds it
    # to the exact AP, Evidence, and source assertion before emitting MENTIONS.
    target_value = PropositionValue(kind=PropositionValueKind.TEXT, text="a contextual description")
    changed_scope = _scope(mention=True, target_value=target_value)
    result = evaluate_proposition_evidence_directions((changed_scope,), budget=_budget())
    assert result.assessments[0].direction_status is DirectionStatus.RESOLVED_MENTION
    assert result.assessments[0].mention_basis_ref == "mention-basis-1"
    assert result.links[0].relation is EvidencePropositionRelation.MENTIONS
    assert changed_scope.targets[0].assertion.source_proposition_id == PROP_ID


def test_related_without_matching_mention_basis_stays_unresolved() -> None:
    result = evaluate_proposition_evidence_directions(
        (
            _scope(
                (_source_assertion(),),
                target_value=PropositionValue(kind=PropositionValueKind.TEXT, text="other"),
                explicit_association=True,
            ),
        ),
        budget=_budget(),
    )
    assert result.assessments[0].direction_status is DirectionStatus.UNRESOLVED
    assert DirectionReason.RELATED_WITHOUT_MENTION_BASIS in result.assessments[0].reason_codes
    assert result.links == ()


def test_multi_assertion_support_and_contradiction_suppress_link() -> None:
    first = _source_assertion("1" * 64, value=True)
    second = _source_assertion("2" * 64, value=False)
    result = evaluate_proposition_evidence_directions((_scope((first, second)),), budget=_budget())
    assert {item.direction_status for item in result.assessments} == {
        DirectionStatus.RESOLVED_SUPPORT,
        DirectionStatus.RESOLVED_CONTRADICTION,
    }
    assert len(result.aggregates) == 1
    assert DirectionReason.INTRA_EVIDENCE_DIRECTION_CONFLICT in result.aggregates[0].reason_codes
    assert result.links == ()


def test_same_direction_multiple_assertions_aggregate_deterministically() -> None:
    first_source = _source_assertion("1" * 64)
    second_source = _source_assertion("2" * 64)
    scope = _scope((first_source, second_source))
    first = evaluate_proposition_evidence_directions((scope,), budget=_budget())
    second = evaluate_proposition_evidence_directions((scope,), budget=_budget())
    reversed_scope = _scope((second_source, first_source))
    reordered = evaluate_proposition_evidence_directions((reversed_scope,), budget=_budget())
    assert first == second
    assert first == reordered
    assert len(first.links) == 1
    assert len(first.aggregates[0].source_assertion_refs) == 2


def test_multi_proposition_directions_do_not_leak_between_propositions() -> None:
    result = evaluate_proposition_evidence_directions(
        (_scope(target_ids=(PROP_ID, UUID("50000000-0000-0000-0000-000000000002"))),),
        budget=_budget(),
    )
    assert len(result.assessments) == 2
    assert {item.proposition_ref for item in result.assessments} == {
        PROP_ID,
        UUID("50000000-0000-0000-0000-000000000002"),
    }
    assert all(
        link.proposition_id in {PROP_ID, UUID("50000000-0000-0000-0000-000000000002")}
        for link in result.links
    )


def test_same_ap_and_evidence_in_different_claims_remain_separate_scopes() -> None:
    first = _scope()
    other_claim = UUID("10000000-0000-0000-0000-000000000002")
    second = _scope(claim_id=other_claim)
    result = evaluate_proposition_evidence_directions((first, second), budget=_budget())
    assert len(result.aggregates) == 2
    assert {item.claim_ref for item in result.aggregates} == {CLAIM_ID, other_claim}
    assert {item.claim_id for item in result.links} == {CLAIM_ID, other_claim}


def test_generated_link_is_accepted_by_existing_v22_b1_consumer() -> None:
    scope = _scope()
    result = evaluate_proposition_evidence_directions(
        (scope,), budget=_budget(), identity_alignments=_alignment_facts(scope)
    )
    grouping = group_propositions(
        (scope.targets[0].membership,),
        result.links,
        (
            PropositionAssertionBinding(
                proposition_ref=PROP_ID,
                assertion_ref=scope.targets[0].assertion.assertion_ref,
            ),
        ),
        (),
    )
    assert len(grouping.groups) == 1
    assert grouping.groups[0].evidence_memberships[0].evidence_ref == EVIDENCE_ID
    assert grouping.groups[0].evidence_memberships[0].provenance_links == result.links


def test_budget_marks_unprocessed_candidates_without_rolling_back_completed() -> None:
    scope = _scope(
        (_source_assertion("1" * 64), _source_assertion("2" * 64)),
        target_ids=(PROP_ID, UUID("50000000-0000-0000-0000-000000000002")),
    )
    result = evaluate_proposition_evidence_directions(
        (scope,), budget=_budget(max_candidate_pairs=2, max_direction_assessments=2)
    )
    assert (
        sum(
            item.direction_status is DirectionStatus.RESOLVED_SUPPORT for item in result.assessments
        )
        == 2
    )
    assert (
        sum(
            item.direction_status is DirectionStatus.NOT_PROCESSED_BUDGET
            for item in result.assessments
        )
        == 2
    )
    assert result.deferred_candidate_count == 2
    assert result.budget_state.provider_ap_decomposition_calls == 0
    assert result.budget_state.candidate_pairs == 2


def test_budget_limits_are_explicit_and_zero_is_not_a_quality_judgment() -> None:
    result = evaluate_proposition_evidence_directions(
        (_scope(),), budget=_budget(max_evidence_inputs=0)
    )
    assert result.assessments[0].direction_status is DirectionStatus.NOT_PROCESSED_BUDGET
    assert result.links == ()


def test_claim_evidence_or_assertion_binding_mismatch_is_rejected() -> None:
    scope = _scope()
    with pytest.raises(ValueError, match="source_assertion_evidence_provenance_mismatch"):
        EvidenceDirectionScope(
            claim=scope.claim,
            claim_evidence=scope.claim_evidence,
            targets=scope.targets,
            source_assertions=scope.source_assertions,
            source_bindings=(
                scope.source_bindings[0].model_copy(
                    update={"evidence_id": UUID("20000000-0000-0000-0000-000000000002")}
                ),
            ),
        )


def test_input_models_are_not_mutated() -> None:
    scope = _scope()
    before = scope.model_dump(mode="json")
    evaluate_proposition_evidence_directions((scope,), budget=_budget())
    assert scope.model_dump(mode="json") == before


def _alignment_facts(
    scope: EvidenceDirectionScope,
    *,
    subject: SemanticAlignmentStatus = SemanticAlignmentStatus.SAME,
    predicate: SemanticAlignmentStatus = SemanticAlignmentStatus.SAME,
) -> tuple[SemanticIdentityAlignmentFact, ...]:
    target_ref = str(scope.targets[0].proposition.proposition_id)
    source = scope.source_assertions[0]
    return tuple(
        SemanticIdentityAlignmentFact(
            alignment_ref=("a" if role is SemanticIdentityFieldRole.SUBJECT else "b") * 64,
            target_proposition_ref=target_ref,
            source_assertion_ref=source.assertion_ref,
            field_role=role,
            alignment_status=status,
            reason_code=(
                SemanticIdentityAlignmentReasonCode.IDENTITY_ALIGNED
                if status is SemanticAlignmentStatus.SAME
                else SemanticIdentityAlignmentReasonCode.IDENTITY_DISTINCT
                if status is SemanticAlignmentStatus.DIFFERENT
                else SemanticIdentityAlignmentReasonCode.AMBIGUOUS
            ),
            target_semantic_ref=("c" if role is SemanticIdentityFieldRole.SUBJECT else "d") * 64,
            source_semantic_ref=("e" if role is SemanticIdentityFieldRole.SUBJECT else "f") * 64,
            source_grounding_refs=("evidence-span-1",),
            basis_refs=(SPAN,),
        )
        for role, status in (
            (SemanticIdentityFieldRole.SUBJECT, subject),
            (SemanticIdentityFieldRole.PREDICATE, predicate),
        )
    )


def test_pair_local_alignment_flows_through_frozen_a2_a3_to_support() -> None:
    scope = _scope()
    before = scope.model_dump(mode="json")
    result = evaluate_proposition_evidence_directions(
        (scope,), budget=_budget(), identity_alignments=_alignment_facts(scope)
    )
    assert result.assessments[0].direction_status is DirectionStatus.RESOLVED_SUPPORT
    assert result.links[0].relation is EvidencePropositionRelation.SUPPORTS
    assert scope.model_dump(mode="json") == before


def test_pair_local_alignment_flows_through_frozen_a2_a3_to_contradiction() -> None:
    scope = _scope((_source_assertion(value=False),))
    result = evaluate_proposition_evidence_directions(
        (scope,), budget=_budget(), identity_alignments=_alignment_facts(scope)
    )
    assert result.assessments[0].direction_status is DirectionStatus.RESOLVED_CONTRADICTION
    assert result.links[0].relation is EvidencePropositionRelation.CONTRADICTS


def test_same_surface_identity_without_alignment_is_unresolved() -> None:
    scope = _scope()
    result = evaluate_proposition_evidence_directions(
        (scope,),
        budget=_budget(),
        identity_alignments=_alignment_facts(
            scope,
            subject=SemanticAlignmentStatus.UNKNOWN,
            predicate=SemanticAlignmentStatus.UNKNOWN,
        ),
    )
    assert result.assessments[0].direction_status is DirectionStatus.UNRESOLVED
    assert result.links == ()
    assert DirectionReason.IDENTITY_UNRESOLVED in result.assessments[0].reason_codes


def test_predicate_unknown_blocks_direction_even_when_subject_is_same() -> None:
    scope = _scope()
    facts = _alignment_facts(scope, predicate=SemanticAlignmentStatus.UNKNOWN)
    result = evaluate_proposition_evidence_directions(
        (scope,), budget=_budget(), identity_alignments=facts
    )
    assert result.assessments[0].direction_status is DirectionStatus.UNRESOLVED
    assert result.links == ()


def test_different_identity_is_pair_scoped_and_incompatible() -> None:
    from app.domain.semantic_identity_alignment import project_pair_identities

    scope = _scope()
    facts = _alignment_facts(scope, subject=SemanticAlignmentStatus.DIFFERENT)
    left, right = project_pair_identities(facts[0])
    assert left.status is SemanticIdentityStatus.RESOLVED
    assert right.status is SemanticIdentityStatus.RESOLVED
    assert left.key != right.key
    result = evaluate_proposition_evidence_directions(
        (scope,), budget=_budget(), identity_alignments=facts
    )
    assert result.assessments[0].direction_status is DirectionStatus.UNRESOLVED
    assert result.links == ()


def test_duplicate_pair_field_alignment_fails_closed() -> None:
    scope = _scope()
    facts = _alignment_facts(scope)
    result = evaluate_proposition_evidence_directions(
        (scope,), budget=_budget(), identity_alignments=(*facts, facts[0])
    )
    assert result.assessments[0].direction_status is DirectionStatus.INVALID_INPUT
    assert result.links == ()


def test_budget_exhaustion_does_not_map_to_insufficient_link() -> None:
    result = evaluate_proposition_evidence_directions(
        (_scope(),), budget=_budget(max_candidate_pairs=0)
    )
    assert result.assessments[0].direction_status is DirectionStatus.NOT_PROCESSED_BUDGET
    assert all(
        link.relation is not EvidencePropositionRelation.INSUFFICIENT for link in result.links
    )


def test_result_references_are_deterministic_and_contain_no_quote_text() -> None:
    scope = _scope()
    first = evaluate_proposition_evidence_directions((scope,), budget=_budget())
    second = evaluate_proposition_evidence_directions((scope,), budget=_budget())
    assert first.assessments[0].assessment_ref == second.assessments[0].assessment_ref
    assert "The system is active" not in first.assessments[0].model_dump_json()


def test_provider_budget_is_not_consumed_by_step_two() -> None:
    result = evaluate_proposition_evidence_directions((_scope(),), budget=_budget())
    assert result.budget_state.provider_ap_decomposition_calls == 0
    assert result.budget_state.provider_source_semantic_extraction_calls == 0
    assert result.budget_state.provider_identity_resolution_calls == 0
