from __future__ import annotations

from uuid import UUID

import pytest
from app.domain.gap_research_intent import (
    DesiredEvidencePurpose,
)
from app.domain.proposition_research_gap import (
    DesiredEvidenceRelation,
    GapPriorityClass,
    GapPriorityInputs,
    GapSearchability,
    PropositionGapLifecycle,
    PropositionGapReasonCode,
    PropositionResearchGap,
    PropositionResearchGapType,
    RequiredEvidenceCharacteristics,
)
from app.domain.research_context import ResearchContext
from app.services.gap_research_intent import GapResearchIntentService

RUN = UUID("50000000-0000-0000-0000-000000000001")
CLAIM = UUID("20000000-0000-0000-0000-000000000001")
PROP = UUID("10000000-0000-0000-0000-000000000001")
GROUP = "a" * 64


def _gap(
    gap_type: PropositionResearchGapType = PropositionResearchGapType.MISSING_SUPPORT,
    *,
    searchability: GapSearchability = GapSearchability.SEARCHABLE,
    relations: tuple[DesiredEvidenceRelation, ...] = (DesiredEvidenceRelation.SUPPORTS,),
    required_owner_independence: bool = False,
    excluded_owner_keys: tuple[str, ...] = (),
    context_refs: tuple[str, ...] = (),
    time_scope_refs: tuple[str, ...] = (),
    a2_refs: tuple[str, ...] = (),
    support_refs: tuple[str, ...] = (),
    contradict_refs: tuple[str, ...] = (),
    conflict_refs: tuple[str, ...] = (),
    attempt: int = 0,
    gap_ref: str = "b" * 64,
    budget_state: dict[str, int] | None = None,
) -> PropositionResearchGap:
    required = RequiredEvidenceCharacteristics(
        target_group_ref=GROUP,
        target_proposition_refs=(PROP,),
        desired_relations=tuple(sorted(set(relations), key=lambda item: item.value)),
        required_owner_independence=required_owner_independence,
        excluded_owner_keys=excluded_owner_keys,
        required_context_refs=context_refs,
        required_time_version_scope_condition_refs=time_scope_refs,
        a2_conflict_pair_refs=a2_refs,
        support_side_evidence_refs=support_refs,
        contradict_side_evidence_refs=contradict_refs,
        conflict_group_refs=conflict_refs,
    )
    return PropositionResearchGap(
        gap_ref=gap_ref,
        run_ref=RUN,
        target_group_ref=GROUP,
        target_proposition_refs=(PROP,),
        parent_claim_refs=(CLAIM,),
        objective_scope_ref="scope:project-objective",
        verification_assessment_ref="c" * 64,
        verification_assessment_hash="d" * 64,
        gap_type=gap_type,
        searchability=searchability,
        reason_codes=(PropositionGapReasonCode.NO_ELIGIBLE_SUPPORT,),
        required_evidence=required,
        existing_evidence_refs=("evidence:existing-1",),
        existing_owner_refs=("owner:existing-1",),
        existing_conflict_refs=tuple(sorted(conflict_refs)),
        source_reliability_refs=("source-assessment:1",),
        source_reliability_reason_codes=("LINEAGE_UNKNOWN",),
        lifecycle_state=(
            PropositionGapLifecycle.OPEN
            if searchability is GapSearchability.SEARCHABLE
            else PropositionGapLifecycle.BLOCKED
        ),
        attempt_count=attempt,
        priority_inputs=GapPriorityInputs(
            priority_class=(
                GapPriorityClass.ACTIVE_CONFLICT
                if gap_type is PropositionResearchGapType.CONFLICT_RESOLUTION
                else GapPriorityClass.MISSING_SUPPORT
            ),
            parent_research_priority=1,
            attempt_count=attempt,
        ),
        budget_state_ref="e" * 64,
        budget_state=budget_state,
    )


def test_missing_support_maps_only_to_support_and_preserves_proposition_identity() -> None:
    result = GapResearchIntentService().derive((_gap(),))
    assert len(result.intents) == 1
    intent = result.intents[0]
    assert intent.desired_evidence_purpose is DesiredEvidencePurpose.SUPPORT
    assert intent.target_proposition_refs == (PROP,)
    assert intent.target_group_ref == GROUP
    assert intent.parent_claim_refs == (CLAIM,)
    assert intent.required_evidence.desired_relations == (DesiredEvidenceRelation.SUPPORTS,)
    assert not hasattr(intent, "query_text")


def test_independent_corroboration_preserves_owner_exclusions_and_parallel_legacy_context() -> None:
    gap = _gap(
        PropositionResearchGapType.MISSING_INDEPENDENT_CORROBORATION,
        required_owner_independence=True,
        excluded_owner_keys=("owner:existing-1",),
    )
    legacy = ResearchContext(
        evidence_failure_reason="INSUFFICIENT_INDEPENDENT_SOURCES",
        query_hints=("legacy hint",),
        existing_source_owners=("owner:existing-1",),
    )
    service = GapResearchIntentService()
    result = service.derive((gap,))
    projected = service.project(result.intents, legacy_context=legacy)
    assert (
        result.intents[0].desired_evidence_purpose
        is DesiredEvidencePurpose.INDEPENDENT_CORROBORATION
    )
    assert result.intents[0].required_evidence.excluded_owner_keys == ("owner:existing-1",)
    assert projected.research_context is not None
    assert projected.research_context.query_hints == legacy.query_hints
    assert projected.research_context.evidence_failure_reason == legacy.evidence_failure_reason
    assert projected.research_context.proposition_gap_intents == result.intents
    restored = ResearchContext.from_mapping(projected.research_context.as_dict())
    assert restored.proposition_gap_intents == result.intents


def test_conflict_intent_retains_both_sides_and_is_not_plain_support() -> None:
    gap = _gap(
        PropositionResearchGapType.CONFLICT_RESOLUTION,
        relations=(
            DesiredEvidenceRelation.SUPPORTS,
            DesiredEvidenceRelation.CONTRADICTS,
            DesiredEvidenceRelation.DISAMBIGUATES,
        ),
        support_refs=("evidence:support-side",),
        contradict_refs=("evidence:contradict-side",),
        conflict_refs=("conflict:1",),
    )
    intent = GapResearchIntentService().derive((gap,)).intents[0]
    assert intent.desired_evidence_purpose is DesiredEvidencePurpose.CONFLICT_DISAMBIGUATION
    assert intent.existing_conflict_refs == ("conflict:1",)
    assert intent.required_evidence.support_side_evidence_refs == ("evidence:support-side",)
    assert intent.required_evidence.contradict_side_evidence_refs == ("evidence:contradict-side",)
    assert set(intent.required_evidence.desired_relations) == {
        DesiredEvidenceRelation.SUPPORTS,
        DesiredEvidenceRelation.CONTRADICTS,
        DesiredEvidenceRelation.DISAMBIGUATES,
    }


@pytest.mark.parametrize(
    ("gap_type", "kwargs", "expected_purpose"),
    (
        (
            PropositionResearchGapType.CONTEXT_DISAMBIGUATION,
            {
                "relations": (DesiredEvidenceRelation.DISAMBIGUATES,),
                "time_scope_refs": ("formal:time-scope-1",),
            },
            DesiredEvidencePurpose.CONTEXT_RESOLUTION,
        ),
        (
            PropositionResearchGapType.SCOPE_COMPLETION,
            {},
            DesiredEvidencePurpose.SCOPE_EXTENSION,
        ),
    ),
)
def test_context_and_scope_intents_keep_formal_refs(gap_type, kwargs, expected_purpose) -> None:
    intent = GapResearchIntentService().derive((_gap(gap_type, **kwargs),)).intents[0]
    assert intent.desired_evidence_purpose is expected_purpose
    assert intent.objective_scope_ref == "scope:project-objective"
    if gap_type is PropositionResearchGapType.CONTEXT_DISAMBIGUATION:
        assert intent.required_evidence.required_time_version_scope_condition_refs == (
            "formal:time-scope-1",
        )


@pytest.mark.parametrize(
    ("gap_type", "searchability"),
    (
        (PropositionResearchGapType.MISSING_SUPPORT, GapSearchability.NON_SEARCHABLE),
        (PropositionResearchGapType.MISSING_SUPPORT, GapSearchability.UNKNOWN_SEARCHABILITY),
        (PropositionResearchGapType.SEMANTIC_CAPABILITY_BLOCKED, GapSearchability.NON_SEARCHABLE),
        (PropositionResearchGapType.SEMANTIC_WORK_NOT_PROCESSED, GapSearchability.NON_SEARCHABLE),
    ),
)
def test_nonsearchable_unknown_and_blocked_gaps_emit_no_intent(gap_type, searchability) -> None:
    result = GapResearchIntentService().derive(
        (_gap(gap_type, searchability=searchability),)
    )
    assert result.intents == ()
    assert result.issues == ()


def test_dedup_identity_depends_on_gap_and_attempt_not_query_and_is_deterministic() -> None:
    gap = _gap()
    first = GapResearchIntentService().derive((gap, gap)).intents
    second = GapResearchIntentService().derive((gap,)).intents
    later_attempt = GapResearchIntentService().derive((_gap(attempt=1),)).intents[0]
    assert len(first) == 1
    assert first == second
    assert first[0].gap_ref == later_attempt.gap_ref
    assert first[0].intent_ref != later_attempt.intent_ref
    assert "query" not in first[0].model_dump(mode="json")


def test_budget_snapshot_is_preserved_as_immutable_canonical_pairs() -> None:
    gap = _gap(budget_state={"searches_remaining": 2, "readers_remaining": 1})
    intent = GapResearchIntentService().derive((gap,)).intents[0]
    assert intent.budget_state_ref == "e" * 64
    assert intent.budget_state == (("readers_remaining", 1), ("searches_remaining", 2))


def test_malformed_gap_fails_closed_without_blocking_other_valid_gap() -> None:
    invalid = _gap(
        PropositionResearchGapType.MISSING_INDEPENDENT_CORROBORATION,
        gap_ref="f" * 64,
    )
    result = GapResearchIntentService().derive((invalid, _gap()))
    assert len(result.intents) == 1
    assert result.intents[0].gap_ref == "b" * 64
    assert result.issues


def test_empty_intent_projection_keeps_legacy_context_serialization_unchanged() -> None:
    context = ResearchContext(query_hints=("legacy",))
    legacy_shape = {
        "evidence_failure_reason": None,
        "missing_evidence_type": None,
        "refined_need_type": None,
        "query_hints": ["legacy"],
        "existing_source_owners": [],
        "existing_domains": [],
        "required_source_count": None,
        "missing_source_count": None,
        "target_claim_id": None,
    }
    assert context.as_dict() == legacy_shape
    restored = ResearchContext.from_mapping(legacy_shape)
    assert restored == context
    with pytest.raises(ValueError, match="must_be_a_sequence"):
        ResearchContext.from_mapping({"proposition_gap_intents": "malformed"})


def test_research_context_rejects_mutable_or_unvalidated_intent_collections() -> None:
    intent = GapResearchIntentService().derive((_gap(),)).intents[0]
    with pytest.raises(TypeError, match="immutable_tuple"):
        ResearchContext(proposition_gap_intents=[intent])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="validated_intents"):
        ResearchContext(proposition_gap_intents=(object(),))  # type: ignore[arg-type]


def test_priority_order_is_inherited_from_gap_priority() -> None:
    regular = _gap(gap_ref="b" * 64)
    conflict = _gap(
        PropositionResearchGapType.CONFLICT_RESOLUTION,
        relations=(
            DesiredEvidenceRelation.SUPPORTS,
            DesiredEvidenceRelation.CONTRADICTS,
            DesiredEvidenceRelation.DISAMBIGUATES,
        ),
        support_refs=("support",),
        contradict_refs=("contradict",),
        conflict_refs=("conflict",),
        gap_ref="c" * 64,
    )
    result = GapResearchIntentService().derive((regular, conflict))
    assert result.intents[0].gap_type is PropositionResearchGapType.CONFLICT_RESOLUTION


def test_resume_rebuilds_identical_intents_and_planning_projection() -> None:
    gaps = (_gap(),)
    first_service = GapResearchIntentService()
    second_service = GapResearchIntentService()
    first_intents = first_service.derive(gaps).intents
    second_intents = second_service.derive(gaps).intents
    legacy = ResearchContext(query_hints=("legacy",))
    first = first_service.project(first_intents, legacy_context=legacy)
    second = second_service.project(second_intents, legacy_context=legacy)
    assert first_intents == second_intents
    assert first.projection == second.projection
    assert first.research_context == second.research_context
