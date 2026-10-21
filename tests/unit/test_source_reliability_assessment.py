from dataclasses import FrozenInstanceError, fields

import pytest
from app.domain.source_reliability_assessment import (
    SCHEMA_VERSION,
    ClaimInputKind,
    ClaimSourceAuthorityFit,
    ClaimSourceFitInput,
    ExplicitFact,
    FactStatus,
    IdentityClarity,
    Originality,
    RecencyFit,
    ReliabilityDimension,
    SourcePropositionRelation,
    SourceReliabilityInput,
    assess_source_reliability,
)


def _base(
    *,
    source_ref: str = "source:one",
    claim_ref: str = "claim:one",
    relation: SourcePropositionRelation = SourcePropositionRelation.UNKNOWN,
    relation_refs: tuple[str, ...] = (),
    authority_scope_match: ExplicitFact | None = None,
    **kwargs: object,
) -> SourceReliabilityInput:
    return SourceReliabilityInput(
        source_ref=source_ref,
        claim_source_fit=ClaimSourceFitInput(
            claim_ref=claim_ref,
            input_kind=ClaimInputKind.PROPOSITION,
            source_relation=relation,
            relation_basis_refs=relation_refs,
            authority_scope_match=authority_scope_match or ExplicitFact(),
        ),
        **kwargs,
    )


def _value(
    value: SourceReliabilityInput,
    dimension: ReliabilityDimension,
) -> str:
    return assess_source_reliability(value).dimension(dimension).value


def test_default_assessment_is_unknown_without_inference_or_score() -> None:
    result = assess_source_reliability(_base())

    assert result.schema_version == SCHEMA_VERSION
    assert result.dimension(ReliabilityDimension.IDENTITY_CLARITY).value == "UNKNOWN"
    assert result.dimension(ReliabilityDimension.ORIGINALITY).value == "UNKNOWN"
    assert result.dimension(ReliabilityDimension.DIRECTNESS).value == "UNKNOWN"
    assert result.dimension(ReliabilityDimension.AUTHORITY_FIT).value == "UNRESOLVED"
    assert result.dimension(ReliabilityDimension.LINEAGE_STATUS).value == "UNKNOWN"
    assert not hasattr(result, "score")
    assert not hasattr(result, "verification_status")


@pytest.mark.parametrize(
    ("identity", "refs", "expected"),
    [
        (IdentityClarity.KNOWN, ("source-record:1",), "KNOWN"),
        (IdentityClarity.AMBIGUOUS, ("source-record:2",), "AMBIGUOUS"),
        (IdentityClarity.KNOWN, (), "UNKNOWN"),
        (IdentityClarity.UNKNOWN, ("untrusted-url-looks-official",), "UNKNOWN"),
    ],
)
def test_identity_requires_explicit_basis(
    identity: IdentityClarity, refs: tuple[str, ...], expected: str
) -> None:
    result = assess_source_reliability(_base(identity_clarity=identity, identity_basis_refs=refs))
    assert result.dimension(ReliabilityDimension.IDENTITY_CLARITY).value == expected


def test_known_identity_does_not_imply_authority() -> None:
    result = assess_source_reliability(
        _base(
            identity_clarity=IdentityClarity.KNOWN,
            identity_basis_refs=("identity-record",),
        )
    )
    assert result.dimension(ReliabilityDimension.IDENTITY_CLARITY).value == "KNOWN"
    assert result.dimension(ReliabilityDimension.AUTHORITY_FIT).value == "UNRESOLVED"


def test_explicit_original_source_is_supported_with_provenance() -> None:
    result = assess_source_reliability(
        _base(originality=Originality.ORIGINAL, originality_basis_refs=("source-doc",))
    )
    assert result.dimension(ReliabilityDimension.ORIGINALITY).value == "ORIGINAL"
    assert result.dimension(ReliabilityDimension.LINEAGE_STATUS).value == "ORIGINAL"
    assert result.dimension(ReliabilityDimension.LINEAGE_STATUS).basis_refs == ("source-doc",)


def test_explicit_original_source_link_means_derived_lineage() -> None:
    result = assess_source_reliability(_base(original_source_ref="source:origin"))
    assert (
        result.dimension(ReliabilityDimension.ORIGINALITY).value
        == "DERIVED_FROM_KNOWN_SOURCE"
    )
    assert (
        result.dimension(ReliabilityDimension.LINEAGE_STATUS).value == "DERIVED_FROM_KNOWN_SOURCE"
    )
    assert result.dimension(ReliabilityDimension.LINEAGE_STATUS).basis_refs == ("source:origin",)


def test_shared_origin_is_explicit_and_distinct_from_owner_identity() -> None:
    result = assess_source_reliability(_base(shared_origin_ref="lineage:shared"))
    assert result.dimension(ReliabilityDimension.LINEAGE_STATUS).value == "SHARED_ORIGIN_KNOWN"
    assert "source_owner_key" not in {item.name for item in fields(SourceReliabilityInput)}


def test_unknown_lineage_stays_unknown_and_does_not_assume_url_originality() -> None:
    result = assess_source_reliability(_base(source_ref="https://official-looking.example"))
    assert result.dimension(ReliabilityDimension.ORIGINALITY).value == "UNKNOWN"
    assert result.dimension(ReliabilityDimension.LINEAGE_STATUS).value == "UNKNOWN"


def test_conflicting_lineage_refs_fail_closed() -> None:
    result = assess_source_reliability(
        _base(original_source_ref="source:a", shared_origin_ref="source:b")
    )
    assert result.dimension(ReliabilityDimension.LINEAGE_STATUS).value == "UNKNOWN"
    assert "LINEAGE_CONFLICTING_PROVENANCE" in result.reason_codes


@pytest.mark.parametrize(
    ("relation", "authority", "expected"),
    [
        (
            SourcePropositionRelation.FORMAL_AUTHORITY,
            ExplicitFact(FactStatus.SUPPORTED, ("scope-record",)),
            ClaimSourceAuthorityFit.DIRECTLY_AUTHORITATIVE,
        ),
        (
            SourcePropositionRelation.FORMAL_AUTHORITY,
            ExplicitFact(FactStatus.NOT_SUPPORTED, ("scope-check",)),
            ClaimSourceAuthorityFit.UNRESOLVED,
        ),
        (
            SourcePropositionRelation.FORMAL_AUTHORITY,
            ExplicitFact(FactStatus.SUPPORTED),
            ClaimSourceAuthorityFit.UNRESOLVED,
        ),
        (
            SourcePropositionRelation.UNKNOWN,
            ExplicitFact(FactStatus.SUPPORTED, ("scope-record",)),
            ClaimSourceAuthorityFit.UNRESOLVED,
        ),
    ],
)
def test_formal_authority_requires_relation_and_scope_basis(
    relation: SourcePropositionRelation,
    authority: ExplicitFact,
    expected: ClaimSourceAuthorityFit,
) -> None:
    result = assess_source_reliability(
        _base(
            relation=relation,
            relation_refs=("relation-record",),
            authority_scope_match=authority,
        )
    )
    assert result.claim_source_fit.assessment is expected


def test_same_source_can_have_different_fit_for_different_propositions() -> None:
    direct = assess_source_reliability(
        _base(
            source_ref="source:filing",
            claim_ref="proposition:self-reported-fact",
            relation=SourcePropositionRelation.SELF_REPORTED,
            relation_refs=("filing:statement",),
        )
    )
    marketing = assess_source_reliability(
        _base(
            source_ref="source:filing",
            claim_ref="proposition:comparative-superiority",
            relation=SourcePropositionRelation.UNKNOWN,
        )
    )
    assert direct.claim_source_fit.assessment is ClaimSourceAuthorityFit.DIRECT_BUT_SELF_REPORTED
    assert marketing.claim_source_fit.assessment is ClaimSourceAuthorityFit.UNRESOLVED


def test_self_report_is_separate_from_directness_authority_and_conflict() -> None:
    result = assess_source_reliability(
        _base(
            relation=SourcePropositionRelation.SELF_REPORTED,
            relation_refs=("statement:1",),
            self_report=ExplicitFact(FactStatus.SUPPORTED, ("statement:1",)),
            conflict_of_interest=ExplicitFact(FactStatus.UNKNOWN),
        )
    )
    assert result.dimension(ReliabilityDimension.SELF_REPORT_STATUS).value == "SUPPORTED"
    assert result.dimension(ReliabilityDimension.DIRECTNESS).value == "DIRECT"
    assert result.dimension(ReliabilityDimension.AUTHORITY_FIT).value == "DIRECT_BUT_SELF_REPORTED"
    assert result.dimension(ReliabilityDimension.CONFLICT_OF_INTEREST).value == "UNKNOWN"


def test_self_report_does_not_auto_reject_or_auto_establish_authority() -> None:
    result = assess_source_reliability(
        _base(self_report=ExplicitFact(FactStatus.SUPPORTED, ("source-statement",)))
    )
    assert result.dimension(ReliabilityDimension.SELF_REPORT_STATUS).value == "SUPPORTED"
    assert result.claim_source_fit.assessment is ClaimSourceAuthorityFit.UNRESOLVED


@pytest.mark.parametrize(
    ("relation", "fit", "directness"),
    [
        (SourcePropositionRelation.SELF_REPORTED, "DIRECT_BUT_SELF_REPORTED", "DIRECT"),
        (SourcePropositionRelation.SECONDARY_REPORT, "SECONDARY_SUPPORT", "INDIRECT"),
        (SourcePropositionRelation.INDIRECT, "INDIRECT", "INDIRECT"),
        (SourcePropositionRelation.UNKNOWN, "UNRESOLVED", "UNKNOWN"),
    ],
)
def test_claim_source_fit_and_directness_remain_separate(
    relation: SourcePropositionRelation, fit: str, directness: str
) -> None:
    result = assess_source_reliability(
        _base(relation=relation, relation_refs=("relation-evidence",))
    )
    assert result.dimension(ReliabilityDimension.AUTHORITY_FIT).value == fit
    assert result.dimension(ReliabilityDimension.DIRECTNESS).value == directness


def test_relation_without_basis_becomes_unknown() -> None:
    result = assess_source_reliability(_base(relation=SourcePropositionRelation.SELF_REPORTED))
    assert result.dimension(ReliabilityDimension.DIRECTNESS).value == "UNKNOWN"
    assert result.dimension(ReliabilityDimension.AUTHORITY_FIT).value == "UNRESOLVED"
    assert "RELATION_BASIS_MISSING" in result.reason_codes


@pytest.mark.parametrize(
    ("field", "dimension"),
    [
        ("expertise_fit", ReliabilityDimension.EXPERTISE_FIT),
        ("transparency", ReliabilityDimension.TRANSPARENCY),
        ("conflict_of_interest", ReliabilityDimension.CONFLICT_OF_INTEREST),
    ],
)
def test_explicit_metadata_dimensions_require_basis_refs(
    field: str, dimension: ReliabilityDimension
) -> None:
    result = assess_source_reliability(_base(**{field: ExplicitFact(FactStatus.SUPPORTED)}))
    assert result.dimension(dimension).value == "UNKNOWN"


def test_secondary_reporting_is_not_automatically_unreliable_or_original() -> None:
    result = assess_source_reliability(
        _base(
            relation=SourcePropositionRelation.SECONDARY_REPORT,
            relation_refs=("reporting-record",),
        )
    )
    assert result.claim_source_fit.assessment is ClaimSourceAuthorityFit.SECONDARY_SUPPORT
    assert result.dimension(ReliabilityDimension.ORIGINALITY).value == "UNKNOWN"
    assert result.dimension(ReliabilityDimension.LINEAGE_STATUS).value == "UNKNOWN"


def test_recency_is_not_applicable_only_with_explicit_non_temporal_basis() -> None:
    result = assess_source_reliability(
        _base(
            proposition_temporal_sensitivity=ExplicitFact(
                FactStatus.NOT_SUPPORTED, ("proposition-review",)
            ),
            recency_fit=RecencyFit.MISMATCH,
        )
    )
    recency = result.dimension(ReliabilityDimension.RECENCY_FIT)
    assert recency.value == "NOT_APPLICABLE"
    assert recency.materiality is False


def test_recency_mismatch_requires_temporal_and_recency_basis() -> None:
    result = assess_source_reliability(
        _base(
            proposition_temporal_sensitivity=ExplicitFact(
                FactStatus.SUPPORTED, ("temporal-requirement",)
            ),
            recency_fit=RecencyFit.MISMATCH,
            recency_basis_refs=("publication-date-comparison",),
        )
    )
    recency = result.dimension(ReliabilityDimension.RECENCY_FIT)
    assert recency.value == "MISMATCH"
    assert recency.materiality is True


def test_fetch_time_or_missing_publication_time_cannot_establish_recency() -> None:
    result = assess_source_reliability(_base(snapshot_ref="snapshot:fetched-today"))
    assert result.dimension(ReliabilityDimension.RECENCY_FIT).value == "UNKNOWN"


def test_legacy_claim_type_signal_is_never_new_fit_truth() -> None:
    legacy = ClaimSourceFitInput(
        claim_ref="legacy-claim",
        input_kind=ClaimInputKind.LEGACY_CLAIM_BRIDGE,
        legacy_claim_type_signal="market_statistic",
        source_relation=SourcePropositionRelation.UNKNOWN,
    )
    result = assess_source_reliability(
        SourceReliabilityInput(source_ref="source", claim_source_fit=legacy)
    )
    assert result.claim_source_fit.claim_type_status == "LEGACY_SIGNAL_ONLY"
    assert result.claim_source_fit.assessment is ClaimSourceAuthorityFit.UNRESOLVED


def test_legacy_keyword_signal_does_not_change_authority_result() -> None:
    a = assess_source_reliability(
        _base(
            claim_ref="plain",
            relation=SourcePropositionRelation.UNKNOWN,
        )
    )
    legacy_input = ClaimSourceFitInput(
        claim_ref="legacy",
        input_kind=ClaimInputKind.LEGACY_CLAIM_BRIDGE,
        legacy_claim_type_signal="comparative",
        source_relation=SourcePropositionRelation.UNKNOWN,
    )
    b = assess_source_reliability(
        SourceReliabilityInput(source_ref="source:one", claim_source_fit=legacy_input)
    )
    assert a.claim_source_fit.assessment is b.claim_source_fit.assessment


def test_assessment_is_deterministic_and_basis_order_is_normalized() -> None:
    first = _base(
        identity_clarity=IdentityClarity.KNOWN,
        identity_basis_refs=("basis:b", "basis:a"),
    )
    second = _base(
        identity_clarity=IdentityClarity.KNOWN,
        identity_basis_refs=("basis:a", "basis:b"),
    )
    assert assess_source_reliability(first) == assess_source_reliability(second)


def test_assessment_does_not_mutate_input() -> None:
    source = _base(
        relation=SourcePropositionRelation.FORMAL_AUTHORITY,
        relation_refs=("relation",),
        authority_scope_match=ExplicitFact(FactStatus.SUPPORTED, ("scope",)),
    )
    before = source
    assess_source_reliability(source)
    assert source == before


def test_contract_objects_are_immutable() -> None:
    assessment = assess_source_reliability(_base())
    with pytest.raises(FrozenInstanceError):
        assessment.source_ref = "changed"  # type: ignore[misc]


def test_no_reliability_aggregation_or_verification_output_exists() -> None:
    result = assess_source_reliability(_base())
    assert {item.dimension for item in result.dimensions} == set(ReliabilityDimension)
    assert not any(hasattr(result, name) for name in ("score", "confidence", "verified"))


def test_direct_authority_and_self_report_can_coexist_without_score() -> None:
    result = assess_source_reliability(
        _base(
            relation=SourcePropositionRelation.SELF_REPORTED,
            relation_refs=("filing-statement",),
            self_report=ExplicitFact(FactStatus.SUPPORTED, ("filing-statement",)),
            conflict_of_interest=ExplicitFact(FactStatus.SUPPORTED, ("interest-record",)),
        )
    )
    assert result.dimension(ReliabilityDimension.SELF_REPORT_STATUS).value == "SUPPORTED"
    assert result.dimension(ReliabilityDimension.CONFLICT_OF_INTEREST).value == "SUPPORTED"
    assert not hasattr(result, "score")
