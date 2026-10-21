"""Proposition-relative source reliability facts for shadow evaluation.

This contract intentionally does not calculate a trust score, infer authority
from URLs/roles, or alter the legacy evidence acceptance path.  Affirmative
facts require explicit provenance references; absent or unsupported inputs
remain UNKNOWN.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

SCHEMA_VERSION = "source-reliability-assessment.v1"


class ClaimInputKind(StrEnum):
    PROPOSITION = "PROPOSITION"
    LEGACY_CLAIM_BRIDGE = "LEGACY_CLAIM_BRIDGE"


class ClaimTypeSignalStatus(StrEnum):
    LEGACY_SIGNAL_ONLY = "LEGACY_SIGNAL_ONLY"
    CLAIM_TYPE_UNKNOWN = "CLAIM_TYPE_UNKNOWN"


class FactStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    UNKNOWN = "UNKNOWN"


class IdentityClarity(StrEnum):
    KNOWN = "KNOWN"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"


class Originality(StrEnum):
    ORIGINAL = "ORIGINAL"
    DERIVED_FROM_KNOWN_SOURCE = "DERIVED_FROM_KNOWN_SOURCE"
    SHARED_ORIGIN_KNOWN = "SHARED_ORIGIN_KNOWN"
    UNKNOWN = "UNKNOWN"


class SourcePropositionRelation(StrEnum):
    DIRECT_OBSERVATION = "DIRECT_OBSERVATION"
    FORMAL_AUTHORITY = "FORMAL_AUTHORITY"
    SELF_REPORTED = "SELF_REPORTED"
    SECONDARY_REPORT = "SECONDARY_REPORT"
    INDIRECT = "INDIRECT"
    UNKNOWN = "UNKNOWN"


class Directness(StrEnum):
    DIRECT = "DIRECT"
    INDIRECT = "INDIRECT"
    UNKNOWN = "UNKNOWN"


class ClaimSourceAuthorityFit(StrEnum):
    DIRECTLY_AUTHORITATIVE = "DIRECTLY_AUTHORITATIVE"
    DIRECT_BUT_SELF_REPORTED = "DIRECT_BUT_SELF_REPORTED"
    SECONDARY_SUPPORT = "SECONDARY_SUPPORT"
    INDIRECT = "INDIRECT"
    UNRESOLVED = "UNRESOLVED"


class RecencyFit(StrEnum):
    FIT = "FIT"
    MISMATCH = "MISMATCH"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNKNOWN = "UNKNOWN"


class ReliabilityDimension(StrEnum):
    IDENTITY_CLARITY = "IDENTITY_CLARITY"
    ORIGINALITY = "ORIGINALITY"
    DIRECTNESS = "DIRECTNESS"
    AUTHORITY_FIT = "AUTHORITY_FIT"
    EXPERTISE_FIT = "EXPERTISE_FIT"
    SELF_REPORT_STATUS = "SELF_REPORT_STATUS"
    TRANSPARENCY = "TRANSPARENCY"
    RECENCY_FIT = "RECENCY_FIT"
    CONFLICT_OF_INTEREST = "CONFLICT_OF_INTEREST"
    LINEAGE_STATUS = "LINEAGE_STATUS"


@dataclass(frozen=True, slots=True)
class ExplicitFact:
    """A typed fact and the provenance references that support it."""

    status: FactStatus = FactStatus.UNKNOWN
    basis_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ClaimSourceFitInput:
    """Minimal bridge input; legacy claim text is not a V2 proposition."""

    claim_ref: str
    input_kind: ClaimInputKind
    legacy_claim_type_signal: str | None = None
    source_relation: SourcePropositionRelation = SourcePropositionRelation.UNKNOWN
    relation_basis_refs: tuple[str, ...] = ()
    authority_scope_match: ExplicitFact = ExplicitFact()


@dataclass(frozen=True, slots=True)
class SourceReliabilityInput:
    source_ref: str
    claim_source_fit: ClaimSourceFitInput
    snapshot_ref: str | None = None
    identity_clarity: IdentityClarity = IdentityClarity.UNKNOWN
    identity_basis_refs: tuple[str, ...] = ()
    originality: Originality = Originality.UNKNOWN
    originality_basis_refs: tuple[str, ...] = ()
    expertise_fit: ExplicitFact = ExplicitFact()
    self_report: ExplicitFact = ExplicitFact()
    transparency: ExplicitFact = ExplicitFact()
    proposition_temporal_sensitivity: ExplicitFact = ExplicitFact()
    recency_fit: RecencyFit = RecencyFit.UNKNOWN
    recency_basis_refs: tuple[str, ...] = ()
    conflict_of_interest: ExplicitFact = ExplicitFact()
    original_source_ref: str | None = None
    shared_origin_ref: str | None = None


@dataclass(frozen=True, slots=True)
class ReliabilityDimensionAssessment:
    dimension: ReliabilityDimension
    value: StrEnum
    basis_refs: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    materiality: bool | None = None


@dataclass(frozen=True, slots=True)
class ClaimSourceFitAssessment:
    claim_ref: str
    input_kind: ClaimInputKind
    assessment: ClaimSourceAuthorityFit
    claim_type_status: ClaimTypeSignalStatus
    basis_refs: tuple[str, ...]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceReliabilityAssessment:
    schema_version: str
    source_ref: str
    snapshot_ref: str | None
    claim_source_fit: ClaimSourceFitAssessment
    dimensions: tuple[ReliabilityDimensionAssessment, ...]
    reason_codes: tuple[str, ...]

    def dimension(self, key: ReliabilityDimension) -> ReliabilityDimensionAssessment:
        """Return one dimension without exposing an aggregate score."""

        for item in self.dimensions:
            if item.dimension is key:
                return item
        raise KeyError(key)


def assess_source_reliability(
    value: SourceReliabilityInput,
) -> SourceReliabilityAssessment:
    """Build an immutable, evidence-backed assessment without inference."""

    fit_input = value.claim_source_fit
    reasons: list[str] = []

    identity = value.identity_clarity
    identity_refs = _refs_or_unknown(identity.value, value.identity_basis_refs)
    if identity.value != IdentityClarity.UNKNOWN.value and not identity_refs:
        identity = IdentityClarity.UNKNOWN
        reasons.append("IDENTITY_BASIS_MISSING")

    originality = value.originality
    originality_refs = _refs_or_unknown(originality.value, value.originality_basis_refs)
    if originality.value != Originality.UNKNOWN.value and not originality_refs:
        originality = Originality.UNKNOWN
        reasons.append("ORIGINALITY_BASIS_MISSING")
    if originality in {
        Originality.DERIVED_FROM_KNOWN_SOURCE,
        Originality.SHARED_ORIGIN_KNOWN,
    } and not (value.original_source_ref or value.shared_origin_ref):
        originality = Originality.UNKNOWN
        originality_refs = ()
        reasons.append("ORIGINALITY_LINEAGE_REF_MISSING")

    lineage_value = _lineage_value(value)
    lineage_refs = _lineage_refs(value)
    if value.original_source_ref and value.shared_origin_ref:
        reasons.append("LINEAGE_CONFLICTING_PROVENANCE")
    if value.original_source_ref or value.shared_origin_ref:
        originality = Originality(lineage_value)
        originality_refs = lineage_refs
    if lineage_value is not Originality.UNKNOWN and not lineage_refs:
        lineage_value = Originality.UNKNOWN
        reasons.append("LINEAGE_BASIS_MISSING")

    relation = fit_input.source_relation
    relation_is_supported = bool(fit_input.relation_basis_refs)
    directness = Directness.UNKNOWN
    if relation_is_supported:
        if relation in {
            SourcePropositionRelation.DIRECT_OBSERVATION,
            SourcePropositionRelation.FORMAL_AUTHORITY,
            SourcePropositionRelation.SELF_REPORTED,
        }:
            directness = Directness.DIRECT
        elif relation in {
            SourcePropositionRelation.SECONDARY_REPORT,
            SourcePropositionRelation.INDIRECT,
        }:
            directness = Directness.INDIRECT
    elif relation is not SourcePropositionRelation.UNKNOWN:
        reasons.append("RELATION_BASIS_MISSING")

    authority_fit = _authority_fit(fit_input)
    if authority_fit is ClaimSourceAuthorityFit.UNRESOLVED:
        reasons.append("AUTHORITY_UNRESOLVED")

    expertise = _fact_value(value.expertise_fit)
    self_report = _fact_value(value.self_report)
    transparency = _fact_value(value.transparency)
    conflict = _fact_value(value.conflict_of_interest)

    if (
        value.proposition_temporal_sensitivity.status is FactStatus.NOT_SUPPORTED
        and value.proposition_temporal_sensitivity.basis_refs
    ):
        recency = RecencyFit.NOT_APPLICABLE
        recency_refs = value.proposition_temporal_sensitivity.basis_refs
    elif (
        value.proposition_temporal_sensitivity.status is FactStatus.SUPPORTED
        and value.proposition_temporal_sensitivity.basis_refs
        and value.recency_fit is not RecencyFit.UNKNOWN
        and value.recency_basis_refs
    ):
        recency = value.recency_fit
        recency_refs = value.recency_basis_refs
    else:
        recency = RecencyFit.UNKNOWN
        recency_refs = ()
        if value.recency_fit is not RecencyFit.UNKNOWN and not value.recency_basis_refs:
            reasons.append("RECENCY_BASIS_MISSING")

    fit_status, fit_reasons = _assess_claim_source_fit(fit_input)
    reasons.extend(fit_reasons)
    claim_type_status = (
        ClaimTypeSignalStatus.LEGACY_SIGNAL_ONLY
        if fit_input.input_kind is ClaimInputKind.LEGACY_CLAIM_BRIDGE
        and fit_input.legacy_claim_type_signal
        else ClaimTypeSignalStatus.CLAIM_TYPE_UNKNOWN
    )

    dimensions = (
        _dimension(
            ReliabilityDimension.IDENTITY_CLARITY,
            identity,
            identity_refs,
            () if identity is not IdentityClarity.UNKNOWN else ("SOURCE_IDENTITY_UNKNOWN",),
        ),
        _dimension(
            ReliabilityDimension.ORIGINALITY,
            originality,
            originality_refs,
            () if originality is not Originality.UNKNOWN else ("ORIGINALITY_UNKNOWN",),
        ),
        _dimension(
            ReliabilityDimension.DIRECTNESS,
            directness,
            fit_input.relation_basis_refs if directness is not Directness.UNKNOWN else (),
            () if directness is not Directness.UNKNOWN else ("DIRECTNESS_UNKNOWN",),
        ),
        _dimension(
            ReliabilityDimension.AUTHORITY_FIT,
            authority_fit,
            _authority_refs(fit_input)
            if authority_fit is not ClaimSourceAuthorityFit.UNRESOLVED
            else (),
            ()
            if authority_fit is not ClaimSourceAuthorityFit.UNRESOLVED
            else ("AUTHORITY_UNRESOLVED",),
            materiality=True,
        ),
        _dimension(
            ReliabilityDimension.EXPERTISE_FIT,
            expertise,
            _effective_refs(value.expertise_fit),
            _fact_reason(value.expertise_fit, "EXPERTISE_BASIS_MISSING"),
        ),
        _dimension(
            ReliabilityDimension.SELF_REPORT_STATUS,
            self_report,
            _effective_refs(value.self_report),
            _fact_reason(value.self_report, "SELF_REPORT_BASIS_MISSING"),
        ),
        _dimension(
            ReliabilityDimension.TRANSPARENCY,
            transparency,
            _effective_refs(value.transparency),
            _fact_reason(value.transparency, "TRANSPARENCY_BASIS_MISSING"),
        ),
        _dimension(
            ReliabilityDimension.RECENCY_FIT,
            recency,
            recency_refs,
            () if recency is not RecencyFit.UNKNOWN else ("RECENCY_UNKNOWN",),
            materiality=(
                True
                if (
                    value.proposition_temporal_sensitivity.status is FactStatus.SUPPORTED
                    and value.proposition_temporal_sensitivity.basis_refs
                )
                else False
                if recency is RecencyFit.NOT_APPLICABLE
                else None
            ),
        ),
        _dimension(
            ReliabilityDimension.CONFLICT_OF_INTEREST,
            conflict,
            _effective_refs(value.conflict_of_interest),
            _fact_reason(
                value.conflict_of_interest,
                "CONFLICT_OF_INTEREST_BASIS_MISSING",
            ),
        ),
        _dimension(
            ReliabilityDimension.LINEAGE_STATUS,
            lineage_value,
            lineage_refs,
            () if lineage_value is not Originality.UNKNOWN else ("LINEAGE_UNKNOWN",),
        ),
    )
    return SourceReliabilityAssessment(
        schema_version=SCHEMA_VERSION,
        source_ref=value.source_ref,
        snapshot_ref=value.snapshot_ref,
        claim_source_fit=ClaimSourceFitAssessment(
            claim_ref=fit_input.claim_ref,
            input_kind=fit_input.input_kind,
            assessment=fit_status,
            claim_type_status=claim_type_status,
            basis_refs=_authority_refs(fit_input),
            reason_codes=tuple(fit_reasons),
        ),
        dimensions=dimensions,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def _refs_or_unknown(value: str, refs: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(set(refs))) if value != "UNKNOWN" and refs else ()


def _fact_value(value: ExplicitFact) -> FactStatus:
    if value.status is FactStatus.UNKNOWN:
        return FactStatus.UNKNOWN
    return value.status if value.basis_refs else FactStatus.UNKNOWN


def _effective_refs(value: ExplicitFact) -> tuple[str, ...]:
    return (
        tuple(sorted(set(value.basis_refs)))
        if value.status is not FactStatus.UNKNOWN and value.basis_refs
        else ()
    )


def _fact_reason(value: ExplicitFact, missing_basis_reason: str) -> tuple[str, ...]:
    if value.status is FactStatus.UNKNOWN:
        return ("FACT_UNKNOWN",)
    if not value.basis_refs:
        return (missing_basis_reason,)
    return ()


def _authority_refs(value: ClaimSourceFitInput) -> tuple[str, ...]:
    if value.source_relation is SourcePropositionRelation.FORMAL_AUTHORITY:
        return tuple(
            sorted(set((*value.relation_basis_refs, *value.authority_scope_match.basis_refs)))
        )
    return tuple(sorted(set(value.relation_basis_refs)))


def _authority_fit(value: ClaimSourceFitInput) -> ClaimSourceAuthorityFit:
    if not value.relation_basis_refs:
        return ClaimSourceAuthorityFit.UNRESOLVED
    if value.source_relation is SourcePropositionRelation.SELF_REPORTED:
        return ClaimSourceAuthorityFit.DIRECT_BUT_SELF_REPORTED
    if value.source_relation is SourcePropositionRelation.SECONDARY_REPORT:
        return ClaimSourceAuthorityFit.SECONDARY_SUPPORT
    if value.source_relation is SourcePropositionRelation.INDIRECT:
        return ClaimSourceAuthorityFit.INDIRECT
    if value.source_relation is SourcePropositionRelation.FORMAL_AUTHORITY:
        scope = value.authority_scope_match
        if scope.status is FactStatus.SUPPORTED and scope.basis_refs:
            return ClaimSourceAuthorityFit.DIRECTLY_AUTHORITATIVE
    return ClaimSourceAuthorityFit.UNRESOLVED


def _assess_claim_source_fit(
    value: ClaimSourceFitInput,
) -> tuple[ClaimSourceAuthorityFit, tuple[str, ...]]:
    assessment = _authority_fit(value)
    reasons: list[str] = []
    if assessment is ClaimSourceAuthorityFit.UNRESOLVED:
        reasons.append("AUTHORITY_UNRESOLVED")
    if (
        value.source_relation is SourcePropositionRelation.FORMAL_AUTHORITY
        and value.authority_scope_match.status is FactStatus.NOT_SUPPORTED
        and value.authority_scope_match.basis_refs
    ):
        reasons.append("AUTHORITY_SCOPE_NOT_MATCHED")
    if (
        value.source_relation is not SourcePropositionRelation.UNKNOWN
        and not value.relation_basis_refs
    ):
        reasons.append("RELATION_BASIS_MISSING")
    return assessment, tuple(dict.fromkeys(reasons))


def _lineage_value(value: SourceReliabilityInput) -> Originality:
    if value.original_source_ref and value.shared_origin_ref:
        return Originality.UNKNOWN
    if value.shared_origin_ref:
        return Originality.SHARED_ORIGIN_KNOWN
    if value.original_source_ref:
        if value.original_source_ref == value.source_ref:
            return Originality.UNKNOWN
        return Originality.DERIVED_FROM_KNOWN_SOURCE
    if value.originality is Originality.ORIGINAL and value.originality_basis_refs:
        return Originality.ORIGINAL
    return Originality.UNKNOWN


def _lineage_refs(value: SourceReliabilityInput) -> tuple[str, ...]:
    refs = tuple(
        ref
        for ref in (
            value.original_source_ref,
            value.shared_origin_ref,
            *(value.originality_basis_refs if value.originality is Originality.ORIGINAL else ()),
        )
        if ref
    )
    return tuple(sorted(set(refs)))


def _dimension(
    key: ReliabilityDimension,
    value: StrEnum,
    basis_refs: tuple[str, ...] = (),
    reason_codes: tuple[str, ...] = (),
    *,
    materiality: bool | None = None,
) -> ReliabilityDimensionAssessment:
    return ReliabilityDimensionAssessment(
        dimension=key,
        value=value,
        basis_refs=basis_refs,
        reason_codes=reason_codes,
        materiality=materiality,
    )
