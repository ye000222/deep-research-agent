"""Deterministic, reference-only aggregation of existing evidence assessments.

This domain contract observes V2.3 formation/document/source assessments and
existing grounding/acceptance observations. It does not decide acceptance,
truth, verification, or source reliability, and it has no persistence or I/O.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.domain.atomic_proposition import PropositionEvidenceLink
from app.domain.document_evidence_context import (
    CanonicalEvidenceSpan,
    ContextWindow,
    DocumentCompleteness,
    ProcessingChunk,
    ReadableDocument,
    RelevantSpan,
)
from app.domain.evidence_formation_diagnostics import (
    EvidenceFormationDiagnosticEvent,
    EvidenceFormationOutcome,
    EvidenceFormationStage,
    EvidenceFormationTrace,
)
from app.domain.source_reliability_assessment import (
    ClaimSourceFitAssessment,
    ReliabilityDimension,
    SourceReliabilityAssessment,
)
from app.domain.verification_assertion import AssertionGroundingUsability

EVIDENCE_RELIABILITY_ASSESSMENT_VERSION = "evidence-reliability-assessment.v1"


class ReliabilityAssessmentStatus(StrEnum):
    ASSESSED = "ASSESSED"
    ASSESSED_WITH_LIMITATIONS = "ASSESSED_WITH_LIMITATIONS"
    UNKNOWN = "UNKNOWN"


class ReliabilityReasonCode(StrEnum):
    FORMATION_AVAILABLE = "FORMATION_AVAILABLE"
    FORMATION_INCOMPLETE = "FORMATION_INCOMPLETE"
    FORMATION_MISSING = "FORMATION_MISSING"
    GROUNDING_AVAILABLE = "GROUNDING_AVAILABLE"
    GROUNDING_PARTIAL = "GROUNDING_PARTIAL"
    GROUNDING_MISSING = "GROUNDING_MISSING"
    SOURCE_ASSESSMENT_AVAILABLE = "SOURCE_ASSESSMENT_AVAILABLE"
    SOURCE_IDENTITY_UNKNOWN = "SOURCE_IDENTITY_UNKNOWN"
    SOURCE_FIT_UNKNOWN = "SOURCE_FIT_UNKNOWN"
    V22_GROUNDING_USABILITY_UNKNOWN = "V22_GROUNDING_USABILITY_UNKNOWN"
    V22_GROUNDING_USABILITY_INSUFFICIENT = "V22_GROUNDING_USABILITY_INSUFFICIENT"
    LINEAGE_UNKNOWN = "LINEAGE_UNKNOWN"
    FULLY_ASSESSED = "FULLY_ASSESSED"
    ASSESSED_WITH_LIMITATIONS = "ASSESSED_WITH_LIMITATIONS"
    INSUFFICIENT_INPUT = "INSUFFICIENT_INPUT"


class KnownLimitation(StrEnum):
    FORMATION_INCOMPLETE = "FORMATION_INCOMPLETE"
    ACCEPTANCE_UNKNOWN = "ACCEPTANCE_UNKNOWN"
    ACCEPTANCE_REFERENCE_MISSING = "ACCEPTANCE_REFERENCE_MISSING"
    DOCUMENT_COMPLETENESS_UNKNOWN = "DOCUMENT_COMPLETENESS_UNKNOWN"
    DOCUMENT_PARTIAL = "DOCUMENT_PARTIAL"
    CONTEXT_INCOMPLETE = "CONTEXT_INCOMPLETE"
    GROUNDING_PARTIAL = "GROUNDING_PARTIAL"
    SOURCE_ASSESSMENT_MISSING = "SOURCE_ASSESSMENT_MISSING"
    SOURCE_IDENTITY_UNKNOWN = "SOURCE_IDENTITY_UNKNOWN"
    SOURCE_ORIGINALITY_UNKNOWN = "SOURCE_ORIGINALITY_UNKNOWN"
    SOURCE_DIRECTNESS_UNKNOWN = "SOURCE_DIRECTNESS_UNKNOWN"
    SOURCE_AUTHORITY_UNRESOLVED = "SOURCE_AUTHORITY_UNRESOLVED"
    SOURCE_EXPERTISE_UNKNOWN = "SOURCE_EXPERTISE_UNKNOWN"
    SOURCE_SELF_REPORT_UNKNOWN = "SOURCE_SELF_REPORT_UNKNOWN"
    SOURCE_TRANSPARENCY_UNKNOWN = "SOURCE_TRANSPARENCY_UNKNOWN"
    SOURCE_RECENCY_UNKNOWN = "SOURCE_RECENCY_UNKNOWN"
    CONFLICT_OF_INTEREST_UNKNOWN = "CONFLICT_OF_INTEREST_UNKNOWN"
    LINEAGE_UNKNOWN = "LINEAGE_UNKNOWN"
    SOURCE_FIT_UNKNOWN = "SOURCE_FIT_UNKNOWN"
    V22_GROUNDING_UNRESOLVED = "V22_GROUNDING_UNRESOLVED"


class ObservedState(StrEnum):
    YES = "YES"
    NO = "NO"
    UNKNOWN = "UNKNOWN"


class QuoteExactness(StrEnum):
    EXACT = "EXACT"
    NOT_EXACT = "NOT_EXACT"
    UNKNOWN = "UNKNOWN"


class ContextAlignment(StrEnum):
    ALIGNED = "ALIGNED"
    MISALIGNED = "MISALIGNED"
    UNKNOWN = "UNKNOWN"


class AcceptanceObservation(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


class ExistingGroundingObservation(BaseModel):
    """A reference and already-computed grounding facts; never runs grounding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grounding_ref: str = Field(min_length=1, max_length=256)
    source_span_available: ObservedState
    quote_exactness: QuoteExactness
    context_alignment: ContextAlignment
    v22_grounding_usability: AssertionGroundingUsability | None = None
    proposition_evidence_link: PropositionEvidenceLink | None = None


class EvidenceFormationAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    formation_trace_ref: str
    document_available: ObservedState
    candidate_formed: ObservedState
    span_located: ObservedState
    quote_verified: ObservedState
    context_available: ObservedState
    extraction_status: EvidenceFormationOutcome
    acceptance_observation: AcceptanceObservation
    persistence_observation: ObservedState


class GroundingAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    grounding_ref: str
    source_span_available: ObservedState
    canonical_location_available: ObservedState
    quote_exactness: QuoteExactness
    context_alignment: ContextAlignment
    document_ref: str | None = None
    snapshot_ref: str | None = None
    document_span_ref: str | None = None
    context_ref: str | None = None
    relevant_span_ref: str | None = None
    processing_chunk_refs: tuple[str, ...] = ()
    proposition_evidence_link_ref: str | None = None
    v22_grounding_usability: AssertionGroundingUsability | None = None


class EvidenceReliabilityAssessment(BaseModel):
    """Immutable aggregation output. Contains provenance references, not text."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = EVIDENCE_RELIABILITY_ASSESSMENT_VERSION
    assessment_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_ref: str = Field(min_length=1, max_length=256)
    assessment_status: ReliabilityAssessmentStatus
    formation: EvidenceFormationAssessment | None = None
    grounding: GroundingAssessment | None = None
    source_assessment_ref: str | None = None
    claim_source_fit_ref: str | None = None
    acceptance_ref: str | None = None
    document_completeness: DocumentCompleteness | None = None
    known_limitations: tuple[KnownLimitation, ...] = ()
    reason_codes: tuple[ReliabilityReasonCode, ...]


class OptionalVerificationReliabilityInput(BaseModel):
    """Versioned future adapter shape; intentionally not wired into V2.2."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assessment_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str = EVIDENCE_RELIABILITY_ASSESSMENT_VERSION
    assessment_status: ReliabilityAssessmentStatus
    known_limitations: tuple[KnownLimitation, ...]


def aggregate_evidence_reliability(
    *,
    evidence_ref: str,
    formation_trace: EvidenceFormationTrace | None,
    grounding: ExistingGroundingObservation | None,
    document: ReadableDocument | None,
    document_span: CanonicalEvidenceSpan | None,
    context_window: ContextWindow | None,
    relevant_span: RelevantSpan | None,
    processing_chunks: tuple[ProcessingChunk, ...] = (),
    source_assessment: SourceReliabilityAssessment | None,
    claim_source_fit: ClaimSourceFitAssessment | None,
    acceptance_ref: str | None,
) -> EvidenceReliabilityAssessment:
    """Aggregate explicit inputs without changing or retaining their content."""

    limitations: set[KnownLimitation] = set()
    reasons: set[ReliabilityReasonCode] = set()
    if formation_trace is None:
        formation = None
        reasons.add(ReliabilityReasonCode.FORMATION_MISSING)
    else:
        formation = _formation_projection(
            formation_trace,
            evidence_ref=evidence_ref,
            document_available=document is not None,
            span_located=document_span is not None,
            context_available=_context_covers_span(context_window, document_span),
        )
        reasons.add(
            ReliabilityReasonCode.FORMATION_AVAILABLE
            if formation.acceptance_observation is not AcceptanceObservation.UNKNOWN
            and formation.extraction_status is EvidenceFormationOutcome.SUCCEEDED
            and formation.candidate_formed is ObservedState.YES
            else ReliabilityReasonCode.FORMATION_INCOMPLETE
        )
        if formation.acceptance_observation is AcceptanceObservation.UNKNOWN:
            limitations.add(KnownLimitation.ACCEPTANCE_UNKNOWN)
            limitations.add(KnownLimitation.FORMATION_INCOMPLETE)
        if (
            formation.extraction_status is not EvidenceFormationOutcome.SUCCEEDED
            or formation.candidate_formed is not ObservedState.YES
        ):
            limitations.add(KnownLimitation.FORMATION_INCOMPLETE)

    grounded = _grounding_projection(
        grounding=grounding,
        document=document,
        document_span=document_span,
        context_window=context_window,
        relevant_span=relevant_span,
        processing_chunks=processing_chunks,
    )
    if grounded is None:
        reasons.add(ReliabilityReasonCode.GROUNDING_MISSING)
    elif (
        grounded.source_span_available is ObservedState.YES
        and grounded.canonical_location_available is ObservedState.YES
        and grounded.quote_exactness is QuoteExactness.EXACT
        and grounded.context_alignment is ContextAlignment.ALIGNED
    ):
        reasons.add(ReliabilityReasonCode.GROUNDING_AVAILABLE)
    else:
        reasons.add(ReliabilityReasonCode.GROUNDING_PARTIAL)
        limitations.add(KnownLimitation.GROUNDING_PARTIAL)
    if (
        grounded is not None
        and grounded.v22_grounding_usability is not None
        and grounded.v22_grounding_usability is not AssertionGroundingUsability.SUFFICIENT
    ):
        limitations.add(KnownLimitation.V22_GROUNDING_UNRESOLVED)
        reasons.add(
            ReliabilityReasonCode.V22_GROUNDING_USABILITY_UNKNOWN
            if grounded.v22_grounding_usability is AssertionGroundingUsability.UNKNOWN
            else ReliabilityReasonCode.V22_GROUNDING_USABILITY_INSUFFICIENT
        )

    if document is not None:
        if document.completeness is DocumentCompleteness.PARTIAL:
            limitations.add(KnownLimitation.DOCUMENT_PARTIAL)
        elif document.completeness is DocumentCompleteness.UNKNOWN:
            limitations.add(KnownLimitation.DOCUMENT_COMPLETENESS_UNKNOWN)
    if context_window is None or not _context_covers_span(context_window, document_span):
        limitations.add(KnownLimitation.CONTEXT_INCOMPLETE)
    if acceptance_ref is None:
        limitations.add(KnownLimitation.ACCEPTANCE_REFERENCE_MISSING)

    _collect_source_limitations(source_assessment, claim_source_fit, limitations, reasons)

    missing_core = (
        formation_trace is None
        or grounding is None
        or document_span is None
        or grounded is None
    )
    if missing_core:
        status = ReliabilityAssessmentStatus.UNKNOWN
        reasons.add(ReliabilityReasonCode.INSUFFICIENT_INPUT)
    elif limitations:
        status = ReliabilityAssessmentStatus.ASSESSED_WITH_LIMITATIONS
        reasons.add(ReliabilityReasonCode.ASSESSED_WITH_LIMITATIONS)
    else:
        status = ReliabilityAssessmentStatus.ASSESSED
        reasons.add(ReliabilityReasonCode.FULLY_ASSESSED)

    formation_ref = formation.formation_trace_ref if formation else None
    grounding_ref = grounded.grounding_ref if grounded else None
    identity = {
        "schema_version": EVIDENCE_RELIABILITY_ASSESSMENT_VERSION,
        "evidence_ref": evidence_ref,
        "formation_trace_ref": formation_ref,
        "grounding_ref": grounding_ref,
        "source_assessment_ref": _source_ref(source_assessment)
        if source_assessment is not None
        else None,
        "source_ref": source_assessment.source_ref if source_assessment else None,
        "claim_source_fit_ref": _claim_fit_ref(claim_source_fit)
        if claim_source_fit is not None
        else None,
        "acceptance_ref": acceptance_ref,
        "status": status.value,
        "limitations": sorted(item.value for item in limitations),
        "reasons": sorted(item.value for item in reasons),
    }
    assessment_ref = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return EvidenceReliabilityAssessment(
        assessment_ref=assessment_ref,
        evidence_ref=evidence_ref,
        assessment_status=status,
        formation=formation,
        grounding=grounded,
        source_assessment_ref=(
            _source_ref(source_assessment) if source_assessment is not None else None
        ),
        claim_source_fit_ref=(
            _claim_fit_ref(claim_source_fit) if claim_source_fit is not None else None
        ),
        acceptance_ref=acceptance_ref,
        document_completeness=document.completeness if document else None,
        known_limitations=tuple(sorted(limitations, key=lambda item: item.value)),
        reason_codes=tuple(sorted(reasons, key=lambda item: item.value)),
    )


def as_optional_verification_input(
    assessment: EvidenceReliabilityAssessment,
) -> OptionalVerificationReliabilityInput:
    """Expose only the assessment reference/status for a future optional adapter."""

    return OptionalVerificationReliabilityInput(
        assessment_ref=assessment.assessment_ref,
        assessment_status=assessment.assessment_status,
        known_limitations=assessment.known_limitations,
    )


def _formation_projection(
    trace: EvidenceFormationTrace,
    *,
    evidence_ref: str,
    document_available: bool,
    span_located: bool,
    context_available: bool,
) -> EvidenceFormationAssessment:
    events = tuple(event for event in trace.events if event.evidence_ref in (None, evidence_ref))
    extraction = _latest_outcome(events, EvidenceFormationStage.EXTRACTION)
    candidate = _latest_outcome(events, EvidenceFormationStage.CANDIDATE_FORMATION)
    acceptance = _latest_outcome(events, EvidenceFormationStage.ACCEPTANCE)
    persistence = _latest_outcome(events, EvidenceFormationStage.PERSISTENCE)
    quote = _latest_outcome(events, EvidenceFormationStage.QUOTE_LOCALIZATION)
    return EvidenceFormationAssessment(
        formation_trace_ref=_trace_ref(trace),
        document_available=_observed(document_available),
        candidate_formed=(
            ObservedState.YES
            if candidate
            in {EvidenceFormationOutcome.SUCCEEDED, EvidenceFormationOutcome.ACCEPTED}
            else ObservedState.NO
            if candidate in {EvidenceFormationOutcome.FAILED, EvidenceFormationOutcome.EMPTY}
            else ObservedState.UNKNOWN
        ),
        span_located=_observed(span_located),
        quote_verified=(
            ObservedState.YES
            if quote is EvidenceFormationOutcome.SUCCEEDED
            else ObservedState.NO
            if quote in {EvidenceFormationOutcome.FAILED, EvidenceFormationOutcome.REJECTED}
            else ObservedState.UNKNOWN
        ),
        context_available=_observed(context_available),
        extraction_status=extraction,
        acceptance_observation=(
            AcceptanceObservation.ACCEPTED
            if acceptance is EvidenceFormationOutcome.ACCEPTED
            else AcceptanceObservation.REJECTED
            if acceptance is EvidenceFormationOutcome.REJECTED
            else AcceptanceObservation.UNKNOWN
        ),
        persistence_observation=(
            ObservedState.YES
            if persistence
            in {EvidenceFormationOutcome.SUCCEEDED, EvidenceFormationOutcome.ACCEPTED}
            else ObservedState.NO
            if persistence in {EvidenceFormationOutcome.FAILED, EvidenceFormationOutcome.REJECTED}
            else ObservedState.UNKNOWN
        ),
    )


def _grounding_projection(
    *,
    grounding: ExistingGroundingObservation | None,
    document: ReadableDocument | None,
    document_span: CanonicalEvidenceSpan | None,
    context_window: ContextWindow | None,
    relevant_span: RelevantSpan | None,
    processing_chunks: tuple[ProcessingChunk, ...],
) -> GroundingAssessment | None:
    if grounding is None:
        return None
    location = (
        ObservedState.YES
        if document_span is not None
        and document_span.source_locator.locator_type.value != "UNKNOWN"
        and document is not None
        and document_span.document_ref == document.document_ref
        and document_span.source_ref == document.source_ref
        and document_span.snapshot_ref == document.snapshot_ref
        else ObservedState.NO
        if document_span is not None
        else ObservedState.UNKNOWN
    )
    relevant_ref = None
    if relevant_span is not None:
        relevant_ref = _sha256(
            f"{relevant_span.document_ref}:{relevant_span.canonical_start}:"
            f"{relevant_span.canonical_end}"
        )
    chunk_refs = tuple(sorted({chunk.chunk_ref for chunk in processing_chunks}))
    return GroundingAssessment(
        grounding_ref=grounding.grounding_ref,
        source_span_available=grounding.source_span_available,
        canonical_location_available=location,
        quote_exactness=grounding.quote_exactness,
        context_alignment=grounding.context_alignment,
        document_ref=document.document_ref if document else None,
        snapshot_ref=document.snapshot_ref if document else None,
        document_span_ref=document_span.span_ref if document_span else None,
        context_ref=context_window.context_ref if context_window else None,
        relevant_span_ref=relevant_ref,
        processing_chunk_refs=chunk_refs,
        proposition_evidence_link_ref=(
            _proposition_evidence_link_ref(grounding.proposition_evidence_link)
            if grounding.proposition_evidence_link is not None
            else None
        ),
        v22_grounding_usability=grounding.v22_grounding_usability,
    )


def _collect_source_limitations(
    source: SourceReliabilityAssessment | None,
    fit: ClaimSourceFitAssessment | None,
    limitations: set[KnownLimitation],
    reasons: set[ReliabilityReasonCode],
) -> None:
    if source is None:
        limitations.add(KnownLimitation.SOURCE_ASSESSMENT_MISSING)
        reasons.add(ReliabilityReasonCode.SOURCE_FIT_UNKNOWN)
        return
    reasons.add(ReliabilityReasonCode.SOURCE_ASSESSMENT_AVAILABLE)
    if fit is None or fit.assessment.value == "UNRESOLVED":
        limitations.add(KnownLimitation.SOURCE_FIT_UNKNOWN)
        reasons.add(ReliabilityReasonCode.SOURCE_FIT_UNKNOWN)
    dimensions = {item.dimension: item.value.value for item in source.dimensions}
    unknown_dimension_map = {
        ReliabilityDimension.IDENTITY_CLARITY: (
            "UNKNOWN",
            KnownLimitation.SOURCE_IDENTITY_UNKNOWN,
            ReliabilityReasonCode.SOURCE_IDENTITY_UNKNOWN,
        ),
        ReliabilityDimension.ORIGINALITY: (
            "UNKNOWN",
            KnownLimitation.SOURCE_ORIGINALITY_UNKNOWN,
            None,
        ),
        ReliabilityDimension.DIRECTNESS: (
            "UNKNOWN",
            KnownLimitation.SOURCE_DIRECTNESS_UNKNOWN,
            None,
        ),
        ReliabilityDimension.AUTHORITY_FIT: (
            "UNRESOLVED",
            KnownLimitation.SOURCE_AUTHORITY_UNRESOLVED,
            None,
        ),
        ReliabilityDimension.EXPERTISE_FIT: (
            "UNKNOWN",
            KnownLimitation.SOURCE_EXPERTISE_UNKNOWN,
            None,
        ),
        ReliabilityDimension.SELF_REPORT_STATUS: (
            "UNKNOWN",
            KnownLimitation.SOURCE_SELF_REPORT_UNKNOWN,
            None,
        ),
        ReliabilityDimension.TRANSPARENCY: (
            "UNKNOWN",
            KnownLimitation.SOURCE_TRANSPARENCY_UNKNOWN,
            None,
        ),
        ReliabilityDimension.RECENCY_FIT: (
            "UNKNOWN",
            KnownLimitation.SOURCE_RECENCY_UNKNOWN,
            None,
        ),
        ReliabilityDimension.CONFLICT_OF_INTEREST: (
            "UNKNOWN",
            KnownLimitation.CONFLICT_OF_INTEREST_UNKNOWN,
            None,
        ),
        ReliabilityDimension.LINEAGE_STATUS: (
            "UNKNOWN",
            KnownLimitation.LINEAGE_UNKNOWN,
            ReliabilityReasonCode.LINEAGE_UNKNOWN,
        ),
    }
    for dimension, (unknown_value, limitation, reason) in unknown_dimension_map.items():
        if dimensions.get(dimension, "UNKNOWN") == unknown_value:
            limitations.add(limitation)
            if reason is not None:
                reasons.add(reason)


def _latest_outcome(
    events: tuple[EvidenceFormationDiagnosticEvent, ...],
    stage: EvidenceFormationStage,
) -> EvidenceFormationOutcome:
    matching = [event for event in events if event.stage is stage]
    if not matching:
        return EvidenceFormationOutcome.NOT_OBSERVED
    return matching[-1].outcome


def _context_covers_span(
    context: ContextWindow | None,
    span: CanonicalEvidenceSpan | None,
) -> bool:
    return bool(
        context is not None
        and span is not None
        and context.document_ref == span.document_ref
        and context.evidence_span_ref == span.span_ref
        and context.window_start <= span.char_start
        and context.window_end >= span.char_end
    )


def _observed(value: bool) -> ObservedState:
    return ObservedState.YES if value else ObservedState.NO


def _trace_ref(trace: EvidenceFormationTrace) -> str:
    return _sha256("|".join(event.event_ref for event in trace.events))


def _source_ref(source: SourceReliabilityAssessment) -> str:
    return _canonical_dataclass_hash(source)


def _claim_fit_ref(fit: ClaimSourceFitAssessment) -> str:
    return _canonical_dataclass_hash(fit)


def _canonical_dataclass_hash(
    value: SourceReliabilityAssessment | ClaimSourceFitAssessment,
) -> str:
    payload = json.dumps(
        asdict(value),
        default=lambda item: item.value if isinstance(item, StrEnum) else str(item),
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256(payload)


def _proposition_evidence_link_ref(link: PropositionEvidenceLink) -> str:
    """Hash link identity without serializing quote or owner metadata."""

    payload = {
        "proposition_id": str(link.proposition_id),
        "evidence_id": str(link.evidence_id),
        "claim_id": str(link.claim_id),
        "source_id": str(link.source_id),
        "run_id": str(link.run_id),
        "relation": link.relation.value,
        "span_start": link.span_start,
        "span_end": link.span_end,
        "method": link.method,
        "question_id": link.question_id,
        "dimension_key": link.dimension_key,
    }
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
