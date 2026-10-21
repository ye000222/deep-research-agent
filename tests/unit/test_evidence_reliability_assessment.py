from __future__ import annotations

from typing import TypedDict
from uuid import UUID

from app.domain.atomic_proposition import EvidencePropositionRelation, PropositionEvidenceLink
from app.domain.document_evidence_context import (
    CanonicalEvidenceSpan,
    ContextWindow,
    DocumentCompleteness,
    DocumentSourceSegment,
    MappedTextWindow,
    ProcessingChunk,
    ReadableDocument,
    RelevantSpan,
    SourceLocator,
    SourceLocatorType,
    build_context_window,
    build_readable_document,
    locate_canonical_evidence_span,
)
from app.domain.evidence_formation_diagnostics import (
    EvidenceFormationDiagnosticEvent,
    EvidenceFormationOutcome,
    EvidenceFormationStage,
    EvidenceFormationTrace,
)
from app.domain.evidence_reliability_assessment import (
    AcceptanceObservation,
    ContextAlignment,
    ExistingGroundingObservation,
    KnownLimitation,
    QuoteExactness,
    ReliabilityAssessmentStatus,
    ReliabilityReasonCode,
    aggregate_evidence_reliability,
    as_optional_verification_input,
)
from app.domain.source_reliability_assessment import (
    ClaimInputKind,
    ClaimSourceFitAssessment,
    ClaimSourceFitInput,
    ExplicitFact,
    FactStatus,
    IdentityClarity,
    Originality,
    RecencyFit,
    SourcePropositionRelation,
    SourceReliabilityAssessment,
    SourceReliabilityInput,
    assess_source_reliability,
)
from app.domain.verification_assertion import AssertionGroundingUsability


class AggregationInputs(TypedDict):
    evidence_ref: str
    formation_trace: EvidenceFormationTrace | None
    grounding: ExistingGroundingObservation | None
    document: ReadableDocument
    document_span: CanonicalEvidenceSpan | None
    context_window: ContextWindow | None
    relevant_span: RelevantSpan | None
    processing_chunks: tuple[ProcessingChunk, ...]
    source_assessment: SourceReliabilityAssessment | None
    claim_source_fit: ClaimSourceFitAssessment
    acceptance_ref: str | None


def _trace(*, complete: bool = True) -> EvidenceFormationTrace:
    evidence_ref = "evidence:one"
    stages = (
        (EvidenceFormationStage.EXTRACTION, EvidenceFormationOutcome.SUCCEEDED),
        (EvidenceFormationStage.CANDIDATE_FORMATION, EvidenceFormationOutcome.SUCCEEDED),
        (
            EvidenceFormationStage.ACCEPTANCE,
            (
                EvidenceFormationOutcome.ACCEPTED
                if complete
                else EvidenceFormationOutcome.NOT_OBSERVED
            ),
        ),
        (EvidenceFormationStage.QUOTE_LOCALIZATION, EvidenceFormationOutcome.SUCCEEDED),
        (EvidenceFormationStage.PERSISTENCE, EvidenceFormationOutcome.SUCCEEDED),
    )
    events = tuple(
        EvidenceFormationDiagnosticEvent.create(
            run_ref="run:one",
            target_ref="claim:one",
            evidence_ref=evidence_ref,
            stage=stage,
            outcome=outcome,
        )
        for stage, outcome in stages
    )
    return EvidenceFormationTrace(
        run_ref="run:one", target_ref="claim:one", events=events
    )


def _source(*, unknown_lineage: bool = False) -> SourceReliabilityAssessment:
    return assess_source_reliability(
        SourceReliabilityInput(
            source_ref="source:one",
            snapshot_ref="snapshot:one",
            claim_source_fit=ClaimSourceFitInput(
                claim_ref="claim:one",
                input_kind=ClaimInputKind.PROPOSITION,
                source_relation=SourcePropositionRelation.FORMAL_AUTHORITY,
                relation_basis_refs=("relation:evidence",),
                authority_scope_match=ExplicitFact(FactStatus.SUPPORTED, ("scope:evidence",)),
            ),
            identity_clarity=IdentityClarity.KNOWN,
            identity_basis_refs=("identity:evidence",),
            originality=(Originality.UNKNOWN if unknown_lineage else Originality.ORIGINAL),
            originality_basis_refs=() if unknown_lineage else ("origin:evidence",),
            original_source_ref=None,
            expertise_fit=ExplicitFact(FactStatus.SUPPORTED, ("expertise:evidence",)),
            self_report=ExplicitFact(FactStatus.NOT_SUPPORTED, ("report:evidence",)),
            transparency=ExplicitFact(FactStatus.SUPPORTED, ("transparency:evidence",)),
            proposition_temporal_sensitivity=ExplicitFact(
                FactStatus.NOT_SUPPORTED, ("temporal:evidence",)
            ),
            recency_fit=RecencyFit.FIT,
            recency_basis_refs=("recency:evidence",),
            conflict_of_interest=ExplicitFact(FactStatus.NOT_SUPPORTED, ("coi:evidence",)),
        )
    )


def _inputs(
    *,
    unknown_lineage: bool = False,
    incomplete: bool = False,
    context_alignment: ContextAlignment = ContextAlignment.ALIGNED,
    v22_grounding_usability: AssertionGroundingUsability | None = None,
    proposition_evidence_link: PropositionEvidenceLink | None = None,
) -> AggregationInputs:
    text = "A verified statement appears here."
    document = build_readable_document(
        source_ref="source:one",
        snapshot_content_hash="snapshot-content-hash",
        canonical_text=text,
        available_text_length=len(text),
        document_type="html",
        was_truncated=False,
        source_segments=(
            DocumentSourceSegment(
                canonical_start=0,
                canonical_end=len(text),
                locator=SourceLocator(
                    locator_type=SourceLocatorType.DOCUMENT_BLOCK, block_ordinal=0
                ),
                ordinal=0,
            ),
        ),
    )
    span = locate_canonical_evidence_span(
        document,
        "A verified statement appears here.",
        (MappedTextWindow(text, 0, len(text), "chunk:one"),),
    ).span
    assert span is not None
    context = build_context_window(document, span)
    return {
        "evidence_ref": "evidence:one",
        "formation_trace": _trace(complete=not incomplete),
        "grounding": ExistingGroundingObservation(
            grounding_ref="grounding:one",
            source_span_available="YES",
            quote_exactness=QuoteExactness.EXACT,
            context_alignment=context_alignment,
            v22_grounding_usability=v22_grounding_usability,
            proposition_evidence_link=proposition_evidence_link,
        ),
        "document": document,
        "document_span": span,
        "context_window": context,
        "relevant_span": None,
        "processing_chunks": (
            ProcessingChunk(
                chunk_ref="a" * 64,
                document_ref=document.document_ref,
                canonical_start=0,
                canonical_end=7,
                text=text[:7],
                ordinal=0,
            ),
            ProcessingChunk(
                chunk_ref="b" * 64,
                document_ref=document.document_ref,
                canonical_start=7,
                canonical_end=len(text),
                text=text[7:],
                ordinal=1,
            ),
        ),
        "source_assessment": _source(unknown_lineage=unknown_lineage),
        "claim_source_fit": _source(unknown_lineage=unknown_lineage).claim_source_fit,
        "acceptance_ref": "acceptance:one",
    }


def test_complete_inputs_are_assessed_without_creating_a_score() -> None:
    result = aggregate_evidence_reliability(**_inputs())

    assert result.assessment_status is ReliabilityAssessmentStatus.ASSESSED
    assert result.reason_codes == (
        ReliabilityReasonCode.FORMATION_AVAILABLE,
        ReliabilityReasonCode.FULLY_ASSESSED,
        ReliabilityReasonCode.GROUNDING_AVAILABLE,
        ReliabilityReasonCode.SOURCE_ASSESSMENT_AVAILABLE,
    )
    assert not hasattr(result, "reliability_score")
    assert not hasattr(result, "confidence_score")
    assert not hasattr(result, "accepted")
    assert not hasattr(result, "verification_status")


def test_unknown_source_lineage_is_preserved_as_a_limitation() -> None:
    result = aggregate_evidence_reliability(**_inputs(unknown_lineage=True))

    assert result.assessment_status is ReliabilityAssessmentStatus.ASSESSED_WITH_LIMITATIONS
    assert KnownLimitation.LINEAGE_UNKNOWN in result.known_limitations
    assert ReliabilityReasonCode.LINEAGE_UNKNOWN in result.reason_codes


def test_missing_document_span_makes_assessment_unknown() -> None:
    inputs = _inputs()
    inputs["document_span"] = None
    result = aggregate_evidence_reliability(**inputs)

    assert result.assessment_status is ReliabilityAssessmentStatus.UNKNOWN
    assert ReliabilityReasonCode.INSUFFICIENT_INPUT in result.reason_codes


def test_missing_formation_or_grounding_is_unknown() -> None:
    for key in ("formation_trace", "grounding"):
        inputs = _inputs()
        inputs[key] = None
        result = aggregate_evidence_reliability(**inputs)
        assert result.assessment_status is ReliabilityAssessmentStatus.UNKNOWN
        assert ReliabilityReasonCode.INSUFFICIENT_INPUT in result.reason_codes


def test_incomplete_formation_is_preserved_not_reinterpreted() -> None:
    result = aggregate_evidence_reliability(**_inputs(incomplete=True))

    assert result.assessment_status is ReliabilityAssessmentStatus.ASSESSED_WITH_LIMITATIONS
    assert result.formation is not None
    assert result.formation.acceptance_observation is AcceptanceObservation.UNKNOWN
    assert KnownLimitation.FORMATION_INCOMPLETE in result.known_limitations
    assert KnownLimitation.ACCEPTANCE_UNKNOWN in result.known_limitations


def test_aggregation_does_not_mutate_any_input() -> None:
    inputs = _inputs()
    trace = inputs["formation_trace"]
    assert trace is not None
    before = trace.model_dump(mode="json")

    aggregate_evidence_reliability(**inputs)

    assert trace.model_dump(mode="json") == before
    assert inputs["document"].canonical_text == "A verified statement appears here."


def test_processing_input_order_does_not_change_result() -> None:
    inputs = _inputs()
    first = aggregate_evidence_reliability(**inputs)
    inputs["processing_chunks"] = tuple(reversed(inputs["processing_chunks"]))
    second = aggregate_evidence_reliability(**inputs)

    assert first == second
    assert first.assessment_ref == second.assessment_ref


def test_quote_and_source_text_are_not_copied_to_assessment() -> None:
    inputs = _inputs()
    result = aggregate_evidence_reliability(**inputs)
    serialized = result.model_dump_json()
    span = inputs["document_span"]
    assert span is not None

    assert span.exact_text not in serialized
    assert inputs["document"].canonical_text not in serialized
    assert "quote_reference" not in serialized


def test_optional_v22_adapter_is_reference_only_and_does_not_change_v22() -> None:
    assessment = aggregate_evidence_reliability(**_inputs())
    adapter = as_optional_verification_input(assessment)

    assert adapter.assessment_ref == assessment.assessment_ref
    assert adapter.assessment_status is ReliabilityAssessmentStatus.ASSESSED
    assert not hasattr(adapter, "verification_status")
    assert not hasattr(adapter, "accepted")


def test_existing_v22_grounding_unknown_is_preserved_as_a_limitation() -> None:
    inputs = _inputs(v22_grounding_usability=AssertionGroundingUsability.UNKNOWN)
    result = aggregate_evidence_reliability(**inputs)

    assert result.assessment_status is ReliabilityAssessmentStatus.ASSESSED_WITH_LIMITATIONS
    assert KnownLimitation.V22_GROUNDING_UNRESOLVED in result.known_limitations
    assert result.grounding is not None
    assert result.grounding.v22_grounding_usability is AssertionGroundingUsability.UNKNOWN


def test_proposition_evidence_link_is_referenced_without_quote_or_owner_data() -> None:
    link = PropositionEvidenceLink(
        proposition_id=UUID(int=1),
        evidence_id=UUID(int=2),
        claim_id=UUID(int=3),
        relation=EvidencePropositionRelation.SUPPORTS,
        source_id=UUID(int=4),
        source_owner_key="owner-secret-like-value",
        quote_reference="private quote text",
        span_start=5,
        span_end=10,
        method="existing-link",
        run_id=UUID(int=6),
        question_id="q1",
        dimension_key="d1",
    )
    inputs = _inputs(
        v22_grounding_usability=AssertionGroundingUsability.SUFFICIENT,
        proposition_evidence_link=link,
    )
    result = aggregate_evidence_reliability(**inputs)
    serialized = result.model_dump_json()

    assert result.grounding is not None
    assert result.grounding.proposition_evidence_link_ref is not None
    assert "private quote text" not in serialized
    assert "owner-secret-like-value" not in serialized


def test_partial_document_remains_an_explicit_limitation() -> None:
    inputs = _inputs()
    inputs["document"] = inputs["document"].model_copy(
        update={"completeness": DocumentCompleteness.PARTIAL, "was_truncated": True}
    )
    # Span identity still belongs to the same canonical document; only the
    # completeness observation changes in this deterministic aggregation test.
    result = aggregate_evidence_reliability(**inputs)

    assert result.assessment_status is ReliabilityAssessmentStatus.ASSESSED_WITH_LIMITATIONS
    assert KnownLimitation.DOCUMENT_PARTIAL in result.known_limitations
