from __future__ import annotations

from app.domain.evidence_formation_diagnostics import (
    EvidenceAcceptanceGate,
    EvidenceAcceptanceGateOutcome,
    EvidenceFormationDiagnosticEvent,
    EvidenceFormationOutcome,
    EvidenceFormationReasonCode,
    EvidenceFormationStage,
    acceptance_gate_observations,
    build_evidence_formation_summary,
    build_evidence_formation_traces,
    candidate_diagnostic_ref,
    normalize_evidence_formation_reason,
)


def _event(
    stage: EvidenceFormationStage,
    outcome: EvidenceFormationOutcome,
    *,
    page: str = "page-1",
    candidate: str | None = None,
    evidence: str | None = None,
    reason: str | None = None,
    facts: dict[str, str | int | float | bool | None] | None = None,
) -> EvidenceFormationDiagnosticEvent:
    return EvidenceFormationDiagnosticEvent.create(
        run_ref="run-1",
        target_ref="target-1",
        page_ref=page,
        source_ref="source-1",
        snapshot_ref="snapshot-1",
        candidate_ref=candidate,
        evidence_ref=evidence,
        stage=stage,
        outcome=outcome,
        reason_codes=(
            ()
            if reason is None
            else (
                normalize_evidence_formation_reason(reason)
                or EvidenceFormationReasonCode.OTHER,
            )
        ),
        raw_reason=reason,
        facts=facts,
    )


def test_reason_normalization_preserves_raw_detail_and_distinguishes_gates() -> None:
    assert normalize_evidence_formation_reason("claim_quote_entailment_failed") is (
        EvidenceFormationReasonCode.CLAIM_QUOTE_ENTAILMENT_FAILED
    )
    assert normalize_evidence_formation_reason("scope_mismatch") is (
        EvidenceFormationReasonCode.SCOPE_MISMATCH
    )
    assert normalize_evidence_formation_reason("unrecognized detail") is (
        EvidenceFormationReasonCode.OTHER
    )
    event = _event(
        EvidenceFormationStage.ACCEPTANCE,
        EvidenceFormationOutcome.REJECTED,
        candidate="candidate-1",
        reason="claim_quote_entailment_failed",
    )
    assert event.raw_reason == "claim_quote_entailment_failed"
    assert event.reason_codes == (
        EvidenceFormationReasonCode.CLAIM_QUOTE_ENTAILMENT_FAILED,
    )


def test_acceptance_gate_observations_stop_after_the_observed_first_failure() -> None:
    observations = acceptance_gate_observations(
        raw_rejection_reason="scope_mismatch",
        quote_matched=True,
        evidence_score=0.9,
        reliability=0.9,
        relevance=0.9,
        confidence=0.9,
    )
    by_gate = {observation.gate: observation.outcome for observation in observations}
    assert by_gate[EvidenceAcceptanceGate.INJECTION] is EvidenceAcceptanceGateOutcome.PASSED
    assert by_gate[EvidenceAcceptanceGate.QUOTE_MATCH] is EvidenceAcceptanceGateOutcome.PASSED
    assert by_gate[EvidenceAcceptanceGate.CLAIM_QUOTE_ENTAILMENT] is (
        EvidenceAcceptanceGateOutcome.PASSED
    )
    assert by_gate[EvidenceAcceptanceGate.SCOPE] is EvidenceAcceptanceGateOutcome.REJECTED
    assert by_gate[EvidenceAcceptanceGate.NUMERIC_SCOPE] is (
        EvidenceAcceptanceGateOutcome.NOT_EVALUATED
    )


def test_unknown_rejection_reason_does_not_fabricate_gate_results() -> None:
    observations = acceptance_gate_observations(
        raw_rejection_reason="new reason not in this version",
        quote_matched=None,
        evidence_score=0.0,
        reliability=0.0,
        relevance=0.0,
        confidence=0.0,
    )
    assert all(
        observation.outcome is EvidenceAcceptanceGateOutcome.NOT_EVALUATED
        for observation in observations
    )


def test_empty_failed_rejected_localization_and_persistence_are_distinct() -> None:
    events = [
        _event(EvidenceFormationStage.EXTRACTION, EvidenceFormationOutcome.EMPTY),
        _event(EvidenceFormationStage.EXTRACTION, EvidenceFormationOutcome.FAILED, page="page-2"),
        _event(
            EvidenceFormationStage.ACCEPTANCE,
            EvidenceFormationOutcome.REJECTED,
            candidate="candidate-1",
            reason="evidence_relevance_below_threshold",
        ),
        _event(
            EvidenceFormationStage.QUOTE_LOCALIZATION,
            EvidenceFormationOutcome.FAILED,
            candidate="candidate-2",
            reason="quote_not_located_for_graph",
        ),
        _event(
            EvidenceFormationStage.PERSISTENCE,
            EvidenceFormationOutcome.SUCCEEDED,
            candidate="candidate-3",
            evidence="evidence-1",
            facts={"accepted_and_located": True},
        ),
    ]
    summary = build_evidence_formation_summary(events)
    assert summary.extraction_empty == 1
    assert summary.extraction_failed == 1
    assert summary.candidates_rejected == 1
    assert summary.quote_localization_failed == 1
    assert summary.evidence_persisted == 1
    assert summary.accepted_located_evidence_persisted == 1


def test_traces_group_by_page_and_downstream_unavailable_is_not_failure() -> None:
    events = [
        _event(EvidenceFormationStage.PAGE_READ, EvidenceFormationOutcome.SUCCEEDED),
        _event(
            EvidenceFormationStage.EXTRACTION_REQUEST,
            EvidenceFormationOutcome.SUCCEEDED,
        ),
        _event(
            EvidenceFormationStage.CANDIDATE_FORMATION,
            EvidenceFormationOutcome.SUCCEEDED,
            candidate="candidate-1",
        ),
        _event(
            EvidenceFormationStage.PAGE_READ,
            EvidenceFormationOutcome.SUCCEEDED,
            page="page-2",
        ),
        _event(
            EvidenceFormationStage.EXTRACTION,
            EvidenceFormationOutcome.EMPTY,
            page="page-2",
        ),
    ]
    traces = build_evidence_formation_traces(events)
    assert len(traces) == 2
    assert {trace.page_ref: len(trace.events) for trace in traces} == {
        "page-1": 3,
        "page-2": 2,
    }
    assert all(
        trace.proposition_formation is EvidenceFormationOutcome.NOT_AVAILABLE
        and trace.source_grounding is EvidenceFormationOutcome.NOT_AVAILABLE
        and trace.verification is EvidenceFormationOutcome.NOT_AVAILABLE
        for trace in traces
    )


def test_downstream_alignment_joins_by_evidence_and_closure_stays_target_scoped() -> None:
    persisted = _event(
        EvidenceFormationStage.PERSISTENCE,
        EvidenceFormationOutcome.SUCCEEDED,
        page="page-1",
        candidate="candidate-1",
        evidence="evidence-1",
        facts={"accepted_and_located": True},
    )
    alignment = EvidenceFormationDiagnosticEvent.create(
        run_ref="run-1",
        target_ref="target-1",
        source_ref="source-1",
        evidence_ref="evidence-1",
        stage=EvidenceFormationStage.POST_PERSISTENCE_ALIGNMENT,
        outcome=EvidenceFormationOutcome.SUCCEEDED,
        upstream_refs=("evidence-1", "alignment-1"),
    )
    closure = EvidenceFormationDiagnosticEvent.create(
        run_ref="run-1",
        target_ref="target-1",
        stage=EvidenceFormationStage.CLOSURE_EVALUATION,
        outcome=EvidenceFormationOutcome.SUCCEEDED,
        upstream_refs=("evidence-1",),
        facts={"gap_ref": "gap-1"},
    )

    traces = build_evidence_formation_traces((persisted, alignment, closure))
    page_trace = next(trace for trace in traces if trace.page_ref == "page-1")
    target_trace = next(trace for trace in traces if trace.page_ref is None)
    assert alignment in page_trace.events
    assert closure in target_trace.events


def test_summary_is_order_independent_and_deduplicates_event_and_entity_refs() -> None:
    page_read = _event(EvidenceFormationStage.PAGE_READ, EvidenceFormationOutcome.SUCCEEDED)
    candidate = _event(
        EvidenceFormationStage.CANDIDATE_FORMATION,
        EvidenceFormationOutcome.SUCCEEDED,
        candidate="candidate-1",
    )
    accepted = _event(
        EvidenceFormationStage.ACCEPTANCE,
        EvidenceFormationOutcome.ACCEPTED,
        candidate="candidate-1",
    )
    persisted = _event(
        EvidenceFormationStage.PERSISTENCE,
        EvidenceFormationOutcome.SUCCEEDED,
        candidate="candidate-1",
        evidence="evidence-1",
        facts={"accepted_and_located": True},
    )
    events = [page_read, candidate, accepted, persisted]
    forward = build_evidence_formation_summary([*events, page_read])
    reverse = build_evidence_formation_summary(list(reversed(events)))
    assert forward == reverse
    assert forward.pages_read == 1
    assert forward.candidates_formed == 1
    assert forward.candidates_accepted == 1
    assert forward.evidence_persisted == 1


def test_candidate_reference_is_deterministic_and_payload_sensitive() -> None:
    common = {
        "run_ref": "run-1",
        "source_ref": "source-1",
        "snapshot_content_hash": "a" * 64,
        "ordinal": 0,
    }
    first = candidate_diagnostic_ref(**common, canonical_payload={"claim": "A"})
    replay = candidate_diagnostic_ref(**common, canonical_payload={"claim": "A"})
    changed = candidate_diagnostic_ref(**common, canonical_payload={"claim": "B"})
    assert first == replay
    assert first != changed
    assert first.startswith("candidate:")


def test_event_payload_does_not_store_page_or_candidate_text() -> None:
    event = _event(
        EvidenceFormationStage.CANDIDATE_FORMATION,
        EvidenceFormationOutcome.SUCCEEDED,
        candidate="candidate-1",
        facts={"claim_length": 42, "quote_length": 18},
    )
    payload = event.model_dump_json()
    assert "whole page contents" not in payload
    assert "raw claim text" not in payload
    assert "claim_length" in payload
    assert "candidate-1" in payload
