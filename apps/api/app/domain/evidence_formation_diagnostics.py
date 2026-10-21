"""Deterministic, observational diagnostics for the production evidence funnel.

These contracts describe events already produced by the ResearchLoop. They do
not make acceptance, localization, persistence, or verification decisions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

EVIDENCE_FORMATION_DIAGNOSTIC_VERSION = "evidence-formation-diagnostic.v1"
EVIDENCE_CANDIDATE_DIAGNOSTIC_VERSION = "evidence-candidate-diagnostic.v1"


class EvidenceFormationStage(StrEnum):
    PAGE_READ = "PAGE_READ"
    TRIAGE = "TRIAGE"
    SELECTION = "SELECTION"
    EXTRACTION_REQUEST = "EXTRACTION_REQUEST"
    EXTRACTION = "EXTRACTION"
    CANDIDATE_FORMATION = "CANDIDATE_FORMATION"
    ACCEPTANCE = "ACCEPTANCE"
    QUOTE_LOCALIZATION = "QUOTE_LOCALIZATION"
    PERSISTENCE = "PERSISTENCE"
    POST_PERSISTENCE_ALIGNMENT = "POST_PERSISTENCE_ALIGNMENT"
    CLOSURE_EVALUATION = "CLOSURE_EVALUATION"
    PROPOSITION_FORMATION = "PROPOSITION_FORMATION"
    SOURCE_GROUNDING = "SOURCE_GROUNDING"
    VERIFICATION = "VERIFICATION"


class EvidenceFormationOutcome(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    EMPTY = "EMPTY"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    NOT_OBSERVED = "NOT_OBSERVED"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class EvidenceFormationReasonCode(StrEnum):
    BODY_TOO_SHORT = "BODY_TOO_SHORT"
    PROMPT_INJECTION_DETECTED = "PROMPT_INJECTION_DETECTED"
    LOGIN_OR_NAVIGATION_PAGE = "LOGIN_OR_NAVIGATION_PAGE"
    TOPIC_MISMATCH = "TOPIC_MISMATCH"
    ALREADY_PROCESSED = "ALREADY_PROCESSED"
    DUPLICATE_SOURCE = "DUPLICATE_SOURCE"
    DUPLICATE_EVIDENCE = "DUPLICATE_EVIDENCE"
    EXTRACTION_BUDGET = "EXTRACTION_BUDGET"
    QUESTION_BUDGET = "QUESTION_BUDGET"
    TOKEN_BUDGET = "TOKEN_BUDGET"
    SOURCE_DIVERSITY = "SOURCE_DIVERSITY"
    LOW_EVIDENCE_VALUE = "LOW_EVIDENCE_VALUE"
    QUESTION_MISMATCH = "QUESTION_MISMATCH"
    PAGE_LIMIT = "PAGE_LIMIT"
    QUOTE_NOT_FOUND_IN_SOURCE = "QUOTE_NOT_FOUND_IN_SOURCE"
    QUOTE_LOCALIZATION_AMBIGUOUS = "QUOTE_LOCALIZATION_AMBIGUOUS"
    QUOTE_LOCALIZATION_UNMAPPED = "QUOTE_LOCALIZATION_UNMAPPED"
    CLAIM_QUOTE_ENTAILMENT_FAILED = "CLAIM_QUOTE_ENTAILMENT_FAILED"
    SCOPE_MISMATCH = "SCOPE_MISMATCH"
    NUMERIC_SCOPE_INCONSISTENT = "NUMERIC_SCOPE_INCONSISTENT"
    SOURCE_RELIABILITY_BELOW_THRESHOLD = "SOURCE_RELIABILITY_BELOW_THRESHOLD"
    SOURCE_ROLE_MISMATCH = "SOURCE_ROLE_MISMATCH"
    EVIDENCE_RELEVANCE_BELOW_THRESHOLD = "EVIDENCE_RELEVANCE_BELOW_THRESHOLD"
    EVIDENCE_CONFIDENCE_BELOW_THRESHOLD = "EVIDENCE_CONFIDENCE_BELOW_THRESHOLD"
    EVIDENCE_SCORE_BELOW_THRESHOLD = "EVIDENCE_SCORE_BELOW_THRESHOLD"
    QUOTE_NOT_LOCATED_FOR_GRAPH = "QUOTE_NOT_LOCATED_FOR_GRAPH"
    DEADLINE_EXHAUSTED = "DEADLINE_EXHAUSTED"
    MODEL_ERROR = "MODEL_ERROR"
    STRUCTURED_OUTPUT_INVALID = "STRUCTURED_OUTPUT_INVALID"
    EVIDENCE_PERSISTENCE_FAILED = "EVIDENCE_PERSISTENCE_FAILED"
    OTHER = "OTHER"


class EvidenceAcceptanceGate(StrEnum):
    INJECTION = "INJECTION"
    QUOTE_MATCH = "QUOTE_MATCH"
    CLAIM_QUOTE_ENTAILMENT = "CLAIM_QUOTE_ENTAILMENT"
    SCOPE = "SCOPE"
    NUMERIC_SCOPE = "NUMERIC_SCOPE"
    SOURCE_RELIABILITY = "SOURCE_RELIABILITY"
    SOURCE_ROLE_FIT = "SOURCE_ROLE_FIT"
    RELEVANCE = "RELEVANCE"
    CONFIDENCE = "CONFIDENCE"
    EVIDENCE_SCORE = "EVIDENCE_SCORE"


class EvidenceAcceptanceGateOutcome(StrEnum):
    PASSED = "PASSED"
    REJECTED = "REJECTED"
    NOT_EVALUATED = "NOT_EVALUATED"


class EvidenceAcceptanceGateObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    gate: EvidenceAcceptanceGate
    outcome: EvidenceAcceptanceGateOutcome
    observed_value: float | bool | None = None
    threshold: float | None = None


class EvidenceFormationDiagnosticEvent(BaseModel):
    """Small immutable event payload stored in the existing run event stream."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    diagnostic_version: str = EVIDENCE_FORMATION_DIAGNOSTIC_VERSION
    event_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_ref: str
    target_ref: str
    page_ref: str | None = None
    source_ref: str | None = None
    snapshot_ref: str | None = None
    candidate_ref: str | None = None
    evidence_ref: str | None = None
    stage: EvidenceFormationStage
    outcome: EvidenceFormationOutcome
    reason_codes: tuple[EvidenceFormationReasonCode, ...] = ()
    raw_reason: str | None = Field(default=None, max_length=200)
    upstream_refs: tuple[str, ...] = ()
    gate_observations: tuple[EvidenceAcceptanceGateObservation, ...] = ()
    facts: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        run_ref: str,
        target_ref: str,
        stage: EvidenceFormationStage,
        outcome: EvidenceFormationOutcome,
        page_ref: str | None = None,
        source_ref: str | None = None,
        snapshot_ref: str | None = None,
        candidate_ref: str | None = None,
        evidence_ref: str | None = None,
        reason_codes: Iterable[EvidenceFormationReasonCode] = (),
        raw_reason: str | None = None,
        upstream_refs: Iterable[str] = (),
        gate_observations: Iterable[EvidenceAcceptanceGateObservation] = (),
        facts: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> EvidenceFormationDiagnosticEvent:
        reasons = tuple(sorted(set(reason_codes), key=lambda value: value.value))
        upstream = tuple(sorted(set(upstream_refs)))
        gates = tuple(gate_observations)
        safe_facts = dict(sorted((facts or {}).items()))
        identity = {
            "diagnostic_version": EVIDENCE_FORMATION_DIAGNOSTIC_VERSION,
            "run_ref": run_ref,
            "target_ref": target_ref,
            "page_ref": page_ref,
            "source_ref": source_ref,
            "snapshot_ref": snapshot_ref,
            "candidate_ref": candidate_ref,
            "evidence_ref": evidence_ref,
            "stage": stage.value,
            "outcome": outcome.value,
            "reason_codes": [value.value for value in reasons],
            "raw_reason": raw_reason,
            "upstream_refs": upstream,
            "gate_observations": [value.model_dump(mode="json") for value in gates],
            "facts": safe_facts,
        }
        event_ref = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(
            event_ref=event_ref,
            run_ref=run_ref,
            target_ref=target_ref,
            page_ref=page_ref,
            source_ref=source_ref,
            snapshot_ref=snapshot_ref,
            candidate_ref=candidate_ref,
            evidence_ref=evidence_ref,
            stage=stage,
            outcome=outcome,
            reason_codes=reasons,
            raw_reason=raw_reason,
            upstream_refs=upstream,
            gate_observations=gates,
            facts=safe_facts,
        )


class EvidenceFormationTrace(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_ref: str
    target_ref: str
    page_ref: str | None = None
    source_refs: tuple[str, ...] = ()
    events: tuple[EvidenceFormationDiagnosticEvent, ...]
    proposition_formation: EvidenceFormationOutcome = EvidenceFormationOutcome.NOT_AVAILABLE
    source_grounding: EvidenceFormationOutcome = EvidenceFormationOutcome.NOT_AVAILABLE
    verification: EvidenceFormationOutcome = EvidenceFormationOutcome.NOT_AVAILABLE


class EvidenceFormationFunnelSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    diagnostic_version: str = EVIDENCE_FORMATION_DIAGNOSTIC_VERSION
    pages_read: int = Field(ge=0)
    pages_triage_skipped: int = Field(ge=0)
    pages_sent_to_extraction: int = Field(ge=0)
    extraction_empty: int = Field(ge=0)
    extraction_failed: int = Field(ge=0)
    candidates_formed: int = Field(ge=0)
    candidates_accepted: int = Field(ge=0)
    candidates_rejected: int = Field(ge=0)
    quote_localization_failed: int = Field(ge=0)
    evidence_persisted: int = Field(ge=0)
    accepted_located_evidence_persisted: int = Field(ge=0)


_REASON_MAP: dict[str, EvidenceFormationReasonCode] = {
    reason.value.lower(): reason
    for reason in EvidenceFormationReasonCode
    if reason is not EvidenceFormationReasonCode.OTHER
}
_REASON_MAP.update(
    {
        "topic_mismatch": EvidenceFormationReasonCode.TOPIC_MISMATCH,
        "prompt_injection_detected": EvidenceFormationReasonCode.PROMPT_INJECTION_DETECTED,
        "quote_not_found_in_source": EvidenceFormationReasonCode.QUOTE_NOT_FOUND_IN_SOURCE,
        "quote_localization_ambiguous": EvidenceFormationReasonCode.QUOTE_LOCALIZATION_AMBIGUOUS,
        "quote_localization_unmapped": EvidenceFormationReasonCode.QUOTE_LOCALIZATION_UNMAPPED,
        "claim_quote_entailment_failed": EvidenceFormationReasonCode.CLAIM_QUOTE_ENTAILMENT_FAILED,
        "numeric_scope_inconsistent": EvidenceFormationReasonCode.NUMERIC_SCOPE_INCONSISTENT,
        "source_reliability_below_threshold": (
            EvidenceFormationReasonCode.SOURCE_RELIABILITY_BELOW_THRESHOLD
        ),
        "source_role_mismatch": EvidenceFormationReasonCode.SOURCE_ROLE_MISMATCH,
        "evidence_relevance_below_threshold": (
            EvidenceFormationReasonCode.EVIDENCE_RELEVANCE_BELOW_THRESHOLD
        ),
        "evidence_confidence_below_threshold": (
            EvidenceFormationReasonCode.EVIDENCE_CONFIDENCE_BELOW_THRESHOLD
        ),
        "evidence_score_below_threshold": (
            EvidenceFormationReasonCode.EVIDENCE_SCORE_BELOW_THRESHOLD
        ),
        "quote_not_located_for_graph": EvidenceFormationReasonCode.QUOTE_NOT_LOCATED_FOR_GRAPH,
        "deadline_exhausted": EvidenceFormationReasonCode.DEADLINE_EXHAUSTED,
        "extraction_failed": EvidenceFormationReasonCode.MODEL_ERROR,
        "evidence_persistence_failed": EvidenceFormationReasonCode.EVIDENCE_PERSISTENCE_FAILED,
    }
)


def normalize_evidence_formation_reason(
    raw_reason: str | None,
) -> EvidenceFormationReasonCode | None:
    if raw_reason is None:
        return None
    return _REASON_MAP.get(raw_reason.strip().casefold(), EvidenceFormationReasonCode.OTHER)


def candidate_diagnostic_ref(
    *,
    run_ref: str,
    source_ref: str,
    snapshot_content_hash: str,
    ordinal: int,
    canonical_payload: Mapping[str, Any],
) -> str:
    """Stable per-run correlation ref, not a semantic proposition identity."""

    canonical_payload_json = json.dumps(
        canonical_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    canonical_payload_hash = hashlib.sha256(canonical_payload_json.encode("utf-8")).hexdigest()
    identity = {
        "version": EVIDENCE_CANDIDATE_DIAGNOSTIC_VERSION,
        "run_ref": run_ref,
        "source_ref": source_ref,
        "snapshot_content_hash": snapshot_content_hash,
        "ordinal": ordinal,
        "payload_hash": canonical_payload_hash,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"candidate:{digest}"


def acceptance_gate_observations(
    *,
    raw_rejection_reason: str | None,
    quote_matched: bool | None,
    evidence_score: float,
    reliability: float,
    relevance: float,
    confidence: float,
) -> tuple[EvidenceAcceptanceGateObservation, ...]:
    """Describe the existing first-failure gate order without rerunning gates."""

    reason_to_gate = {
        "prompt_injection_detected": EvidenceAcceptanceGate.INJECTION,
        "quote_not_found_in_source": EvidenceAcceptanceGate.QUOTE_MATCH,
        "claim_quote_entailment_failed": EvidenceAcceptanceGate.CLAIM_QUOTE_ENTAILMENT,
        "scope_mismatch": EvidenceAcceptanceGate.SCOPE,
        "numeric_scope_inconsistent": EvidenceAcceptanceGate.NUMERIC_SCOPE,
        "source_reliability_below_threshold": EvidenceAcceptanceGate.SOURCE_RELIABILITY,
        "source_role_mismatch": EvidenceAcceptanceGate.SOURCE_ROLE_FIT,
        "evidence_relevance_below_threshold": EvidenceAcceptanceGate.RELEVANCE,
        "evidence_confidence_below_threshold": EvidenceAcceptanceGate.CONFIDENCE,
        "evidence_score_below_threshold": EvidenceAcceptanceGate.EVIDENCE_SCORE,
    }
    order = tuple(EvidenceAcceptanceGate)
    failed_gate = reason_to_gate.get((raw_rejection_reason or "").casefold())
    observed_values: dict[EvidenceAcceptanceGate, float | bool | None] = {
        EvidenceAcceptanceGate.QUOTE_MATCH: quote_matched,
        EvidenceAcceptanceGate.SOURCE_RELIABILITY: reliability,
        EvidenceAcceptanceGate.RELEVANCE: relevance,
        EvidenceAcceptanceGate.CONFIDENCE: confidence,
        EvidenceAcceptanceGate.EVIDENCE_SCORE: evidence_score,
    }
    observations: list[EvidenceAcceptanceGateObservation] = []
    stopped = False
    for gate in order:
        if stopped:
            outcome = EvidenceAcceptanceGateOutcome.NOT_EVALUATED
        elif failed_gate is gate:
            outcome = EvidenceAcceptanceGateOutcome.REJECTED
            stopped = True
        elif failed_gate is None and raw_rejection_reason is not None:
            outcome = EvidenceAcceptanceGateOutcome.NOT_EVALUATED
        else:
            outcome = EvidenceAcceptanceGateOutcome.PASSED
        observations.append(
            EvidenceAcceptanceGateObservation(
                gate=gate,
                outcome=outcome,
                observed_value=observed_values.get(gate),
            )
        )
    return tuple(observations)


def build_evidence_formation_summary(
    events: Iterable[EvidenceFormationDiagnosticEvent],
) -> EvidenceFormationFunnelSummary:
    """Aggregate a canonical event set; duplicate inputs/order do not matter."""

    unique = {event.event_ref: event for event in events}
    values = tuple(unique[key] for key in sorted(unique))

    def refs(stage: EvidenceFormationStage, outcome: EvidenceFormationOutcome) -> set[str]:
        return {
            ref
            for event in values
            if event.stage is stage and event.outcome is outcome
            if (ref := _count_ref(event)) is not None
        }

    return EvidenceFormationFunnelSummary(
        pages_read=len(refs(EvidenceFormationStage.PAGE_READ, EvidenceFormationOutcome.SUCCEEDED)),
        pages_triage_skipped=len(
            refs(EvidenceFormationStage.TRIAGE, EvidenceFormationOutcome.SKIPPED)
        ),
        pages_sent_to_extraction=len(
            refs(EvidenceFormationStage.EXTRACTION_REQUEST, EvidenceFormationOutcome.SUCCEEDED)
        ),
        extraction_empty=len(
            refs(EvidenceFormationStage.EXTRACTION, EvidenceFormationOutcome.EMPTY)
        ),
        extraction_failed=len(
            refs(EvidenceFormationStage.EXTRACTION, EvidenceFormationOutcome.FAILED)
        ),
        candidates_formed=len(
            refs(EvidenceFormationStage.CANDIDATE_FORMATION, EvidenceFormationOutcome.SUCCEEDED)
        ),
        candidates_accepted=len(
            refs(EvidenceFormationStage.ACCEPTANCE, EvidenceFormationOutcome.ACCEPTED)
        ),
        candidates_rejected=len(
            refs(EvidenceFormationStage.ACCEPTANCE, EvidenceFormationOutcome.REJECTED)
        ),
        quote_localization_failed=len(
            refs(EvidenceFormationStage.QUOTE_LOCALIZATION, EvidenceFormationOutcome.FAILED)
        ),
        evidence_persisted=len(
            refs(EvidenceFormationStage.PERSISTENCE, EvidenceFormationOutcome.SUCCEEDED)
        ),
        accepted_located_evidence_persisted=len(
            {
                event.evidence_ref
                for event in values
                if event.stage is EvidenceFormationStage.PERSISTENCE
                and event.outcome is EvidenceFormationOutcome.SUCCEEDED
                and event.facts.get("accepted_and_located") is True
                and event.evidence_ref is not None
            }
        ),
    )


def build_evidence_formation_traces(
    events: Iterable[EvidenceFormationDiagnosticEvent],
) -> tuple[EvidenceFormationTrace, ...]:
    """Group by run/target/page, joining downstream evidence refs when possible."""

    unique = {event.event_ref: event for event in events}
    evidence_pages: dict[tuple[str, str, str], set[str]] = {}
    for event in unique.values():
        if (
            event.stage is EvidenceFormationStage.PERSISTENCE
            and event.outcome is EvidenceFormationOutcome.SUCCEEDED
            and event.evidence_ref is not None
            and event.page_ref is not None
        ):
            key = (event.run_ref, event.target_ref, event.evidence_ref)
            evidence_pages.setdefault(key, set()).add(event.page_ref)

    grouped: dict[tuple[str, str, str | None], list[EvidenceFormationDiagnosticEvent]] = {}
    for event in unique.values():
        page_refs = (
            {event.page_ref}
            if event.page_ref is not None
            else evidence_pages.get(
                (event.run_ref, event.target_ref, event.evidence_ref or ""), set()
            )
        )
        if page_refs:
            for page_ref in page_refs:
                page_key = (event.run_ref, event.target_ref, page_ref)
                grouped.setdefault(page_key, []).append(event)
        elif event.stage is EvidenceFormationStage.CLOSURE_EVALUATION:
            target_key = (event.run_ref, event.target_ref, None)
            grouped.setdefault(target_key, []).append(event)
    traces = [
        EvidenceFormationTrace(
            run_ref=key[0],
            target_ref=key[1],
            page_ref=key[2],
            source_refs=tuple(sorted({event.source_ref for event in values if event.source_ref})),
            events=tuple(sorted(values, key=lambda event: event.event_ref)),
        )
        for key, values in grouped.items()
    ]
    return tuple(
        sorted(
            traces,
            key=lambda trace: (
                trace.run_ref,
                trace.target_ref,
                trace.page_ref or "",
            ),
        )
    )


def _count_ref(event: EvidenceFormationDiagnosticEvent) -> str | None:
    if event.stage in {
        EvidenceFormationStage.PAGE_READ,
        EvidenceFormationStage.TRIAGE,
        EvidenceFormationStage.SELECTION,
        EvidenceFormationStage.EXTRACTION_REQUEST,
        EvidenceFormationStage.EXTRACTION,
    }:
        return event.page_ref
    if event.stage in {
        EvidenceFormationStage.CANDIDATE_FORMATION,
        EvidenceFormationStage.ACCEPTANCE,
        EvidenceFormationStage.QUOTE_LOCALIZATION,
    }:
        return event.candidate_ref
    if event.stage is EvidenceFormationStage.PERSISTENCE:
        return event.evidence_ref
    return None
