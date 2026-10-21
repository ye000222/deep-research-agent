"""Provider-capable, non-persistent v3.3 source semantic runtime.

The runtime is deliberately not wired into ResearchLoop. It parses a versioned
bridge envelope, validates the nested frozen v3.3 response, binds exact spans
to caller-supplied canonical provenance, and returns only typed derived data.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

from app.domain.assertion_composition import AssertionGroupOperator
from app.domain.document_evidence_context import CanonicalEvidenceSpan, ContextWindow
from app.domain.providers import AdapterType, CanonicalModelRequest, ContentPart
from app.domain.source_atomic_assertion import (
    SOURCE_ATOMIC_ASSERTION_SCHEMA_VERSION,
    SOURCE_SEMANTIC_RUNTIME_RESPONSE_SCHEMA_VERSION,
    SourceAssertionComposition,
    SourceAssertionEligibility,
    SourceAssertionField,
    SourceAtomicAssertion,
    SourceAtomicAssertionCandidateV1,
    SourceFieldGrounding,
    SourceGroundingState,
    SourceSemanticRuntimeResponseV1,
)
from app.domain.source_context_extraction import (
    AssertionStatus,
    GroundedSpan,
    SourceAssertionAnchor,
    SpanEvidence,
)
from app.domain.source_context_extraction_v3_1 import (
    SemanticEvidenceBindingV31,
    SemanticEvidenceRoleV31,
    SourceEvidenceSpanV31,
)
from app.domain.source_context_extraction_v3_3 import (
    SOURCE_CONTEXT_EXTRACTION_PROMPT_V33,
    SOURCE_CONTEXT_EXTRACTION_SCHEMA_V33,
    MeasurementAdmissibilityV33,
    SourceExtractionResponseV33,
)
from app.domain.truth_condition_context import (
    AttachmentStatus,
    ContextOrigin,
    TruthConditionContext,
    _signature,
)
from app.services.source_context_extractor_v3_3 import (
    SOURCE_CONTEXT_V33_INSTRUCTIONS,
    evaluate_measurement_admissibility_v33,
)

SOURCE_SEMANTIC_RUNTIME_PROMPT_V1 = "source-semantic-runtime.generic.v1"
SOURCE_SEMANTIC_RUNTIME_MAX_OUTPUT_TOKENS = 12_000


class StructuredGatewayV33(Protocol):
    async def generate_structured(
        self,
        *,
        adapter_type: AdapterType,
        base_url: str,
        api_key: SecretStr,
        request: CanonicalModelRequest,
        allow_regeneration: bool = True,
    ) -> Any: ...


class SourceSemanticRuntimeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_ref: str = Field(min_length=1, max_length=2_000)
    snapshot_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_evidence_span: CanonicalEvidenceSpan | None
    context_window: ContextWindow | None = None
    max_assertions: int = Field(default=64, ge=1, le=64)

    @model_validator(mode="after")
    def validate_provenance(self) -> SourceSemanticRuntimeInput:
        span = self.canonical_evidence_span
        if span is None:
            return self
        if span.source_ref != self.source_ref or span.snapshot_ref != self.snapshot_ref:
            raise ValueError("canonical_span_source_or_snapshot_mismatch")
        if span.char_end - span.char_start != len(span.exact_text):
            raise ValueError("canonical_span_offsets_do_not_match_text_length")
        if hashlib.sha256(span.exact_text.encode("utf-8")).hexdigest() != span.content_sha256:
            raise ValueError("canonical_span_content_hash_mismatch")
        window = self.context_window
        if window is not None:
            if window.document_ref != span.document_ref:
                raise ValueError("context_window_document_mismatch")
            if window.evidence_span_ref != span.span_ref:
                raise ValueError("context_window_evidence_ref_mismatch")
            if window.window_end - window.window_start != len(window.text):
                raise ValueError("context_window_offsets_do_not_match_text_length")
            if not (
                window.window_start <= span.char_start
                and span.char_end <= window.window_end
                and window.text[
                    span.char_start
                    - window.window_start : span.char_end
                    - window.window_start
                ]
                == span.exact_text
            ):
                raise ValueError("context_window_does_not_contain_canonical_span")
        return self


class SourceSemanticRuntimeStatus(StrEnum):
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_UNRESOLVED = "COMPLETED_WITH_UNRESOLVED"
    EMPTY = "EMPTY"
    INVALID = "INVALID"
    FAILED = "FAILED"
    NOT_PROCESSED = "NOT_PROCESSED"


class SourceSemanticRuntimeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_version: str = "source-semantic-runtime-v1"
    runtime_status: SourceSemanticRuntimeStatus
    source_ref: str
    snapshot_ref: str
    canonical_span_ref: str | None = None
    context_window_ref: str | None = None
    assertions: tuple[SourceAtomicAssertion, ...] = ()
    anchors: tuple[SourceAssertionAnchor, ...] = ()
    source_evidence_spans: tuple[SourceEvidenceSpanV31, ...] = ()
    truth_condition_contexts: tuple[TruthConditionContext, ...] = ()
    source_extraction: SourceExtractionResponseV33 | None = None
    semantic_evidence_bindings: tuple[SemanticEvidenceBindingV31, ...] = ()
    composition: SourceAssertionComposition | None = None
    measurement_admissibility: dict[str, MeasurementAdmissibilityV33] = Field(
        default_factory=dict
    )
    unresolved_items: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    schema_versions: dict[str, str] = Field(default_factory=dict)
    model_call_count: int = Field(default=0, ge=0, le=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)


class SourceSemanticExtractionV33Runtime:
    """One-call structured runtime; no persistence, retry, fallback, or ResearchLoop hook."""

    def __init__(
        self,
        gateway: StructuredGatewayV33,
        *,
        adapter_type: AdapterType,
        base_url: str,
        api_key: SecretStr,
        model: str,
        max_calls: int = 1,
    ) -> None:
        if max_calls != 1:
            raise ValueError("source semantic runtime call limit must equal one")
        self._gateway = gateway
        self._adapter_type = adapter_type
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._called = False

    async def extract(
        self, runtime_input: SourceSemanticRuntimeInput
    ) -> SourceSemanticRuntimeResult:
        if self._called:
            return _result(
                runtime_input,
                SourceSemanticRuntimeStatus.NOT_PROCESSED,
                reason_codes=("CALL_LIMIT_REACHED",),
            )
        self._called = True
        span = runtime_input.canonical_evidence_span
        if span is None:
            return _result(
                runtime_input,
                SourceSemanticRuntimeStatus.NOT_PROCESSED,
                reason_codes=("SOURCE_SPAN_UNAVAILABLE",),
            )

        request = CanonicalModelRequest(
            task_kind="source_semantic_extraction_v3_3",
            role="source_semantic_runtime_v3_3",
            model=self._model,
            instructions="\n".join(
                (
                    SOURCE_CONTEXT_V33_INSTRUCTIONS,
                    "Return the source-semantic-runtime-response-v1 object only.",
                    "Keep source_extraction as a source-context-extraction-schema-v3.3 object.",
                    "SourceAtomicAssertion is source-authored semantics, not a target proposition.",
                    "Use only supplied span refs and context refs; do not infer identities.",
                    "Every semantic field must cite exact evidence refs; UNKNOWN is allowed.",
                    "Preserve attribution, modality and AND/OR/XOR/UNKNOWN composition.",
                    "Never return reasoning text or copy the context window as an evidence span.",
                    "For measurement semantics, follow the frozen v3.3 admissibility contract.",
                )
            ),
            content_parts=(
                ContentPart(
                    kind="json",
                    value={
                        "source_ref": runtime_input.source_ref,
                        "snapshot_ref": runtime_input.snapshot_ref,
                        "canonical_span": {
                            "span_ref": span.span_ref,
                            "char_start": span.char_start,
                            "char_end": span.char_end,
                            "exact_text": span.exact_text,
                        },
                        "context_window": (
                            {
                                "context_ref": runtime_input.context_window.context_ref,
                                "window_start": runtime_input.context_window.window_start,
                                "window_end": runtime_input.context_window.window_end,
                                "text": runtime_input.context_window.text,
                            }
                            if runtime_input.context_window is not None
                            else None
                        ),
                        "max_assertions": runtime_input.max_assertions,
                    },
                ),
            ),
            response_contract=SourceSemanticRuntimeResponseV1.model_json_schema(),
            generation_parameters={"temperature": 0.0, "reasoning_enabled": False},
            max_output_tokens=SOURCE_SEMANTIC_RUNTIME_MAX_OUTPUT_TOKENS,
            context_manifest_id=uuid4(),
            metadata={
                "schema_version": SOURCE_SEMANTIC_RUNTIME_RESPONSE_SCHEMA_VERSION,
                "source_schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V33,
                "source_prompt_version": SOURCE_CONTEXT_EXTRACTION_PROMPT_V33,
                "runtime_prompt_version": SOURCE_SEMANTIC_RUNTIME_PROMPT_V1,
                "execution_mode": "SHADOW",
                "retry_mode": "none",
            },
        )
        started = perf_counter()
        try:
            raw = await self._gateway.generate_structured(
                adapter_type=self._adapter_type,
                base_url=self._base_url,
                api_key=self._api_key,
                request=request,
                allow_regeneration=False,
            )
        except Exception:
            return _result(
                runtime_input,
                SourceSemanticRuntimeStatus.FAILED,
                reason_codes=("GATEWAY_ERROR",),
                model_call_count=1,
                latency_ms=round((perf_counter() - started) * 1000),
            )
        latency_ms = round((perf_counter() - started) * 1000)
        raw_payload = getattr(raw, "parsed_object", None)
        if raw_payload is None:
            text = getattr(raw, "text", None)
            try:
                raw_payload = json.loads(text) if isinstance(text, str) else None
            except (json.JSONDecodeError, TypeError):
                raw_payload = None
        if not isinstance(raw_payload, dict):
            return _result(
                runtime_input,
                SourceSemanticRuntimeStatus.INVALID,
                reason_codes=("STRUCTURED_OUTPUT_INVALID",),
                model_call_count=1,
                latency_ms=latency_ms,
            )
        try:
            response = SourceSemanticRuntimeResponseV1.model_validate(raw_payload)
        except ValidationError:
            return _result(
                runtime_input,
                SourceSemanticRuntimeStatus.INVALID,
                reason_codes=("SCHEMA_INVALID",),
                model_call_count=1,
                latency_ms=latency_ms,
            )
        try:
            result = _assemble(runtime_input, response)
        except (ValueError, ValidationError, KeyError, TypeError):
            return _result(
                runtime_input,
                SourceSemanticRuntimeStatus.INVALID,
                reason_codes=("SEMANTIC_OR_PROVENANCE_VALIDATION_FAILED",),
                model_call_count=1,
                latency_ms=latency_ms,
            )
        except Exception:
            return _result(
                runtime_input,
                SourceSemanticRuntimeStatus.FAILED,
                reason_codes=("INTERNAL_EXECUTION_ERROR",),
                model_call_count=1,
                latency_ms=latency_ms,
            )
        usage = getattr(raw, "usage", None)
        return result.model_copy(
            update={
                "model_call_count": 1,
                "input_tokens": _usage(usage, "input_tokens"),
                "output_tokens": _usage(usage, "output_tokens"),
                "latency_ms": latency_ms,
            }
        )


def _result(
    runtime_input: SourceSemanticRuntimeInput,
    status: SourceSemanticRuntimeStatus,
    *,
    reason_codes: tuple[str, ...],
    model_call_count: int = 0,
    latency_ms: int = 0,
) -> SourceSemanticRuntimeResult:
    span = runtime_input.canonical_evidence_span
    return SourceSemanticRuntimeResult(
        runtime_status=status,
        source_ref=runtime_input.source_ref,
        snapshot_ref=runtime_input.snapshot_ref,
        canonical_span_ref=span.span_ref if span else None,
        context_window_ref=(
            runtime_input.context_window.context_ref if runtime_input.context_window else None
        ),
        reason_codes=reason_codes,
        model_call_count=model_call_count,
        latency_ms=latency_ms,
        schema_versions={
            "runtime": "source-semantic-runtime-v1",
            "response": SOURCE_SEMANTIC_RUNTIME_RESPONSE_SCHEMA_VERSION,
            "source": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V33,
            "source_prompt": SOURCE_CONTEXT_EXTRACTION_PROMPT_V33,
            "assertion": SOURCE_ATOMIC_ASSERTION_SCHEMA_VERSION,
        },
    )


def _assemble(
    runtime_input: SourceSemanticRuntimeInput,
    response: SourceSemanticRuntimeResponseV1,
) -> SourceSemanticRuntimeResult:
    canonical = runtime_input.canonical_evidence_span
    if canonical is None:
        raise ValueError("SOURCE_SPAN_UNAVAILABLE")
    extraction = response.source_extraction
    span_offsets: dict[str, tuple[int, int]] = {}
    evidence_spans: dict[str, SourceEvidenceSpanV31] = {}
    for item in extraction.evidence_spans:
        locations = _occurrences(canonical.exact_text, item.exact_text)
        if item.occurrence_index >= len(locations):
            raise ValueError("SOURCE_EVIDENCE_SPAN_NOT_IN_CANONICAL_SPAN")
        start = canonical.char_start + locations[item.occurrence_index]
        end = start + len(item.exact_text)
        evidence_spans[item.evidence_ref] = item
        span_offsets[item.evidence_ref] = (start, end)
    _validate_v33_graph(extraction, evidence_spans)

    local_candidates = {item.local_ref: item for item in response.assertion_candidates}
    if len(local_candidates) != len(response.assertion_candidates):
        raise ValueError("DUPLICATE_ASSERTION_LOCAL_REF")
    if len(local_candidates) > runtime_input.max_assertions:
        raise ValueError("MAX_ASSERTIONS_EXCEEDED")
    source_assertions = {item.source_assertion_ref: item for item in extraction.source_assertions}
    if len(source_assertions) != len(extraction.source_assertions):
        raise ValueError("DUPLICATE_SOURCE_ASSERTION_REF")

    for candidate in response.assertion_candidates:
        source_assertion = source_assertions.get(candidate.source_assertion_ref)
        if source_assertion is None or not set(candidate.evidence_refs).issubset(
            source_assertion.evidence_refs
        ):
            raise ValueError("SOURCE_ASSERTION_PROVENANCE_MISMATCH")
        if not candidate.evidence_refs or any(
            ref not in evidence_spans for ref in candidate.evidence_refs
        ):
            raise ValueError("ASSERTION_EVIDENCE_REF_INVALID")
        if any(
            not refs or any(ref not in candidate.evidence_refs for ref in refs)
            for refs in candidate.field_evidence_refs.values()
        ):
            raise ValueError("FIELD_EVIDENCE_REF_INVALID")
        _validate_candidate_bindings(extraction, candidate)

    contexts = _normalize_contexts(
        response.truth_condition_contexts, local_candidates, span_offsets
    )
    context_refs = {item.context_ref for item in contexts}
    assertions: list[SourceAtomicAssertion] = []
    local_to_ref: dict[str, str] = {}
    for candidate in response.assertion_candidates:
        if any(ref not in context_refs for ref in candidate.truth_condition_context_refs):
            raise ValueError("TRUTH_CONDITION_CONTEXT_REF_INVALID")
        _validate_context_grounding_refs(candidate, contexts)
        assertion_ref = _assertion_ref(runtime_input, candidate, contexts)
        local_to_ref[candidate.local_ref] = assertion_ref
        grounding = {
            field: _field_grounding(
                field,
                refs,
                evidence_spans,
                span_offsets,
                candidate,
                polarity_basis_valid=_polarity_basis_valid(
                    candidate, extraction, span_offsets
                ),
            )
            for field, refs in candidate.field_evidence_refs.items()
        }
        context_by_ref = {item.context_ref: item for item in contexts}
        attribution_refs = tuple(
            sorted(
                ref
                for ref in candidate.truth_condition_context_refs
                if context_by_ref[ref].context_type.value == "EPISTEMIC"
                and any(
                    attachment.relation.value == "ATTRIBUTES"
                    and attachment.status is AttachmentStatus.RESOLVED
                    for attachment in context_by_ref[ref].attachments
                )
            )
        )
        modality_refs = tuple(
            sorted(
                ref
                for ref in candidate.truth_condition_context_refs
                if context_by_ref[ref].context_type.value == "EPISTEMIC"
            )
        )
        assertions.append(
            SourceAtomicAssertion(
                assertion_ref=assertion_ref,
                source_ref=runtime_input.source_ref,
                snapshot_ref=runtime_input.snapshot_ref,
                source_assertion_anchor_refs=(candidate.source_assertion_ref,),
                canonical_evidence_span_ref=canonical.span_ref,
                evidence_span_refs=tuple(sorted(set(candidate.evidence_refs))),
                source_evidence_refs=tuple(sorted(set(candidate.evidence_refs))),
                subject=candidate.subject,
                predicate=candidate.predicate,
                value=candidate.value,
                polarity=candidate.polarity,
                truth_condition_context_refs=tuple(sorted(candidate.truth_condition_context_refs)),
                attribution_context_refs=attribution_refs,
                modality_context_refs=modality_refs,
                composition_ref=candidate.composition_ref,
                verification_eligibility=candidate.verification_eligibility,
                field_grounding=tuple(
                    grounding[field].model_copy(
                        update={"field": field, "evidence_refs": tuple(sorted(set(refs)))}
                    )
                    for field, refs in sorted(
                        candidate.field_evidence_refs.items(),
                        key=lambda item: item[0].value,
                    )
                ),
            )
        )

    contexts = _bind_context_targets(contexts, local_to_ref)
    composition = _normalize_composition(response.composition, local_to_ref)
    if response.composition is not None:
        group_refs = {group.group_ref for group in response.composition.groups}
        if any(
            candidate.composition_ref is not None and candidate.composition_ref not in group_refs
            for candidate in response.assertion_candidates
        ):
            raise ValueError("ASSERTION_COMPOSITION_REF_INVALID")
    assertions = _apply_composition_eligibility(assertions, composition)
    assertions = list({item.assertion_ref: item for item in assertions}.values())
    anchors = _build_anchors(extraction, evidence_spans, span_offsets)
    measurement = evaluate_measurement_admissibility_v33(extraction, canonical.exact_text)
    unresolved: list[str] = list(response.unresolved_items)
    if response.status == "UNRESOLVED":
        unresolved.extend(response.unresolved_items)
    if extraction.status == "UNRESOLVED":
        unresolved.extend(extraction.unresolved_items)
    if any(
        dimension.semantic_support is SourceGroundingState.UNKNOWN
        for assertion in assertions
        for dimension in assertion.field_grounding
    ):
        unresolved.append("SEMANTIC_SUPPORT_UNRESOLVED")
    if any(
        assertion.subject.semantic_identity.status.value == "UNRESOLVED"
        or assertion.predicate.semantic_identity.status.value == "UNRESOLVED"
        or assertion.verification_eligibility is SourceAssertionEligibility.UNKNOWN
        for assertion in assertions
    ):
        unresolved.append("IDENTITY_OR_ELIGIBILITY_UNRESOLVED")
    if any(
        context.scope.kind.value == "UNKNOWN"
        or context.certainty == "UNKNOWN"
        or any(
            attachment.status is AttachmentStatus.UNKNOWN
            or attachment.relation.value == "UNKNOWN"
            for attachment in context.attachments
        )
        for context in contexts
    ):
        unresolved.append("TRUTH_CONDITION_ATTACHMENT_UNRESOLVED")
    if any(value is MeasurementAdmissibilityV33.UNKNOWN for value in measurement.values()):
        unresolved.append("MEASUREMENT_ADMISSIBILITY_UNKNOWN")
    status = (
        SourceSemanticRuntimeStatus.EMPTY
        if not assertions
        else SourceSemanticRuntimeStatus.COMPLETED_WITH_UNRESOLVED
        if unresolved
        else SourceSemanticRuntimeStatus.COMPLETED
    )
    return SourceSemanticRuntimeResult(
        runtime_status=status,
        source_ref=runtime_input.source_ref,
        snapshot_ref=runtime_input.snapshot_ref,
        canonical_span_ref=canonical.span_ref,
        context_window_ref=(
            runtime_input.context_window.context_ref if runtime_input.context_window else None
        ),
        assertions=tuple(sorted(assertions, key=lambda item: item.assertion_ref)),
        anchors=anchors,
        source_evidence_spans=tuple(
            sorted(extraction.evidence_spans, key=lambda item: item.evidence_ref)
        ),
        truth_condition_contexts=contexts,
        source_extraction=extraction,
        semantic_evidence_bindings=tuple(
            sorted(extraction.evidence_bindings, key=lambda item: item.binding_ref)
        ),
        composition=composition,
        measurement_admissibility=dict(sorted(measurement.items())),
        unresolved_items=tuple(sorted(set(unresolved))),
        reason_codes=("SOURCE_SEMANTICS_EXTRACTED",),
        schema_versions={
            "runtime": "source-semantic-runtime-v1",
            "response": SOURCE_SEMANTIC_RUNTIME_RESPONSE_SCHEMA_VERSION,
            "source": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V33,
            "source_prompt": SOURCE_CONTEXT_EXTRACTION_PROMPT_V33,
            "assertion": SOURCE_ATOMIC_ASSERTION_SCHEMA_VERSION,
        },
    )


def _normalize_contexts(
    contexts: tuple[TruthConditionContext, ...],
    candidates: dict[str, SourceAtomicAssertionCandidateV1],
    span_offsets: dict[str, tuple[int, int]],
) -> tuple[TruthConditionContext, ...]:
    refs = {item.context_ref for item in contexts}
    if len(refs) != len(contexts):
        raise ValueError("DUPLICATE_TRUTH_CONDITION_CONTEXT_REF")
    candidate_refs = set(candidates)
    normalized: list[TruthConditionContext] = []
    for context in contexts:
        if context.origin is not ContextOrigin.SOURCE:
            raise ValueError("TRUTH_CONDITION_CONTEXT_ORIGIN_INVALID")
        if (
            context.source_start is not None
            and context.source_end is not None
            and not any(
                start <= context.source_start <= context.source_end <= end
                for start, end in span_offsets.values()
            )
        ):
            raise ValueError("TRUTH_CONDITION_CONTEXT_SPAN_OUTSIDE_CANONICAL_EVIDENCE")
        for target in context.scope.target_refs:
            if target not in candidate_refs:
                raise ValueError("TRUTH_CONDITION_CONTEXT_TARGET_REF_INVALID")
        for attachment in context.attachments:
            if attachment.status is AttachmentStatus.RESOLVED:
                if attachment.target_ref not in candidate_refs:
                    raise ValueError("TRUTH_CONDITION_ATTACHMENT_TARGET_INVALID")
            elif attachment.target_ref is not None:
                raise ValueError("UNKNOWN_CONTEXT_ATTACHMENT_HAS_TARGET")
        signature = _signature(context.context_type, context.semantic_value)
        normalized.append(context.model_copy(update={"semantic_signature": signature}))
    return tuple(sorted(normalized, key=lambda item: item.context_ref))


def _bind_context_targets(
    contexts: tuple[TruthConditionContext, ...], local_to_ref: dict[str, str]
) -> tuple[TruthConditionContext, ...]:
    bound: list[TruthConditionContext] = []
    for context in contexts:
        scope = context.scope.model_copy(
            update={
                "target_refs": tuple(
                    sorted(local_to_ref[ref] for ref in context.scope.target_refs)
                )
            }
        )
        attachments = tuple(
            attachment.model_copy(
                update={
                    "target_ref": local_to_ref[attachment.target_ref]
                    if attachment.target_ref is not None
                    else None
                }
            )
            for attachment in context.attachments
        )
        bound.append(context.model_copy(update={"scope": scope, "attachments": attachments}))
    return tuple(bound)


def _validate_candidate_bindings(
    extraction: SourceExtractionResponseV33,
    candidate: SourceAtomicAssertionCandidateV1,
) -> None:
    bindings = extraction.evidence_bindings
    expected_roles = {
        SourceAssertionField.SUBJECT: {SemanticEvidenceRoleV31.ASSERTION_EVIDENCE},
        SourceAssertionField.PREDICATE: {SemanticEvidenceRoleV31.RELATION_OPERATOR},
        SourceAssertionField.VALUE: {SemanticEvidenceRoleV31.ASSERTION_EVIDENCE},
        SourceAssertionField.POLARITY: {SemanticEvidenceRoleV31.ASSERTION_EVIDENCE},
        SourceAssertionField.CONTEXT: {
            SemanticEvidenceRoleV31.CONTEXT_VALUE,
            SemanticEvidenceRoleV31.CONDITION,
            SemanticEvidenceRoleV31.CONFIGURATION_VALUE,
            SemanticEvidenceRoleV31.SCOPE_VALUE,
            SemanticEvidenceRoleV31.POPULATION_VALUE,
            SemanticEvidenceRoleV31.LOCATION_VALUE,
            SemanticEvidenceRoleV31.SCENARIO_VALUE,
            SemanticEvidenceRoleV31.TEMPORAL_VALUE,
            SemanticEvidenceRoleV31.VERSION_VALUE,
            SemanticEvidenceRoleV31.SAMPLE_SCOPE_VALUE,
            SemanticEvidenceRoleV31.MEASUREMENT_IDENTITY,
            SemanticEvidenceRoleV31.MEASUREMENT_DEFINITION,
            SemanticEvidenceRoleV31.MEASUREMENT_VALUE,
            SemanticEvidenceRoleV31.MEASUREMENT_UNIT,
            SemanticEvidenceRoleV31.MEASUREMENT_COMPARATOR,
            SemanticEvidenceRoleV31.MEASUREMENT_BASIS,
            SemanticEvidenceRoleV31.MEASUREMENT_TARGET,
            SemanticEvidenceRoleV31.MEASUREMENT_QUALIFIER,
            SemanticEvidenceRoleV31.METHOD_RELATION,
            SemanticEvidenceRoleV31.METHOD_VALUE,
        },
        SourceAssertionField.ATTRIBUTION: {
            SemanticEvidenceRoleV31.EPISTEMIC_AGENT,
            SemanticEvidenceRoleV31.EPISTEMIC_OPERATOR,
        },
        SourceAssertionField.MODALITY: {SemanticEvidenceRoleV31.MODALITY},
    }
    if set(candidate.field_evidence_refs) != set(expected_roles).intersection(
        candidate.field_evidence_refs
    ):
        raise ValueError("FIELD_EVIDENCE_ROLE_UNKNOWN")
    for field, refs in candidate.field_evidence_refs.items():
        target_kind = (
            "CONTEXT"
            if field
            in {
                SourceAssertionField.CONTEXT,
                SourceAssertionField.ATTRIBUTION,
                SourceAssertionField.MODALITY,
            }
            else "ASSERTION"
        )
        target_refs = (
            set(candidate.truth_condition_context_refs)
            if target_kind == "CONTEXT"
            else {candidate.source_assertion_ref}
        )
        for evidence_ref in refs:
            if not any(
                binding.evidence_ref == evidence_ref
                and binding.target_kind.value == target_kind
                and binding.target_ref in target_refs
                and binding.semantic_role in expected_roles[field]
                for binding in bindings
            ):
                raise ValueError("FIELD_SEMANTIC_BINDING_MISSING")


def _validate_context_grounding_refs(
    candidate: SourceAtomicAssertionCandidateV1,
    contexts: tuple[TruthConditionContext, ...],
) -> None:
    fields = set(candidate.field_evidence_refs)
    selected = [
        context
        for context in contexts
        if context.context_ref in candidate.truth_condition_context_refs
    ]
    for context in selected:
        if context.context_type.value == "EPISTEMIC":
            required = {SourceAssertionField.ATTRIBUTION, SourceAssertionField.MODALITY}
            if not fields.intersection(required):
                raise ValueError("EPISTEMIC_CONTEXT_GROUNDING_REF_MISSING")
        elif SourceAssertionField.CONTEXT not in fields:
            raise ValueError("MATERIAL_CONTEXT_GROUNDING_REF_MISSING")


def _field_grounding(
    field: SourceAssertionField,
    refs: tuple[str, ...],
    spans: dict[str, SourceEvidenceSpanV31],
    offsets: dict[str, tuple[int, int]],
    candidate: SourceAtomicAssertionCandidateV1,
    *,
    polarity_basis_valid: bool = False,
) -> SourceFieldGrounding:
    valid = all(ref in spans and ref in offsets for ref in refs)
    value_text: str | None = None
    if field is SourceAssertionField.SUBJECT:
        value_text = candidate.subject.source_representation
    elif field is SourceAssertionField.PREDICATE:
        value_text = candidate.predicate.source_representation
    elif field is SourceAssertionField.VALUE and candidate.value.kind.value in {"text", "entity"}:
        value_text = candidate.value.text
    textual = valid and value_text is not None and all(
        value_text in spans[ref].exact_text for ref in refs
    )
    if field is SourceAssertionField.POLARITY:
        # Polarity has no standalone lexical value. Its source basis is the
        # cited assertion evidence bound to the same assertion and overlapping
        # the cited relation-operator span. This is reference/graph validation,
        # not an inference from the absence of a negation token.
        textual = valid and polarity_basis_valid
    # The v3.3 semantic binding was checked against the exact target, role and
    # source span before this function is called. This establishes source-side
    # grounding only; it is not target-claim entailment or verification.
    semantic = (
        SourceGroundingState.VALID
        if valid and (field is not SourceAssertionField.POLARITY or polarity_basis_valid)
        else SourceGroundingState.UNKNOWN
    )
    structural = valid and bool(refs)
    return SourceFieldGrounding(
        field=field,
        evidence_refs=tuple(sorted(set(refs))),
        textual_grounding=(
            SourceGroundingState.INVALID
            if not valid
            else SourceGroundingState.VALID
            if textual
            else SourceGroundingState.UNKNOWN
        ),
        semantic_support=semantic,
        structural_grounding=(
            SourceGroundingState.VALID if structural else SourceGroundingState.INVALID
        ),
        reason_codes=(
            *(
                (
                    "ASSERTION_LEVEL_POLARITY_BASIS",
                )
                if field is SourceAssertionField.POLARITY and polarity_basis_valid
                else ("EXACT_LEXICAL_GROUNDING",)
                if textual
                else ("TEXTUAL_GROUNDING_UNRESOLVED",)
            ),
            "V33_SEMANTIC_BINDING_VALIDATED",
        ),
    )


def _polarity_basis_valid(
    candidate: SourceAtomicAssertionCandidateV1,
    extraction: SourceExtractionResponseV33,
    offsets: dict[str, tuple[int, int]],
) -> bool:
    """Validate field-specific assertion and relation evidence for polarity.

    The frozen extraction contract has no polarity-specific role. It does have
    an assertion-evidence binding for the polarity field and a relation-operator
    binding for the predicate. Requiring both to resolve to this exact source
    assertion and overlapping source spans supplies an assertion-level basis
    without adding a role, changing v3.3, or treating missing negation text as
    proof of positive polarity.
    """
    if candidate.polarity.value == "unknown":
        return False
    assertion_evidence_refs = set(
        candidate.field_evidence_refs.get(SourceAssertionField.POLARITY, ())
    )
    predicate_evidence_refs = set(
        candidate.field_evidence_refs.get(SourceAssertionField.PREDICATE, ())
    )
    if not assertion_evidence_refs or not predicate_evidence_refs:
        return False
    assertion = next(
        (
            item
            for item in extraction.source_assertions
            if item.source_assertion_ref == candidate.source_assertion_ref
        ),
        None,
    )
    if assertion is None:
        return False
    if assertion.assertion_status.value == "UNKNOWN":
        return False
    assertion_refs = set(assertion.evidence_refs)
    if not assertion_evidence_refs.issubset(assertion_refs):
        return False
    bindings = extraction.evidence_bindings
    span_text_by_ref = {item.evidence_ref: item.exact_text for item in extraction.evidence_spans}
    polarity_bindings = {
        item.evidence_ref
        for item in bindings
        if item.target_kind.value == "ASSERTION"
        and item.target_ref == candidate.source_assertion_ref
        and item.semantic_role is SemanticEvidenceRoleV31.ASSERTION_EVIDENCE
        and item.evidence_ref in assertion_evidence_refs
    }
    predicate_bindings = {
        item.evidence_ref
        for item in bindings
        if item.target_kind.value == "ASSERTION"
        and item.target_ref == candidate.source_assertion_ref
        and item.semantic_role is SemanticEvidenceRoleV31.RELATION_OPERATOR
        and item.evidence_ref in predicate_evidence_refs
        and candidate.predicate.source_representation
        in span_text_by_ref.get(item.evidence_ref, "")
    }
    if not polarity_bindings or not predicate_bindings:
        return False
    return any(
        max(offsets[polarity_ref][0], offsets[predicate_ref][0])
        < min(offsets[polarity_ref][1], offsets[predicate_ref][1])
        for polarity_ref in polarity_bindings
        for predicate_ref in predicate_bindings
        if polarity_ref in offsets and predicate_ref in offsets
    )


def _normalize_composition(
    composition: SourceAssertionComposition | None,
    local_to_ref: dict[str, str],
) -> SourceAssertionComposition | None:
    if composition is None:
        return None
    group_refs = {group.group_ref for group in composition.groups}
    _validate_composition_cycles(composition)
    for group in composition.groups:
        for member in group.members:
            if member.assertion_ref is not None and member.assertion_ref not in local_to_ref:
                raise ValueError("COMPOSITION_ASSERTION_REF_INVALID")
            if member.group_ref is not None and member.group_ref not in group_refs:
                raise ValueError("COMPOSITION_GROUP_REF_INVALID")
    for member in composition.root_members:
        if member.assertion_ref is not None and member.assertion_ref not in local_to_ref:
            raise ValueError("COMPOSITION_ROOT_ASSERTION_REF_INVALID")
        if member.group_ref is not None and member.group_ref not in group_refs:
            raise ValueError("COMPOSITION_ROOT_GROUP_REF_INVALID")
    groups = tuple(
        group.model_copy(
            update={
                "members": tuple(
                    member.model_copy(
                        update={
                            "assertion_ref": local_to_ref[member.assertion_ref]
                            if member.assertion_ref is not None
                            else None
                        }
                    )
                    for member in group.members
                )
            }
        )
        for group in composition.groups
    )
    roots = tuple(
        member.model_copy(
            update={
                "assertion_ref": local_to_ref[member.assertion_ref]
                if member.assertion_ref is not None
                else None
            }
        )
        for member in composition.root_members
    )
    normalized = composition.model_copy(update={"groups": groups, "root_members": roots})
    return _normalize_composition_eligibility(normalized)


def _normalize_composition_eligibility(
    composition: SourceAssertionComposition,
) -> SourceAssertionComposition:
    groups = {group.group_ref: group for group in composition.groups}
    group_context: dict[str, SourceAssertionEligibility | None] = {}

    def combine(
        current: SourceAssertionEligibility | None,
        inherited: SourceAssertionEligibility | None,
    ) -> SourceAssertionEligibility | None:
        if current is SourceAssertionEligibility.NOT_INDEPENDENTLY_ASSERTED:
            return current
        if inherited is SourceAssertionEligibility.NOT_INDEPENDENTLY_ASSERTED:
            return inherited
        if (
            current is SourceAssertionEligibility.UNKNOWN
            or inherited is SourceAssertionEligibility.UNKNOWN
        ):
            return SourceAssertionEligibility.UNKNOWN
        return current

    def enter_group(
        group_ref: str, inherited: SourceAssertionEligibility | None
    ) -> None:
        group = groups[group_ref]
        group_inherited = inherited
        if group.operator in {AssertionGroupOperator.OR, AssertionGroupOperator.XOR}:
            group_inherited = SourceAssertionEligibility.NOT_INDEPENDENTLY_ASSERTED
        elif group.operator is AssertionGroupOperator.UNKNOWN:
            group_inherited = combine(group_inherited, SourceAssertionEligibility.UNKNOWN)
        group_context[group_ref] = combine(group_context.get(group_ref), group_inherited)
        for child in group.members:
            if child.group_ref is not None:
                enter_group(child.group_ref, group_inherited)

    for root in composition.root_members:
        if root.group_ref is not None:
            enter_group(root.group_ref, None)

    def update_leaf(member: Any, inherited: SourceAssertionEligibility | None) -> Any:
        return member.model_copy(
            update={"eligibility": combine(member.eligibility, inherited)}
        )

    normalized_groups = tuple(
        group.model_copy(
            update={
                "members": tuple(
                    update_leaf(member, group_context.get(group.group_ref))
                    if member.assertion_ref is not None
                    else member
                    for member in group.members
                )
            }
        )
        for group in composition.groups
    )
    normalized_roots = tuple(
        update_leaf(member, None) if member.assertion_ref is not None else member
        for member in composition.root_members
    )
    return composition.model_copy(
        update={"groups": normalized_groups, "root_members": normalized_roots}
    )


def _validate_composition_cycles(composition: SourceAssertionComposition) -> None:
    groups = {group.group_ref: group for group in composition.groups}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(group_ref: str) -> None:
        if group_ref in visiting:
            raise ValueError("COMPOSITION_CYCLE")
        if group_ref in visited:
            return
        visiting.add(group_ref)
        for member in groups[group_ref].members:
            if member.group_ref is not None:
                visit(member.group_ref)
        visiting.remove(group_ref)
        visited.add(group_ref)

    for member in composition.root_members:
        if member.group_ref is not None:
            visit(member.group_ref)
    if visited != set(groups):
        raise ValueError("UNREFERENCED_COMPOSITION_GROUP")


def _validate_v33_graph(
    response: SourceExtractionResponseV33,
    evidence_spans: dict[str, SourceEvidenceSpanV31],
) -> None:
    evidence_refs = set(evidence_spans)
    assertion_refs = {item.source_assertion_ref for item in response.source_assertions}
    context_refs = {item.context_ref for item in response.source_contexts}
    attachment_refs = {item.attachment_ref for item in response.source_attachments}
    scope_refs = {item.source_scope_ref for item in response.source_scopes}
    relation_refs = {item.relation_ref for item in response.semantic_relations}
    if len(assertion_refs) != len(response.source_assertions):
        raise ValueError("DUPLICATE_SOURCE_ASSERTION_REF")
    if len(context_refs) != len(response.source_contexts):
        raise ValueError("DUPLICATE_SOURCE_CONTEXT_REF")
    if len(attachment_refs) != len(response.source_attachments):
        raise ValueError("DUPLICATE_SOURCE_ATTACHMENT_REF")
    if len(scope_refs) != len(response.source_scopes):
        raise ValueError("DUPLICATE_SOURCE_SCOPE_REF")
    if len(relation_refs) != len(response.semantic_relations):
        raise ValueError("DUPLICATE_SEMANTIC_RELATION_REF")
    for assertion in response.source_assertions:
        if not set(assertion.evidence_refs).issubset(evidence_refs):
            raise ValueError("SOURCE_ASSERTION_EVIDENCE_REF_INVALID")
        if assertion.source_scope_ref is not None and assertion.source_scope_ref not in scope_refs:
            raise ValueError("SOURCE_ASSERTION_SCOPE_REF_INVALID")
    for attachment in response.source_attachments:
        if attachment.context_ref not in context_refs:
            raise ValueError("SOURCE_ATTACHMENT_CONTEXT_REF_INVALID")
        if not set(attachment.source_assertion_refs).issubset(assertion_refs):
            raise ValueError("SOURCE_ATTACHMENT_ASSERTION_REF_INVALID")
        if not set(attachment.evidence_refs).issubset(evidence_refs):
            raise ValueError("SOURCE_ATTACHMENT_EVIDENCE_REF_INVALID")
        if (
            attachment.source_scope_ref is not None
            and attachment.source_scope_ref not in scope_refs
        ):
            raise ValueError("SOURCE_ATTACHMENT_SCOPE_REF_INVALID")
    for scope in response.source_scopes:
        if not set(scope.source_assertion_refs).issubset(assertion_refs):
            raise ValueError("SOURCE_SCOPE_ASSERTION_REF_INVALID")
        if not set(scope.evidence_refs).issubset(evidence_refs):
            raise ValueError("SOURCE_SCOPE_EVIDENCE_REF_INVALID")
    for binding in response.evidence_bindings:
        if binding.evidence_ref not in evidence_refs:
            raise ValueError("SEMANTIC_BINDING_EVIDENCE_REF_INVALID")
        targets = {
            "ASSERTION": assertion_refs,
            "CONTEXT": context_refs,
            "ATTACHMENT": attachment_refs,
            "SCOPE": scope_refs,
            "RELATION": relation_refs,
        }.get(binding.target_kind.value, set())
        if binding.target_ref not in targets:
            raise ValueError("SEMANTIC_BINDING_TARGET_REF_INVALID")
    for relation in response.semantic_relations:
        source_refs = {
            "ASSERTION": assertion_refs,
            "CONTEXT": context_refs,
            "ATTACHMENT": attachment_refs,
            "SCOPE": scope_refs,
            "RELATION": relation_refs,
        }.get(relation.source_kind.value, set())
        target_refs = {
            "ASSERTION": assertion_refs,
            "CONTEXT": context_refs,
            "ATTACHMENT": attachment_refs,
            "SCOPE": scope_refs,
            "RELATION": relation_refs,
        }.get(relation.target_kind.value, set())
        if relation.source_ref not in source_refs or relation.target_ref not in target_refs:
            raise ValueError("SEMANTIC_RELATION_TARGET_REF_INVALID")
        if not set(relation.evidence_refs).issubset(evidence_refs):
            raise ValueError("SEMANTIC_RELATION_EVIDENCE_REF_INVALID")


def _apply_composition_eligibility(
    assertions: list[SourceAtomicAssertion],
    composition: SourceAssertionComposition | None,
) -> list[SourceAtomicAssertion]:
    if composition is None:
        return assertions
    groups = {group.group_ref: group for group in composition.groups}
    required: dict[str, SourceAssertionEligibility] = {}

    def apply_member(member: Any, inherited: SourceAssertionEligibility | None = None) -> None:
        if member.assertion_ref is not None:
            current = member.eligibility or SourceAssertionEligibility.UNKNOWN
            if inherited is SourceAssertionEligibility.NOT_INDEPENDENTLY_ASSERTED:
                current = SourceAssertionEligibility.NOT_INDEPENDENTLY_ASSERTED
            elif inherited is SourceAssertionEligibility.UNKNOWN:
                current = SourceAssertionEligibility.UNKNOWN
            previous = required.get(member.assertion_ref)
            if previous is SourceAssertionEligibility.NOT_INDEPENDENTLY_ASSERTED or (
                previous is SourceAssertionEligibility.UNKNOWN
                and current is SourceAssertionEligibility.INDEPENDENTLY_ASSERTED
            ):
                current = previous
            required[member.assertion_ref] = current
            return
        group = groups[member.group_ref]
        nested = inherited
        if group.operator in {AssertionGroupOperator.OR, AssertionGroupOperator.XOR}:
            nested = SourceAssertionEligibility.NOT_INDEPENDENTLY_ASSERTED
        elif group.operator is AssertionGroupOperator.UNKNOWN:
            nested = SourceAssertionEligibility.UNKNOWN
        for child in group.members:
            apply_member(child, nested)

    for root in composition.root_members:
        apply_member(root)
    return [
        item.model_copy(
            update={
                "verification_eligibility": required.get(
                    item.assertion_ref, item.verification_eligibility
                )
            }
        )
        for item in assertions
    ]


def _assertion_ref(
    runtime_input: SourceSemanticRuntimeInput,
    candidate: SourceAtomicAssertionCandidateV1,
    contexts: tuple[TruthConditionContext, ...],
) -> str:
    context_refs = set(candidate.truth_condition_context_refs)
    selected_contexts = [
        item.model_dump(mode="json") for item in contexts if item.context_ref in context_refs
    ]
    payload = {
        "schema": SOURCE_ATOMIC_ASSERTION_SCHEMA_VERSION,
        "source": runtime_input.source_ref,
        "snapshot": runtime_input.snapshot_ref,
        "canonical_span": runtime_input.canonical_evidence_span.span_ref
        if runtime_input.canonical_evidence_span
        else None,
        "source_assertion_ref": candidate.source_assertion_ref,
        "evidence_refs": sorted(candidate.evidence_refs),
        "subject": candidate.subject.model_dump(mode="json"),
        "predicate": candidate.predicate.model_dump(mode="json"),
        "value": candidate.value.model_dump(mode="json"),
        "polarity": candidate.polarity.value,
        "contexts": selected_contexts,
        "verification_eligibility": candidate.verification_eligibility.value,
        "composition_ref": candidate.composition_ref,
        "field_evidence_refs": {
            field.value: sorted(refs)
            for field, refs in sorted(
                candidate.field_evidence_refs.items(), key=lambda item: item[0].value
            )
        },
    }
    normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _build_anchors(
    extraction: SourceExtractionResponseV33,
    spans: dict[str, SourceEvidenceSpanV31],
    offsets: dict[str, tuple[int, int]],
) -> tuple[SourceAssertionAnchor, ...]:
    anchors: list[SourceAssertionAnchor] = []
    for assertion in extraction.source_assertions:
        for ref in assertion.evidence_refs:
            if ref not in spans or ref not in offsets:
                continue
            span = spans[ref]
            start, end = offsets[ref]
            anchors.append(
                SourceAssertionAnchor(
                    source_assertion_ref=assertion.source_assertion_ref,
                    source_evidence=SpanEvidence(
                        exact_text=span.exact_text,
                        occurrence_index=span.occurrence_index,
                        advisory_start=start,
                        advisory_end=end,
                    ),
                    grounded_span=GroundedSpan(
                        exact_text=span.exact_text,
                        start_offset=start,
                        end_offset=end,
                        occurrence_index=span.occurrence_index,
                    ),
                    source_scope_ref=assertion.source_scope_ref,
                    assertion_status=AssertionStatus(assertion.assertion_status.value),
                    grounding_status="GROUNDING_VALID",
                )
            )
    return tuple(
        sorted(
            anchors,
            key=lambda item: (
                item.source_assertion_ref,
                item.grounded_span.start_offset if item.grounded_span else -1,
            ),
        )
    )


def _occurrences(text: str, needle: str) -> tuple[int, ...]:
    if not needle:
        return ()
    positions: list[int] = []
    cursor = 0
    while True:
        position = text.find(needle, cursor)
        if position < 0:
            return tuple(positions)
        positions.append(position)
        cursor = position + 1


def _usage(usage: Any, field: str) -> int:
    value = getattr(usage, field, 0)
    return value if isinstance(value, int) and value >= 0 else 0
