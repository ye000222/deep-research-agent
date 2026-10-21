"""One-call v3.4 source runtime with deterministic typed-value grounding."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any

from pydantic import ConfigDict, ValidationError

from app.domain.source_atomic_assertion import (
    SourceAssertionField,
    SourceFieldGrounding,
    SourceGroundingState,
)
from app.domain.source_context_extraction_v3_1 import SemanticEvidenceRoleV31
from app.domain.source_context_extraction_v3_4 import (
    SOURCE_CONTEXT_EXTRACTION_DOMAIN_V34,
    SOURCE_CONTEXT_EXTRACTION_INSTRUCTIONS_V34,
    SOURCE_CONTEXT_EXTRACTION_PROMPT_V34,
    SOURCE_CONTEXT_EXTRACTION_SCHEMA_V34,
    SOURCE_SEMANTIC_RUNTIME_PROMPT_V2,
    SOURCE_SEMANTIC_RUNTIME_RESPONSE_V2,
    SourceExtractionResponseV34,
    SourceSemanticRuntimeResponseV2,
    v2_response_to_v1_payload,
)
from app.domain.source_typed_value_grounding import (
    SOURCE_TYPED_VALUE_GROUNDING_CONTRACT_VERSION,
    SOURCE_TYPED_VALUE_PARSER_VERSION,
    SourceEvidenceSpanInput,
    SourceTypedValueGroundingAssessment,
    TypedValueEvidenceBinding,
    assess_typed_value_grounding,
)
from app.services.source_semantic_runtime_v3_3 import (
    SOURCE_SEMANTIC_RUNTIME_MAX_OUTPUT_TOKENS,
    SourceSemanticExtractionV33Runtime,
    SourceSemanticRuntimeInput,
    SourceSemanticRuntimeResult,
    SourceSemanticRuntimeStatus,
    _assemble,
)


class SourceSemanticRuntimeResultV34(SourceSemanticRuntimeResult):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_version: str = "source-semantic-runtime-v2"
    source_extraction: SourceExtractionResponseV34 | None = None
    typed_value_grounding_assessments: tuple[SourceTypedValueGroundingAssessment, ...] = ()
    typed_value_bindings: tuple[TypedValueEvidenceBinding, ...] = ()


class SourceSemanticExtractionV34Runtime(SourceSemanticExtractionV33Runtime):
    """Strict one-call runtime; old v3.3 assembly remains the base projection."""

    async def extract(
        self, runtime_input: SourceSemanticRuntimeInput
    ) -> SourceSemanticRuntimeResultV34:
        # Do not call the parent implementation: its response contract is frozen v3.3.
        if self._called:
            return _v34_result(
                runtime_input, SourceSemanticRuntimeStatus.NOT_PROCESSED, "CALL_LIMIT_REACHED"
            )
        self._called = True
        canonical = runtime_input.canonical_evidence_span
        if canonical is None:
            return _v34_result(
                runtime_input, SourceSemanticRuntimeStatus.NOT_PROCESSED, "SOURCE_SPAN_UNAVAILABLE"
            )

        from uuid import uuid4

        from app.domain.providers import CanonicalModelRequest, ContentPart

        request = CanonicalModelRequest(
            task_kind="source_semantic_extraction_v3_4",
            role="source_semantic_runtime_v3_4",
            model=self._model,
            instructions=SOURCE_CONTEXT_EXTRACTION_INSTRUCTIONS_V34,
            content_parts=(
                ContentPart(
                    kind="json",
                    value={
                        "source_ref": runtime_input.source_ref,
                        "snapshot_ref": runtime_input.snapshot_ref,
                        "canonical_span": {
                            "span_ref": canonical.span_ref,
                            "char_start": canonical.char_start,
                            "char_end": canonical.char_end,
                            "exact_text": canonical.exact_text,
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
            response_contract=SourceSemanticRuntimeResponseV2.model_json_schema(),
            generation_parameters={"temperature": 0.0, "reasoning_enabled": False},
            max_output_tokens=SOURCE_SEMANTIC_RUNTIME_MAX_OUTPUT_TOKENS,
            context_manifest_id=uuid4(),
            metadata={
                "schema_version": SOURCE_SEMANTIC_RUNTIME_RESPONSE_V2,
                "source_schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V34,
                "source_domain_version": SOURCE_CONTEXT_EXTRACTION_DOMAIN_V34,
                "source_prompt_version": SOURCE_CONTEXT_EXTRACTION_PROMPT_V34,
                "value_grounding_contract": SOURCE_TYPED_VALUE_GROUNDING_CONTRACT_VERSION,
                "numeric_parser_version": SOURCE_TYPED_VALUE_PARSER_VERSION,
                "runtime_prompt_version": SOURCE_SEMANTIC_RUNTIME_PROMPT_V2,
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
            return _v34_result(
                runtime_input,
                SourceSemanticRuntimeStatus.FAILED,
                "GATEWAY_ERROR",
                model_call_count=1,
                latency_ms=round((perf_counter() - started) * 1000),
            )
        latency = round((perf_counter() - started) * 1000)
        payload = getattr(raw, "parsed_object", None)
        if payload is None:
            text = getattr(raw, "text", None)
            try:
                payload = json.loads(text) if isinstance(text, str) else None
            except (json.JSONDecodeError, TypeError):
                payload = None
        if not isinstance(payload, dict):
            return _v34_result(
                runtime_input,
                SourceSemanticRuntimeStatus.INVALID,
                "STRUCTURED_OUTPUT_INVALID",
                1,
                latency,
            )
        try:
            response = SourceSemanticRuntimeResponseV2.model_validate(payload)
            legacy_payload = v2_response_to_v1_payload(response)
            from app.domain.source_atomic_assertion import SourceSemanticRuntimeResponseV1

            legacy_response = SourceSemanticRuntimeResponseV1.model_validate(legacy_payload)
            base_result = _assemble(runtime_input, legacy_response)
            assessments, assertions = _ground_response(runtime_input, response, base_result)
        except (ValidationError, ValueError, KeyError, TypeError):
            return _v34_result(
                runtime_input,
                SourceSemanticRuntimeStatus.INVALID,
                "VALUE_GROUNDING_OR_PROVENANCE_INVALID",
                1,
                latency,
            )
        usage = getattr(raw, "usage", None)
        dump = base_result.model_dump(mode="python")
        dump.update(
            runtime_version="source-semantic-runtime-v2",
            source_extraction=response.source_extraction,
            typed_value_bindings=response.source_extraction.typed_value_bindings,
            assertions=assertions,
            typed_value_grounding_assessments=assessments,
            schema_versions={
                **base_result.schema_versions,
                "runtime": "source-semantic-runtime-v2",
                "response": SOURCE_SEMANTIC_RUNTIME_RESPONSE_V2,
                "source": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V34,
                "source_domain": SOURCE_CONTEXT_EXTRACTION_DOMAIN_V34,
                "source_prompt": SOURCE_CONTEXT_EXTRACTION_PROMPT_V34,
                "value_grounding": SOURCE_TYPED_VALUE_GROUNDING_CONTRACT_VERSION,
                "numeric_parser": SOURCE_TYPED_VALUE_PARSER_VERSION,
            },
            model_call_count=1,
            input_tokens=_usage(usage, "input_tokens"),
            output_tokens=_usage(usage, "output_tokens"),
            latency_ms=latency,
        )
        result = SourceSemanticRuntimeResultV34.model_validate(dump)
        if any(item.status is not SourceGroundingState.VALID for item in assessments):
            unresolved = tuple(
                sorted(set((*result.unresolved_items, "TYPED_VALUE_GROUNDING_UNRESOLVED")))
            )
            result = result.model_copy(
                update={
                    "runtime_status": SourceSemanticRuntimeStatus.COMPLETED_WITH_UNRESOLVED,
                    "unresolved_items": unresolved,
                }
            )
        return result


def _ground_response(
    runtime_input: SourceSemanticRuntimeInput,
    response: SourceSemanticRuntimeResponseV2,
    base_result: SourceSemanticRuntimeResult,
) -> tuple[tuple[SourceTypedValueGroundingAssessment, ...], tuple[Any, ...]]:
    extraction = response.source_extraction
    spans = tuple(_resolved_span_input(item, runtime_input) for item in extraction.evidence_spans)
    binding_by_ref: dict[str, list[Any]] = {}
    for binding in extraction.typed_value_bindings:
        binding_by_ref.setdefault(binding.assertion_ref, []).append(binding)
    candidate_refs = {item.source_assertion_ref for item in response.assertion_candidates}
    if set(binding_by_ref) - candidate_refs:
        raise ValueError("TYPED_VALUE_BINDING_ASSERTION_REF_INVALID")
    legacy_bindings = extraction.evidence_bindings
    assessments: list[SourceTypedValueGroundingAssessment] = []
    updated: list[Any] = []
    assertion_by_anchor = {
        anchor: assertion
        for assertion in base_result.assertions
        for anchor in assertion.source_assertion_anchor_refs
    }
    for candidate in response.assertion_candidates:
        assertion = assertion_by_anchor.get(candidate.source_assertion_ref)
        if assertion is None:
            raise ValueError("TYPED_VALUE_ASSERTION_REF_INVALID")
        value_refs = set(candidate.field_evidence_refs.get(SourceAssertionField.VALUE, ()))
        proposed = tuple(binding_by_ref.get(candidate.source_assertion_ref, ()))
        for binding in proposed:
            if binding.evidence_ref not in value_refs:
                raise ValueError("TYPED_VALUE_BINDING_NOT_IN_VALUE_FIELD_REFS")
            if not any(
                item.evidence_ref == binding.evidence_ref
                and item.target_kind.value == "ASSERTION"
                and item.target_ref == candidate.source_assertion_ref
                and item.semantic_role is SemanticEvidenceRoleV31.ASSERTION_EVIDENCE
                for item in legacy_bindings
            ):
                raise ValueError("TYPED_VALUE_SEMANTIC_BINDING_MISSING")
        assessment = assess_typed_value_grounding(
            assertion_ref=candidate.source_assertion_ref,
            value=candidate.value,
            bindings=proposed,
            spans=spans,
            source_ref=runtime_input.source_ref,
            snapshot_ref=runtime_input.snapshot_ref,
        ).model_copy(update={"assertion_ref": assertion.assertion_ref})
        assessments.append(assessment)
        by_field = {item.field: item for item in assertion.field_grounding}
        previous = by_field.get(SourceAssertionField.VALUE)
        if previous is None:
            raise ValueError("VALUE_FIELD_GROUNDING_MISSING")
        textual, semantic, structural = _aggregate_component_states(assessment)
        by_field[SourceAssertionField.VALUE] = SourceFieldGrounding(
            field=SourceAssertionField.VALUE,
            evidence_refs=previous.evidence_refs,
            textual_grounding=textual,
            semantic_support=semantic,
            structural_grounding=structural,
            reason_codes=(
                *assessment.reason_codes,
                "VALUE_LEVEL_BINDING_VALIDATED",
            ),
        )
        updated.append(
            assertion.model_copy(
                update={
                    "field_grounding": tuple(
                        by_field[field] for field in sorted(by_field, key=lambda item: item.value)
                    )
                }
            )
        )
    updated_refs = {value.assertion_ref for value in updated}
    final_assertions = tuple(
        sorted(
            (
                *updated,
                *(
                    item
                    for item in base_result.assertions
                    if item.assertion_ref not in updated_refs
                ),
            ),
            key=lambda item: item.assertion_ref,
        )
    )
    return tuple(sorted(assessments, key=lambda item: item.assertion_ref)), final_assertions


def _resolved_span_input(
    item: Any, runtime_input: SourceSemanticRuntimeInput
) -> SourceEvidenceSpanInput:
    canonical = runtime_input.canonical_evidence_span
    if canonical is None:
        raise ValueError("SOURCE_SPAN_UNAVAILABLE")
    cursor = 0
    selected_start: int | None = None
    for _index in range(item.occurrence_index + 1):
        offset = canonical.exact_text.find(item.exact_text, cursor)
        if offset < 0:
            raise ValueError("SOURCE_EVIDENCE_SPAN_NOT_IN_CANONICAL_SPAN")
        selected_start = offset
        cursor = offset + 1
    assert selected_start is not None
    start = canonical.char_start + selected_start
    return SourceEvidenceSpanInput(
        evidence_ref=item.evidence_ref,
        exact_text=item.exact_text,
        occurrence_index=item.occurrence_index,
        char_start=start,
        char_end=start + len(item.exact_text),
    )


def rehydrate_typed_value_grounding(
    result: SourceSemanticRuntimeResultV34,
    canonical_span: Any,
) -> SourceSemanticRuntimeResultV34:
    """Re-run deterministic normalization from frozen bindings and receipt text."""

    if (
        canonical_span is None
        or result.canonical_span_ref != canonical_span.span_ref
        or result.source_ref != canonical_span.source_ref
        or result.snapshot_ref != canonical_span.snapshot_ref
    ):
        raise ValueError("TYPED_VALUE_REHYDRATION_PROVENANCE_MISMATCH")
    spans_by_ref: dict[str, SourceEvidenceSpanInput] = {}
    bindings_by_assertion: dict[str, list[Any]] = {}
    for binding in result.typed_value_bindings:
        left = binding.char_start - canonical_span.char_start
        right = binding.char_end - canonical_span.char_start
        if left < 0 or right > len(canonical_span.exact_text) or right <= left:
            raise ValueError("TYPED_VALUE_REHYDRATION_SPAN_OUT_OF_BOUNDS")
        exact = canonical_span.exact_text[left:right]
        proposed = SourceEvidenceSpanInput(
            evidence_ref=binding.evidence_ref,
            exact_text=exact,
            occurrence_index=binding.occurrence_index,
            char_start=binding.char_start,
            char_end=binding.char_end,
        )
        previous = spans_by_ref.get(binding.evidence_ref)
        if previous is not None and previous != proposed:
            raise ValueError("TYPED_VALUE_REHYDRATION_SPAN_REF_COLLISION")
        spans_by_ref[binding.evidence_ref] = proposed
        bindings_by_assertion.setdefault(binding.assertion_ref, []).append(binding)
    old_by_anchor = {
        anchor: assertion
        for assertion in result.assertions
        for anchor in assertion.source_assertion_anchor_refs
    }
    old_assessments = {
        item.assertion_ref: item for item in result.typed_value_grounding_assessments
    }
    updated: list[Any] = []
    new_assessments: list[SourceTypedValueGroundingAssessment] = []
    for anchor, assertion in old_by_anchor.items():
        assessment = assess_typed_value_grounding(
            assertion_ref=anchor,
            value=assertion.value,
            bindings=tuple(bindings_by_assertion.get(anchor, ())),
            spans=tuple(spans_by_ref.values()),
            source_ref=result.source_ref,
            snapshot_ref=result.snapshot_ref,
        ).model_copy(update={"assertion_ref": assertion.assertion_ref})
        previous_assessment = old_assessments.get(assertion.assertion_ref)
        if previous_assessment is None or previous_assessment != assessment:
            raise ValueError("TYPED_VALUE_REHYDRATION_RESULT_MISMATCH")
        new_assessments.append(assessment)
        by_field = {item.field: item for item in assertion.field_grounding}
        old_value = by_field.get(SourceAssertionField.VALUE)
        if old_value is None:
            raise ValueError("TYPED_VALUE_REHYDRATION_VALUE_GROUNDING_MISSING")
        textual, semantic, structural = _aggregate_component_states(assessment)
        by_field[SourceAssertionField.VALUE] = SourceFieldGrounding(
            field=SourceAssertionField.VALUE,
            evidence_refs=old_value.evidence_refs,
            textual_grounding=textual,
            semantic_support=semantic,
            structural_grounding=structural,
            # Keep the durable projection byte-for-byte semantically stable;
            # rehydration is validation, not a new grounding event.
            reason_codes=old_value.reason_codes,
        )
        updated.append(
            assertion.model_copy(
                update={
                    "field_grounding": tuple(
                        by_field[field] for field in sorted(by_field, key=lambda item: item.value)
                    )
                }
            )
        )
    updated_by_ref = {item.assertion_ref: item for item in updated}
    assertions = tuple(updated_by_ref.get(item.assertion_ref, item) for item in result.assertions)
    return result.model_copy(
        update={
            "assertions": assertions,
            "typed_value_grounding_assessments": tuple(
                sorted(new_assessments, key=lambda item: item.assertion_ref)
            ),
        }
    )


def _aggregate_component_states(
    assessment: SourceTypedValueGroundingAssessment,
) -> tuple[SourceGroundingState, SourceGroundingState, SourceGroundingState]:
    if not assessment.components:
        return (assessment.status, assessment.status, assessment.status)
    return tuple(
        SourceGroundingState.INVALID
        if any(
            getattr(component, field) is SourceGroundingState.INVALID
            for component in assessment.components
        )
        else SourceGroundingState.VALID
        if all(
            getattr(component, field) is SourceGroundingState.VALID
            for component in assessment.components
        )
        else SourceGroundingState.UNKNOWN
        for field in ("textual_grounding", "semantic_support", "structural_grounding")
    )  # type: ignore[return-value]


def _v34_result(
    runtime_input: SourceSemanticRuntimeInput,
    status: SourceSemanticRuntimeStatus,
    reason: str,
    model_call_count: int = 0,
    latency_ms: int = 0,
) -> SourceSemanticRuntimeResultV34:
    base = SourceSemanticRuntimeResult(
        runtime_status=status,
        source_ref=runtime_input.source_ref,
        snapshot_ref=runtime_input.snapshot_ref,
        canonical_span_ref=(
            runtime_input.canonical_evidence_span.span_ref
            if runtime_input.canonical_evidence_span
            else None
        ),
        context_window_ref=(
            runtime_input.context_window.context_ref if runtime_input.context_window else None
        ),
        reason_codes=(reason,),
        model_call_count=model_call_count,
        latency_ms=latency_ms,
        schema_versions={
            "runtime": "source-semantic-runtime-v2",
            "response": SOURCE_SEMANTIC_RUNTIME_RESPONSE_V2,
            "source": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V34,
            "source_domain": SOURCE_CONTEXT_EXTRACTION_DOMAIN_V34,
            "source_prompt": SOURCE_CONTEXT_EXTRACTION_PROMPT_V34,
            "value_grounding": SOURCE_TYPED_VALUE_GROUNDING_CONTRACT_VERSION,
            "numeric_parser": SOURCE_TYPED_VALUE_PARSER_VERSION,
        },
    )
    payload = base.model_dump(mode="python")
    payload["runtime_version"] = "source-semantic-runtime-v2"
    return SourceSemanticRuntimeResultV34.model_validate(payload)


def _usage(usage: Any, key: str) -> int:
    value = usage.get(key, 0) if isinstance(usage, dict) else getattr(usage, key, 0)
    return value if isinstance(value, int) and value >= 0 else 0
