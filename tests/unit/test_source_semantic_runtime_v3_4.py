from __future__ import annotations

import hashlib
from typing import Any

import pytest
from app.domain.document_evidence_context import (
    CanonicalEvidenceSpan,
    SourceLocator,
    SourceLocatorType,
)
from app.domain.providers import AdapterType, CanonicalModelResult, TokenUsage, UsageAccuracy
from app.domain.source_context_extraction_v3_4 import (
    SOURCE_CONTEXT_EXTRACTION_SCHEMA_V34,
    SOURCE_SEMANTIC_RUNTIME_RESPONSE_V2,
)
from app.domain.source_typed_value_grounding import SourceGroundingState
from app.services.source_semantic_runtime_v3_3 import SourceSemanticRuntimeInput
from app.services.source_semantic_runtime_v3_4 import (
    SourceSemanticExtractionV34Runtime,
    SourceSemanticRuntimeStatus,
    rehydrate_typed_value_grounding,
)
from pydantic import SecretStr

SOURCE = "source:synthetic"
SNAPSHOT = hashlib.sha256(b"snapshot").hexdigest()
CANONICAL_TEXT = "Device success rate is 12.4%."
VALUE_START = CANONICAL_TEXT.index("12.4%")


def _response(
    value: str = "12.4%", *, typed_magnitude: float = 12.4, polarity: str = "asserted"
) -> dict[str, Any]:
    return {
        "schema_version": SOURCE_SEMANTIC_RUNTIME_RESPONSE_V2,
        "status": "RESOLVED",
        "source_extraction": {
            "schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V34,
            "status": "RESOLVED",
            "source_assertions": [
                {
                    "source_assertion_ref": "A1",
                    "evidence_refs": ["whole", "value"],
                    "assertion_status": "ASSERTED",
                }
            ],
            "source_contexts": [],
            "evidence_spans": [
                {"evidence_ref": "whole", "exact_text": CANONICAL_TEXT},
                {"evidence_ref": "value", "exact_text": value},
            ],
            "evidence_bindings": [
                {
                    "binding_ref": "b-subject",
                    "evidence_ref": "whole",
                    "semantic_role": "ASSERTION_EVIDENCE",
                    "target_kind": "ASSERTION",
                    "target_ref": "A1",
                },
                {
                    "binding_ref": "b-predicate",
                    "evidence_ref": "whole",
                    "semantic_role": "RELATION_OPERATOR",
                    "target_kind": "ASSERTION",
                    "target_ref": "A1",
                },
                {
                    "binding_ref": "b-value",
                    "evidence_ref": "value",
                    "semantic_role": "ASSERTION_EVIDENCE",
                    "target_kind": "ASSERTION",
                    "target_ref": "A1",
                },
                {
                    "binding_ref": "b-polarity",
                    "evidence_ref": "whole",
                    "semantic_role": "ASSERTION_EVIDENCE",
                    "target_kind": "ASSERTION",
                    "target_ref": "A1",
                },
            ],
            "typed_value_bindings": [
                {
                    "binding_ref": "tv-magnitude",
                    "assertion_ref": "A1",
                    "value_field_role": "VALUE",
                    "evidence_ref": "value",
                    "occurrence_index": 0,
                    "char_start": VALUE_START,
                    "char_end": VALUE_START + len(value),
                    "component_role": "NUMERIC_MAGNITUDE",
                    "source_ref": SOURCE,
                    "snapshot_ref": SNAPSHOT,
                    "contract_version": "source-typed-value-grounding-v1",
                },
                {
                    "binding_ref": "tv-unit",
                    "assertion_ref": "A1",
                    "value_field_role": "VALUE",
                    "evidence_ref": "value",
                    "occurrence_index": 0,
                    "char_start": VALUE_START,
                    "char_end": VALUE_START + len(value),
                    "component_role": "UNIT",
                    "source_ref": SOURCE,
                    "snapshot_ref": SNAPSHOT,
                    "contract_version": "source-typed-value-grounding-v1",
                },
            ],
            "semantic_relations": [],
            "measurement_bases": [],
            "unresolved_items": [],
        },
        "assertion_candidates": [
            {
                "local_ref": "candidate-1",
                "source_assertion_ref": "A1",
                "evidence_refs": ["whole", "value"],
                "subject": {
                    "source_representation": "Device",
                    "semantic_identity": {"status": "UNRESOLVED"},
                },
                "predicate": {
                    "source_representation": "success rate",
                    "semantic_identity": {"status": "UNRESOLVED"},
                },
                "value": {
                    "kind": "quantity",
                    "numeric": {"operator": "exact", "value": typed_magnitude, "unit": "%"},
                },
                "polarity": polarity,
                "field_evidence_refs": {
                    "SUBJECT": ["whole"],
                    "PREDICATE": ["whole"],
                    "VALUE": ["value"],
                    "POLARITY": ["whole"],
                },
                "truth_condition_context_refs": [],
                "verification_eligibility": "INDEPENDENTLY_ASSERTED",
            }
        ],
        "truth_condition_contexts": [],
        "composition": None,
        "unresolved_items": [],
    }


class FakeGateway:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.calls = 0
        self.kwargs: dict[str, Any] | None = None

    async def generate_structured(self, **kwargs: Any) -> CanonicalModelResult:
        self.calls += 1
        self.kwargs = kwargs
        return CanonicalModelResult(
            parsed_object=self.payload,
            usage=TokenUsage(
                input_tokens=3, output_tokens=5, total_tokens=8, accuracy=UsageAccuracy.EXACT
            ),
        )


def _input() -> SourceSemanticRuntimeInput:
    return SourceSemanticRuntimeInput(
        source_ref=SOURCE,
        snapshot_ref=SNAPSHOT,
        canonical_evidence_span=CanonicalEvidenceSpan(
            span_ref=hashlib.sha256(b"canonical").hexdigest(),
            document_ref=hashlib.sha256(b"document").hexdigest(),
            source_ref=SOURCE,
            snapshot_ref=SNAPSHOT,
            char_start=0,
            char_end=len(CANONICAL_TEXT),
            exact_text=CANONICAL_TEXT,
            content_sha256=hashlib.sha256(CANONICAL_TEXT.encode()).hexdigest(),
            source_locator=SourceLocator(locator_type=SourceLocatorType.UNKNOWN),
        ),
    )


def _run(coroutine: Any) -> Any:
    try:
        coroutine.send(None)
    except StopIteration as done:
        return done.value
    raise AssertionError("Fake Gateway unexpectedly awaited external work")


def _runtime(gateway: FakeGateway) -> SourceSemanticExtractionV34Runtime:
    return SourceSemanticExtractionV34Runtime(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://example.invalid/v1",
        api_key=SecretStr("fake-only"),
        model="fake-model",
    )


def test_v34_runtime_grounds_quantity_and_records_single_fake_call() -> None:
    gateway = FakeGateway(_response())
    result = _run(_runtime(gateway).extract(_input()))
    assert gateway.calls == 1
    assert gateway.kwargs["allow_regeneration"] is False
    assert gateway.kwargs["request"].task_kind == "source_semantic_extraction_v3_4"
    assert "NUMERIC_MAGNITUDE" in gateway.kwargs["request"].instructions
    # The numeric value is grounded; overall resolution remains conservative
    # because this fixture intentionally leaves semantic identity unresolved.
    assert result.runtime_status is SourceSemanticRuntimeStatus.COMPLETED_WITH_UNRESOLVED
    assert "IDENTITY_OR_ELIGIBILITY_UNRESOLVED" in result.unresolved_items
    assert result.model_call_count == 1
    assert result.typed_value_grounding_assessments[0].status is SourceGroundingState.VALID
    value_grounding = next(
        item for item in result.assertions[0].field_grounding if item.field.value == "VALUE"
    )
    assert value_grounding.textual_grounding is SourceGroundingState.VALID
    assert value_grounding.semantic_support is SourceGroundingState.VALID
    assert value_grounding.structural_grounding is SourceGroundingState.VALID
    assert result.schema_versions["source"] == SOURCE_CONTEXT_EXTRACTION_SCHEMA_V34


def test_wrong_percentage_value_is_invalid_without_retry_or_repair() -> None:
    gateway = FakeGateway(_response(typed_magnitude=12.5))
    result = _run(_runtime(gateway).extract(_input()))
    assert gateway.calls == 1
    assert result.typed_value_grounding_assessments[0].status is SourceGroundingState.INVALID
    value_grounding = next(
        item for item in result.assertions[0].field_grounding if item.field.value == "VALUE"
    )
    assert value_grounding.textual_grounding is SourceGroundingState.VALID
    assert value_grounding.semantic_support is SourceGroundingState.INVALID
    assert value_grounding.structural_grounding is SourceGroundingState.VALID


@pytest.mark.parametrize(
    "invalid_binding",
    ("unknown_assertion", "wrong_value_evidence", "span_mismatch"),
)
def test_invalid_typed_binding_provenance_never_becomes_valid(
    invalid_binding: str,
) -> None:
    payload = _response()
    binding = payload["source_extraction"]["typed_value_bindings"][0]
    if invalid_binding == "unknown_assertion":
        binding["assertion_ref"] = "MISSING_ASSERTION"
    elif invalid_binding == "wrong_value_evidence":
        binding["evidence_ref"] = "whole"
    else:
        binding["char_start"] = len(CANONICAL_TEXT) + 1
        binding["char_end"] = binding["char_start"] + len("12.4%")

    gateway = FakeGateway(payload)
    result = _run(_runtime(gateway).extract(_input()))

    assert gateway.calls == 1
    assert all(
        assessment.status is not SourceGroundingState.VALID
        for assessment in result.typed_value_grounding_assessments
    )
    if invalid_binding == "span_mismatch":
        assert result.runtime_status is SourceSemanticRuntimeStatus.COMPLETED_WITH_UNRESOLVED
        assert result.typed_value_grounding_assessments[0].status is SourceGroundingState.INVALID
    else:
        assert result.runtime_status is SourceSemanticRuntimeStatus.INVALID
        assert result.assertions == ()


def test_v34_gateway_exception_returns_failed_without_semantic_product() -> None:
    class FailingGateway:
        calls = 0

        async def generate_structured(self, **_kwargs: Any) -> Any:
            self.calls += 1
            raise RuntimeError("fake V34 extraction transport failure")

    gateway = FailingGateway()
    result = _run(_runtime(gateway).extract(_input()))

    assert gateway.calls == 1
    assert result.runtime_status is SourceSemanticRuntimeStatus.FAILED
    assert result.model_call_count == 1
    assert result.assertions == ()
    assert result.typed_value_grounding_assessments == ()


def test_durable_binding_rehydrates_to_identical_grounding_without_provider() -> None:
    gateway = FakeGateway(_response())
    original = _run(_runtime(gateway).extract(_input()))
    restored = original.model_copy(
        update={
            "source_evidence_spans": (),
            "source_extraction": original.source_extraction.model_copy(
                update={"evidence_spans": ()}
            ),
        }
    )
    replayed = rehydrate_typed_value_grounding(restored, _input().canonical_evidence_span)
    assert gateway.calls == 1
    assert replayed.typed_value_grounding_assessments == original.typed_value_grounding_assessments
    assert replayed.assertions == original.assertions


def test_v33_response_is_not_silently_upgraded_by_v34_runtime() -> None:
    legacy = {
        "schema_version": "source-semantic-runtime-response-v1",
        "status": "RESOLVED",
        "source_extraction": {
            "schema_version": "source-context-extraction-schema-v3.3",
        },
    }
    gateway = FakeGateway(legacy)
    result = _run(_runtime(gateway).extract(_input()))
    assert gateway.calls == 1
    assert result.runtime_status is SourceSemanticRuntimeStatus.INVALID
    assert result.assertions == ()
