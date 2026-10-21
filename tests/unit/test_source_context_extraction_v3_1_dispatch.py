from __future__ import annotations

import asyncio
from typing import Any

import pytest
from app.domain.providers import AdapterType, CanonicalModelResult, TokenUsage, UsageAccuracy
from app.domain.source_context_extraction_v3_1 import (
    SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31,
    GenericContextSemanticValueV31,
    MeasurementSemanticsV31,
    SourceContextV31,
    SourceExtractionResponseV31,
    V31ValidationStatus,
)
from app.domain.truth_condition_context import ContextType
from app.services.source_context_extractor_v3_1 import SourceContextExtractorV31
from pydantic import SecretStr, ValidationError


def response(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    attachments = []
    bindings = [
        {
            "binding_ref": "bA",
            "evidence_ref": "eA",
            "semantic_role": "ASSERTION_EVIDENCE",
            "target_kind": "ASSERTION",
            "target_ref": "A",
        }
    ]
    for index, context in enumerate(contexts):
        ref = context["context_ref"]
        attachment_ref = f"AT{index}"
        attachments.append(
            {
                "attachment_ref": attachment_ref,
                "context_ref": ref,
                "source_assertion_refs": ["A"],
                "relation": "APPLIES_TO",
                "status": "RESOLVED",
                "evidence_refs": ["eA"],
            }
        )
        bindings.append(
            {
                "binding_ref": f"b{attachment_ref}",
                "evidence_ref": "eA",
                "semantic_role": "ATTACHMENT",
                "target_kind": "ATTACHMENT",
                "target_ref": attachment_ref,
            }
        )
        if context["context_type"] == "MEASUREMENT":
            for field, role in (
                ("measurement_identity", "MEASUREMENT_IDENTITY"),
                ("formal_definition", "MEASUREMENT_DEFINITION"),
                ("observed_value", "MEASUREMENT_VALUE"),
                ("unit", "MEASUREMENT_UNIT"),
                ("comparator", "MEASUREMENT_COMPARATOR"),
                ("basis_ref", "MEASUREMENT_BASIS"),
                ("target_object", "MEASUREMENT_TARGET"),
            ):
                if context["semantic_value"].get(field) is not None:
                    bindings.append(
                        {
                            "binding_ref": f"b{ref}{role}",
                            "evidence_ref": "eA",
                            "semantic_role": role,
                            "target_kind": "CONTEXT",
                            "target_ref": ref,
                        }
                    )
            if context["semantic_value"].get("qualifiers"):
                bindings.append(
                    {
                        "binding_ref": f"b{ref}Q",
                        "evidence_ref": "eA",
                        "semantic_role": "MEASUREMENT_QUALIFIER",
                        "target_kind": "CONTEXT",
                        "target_ref": ref,
                    }
                )

    return {
        "schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31,
        "status": "RESOLVED",
        "source_assertions": [
            {"source_assertion_ref": "A", "evidence_refs": ["eA"], "assertion_status": "ASSERTED"}
        ],
        "source_contexts": contexts,
        "source_attachments": attachments,
        "source_scopes": [],
        "evidence_spans": [{"evidence_ref": "eA", "exact_text": "Anchor P."}],
        "evidence_bindings": bindings,
        "semantic_relations": [],
        "measurement_bases": [],
        "unresolved_items": [],
    }


class FakeGateway:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.calls = 0

    async def generate_structured(self, **_: Any) -> CanonicalModelResult:
        self.calls += 1
        return CanonicalModelResult(
            parsed_object=self.payload,
            usage=TokenUsage(
                input_tokens=2, output_tokens=3, total_tokens=5, accuracy=UsageAccuracy.EXACT
            ),
        )


def extractor(payload: dict[str, Any]) -> tuple[SourceContextExtractorV31, FakeGateway]:
    gateway = FakeGateway(payload)
    return (
        SourceContextExtractorV31(
            gateway,
            adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
            base_url="https://fake.invalid/v1",
            api_key=SecretStr("not-real"),
            model="fake",
            max_calls=1,
        ),
        gateway,
    )


@pytest.mark.parametrize("context_type", [item.value for item in ContextType])
def test_every_context_type_has_one_deterministic_payload_branch(context_type: str) -> None:
    value: dict[str, Any]
    if context_type == ContextType.MEASUREMENT.value:
        value = {"measurement_identity": "latency"}
    elif context_type == ContextType.SCOPE.value:
        # Historical D4-J collision: this is a generic SCOPE value, not a measurement.
        value = {"scope_category": "edge deployment"}
    else:
        value = {"payload": f"{context_type.lower()}-value"}

    parsed = SourceContextV31.model_validate(
        {"context_ref": "C", "context_type": context_type, "semantic_value": value}
    )
    if context_type == ContextType.MEASUREMENT.value:
        assert isinstance(parsed.semantic_value, MeasurementSemanticsV31)
    else:
        assert isinstance(parsed.semantic_value, GenericContextSemanticValueV31)

    serialized = parsed.model_dump(mode="json", exclude_none=True)
    if context_type == ContextType.MEASUREMENT.value:
        assert serialized["semantic_value"]["measurement_identity"] == "latency"
        assert serialized["semantic_value"]["qualifiers"] == []
    else:
        assert serialized["semantic_value"] == value
    round_tripped = SourceContextV31.model_validate(serialized)
    assert round_tripped == parsed
    assert type(round_tripped.semantic_value) is type(parsed.semantic_value)


def test_unknown_and_missing_context_discriminators_fail_closed() -> None:
    with pytest.raises(ValidationError):
        SourceContextV31.model_validate(
            {"context_ref": "C", "context_type": "FUTURE_CONTEXT", "semantic_value": {}}
        )
    with pytest.raises(ValidationError):
        SourceContextV31.model_validate({"context_ref": "C", "semantic_value": {}})


@pytest.mark.parametrize(
    "semantic_value",
    [
        {},
        {
            "measurement_identity": "latency",
            "observed_value": {"kind": "INTEGER", "raw_value": "2", "value": True},
        },
        {"measurement_identity": "latency", "unexpected": "must not escape typed validation"},
    ],
)
def test_measurement_context_never_escapes_typed_validation(semantic_value: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SourceContextV31.model_validate(
            {"context_ref": "M", "context_type": "MEASUREMENT", "semantic_value": semantic_value}
        )


def test_scope_discriminator_controls_generic_dispatch_without_scope_owner_changes() -> None:
    parsed = SourceContextV31.model_validate(
        {
            "context_ref": "S",
            "context_type": "SCOPE",
            "semantic_value": {"scope_category": "MVTec AD dataset"},
        }
    )
    assert isinstance(parsed.semantic_value, GenericContextSemanticValueV31)
    assert parsed.semantic_value.root == {"scope_category": "MVTec AD dataset"}
    assert SourceContextV31.model_validate(parsed.model_dump(mode="json")) == parsed


def test_wrong_typed_domain_object_is_not_auto_corrected() -> None:
    with pytest.raises(ValidationError):
        SourceContextV31.model_validate(
            {
                "context_ref": "S",
                "context_type": ContextType.SCOPE,
                "semantic_value": MeasurementSemanticsV31(measurement_identity="latency"),
            }
        )


def test_fake_gateway_same_path_scope_only_historical_failure_is_closed() -> None:
    payload = response(
        [
            {
                "context_ref": "S1",
                "context_type": "SCOPE",
                "semantic_value": {"scope_category": "industrial edge deployment"},
            },
            {
                "context_ref": "S2",
                "context_type": "SCOPE",
                "semantic_value": {"scope_category": "MVTec AD"},
            },
        ]
    )
    service, gateway = extractor(payload)
    result = asyncio.run(service.extract("Anchor P."))
    assert result.status is V31ValidationStatus.RESOLVED
    assert gateway.calls == 1
    assert result.model_call_count == 1


def test_fake_gateway_same_path_mixed_contexts_dispatch_independently() -> None:
    payload = response(
        [
            {
                "context_ref": "S",
                "context_type": "SCOPE",
                "semantic_value": {"scope_category": "industrial plants"},
            },
            {
                "context_ref": "M",
                "context_type": "MEASUREMENT",
                "semantic_value": {"measurement_identity": "latency", "unit": "ms"},
            },
            {
                "context_ref": "Q",
                "context_type": "CONDITION",
                "semantic_value": {"condition": "during inference"},
            },
        ]
    )
    service, gateway = extractor(payload)
    result = asyncio.run(service.extract("Anchor P."))
    assert result.status is V31ValidationStatus.RESOLVED
    assert gateway.calls == 1

    parsed = SourceExtractionResponseV31.model_validate(payload)
    assert isinstance(parsed.source_contexts[0].semantic_value, GenericContextSemanticValueV31)
    assert isinstance(parsed.source_contexts[1].semantic_value, MeasurementSemanticsV31)
    assert isinstance(parsed.source_contexts[2].semantic_value, GenericContextSemanticValueV31)
    assert parsed.source_contexts[0].semantic_value.root == {"scope_category": "industrial plants"}
    assert parsed.source_contexts[2].semantic_value.root == {"condition": "during inference"}


def test_fake_gateway_same_path_supports_multiple_scopes_and_measurements() -> None:
    payload = response(
        [
            {
                "context_ref": "S1",
                "context_type": "SCOPE",
                "semantic_value": {"scope_category": "plants"},
            },
            {
                "context_ref": "S2",
                "context_type": "SCOPE",
                "semantic_value": {"scope_category": "lines"},
            },
            {
                "context_ref": "M1",
                "context_type": "MEASUREMENT",
                "semantic_value": {"measurement_identity": "latency"},
            },
            {
                "context_ref": "M2",
                "context_type": "MEASUREMENT",
                "semantic_value": {"measurement_identity": "throughput"},
            },
        ]
    )
    service, gateway = extractor(payload)
    result = asyncio.run(service.extract("Anchor P."))
    assert result.status is V31ValidationStatus.RESOLVED
    assert gateway.calls == 1
    parsed = SourceExtractionResponseV31.model_validate(payload)
    assert [type(item.semantic_value) for item in parsed.source_contexts] == [
        GenericContextSemanticValueV31,
        GenericContextSemanticValueV31,
        MeasurementSemanticsV31,
        MeasurementSemanticsV31,
    ]


def test_fake_gateway_invalid_measurement_is_rejected_after_one_call() -> None:
    payload = response(
        [
            {
                "context_ref": "M",
                "context_type": "MEASUREMENT",
                "semantic_value": {"scope": "not a measurement payload"},
            }
        ]
    )
    service, gateway = extractor(payload)
    result = asyncio.run(service.extract("Anchor P."))
    assert result.status is V31ValidationStatus.UNRESOLVED
    assert "schema_validation_failed" in result.diagnostics
    assert gateway.calls == 1
