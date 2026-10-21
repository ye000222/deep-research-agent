from __future__ import annotations

import asyncio
from typing import Any

import pytest
from app.domain.providers import AdapterType, CanonicalModelResult, TokenUsage, UsageAccuracy
from app.domain.source_context_extraction_v3_2 import (
    SOURCE_CONTEXT_EXTRACTION_SCHEMA_V32,
    GenericContextSemanticValueV32,
    MeasurementSemanticsV32,
    SourceContextV32,
)
from app.domain.truth_condition_context import ContextType
from app.services.source_context_extractor_v3_2 import SourceContextExtractorV32
from pydantic import SecretStr, ValidationError


def integer_value(value: int = 7) -> dict[str, Any]:
    return {"kind": "INTEGER", "raw_value": str(value), "value": value}


@pytest.mark.parametrize(
    "semantic_value",
    [
        {"measurement_identity": "latency"},
        {"observed_value": integer_value()},
        {"measurement_identity": "latency", "observed_value": integer_value()},
        {"observed_value": integer_value(), "unit": "ms"},
        {"formal_definition": "Elapsed time between request and response."},
        {
            "threshold": {
                "comparator": "GTE",
                "value": integer_value(95),
                "raw_text": "at least 95",
            }
        },
        {"measurement_identity": "latency", "scope_category": "edge devices"},
        {"observed_value": integer_value(), "basis_ref": "B1"},
    ],
)
def test_intrinsic_measurement_anchors_are_admissible(semantic_value: dict[str, Any]) -> None:
    parsed = SourceContextV32.model_validate(
        {"context_ref": "M1", "context_type": "MEASUREMENT", "semantic_value": semantic_value}
    )
    assert isinstance(parsed.semantic_value, MeasurementSemanticsV32)


@pytest.mark.parametrize(
    "semantic_value",
    [
        {"scope_category": "edge devices"},
        {"unit": "ms"},
        {"basis_ref": "B1"},
        {"comparator": "GTE"},
        {"target_object": "device"},
        {"qualifiers": ["average"]},
        {"scope_category": "edge", "unit": "ms"},
        {},
    ],
)
def test_modifier_only_measurement_is_rejected(semantic_value: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="measurement_requires_intrinsic_semantic_anchor"):
        SourceContextV32.model_validate(
            {"context_ref": "M1", "context_type": "MEASUREMENT", "semantic_value": semantic_value}
        )


def test_scope_scope_category_payload_remains_generic_and_round_trips() -> None:
    scope = SourceContextV32.model_validate(
        {
            "context_ref": "S1",
            "context_type": "SCOPE",
            "semantic_value": {"scope_category": "MVTec AD"},
        }
    )
    assert isinstance(scope.semantic_value, GenericContextSemanticValueV32)
    assert scope.semantic_value.root == {"scope_category": "MVTec AD"}
    assert SourceContextV32.model_validate(scope.model_dump(mode="json")) == scope


def test_measurement_round_trip_preserves_fields_without_insertion() -> None:
    original = SourceContextV32.model_validate(
        {
            "context_ref": "M1",
            "context_type": "MEASUREMENT",
            "semantic_value": {"observed_value": integer_value(), "unit": "ms"},
        }
    )
    serialized = original.model_dump(mode="json", exclude_none=True)
    assert set(serialized["semantic_value"]) == {"observed_value", "unit", "qualifiers"}
    assert SourceContextV32.model_validate(serialized) == original


def test_wrong_discriminator_and_generic_dict_cannot_bypass_measurement_contract() -> None:
    with pytest.raises(ValidationError):
        SourceContextV32.model_validate(
            {"context_ref": "M1", "context_type": "FUTURE_CONTEXT", "semantic_value": {}}
        )
    generic = GenericContextSemanticValueV32({"scope_category": "edge"})
    with pytest.raises(ValidationError):
        SourceContextV32.model_validate(
            {
                "context_ref": "M1",
                "context_type": ContextType.MEASUREMENT,
                "semantic_value": generic,
            }
        )


def valid_response(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    attachments: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = [
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
            sem = context["semantic_value"]
            roles = {
                "measurement_identity": "MEASUREMENT_IDENTITY",
                "formal_definition": "MEASUREMENT_DEFINITION",
                "observed_value": "MEASUREMENT_VALUE",
                "unit": "MEASUREMENT_UNIT",
                "comparator": "MEASUREMENT_COMPARATOR",
                "threshold": "MEASUREMENT_COMPARATOR",
                "basis_ref": "MEASUREMENT_BASIS",
                "scope_category": "SCOPE_VALUE",
                "target_object": "MEASUREMENT_TARGET",
                "qualifiers": "MEASUREMENT_QUALIFIER",
            }
            for field, role in roles.items():
                value = sem.get(field)
                if value is not None and value != []:
                    bindings.append(
                        {
                            "binding_ref": f"b{ref}{role}{len(bindings)}",
                            "evidence_ref": "eA",
                            "semantic_role": role,
                            "target_kind": "CONTEXT",
                            "target_ref": ref,
                        }
                    )

    return {
        "schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V32,
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
        self.request = None

    async def generate_structured(self, **kwargs: Any) -> CanonicalModelResult:
        self.calls += 1
        self.request = kwargs["request"]
        assert kwargs["allow_regeneration"] is False
        return CanonicalModelResult(
            parsed_object=self.payload,
            usage=TokenUsage(
                input_tokens=2,
                output_tokens=3,
                total_tokens=5,
                accuracy=UsageAccuracy.EXACT,
            ),
        )


def service(payload: dict[str, Any]) -> tuple[SourceContextExtractorV32, FakeGateway]:
    gateway = FakeGateway(payload)
    return (
        SourceContextExtractorV32(
            gateway,
            adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
            base_url="https://fake.invalid/v1",
            api_key=SecretStr("not-real"),
            model="fake",
        ),
        gateway,
    )


def test_same_path_fake_accepts_measurement_scope_and_multiple_contexts() -> None:
    payload = valid_response(
        [
            {
                "context_ref": "S",
                "context_type": "SCOPE",
                "semantic_value": {"scope_category": "plants"},
            },
            {
                "context_ref": "M1",
                "context_type": "MEASUREMENT",
                "semantic_value": {
                    "measurement_identity": "latency",
                    "observed_value": integer_value(),
                },
            },
            {
                "context_ref": "M2",
                "context_type": "MEASUREMENT",
                "semantic_value": {
                    "threshold": {
                        "comparator": "GTE",
                        "value": integer_value(95),
                        "raw_text": "at least 95",
                    },
                    "comparator": "GTE",
                },
            },
        ]
    )
    extractor, gateway = service(payload)
    result = asyncio.run(extractor.extract("Anchor P."))
    assert gateway.calls == 1
    assert result.model_call_count == 1
    assert result.schema_version == SOURCE_CONTEXT_EXTRACTION_SCHEMA_V32
    assert gateway.request.task_kind == "source_context_shadow_extraction_v3_2"
    assert gateway.request.metadata["execution_mode"] == "SHADOW"
    assert result.status.value == "RESOLVED"


def test_same_path_fake_rejects_modifier_only_measurement_after_one_call() -> None:
    payload = valid_response(
        [
            {
                "context_ref": "M",
                "context_type": "MEASUREMENT",
                "semantic_value": {"scope_category": "plants"},
            }
        ]
    )
    extractor, gateway = service(payload)
    result = asyncio.run(extractor.extract("Anchor P."))
    assert gateway.calls == 1
    assert result.status.value == "UNRESOLVED"
    assert "schema_validation_failed" in result.diagnostics
