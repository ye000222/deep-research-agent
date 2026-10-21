from __future__ import annotations

import asyncio
import re
from typing import Any

from app.domain.atomic_proposition import PropositionValue
from app.domain.providers import AdapterType, CanonicalModelResult, TokenUsage, UsageAccuracy
from app.services.atomic_proposition_decomposer import AtomicPropositionDecomposer
from app.services.atomic_proposition_validation import (
    ATOMIC_DECOMPOSITION_PROMPT_VERSION,
    ShadowDecompositionResponse,
    validate_decomposition_response,
)
from pydantic import SecretStr, ValidationError


def _schema_accepts(value: Any, schema: dict[str, Any], root: dict[str, Any]) -> bool:
    if "$ref" in schema:
        resolved: Any = root
        for segment in schema["$ref"].removeprefix("#/").split("/"):
            resolved = resolved[segment]
        return _schema_accepts(value, resolved, root)
    for keyword in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(keyword)
        if branches is None:
            continue
        results = [_schema_accepts(value, branch, root) for branch in branches]
        if keyword == "allOf" and not all(results):
            return False
        if keyword == "anyOf" and not any(results):
            return False
        if keyword == "oneOf" and sum(results) != 1:
            return False
    if "if" in schema:
        branch = "then" if _schema_accepts(value, schema["if"], root) else "else"
        if branch in schema and not _schema_accepts(value, schema[branch], root):
            return False
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    expected_type = schema.get("type")
    if expected_type is not None:
        expected_types = expected_type if isinstance(expected_type, list) else [expected_type]
        matched = any(
            (kind == "object" and isinstance(value, dict))
            or (kind == "array" and isinstance(value, list | tuple))
            or (kind == "string" and isinstance(value, str))
            or (kind == "boolean" and isinstance(value, bool))
            or (kind == "integer" and isinstance(value, int) and not isinstance(value, bool))
            or (kind == "number" and isinstance(value, int | float) and not isinstance(value, bool))
            or (kind == "null" and value is None)
            for kind in expected_types
        )
        if not matched:
            return False
    if isinstance(value, dict):
        if any(key not in value for key in schema.get("required", [])):
            return False
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and any(
            key not in properties for key in value
        ):
            return False
        if any(
            key in value and not _schema_accepts(value[key], child, root)
            for key, child in properties.items()
        ):
            return False
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            return False
        if len(value) > schema.get("maxLength", float("inf")):
            return False
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            return False
    if isinstance(value, list | tuple):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get(
            "maxItems", float("inf")
        ):
            return False
        if "items" in schema and any(
            not _schema_accepts(item, schema["items"], root) for item in value
        ):
            return False
    if isinstance(value, int | float) and not isinstance(value, bool):
        if value < schema.get("minimum", float("-inf")):
            return False
        if value > schema.get("maximum", float("inf")):
            return False
    return True


def _response(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "atomic-proposition-schema-v2",
        "decomposition_status": "single_proposition",
        "propositions": [
            {
                "proposition_text": "Synthetic typed-value assertion.",
                "subject": "Synthetic subject",
                "predicate": "has value",
                "typed_value": value,
                "polarity": "asserted",
            }
        ],
    }


def _matrix_case(value: dict[str, Any], expected: bool) -> None:
    schema = ShadowDecompositionResponse.model_json_schema()
    typed_schema = schema["$defs"]["PropositionPayload"]["properties"]["typed_value"]
    schema_result = _schema_accepts(value, typed_schema, schema)
    try:
        parsed = ShadowDecompositionResponse.model_validate(_response(value))
        response_model_accepts = True
    except ValidationError:
        parsed = None
        response_model_accepts = False
    runtime_accepts = validate_decomposition_response(
        _response(value), claim_text="Synthetic typed-value assertion."
    ).valid
    try:
        PropositionValue.model_validate(value)
        domain_accepts = True
    except ValidationError:
        domain_accepts = False

    assert schema_result is expected, value
    assert response_model_accepts is expected, value
    assert runtime_accepts is expected, value
    assert domain_accepts is expected, value
    if expected:
        assert parsed is not None


def test_all_supported_value_kinds_accept_valid_generic_shapes() -> None:
    valid_values = (
        {"kind": "text", "text": "a statement"},
        {"kind": "entity", "text": "an entity"},
        {"kind": "quantity", "numeric": {"operator": "exact", "value": 3}},
        {"kind": "range", "numeric": {"operator": "range", "lower_bound": 1, "upper_bound": 3}},
        {"kind": "boolean", "boolean": False},
        {"kind": "set", "members": ["alpha", "beta"]},
        {
            "kind": "comparison",
            "comparison_target": "system B",
            "comparison_metric": "latency",
            "comparison_metric_status": "explicit",
            "numeric": {"operator": "approx", "value": 4, "unit": "ms"},
        },
        {
            "kind": "comparison",
            "comparison_target": "system C",
            "comparison_metric": None,
            "comparison_metric_status": "unspecified",
        },
        {
            "kind": "temporal_change",
            "temporal_from": "period A",
            "temporal_to": "period B",
            "numeric": {"operator": "gt", "value": 0},
        },
        {"kind": "unknown"},
    )
    for value in valid_values:
        _matrix_case(value, True)


def test_all_supported_value_kinds_reject_invalid_generic_shapes() -> None:
    invalid_values = (
        {"kind": "text", "text": ""},
        {"kind": "entity", "numeric": {"operator": "exact", "value": 1}},
        {"kind": "quantity"},
        {"kind": "quantity", "numeric": {"operator": "exact"}},
        {"kind": "range", "numeric": {"operator": "range", "lower_bound": 3}},
        {"kind": "boolean"},
        {"kind": "boolean", "boolean": 1},
        {"kind": "set", "members": []},
        {"kind": "set", "members": ["  "]},
        {"kind": "comparison", "comparison_target": "system B", "comparison_metric": "latency"},
        {
            "kind": "comparison",
            "comparison_target": "system B",
            "comparison_metric": None,
            "comparison_metric_status": "explicit",
        },
        {
            "kind": "comparison",
            "comparison_target": "system B",
            "comparison_metric_status": "unspecified",
            "comparison_metric": "latency",
        },
        {
            "kind": "comparison",
            "comparison_target": "system B",
            "comparison_metric_status": "explicit",
            "numeric": {"operator": "exact", "value": 1},
        },
        {
            "kind": "comparison",
            "comparison_metric": "latency",
            "comparison_metric_status": "explicit",
            "numeric": {"operator": "exact", "value": 1},
        },
        {
            "kind": "temporal_change",
            "temporal_from": "period A",
            "numeric": {"operator": "exact", "value": 1},
        },
        {"kind": "unknown", "text": "incompatible"},
        {"kind": "not_a_supported_kind"},
    )
    for value in invalid_values:
        _matrix_case(value, False)


def test_model_visible_schema_allows_qualitative_and_quantitative_comparison() -> None:
    schema = ShadowDecompositionResponse.model_json_schema()
    typed_schema = schema["$defs"]["PropositionPayload"]["properties"]["typed_value"]
    mapping = typed_schema["discriminator"]["mapping"]
    comparison = schema["$defs"][mapping["comparison"].rsplit("/", 1)[-1]]

    assert typed_schema["discriminator"]["propertyName"] == "kind"
    assert "oneOf" in typed_schema
    assert comparison["required"] == ["kind", "comparison_target", "comparison_metric_status"]
    assert "numeric" not in comparison["required"]
    assert "temporal_from" not in comparison["properties"]
    _matrix_case(
        {
            "kind": "comparison",
            "comparison_target": "system B",
            "comparison_metric": None,
            "comparison_metric_status": "unspecified",
        },
        True,
    )


def test_exact_gateway_request_uses_the_kind_specific_generated_contract() -> None:
    class ContractCaptureGateway:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def generate_structured(self, **kwargs: Any) -> CanonicalModelResult:
            self.calls.append(kwargs)
            return CanonicalModelResult(
                text=None,
                parsed_object={"invalid": True},
                usage=TokenUsage(accuracy=UsageAccuracy.UNAVAILABLE),
            )

    fake_gateway = ContractCaptureGateway()
    decomposer = AtomicPropositionDecomposer(
        fake_gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://not-contacted.invalid/v1",
        api_key=SecretStr("synthetic-only"),
        model="synthetic-model",
        max_calls=1,
    )
    asyncio.run(decomposer.decompose_claim("Synthetic assertion."))
    request = fake_gateway.calls[0]["request"]
    assert request.metadata["prompt_version"] == ATOMIC_DECOMPOSITION_PROMPT_VERSION
    assert request.response_contract == ShadowDecompositionResponse.model_json_schema()
    typed_value = request.response_contract["$defs"]["PropositionPayload"]["properties"][
        "typed_value"
    ]
    comparison_ref = typed_value["discriminator"]["mapping"]["comparison"].rsplit("/", 1)[-1]
    comparison_schema = request.response_contract["$defs"][comparison_ref]
    assert "comparison_metric_status" in comparison_schema["required"]
    assert "numeric" not in comparison_schema["required"]
    assert len(fake_gateway.calls) == 1


def test_numeric_operator_schema_is_discriminated_and_documents_range_ordering_limit() -> None:
    definitions = ShadowDecompositionResponse.model_json_schema()["$defs"]
    numeric = definitions["_QuantityValuePayload"]["properties"]["numeric"]
    scalar = definitions["_ScalarNumericPayload"]
    ranged = definitions["_RangeNumericPayload"]
    assert numeric["discriminator"]["propertyName"] == "operator"
    assert "value" in scalar["required"]
    assert "lower_bound" in ranged["required"] and "upper_bound" in ranged["required"]
    assert "ordered" in ranged["description"].lower()


def test_reversed_bounds_are_schema_documented_and_runtime_fails_closed() -> None:
    value = {
        "kind": "range",
        "numeric": {"operator": "range", "lower_bound": 5, "upper_bound": 2},
    }
    schema = ShadowDecompositionResponse.model_json_schema()
    typed_schema = schema["$defs"]["PropositionPayload"]["properties"]["typed_value"]
    assert _schema_accepts(value, typed_schema, schema)
    try:
        ShadowDecompositionResponse.model_validate(_response(value))
    except ValidationError:
        response_model_accepts = False
    else:
        response_model_accepts = True
    assert not response_model_accepts
    assert not validate_decomposition_response(
        _response(value), claim_text="Synthetic range."
    ).valid
    try:
        PropositionValue.model_validate(value)
    except ValidationError:
        domain_accepts = False
    else:
        domain_accepts = True
    assert not domain_accepts


def test_domain_model_rejects_incompatible_fields_and_non_json_numeric_coercions() -> None:
    invalid_values = (
        {
            "kind": "comparison",
            "comparison_target": "B",
            "comparison_metric": "m",
            "numeric": {"operator": "exact", "value": 1},
            "temporal_from": "period A",
        },
        {"kind": "quantity", "numeric": {"operator": "exact", "value": "3"}},
    )
    for value in invalid_values:
        try:
            PropositionValue.model_validate(value)
        except ValidationError:
            continue
        raise AssertionError(f"domain accepted incompatible typed value: {value}")
