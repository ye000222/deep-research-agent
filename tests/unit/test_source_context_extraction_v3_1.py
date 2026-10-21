from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from app.domain.providers import AdapterType, CanonicalModelResult, TokenUsage, UsageAccuracy
from app.domain.source_context_extraction_v3_1 import (
    SOURCE_CONTEXT_EXTRACTION_PROMPT_V31,
    SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31,
    SourceExtractionResponseV31,
    V31ValidationStatus,
    validate_scope_ownership_v31,
    validate_source_context_v31,
)
from app.services.source_context_extractor_v3_1 import (
    SOURCE_CONTEXT_V31_INSTRUCTIONS,
    SourceContextExtractorV31,
)
from pydantic import SecretStr, ValidationError


def base(claim: str = "Anchor P.") -> dict[str, Any]:
    return {
        "schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31,
        "status": "RESOLVED",
        "source_assertions": [
            {"source_assertion_ref": "A", "evidence_refs": ["eA"], "assertion_status": "ASSERTED"}
        ],
        "source_contexts": [],
        "source_attachments": [],
        "source_scopes": [],
        "evidence_spans": [{"evidence_ref": "eA", "exact_text": "Anchor P."}],
        "evidence_bindings": [
            {
                "binding_ref": "bA",
                "evidence_ref": "eA",
                "semantic_role": "ASSERTION_EVIDENCE",
                "target_kind": "ASSERTION",
                "target_ref": "A",
            }
        ],
        "semantic_relations": [],
        "measurement_bases": [],
        "unresolved_items": [],
    }


def add_scope(
    data: dict[str, Any],
    *,
    members: list[str] | None = None,
    evidence: list[str] | None = None,
    status: str = "RESOLVED",
) -> None:
    data["source_scopes"].append(
        {
            "source_scope_ref": "S",
            "kind": "CLAIM",
            "source_assertion_refs": members or ["A"],
            "evidence_refs": evidence or ["eA"],
            "resolution_status": status,
        }
    )


def add_measurement(
    data: dict[str, Any],
    *,
    semantics: dict[str, Any] | None = None,
    roles: list[str] | None = None,
    target_ref: str = "M",
) -> dict[str, Any]:
    data["source_contexts"].append(
        {
            "context_ref": "M",
            "context_type": "MEASUREMENT",
            "semantic_value": semantics or {"measurement_identity": "latency"},
            "lexical_value": "latency",
            "materiality": "MATERIAL",
        }
    )
    data["source_attachments"].append(
        {
            "attachment_ref": "AT-M",
            "context_ref": "M",
            "source_assertion_refs": ["A"],
            "relation": "APPLIES_TO",
            "status": "RESOLVED",
            "evidence_refs": ["eA"],
        }
    )
    role_list = roles or ["MEASUREMENT_IDENTITY"]
    for index, role in enumerate(role_list):
        data["evidence_bindings"].append(
            {
                "binding_ref": f"bM{index}",
                "evidence_ref": "eA",
                "semantic_role": role,
                "target_kind": "CONTEXT",
                "target_ref": target_ref,
            }
        )
    data["evidence_bindings"].append(
        {
            "binding_ref": "bAT-M",
            "evidence_ref": "eA",
            "semantic_role": "ATTACHMENT",
            "target_kind": "ATTACHMENT",
            "target_ref": "AT-M",
        }
    )
    return data


def checked(data: dict[str, Any], claim: str = "Anchor P."):
    return validate_source_context_v31(SourceExtractionResponseV31.model_validate(data), claim)


def test_scope_single_and_multi_member_are_canonical_without_duplicate_binding() -> None:
    single = base()
    add_scope(single)
    assert checked(single).status is V31ValidationStatus.RESOLVED
    assert not any(x["semantic_role"] == "SCOPE_STRUCTURE" for x in single["evidence_bindings"])

    multi = base()
    multi["source_assertions"].append(
        {"source_assertion_ref": "B", "evidence_refs": ["eA"], "assertion_status": "ASSERTED"}
    )
    multi["evidence_bindings"].append(
        {
            "binding_ref": "bB",
            "evidence_ref": "eA",
            "semantic_role": "ASSERTION_EVIDENCE",
            "target_kind": "ASSERTION",
            "target_ref": "B",
        }
    )
    add_scope(multi, members=["A", "B"])
    assert checked(multi).status is V31ValidationStatus.RESOLVED


@pytest.mark.parametrize(
    ("members", "evidence", "expected"),
    [
        (["missing"], ["eA"], V31ValidationStatus.INVALID),
        (["A"], ["missing"], V31ValidationStatus.INVALID),
    ],
)
def test_invalid_scope_refs_fail_closed(members, evidence, expected) -> None:
    data = base()
    add_scope(data, members=members, evidence=evidence)
    assert checked(data).status is expected


def test_empty_scope_evidence_is_schema_invalid() -> None:
    data = base()
    add_scope(data, evidence=[])
    data["source_scopes"][0]["evidence_refs"] = []
    with pytest.raises(ValidationError):
        SourceExtractionResponseV31.model_validate(data)


@pytest.mark.parametrize("scope_status", ["PARTIAL", "UNKNOWN", "UNSUPPORTED"])
def test_unresolved_scope_remains_partial_not_semantically_promoted(scope_status: str) -> None:
    data = base()
    add_scope(data, status=scope_status)
    result = checked(data)
    assert result.status is V31ValidationStatus.PARTIAL
    assert result.semantic_support == "NOT_EVALUATED"


def test_duplicate_scope_members_are_invalid() -> None:
    data = base()
    data["source_assertions"].append(
        {"source_assertion_ref": "B", "evidence_refs": ["eA"], "assertion_status": "ASSERTED"}
    )
    add_scope(data, members=["A", "A"])
    assert checked(data).status is V31ValidationStatus.INVALID


@pytest.mark.parametrize(
    ("semantics", "roles"),
    [
        (
            {
                "measurement_identity": "latency",
                "observed_value": {"kind": "INTEGER", "raw_value": "12", "value": 12},
            },
            ["MEASUREMENT_IDENTITY", "MEASUREMENT_VALUE"],
        ),
        (
            {
                "measurement_identity": "latency",
                "observed_value": {"kind": "DECIMAL", "raw_value": "1.25", "value": "1.25"},
                "unit": "ms",
            },
            ["MEASUREMENT_IDENTITY", "MEASUREMENT_VALUE", "MEASUREMENT_UNIT"],
        ),
        (
            {
                "measurement_identity": "latency",
                "observed_value": {
                    "kind": "RANGE",
                    "raw_value": "1-3",
                    "lower_bound": "1",
                    "upper_bound": "3",
                },
            },
            ["MEASUREMENT_IDENTITY", "MEASUREMENT_VALUE"],
        ),
        ({"measurement_identity": "latency"}, ["MEASUREMENT_IDENTITY"]),
        (
            {"measurement_identity": "latency", "formal_definition": "explicit definition"},
            ["MEASUREMENT_IDENTITY", "MEASUREMENT_DEFINITION"],
        ),
        (
            {
                "measurement_identity": "latency",
                "observed_value": {"kind": "APPROXIMATE", "raw_value": "about 3", "value": "3"},
            },
            ["MEASUREMENT_IDENTITY", "MEASUREMENT_VALUE"],
        ),
        (
            {
                "measurement_identity": "class",
                "observed_value": {"kind": "TEXT", "raw_value": "high", "value": "high"},
            },
            ["MEASUREMENT_IDENTITY", "MEASUREMENT_VALUE"],
        ),
        (
            {
                "measurement_identity": "latency",
                "comparator": "GTE",
                "threshold": {
                    "comparator": "GTE",
                    "value": {"kind": "INTEGER", "raw_value": "10", "value": 10},
                    "raw_text": "at least 10",
                },
            },
            ["MEASUREMENT_IDENTITY", "MEASUREMENT_COMPARATOR"],
        ),
        (
            {
                "measurement_identity": "latency",
                "observed_value": {
                    "kind": "BOUND",
                    "raw_value": "<5",
                    "comparator": "LT",
                    "value": "5",
                },
            },
            ["MEASUREMENT_IDENTITY", "MEASUREMENT_VALUE"],
        ),
        (
            {
                "measurement_identity": "latency",
                "observed_value": {"kind": "INTEGER", "raw_value": "10", "value": 10},
                "scope_category": "group",
                "target_object": "service",
                "qualifiers": ["warm run"],
            },
            [
                "MEASUREMENT_IDENTITY",
                "MEASUREMENT_VALUE",
                "SCOPE_VALUE",
                "MEASUREMENT_TARGET",
                "MEASUREMENT_QUALIFIER",
            ],
        ),
    ],
)
def test_typed_measurement_facts_and_roles(semantics: dict[str, Any], roles: list[str]) -> None:
    data = base()
    add_measurement(data, semantics=semantics, roles=roles)
    assert checked(data).status is V31ValidationStatus.RESOLVED


def test_measurement_result_does_not_imply_definition_unit_or_basis() -> None:
    data = base()
    semantics = {
        "measurement_identity": "score",
        "observed_value": {"kind": "DECIMAL", "raw_value": "0.9", "value": "0.9"},
    }
    add_measurement(data, semantics=semantics, roles=["MEASUREMENT_IDENTITY", "MEASUREMENT_VALUE"])
    result = checked(data)
    parsed = result.source_contexts[0].semantic_value
    assert parsed.formal_definition is None
    assert parsed.unit is None
    assert parsed.basis_ref is None


def test_definition_role_is_not_repaired_when_only_observed_value_exists() -> None:
    data = base()
    add_measurement(
        data,
        semantics={
            "measurement_identity": "score",
            "observed_value": {"kind": "DECIMAL", "raw_value": "0.9", "value": "0.9"},
        },
        roles=["MEASUREMENT_IDENTITY", "MEASUREMENT_DEFINITION"],
    )
    result = checked(data)
    assert result.status is V31ValidationStatus.INVALID
    assert any(
        item == "measurement_binding_missing:measurement_value" for item in result.diagnostics
    )
    assert any(
        item == "measurement_role_without_matching_field:measurement_definition"
        for item in result.diagnostics
    )


def test_measurement_binding_to_wrong_context_fails_closed() -> None:
    data = base()
    add_measurement(
        data, semantics={"measurement_identity": "score"}, roles=["MEASUREMENT_IDENTITY"]
    )
    data["evidence_bindings"][1]["target_kind"] = "ASSERTION"
    data["evidence_bindings"][1]["target_ref"] = "A"
    result = checked(data)
    assert result.status is V31ValidationStatus.INVALID
    assert "binding_role_target_mismatch" in result.diagnostics


def test_measurement_attachment_and_shared_condition_are_canonical() -> None:
    data = base()
    data["source_contexts"].append(
        {
            "context_ref": "COND",
            "context_type": "CONDITION",
            "semantic_value": {"value": "same condition"},
        }
    )
    data["source_attachments"].append(
        {
            "attachment_ref": "AT-COND",
            "context_ref": "COND",
            "source_assertion_refs": ["A"],
            "relation": "CONDITIONS",
            "status": "RESOLVED",
            "evidence_refs": ["eA"],
        }
    )
    data["evidence_bindings"].append(
        {
            "binding_ref": "bCOND",
            "evidence_ref": "eA",
            "semantic_role": "ATTACHMENT",
            "target_kind": "ATTACHMENT",
            "target_ref": "AT-COND",
        }
    )
    add_measurement(
        data,
        semantics={
            "measurement_identity": "count",
            "observed_value": {"kind": "INTEGER", "raw_value": "3", "value": 3},
        },
        roles=["MEASUREMENT_IDENTITY", "MEASUREMENT_VALUE"],
    )
    data["source_contexts"].append(
        {
            "context_ref": "M2",
            "context_type": "MEASUREMENT",
            "semantic_value": {"measurement_identity": "duration"},
        }
    )
    data["source_attachments"].append(
        {
            "attachment_ref": "AT-M2",
            "context_ref": "M2",
            "source_assertion_refs": ["A"],
            "relation": "APPLIES_TO",
            "status": "RESOLVED",
            "evidence_refs": ["eA"],
        }
    )
    data["evidence_bindings"].extend(
        [
            {
                "binding_ref": "bM2",
                "evidence_ref": "eA",
                "semantic_role": "MEASUREMENT_IDENTITY",
                "target_kind": "CONTEXT",
                "target_ref": "M2",
            },
            {
                "binding_ref": "bAT-M2",
                "evidence_ref": "eA",
                "semantic_role": "ATTACHMENT",
                "target_kind": "ATTACHMENT",
                "target_ref": "AT-M2",
            },
        ]
    )
    data["source_scopes"].append(
        {
            "source_scope_ref": "S",
            "kind": "CLAIM",
            "source_assertion_refs": ["A"],
            "evidence_refs": ["eA"],
        }
    )
    assert checked(data).status is V31ValidationStatus.RESOLVED


class FakeGateway:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def generate_structured(self, **kwargs: Any) -> CanonicalModelResult:
        self.calls.append(kwargs)
        return CanonicalModelResult(
            parsed_object=self.payload,
            usage=TokenUsage(
                input_tokens=4, output_tokens=6, total_tokens=10, accuracy=UsageAccuracy.EXACT
            ),
        )


def test_fake_gateway_uses_v31_contract_and_never_regenerates() -> None:
    data = base()
    add_scope(data, members=["A"])
    gateway = FakeGateway(data)
    extractor = SourceContextExtractorV31(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://fake.invalid/v1",
        api_key=SecretStr("not-real"),
        model="fake",
        max_calls=1,
    )
    result = asyncio.run(extractor.extract("Anchor P.", claim_ref="synthetic"))
    assert result.status is V31ValidationStatus.RESOLVED
    assert result.model_call_count == 1
    assert len(gateway.calls) == 1
    call = gateway.calls[0]
    assert call["allow_regeneration"] is False
    request = call["request"]
    assert request.metadata["schema_version"] == SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31
    assert request.metadata["prompt_version"] == SOURCE_CONTEXT_EXTRACTION_PROMPT_V31
    assert request.metadata["execution_mode"] == "SHADOW"
    assert "SourceScopeNode is the canonical owner" in request.instructions
    assert "formal_definition is optional" in request.instructions
    assert result.input_tokens == 4 and result.output_tokens == 6


def test_fake_gateway_rejects_bad_role_without_repair_or_second_call() -> None:
    data = base()
    add_measurement(
        data, semantics={"measurement_identity": "score"}, roles=["MEASUREMENT_DEFINITION"]
    )
    gateway = FakeGateway(data)
    extractor = SourceContextExtractorV31(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://fake.invalid/v1",
        api_key=SecretStr("not-real"),
        model="fake",
    )
    result = asyncio.run(extractor.extract("Anchor P."))
    assert result.status is V31ValidationStatus.INVALID
    assert len(gateway.calls) == 1
    assert gateway.calls[0]["allow_regeneration"] is False


def test_fake_gateway_schema_failure_consumes_only_single_call() -> None:
    data = base()
    data["schema_version"] = "wrong"
    gateway = FakeGateway(data)
    extractor = SourceContextExtractorV31(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://fake.invalid/v1",
        api_key=SecretStr("not-real"),
        model="fake",
    )
    result = asyncio.run(extractor.extract("Anchor P."))
    assert result.status is V31ValidationStatus.UNRESOLVED
    assert "schema_validation_failed" in result.diagnostics
    assert len(gateway.calls) == 1


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("typed_value", V31ValidationStatus.RESOLVED),
        ("identity_only", V31ValidationStatus.RESOLVED),
        ("value_without_unit", V31ValidationStatus.RESOLVED),
        ("wrong_definition_role", V31ValidationStatus.INVALID),
        ("invalid_scope", V31ValidationStatus.INVALID),
    ],
)
def test_fake_gateway_measurement_and_scope_contract_matrix(case, expected) -> None:
    data = base()
    if case == "typed_value":
        add_measurement(
            data,
            semantics={
                "measurement_identity": "latency",
                "observed_value": {"kind": "DECIMAL", "raw_value": "2.5", "value": "2.5"},
                "unit": "ms",
            },
            roles=["MEASUREMENT_IDENTITY", "MEASUREMENT_VALUE", "MEASUREMENT_UNIT"],
        )
    elif case == "identity_only":
        add_measurement(data, semantics={"measurement_identity": "latency"})
    elif case == "value_without_unit":
        add_measurement(
            data,
            semantics={
                "measurement_identity": "score",
                "observed_value": {"kind": "DECIMAL", "raw_value": "0.9", "value": "0.9"},
            },
            roles=["MEASUREMENT_IDENTITY", "MEASUREMENT_VALUE"],
        )
    elif case == "wrong_definition_role":
        add_measurement(
            data,
            semantics={
                "measurement_identity": "score",
                "observed_value": {"kind": "INTEGER", "raw_value": "9", "value": 9},
            },
            roles=["MEASUREMENT_IDENTITY", "MEASUREMENT_DEFINITION"],
        )
    else:
        add_scope(data, members=["absent"])
    gateway = FakeGateway(data)
    extractor = SourceContextExtractorV31(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://fake.invalid/v1",
        api_key=SecretStr("not-real"),
        model="fake",
    )
    result = asyncio.run(extractor.extract("Anchor P."))
    assert result.status is expected
    assert len(gateway.calls) == 1
    assert gateway.calls[0]["allow_regeneration"] is False


def test_fake_gateway_accepts_multiple_measurement_contexts() -> None:
    data = base()
    add_measurement(data, semantics={"measurement_identity": "latency"})
    data["source_contexts"].append(
        {
            "context_ref": "M2",
            "context_type": "MEASUREMENT",
            "semantic_value": {"measurement_identity": "throughput"},
        }
    )
    data["source_attachments"].append(
        {
            "attachment_ref": "AT-M2",
            "context_ref": "M2",
            "source_assertion_refs": ["A"],
            "relation": "APPLIES_TO",
            "status": "RESOLVED",
            "evidence_refs": ["eA"],
        }
    )
    data["evidence_bindings"].extend(
        [
            {
                "binding_ref": "bM2",
                "evidence_ref": "eA",
                "semantic_role": "MEASUREMENT_IDENTITY",
                "target_kind": "CONTEXT",
                "target_ref": "M2",
            },
            {
                "binding_ref": "bAT-M2",
                "evidence_ref": "eA",
                "semantic_role": "ATTACHMENT",
                "target_kind": "ATTACHMENT",
                "target_ref": "AT-M2",
            },
        ]
    )
    gateway = FakeGateway(data)
    extractor = SourceContextExtractorV31(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://fake.invalid/v1",
        api_key=SecretStr("not-real"),
        model="fake",
    )
    assert asyncio.run(extractor.extract("Anchor P.")).status is V31ValidationStatus.RESOLVED
    assert len(gateway.calls) == 1


def test_schema_and_prompt_are_explicitly_versioned() -> None:
    schema = SourceExtractionResponseV31.model_json_schema()
    assert SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31 == "source-context-extraction-schema-v3.1"
    assert SOURCE_CONTEXT_EXTRACTION_PROMPT_V31 == "source-context-extraction.generic.v3.1"
    assert "MeasurementSemanticsV31" in str(schema)
    assert "identity" in SOURCE_CONTEXT_V31_INSTRUCTIONS


def test_generic_fixture_inventory_covers_required_contract_cases() -> None:
    fixture_path = Path("tests/fixtures/source_context_extraction_v3_1_cases.json")
    cases = json.loads(fixture_path.read_text(encoding="utf-8"))["cases"]
    assert len(cases) == 24
    assert len({case["id"] for case in cases}) == 24


def test_basis_is_independent_and_requires_its_own_evidence_role() -> None:
    data = base()
    semantics = {"measurement_identity": "rate", "basis_ref": "BASIS"}
    add_measurement(data, semantics=semantics, roles=["MEASUREMENT_IDENTITY", "MEASUREMENT_BASIS"])
    data["measurement_bases"].append(
        {
            "basis_ref": "BASIS",
            "measurement_context_ref": "M",
            "basis_kind": "DENOMINATOR",
            "semantic_value": {"denominator": "item"},
            "evidence_refs": ["eA"],
        }
    )
    assert checked(data).status is V31ValidationStatus.RESOLVED

    absent = base()
    add_measurement(
        absent, semantics={"measurement_identity": "rate"}, roles=["MEASUREMENT_IDENTITY"]
    )
    assert not checked(absent).measurement_bases


def test_frozen_d4g_scope_only_replay_closes_duplicate_binding_blocker() -> None:
    root = Path("artifacts/v2_1r_d4g_b_execution_attempt_1")
    claim_files = [
        root / "claim_01_01a0c86f-e82d-7dd8-b42f-d48bba5b879e.json",
        root / "claim_03_01a0c41c-a7cc-73d2-9f55-6e19ede5be76.json",
        root / "claim_05_01a0c873-f07c-74b5-94e6-0c694bf0382b.json",
    ]
    assert all(path.exists() for path in claim_files)
    for path in claim_files:
        frozen = json.loads(path.read_text(encoding="utf-8"))
        raw = json.loads(frozen["raw_provider_structured_response"]["raw_structured_content"])
        raw["schema_version"] = SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31
        scope_only = {
            "schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V31,
            "status": "RESOLVED",
            "source_assertions": raw["source_assertions"],
            "source_contexts": [],
            "source_attachments": [],
            "source_scopes": raw["source_scopes"],
            "evidence_spans": raw["evidence_spans"],
            "evidence_bindings": [],
            "semantic_relations": [],
            "measurement_bases": [],
            "unresolved_items": [],
        }
        response = SourceExtractionResponseV31.model_validate(scope_only)
        assessment = validate_scope_ownership_v31(response, frozen["claim_text"])
        assert assessment.status is V31ValidationStatus.RESOLVED
        assert "scope_structure_binding_missing" not in assessment.diagnostics
        assert all(
            ref in {span.evidence_ref for span in response.evidence_spans}
            for scope in response.source_scopes
            for ref in scope.evidence_refs
        )
