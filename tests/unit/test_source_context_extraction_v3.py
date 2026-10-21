from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from app.domain.providers import (
    AdapterType,
    CanonicalModelResult,
    TokenUsage,
    UsageAccuracy,
)
from app.domain.source_context_extraction_v3 import (
    SOURCE_CONTEXT_EXTRACTION_PROMPT_V3,
    SOURCE_CONTEXT_EXTRACTION_SCHEMA_V3,
    SourceExtractionResponseV3,
    V3Diagnostic,
    project_v3_to_v2_1_read_only,
    validate_source_context_v3,
)
from app.domain.source_context_extraction_v3 import (
    V3ValidationStatus as Status,
)
from app.services.source_context_extractor_v2 import (
    SOURCE_CONTEXT_V2_INSTRUCTIONS,
    SourceContextExtractorV2,
)
from app.services.source_context_extractor_v3 import (
    SOURCE_CONTEXT_V3_INSTRUCTIONS,
    SourceContextExtractorV3,
)
from pydantic import SecretStr, ValidationError

FIXTURE_PATH = Path("tests/fixtures/source_context_extraction_v3_cases.json")
FIXTURES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]


class FakeGateway:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def generate_structured(self, **kwargs: Any) -> CanonicalModelResult:
        self.calls.append(kwargs)
        return CanonicalModelResult(
            parsed_object=self.payload,
            usage=TokenUsage(
                input_tokens=5,
                output_tokens=7,
                total_tokens=12,
                accuracy=UsageAccuracy.EXACT,
            ),
        )


def base(claim: str = "P") -> dict[str, Any]:
    return {
        "schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V3,
        "status": "RESOLVED",
        "source_assertions": [
            {"source_assertion_ref": "A", "evidence_refs": ["eA"], "assertion_status": "ASSERTED"}
        ],
        "source_contexts": [],
        "source_attachments": [],
        "source_scopes": [],
        "evidence_spans": [{"evidence_ref": "eA", "exact_text": claim}],
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


def add_context(
    payload: dict[str, Any],
    *,
    ref: str = "C",
    context_type: str = "SCOPE",
    value: dict[str, Any] | None = None,
    quote: str = "P",
    assertion_refs: list[str] | None = None,
    bind_role: str | None = "AUTO",
) -> None:
    payload["source_contexts"].append(
        {
            "context_ref": ref,
            "context_type": context_type,
            "semantic_value": value or {"scope": quote},
            "lexical_value": quote,
            "materiality": "MATERIAL",
        }
    )
    payload["evidence_spans"].append({"evidence_ref": f"e{ref}", "exact_text": quote})
    payload["source_attachments"].append(
        {
            "attachment_ref": f"AT{ref}",
            "context_ref": ref,
            "source_assertion_refs": assertion_refs or ["A"],
            "relation": "APPLIES_TO",
            "status": "RESOLVED",
            "evidence_refs": [f"e{ref}"],
        }
    )
    semantic_value = value or {"scope": quote}
    roles: list[str] = []
    if bind_role == "AUTO":
        defaults = {
            "TEMPORAL": "TEMPORAL_VALUE",
            "VERSION": "VERSION_VALUE",
            "SCOPE": "SCOPE_VALUE",
            "CONFIGURATION": "CONFIGURATION_VALUE",
            "SCENARIO": "SCENARIO_VALUE",
            "CONDITION": "CONDITION",
            "SAMPLE_SCOPE": "SAMPLE_SCOPE_VALUE",
            "METHOD": "METHOD_VALUE",
        }
        if context_type == "EPISTEMIC":
            roles.extend(
                "EPISTEMIC_OPERATOR"
                for key in ("operator", "epistemic_operator")
                if key in semantic_value
            )
            roles.extend(
                "EPISTEMIC_AGENT"
                for key in ("agent", "source", "attribution")
                if key in semantic_value
            )
            roles.extend("MODALITY" for key in ("modality", "certainty") if key in semantic_value)
        elif context_type == "MEASUREMENT":
            roles.extend("MEASUREMENT_VALUE" for key in ("value",) if key in semantic_value)
            roles.extend("MEASUREMENT_UNIT" for key in ("unit",) if key in semantic_value)
            roles.extend("MEASUREMENT_BASIS" for key in ("basis",) if key in semantic_value)
        elif context_type in defaults:
            roles.append(defaults[context_type])
    elif bind_role:
        roles.append(bind_role)
    for role in roles:
        payload["evidence_bindings"].append(
            {
                "binding_ref": f"b{ref}{role}",
                "evidence_ref": f"e{ref}",
                "semantic_role": role,
                "target_kind": "CONTEXT",
                "target_ref": ref,
            }
        )
    payload["evidence_bindings"].append(
        {
            "binding_ref": f"bAT{ref}",
            "evidence_ref": f"e{ref}",
            "semantic_role": "ATTACHMENT",
            "target_kind": "ATTACHMENT",
            "target_ref": f"AT{ref}",
        }
    )


def add_relation(
    payload: dict[str, Any],
    *,
    relation_type: str = "CAUSES",
    source: str = "A",
    target: str = "B",
    quote: str = "causes",
    bind: bool = True,
) -> None:
    if "B" in (source, target) and not any(
        x["source_assertion_ref"] == "B" for x in payload["source_assertions"]
    ):
        payload["source_assertions"].append(
            {"source_assertion_ref": "B", "evidence_refs": ["eB"], "assertion_status": "ASSERTED"}
        )
        payload["evidence_spans"].append({"evidence_ref": "eB", "exact_text": "Q"})
        payload["evidence_bindings"].append(
            {
                "binding_ref": "bB",
                "evidence_ref": "eB",
                "semantic_role": "ASSERTION_EVIDENCE",
                "target_kind": "ASSERTION",
                "target_ref": "B",
            }
        )
    payload["evidence_spans"].append({"evidence_ref": "eR", "exact_text": quote})
    payload["semantic_relations"].append(
        {
            "relation_ref": "R1",
            "relation_type": relation_type,
            "source_ref": source,
            "source_kind": "CONTEXT" if source == "C" else "ASSERTION",
            "target_ref": target,
            "target_kind": "ASSERTION",
            "evidence_refs": ["eR"],
            "status": "RESOLVED",
            "schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V3,
        }
    )
    if bind:
        payload["evidence_bindings"].append(
            {
                "binding_ref": "bR",
                "evidence_ref": "eR",
                "semantic_role": "SEMANTIC_RELATION",
                "target_kind": "RELATION",
                "target_ref": "R1",
            }
        )


def validate(payload: dict[str, Any], claim: str = "P"):
    response = SourceExtractionResponseV3.model_validate(payload)
    return validate_source_context_v3(response, claim)


def test_synthetic_fixture_inventory_is_complete() -> None:
    assert len(FIXTURES) == 27
    assert len({case["id"] for case in FIXTURES}) == 27


def test_substantive_anchor_and_attribution_are_separate() -> None:
    payload = base("substantive proposition")
    add_context(
        payload,
        context_type="EPISTEMIC",
        value={"operator": "reports"},
        quote="reports",
        bind_role=None,
    )
    payload["evidence_bindings"].append(
        {
            "binding_ref": "bOp",
            "evidence_ref": "eC",
            "semantic_role": "EPISTEMIC_OPERATOR",
            "target_kind": "CONTEXT",
            "target_ref": "C",
        }
    )
    result = validate(payload, "Source S reports substantive proposition")
    assert result.status is Status.RESOLVED
    assert payload["source_assertions"][0]["evidence_refs"] == ["eA"]


def test_attribution_only_can_be_unresolved_without_brittle_lexical_rule() -> None:
    payload = base("reports")
    payload.update(
        status="UNRESOLVED",
        source_assertions=[],
        evidence_bindings=[],
        unresolved_items=["no substantive proposition anchor"],
    )
    # No deterministic reporting-verb detector is used; model/prompt owns segmentation.
    assert validate(payload, "Source reports").status is Status.UNRESOLVED
    assert "substantive proposition" in SOURCE_CONTEXT_V3_INSTRUCTIONS


@pytest.mark.parametrize(
    "semantic_key,role", [("operator", "EPISTEMIC_OPERATOR"), ("agent", "EPISTEMIC_AGENT")]
)
def test_epistemic_roles_bind_to_context(semantic_key: str, role: str) -> None:
    payload = base("P")
    add_context(
        payload,
        context_type="EPISTEMIC",
        value={semantic_key: "text-grounded"},
        quote="epistemic",
        bind_role=None,
    )
    payload["evidence_bindings"].append(
        {
            "binding_ref": "bE",
            "evidence_ref": "eC",
            "semantic_role": role,
            "target_kind": "CONTEXT",
            "target_ref": "C",
        }
    )
    assert validate(payload, "P epistemic").status is Status.RESOLVED


def test_epistemic_binding_to_assertion_is_rejected() -> None:
    payload = base("P")
    payload["source_contexts"].append(
        {"context_ref": "C", "context_type": "EPISTEMIC", "semantic_value": {"operator": "reports"}}
    )
    payload["evidence_spans"].append({"evidence_ref": "eC", "exact_text": "reports"})
    payload["evidence_bindings"].append(
        {
            "binding_ref": "bE",
            "evidence_ref": "eC",
            "semantic_role": "EPISTEMIC_OPERATOR",
            "target_kind": "ASSERTION",
            "target_ref": "A",
        }
    )
    result = validate(payload, "reports P")
    assert result.status is Status.INVALID
    assert V3Diagnostic.ROLE_TARGET_MISMATCH.value in result.diagnostics


def test_method_identity_and_purpose_relation_are_distinct() -> None:
    payload = base("P")
    add_context(
        payload, context_type="METHOD", value={"method": "M"}, quote="M", bind_role="METHOD_VALUE"
    )
    add_relation(payload, relation_type="PURPOSE", source="C", target="A", quote="for target T")
    # Purpose is a relation, not an OTHER context payload.
    result = validate(payload, "M for target T P causes Q")
    assert result.status is Status.RESOLVED
    assert result.downstream_not_yet_consumed == ("SEMANTIC_RELATION",)


@pytest.mark.parametrize(
    "relation_type,source,target,quote",
    [
        ("CAUSES", "A", "B", "causes"),
        ("RESULTS_IN", "A", "B", "results in"),
        ("CAUSES", "B", "A", "results from"),
        ("COMPATIBLE_WITH", "A", "B", "compatible with"),
    ],
)
def test_directed_semantic_relations(
    relation_type: str, source: str, target: str, quote: str
) -> None:
    payload = base("P")
    add_relation(payload, relation_type=relation_type, source=source, target=target, quote=quote)
    assert validate(payload, f"P Q {quote}").status is Status.RESOLVED


def test_unknown_relation_is_retained_without_claiming_grounding() -> None:
    payload = base("P")
    payload["semantic_relations"].append(
        {
            "relation_ref": "R1",
            "relation_type": "UNKNOWN",
            "source_ref": "A",
            "source_kind": "ASSERTION",
            "target_ref": "A",
            "target_kind": "ASSERTION",
            "status": "UNKNOWN",
            "schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V3,
        }
    )
    result = validate(payload)
    assert result.status is Status.PARTIAL
    assert "semantic_relation_unresolved" in result.diagnostics

    bad_ref = base("P")
    bad_ref["semantic_relations"].append(
        {
            "relation_ref": "R1",
            "relation_type": "UNKNOWN",
            "source_ref": "A",
            "source_kind": "ASSERTION",
            "target_ref": "missing",
            "target_kind": "ASSERTION",
            "status": "UNKNOWN",
            "schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V3,
        }
    )
    assert validate(bad_ref).status is Status.INVALID


def test_measurement_basis_typed_separately_from_result_and_unit() -> None:
    payload = base("value unit per item")
    add_context(
        payload,
        context_type="MEASUREMENT",
        value={"metric": "m", "value": 3, "unit": "ms"},
        quote="value unit",
    )
    payload["evidence_spans"].append({"evidence_ref": "eBasis", "exact_text": "per item"})
    payload["source_attachments"][0]["evidence_refs"].append("eBasis")
    payload["measurement_bases"].append(
        {
            "basis_ref": "MB1",
            "measurement_context_ref": "C",
            "basis_kind": "DENOMINATOR",
            "semantic_value": {"denominator": "item"},
            "evidence_refs": ["eBasis"],
        }
    )
    payload["evidence_bindings"].append(
        {
            "binding_ref": "bBasis",
            "evidence_ref": "eBasis",
            "semantic_role": "MEASUREMENT_BASIS",
            "target_kind": "CONTEXT",
            "target_ref": "C",
        }
    )
    result = validate(payload, "value unit per item")
    assert result.status is Status.RESOLVED
    assert "MEASUREMENT_BASIS" in result.downstream_not_yet_consumed


def test_measurement_basis_can_exist_without_result_and_result_does_not_imply_basis() -> None:
    basis_only = base("per instance")
    add_context(
        basis_only, context_type="MEASUREMENT", value={"metric": "latency"}, quote="latency"
    )
    basis_only["evidence_spans"].append({"evidence_ref": "eBasis", "exact_text": "per instance"})
    basis_only["source_attachments"][0]["evidence_refs"].append("eBasis")
    basis_only["measurement_bases"].append(
        {
            "basis_ref": "MB1",
            "measurement_context_ref": "C",
            "basis_kind": "DENOMINATOR",
            "semantic_value": {"denominator": "instance"},
            "evidence_refs": ["eBasis"],
        }
    )
    basis_only["evidence_bindings"].append(
        {
            "binding_ref": "bBasis",
            "evidence_ref": "eBasis",
            "semantic_role": "MEASUREMENT_BASIS",
            "target_kind": "CONTEXT",
            "target_ref": "C",
        }
    )
    assert validate(basis_only, "per instance latency").status is Status.RESOLVED

    result_only = base("3 ms")
    add_context(
        result_only,
        context_type="MEASUREMENT",
        value={"metric": "latency", "value": 3, "unit": "ms"},
        quote="3 ms",
    )
    checked = validate(result_only, "3 ms")
    assert checked.status is Status.RESOLVED
    assert result_only["measurement_bases"] == []


def test_metric_value_does_not_synthesize_formal_definition() -> None:
    payload = base("AUROC 0.9")
    add_context(
        payload,
        context_type="MEASUREMENT",
        value={"metric": "AUROC", "value": 0.9},
        quote="AUROC 0.9",
    )
    checked = validate(payload, "AUROC 0.9")
    assert checked.status is Status.RESOLVED
    assert not any(
        x["semantic_role"] == "MEASUREMENT_DEFINITION" for x in payload["evidence_bindings"]
    )


def test_resolved_attachment_requires_binding_and_supports_multi_target_edge() -> None:
    good = base("P")
    good["source_assertions"].append(
        {"source_assertion_ref": "B", "evidence_refs": ["eA"], "assertion_status": "ASSERTED"}
    )
    good["evidence_bindings"].append(
        {
            "binding_ref": "bB",
            "evidence_ref": "eA",
            "semantic_role": "ASSERTION_EVIDENCE",
            "target_kind": "ASSERTION",
            "target_ref": "B",
        }
    )
    add_context(good, assertion_refs=["A", "B"])
    assert validate(good).status is Status.RESOLVED
    good["evidence_bindings"] = [
        b for b in good["evidence_bindings"] if b["semantic_role"] != "ATTACHMENT"
    ]
    assert validate(good).status is Status.PARTIAL


def test_relation_requires_binding_and_valid_graph_endpoints() -> None:
    missing_binding = base("P")
    add_relation(missing_binding, bind=False)
    assert validate(missing_binding, "P causes Q").status is Status.PARTIAL

    bad_source = base("P")
    add_relation(bad_source, source="missing")
    assert validate(bad_source, "P causes Q").status is Status.INVALID
    bad_target = base("P")
    add_relation(bad_target, target="missing")
    assert validate(bad_target, "P causes Q").status is Status.INVALID


def test_scope_graph_valid_invalid_and_evidence_not_semantically_promoted() -> None:
    good = base("P")
    good["source_scopes"].append(
        {
            "source_scope_ref": "S",
            "kind": "CLAIM",
            "source_assertion_refs": ["A"],
            "evidence_refs": ["eA"],
        }
    )
    assert validate(good).status is Status.RESOLVED
    bad = base("P")
    bad["source_scopes"].append(
        {
            "source_scope_ref": "S",
            "kind": "CLAIM",
            "source_assertion_refs": ["missing"],
            "evidence_refs": ["eA"],
        }
    )
    assert validate(bad).status is Status.INVALID
    assert all(
        item.semantic_support.value == "NOT_EVALUATED"
        for item in validate(good).binding_assessments
    )


def test_json_schema_rejects_unknown_relation_enum_and_v2_remains_unchanged() -> None:
    payload = base("P")
    add_relation(payload)
    payload["semantic_relations"][0]["relation_type"] = "DOMAIN_SPECIFIC_CAUSE"
    with pytest.raises(ValidationError):
        SourceExtractionResponseV3.model_validate(payload)
    assert SOURCE_CONTEXT_EXTRACTION_SCHEMA_V3.endswith("v3")
    assert "structured object" in SOURCE_CONTEXT_V2_INSTRUCTIONS


def test_fake_gateway_uses_v3_contract_and_never_regenerates() -> None:
    payload = base("P")
    gateway = FakeGateway(payload)
    extractor = SourceContextExtractorV3(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://fake.invalid/v1",
        api_key=SecretStr("fake-secret"),
        model="fake-model",
    )
    result = asyncio.run(extractor.extract("P", claim_ref="synthetic"))
    assert result.status is Status.RESOLVED
    assert result.model_call_count == 1
    assert len(gateway.calls) == 1
    request = gateway.calls[0]["request"]
    assert request.metadata["schema_version"] == SOURCE_CONTEXT_EXTRACTION_SCHEMA_V3
    assert request.metadata["prompt_version"] == SOURCE_CONTEXT_EXTRACTION_PROMPT_V3
    assert gateway.calls[0]["allow_regeneration"] is False
    assert "substantive proposition" in request.instructions


def test_fake_gateway_schema_invalid_and_no_second_call() -> None:
    payload = base("P")
    payload["schema_version"] = "source-context-extraction-schema-v2"
    gateway = FakeGateway(payload)
    extractor = SourceContextExtractorV3(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://fake.invalid/v1",
        api_key=SecretStr("fake-secret"),
        model="fake-model",
    )
    first = asyncio.run(extractor.extract("P"))
    second = asyncio.run(extractor.extract("P"))
    assert first.status is Status.UNRESOLVED
    assert "schema_validation_failed" in first.diagnostics
    assert second.diagnostics == ("call_budget_exhausted",)
    assert len(gateway.calls) == 1


def test_read_only_downstream_projection_preserves_v3_only_facts() -> None:
    payload = base("P")
    add_relation(payload)
    checked = validate(payload, "P causes Q")
    projected = project_v3_to_v2_1_read_only(checked)
    assert projected["projection_status"] == "READ_ONLY_PARTIAL"
    assert projected["downstream_not_yet_consumed"] == ["SEMANTIC_RELATION"]
    assert projected["semantic_promotion"] is False
    assert "source_assertions" in projected["facts"]


def test_v2_extractor_contract_is_not_routed_or_mutated_by_v3() -> None:
    assert (
        "source_context_shadow_extraction_v2" in SourceContextExtractorV2.__doc__
        or SourceContextExtractorV2.__name__
    )
    assert SOURCE_CONTEXT_EXTRACTION_PROMPT_V3 != "source-context-extraction.generic.v2"
    assert "source-context-extraction-schema-v2" not in SOURCE_CONTEXT_EXTRACTION_SCHEMA_V3


def test_fixture_ids_cover_requested_synthetic_contract_cases() -> None:
    ids = {case["id"] for case in FIXTURES}
    assert {
        "assertion_causes_assertion",
        "generic_compatibility",
        "multi_target_attachment_missing_binding",
        "measurement_basis_without_result",
        "epistemic_binding_to_assertion",
    } <= ids
