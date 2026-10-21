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
from app.domain.source_context_extraction import (
    SourceContextExtractionResult,
    SourceExtractionResponse,
)
from app.domain.source_context_grounding_v2 import (
    EvidenceTargetKind,
    GroundingStatusV2,
    MaterialNodeUsability,
    SourceExtractionResponseV2,
)
from app.domain.source_context_grounding_v2 import (
    GroundingDimensionStatus as DS,
)
from app.domain.source_context_grounding_v2 import (
    SemanticEvidenceRole as Role,
)
from app.services.source_context_extractor_v2 import (
    SOURCE_CONTEXT_V2_INSTRUCTIONS,
    SourceContextExtractorV2,
    legacy_v1_compatibility_diagnostics,
    validate_source_grounding_v2,
)
from pydantic import SecretStr

FIXTURE_PATH = Path("tests/fixtures/source_context_grounding_v2_cases.json")
FIXTURE_CASES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]
assert len(FIXTURE_CASES) == 20


class FakeGateway:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def generate_structured(self, **kwargs: Any) -> CanonicalModelResult:
        self.calls.append(kwargs)
        return CanonicalModelResult(
            parsed_object=self.payload,
            usage=TokenUsage(
                input_tokens=17,
                output_tokens=23,
                total_tokens=40,
                accuracy=UsageAccuracy.EXACT,
            ),
        )


def _base(claim: str) -> dict[str, Any]:
    return {
        "schema_version": "source-context-extraction-schema-v2",
        "status": "RESOLVED",
        "source_assertions": [
            {
                "source_assertion_ref": "a0",
                "evidence_refs": ["e_assertion"],
                "assertion_status": "ASSERTED",
            }
        ],
        "source_contexts": [],
        "source_attachments": [],
        "source_scopes": [],
        "evidence_spans": [
            {
                "evidence_ref": "e_assertion",
                "exact_text": claim,
                "occurrence_index": 0,
            }
        ],
        "evidence_bindings": [
            {
                "binding_ref": "b_assertion",
                "evidence_ref": "e_assertion",
                "semantic_role": "ASSERTION_EVIDENCE",
                "target_kind": "ASSERTION",
                "target_ref": "a0",
            }
        ],
        "unresolved_items": [],
    }


def _context(
    payload: dict[str, Any],
    claim: str,
    *,
    context_type: str,
    semantic: dict[str, Any],
    quote: str,
    role: Role,
    binding_ref: str = "b_context",
    evidence_ref: str = "e_context",
) -> None:
    payload["evidence_spans"].append(
        {
            "evidence_ref": evidence_ref,
            "exact_text": quote,
            "occurrence_index": 0,
        }
    )
    payload["source_contexts"].append(
        {
            "context_ref": "c0",
            "context_type": context_type,
            "semantic_value": semantic,
            "materiality": "MATERIAL",
        }
    )
    payload["evidence_bindings"].append(
        {
            "binding_ref": binding_ref,
            "evidence_ref": evidence_ref,
            "semantic_role": role.value,
            "target_kind": "CONTEXT",
            "target_ref": "c0",
        }
    )
    payload["source_attachments"].append(
        {
            "attachment_ref": "x0",
            "context_ref": "c0",
            "source_assertion_refs": ["a0"],
            "relation": "APPLIES_TO",
            "status": "RESOLVED",
            "evidence_refs": [evidence_ref],
        }
    )
    payload["evidence_bindings"].append(
        {
            "binding_ref": "b_attach",
            "evidence_ref": evidence_ref,
            "semantic_role": "ATTACHMENT",
            "target_kind": "ATTACHMENT",
            "target_ref": "x0",
        }
    )


def _shared_scope_payload() -> tuple[str, dict[str, Any]]:
    claim = "In 2025 Atlas grew and Nova expanded."
    payload = {
        "schema_version": "source-context-extraction-schema-v2",
        "status": "RESOLVED",
        "source_assertions": [
            {"source_assertion_ref": "a0", "evidence_refs": ["ea"], "assertion_status": "ASSERTED"},
            {"source_assertion_ref": "a1", "evidence_refs": ["eb"], "assertion_status": "ASSERTED"},
        ],
        "source_contexts": [
            {
                "context_ref": "c0",
                "context_type": "TEMPORAL",
                "semantic_value": {"year": 2025},
                "materiality": "MATERIAL",
            }
        ],
        "source_attachments": [
            {
                "attachment_ref": "x0",
                "context_ref": "c0",
                "source_assertion_refs": ["a0", "a1"],
                "relation": "APPLIES_TO",
                "status": "RESOLVED",
                "source_scope_ref": "g0",
                "evidence_refs": [],
            }
        ],
        "source_scopes": [
            {
                "source_scope_ref": "g0",
                "kind": "ASSERTION_SET",
                "source_assertion_refs": ["a0", "a1"],
                "evidence_refs": ["eg"],
            }
        ],
        "evidence_spans": [
            {"evidence_ref": "ea", "exact_text": "Atlas grew", "occurrence_index": 0},
            {"evidence_ref": "eb", "exact_text": "Nova expanded", "occurrence_index": 0},
            {"evidence_ref": "ey", "exact_text": "2025", "occurrence_index": 0},
            {"evidence_ref": "eg", "exact_text": claim, "occurrence_index": 0},
        ],
        "evidence_bindings": [
            {
                "binding_ref": "ba",
                "evidence_ref": "ea",
                "semantic_role": "ASSERTION_EVIDENCE",
                "target_kind": "ASSERTION",
                "target_ref": "a0",
            },
            {
                "binding_ref": "bb",
                "evidence_ref": "eb",
                "semantic_role": "ASSERTION_EVIDENCE",
                "target_kind": "ASSERTION",
                "target_ref": "a1",
            },
            {
                "binding_ref": "by",
                "evidence_ref": "ey",
                "semantic_role": "TEMPORAL_VALUE",
                "target_kind": "CONTEXT",
                "target_ref": "c0",
            },
            {
                "binding_ref": "bg",
                "evidence_ref": "eg",
                "semantic_role": "SCOPE_STRUCTURE",
                "target_kind": "SCOPE",
                "target_ref": "g0",
            },
        ],
        "unresolved_items": [],
    }
    return claim, payload


def _case_payload(kind: str) -> tuple[str, dict[str, Any]]:
    cases = {
        "simple": ("Atlas reached 8 units.", None),
        "epistemic": ("Study A指出Atlas增长。", "epistemic"),
        "prediction": ("Atlas may grow.", "prediction"),
        "method": ("Using CNN detects defects.", "method"),
        "method_missing": ("CNN detects defects.", "method_missing"),
        "condition": ("Demand rises if orders increase.", "condition"),
        "scope": ("Rural households adopted Atlas.", "scope"),
        "measurement": ("Revenue was 8 USD.", "measurement"),
        "fabricated_span": ("Atlas grew.", "fabricated_span"),
        "semantic_ambiguous": ("据悉Atlas增长。", "semantic_ambiguous"),
        "dangling_ref": ("Atlas grew.", "dangling_ref"),
        "repeated": ("Atlas grew; Atlas grew.", "repeated"),
        "nested": ("In 2025 Atlas grew.", "nested"),
        "fabricated_agent": ("Atlas grew.", "fabricated_agent"),
        "wrong_clause": ("Acme reports; Nova grew.", "wrong_clause"),
        "canonical_nonliteral": ("报告指出Atlas增长。", "canonical_nonliteral"),
        "direct_attachment": ("In 2025 Atlas grew.", "direct_attachment"),
        "span_reuse": ("Atlas may grow.", "span_reuse"),
    }
    if kind in {"shared_scope", "shared_attachment"}:
        return _shared_scope_payload()
    claim, shape = cases[kind]
    payload = _base(claim)
    if shape is None:
        return claim, payload
    if shape == "epistemic":
        _context(
            payload,
            claim,
            context_type="EPISTEMIC",
            semantic={"operator": "REPORTS", "agent": "Study A"},
            quote="指出",
            role=Role.EPISTEMIC_OPERATOR,
        )
        payload["evidence_spans"].append(
            {"evidence_ref": "e_agent", "exact_text": "Study A", "occurrence_index": 0}
        )
        payload["evidence_bindings"].append(
            {
                "binding_ref": "b_agent",
                "evidence_ref": "e_agent",
                "semantic_role": "EPISTEMIC_AGENT",
                "target_kind": "CONTEXT",
                "target_ref": "c0",
            }
        )
    elif shape == "prediction":
        _context(
            payload,
            claim,
            context_type="EPISTEMIC",
            semantic={"modality": "may"},
            quote="may",
            role=Role.MODALITY,
        )
    elif shape == "method":
        _context(
            payload,
            claim,
            context_type="METHOD",
            semantic={"method_relation": "Using", "method": "CNN"},
            quote="Using",
            role=Role.METHOD_RELATION,
        )
        payload["evidence_spans"].append(
            {"evidence_ref": "e_method", "exact_text": "CNN", "occurrence_index": 0}
        )
        payload["evidence_bindings"].append(
            {
                "binding_ref": "b_method",
                "evidence_ref": "e_method",
                "semantic_role": "METHOD_VALUE",
                "target_kind": "CONTEXT",
                "target_ref": "c0",
            }
        )
        payload["evidence_bindings"][1]["related_binding_ref"] = "b_method"
    elif shape == "method_missing":
        payload["source_contexts"].append(
            {
                "context_ref": "c0",
                "context_type": "METHOD",
                "semantic_value": {"method": "CNN"},
                "materiality": "MATERIAL",
            }
        )
        payload["evidence_spans"].append(
            {"evidence_ref": "e_method", "exact_text": "CNN", "occurrence_index": 0}
        )
        payload["evidence_bindings"].append(
            {
                "binding_ref": "b_method",
                "evidence_ref": "e_method",
                "semantic_role": "METHOD_VALUE",
                "target_kind": "CONTEXT",
                "target_ref": "c0",
            }
        )
        payload["source_attachments"].append(
            {
                "attachment_ref": "x0",
                "context_ref": "c0",
                "source_assertion_refs": ["a0"],
                "relation": "APPLIES_TO",
                "status": "UNKNOWN",
                "evidence_refs": [],
            }
        )
    elif shape == "condition":
        _context(
            payload,
            claim,
            context_type="CONDITION",
            semantic={"condition": "orders increase"},
            quote="orders increase",
            role=Role.CONDITION,
        )
    elif shape == "scope":
        _context(
            payload,
            claim,
            context_type="SCOPE",
            semantic={"population": "Rural households"},
            quote="Rural households",
            role=Role.SCOPE_VALUE,
        )
    elif shape == "measurement":
        _context(
            payload,
            claim,
            context_type="MEASUREMENT",
            semantic={"definition": "Revenue"},
            quote="Revenue",
            role=Role.MEASUREMENT_DEFINITION,
        )
    elif shape == "fabricated_span":
        payload["evidence_spans"][0]["exact_text"] = "Invented assertion"
    elif shape == "semantic_ambiguous":
        _context(
            payload,
            claim,
            context_type="EPISTEMIC",
            semantic={"operator": "REPORTS"},
            quote="据悉",
            role=Role.EPISTEMIC_OPERATOR,
        )
    elif shape == "dangling_ref":
        payload["evidence_bindings"][0]["evidence_ref"] = "missing_evidence"
    elif shape == "repeated":
        payload["source_assertions"] = [
            {"source_assertion_ref": "a0", "evidence_refs": ["e0"], "assertion_status": "ASSERTED"},
            {"source_assertion_ref": "a1", "evidence_refs": ["e1"], "assertion_status": "ASSERTED"},
        ]
        payload["evidence_spans"] = [
            {"evidence_ref": "e0", "exact_text": "Atlas grew", "occurrence_index": 0},
            {"evidence_ref": "e1", "exact_text": "Atlas grew", "occurrence_index": 1},
        ]
        payload["evidence_bindings"] = [
            {
                "binding_ref": "b0",
                "evidence_ref": "e0",
                "semantic_role": "ASSERTION_EVIDENCE",
                "target_kind": "ASSERTION",
                "target_ref": "a0",
            },
            {
                "binding_ref": "b1",
                "evidence_ref": "e1",
                "semantic_role": "ASSERTION_EVIDENCE",
                "target_kind": "ASSERTION",
                "target_ref": "a1",
            },
        ]
    elif shape == "nested":
        payload["evidence_spans"].append(
            {"evidence_ref": "e_year", "exact_text": "2025", "occurrence_index": 0}
        )
        payload["source_contexts"].append(
            {
                "context_ref": "c0",
                "context_type": "TEMPORAL",
                "semantic_value": {"year": 2025},
                "materiality": "MATERIAL",
            }
        )
        payload["evidence_bindings"].append(
            {
                "binding_ref": "b_year",
                "evidence_ref": "e_year",
                "semantic_role": "TEMPORAL_VALUE",
                "target_kind": "CONTEXT",
                "target_ref": "c0",
            }
        )
        payload["source_attachments"].append(
            {
                "attachment_ref": "x0",
                "context_ref": "c0",
                "source_assertion_refs": ["a0"],
                "relation": "APPLIES_TO",
                "status": "UNKNOWN",
                "evidence_refs": [],
            }
        )
    elif shape == "fabricated_agent":
        _context(
            payload,
            claim,
            context_type="EPISTEMIC",
            semantic={"agent": "Retrieval Index"},
            quote="Atlas",
            role=Role.EPISTEMIC_AGENT,
        )
    elif shape == "wrong_clause":
        _context(
            payload,
            claim,
            context_type="EPISTEMIC",
            semantic={"agent": "Acme"},
            quote="Nova grew",
            role=Role.EPISTEMIC_AGENT,
        )
    elif shape == "canonical_nonliteral":
        _context(
            payload,
            claim,
            context_type="EPISTEMIC",
            semantic={"operator": "REPORTS"},
            quote="指出",
            role=Role.EPISTEMIC_OPERATOR,
        )
    elif shape == "direct_attachment":
        _context(
            payload,
            claim,
            context_type="TEMPORAL",
            semantic={"year": 2025},
            quote="2025",
            role=Role.TEMPORAL_VALUE,
        )
    elif shape == "span_reuse":
        _context(
            payload,
            claim,
            context_type="EPISTEMIC",
            semantic={"operator": "may", "modality": "may"},
            quote="may",
            role=Role.EPISTEMIC_OPERATOR,
        )
        payload["evidence_bindings"].append(
            {
                "binding_ref": "b_modality",
                "evidence_ref": "e_context",
                "semantic_role": "MODALITY",
                "target_kind": "CONTEXT",
                "target_ref": "c0",
            }
        )
    return claim, payload


@pytest.mark.parametrize("case", FIXTURE_CASES, ids=lambda case: case["case_id"])
def test_fake_gateway_v2_fixture_matrix(case: dict[str, str]) -> None:
    claim, payload = _case_payload(case["kind"])
    response = SourceExtractionResponseV2.model_validate(payload)
    result = validate_source_grounding_v2(response, claim, claim_ref=case["case_id"])
    assert result.claim_ref == case["case_id"]
    assert result.schema_version == "source-context-extraction-schema-v2"
    if case["kind"] == "fabricated_span":
        assert result.status is GroundingStatusV2.INVALID
    if case["kind"] in {"semantic_ambiguous", "canonical_nonliteral", "epistemic"}:
        assert result.status is GroundingStatusV2.AMBIGUOUS
    if case["kind"] == "method_missing":
        context = next(n for n in result.node_assessments if n.target_ref == "c0")
        assert context.dimensions.semantic_support is DS.MISSING_EVIDENCE
    if case["kind"] == "dangling_ref":
        assert "dangling_evidence_ref" in result.diagnostics
        assert result.status is GroundingStatusV2.INVALID
    if case["kind"] == "repeated":
        assert result.status is GroundingStatusV2.VALID
    if case["kind"] == "nested":
        span = next(s for s in result.evidence_spans if s.evidence_ref == "e_year")
        assert claim[span.start_offset : span.end_offset] == "2025"
        assert result.status is not GroundingStatusV2.INVALID
    if case["kind"] == "wrong_clause":
        assert result.status is GroundingStatusV2.INVALID


def test_fake_gateway_v2_exact_schema_prompt_and_one_call_boundary() -> None:
    claim, payload = _case_payload("simple")
    gateway = FakeGateway(payload)
    extractor = SourceContextExtractorV2(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://fake.invalid/v1",
        api_key=SecretStr("fake-secret"),
        model="fake-model",
        max_calls=1,
    )
    first = asyncio.run(extractor.extract(claim, claim_ref="synthetic"))
    second = asyncio.run(extractor.extract("another claim"))
    call = gateway.calls[0]
    request = call["request"]
    assert len(gateway.calls) == 1
    assert call["allow_regeneration"] is False
    assert request.metadata == {
        "prompt_version": "source-context-extraction.generic.v2",
        "schema_version": "source-context-extraction-schema-v2",
        "execution_mode": "SHADOW",
        "fusion_policy": "NONE",
    }
    assert request.response_contract == SourceExtractionResponseV2.model_json_schema()
    assert request.content_parts[0].value == {"claim_text": claim}
    assert first.status is GroundingStatusV2.VALID
    assert first.model_call_count == 1 and first.input_tokens == 17 and first.output_tokens == 23
    assert second.model_call_count == 0 and "call_budget_exhausted" in second.diagnostics
    assert "Atomic Proposition" in request.instructions
    assert "retrieval provenance" in request.instructions


def test_schema_v2_rejects_unknown_roles_and_provider_self_grounding_fields() -> None:
    _, payload = _case_payload("simple")
    payload["evidence_bindings"][0]["semantic_role"] = "CLAIM_SPECIFIC_MAGIC_ROLE"
    with pytest.raises(ValueError):
        SourceExtractionResponseV2.model_validate(payload)
    _, payload = _case_payload("simple")
    payload["evidence_bindings"][0]["support_status"] = "VALID"
    with pytest.raises(ValueError):
        SourceExtractionResponseV2.model_validate(payload)
    _, payload = _case_payload("simple")
    payload["evidence_spans"][0]["start_offset"] = 0
    with pytest.raises(ValueError):
        SourceExtractionResponseV2.model_validate(payload)


def test_role_specific_epistemic_and_method_support_do_not_use_enum_or_cue_lists() -> None:
    claim, payload = _case_payload("epistemic")
    result = validate_source_grounding_v2(SourceExtractionResponseV2.model_validate(payload), claim)
    operator = next(
        b for b in result.evidence_bindings if b.semantic_role is Role.EPISTEMIC_OPERATOR
    )
    agent = next(b for b in result.evidence_bindings if b.semantic_role is Role.EPISTEMIC_AGENT)
    assert operator.support_status is DS.AMBIGUOUS  # canonical REPORTS is not required as a literal
    assert agent.support_status is DS.VALID

    claim, payload = _case_payload("method")
    result = validate_source_grounding_v2(SourceExtractionResponseV2.model_validate(payload), claim)
    method_value = next(b for b in result.evidence_bindings if b.semantic_role is Role.METHOD_VALUE)
    assert method_value.support_status is DS.VALID
    # The relation is conservatively ambiguous unless its semantic payload itself is grounded.
    relation = next(b for b in result.evidence_bindings if b.semantic_role is Role.METHOD_RELATION)
    assert relation.support_status is DS.VALID


def test_false_accept_hard_negatives_fail_closed() -> None:
    claim = "Atlas grew; Atlas grew."
    payload = _base(claim)
    payload["evidence_spans"][0]["exact_text"] = "Atlas grew"
    payload["evidence_spans"][0]["occurrence_index"] = 4
    result = validate_source_grounding_v2(SourceExtractionResponseV2.model_validate(payload), claim)
    assert result.status is GroundingStatusV2.INVALID

    claim, payload = _case_payload("wrong_clause")
    result = validate_source_grounding_v2(SourceExtractionResponseV2.model_validate(payload), claim)
    assert result.status is GroundingStatusV2.INVALID

    claim, payload = _case_payload("semantic_ambiguous")
    result = validate_source_grounding_v2(SourceExtractionResponseV2.model_validate(payload), claim)
    context = next(n for n in result.node_assessments if n.target_ref == "c0")
    assert context.usability is MaterialNodeUsability.SEMANTIC_SUPPORT_AMBIGUOUS

    claim, payload = _case_payload("method")
    payload["evidence_bindings"][1]["related_binding_ref"] = "nonexistent_value_binding"
    result = validate_source_grounding_v2(SourceExtractionResponseV2.model_validate(payload), claim)
    assert result.status is GroundingStatusV2.INVALID
    assert "method_value_relation_unlinked" in result.diagnostics

    claim, payload = _shared_scope_payload()
    payload["evidence_bindings"] = [
        binding
        for binding in payload["evidence_bindings"]
        if binding["semantic_role"] != "SCOPE_STRUCTURE"
    ]
    result = validate_source_grounding_v2(SourceExtractionResponseV2.model_validate(payload), claim)
    assert result.status is GroundingStatusV2.INVALID
    assert "invalid_scope_graph" in result.diagnostics

    claim, payload = _case_payload("direct_attachment")
    payload["source_attachments"][0]["source_assertion_refs"] = ["missing_assertion"]
    result = validate_source_grounding_v2(SourceExtractionResponseV2.model_validate(payload), claim)
    assert result.status is GroundingStatusV2.INVALID
    assert "dangling_target_ref" in result.diagnostics


def test_shared_scope_attachment_is_structurally_valid_without_edge_quote_duplication() -> None:
    claim, payload = _shared_scope_payload()
    result = validate_source_grounding_v2(SourceExtractionResponseV2.model_validate(payload), claim)
    assert "invalid_scope_graph" not in result.diagnostics
    attach = next(
        n for n in result.node_assessments if n.target_kind is EvidenceTargetKind.ATTACHMENT
    )
    assert attach.dimensions.structural_grounding is DS.VALID
    assert attach.usability is not MaterialNodeUsability.UNGROUNDED_UNUSABLE


def test_material_usability_aggregation_fail_closed() -> None:
    claim, payload = _case_payload("canonical_nonliteral")
    result = validate_source_grounding_v2(SourceExtractionResponseV2.model_validate(payload), claim)
    context = next(n for n in result.node_assessments if n.target_ref == "c0")
    assert context.usability is MaterialNodeUsability.SEMANTIC_SUPPORT_AMBIGUOUS
    claim, payload = _case_payload("method_missing")
    result = validate_source_grounding_v2(SourceExtractionResponseV2.model_validate(payload), claim)
    context = next(n for n in result.node_assessments if n.target_ref == "c0")
    assert context.usability is MaterialNodeUsability.PARTIALLY_GROUNDED_UNRESOLVED


def test_v1_compatibility_never_manufactures_v2_bindings() -> None:
    legacy = SourceExtractionResponse.model_validate(
        {
            "schema_version": "source-context-extraction-schema-v1",
            "status": "RESOLVED",
            "source_assertions": [
                {
                    "source_assertion_ref": "a0",
                    "source_evidence": {"exact_text": "Atlas grew"},
                    "assertion_status": "ASSERTED",
                }
            ],
            "source_contexts": [
                {
                    "context_ref": "c0",
                    "context_type": "TEMPORAL",
                    "semantic_value": {"year": 2025},
                    "source_evidence": {"exact_text": "2025"},
                }
            ],
            "source_attachments": [],
            "source_scopes": [],
            "unresolved_items": [],
        }
    )
    claim_result = SourceContextExtractionResult(
        claim_ref="legacy",
        claim_hash="0" * 64,
        model_status="MODEL_OUTPUT_VALID",
        grounding_status="GROUNDING_VALID",
        source_contexts=(),
    )
    assert legacy.source_contexts
    assert legacy_v1_compatibility_diagnostics(claim_result) == ()
    # A legacy result that contains a context is explicitly marked unavailable.
    from app.domain.source_context_extraction import SourceContextNode, SpanEvidence

    node = SourceContextNode(
        context_ref="c0",
        context_type="TEMPORAL",
        semantic_value={"year": 2025},
        canonical_value={},
        lexical_value="2025",
        source_evidence=SpanEvidence(exact_text="2025"),
    )
    with_context = claim_result.model_copy(update={"source_contexts": (node,)})
    assert legacy_v1_compatibility_diagnostics(with_context) == (
        "LEGACY_EVIDENCE_BINDING_UNAVAILABLE",
    )


def test_auxiliary_detector_is_non_fusing_and_claim_payload_isolated() -> None:
    claim, payload = _case_payload("simple")
    gateway = FakeGateway(payload)
    extractor = SourceContextExtractorV2(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://fake.invalid",
        api_key=SecretStr("secret"),
        model="fake",
    )
    result = asyncio.run(extractor.extract(claim))
    request = gateway.calls[0]["request"]
    assert request.metadata["fusion_policy"] == "NONE"
    assert set(request.content_parts[0].value) == {"claim_text"}
    assert not hasattr(result, "auxiliary_detector_comparison")
    assert "Atomic Proposition" in SOURCE_CONTEXT_V2_INSTRUCTIONS
