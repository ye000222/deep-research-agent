from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from app.domain.providers import AdapterType, TokenUsage, UsageAccuracy
from app.domain.semantic_support_verification import (
    SEMANTIC_SUPPORT_VERIFIER_PROMPT_VERSION,
    SEMANTIC_SUPPORT_VERIFIER_SCHEMA_VERSION,
    BindingMateriality,
    BindingMaterialityRecord,
    SemanticSupportRoute,
    SemanticSupportSource,
    SemanticVerifierVerdict,
    VerifierFailureStatus,
)
from app.domain.source_context_extraction import (
    SourceAttachmentRelation,
    SourceScopeKind,
)
from app.domain.source_context_grounding_v2 import (
    DimensionGrounding,
    EvidenceTargetKind,
    GroundingDimensionStatus,
    GroundingStatusV2,
    SemanticEvidenceBinding,
    SemanticEvidenceRole,
    SourceAssertionV2,
    SourceAttachmentV2,
    SourceContextGroundingResultV2,
    SourceContextV2,
    SourceEvidenceSpan,
    SourceScopeV2,
)
from app.domain.truth_condition_context import ContextType
from app.services.hybrid_semantic_support import (
    build_semantic_verification_item,
    evaluate_hybrid_semantic_support,
    route_semantic_support,
)

FIXTURES = Path("tests/fixtures/semantic_support_hybrid_v1_cases.json")


class FakeGateway:
    def __init__(self, payload: dict[str, Any] | None = None, *, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def generate_structured(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            parsed_object=self.payload,
            text=None,
            usage=TokenUsage(
                input_tokens=17,
                output_tokens=9,
                total_tokens=26,
                accuracy=UsageAccuracy.EXACT,
            ),
        )


def _verdict(ref: str, verdict: str) -> dict[str, str]:
    return {
        "binding_ref": ref,
        "verdict": verdict,
        "reason_category": ("DIRECT_SUPPORT" if verdict == "SUPPORTED" else "RELATION_UNSUPPORTED"),
        "evidence_sufficiency": "SUFFICIENT",
        "target_compatibility": "COMPATIBLE" if verdict == "SUPPORTED" else "INCOMPATIBLE",
    }


def _result(
    *,
    role: SemanticEvidenceRole = SemanticEvidenceRole.ATTACHMENT,
    text_status: GroundingDimensionStatus = GroundingDimensionStatus.VALID,
    structural_status: GroundingDimensionStatus = GroundingDimensionStatus.VALID,
    refs: tuple[str, ...] = ("b1",),
    semantic_value: dict[str, Any] | None = None,
) -> tuple[SourceContextGroundingResultV2, str]:
    claim = "The method used 12 units for Acme in 2024."
    quote = "The method used 12 units"
    start = claim.index(quote)
    spans = (
        SourceEvidenceSpan(
            evidence_ref="e1",
            exact_text=quote,
            occurrence_index=0,
            start_offset=start,
            end_offset=start + len(quote),
            claim_hash="synthetic-hash",
            textual_grounding_status=text_status,
        ),
        SourceEvidenceSpan(
            evidence_ref="e2",
            exact_text="for Acme in 2024",
            occurrence_index=0,
            start_offset=claim.index("for Acme"),
            end_offset=len(claim) - 1,
            claim_hash="synthetic-hash",
            textual_grounding_status=GroundingDimensionStatus.VALID,
        ),
    )
    assertion = SourceAssertionV2(
        source_assertion_ref="a1",
        evidence_refs=("e1", "e2"),
        assertion_status="ASSERTED",
        source_scope_ref="s1",
    )
    context = SourceContextV2(
        context_ref="c1",
        context_type=ContextType.METHOD,
        semantic_value=semantic_value or {"method_relation": "used", "method": "method"},
        lexical_value="method",
    )
    attachment = SourceAttachmentV2(
        attachment_ref="at1",
        context_ref="c1",
        source_assertion_refs=("a1",),
        relation=SourceAttachmentRelation.APPLIES_TO,
        status="RESOLVED",
        evidence_refs=("e1",),
        source_scope_ref="s1",
    )
    scope = SourceScopeV2(
        source_scope_ref="s1",
        kind=SourceScopeKind.SINGLE_ASSERTION,
        source_assertion_refs=("a1",),
        evidence_refs=("e1", "e2"),
    )
    target_kind = (
        EvidenceTargetKind.ATTACHMENT
        if role is SemanticEvidenceRole.ATTACHMENT
        else EvidenceTargetKind.CONTEXT
    )
    target_ref = "at1" if target_kind is EvidenceTargetKind.ATTACHMENT else "c1"
    bindings = tuple(
        SemanticEvidenceBinding(
            binding_ref=ref,
            evidence_ref="e1",
            semantic_role=role,
            target_kind=target_kind,
            target_ref=target_ref,
            support_status=GroundingDimensionStatus.AMBIGUOUS,
            dimensions=DimensionGrounding(
                textual_grounding=text_status,
                semantic_support=GroundingDimensionStatus.AMBIGUOUS,
                structural_grounding=structural_status,
            ),
        )
        for ref in refs
    )
    return (
        SourceContextGroundingResultV2(
            claim_ref="synthetic-claim",
            claim_hash="synthetic-hash",
            source_assertions=(assertion,),
            source_contexts=(context,),
            source_attachments=(attachment,),
            source_scopes=(scope,),
            evidence_spans=spans,
            evidence_bindings=bindings,
            status=GroundingStatusV2.AMBIGUOUS,
        ),
        claim,
    )


def _run(
    result: SourceContextGroundingResultV2,
    claim: str,
    gateway: FakeGateway | None = None,
    **kwargs: Any,
) -> Any:
    return asyncio.run(
        evaluate_hybrid_semantic_support(
            result,
            claim,
            gateway=gateway,
            adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT if gateway else None,
            base_url="https://fake.invalid/v1" if gateway else None,
            api_key=__import__("pydantic").SecretStr("not-a-real-secret") if gateway else None,
            model="fake-model" if gateway else None,
            **kwargs,
        )
    )


def test_neutral_fixture_matrix_is_present_and_provider_free() -> None:
    data = json.loads(FIXTURES.read_text(encoding="utf-8"))
    assert data["provider_calls"] == 0
    assert len(data["cases"]) == 22


def test_exact_numeric_support_routes_deterministically_without_gateway() -> None:
    decision = route_semantic_support(
        "b1",
        role=SemanticEvidenceRole.CONTEXT_VALUE,
        target_kind=EvidenceTargetKind.CONTEXT,
        target_ref="c1",
        evidence_span=SourceEvidenceSpan(
            evidence_ref="e1",
            exact_text="12.0",
            occurrence_index=0,
            start_offset=0,
            end_offset=4,
            claim_hash="h",
            textual_grounding_status=GroundingDimensionStatus.VALID,
        ),
        textual_status=GroundingDimensionStatus.VALID,
        structural_status=GroundingDimensionStatus.VALID,
        target_payload={"semantic_value": {"value": 12}},
    )
    assert decision.route is SemanticSupportRoute.DETERMINISTICALLY_SUPPORTED
    assert decision.rule_id == "exact_decimal_value_equivalence"


def test_exact_numeric_mismatch_is_deterministically_rejected() -> None:
    decision = route_semantic_support(
        "b1",
        role=SemanticEvidenceRole.CONTEXT_VALUE,
        target_kind=EvidenceTargetKind.CONTEXT,
        target_ref="c1",
        evidence_span=SourceEvidenceSpan(
            evidence_ref="e1",
            exact_text="18",
            occurrence_index=0,
            start_offset=0,
            end_offset=2,
            claim_hash="h",
            textual_grounding_status=GroundingDimensionStatus.VALID,
        ),
        textual_status=GroundingDimensionStatus.VALID,
        structural_status=GroundingDimensionStatus.VALID,
        target_payload={"semantic_value": {"value": 12}},
    )
    assert decision.route is SemanticSupportRoute.DETERMINISTICALLY_REJECTED


@pytest.mark.parametrize(
    ("member", "expected"),
    [
        (True, SemanticSupportRoute.DETERMINISTICALLY_SUPPORTED),
        (False, SemanticSupportRoute.DETERMINISTICALLY_REJECTED),
    ],
)
def test_assertion_evidence_requires_explicit_graph_membership(
    member: bool, expected: SemanticSupportRoute
) -> None:
    decision = route_semantic_support(
        "b1",
        role=SemanticEvidenceRole.ASSERTION_EVIDENCE,
        target_kind=EvidenceTargetKind.ASSERTION,
        target_ref="a1",
        evidence_span=SourceEvidenceSpan(
            evidence_ref="e1",
            exact_text="Acme",
            occurrence_index=0,
            start_offset=0,
            end_offset=4,
            claim_hash="h",
            textual_grounding_status=GroundingDimensionStatus.VALID,
        ),
        textual_status=GroundingDimensionStatus.VALID,
        structural_status=GroundingDimensionStatus.VALID,
        target_payload={},
        assertion_membership_valid=member,
    )
    assert decision.route is expected


@pytest.mark.parametrize(
    ("role", "verdict", "expected"),
    [
        (SemanticEvidenceRole.ATTACHMENT, "SUPPORTED", GroundingDimensionStatus.VALID),
        (SemanticEvidenceRole.ATTACHMENT, "NOT_SUPPORTED", GroundingDimensionStatus.INVALID),
        (SemanticEvidenceRole.ATTACHMENT, "AMBIGUOUS", GroundingDimensionStatus.AMBIGUOUS),
        (
            SemanticEvidenceRole.ATTACHMENT,
            "INSUFFICIENT_EVIDENCE",
            GroundingDimensionStatus.MISSING_EVIDENCE,
        ),
        (SemanticEvidenceRole.METHOD_RELATION, "SUPPORTED", GroundingDimensionStatus.VALID),
        (SemanticEvidenceRole.METHOD_RELATION, "NOT_SUPPORTED", GroundingDimensionStatus.INVALID),
        (SemanticEvidenceRole.EPISTEMIC_OPERATOR, "SUPPORTED", GroundingDimensionStatus.VALID),
        (SemanticEvidenceRole.EPISTEMIC_AGENT, "NOT_SUPPORTED", GroundingDimensionStatus.INVALID),
        (SemanticEvidenceRole.CONDITION, "NOT_SUPPORTED", GroundingDimensionStatus.INVALID),
        (
            SemanticEvidenceRole.MEASUREMENT_DEFINITION,
            "NOT_SUPPORTED",
            GroundingDimensionStatus.INVALID,
        ),
        (SemanticEvidenceRole.MEASUREMENT_VALUE, "NOT_SUPPORTED", GroundingDimensionStatus.INVALID),
    ],
)
def test_fake_gateway_verdict_mapping(
    role: SemanticEvidenceRole, verdict: str, expected: GroundingDimensionStatus
) -> None:
    result, claim = _result(role=role)
    gateway = FakeGateway(
        {
            "schema_version": "semantic-support-verifier-schema-v1",
            "verdicts": [_verdict("b1", verdict)],
        }
    )
    output = _run(result, claim, gateway)
    assert output.bindings[0].semantic_support is expected
    assert output.bindings[0].support_source is SemanticSupportSource.INDEPENDENT_VERIFIER
    assert output.bindings[0].verifier_result is not None
    assert output.bindings[0].verifier_result.reason_category.value in {
        "DIRECT_SUPPORT",
        "RELATION_UNSUPPORTED",
    }
    assert len(gateway.calls) == 1
    assert output.verifier.call_count == 1
    assert output.verifier.provider_identity is None
    assert output.verifier.model_identity is None
    assert output.verifier.total_tokens == 26
    sent = gateway.calls[0]["request"].content_parts[0].value
    assert "support_status" not in json.dumps(sent)
    assert "confidence" not in json.dumps(sent)


@pytest.mark.parametrize(
    "role",
    [
        SemanticEvidenceRole.ATTACHMENT,
        SemanticEvidenceRole.EPISTEMIC_OPERATOR,
        SemanticEvidenceRole.EPISTEMIC_AGENT,
        SemanticEvidenceRole.METHOD_RELATION,
        SemanticEvidenceRole.MEASUREMENT_VALUE,
        SemanticEvidenceRole.MEASUREMENT_DEFINITION,
        SemanticEvidenceRole.CONDITION,
    ],
)
def test_hard_negative_semantic_associations_remain_rejected(
    role: SemanticEvidenceRole,
) -> None:
    result, claim = _result(role=role)
    gateway = FakeGateway(
        {
            "schema_version": "semantic-support-verifier-schema-v1",
            "verdicts": [_verdict("b1", "NOT_SUPPORTED")],
        }
    )
    output = _run(result, claim, gateway)
    assert len(gateway.calls) == 1
    assert output.bindings[0].semantic_support is GroundingDimensionStatus.INVALID
    assert output.bindings[0].verifier_verdict is SemanticVerifierVerdict.NOT_SUPPORTED


@pytest.mark.parametrize(
    ("textual", "structural"),
    [
        (GroundingDimensionStatus.INVALID, GroundingDimensionStatus.VALID),
        (GroundingDimensionStatus.VALID, GroundingDimensionStatus.INVALID),
    ],
)
def test_invalid_text_or_structure_never_calls_verifier(
    textual: GroundingDimensionStatus, structural: GroundingDimensionStatus
) -> None:
    result, claim = _result(text_status=textual, structural_status=structural)
    gateway = FakeGateway(
        {
            "schema_version": "semantic-support-verifier-schema-v1",
            "verdicts": [_verdict("b1", "SUPPORTED")],
        }
    )
    output = _run(result, claim, gateway)
    assert gateway.calls == []
    assert output.verifier.call_count == 0
    assert output.bindings[0].semantic_support is GroundingDimensionStatus.NOT_EVALUATED


def test_multiple_bindings_share_one_claim_scoped_verifier_call() -> None:
    result, claim = _result(refs=("b1", "b2", "b3"))
    gateway = FakeGateway(
        {
            "schema_version": "semantic-support-verifier-schema-v1",
            "verdicts": [_verdict(ref, "SUPPORTED") for ref in ("b1", "b2", "b3")],
        }
    )
    output = _run(result, claim, gateway)
    assert len(gateway.calls) == 1
    assert output.verifier.input_binding_refs == ("b1", "b2", "b3")


def test_no_inference_bindings_means_zero_verifier_calls() -> None:
    result, claim = _result(
        role=SemanticEvidenceRole.CONTEXT_VALUE,
        semantic_value={"value": "The method used 12 units"},
    )
    gateway = FakeGateway()
    output = _run(result, claim, gateway)
    assert gateway.calls == []
    assert output.verifier.call_count == 0
    assert output.bindings[0].semantic_support is GroundingDimensionStatus.VALID


@pytest.mark.parametrize(
    ("payload", "failure"),
    [
        ({"schema_version": "wrong", "verdicts": []}, VerifierFailureStatus.SCHEMA_INVALID),
        (
            {"schema_version": "semantic-support-verifier-schema-v1", "verdicts": []},
            VerifierFailureStatus.RESPONSE_OMITTED,
        ),
        (
            {
                "schema_version": "semantic-support-verifier-schema-v1",
                "verdicts": [_verdict("unknown-ref", "SUPPORTED")],
            },
            VerifierFailureStatus.BATCH_INVALID,
        ),
        (
            {
                "schema_version": "semantic-support-verifier-schema-v1",
                "verdicts": [_verdict("b1", "SUPPORTED"), _verdict("b1", "NOT_SUPPORTED")],
            },
            VerifierFailureStatus.BATCH_INVALID,
        ),
    ],
)
def test_malformed_or_incomplete_verifier_results_fail_closed(
    payload: dict[str, Any], failure: VerifierFailureStatus
) -> None:
    result, claim = _result()
    output = _run(result, claim, FakeGateway(payload))
    assert output.verifier.failure_status is failure
    assert output.bindings[0].semantic_support in {
        GroundingDimensionStatus.AMBIGUOUS,
        GroundingDimensionStatus.MISSING_EVIDENCE,
    }


def test_gateway_exception_fails_closed_without_retry() -> None:
    result, claim = _result()
    gateway = FakeGateway(error=TimeoutError("synthetic timeout"))
    output = _run(result, claim, gateway)
    assert len(gateway.calls) == 1
    assert output.verifier.call_count == 1
    assert output.verifier.failure_status is VerifierFailureStatus.GATEWAY_ERROR
    assert output.bindings[0].semantic_support is GroundingDimensionStatus.AMBIGUOUS


def test_local_context_is_bounded_and_contains_only_related_source_text() -> None:
    result, claim = _result()
    binding = result.evidence_bindings[0]
    item = build_semantic_verification_item(result, binding, claim)
    assert item is not None
    assert "The method used 12 units" in item.source_local_context
    assert "for Acme in 2024" in item.source_local_context
    assert len(item.source_local_context) <= 8_000
    assert "synthetic-hash" not in item.model_dump_json()


def test_requiredness_is_versioned_representation_and_not_verifier_decision() -> None:
    result, claim = _result(
        role=SemanticEvidenceRole.CONTEXT_VALUE,
        semantic_value={"value": "The method used 12 units"},
    )
    requiredness = BindingMaterialityRecord(
        binding_ref="b1", materiality=BindingMateriality.OPTIONAL_CORROBORATION
    )
    output = _run(result, claim, materiality=(requiredness,))
    assert output.bindings[0].materiality is BindingMateriality.OPTIONAL_CORROBORATION
    assert (
        output.bindings[0].materiality_contract_version
        == "semantic-support-requiredness.generic.v1"
    )


def test_verifier_cannot_create_new_binding_or_overwrite_proposal() -> None:
    result, claim = _result()
    original = result.evidence_bindings[0]
    gateway = FakeGateway(
        {
            "schema_version": "semantic-support-verifier-schema-v1",
            "verdicts": [_verdict("b1", "SUPPORTED")],
        }
    )
    output = _run(result, claim, gateway)
    assert result.evidence_bindings[0] == original
    assert len(output.bindings) == len(result.evidence_bindings)
    assert output.bindings[0].binding_ref == "b1"


def test_structural_only_materiality_is_not_semantically_routed() -> None:
    result, claim = _result(role=SemanticEvidenceRole.ATTACHMENT)
    gateway = FakeGateway()
    output = _run(
        result,
        claim,
        gateway,
        materiality=(
            BindingMaterialityRecord(
                binding_ref="b1", materiality=BindingMateriality.STRUCTURAL_SUPPORT_ONLY
            ),
        ),
    )
    assert gateway.calls == []
    assert output.bindings[0].router_decision.route is SemanticSupportRoute.NOT_APPLICABLE
    assert output.bindings[0].semantic_support is GroundingDimensionStatus.NOT_EVALUATED


def test_verifier_prompt_and_schema_are_separately_versioned_and_independent() -> None:
    result, claim = _result()
    gateway = FakeGateway(
        {
            "schema_version": "semantic-support-verifier-schema-v1",
            "verdicts": [_verdict("b1", "SUPPORTED")],
        }
    )
    _run(result, claim, gateway)
    request = gateway.calls[0]["request"]
    assert request.metadata["prompt_version"] == SEMANTIC_SUPPORT_VERIFIER_PROMPT_VERSION
    assert request.metadata["schema_version"] == SEMANTIC_SUPPORT_VERIFIER_SCHEMA_VERSION
    assert "confirm another system" in request.instructions
    assert "support_status" not in request.instructions
    assert "support_status" not in json.dumps(request.content_parts[0].value)
    assert request.response_contract["properties"]["verdicts"]


def test_missing_gateway_is_fail_closed_without_claiming_a_call() -> None:
    result, claim = _result()
    output = _run(result, claim)
    assert output.verifier.call_count == 0
    assert output.verifier.failure_status is VerifierFailureStatus.GATEWAY_UNAVAILABLE
    assert output.bindings[0].semantic_support is GroundingDimensionStatus.AMBIGUOUS


def test_fake_fixture_cases_have_generic_non_claim_specific_ids() -> None:
    data = json.loads(FIXTURES.read_text(encoding="utf-8"))
    assert all("claim_id" not in item and "expected_claim" not in item for item in data["cases"])
