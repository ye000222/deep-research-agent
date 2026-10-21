from __future__ import annotations

import asyncio
from typing import Any

import pytest
from app.domain.atomic_proposition_contracts import AtomicPropositionExecutionMode
from app.domain.providers import AdapterType, CanonicalModelResult, TokenUsage, UsageAccuracy
from app.services.atomic_proposition_decomposer import (
    ATOMIC_DECOMPOSITION_INSTRUCTIONS,
    AtomicPropositionDecomposer,
    ShadowDecompositionStatus,
)
from app.services.logical_composition_v3 import (
    ATOMIC_DECOMPOSITION_PROMPT_V3,
    ATOMIC_PROPOSITION_SCHEMA_V3,
    V3_COMPOSITION_INSTRUCTIONS,
    V3ShadowDecompositionResponse,
)
from pydantic import SecretStr, ValidationError

CLAIM_ID = "10000000-0000-0000-0000-000000000001"


def _prop(local_ref: str, text: str, *, source_span: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "local_ref": local_ref,
        "proposition_text": text,
        "subject": text.split()[0].rstrip("."),
        "predicate": "is asserted",
        "typed_value": {"kind": "boolean", "boolean": True},
        "polarity": "asserted",
    }
    if source_span is not None:
        result["source_span"] = source_span
    return result


def _ref(member_type: str, ref: str, *, source_span: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {"member_type": member_type, "ref": ref}
    if source_span is not None:
        value["source_span"] = source_span
    return value


def _response(
    claim: str,
    propositions: list[dict[str, Any]],
    *,
    roots: list[dict[str, Any]] | None = None,
    groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": ATOMIC_PROPOSITION_SCHEMA_V3,
        "decomposition_status": ("single_proposition" if len(propositions) == 1 else "decomposed"),
        "propositions": propositions,
    }
    if roots is not None:
        result["assertion_composition"] = {
            "root_expressions": roots,
            "groups": groups or [],
        }
    return result


class FakeGateway:
    def __init__(self, response: dict[str, Any]):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def generate_structured(self, **kwargs: Any) -> CanonicalModelResult:
        self.calls.append(kwargs)
        return CanonicalModelResult(
            text=None,
            parsed_object=self.response,
            usage=TokenUsage(
                input_tokens=3,
                output_tokens=5,
                total_tokens=8,
                accuracy=UsageAccuracy.ESTIMATED,
            ),
            provider_request_id="fake-only",
        )


def _run(response: dict[str, Any], claim: str) -> tuple[Any, FakeGateway]:
    gateway = FakeGateway(response)
    decomposer = AtomicPropositionDecomposer(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://not-contacted.invalid/v1",
        api_key=SecretStr("synthetic-only"),
        model="synthetic-model",
    )
    result = asyncio.run(
        decomposer.decompose_claim(
            claim,
            claim_id=CLAIM_ID,
            schema_version=ATOMIC_PROPOSITION_SCHEMA_V3,
            execution_mode=AtomicPropositionExecutionMode.SHADOW,
        )
    )
    return result, gateway


def _group(
    local_ref: str, operator: str, members: list[dict[str, Any]], claim: str
) -> dict[str, Any]:
    return {
        "local_ref": local_ref,
        "operator": operator,
        "members": members,
        "source_span": claim,
    }


def test_schema_v3_and_prompt_v3_are_shadow_selected_only() -> None:
    assert V3ShadowDecompositionResponse.model_json_schema()["properties"]["schema_version"]
    assert "assertion_composition" in V3ShadowDecompositionResponse.model_fields
    assert (
        "verification_eligibility"
        not in V3ShadowDecompositionResponse.model_json_schema()["$defs"]["V3Reference"][
            "properties"
        ]
    )
    assert ATOMIC_DECOMPOSITION_PROMPT_V3 == "atomic-decomposition.generic.v3"
    assert "alternatives into separate independent assertions" in " ".join(
        V3_COMPOSITION_INSTRUCTIONS.split()
    )
    assert all(token not in V3_COMPOSITION_INSTRUCTIONS for token in ("AnomalyR1", "MVTec", "YOLO"))


def test_single_proposition_stays_sparse_and_v2_stays_default() -> None:
    claim = "The system is ready."
    response = _response(claim, [_prop("p0", claim, source_span=claim)])
    result, gateway = _run(response, claim)
    assert result.status is ShadowDecompositionStatus.RESOLVED
    assert result.assertion_composition is None
    assert result.root_assertions == (
        {
            "member_type": "proposition",
            "ref": "p0",
            "position": 0,
            "source_span": claim,
        },
    )
    assert result.derived_member_eligibility["p0"] == "INDEPENDENTLY_ASSERTED"
    assert result.schema_version == ATOMIC_PROPOSITION_SCHEMA_V3
    assert result.prompt_version == ATOMIC_DECOMPOSITION_PROMPT_V3
    assert result.propositions[0].schema_version == ATOMIC_PROPOSITION_SCHEMA_V3
    assert len(gateway.calls) == 1
    assert gateway.calls[0]["allow_regeneration"] is False
    assert gateway.calls[0]["request"].metadata["schema_version"] == ATOMIC_PROPOSITION_SCHEMA_V3
    assert gateway.calls[0]["request"].metadata["prompt_version"] == ATOMIC_DECOMPOSITION_PROMPT_V3
    assert "assertion_composition" in gateway.calls[0]["request"].response_contract["properties"]
    assert "Never drop qualifiers" in gateway.calls[0]["request"].instructions
    assert ATOMIC_DECOMPOSITION_INSTRUCTIONS in gateway.calls[0]["request"].instructions


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("AND", {"p0": "INDEPENDENTLY_ASSERTED", "p1": "INDEPENDENTLY_ASSERTED"}),
        ("OR", {"p0": "NOT_INDEPENDENTLY_ASSERTED", "p1": "NOT_INDEPENDENTLY_ASSERTED"}),
        ("XOR", {"p0": "NOT_INDEPENDENTLY_ASSERTED", "p1": "NOT_INDEPENDENTLY_ASSERTED"}),
        (
            "AND_OR",
            {
                "p0": "INDEPENDENTLY_ASSERTED",
                "p1": "NOT_INDEPENDENTLY_ASSERTED",
                "p2": "NOT_INDEPENDENTLY_ASSERTED",
            },
        ),
        (
            "OR_AND",
            {
                "p0": "NOT_INDEPENDENTLY_ASSERTED",
                "p1": "NOT_INDEPENDENTLY_ASSERTED",
                "p2": "NOT_INDEPENDENTLY_ASSERTED",
            },
        ),
        (
            "UNKNOWN",
            {"p0": "UNKNOWN", "p1": "UNKNOWN"},
        ),
    ],
)
def test_fake_gateway_composition_matrix(case: str, expected: dict[str, str]) -> None:
    claim = "P and Q and R."
    if case in {"AND", "OR", "XOR", "UNKNOWN"}:
        props = [_prop("p0", "P", source_span="P"), _prop("p1", "Q", source_span="Q")]
        operator = case
        members = [
            _ref("proposition", "p0", source_span="P"),
            _ref("proposition", "p1", source_span="Q"),
        ]
        groups = [_group("g0", operator, members, claim)]
        roots = [_ref("group", "g0", source_span=claim)]
    elif case == "AND_OR":
        props = [
            _prop("p0", "P", source_span="P"),
            _prop("p1", "Q", source_span="Q"),
            _prop("p2", "R", source_span="R"),
        ]
        groups = [
            _group(
                "g0",
                "AND",
                [_ref("proposition", "p0", source_span="P"), _ref("group", "g1")],
                claim,
            ),
            _group(
                "g1",
                "OR",
                [
                    _ref("proposition", "p1", source_span="Q"),
                    _ref("proposition", "p2", source_span="R"),
                ],
                "Q and R",
            ),
        ]
        roots = [_ref("group", "g0", source_span=claim)]
    else:
        props = [
            _prop("p0", "P", source_span="P"),
            _prop("p1", "Q", source_span="Q"),
            _prop("p2", "R", source_span="R"),
        ]
        groups = [
            _group(
                "g0", "OR", [_ref("group", "g1"), _ref("proposition", "p2", source_span="R")], claim
            ),
            _group(
                "g1",
                "AND",
                [
                    _ref("proposition", "p0", source_span="P"),
                    _ref("proposition", "p1", source_span="Q"),
                ],
                "P and Q",
            ),
        ]
        roots = [_ref("group", "g0", source_span=claim)]
    result, gateway = _run(_response(claim, props, roots=roots, groups=groups), claim)
    assert result.status is ShadowDecompositionStatus.RESOLVED, result.validation_issues
    assert result.derived_member_eligibility == expected
    assert result.assertion_composition is not None
    assert len(gateway.calls) == 1
    assert result.validator_result["composition_validation_status"] == "VALID"


def test_multiple_independent_top_level_assertions_need_no_group_boilerplate() -> None:
    claim = "P. Q."
    payload = _response(
        claim,
        [_prop("p0", "P.", source_span="P."), _prop("p1", "Q.", source_span="Q.")],
        roots=[
            _ref("proposition", "p0", source_span="P."),
            _ref("proposition", "p1", source_span="Q."),
        ],
    )
    result, _ = _run(payload, claim)
    assert result.status is ShadowDecompositionStatus.RESOLVED
    assert result.assertion_composition.groups == ()
    assert result.derived_member_eligibility == {
        "p0": "INDEPENDENTLY_ASSERTED",
        "p1": "INDEPENDENTLY_ASSERTED",
    }


def test_condition_causality_and_ambiguous_connective_are_not_misencoded() -> None:
    condition_claim = "If P then Q."
    conditional = _response(
        condition_claim,
        [
            {
                **_prop("p0", "Q when P", source_span=condition_claim),
                "material_qualifiers": [
                    {
                        "category": "condition",
                        "key": "condition",
                        "value": "P",
                        "materiality": "material",
                        "source_span": "If P",
                    }
                ],
            }
        ],
    )
    condition_result, _ = _run(conditional, condition_claim)
    assert condition_result.status is ShadowDecompositionStatus.RESOLVED
    assert condition_result.assertion_composition is None
    assert len(condition_result.propositions) == 1
    assert condition_result.propositions[0].qualifiers[0].value == "P"

    causal_claim = "P because Q."
    causal_prop = _prop("p0", "P causes Q", source_span=causal_claim)
    causal_prop.update(
        {
            "subject": "P",
            "predicate": "causes",
            "typed_value": {"kind": "text", "text": "Q"},
        }
    )
    causal_result, _ = _run(_response(causal_claim, [causal_prop]), causal_claim)
    assert causal_result.status is ShadowDecompositionStatus.RESOLVED
    assert causal_result.assertion_composition is None
    assert causal_result.propositions[0].predicate == "causes"
    assert causal_result.propositions[0].value.text == "Q"

    ambiguous_claim = "P while Q."
    ambiguous = _response(
        ambiguous_claim,
        [_prop("p0", "P", source_span="P"), _prop("p1", "Q", source_span="Q")],
        roots=[_ref("group", "g0", source_span=ambiguous_claim)],
        groups=[
            _group(
                "g0",
                "UNKNOWN",
                [
                    _ref("proposition", "p0", source_span="P"),
                    _ref("proposition", "p1", source_span="Q"),
                ],
                ambiguous_claim,
            )
        ],
    )
    ambiguous_result, _ = _run(ambiguous, ambiguous_claim)
    assert ambiguous_result.status is ShadowDecompositionStatus.RESOLVED
    assert set(ambiguous_result.derived_member_eligibility.values()) == {"UNKNOWN"}


def test_v3_model_cannot_supply_verification_eligibility() -> None:
    payload = _response("P.", [_prop("p0", "P.")])
    payload["assertion_composition"] = {
        "root_expressions": [
            _ref("proposition", "p0") | {"verification_eligibility": "INDEPENDENTLY_ASSERTED"}
        ],
        "groups": [],
    }
    with pytest.raises(ValidationError):
        V3ShadowDecompositionResponse.model_validate(payload)


@pytest.mark.parametrize(
    ("mutation", "expected_status"),
    [
        ("missing_prop", ShadowDecompositionStatus.COMPOSITION_REFERENCE_INVALID),
        ("missing_group", ShadowDecompositionStatus.COMPOSITION_REFERENCE_INVALID),
        ("cycle", ShadowDecompositionStatus.COMPOSITION_CYCLE),
        ("orphan", ShadowDecompositionStatus.COMPOSITION_REFERENCE_INVALID),
        ("duplicate", ShadowDecompositionStatus.COMPOSITION_REFERENCE_INVALID),
        ("operator", ShadowDecompositionStatus.COMPOSITION_SCHEMA_INVALID),
        ("source_span", ShadowDecompositionStatus.COMPOSITION_SEMANTIC_INVALID),
        ("span_conflict", ShadowDecompositionStatus.COMPOSITION_SEMANTIC_INVALID),
        ("duplicate_local_id", ShadowDecompositionStatus.COMPOSITION_REFERENCE_INVALID),
    ],
)
def test_malformed_composition_fails_closed(
    mutation: str, expected_status: ShadowDecompositionStatus
) -> None:
    claim = "P and Q."
    payload = _response(
        claim,
        [_prop("p0", "P", source_span="P"), _prop("p1", "Q", source_span="Q")],
        roots=[_ref("group", "g0", source_span=claim)],
        groups=[
            _group(
                "g0",
                "AND",
                [
                    _ref("proposition", "p0", source_span="P"),
                    _ref("proposition", "p1", source_span="Q"),
                ],
                claim,
            )
        ],
    )
    composition = payload["assertion_composition"]
    if mutation == "missing_prop":
        composition["groups"][0]["members"][1]["ref"] = "p9"
    elif mutation == "missing_group":
        composition["root_expressions"][0]["ref"] = "g9"
    elif mutation == "cycle":
        composition["groups"][0]["members"] = [_ref("group", "g0"), _ref("proposition", "p1")]
    elif mutation == "orphan":
        composition["groups"].append(
            _group("g1", "AND", [_ref("proposition", "p0"), _ref("proposition", "p1")], claim)
        )
    elif mutation == "duplicate":
        composition["groups"][0]["members"][1] = _ref("proposition", "p0")
    elif mutation == "operator":
        composition["groups"][0]["operator"] = "NOR"
    elif mutation == "source_span":
        composition["groups"][0]["members"][0]["source_span"] = "not in Claim"
    elif mutation == "span_conflict":
        composition["groups"][0]["members"][0]["source_span"] = "Q"
    elif mutation == "duplicate_local_id":
        payload["propositions"][1]["local_ref"] = "p0"
    result, gateway = _run(payload, claim)
    assert result.status is expected_status
    assert result.propositions == ()
    assert result.composition_validation_issues
    assert len(gateway.calls) == 1


def test_unknown_connective_must_not_become_boolean_operator() -> None:
    claim = "P but Q."
    response = _response(
        claim,
        [_prop("p0", "P", source_span="P"), _prop("p1", "Q", source_span="Q")],
        roots=[_ref("group", "g0", source_span=claim)],
        groups=[
            _group(
                "g0",
                "UNKNOWN",
                [
                    _ref("proposition", "p0", source_span="P"),
                    _ref("proposition", "p1", source_span="Q"),
                ],
                claim,
            )
        ],
    )
    result, _ = _run(response, claim)
    assert result.derived_member_eligibility == {"p0": "UNKNOWN", "p1": "UNKNOWN"}


def test_v2_shadow_path_remains_readable_and_uses_frozen_versions() -> None:
    claim = "The system is ready."
    gateway = FakeGateway(
        {
            "schema_version": "atomic-proposition-schema-v2",
            "decomposition_status": "single_proposition",
            "propositions": [
                {
                    "proposition_text": claim,
                    "subject": "system",
                    "predicate": "is ready",
                    "typed_value": {"kind": "boolean", "boolean": True},
                    "polarity": "asserted",
                }
            ],
        }
    )
    decomposer = AtomicPropositionDecomposer(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://not-contacted.invalid/v1",
        api_key=SecretStr("synthetic-only"),
        model="synthetic-model",
    )
    result = asyncio.run(decomposer.decompose_claim(claim, claim_id=CLAIM_ID))
    assert result.status is ShadowDecompositionStatus.RESOLVED
    assert result.schema_version == "atomic-proposition-schema-v2"
    assert result.assertion_composition is None
    assert gateway.calls[0]["request"].metadata["schema_version"] == "atomic-proposition-schema-v2"
    assert (
        "assertion_composition" not in gateway.calls[0]["request"].response_contract["properties"]
    )


def test_multiple_propositions_without_explicit_roots_are_unresolved() -> None:
    claim = "P. Q."
    response = _response(claim, [_prop("p0", "P."), _prop("p1", "Q.")])
    result, _ = _run(response, claim)
    assert result.status is ShadowDecompositionStatus.COMPOSITION_UNRESOLVED
    assert result.composition_validation_issues[0].code == "composition_roots_missing"


def test_normalized_shadow_diagnostics_keep_safe_graph_and_derived_gate() -> None:
    claim = "P or Q."
    response = _response(
        claim,
        [_prop("p0", "P", source_span="P"), _prop("p1", "Q", source_span="Q")],
        roots=[_ref("group", "g0", source_span=claim)],
        groups=[
            _group(
                "g0",
                "OR",
                [
                    _ref("proposition", "p0", source_span="P"),
                    _ref("proposition", "p1", source_span="Q"),
                ],
                claim,
            )
        ],
    )
    result, _ = _run(response, claim)
    assert result.validator_result["normalized_composition"]["groups"][0]["operator"] == "OR"
    assert result.validator_result["derived_member_eligibility"] == {
        "p0": "NOT_INDEPENDENTLY_ASSERTED",
        "p1": "NOT_INDEPENDENTLY_ASSERTED",
    }
    assert result.model_structured_result["assertion_composition"]["groups"][0]["operator"] == "OR"
    assert result.validator_result["validation_issue_codes"] == []
    assert all(
        item["status"] == "NOT_APPLICABLE" for item in result.validator_result["qualifier_checks"]
    )
