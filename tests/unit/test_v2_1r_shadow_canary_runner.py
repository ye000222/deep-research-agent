from __future__ import annotations

import asyncio
import json
import socket
from typing import Any

import httpx
import pytest
from app.domain.atomic_proposition_contracts import (
    ATOMIC_DECOMPOSITION_PROMPT_V3,
    ATOMIC_PROPOSITION_SCHEMA_V3,
    AtomicPropositionExecutionMode,
)
from app.domain.providers import AdapterType
from app.llm.adapters import LLMGateway
from app.services.atomic_proposition_decomposer import AtomicPropositionDecomposer
from pydantic import SecretStr

import scripts.run_v2_1r_shadow_canary as canary_module
from scripts.run_v2_1r_shadow_canary import (
    CanaryDecomposer,
    CanaryGateway,
    CanaryRequest,
    CanaryResult,
    build_canary_http_client,
    run_one_claim_canary,
)


@pytest.fixture(autouse=True)
def _isolate_canary_artifacts(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.setattr(canary_module, "CANARY_DIAGNOSTIC_ARTIFACT_DIR", tmp_path)


class FakeDecomposer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.provider_profile_id = "synthetic-profile"

    async def decompose_claim(
        self, claim_text: str, context_metadata: Any = None, **kwargs: Any
    ) -> Any:
        self.calls.append(
            {"claim_text": claim_text, "context_metadata": context_metadata, **kwargs}
        )
        return type(
            "FakeResult",
            (),
            {
                "model_call_count": 1,
                "model_dump": lambda self, mode="python": {
                    "claim_id": "synthetic-id",
                    "provenance": {},
                    "schema_version": "atomic-proposition-schema-v2",
                    "prompt_version": "atomic-decomposition.generic.v2",
                    "status": "RESOLVED",
                    "propositions": [],
                    "validation_issues": [],
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 0,
                    "model_metadata": {},
                    "model_call_count": 1,
                },
            },
        )()


def _request(**changes: Any) -> CanaryRequest:
    values: dict[str, Any] = {
        "claim_id": "synthetic-claim",
        "claim_text": "A caller-supplied synthetic assertion.",
        "provider_profile_id": "synthetic-profile",
        "profile_ready": True,
        "credential_ready": True,
        "schema_version": "atomic-proposition-schema-v2",
        "maximum_calls": 1,
        "holdout_exclusion_established": True,
        "excluded_claim_ids": frozenset({"synthetic-holdout-id"}),
        "context_metadata": {},
    }
    values.update(changes)
    return CanaryRequest(**values)


def _safe_decomposition(result: CanaryResult) -> dict[str, Any]:
    decomposition = result.safe_dict().get("decomposition")
    assert isinstance(decomposition, dict)
    return decomposition


def test_runner_invokes_injected_decomposer_exactly_once() -> None:
    decomposer = FakeDecomposer()
    result = asyncio.run(run_one_claim_canary(_request(), decomposer=decomposer))
    assert result.status == "COMPLETED"
    assert result.reason is None
    assert len(decomposer.calls) == 1
    assert decomposer.calls[0]["claim_id"] == "synthetic-claim"
    assert _safe_decomposition(result)["proposition_count"] == 0
    safe = result.safe_dict()
    assert safe["claim"]["claim_text"] == "A caller-supplied synthetic assertion."
    assert safe["decomposition"]["final_shadow_result"]["status"] == "RESOLVED"
    assert "status_transition_trace" in safe["decomposition"]


def test_runner_blocks_before_invocation_when_any_safety_guard_fails() -> None:
    guards = (
        {"provider_profile_id": None},
        {"profile_ready": False},
        {"credential_ready": False},
        {"schema_version": "unknown-schema"},
        {
            "schema_version": ATOMIC_PROPOSITION_SCHEMA_V3,
            "execution_mode": AtomicPropositionExecutionMode.PRODUCTION,
        },
        {"schema_version": ATOMIC_PROPOSITION_SCHEMA_V3, "prompt_version": "unknown-prompt"},
        {
            "schema_version": ATOMIC_PROPOSITION_SCHEMA_V3,
            "prompt_version": "atomic-decomposition.generic.v2",
        },
        {"maximum_calls": 2},
        {"holdout_exclusion_established": False},
        {"claim_id": "synthetic-holdout-id"},
        {"claim_text": ""},
        {"provider_profile_id": "different-profile"},
    )
    for override in guards:
        decomposer = FakeDecomposer()
        result = asyncio.run(run_one_claim_canary(_request(**override), decomposer=decomposer))
        assert result.status == "BLOCKED"
        assert decomposer.calls == []
        safe = result.safe_dict()
        assert safe["outbound_attempted"] is False
        assert safe["outbound_count"] == 0
        assert safe["response_received"] is False


def test_v3_shadow_canary_resolves_matching_prompt_before_decomposer_call() -> None:
    decomposer = FakeDecomposer()
    result = asyncio.run(
        run_one_claim_canary(
            _request(
                schema_version=ATOMIC_PROPOSITION_SCHEMA_V3,
                prompt_version=ATOMIC_DECOMPOSITION_PROMPT_V3,
            ),
            decomposer=decomposer,
        )
    )

    assert result.status == "COMPLETED"
    assert len(decomposer.calls) == 1
    assert decomposer.calls[0]["schema_version"] == ATOMIC_PROPOSITION_SCHEMA_V3
    assert decomposer.calls[0]["prompt_version"] == ATOMIC_DECOMPOSITION_PROMPT_V3
    assert decomposer.calls[0]["execution_mode"] is AtomicPropositionExecutionMode.SHADOW
    contract = result.safe_dict()["contract_resolution"]
    assert contract["requested_schema_version"] == ATOMIC_PROPOSITION_SCHEMA_V3
    assert contract["resolved_schema_version"] == ATOMIC_PROPOSITION_SCHEMA_V3
    assert contract["resolved_prompt_version"] == ATOMIC_DECOMPOSITION_PROMPT_V3
    assert contract["execution_mode"] == "SHADOW"
    assert contract["contract_resolution_status"] == "RESOLVED"


def test_contract_route_is_persisted_before_decomposer_invocation(monkeypatch: Any) -> None:
    timeline: list[str] = []
    persist = canary_module.persist_safe_canary_artifact

    def observe_persistence(result: CanaryResult, **kwargs: Any) -> Any:
        timeline.append(result.status)
        return persist(result, **kwargs)

    class ObservingDecomposer(FakeDecomposer):
        async def decompose_claim(
            self, claim_text: str, context_metadata: Any = None, **kwargs: Any
        ) -> Any:
            timeline.append("DECOMPOSER_INVOKED")
            return await super().decompose_claim(claim_text, context_metadata, **kwargs)

    monkeypatch.setattr(canary_module, "persist_safe_canary_artifact", observe_persistence)
    result = asyncio.run(
        run_one_claim_canary(
            _request(schema_version=ATOMIC_PROPOSITION_SCHEMA_V3),
            decomposer=ObservingDecomposer(),
        )
    )

    assert result.status == "COMPLETED"
    assert timeline == ["READY_TO_INVOKE", "DECOMPOSER_INVOKED", "COMPLETED"]


@pytest.mark.parametrize(
    "prompt_version",
    [ATOMIC_DECOMPOSITION_PROMPT_V3, None],
    ids=["explicit-v3-prompt", "prompt-derived-from-v3-schema"],
)
def test_v3_real_canary_entrypoint_reaches_intercepted_gateway_without_provider_calls(
    prompt_version: str | None,
) -> None:
    valid_output = {
        "schema_version": ATOMIC_PROPOSITION_SCHEMA_V3,
        "decomposition_status": "unresolved_decomposition",
        "propositions": [],
        "ambiguity_reason": "synthetic offline preflight fixture",
    }

    result, transport, body_seen, gateway = _run_real_gateway_canary(
        valid_output,
        schema_version=ATOMIC_PROPOSITION_SCHEMA_V3,
        prompt_version=prompt_version,
    )

    assert result.status == "COMPLETED"
    assert result.decomposition.schema_version == ATOMIC_PROPOSITION_SCHEMA_V3
    assert result.decomposition.prompt_version == ATOMIC_DECOMPOSITION_PROMPT_V3
    assert result.decomposition.status.value == "UNRESOLVED_DECOMPOSITION"
    assert result.decomposition.validator_result["composition_validation_status"] == "VALID"
    assert transport.outbound_request_count == 1  # intercepted by httpx.MockTransport
    assert len(body_seen) == 1
    assert gateway.gateway_call_count == 1
    assert body_seen[0]["response_format"] == {"type": "json_object"}
    assert "assertion_composition" in json.dumps(body_seen[0]["messages"], ensure_ascii=False)
    assert result.safe_dict()["contract_resolution"]["contract_resolution_status"] == "RESOLVED"


def test_v3_safe_artifact_persists_bounded_source_span_trace() -> None:
    claim = "Alpha signal rose or Beta signal fell."
    alpha = "Alpha signal rose"
    beta = "Beta signal fell"
    structured = {
        "schema_version": ATOMIC_PROPOSITION_SCHEMA_V3,
        "decomposition_status": "decomposed",
        "propositions": [
            {
                "local_ref": "p0",
                "proposition_text": alpha,
                "subject": "Alpha",
                "predicate": "rose",
                "source_span": alpha,
                "typed_value": {"kind": "text", "text": "rose"},
                "polarity": "asserted",
            },
            {
                "local_ref": "p1",
                "proposition_text": beta,
                "subject": "Beta",
                "predicate": "fell",
                "source_span": beta,
                "typed_value": {"kind": "text", "text": "fell"},
                "polarity": "asserted",
            },
        ],
        "assertion_composition": {
            "root_expressions": [{"member_type": "group", "ref": "g0"}],
            "groups": [
                {
                    "local_ref": "g0",
                    "operator": "OR",
                    "source_span": claim[:-1],
                    "members": [
                        {"member_type": "proposition", "ref": "p0", "source_span": alpha},
                        {"member_type": "proposition", "ref": "p1", "source_span": beta},
                    ],
                }
            ],
        },
    }

    result, _, _, _ = _run_real_gateway_canary(
        structured,
        claim_id="10000000-0000-4000-8000-000000000001",
        claim_text=claim,
        schema_version=ATOMIC_PROPOSITION_SCHEMA_V3,
        prompt_version=ATOMIC_DECOMPOSITION_PROMPT_V3,
    )

    serialized = json.loads(result.safe_json())
    decomposition = serialized["decomposition"]
    safe_model = decomposition["model_structured_result"]
    safe_props = safe_model["propositions"]
    assert result.decomposition.validator_result["validation_issue_codes"] == [], (
        result.decomposition.validator_result["validation_issue_codes"]
    )
    assert result.decomposition.status.value == "RESOLVED"
    assert [(item["local_ref"], item["source_span"]) for item in safe_props] == [
        ("p0", alpha),
        ("p1", beta),
    ]
    trace = decomposition["validator_result"]["composition_source_span_trace"]
    assert len(trace) <= 256
    assert decomposition["validator_result"]["composition_source_span_trace_truncated"] is False
    assert any(
        item["parent_ref"] == "g0"
        and item["child_ref"] == "p0"
        and item["parent_source_span"] == claim[:-1]
        and item["child_source_span"] == alpha
        and item["compatibility_result"] == "PASS"
        for item in trace
    )
    assert all(
        item["resolved_proposition_id"]
        for item in trace
        if item["child_ref"] in {"p0", "p1"}
        and item["comparison_rule"]
        == "MEMBER_AND_RESOLVED_PROPOSITION_SPANS_CONTAINMENT_COMPATIBLE"
    )


def test_runner_without_injected_decomposer_fails_closed() -> None:
    result = asyncio.run(run_one_claim_canary(_request(), decomposer=None))
    assert result.status == "BLOCKED"
    assert result.reason == "decomposer_unavailable"


def _gateway_completion(content: str, *, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "id": "synthetic-request",
            "model": "deepseek-flash",
            "choices": [
                {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        },
    )


def _run_real_gateway_canary(
    valid_output: dict[str, Any],
    *,
    claim_id: str = "synthetic-claim",
    claim_text: str = "Synthetic: café 样本。",
    schema_version: str = "atomic-proposition-schema-v2",
    prompt_version: str | None = None,
) -> tuple[Any, Any, list[dict[str, Any]], CanaryGateway]:
    body_seen: list[dict[str, Any]] = []

    async def responder(request: httpx.Request) -> httpx.Response:
        body_seen.append(json.loads(await request.aread()))
        return _gateway_completion(json.dumps(valid_output, ensure_ascii=False))

    client, transport = build_canary_http_client(
        timeout=httpx.Timeout(19.0, connect=4.0, read=13.0, write=6.0, pool=2.0),
        delegate_transport=httpx.MockTransport(responder),
    )
    gateway = CanaryGateway(LLMGateway(client), transport_diagnostics=transport)
    decomposer = AtomicPropositionDecomposer(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://api.deepseek.com/v1",
        api_key=SecretStr("synthetic-secret-never-sent-to-a-real-provider"),
        model="deepseek-flash",
        provider_profile_id="synthetic-profile",
        max_calls=1,
    )
    canary_decomposer = CanaryDecomposer(decomposer, gateway)
    request = _request(
        claim_id=claim_id,
        claim_text=claim_text,
        schema_version=schema_version,
        prompt_version=prompt_version,
    )
    try:
        result = asyncio.run(run_one_claim_canary(request, decomposer=canary_decomposer))
        # Exercise the exact result serializer used for safe output.
        json.loads(result.safe_json())
    finally:
        asyncio.run(client.aclose())
    return result, transport, body_seen, gateway


def test_real_gateway_path_sends_structured_json_and_only_one_outbound_request() -> None:
    valid_output = {
        "schema_version": "atomic-proposition-schema-v2",
        "decomposition_status": "single_proposition",
        "propositions": [
            {
                "proposition_text": "The device detects defects.",
                "subject": "The device",
                "predicate": "detects",
                "typed_value": {"kind": "text", "text": "defects"},
                "polarity": "asserted",
            }
        ],
    }

    result, transport, body_seen, gateway = _run_real_gateway_canary(valid_output)

    assert result.status == "COMPLETED"
    assert _safe_decomposition(result)["model_call_count"] == 1
    assert _safe_decomposition(result)["proposition_count"] == 1
    assert len(result.decomposition.propositions) == 1
    assert result.decomposition.parser_result["proposition_count"] == 1
    assert result.decomposition.validator_result["input_proposition_count"] == 1
    assert transport.outbound_request_count == 1
    assert len(body_seen) == 1
    assert gateway.gateway_call_count == 1
    assert gateway.regeneration_options == [False]
    assert body_seen[0]["response_format"] == {"type": "json_object"}
    assert body_seen[0]["model"] == "deepseek-flash"
    safe_json = result.safe_json()
    serialized = json.loads(safe_json)
    assert serialized["outbound_attempted"] is True
    assert serialized["outbound_count"] == 1
    assert serialized["response_received"] is True
    assert serialized["claim"]["claim_text"] == "Synthetic: café 样本。"
    assert "CANARY_RESULT_BUILD" in serialized["stage_markers"]
    assert "CANARY_RESULT_SERIALIZE" in serialized["stage_markers"]
    assert "DECOMPOSER_PARSE" in serialized["stage_markers"]
    assert "DECOMPOSER_VALIDATE" in serialized["stage_markers"]
    for stage in ("GATEWAY_REQUEST_BUILD", "TRANSPORT_ENTER", "TRANSPORT_EXIT", "GATEWAY_RESPONSE"):
        assert stage in serialized["stage_markers"]
    diagnostics = result.safe_dict()["transport_diagnostics"]
    assert diagnostics == {
        "request_scheme": "https",
        "request_host": "api.deepseek.com",
        "request_port": 443,
        "request_path_template": "/v1/chat/completions",
        "proxy_mode": "DIRECT",
        "trust_env": False,
        "connect_timeout": 4.0,
        "read_timeout": 13.0,
        "write_timeout": 6.0,
        "pool_timeout": 2.0,
        "transport_retry_count": 0,
        "exception_chain_categories": [],
        "outbound_attempted": True,
        "outbound_count": 1,
        "response_received": True,
        "stage_markers": [
            "GATEWAY_REQUEST_BUILD",
            "TRANSPORT_ENTER",
            "TRANSPORT_EXIT",
            "GATEWAY_RESPONSE",
        ],
    }


def test_rejected_result_is_persisted_with_field_diagnostics_and_redacted_secrets(
    tmp_path: Any,
) -> None:
    invalid_output = {
        "schema_version": "atomic-proposition-schema-v2",
        "decomposition_status": "single_proposition",
        "propositions": [
            {
                "proposition_text": "Authorization: Bearer fake-auth api_key=fake-key",
                "predicate": "describes",
                "typed_value": {"kind": "entity", "text": "a synthetic object"},
                "polarity": "asserted",
                "X-Secret-Header": "fake-header",
            }
        ],
    }
    claim = (
        "Synthetic claim; Authorization: Bearer fake-auth api_key=fake-key "
        "credential=fake-credential X-Secret-Header: fake-header"
    )

    result, _, _, _ = _run_real_gateway_canary(invalid_output, claim_text=claim)

    artifacts = list(tmp_path.glob("v2_1r_semantic_canary_*.json"))
    assert len(artifacts) == 1
    serialized = artifacts[0].read_text(encoding="utf-8")
    persisted = json.loads(serialized)
    decomposition = persisted["decomposition"]
    assert result.status == "COMPLETED"
    assert decomposition["status"] == "SCHEMA_INVALID"
    assert decomposition["pre_validation_proposition_count"] == 1
    assert decomposition["model_structured_result"]["model_returned_proposition_count"] == 1
    assert decomposition["validator_result"]["validation_diagnostics"][0]["field_path"] == (
        "propositions[0].subject"
    )
    assert persisted["payload_fidelity_status"] == "PASS"
    assert persisted["outbound_count"] == 1
    for secret in ("fake-auth", "fake-key", "fake-credential", "fake-header"):
        assert secret not in serialized


def _matrix_proposition(
    text: str = "A synthetic fact.",
    *,
    subject: str | None = "subject",
    predicate: str | None = "describes",
    typed_value: dict[str, Any] | None = None,
    polarity: str = "asserted",
    qualifiers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "proposition_text": text,
        "predicate": predicate,
        "typed_value": typed_value or {"kind": "entity", "text": "an object"},
        "polarity": polarity,
    }
    if subject is not None:
        value["subject"] = subject
    if qualifiers is not None:
        value["material_qualifiers"] = qualifiers
    return value


@pytest.mark.parametrize(
    ("case", "response", "claim", "final_status", "issue_code", "stage", "field_path"),
    [
        (
            "A_valid_single",
            {"decomposition_status": "single_proposition", "propositions": [_matrix_proposition()]},
            "A synthetic fact.",
            "RESOLVED",
            None,
            None,
            None,
        ),
        (
            "B_valid_multiple",
            {
                "decomposition_status": "decomposed",
                "propositions": [_matrix_proposition("Fact A."), _matrix_proposition("Fact B.")],
            },
            "Two synthetic facts.",
            "RESOLVED",
            None,
            None,
            None,
        ),
        (
            "C_missing_subject",
            {
                "decomposition_status": "single_proposition",
                "propositions": [_matrix_proposition(subject=None)],
            },
            "A synthetic fact.",
            "SCHEMA_INVALID",
            "missing",
            "PROPOSITION_SCHEMA",
            "propositions[0].subject",
        ),
        (
            "D_missing_predicate",
            {
                "decomposition_status": "single_proposition",
                "propositions": [
                    {
                        "proposition_text": "A synthetic fact.",
                        "subject": "subject",
                        "typed_value": {"kind": "entity", "text": "an object"},
                        "polarity": "asserted",
                    }
                ],
            },
            "A synthetic fact.",
            "SCHEMA_INVALID",
            "missing",
            "PROPOSITION_SCHEMA",
            "propositions[0].predicate",
        ),
        (
            "E_invalid_typed_value",
            {
                "decomposition_status": "single_proposition",
                "propositions": [
                    _matrix_proposition(
                        typed_value={
                            "kind": "quantity",
                            "numeric": {"operator": "range", "lower_bound": 9, "upper_bound": 2},
                        }
                    )
                ],
            },
            "A synthetic range.",
            "SCHEMA_INVALID",
            "value_error",
            "TYPED_VALUE_SHAPE",
            "propositions[0].typed_value.numeric",
        ),
        (
            "F_unsupported_polarity",
            {
                "decomposition_status": "single_proposition",
                "propositions": [_matrix_proposition(polarity="maybe")],
            },
            "A synthetic fact.",
            "SCHEMA_INVALID",
            "enum",
            "POLARITY",
            "propositions[0].polarity",
        ),
        (
            "G_malformed_qualifier",
            {
                "decomposition_status": "single_proposition",
                "propositions": [
                    _matrix_proposition(
                        qualifiers=[
                            {
                                "category": "unsupported",
                                "key": "x",
                                "value": "y",
                                "materiality": "material",
                            }
                        ]
                    )
                ],
            },
            "A synthetic fact.",
            "SCHEMA_INVALID",
            "enum",
            "QUALIFIER_SCHEMA",
            "propositions[0].material_qualifiers[0].category",
        ),
        (
            "H_duplicate",
            {
                "decomposition_status": "decomposed",
                "propositions": [_matrix_proposition("Fact A."), _matrix_proposition(" fact a. ")],
            },
            "Fact A.",
            "STRUCTURED_OUTPUT_INVALID",
            "duplicate_proposition",
            "DUPLICATE_CHECK",
            "propositions[1].proposition_text",
        ),
        (
            "I_qualifier_loss",
            {
                "decomposition_status": "single_proposition",
                "propositions": [_matrix_proposition("The rate is 4%.")],
            },
            "The rate may be 4%.",
            "UNRESOLVED_DECOMPOSITION",
            "potential_modality_loss",
            "QUALIFIER_PRESERVATION",
            "claim.qualifier.modality",
        ),
        (
            "J_decomposed_empty",
            {"decomposition_status": "decomposed", "propositions": []},
            "A synthetic fact.",
            "STRUCTURED_OUTPUT_INVALID",
            "invalid_response_shape",
            "TOP_LEVEL_SHAPE",
            "decomposition_status/propositions/ambiguity_reason",
        ),
        (
            "K_unresolved_empty",
            {
                "decomposition_status": "unresolved_decomposition",
                "propositions": [],
                "ambiguity_reason": "Synthetic ambiguity.",
            },
            "An ambiguous synthetic fact.",
            "UNRESOLVED_DECOMPOSITION",
            None,
            None,
            None,
        ),
        (
            "L_unexpected_field",
            {
                "decomposition_status": "single_proposition",
                "propositions": [_matrix_proposition()],
                "unexpected_field": "ignored by diagnostic whitelist",
            },
            "A synthetic fact.",
            "SCHEMA_INVALID",
            "extra_forbidden",
            "TOP_LEVEL_SCHEMA",
            "<unrecognized_field>",
        ),
    ],
)
def test_fake_gateway_matrix_has_field_diagnostics_without_semantic_changes(
    case: str,
    response: dict[str, Any],
    claim: str,
    final_status: str,
    issue_code: str | None,
    stage: str | None,
    field_path: str | None,
) -> None:
    response = {"schema_version": "atomic-proposition-schema-v2", **response}
    result, transport, _, _ = _run_real_gateway_canary(response, claim_text=claim)

    decomposition = json.loads(result.safe_json())["decomposition"]
    assert result.status == "COMPLETED", case
    assert decomposition["status"] == final_status, case
    assert decomposition["parser_result"]["status"] == "PARSED", case
    assert decomposition["validator_result"]["status"] == (
        "VALID" if issue_code is None or issue_code.startswith("potential_") else "REJECTED"
    ), case
    assert transport.outbound_request_count == 1, case
    if issue_code is not None:
        diagnostic = next(
            item
            for item in decomposition["validator_result"]["validation_diagnostics"]
            if item["issue_code"] == issue_code
        )
        assert diagnostic["validation_stage"] == stage, case
        assert diagnostic["field_path"] == field_path, case
        assert decomposition["pre_validation_proposition_count"] == len(response["propositions"]), (
            case
        )
        assert decomposition["model_structured_result"] is not None, case


def test_exact_entrypoint_serializes_multiple_and_unresolved_fake_results() -> None:
    proposition = {
        "proposition_text": "设备检测缺陷。",
        "subject": "设备",
        "predicate": "检测",
        "typed_value": {"kind": "text", "text": "缺陷"},
        "polarity": "asserted",
    }
    multiple, multiple_transport, _, _ = _run_real_gateway_canary(
        {
            "schema_version": "atomic-proposition-schema-v2",
            "decomposition_status": "decomposed",
            "propositions": [proposition, {**proposition, "proposition_text": "设备识别缺陷。"}],
        }
    )
    assert multiple.status == "COMPLETED"
    assert len(multiple.decomposition.propositions) == 2
    assert json.loads(multiple.safe_json())["decomposition"]["proposition_count"] == 2
    assert multiple_transport.outbound_request_count == 1

    unresolved, unresolved_transport, _, _ = _run_real_gateway_canary(
        {
            "schema_version": "atomic-proposition-schema-v2",
            "decomposition_status": "unresolved_decomposition",
            "propositions": [],
            "ambiguity_reason": "Synthetic wording leaves attachment unclear.",
        }
    )
    assert unresolved.status == "COMPLETED"
    assert unresolved.decomposition.status.value == "UNRESOLVED_DECOMPOSITION"
    assert json.loads(unresolved.safe_json())["decomposition"]["proposition_count"] == 0
    assert unresolved_transport.outbound_request_count == 1


def test_canary_gateway_classifies_error_without_exposing_response_or_credentials() -> None:
    response_body = "private provider body with synthetic-secret and Authorization"

    async def responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=response_body)

    client, transport = build_canary_http_client(
        timeout=httpx.Timeout(5.0),
        delegate_transport=httpx.MockTransport(responder),
    )
    gateway = CanaryGateway(LLMGateway(client), transport_diagnostics=transport)
    decomposer = AtomicPropositionDecomposer(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://api.deepseek.com/v1",
        api_key=SecretStr("synthetic-secret"),
        model="deepseek-flash",
        provider_profile_id="synthetic-profile",
        max_calls=1,
    )
    canary_decomposer = CanaryDecomposer(decomposer, gateway)

    try:
        result = asyncio.run(run_one_claim_canary(_request(), decomposer=canary_decomposer))
    finally:
        asyncio.run(client.aclose())

    safe_result = json.dumps(result.safe_dict())
    assert result.status == "COMPLETED"
    assert result.safe_dict()["gateway_failure"] == {
        "classification": "AUTH",
        "code": "MODEL_AUTHENTICATION_FAILED",
        "detail_code": None,
    }
    safe = result.safe_dict()
    assert safe["outbound_attempted"] is True
    assert safe["outbound_count"] == 1
    assert safe["response_received"] is True
    assert safe["failure_stage"] == "GATEWAY_RESPONSE"
    assert safe["safe_exception_type"] == "ModelGatewayError"
    assert "synthetic-secret" not in safe_result
    assert "private provider body" not in safe_result
    assert "Authorization" not in safe_result


def test_gateway_path_fallback_cannot_exceed_one_outbound_canary_request() -> None:
    delegated_requests: list[httpx.Request] = []

    async def timeout_once(request: httpx.Request) -> httpx.Response:
        delegated_requests.append(request)
        raise httpx.ConnectTimeout("synthetic timeout", request=request)

    client, transport = build_canary_http_client(
        timeout=httpx.Timeout(5.0),
        delegate_transport=httpx.MockTransport(timeout_once),
    )
    gateway = CanaryGateway(LLMGateway(client), transport_diagnostics=transport)
    decomposer = AtomicPropositionDecomposer(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://api.deepseek.com",
        api_key=SecretStr("synthetic-secret"),
        model="deepseek-flash",
        provider_profile_id="synthetic-profile",
        max_calls=1,
    )
    canary_decomposer = CanaryDecomposer(decomposer, gateway)

    try:
        result = asyncio.run(run_one_claim_canary(_request(), decomposer=canary_decomposer))
    finally:
        asyncio.run(client.aclose())

    assert _safe_decomposition(result)["model_call_count"] == 1
    assert len(delegated_requests) == 1
    assert transport.outbound_request_count == 1
    safe = json.loads(result.safe_json())
    assert safe["failure_stage"] == "TRANSPORT_ENTER"
    assert safe["safe_exception_type"] == "ModelGatewayError"
    assert safe["outbound_attempted"] is True
    assert safe["outbound_count"] == 1
    assert safe["response_received"] is False
    assert "synthetic timeout" not in json.dumps(safe)


def test_transport_diagnostics_classify_safe_exception_chain_and_exclude_secrets() -> None:
    delegated_count = 0

    async def dns_failure(request: httpx.Request) -> httpx.Response:
        nonlocal delegated_count
        delegated_count += 1
        try:
            raise socket.gaierror(11001, "sensitive DNS diagnostic")
        except socket.gaierror as cause:
            raise httpx.ConnectError(
                "sensitive transport detail and synthetic-key", request=request
            ) from cause

    timeout = httpx.Timeout(21.0, connect=5.0, read=14.0, write=8.0, pool=3.0)
    client, transport = build_canary_http_client(
        timeout=timeout,
        proxy="http://user:secret@proxy.invalid:7897",
        delegate_transport=httpx.MockTransport(dns_failure),
    )
    gateway = CanaryGateway(LLMGateway(client), transport_diagnostics=transport)
    decomposer = AtomicPropositionDecomposer(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://api.deepseek.com/v1",
        api_key=SecretStr("synthetic-secret"),
        model="deepseek-flash",
        provider_profile_id="synthetic-profile",
        max_calls=1,
    )
    canary_decomposer = CanaryDecomposer(decomposer, gateway)

    try:
        result = asyncio.run(run_one_claim_canary(_request(), decomposer=canary_decomposer))
    finally:
        asyncio.run(client.aclose())

    safe_json = result.safe_json()
    diagnostics = result.safe_dict()["transport_diagnostics"]
    assert result.status == "COMPLETED"
    assert diagnostics["proxy_mode"] == "PROXY"
    assert diagnostics["transport_retry_count"] == 0
    assert diagnostics["exception_chain_categories"] == [
        "UNKNOWN_CONNECT_ERROR",
        "DNS_ERROR",
    ]
    assert transport.outbound_request_count == 1
    assert delegated_count == 1
    for secret in ("synthetic-secret", "synthetic-key", "sensitive", "user", "proxy.invalid"):
        assert secret not in safe_json
