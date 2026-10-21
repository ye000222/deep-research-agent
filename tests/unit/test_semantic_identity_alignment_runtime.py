from __future__ import annotations

from typing import Any

import pytest
from app.domain.providers import AdapterType, CanonicalModelResult, TokenUsage, UsageAccuracy
from app.domain.semantic_identity_alignment import SemanticAlignmentStatus
from app.domain.source_atomic_assertion import SourceGroundingState
from app.services.semantic_identity_alignment_runtime import (
    AlignmentRuntimeStatus,
    IdentityAlignmentPair,
    SemanticIdentityAlignmentRuntime,
    deterministic_pair_ref,
    parse_alignment_response,
)
from pydantic import SecretStr

from tests.unit.test_proposition_evidence_direction import _scope, _source_assertion


class FakeAlignmentGateway:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    async def generate_structured(self, **kwargs: Any) -> CanonicalModelResult:
        self.calls.append(kwargs)
        return CanonicalModelResult(
            parsed_object=self.payload,
            usage=TokenUsage(
                input_tokens=3,
                output_tokens=4,
                total_tokens=7,
                accuracy=UsageAccuracy.EXACT,
            ),
        )


def _run(coroutine: Any) -> Any:
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("Fake alignment gateway unexpectedly suspended")


def _pair(*, grounding: SourceGroundingState = SourceGroundingState.VALID) -> IdentityAlignmentPair:
    scope = _scope((_source_assertion(grounding=grounding),))
    target = scope.targets[0]
    source = scope.source_assertions[0]
    return IdentityAlignmentPair(
        target=target.proposition,
        source=source,
        target_proposition_ref=str(target.proposition.proposition_id),
        source_grounding_refs=("evidence-span-1",),
    )


def _response(
    pair: IdentityAlignmentPair, *, subject: str = "SAME", predicate: str = "SAME"
) -> dict[str, Any]:
    return {
        "schema_version": "semantic-identity-alignment-response-v1",
        "status": "RESOLVED",
        "alignments": [
            {
                "pair_ref": deterministic_pair_ref(
                    pair.target_proposition_ref,
                    pair.source.assertion_ref,
                    "semantic-identity-alignment-response-v1",
                ),
                "subject_relation": subject,
                "predicate_relation": predicate,
                "reason_code": "IDENTITY_ALIGNED"
                if subject == "SAME" and predicate == "SAME"
                else "IDENTITY_DISTINCT"
                if subject == "DIFFERENT" or predicate == "DIFFERENT"
                else "AMBIGUOUS",
            }
        ],
    }


def _runtime(gateway: FakeAlignmentGateway) -> SemanticIdentityAlignmentRuntime:
    return SemanticIdentityAlignmentRuntime(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://example.invalid/v1",
        api_key=SecretStr("fake-only"),
        model="fake-model",
    )


def test_structured_runtime_parses_grounded_same_and_uses_one_call() -> None:
    pair = _pair()
    gateway = FakeAlignmentGateway(_response(pair))
    result = _run(_runtime(gateway).align((pair,), {}))
    assert result.status is AlignmentRuntimeStatus.COMPLETED
    assert result.model_call_count == len(gateway.calls) == 1
    assert [item.alignment_status for item in result.facts] == [
        SemanticAlignmentStatus.SAME,
        SemanticAlignmentStatus.SAME,
    ]
    assert gateway.calls[0]["allow_regeneration"] is False
    assert gateway.calls[0]["request"].response_contract["properties"]["alignments"]


def test_ungrounded_source_demotes_provider_same_to_unknown() -> None:
    pair = _pair(grounding=SourceGroundingState.INVALID)
    facts = parse_alignment_response(_response(pair), (pair,))
    assert facts
    assert all(item.alignment_status is SemanticAlignmentStatus.UNKNOWN for item in facts)
    assert all(not item.source_grounding_refs for item in facts)


def test_same_wording_does_not_substitute_for_unknown_alignment() -> None:
    pair = _pair()
    payload = {
        "schema_version": "semantic-identity-alignment-response-v1",
        "status": "UNRESOLVED",
        "alignments": [],
        "unresolved_reason_codes": ["AMBIGUOUS"],
    }
    facts = parse_alignment_response(payload, (pair,))
    assert all(item.alignment_status is SemanticAlignmentStatus.UNKNOWN for item in facts)


def test_different_surface_wording_can_be_structurally_aligned_without_string_matching() -> None:
    pair = _pair()
    pair = pair.model_copy(
        update={"target": pair.target.model_copy(update={"subject": "the organization"})}
    )
    facts = parse_alignment_response(_response(pair), (pair,))
    subject = next(item for item in facts if item.field_role.value == "SUBJECT")
    assert pair.target.subject != pair.source.subject.source_representation
    assert subject.alignment_status is SemanticAlignmentStatus.SAME


def test_wrong_pair_reference_is_rejected() -> None:
    pair = _pair()
    payload = _response(pair)
    payload["alignments"][0]["pair_ref"] = "f" * 64
    with pytest.raises(ValueError, match="ALIGNMENT_PAIR_SET_MISMATCH"):
        parse_alignment_response(payload, (pair,))


def test_duplicate_pair_response_is_rejected() -> None:
    pair = _pair()
    payload = _response(pair)
    payload["alignments"].append(payload["alignments"][0])
    with pytest.raises(ValueError, match="ALIGNMENT_PAIR_SET_MISMATCH"):
        parse_alignment_response(payload, (pair,))


def test_runtime_second_call_is_blocked_before_gateway() -> None:
    pair = _pair()
    gateway = FakeAlignmentGateway(_response(pair))
    runtime = _runtime(gateway)
    _run(runtime.align((pair,), {}))
    second = _run(runtime.align((pair,), {}))
    assert len(gateway.calls) == 1
    assert second.status is AlignmentRuntimeStatus.NOT_PROCESSED_BUDGET
