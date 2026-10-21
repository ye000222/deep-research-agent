from __future__ import annotations

from decimal import Decimal
from itertools import combinations
from typing import Any
from uuid import UUID, uuid4

import pytest
from app.domain.atomic_proposition import ClaimPropositionLink, NumericValue, PropositionValue
from app.domain.comparability_evaluator import ComparabilityEvaluator
from app.domain.proposition_pair_identity import (
    PAIR_IDENTITY_SCHEMA,
    PairFieldRole,
    PropositionPairInput,
    PropositionPairTarget,
    bind_pair,
    project_pair_views,
)
from app.domain.proposition_support_group import PropositionAssertionBinding, group_propositions
from app.domain.relation_evaluator import RelationEvaluator
from app.domain.verification_assertion import VerificationAssertion
from app.services import proposition_pair_identity_runtime as runtime_module
from app.services.proposition_pair_identity_runtime import (
    PropositionPairSemanticAlignmentRuntime,
    parse_pair_response,
    plan_proposition_pairs,
    request_payload,
    validate_pair_product,
)
from pydantic import SecretStr

from tests.unit.test_semantic_identity_alignment_runtime import FakeAlignmentGateway
from tests.unit.test_semantic_research_shadow import _run_immediate
from tests.unit.test_v22_a1_verification_assertion_contract import _proposition


def _runtime(gateway: Any, max_request_bytes: int = 100_000) -> Any:
    return PropositionPairSemanticAlignmentRuntime(
        gateway,
        adapter_type="openai_compatible_chat",
        base_url="https://example.invalid",
        api_key=SecretStr("fake-only"),
        model="fake-model",
        max_request_bytes=max_request_bytes,
    )


def target(index: int, *, value: str = "12.4", **updates: Any) -> PropositionPairTarget:
    ref = UUID(int=index)
    prop = _proposition(numeric=True).model_copy(
        update={
            "proposition_id": ref,
            "subject": f"device-{index}",
            "predicate": f"rate-{index}",
            "value": PropositionValue(
                kind="quantity",
                numeric=NumericValue(
                    operator="exact",
                    value=Decimal(value),
                    unit="%",
                ),
            ),
        }
    )
    assertion = VerificationAssertion(
        assertion_ref=str(ref),
        source_proposition_id=ref,
        subject=prop.subject,
        predicate=prop.predicate,
        typed_value=prop.value,
        polarity=prop.polarity,
        grounding_usability="SUFFICIENT",
        composition_eligibility="ELIGIBLE",
    )
    return PropositionPairTarget(
        proposition=prop,
        assertion=assertion,
        claim_refs=(UUID(int=100 + index),),
        formation_ref=f"formation-{index}",
        formation_status="FORMATION_SUFFICIENT",
    ).model_copy(update=updates)


def pair() -> PropositionPairInput:
    left, right = target(1), target(2)
    return PropositionPairInput(binding=bind_pair(uuid4(), left, right), left=left, right=right)


def response(pairs: tuple[PropositionPairInput, ...], state: str = "SAME") -> dict[str, Any]:
    reason = {"SAME": "IDENTITY_ALIGNED", "DIFFERENT": "IDENTITY_DISTINCT", "UNKNOWN": "AMBIGUOUS"}
    return {
        "schema_version": PAIR_IDENTITY_SCHEMA,
        "run_ref": str(pairs[0].binding.run_ref),
        "judgments": [
            {
                "pair_ref": item.binding.pair_ref,
                "left_proposition_ref": str(item.binding.left_proposition_ref),
                "right_proposition_ref": str(item.binding.right_proposition_ref),
                "field_role": role.value,
                "alignment_status": state,
                "reason_code": reason[state],
            }
            for item in pairs
            for role in PairFieldRole
        ],
    }


def relation(item: PropositionPairInput, state: str = "SAME") -> Any:
    facts = parse_pair_response(response((item,), state), (item,))
    views = project_pair_views(item, facts)
    return RelationEvaluator.evaluate(*views, ComparabilityEvaluator.evaluate(*views))


def group_count(item: PropositionPairInput, assessment: Any) -> int:
    return len(
        group_propositions(
            tuple(
                ClaimPropositionLink(
                    claim_id=side.claim_refs[0],
                    proposition_id=side.proposition.proposition_id,
                    proposition_index=0,
                    decomposition_status=side.proposition.decomposition.status,
                )
                for side in (item.left, item.right)
            ),
            (),
            tuple(
                PropositionAssertionBinding(
                    proposition_ref=side.proposition.proposition_id,
                    assertion_ref=side.assertion.assertion_ref,
                )
                for side in (item.left, item.right)
            ),
            (assessment,),
        ).groups
    )


def test_different_wording_uses_structured_facts_and_does_not_mutate_targets() -> None:
    item = pair()
    before = item.model_dump(mode="json")
    result = relation(item)
    assert result.relation.value == "same"
    assert item.left.proposition.subject != item.right.proposition.subject
    assert item.left.proposition.predicate != item.right.proposition.predicate
    assert item.model_dump(mode="json") == before
    assert item.left.assertion.subject_identity.status.value == "UNRESOLVED"
    assert result.left_assertion_ref == item.left.assertion.assertion_ref


@pytest.mark.parametrize(
    "state,expected", [("DIFFERENT", "NOT_COMPARABLE"), ("UNKNOWN", "UNKNOWN")]
)
def test_non_same_field_alignment_fails_closed(state: str, expected: str) -> None:
    result = relation(pair(), state)
    assert result.comparability.value == expected
    assert result.relation.value == "unknown"


def test_values_remain_a3_authority() -> None:
    left, right = target(1), target(2, value="13.0")
    item = PropositionPairInput(binding=bind_pair(uuid4(), left, right), left=left, right=right)
    result = relation(item)
    assert result.relation.value == "unknown"
    assert group_count(item, result) == 2


@pytest.mark.parametrize("field", ["attribution", "modality"])
def test_identity_cannot_override_context_fields(field: str) -> None:
    item = pair()
    right = item.right.model_copy(
        update={
            "assertion": item.right.assertion.model_copy(update={field: "one-sided-context"}),
            "proposition": item.right.proposition.model_copy(update={field: "one-sided-context"}),
        }
    )
    item = PropositionPairInput(
        binding=bind_pair(item.binding.run_ref, item.left, right), left=item.left, right=right
    )
    assert relation(item).relation.value == "unknown"


def test_pair_namespace_never_leaks_to_another_pair_or_source_pair() -> None:
    ab = pair()
    ac = PropositionPairInput(
        binding=bind_pair(ab.binding.run_ref, ab.left, target(3)), left=ab.left, right=target(3)
    )
    ab_views = project_pair_views(ab, parse_pair_response(response((ab,)), (ab,)))
    ac_views = project_pair_views(ac, parse_pair_response(response((ac,)), (ac,)))
    assert ab_views[0].subject_identity.namespace != ac_views[0].subject_identity.namespace
    assert "PROPOSITION_PROPOSITION" in ab_views[0].subject_identity.namespace
    assert (
        ComparabilityEvaluator.evaluate(ab_views[0], ac_views[1]).comparability.value == "UNKNOWN"
    )
    with pytest.raises(ValueError, match="BINDING"):
        project_pair_views(ac, parse_pair_response(response((ab,)), (ab,)))


@pytest.mark.parametrize(
    "fault",
    [
        "run",
        "pair",
        "left",
        "right",
        "role",
        "missing",
        "duplicate",
        "conflict",
        "relation",
        "version",
    ],
)
def test_parser_rejects_incomplete_wrong_or_conflicting_response(fault: str) -> None:
    item = pair()
    raw = response((item,))
    judgment = raw["judgments"][0]
    if fault == "run":
        raw["run_ref"] = str(uuid4())
    elif fault == "pair":
        judgment["pair_ref"] = "f" * 64
    elif fault in {"left", "right"}:
        judgment[f"{fault}_proposition_ref"] = str(uuid4())
    elif fault == "role":
        judgment["field_role"] = "VALUE"
    elif fault == "missing":
        raw["judgments"].pop()
    elif fault in {"duplicate", "conflict"}:
        raw["judgments"].append(
            {**judgment, **({"alignment_status": "UNKNOWN"} if fault == "conflict" else {})}
        )
    elif fault == "relation":
        raw["proposition_relation"] = "SAME"
    else:
        raw["schema_version"] = "semantic-identity-alignment-response-v1"
    with pytest.raises(ValueError):
        parse_pair_response(raw, (item,))


def test_candidate_enumeration_is_bounded_and_order_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    count = 0

    def counted(items: Any, size: int) -> Any:
        nonlocal count
        for entry in combinations(items, size):
            count += 1
            yield entry

    monkeypatch.setattr(runtime_module, "combinations", counted)
    targets = tuple(target(index) for index in range(1, 101))
    run = uuid4()
    first = plan_proposition_pairs(
        run, targets, max_pairs=3, batch_size=2, max_request_bytes=100_000
    )
    assert count == first.enumerated_pair_count == len(first.pairs) == 3
    second = plan_proposition_pairs(
        run, tuple(reversed(targets)), max_pairs=3, batch_size=2, max_request_bytes=100_000
    )
    assert first == second
    assert [len(batch) for batch in first.batches] == [2, 1]


@pytest.mark.parametrize(
    "status", ["FORMATION_UNKNOWN", "FORMATION_NOT_PRESERVED", "NOT_PROCESSED"]
)
def test_formation_gate_prevents_affirmative_pair(status: str) -> None:
    plan = plan_proposition_pairs(
        uuid4(),
        (target(1), target(2, formation_status=status)),
        max_pairs=8,
        batch_size=4,
        max_request_bytes=100_000,
    )
    assert not plan.pairs
    with pytest.raises(ValueError):
        bind_pair(uuid4(), target(1), target(2, formation_status=status))


def test_request_size_and_composition_gate() -> None:
    first, second = target(1), target(2)
    plan = plan_proposition_pairs(
        uuid4(), (first, second), max_pairs=4, batch_size=2, max_request_bytes=1
    )
    assert not plan.pairs and len(plan.oversized_pair_refs) == 1
    second = second.model_copy(
        update={
            "assertion": second.assertion.model_copy(
                update={"composition_eligibility": "INELIGIBLE"}
            )
        }
    )
    assert not plan_proposition_pairs(
        uuid4(), (first, second), max_pairs=4, batch_size=2, max_request_bytes=100_000
    ).pairs


def test_real_structured_runtime_and_durable_validation_one_call_only() -> None:
    item = pair()
    gateway = FakeAlignmentGateway(response((item,)))
    runtime = PropositionPairSemanticAlignmentRuntime(
        gateway,
        adapter_type="openai_compatible_chat",
        base_url="https://example.invalid",
        api_key=SecretStr("fake-only"),
        model="fake-model",
        max_request_bytes=100_000,
    )
    result = _run_immediate(runtime.align((item,)))
    assert result.status == "COMPLETED"
    validate_pair_product(result, (item,))
    assert len(gateway.calls) == result.model_call_count == 1
    assert gateway.calls[0]["allow_regeneration"] is False
    request = gateway.calls[0]["request"]
    assert request.response_contract["properties"]["judgments"]
    assert request.content_parts[0].value == request_payload((item,))
    assert not {"source", "evidence", "owner", "supports", "corroboration"} & set(
        request.content_parts[0].value["pairs"][0]
    )
    with pytest.raises(ValueError, match="CALL_BOUND"):
        _run_immediate(runtime.align((item,)))
    assert len(gateway.calls) == 1


def test_material_time_context_is_not_overridden() -> None:
    from app.domain.truth_condition_context import ContextType
    from app.domain.verification_assertion import MaterialContextProjection

    from tests.unit.test_v22_a1_verification_assertion_contract import _context

    targets = []
    for index, year in ((1, "2025"), (2, "2026")):
        item = target(index)
        context = MaterialContextProjection(
            context=_context(ContextType.TEMPORAL, value={"year": int(year)}),
            attachment_resolution="RESOLVED",
        )
        targets.append(
            item.model_copy(
                update={
                    "assertion": item.assertion.model_copy(update={"material_contexts": (context,)})
                }
            )
        )
    left, right = targets
    item = PropositionPairInput(binding=bind_pair(uuid4(), left, right), left=left, right=right)
    result = relation(item)
    assert result.comparability.value == "NOT_COMPARABLE"
    assert result.relation.value == "unknown"
    assert group_count(item, result) == 2


def test_field_judgments_are_independent() -> None:
    item = pair()
    raw = response((item,))
    raw["judgments"][1].update(alignment_status="DIFFERENT", reason_code="IDENTITY_DISTINCT")
    views = project_pair_views(item, parse_pair_response(raw, (item,)))
    assert views[0].subject_identity == views[1].subject_identity
    assert views[0].predicate_identity.key != views[1].predicate_identity.key
    assert ComparabilityEvaluator.evaluate(*views).comparability.value == "NOT_COMPARABLE"


def test_whole_request_size_guard_blocks_gateway() -> None:
    item = pair()
    gateway = FakeAlignmentGateway(response((item,)))
    result = _run_immediate(_runtime(gateway, max_request_bytes=1).align((item,)))
    assert result.status == "INVALID" and result.model_call_count == 0
    assert not gateway.calls


def test_runtime_failure_is_safe_and_never_retries() -> None:
    class TimeoutGateway:
        calls = 0

        async def generate_structured(self, **kwargs: Any) -> Any:
            self.calls += 1
            raise TimeoutError("never serialize a key or response body")

    item = pair()
    gateway = TimeoutGateway()
    runtime = _runtime(gateway)
    result = _run_immediate(runtime.align((item,)))
    assert result.status == "FAILED" and result.model_call_count == 1
    assert "serialize" not in result.model_dump_json()
    with pytest.raises(ValueError):
        _run_immediate(runtime.align((item,)))
    assert gateway.calls == 1


def test_pair_product_serialization_and_input_drift_fail_closed() -> None:
    from app.services.proposition_pair_identity_runtime import PairAlignmentRuntimeResult

    item = pair()
    result = _run_immediate(_runtime(FakeAlignmentGateway(response((item,)))).align((item,)))
    restored = PairAlignmentRuntimeResult.model_validate_json(result.model_dump_json())
    validate_pair_product(restored, (item,))
    other = pair()
    with pytest.raises(ValueError, match="INPUT_MISMATCH"):
        validate_pair_product(restored, (other,))


def test_exact_ref_and_same_claim_pairs_are_not_redispatched() -> None:
    first, second = target(1), target(2)
    second = second.model_copy(update={"claim_refs": first.claim_refs})
    assert not plan_proposition_pairs(
        uuid4(),
        (first, second),
        max_pairs=8,
        batch_size=4,
        max_request_bytes=100_000,
    ).pairs


def test_target_source_and_target_target_keys_are_separate() -> None:
    from app.services.semantic_identity_alignment_runtime import deterministic_pair_ref

    item = pair()
    assert item.binding.pair_ref != deterministic_pair_ref(
        str(item.binding.left_proposition_ref),
        str(item.binding.right_proposition_ref),
        "semantic-identity-alignment-response-v1",
    )
