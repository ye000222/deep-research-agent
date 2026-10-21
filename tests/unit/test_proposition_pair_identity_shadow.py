from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest
from app.domain.evidence_graph import claim_fingerprint
from app.domain.proposition_pair_identity import PAIR_IDENTITY_SCHEMA, PairFieldRole
from app.infrastructure.db.research_tools import PagePersistenceResult
from app.services.semantic_research_shadow import SemanticResearchShadowPipeline

from tests.unit.test_semantic_research_shadow import (
    FakeBindings,
    FakeCipher,
    FakeGateway,
    MemoryArtifactStore,
    _budget,
    _receipt,
    _run_immediate,
)


class PairGateway(FakeGateway):
    def __init__(
        self,
        state: str = "SAME",
        *,
        unknown_middle: bool = False,
        unknown_claim: UUID | None = None,
        malformed: bool = False,
    ) -> None:
        super().__init__()
        self.state = state
        self.unknown_middle = unknown_middle
        self.unknown_claim = unknown_claim
        self.malformed = malformed

    async def generate_structured(self, **kwargs: Any) -> Any:
        request = kwargs["request"]
        if request.task_kind == "proposition_pair_identity_alignment_v1":
            self.calls.append(kwargs)
            payload = request.content_parts[0].value
            judgments = []
            for index, pair in enumerate(payload["pairs"]):
                state = "UNKNOWN" if self.unknown_middle and index == 1 else self.state
                reason = {
                    "SAME": "IDENTITY_ALIGNED",
                    "UNKNOWN": "AMBIGUOUS",
                    "DIFFERENT": "IDENTITY_DISTINCT",
                }[state]
                judgments.extend(
                    {
                        "pair_ref": pair["pair_ref"],
                        "left_proposition_ref": pair["left_proposition_ref"],
                        "right_proposition_ref": pair["right_proposition_ref"],
                        "field_role": role.value,
                        "alignment_status": state,
                        "reason_code": reason,
                    }
                    for role in PairFieldRole
                )
            if self.malformed:
                judgments.pop()
            return SimpleNamespace(
                parsed_object={
                    "schema_version": PAIR_IDENTITY_SCHEMA,
                    "run_ref": payload["run_ref"],
                    "judgments": judgments,
                },
                usage=SimpleNamespace(input_tokens=7, output_tokens=11),
                latency_ms=2,
            )
        result = await super().generate_structured(**kwargs)
        payload = result.parsed_object
        if request.task_kind == "atomic_proposition_shadow_decomposition":
            text = request.content_parts[0].value["claim_text"]
            proposition = payload["propositions"][0]
            proposition["proposition_text"] = text
            proposition["source_span"] = text
            if "装置" in text:
                proposition["subject"] = "该装置"
                proposition["predicate"] = "成功比例"
        elif request.task_kind == "target_proposition_formation_validation_v1":
            if request.content_parts[0].value["claim_ref"] == str(self.unknown_claim):
                for item in payload["judgments"]:
                    item["fields"]["subject"] = "UNKNOWN"
        return result


def cohort(
    count: int = 2, *, different_wording: bool = True
) -> tuple[Any, list[PagePersistenceResult]]:
    run_id, first = _receipt()
    pages = [first]
    for index in range(1, count):
        _, other = _receipt()
        claim = other.claim_receipts[0]
        if different_wording and index == 1:
            text = "该装置的成功比例为12.4%。"
            claim = replace(claim, claim_text=text, claim_content_hash=claim_fingerprint(text))
        pages.append(replace(other, claim_receipts=(claim,)))
    return run_id, pages


def combined(pages: list[PagePersistenceResult]) -> PagePersistenceResult:
    return PagePersistenceResult(
        sum(item.inserted_count for item in pages),
        sum(item.accepted_count for item in pages),
        tuple(claim for page in pages for claim in page.claim_receipts),
    )


def pipeline(
    run_id: Any, gateway: PairGateway, artifacts: MemoryArtifactStore, **limits: Any
) -> SemanticResearchShadowPipeline:
    budget = _budget().model_copy(
        update={
            "max_identity_resolution_calls": 8,
            "proposition_pair_batch_size": 8,
            "proposition_pair_request_bytes": 200_000,
            **limits,
        }
    )
    return SemanticResearchShadowPipeline(
        repository=SimpleNamespace(),
        bindings=FakeBindings(run_id),
        cipher=FakeCipher(),
        gateway=gateway,
        artifacts=artifacts,
        budget=budget,
    )


def pair_calls(gateway: PairGateway) -> list[dict[str, Any]]:
    return [
        item
        for item in gateway.calls
        if item["request"].task_kind == "proposition_pair_identity_alignment_v1"
    ]


def groups(instance: SemanticResearchShadowPipeline) -> Any:
    return instance._last_deterministic["groups"]


def test_full_cross_claim_different_wording_formal_links_a2_a3_b1_b2_c1_c2() -> None:
    run, pages = cohort()
    gateway, artifacts = PairGateway(), MemoryArtifactStore()
    instance = pipeline(run, gateway, artifacts)
    _run_immediate(instance.process_persisted_page(run, combined(pages)))
    assert len(groups(instance)) == 1, [item.model_dump() for item in instance._diagnostics]
    group = groups(instance)[0]
    assert len(group.proposition_refs) == len(group.evidence_memberships) == 2
    links = [link for member in group.evidence_memberships for link in member.provenance_links]
    assert {link.claim_id for link in links} == {page.claim_receipts[0].claim_id for page in pages}
    assert {link.relation.value for link in links} == {"supports"}
    assert {
        item["assessment"]["comparability"] for item in instance._pair_assessment_provenance
    } == {"COMPARABLE"}
    assert {item["assessment"]["relation"] for item in instance._pair_assessment_provenance} == {
        "same"
    }
    assert instance._last_deterministic["corroborations"][0].support_evidence_count == 2
    assert instance._last_deterministic["conflicts"]
    assert instance._last_deterministic["verification"][0].verification_status.value == "SUPPORTED"
    assert len(gateway.calls) == 9 and len(pair_calls(gateway)) == 1
    assert instance.budget_state.provider_identity_resolution_calls == 3
    payload = pair_calls(gateway)[0]["request"].content_parts[0].value
    assert (
        payload["pairs"][0]["left"]["proposition"]["subject"]
        != payload["pairs"][0]["right"]["proposition"]["subject"]
    )
    assert all(
        item.subject_identity.status.value == "UNRESOLVED"
        for item in instance._target_assertions.values()
    )
    assert all(
        item.predicate_identity.status.value == "UNRESOLVED"
        for item in instance._target_assertions.values()
    )


@pytest.mark.parametrize("state", ["UNKNOWN", "DIFFERENT"])
def test_same_wording_without_same_identity_does_not_merge(state: str) -> None:
    run, pages = cohort(different_wording=False)
    gateway, artifacts = PairGateway(state), MemoryArtifactStore()
    instance = pipeline(run, gateway, artifacts)
    _run_immediate(instance.process_persisted_page(run, combined(pages)))
    assert len(groups(instance)) == 2
    assert len(pair_calls(gateway)) == 1
    _run_immediate(instance.process_persisted_page(run, combined(pages)))
    assert len(pair_calls(gateway)) == 1
    fresh = pipeline(run, gateway, artifacts)
    _run_immediate(fresh.process_persisted_page(run, combined(pages)))
    assert len(pair_calls(gateway)) == 1
    assert len(groups(fresh)) == 2
    assert set(fresh._pair_execution_states.values()) == {"COMPLETED"}


def test_ab_bc_same_ac_unknown_keeps_exact_frozen_b1_shape() -> None:
    run, pages = cohort(3)
    gateway, artifacts = PairGateway(unknown_middle=True), MemoryArtifactStore()
    instance = pipeline(run, gateway, artifacts)
    _run_immediate(instance.process_persisted_page(run, combined(pages)))
    assert len(groups(instance)) == 3
    assert {item.grouping_status.value for item in groups(instance)} == {"UNRESOLVED_MEMBERSHIP"}
    assert sorted(
        item["assessment"]["relation"] for item in instance._pair_assessment_provenance
    ) == ["same", "same", "unknown"]
    assert len(pair_calls(gateway)) == 1


def test_formation_unknown_cannot_be_promoted_by_pair_alignment() -> None:
    run, pages = cohort()
    gateway = PairGateway(unknown_claim=pages[0].claim_receipts[0].claim_id)
    instance = pipeline(run, gateway, MemoryArtifactStore())
    _run_immediate(instance.process_persisted_page(run, combined(pages)))
    assert len(groups(instance)) == 2
    assert not pair_calls(gateway)


def test_ap_source_and_ap_ap_share_one_budget_zero_remaining_no_dispatch() -> None:
    run, pages = cohort()
    gateway = PairGateway()
    instance = pipeline(run, gateway, MemoryArtifactStore(), max_identity_resolution_calls=2)
    _run_immediate(instance.process_persisted_page(run, combined(pages)))
    assert instance.budget_state.provider_identity_resolution_calls == 2
    assert not pair_calls(gateway)
    assert len(groups(instance)) == 2
    assert set(instance._pair_execution_states.values()) == {"NOT_PROCESSED_BUDGET"}
    assert any(
        item.status == "NOT_PROCESSED_BUDGET" and item.stage.value == "PROPOSITION_PAIR_IDENTITY"
        for item in instance._diagnostics
    )


def test_preplanned_batches_stop_at_shared_remaining_call_limit_even_after_invalid_output() -> None:
    run, pages = cohort(3)
    gateway = PairGateway(malformed=True)
    instance = pipeline(
        run,
        gateway,
        MemoryArtifactStore(),
        max_identity_resolution_calls=4,
        proposition_pair_batch_size=1,
    )
    _run_immediate(instance.process_persisted_page(run, combined(pages)))
    assert len(pair_calls(gateway)) == 1
    assert instance.budget_state.provider_identity_resolution_calls == 4
    assert len(groups(instance)) == 3


def test_same_instance_and_new_instance_resume_no_calls_without_final_snapshot() -> None:
    run, pages = cohort()
    gateway, artifacts = PairGateway(), MemoryArtifactStore()
    first = pipeline(run, gateway, artifacts)
    receipt = combined(pages)
    _run_immediate(first.process_persisted_page(run, receipt))
    expected = [item.model_dump(mode="json") for item in groups(first)]
    assert len(expected) == 1
    _run_immediate(first.process_persisted_page(run, receipt))
    assert len(gateway.calls) == 9
    for uri in tuple(artifacts.values):
        if "/receipts/" in uri or (
            "/semantic-shadow/" in uri
            and "/work-products/" not in uri
            and "/index/" not in uri
            and "/attempts/" not in uri
        ):
            del artifacts.values[uri]
    fresh = pipeline(run, gateway, artifacts)
    _run_immediate(fresh.process_persisted_page(run, receipt))
    assert len(gateway.calls) == 9, [item.model_dump() for item in fresh._diagnostics]
    assert [item.model_dump(mode="json") for item in groups(fresh)] == expected
    assert fresh.budget_state.provider_identity_resolution_calls == 3


def test_cross_receipt_recovers_old_targets_and_links_without_old_memory_cache() -> None:
    run, pages = cohort()
    gateway, artifacts = PairGateway(), MemoryArtifactStore()
    first = pipeline(run, gateway, artifacts)
    _run_immediate(first.process_persisted_page(run, pages[0]))
    assert len(gateway.calls) == 4
    fresh = pipeline(run, gateway, artifacts)
    _run_immediate(fresh.process_persisted_page(run, pages[1]))
    assert pages[0].claim_receipts[0].claim_id not in fresh._claims
    assert len(groups(fresh)) == 1, [item.model_dump() for item in fresh._diagnostics]
    assert len(groups(fresh)[0].evidence_memberships) == 2
    assert len(gateway.calls) == 9


def test_old_snapshot_does_not_skip_new_stage_or_repeat_old_provider_work() -> None:
    run, pages = cohort()
    gateway, artifacts = PairGateway(), MemoryArtifactStore()
    old = pipeline(run, gateway, artifacts, proposition_pair_batch_size=0)
    _run_immediate(old.process_persisted_page(run, combined(pages)))
    assert len(gateway.calls) == 8 and not pair_calls(gateway)
    for value in artifacts.values.values():
        if (
            isinstance(value, dict)
            and value.get("schema_version") == "semantic-shadow-verification-snapshot-v2"
        ):
            value["schema_version"] = "semantic-shadow-verification-snapshot-v1"
            value.pop("pair_assessments", None)
    fresh = pipeline(run, gateway, artifacts)
    _run_immediate(fresh.process_persisted_page(run, combined(pages)))
    assert len(gateway.calls) == 9
    assert len(groups(fresh)) == 1


def test_pair_product_precedes_completed_marker_and_invalid_result_is_not_regenerated() -> None:
    class RecordedStore(MemoryArtifactStore):
        async def save_json(self, uri: str, value: object) -> str:
            if uri.endswith(".attempt.json") and value["state"] == "COMPLETED":
                assert value["product_ref"] in self.values
            return await super().save_json(uri, value)

    run, pages = cohort()
    gateway, artifacts = PairGateway(malformed=True), RecordedStore()
    first = pipeline(run, gateway, artifacts)
    _run_immediate(first.process_persisted_page(run, combined(pages)))
    assert len(pair_calls(gateway)) == 1
    assert {result.status for result in first._pair_results.values()} == {"INVALID"}
    fresh = pipeline(run, gateway, artifacts)
    _run_immediate(fresh.process_persisted_page(run, combined(pages)))
    assert len(gateway.calls) == 9 and len(groups(fresh)) == 2


def test_failed_pair_result_is_durable_and_not_retried() -> None:
    class TimeoutPairGateway(PairGateway):
        async def generate_structured(self, **kwargs: Any) -> Any:
            if kwargs["request"].task_kind == "proposition_pair_identity_alignment_v1":
                self.calls.append(kwargs)
                raise TimeoutError("sensitive-response-body")
            return await super().generate_structured(**kwargs)

    run, pages = cohort()
    gateway, artifacts = TimeoutPairGateway(), MemoryArtifactStore()
    first = pipeline(run, gateway, artifacts)
    _run_immediate(first.process_persisted_page(run, combined(pages)))
    assert {item.status for item in first._pair_results.values()} == {"FAILED"}
    fresh = pipeline(run, gateway, artifacts)
    _run_immediate(fresh.process_persisted_page(run, combined(pages)))
    assert len(pair_calls(gateway)) == 1
    assert len(groups(fresh)) == 2
    assert "sensitive-response-body" not in str(artifacts.values)


def test_reserved_attempt_without_product_does_not_dispatch_again() -> None:
    run, pages = cohort()
    gateway, artifacts = PairGateway(), MemoryArtifactStore()
    first = pipeline(run, gateway, artifacts)
    _run_immediate(first.process_persisted_page(run, combined(pages)))
    for uri in tuple(artifacts.values):
        if "/proposition-pair-identity-" in uri and not uri.endswith(".attempt.json"):
            del artifacts.values[uri]
    fresh = pipeline(run, gateway, artifacts)
    _run_immediate(fresh.process_persisted_page(run, combined(pages)))
    assert len(pair_calls(gateway)) == 1
    assert len(groups(fresh)) == 2
    assert any(item.reason_code == "PAIR_ATTEMPT_WITHOUT_PRODUCT" for item in fresh._diagnostics)


def test_input_order_produces_same_plan_batch_refs_and_groups() -> None:
    run, pages = cohort(3)
    # The frozen AP parser assigns IDs once. Compare ordering over the same
    # durable proposition refs, not two independent randomized formations.
    seed_store = MemoryArtifactStore()
    seed = pipeline(run, PairGateway(), seed_store, proposition_pair_batch_size=0)
    _run_immediate(seed.process_persisted_page(run, combined(pages)))
    outputs = []
    for ordered in (pages, list(reversed(pages))):
        artifacts = MemoryArtifactStore()
        artifacts.values = deepcopy(seed_store.values)
        instance = pipeline(run, PairGateway(), artifacts)
        _run_immediate(instance.process_persisted_page(run, combined(ordered)))
        outputs.append(
            (
                instance._pair_plan_summary,
                [item.model_dump(mode="json") for item in groups(instance)],
                instance._pair_assessment_provenance,
            )
        )
    assert outputs[0] == outputs[1]
