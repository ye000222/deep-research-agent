from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from app.core.config import Settings
from app.domain.atomic_proposition_contracts import (
    ATOMIC_DECOMPOSITION_PROMPT_V3,
    ATOMIC_PROPOSITION_SCHEMA_V3,
)
from app.domain.document_evidence_context import (
    CanonicalEvidenceSpan,
    SourceLocator,
    SourceLocatorType,
)
from app.domain.evidence_graph import claim_fingerprint
from app.domain.providers import AdapterType, CanonicalModelResult, TokenUsage, UsageAccuracy
from app.domain.semantic_runtime_budget import SemanticRuntimeBudget
from app.domain.source_context_extraction_v3_4 import (
    SOURCE_CONTEXT_EXTRACTION_SCHEMA_V34,
    SOURCE_SEMANTIC_RUNTIME_RESPONSE_V2,
)
from app.infrastructure.artifacts import LocalArtifactStore
from app.infrastructure.db.research_tools import (
    ClaimPersistenceReceipt,
    EvidencePersistenceReceipt,
    PagePersistenceResult,
)
from app.services import semantic_research_shadow as shadow_module
from app.services.research_loop import ResearchLoopService
from app.services.semantic_research_shadow import (
    SemanticResearchShadowPipeline,
    SemanticShadowMode,
)
from pydantic import SecretStr


class FakeGateway:
    def __init__(
        self,
        *,
        source_text: str | None = None,
        source_polarity: str = "asserted",
        infer_fixture_polarity_from_text: bool = False,
        fail_task_kind: str | None = None,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.source_text = source_text
        self.source_polarity = source_polarity
        self.infer_fixture_polarity_from_text = infer_fixture_polarity_from_text
        self.fail_task_kind = fail_task_kind

    async def generate_structured(self, **kwargs: Any) -> CanonicalModelResult:
        self.calls.append(kwargs)
        request = kwargs["request"]
        if request.task_kind == self.fail_task_kind:
            raise RuntimeError("fake V34 extraction transport failure")
        if request.task_kind == "atomic_proposition_shadow_decomposition":
            payload: dict[str, Any] = {
                "schema_version": ATOMIC_PROPOSITION_SCHEMA_V3,
                "decomposition_status": "single_proposition",
                "propositions": [
                    {
                        "local_ref": "p0",
                        "proposition_text": "设备成功率为12.4%。",
                        "subject": "设备",
                        "predicate": "成功率",
                        "typed_value": {
                            "kind": "quantity",
                            "numeric": {"operator": "exact", "value": 12.4, "unit": "%"},
                        },
                        "polarity": "asserted",
                        "source_span": "设备成功率为12.4%。",
                    }
                ],
            }
        elif request.task_kind == "target_proposition_formation_validation_v1":
            request_payload = request.content_parts[0].value
            payload = {
                "schema_version": "target-proposition-formation-response-v1",
                "judgments": [
                    {
                        "claim_ref": request_payload["claim_ref"],
                        "proposition_ref": proposition["proposition_ref"],
                        "claim_proposition_link_ref": proposition["claim_proposition_link_ref"],
                        "fields": {
                            "subject": "PRESERVED",
                            "predicate": "PRESERVED",
                            "typed_value": "PRESERVED",
                            "polarity": "PRESERVED",
                            "material_contexts": "PRESERVED",
                            "attribution": "PRESERVED",
                            "modality": "PRESERVED",
                            "composition": "PRESERVED",
                            "scope_eligibility": "PRESERVED",
                        },
                        "reason_codes": [],
                    }
                    for proposition in request_payload["claim_propositions"]
                ],
            }
        elif request.task_kind == "semantic_identity_alignment_v1":
            request_payload = request.content_parts[0].value
            payload = {
                "schema_version": "semantic-identity-alignment-response-v1",
                "status": "RESOLVED",
                "alignments": [
                    {
                        "pair_ref": pair["pair_ref"],
                        "subject_relation": "SAME",
                        "predicate_relation": "SAME",
                        "reason_code": "IDENTITY_ALIGNED",
                    }
                    for pair in request_payload["pairs"]
                ],
            }
        else:
            request_input = request.content_parts[0].value
            fixture_polarity = self.source_polarity
            if self.infer_fixture_polarity_from_text:
                fixture_polarity = (
                    "negated"
                    if (
                        " not " in request_input["canonical_span"]["exact_text"]
                        or "不是" in request_input["canonical_span"]["exact_text"]
                    )
                    else "asserted"
                )
            payload = _source_v34_response(
                request_input,
                self.source_text,
                fixture_polarity,
            )
        return CanonicalModelResult(
            parsed_object=payload,
            usage=TokenUsage(
                input_tokens=5,
                output_tokens=7,
                total_tokens=12,
                accuracy=UsageAccuracy.EXACT,
            ),
            provider_request_id="fake-only",
        )


class FakeBindings:
    def __init__(self, run_id: Any) -> None:
        self.binding = SimpleNamespace(
            run_id=run_id,
            encrypted_secret=object(),
            credential_id=uuid4(),
            credential_version=1,
            adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
            base_url="https://example.invalid/v1",
            model="fake-model",
        )

    async def get(self, run_id: Any) -> Any:
        assert run_id == self.binding.run_id
        return self.binding


class FakeCipher:
    def decrypt(self, *_args: Any, **_kwargs: Any) -> SecretStr:
        return SecretStr("never-a-real-secret")


class MemoryArtifactStore:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    async def create_json_once(self, artifact_uri: str, value: object) -> bool:
        if artifact_uri in self.values:
            return False
        self.values[artifact_uri] = value
        return True

    async def save_json(self, artifact_uri: str, value: object) -> str:
        self.values[artifact_uri] = value
        return artifact_uri

    async def read_json(self, artifact_uri: str) -> object:
        if artifact_uri not in self.values:
            raise FileNotFoundError(artifact_uri)
        return self.values[artifact_uri]


def _run_immediate(coroutine: Any) -> Any:
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("Fake semantic runtime unexpectedly suspended on external work")


def _receipt(*, evidence_text: str | None = None) -> tuple[Any, PagePersistenceResult]:
    run_id, claim_id, evidence_id, source_id, snapshot_id = (uuid4() for _ in range(5))
    text = "设备成功率为12.4%。"
    snapshot_content_hash = hashlib.sha256(b"snapshot-content").hexdigest()
    canonical_text = evidence_text or text
    span = CanonicalEvidenceSpan(
        span_ref=hashlib.sha256(b"span-ref").hexdigest(),
        document_ref=hashlib.sha256(b"document-ref").hexdigest(),
        source_ref=str(source_id),
        snapshot_ref=hashlib.sha256(snapshot_content_hash.encode()).hexdigest(),
        char_start=0,
        char_end=len(canonical_text),
        exact_text=canonical_text,
        content_sha256=hashlib.sha256(canonical_text.encode()).hexdigest(),
        source_locator=SourceLocator(locator_type=SourceLocatorType.UNKNOWN),
    )
    evidence = EvidencePersistenceReceipt(
        claim_id=claim_id,
        evidence_id=evidence_id,
        source_id=source_id,
        snapshot_id=snapshot_id,
        snapshot_content_hash=snapshot_content_hash,
        question_id="task-001",
        dimension_key="task-001:d1",
        source_owner_key=None,
        canonical_span=span,
        accepted_and_persisted=True,
    )
    claim = ClaimPersistenceReceipt(
        claim_id=claim_id,
        evidence_ids=(evidence_id,),
        source_ids=(source_id,),
        snapshot_ids=(snapshot_id,),
        source_provenance_refs=(
            f"claim:{claim_id}",
            f"evidence:{evidence_id}",
            f"source:{source_id}",
            f"snapshot:{snapshot_id}",
        ),
        source_diversity_key="source-diversity-fingerprint",
        source_owner_key=None,
        claim_content_hash=claim_fingerprint(text),
        claim_text=text,
        evidence_receipts=(evidence,),
        created_at=datetime.now(UTC),
        accepted_and_persisted=True,
    )
    return run_id, PagePersistenceResult(1, 1, (claim,))


def _source_v34_response(
    request_input: dict[str, Any], source_text: str | None, polarity: str
) -> dict[str, Any]:
    canonical = request_input["canonical_span"]
    exact_text = source_text or canonical["exact_text"]
    value_text = "12.4%"
    value_offset = exact_text.index(value_text)
    source_ref = request_input["source_ref"]
    snapshot_ref = request_input["snapshot_ref"]
    predicate = "成功率"
    if polarity == "negated":
        predicate = "成功率"
    return {
        "schema_version": SOURCE_SEMANTIC_RUNTIME_RESPONSE_V2,
        "status": "RESOLVED",
        "source_extraction": {
            "schema_version": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V34,
            "status": "RESOLVED",
            "source_assertions": [
                {
                    "source_assertion_ref": "A1",
                    "evidence_refs": ["whole", "value"],
                    "assertion_status": "ASSERTED",
                }
            ],
            "source_contexts": [],
            "evidence_spans": [
                {"evidence_ref": "whole", "exact_text": exact_text},
                {"evidence_ref": "value", "exact_text": value_text},
            ],
            "evidence_bindings": [
                {
                    "binding_ref": "b-subject",
                    "evidence_ref": "whole",
                    "semantic_role": "ASSERTION_EVIDENCE",
                    "target_kind": "ASSERTION",
                    "target_ref": "A1",
                },
                {
                    "binding_ref": "b-predicate",
                    "evidence_ref": "whole",
                    "semantic_role": "RELATION_OPERATOR",
                    "target_kind": "ASSERTION",
                    "target_ref": "A1",
                },
                {
                    "binding_ref": "b-value",
                    "evidence_ref": "value",
                    "semantic_role": "ASSERTION_EVIDENCE",
                    "target_kind": "ASSERTION",
                    "target_ref": "A1",
                },
                {
                    "binding_ref": "b-polarity",
                    "evidence_ref": "whole",
                    "semantic_role": "ASSERTION_EVIDENCE",
                    "target_kind": "ASSERTION",
                    "target_ref": "A1",
                },
            ],
            "typed_value_bindings": [
                _typed_binding(
                    "tv-magnitude",
                    "NUMERIC_MAGNITUDE",
                    source_ref,
                    snapshot_ref,
                    value_offset,
                    value_text,
                    canonical["char_start"],
                ),
                _typed_binding(
                    "tv-unit",
                    "UNIT",
                    source_ref,
                    snapshot_ref,
                    value_offset,
                    value_text,
                    canonical["char_start"],
                ),
            ],
            "semantic_relations": [],
            "measurement_bases": [],
            "unresolved_items": [],
        },
        "assertion_candidates": [
            {
                "local_ref": "candidate-1",
                "source_assertion_ref": "A1",
                "evidence_refs": ["whole", "value"],
                "subject": {
                    "source_representation": "设备",
                    "semantic_identity": {"status": "UNRESOLVED"},
                },
                "predicate": {
                    "source_representation": predicate,
                    "semantic_identity": {"status": "UNRESOLVED"},
                },
                "value": {
                    "kind": "quantity",
                    "numeric": {"operator": "exact", "value": 12.4, "unit": "%"},
                },
                "polarity": polarity,
                "field_evidence_refs": {
                    "SUBJECT": ["whole"],
                    "PREDICATE": ["whole"],
                    "VALUE": ["value"],
                    "POLARITY": ["whole"],
                },
                "truth_condition_context_refs": [],
                "verification_eligibility": "INDEPENDENTLY_ASSERTED",
            }
        ],
        "truth_condition_contexts": [],
        "composition": None,
        "unresolved_items": [],
    }


def _typed_binding(
    binding_ref: str,
    role: str,
    source_ref: str,
    snapshot_ref: str,
    start: int,
    exact_text: str,
    canonical_start: int,
) -> dict[str, Any]:
    return {
        "binding_ref": binding_ref,
        "assertion_ref": "A1",
        "value_field_role": "VALUE",
        "evidence_ref": "value",
        "occurrence_index": 0,
        "char_start": start + canonical_start,
        "char_end": start + canonical_start + len(exact_text),
        "component_role": role,
        "source_ref": source_ref,
        "snapshot_ref": snapshot_ref,
        "contract_version": "source-typed-value-grounding-v1",
    }


def _budget() -> SemanticRuntimeBudget:
    return SemanticRuntimeBudget(
        max_ap_decomposition_calls=4,
        max_source_semantic_extraction_calls=8,
        max_identity_resolution_calls=4,
        max_formation_validation_calls=4,
        max_evidence_inputs=8,
        max_propositions_per_evidence=8,
        max_source_assertions_per_evidence=8,
        max_candidate_pairs=16,
        max_direction_assessments=16,
    )


def test_semantic_runtime_feature_is_off_by_default() -> None:
    assert Settings(_env_file=None).semantic_runtime_mode == "OFF"


def test_shadow_target_formation_and_source_polarity_create_support_link(
    monkeypatch: Any,
) -> None:
    run_id, persisted = _receipt()
    gateway = FakeGateway()
    source_calls = 0
    actual_runtime = shadow_module.SourceSemanticExtractionV34Runtime

    class CountedSourceRuntime(actual_runtime):
        async def extract(self, runtime_input: Any) -> Any:
            nonlocal source_calls
            source_calls += 1
            return await super().extract(runtime_input)

    monkeypatch.setattr(shadow_module, "SourceSemanticExtractionV34Runtime", CountedSourceRuntime)
    artifacts = MemoryArtifactStore()

    def pipeline() -> SemanticResearchShadowPipeline:
        return SemanticResearchShadowPipeline(
            repository=SimpleNamespace(),
            bindings=FakeBindings(run_id),
            cipher=FakeCipher(),
            gateway=gateway,
            artifacts=artifacts,
            budget=_budget(),
        )

    first = pipeline()
    direction_results: list[Any] = []
    original_direction = shadow_module.evaluate_proposition_evidence_directions

    def capture_direction(*args: Any, **kwargs: Any) -> Any:
        result = original_direction(*args, **kwargs)
        direction_results.append(result)
        return result

    monkeypatch.setattr(
        shadow_module, "evaluate_proposition_evidence_directions", capture_direction
    )
    _run_immediate(first.process_persisted_page(run_id, persisted))
    assert len(gateway.calls) == 4
    assert source_calls == 1
    assert first.budget_state.evidence_inputs == 1
    assert first.budget_state.propositions == 1
    assert first.budget_state.source_assertions == 1
    assert all(call["allow_regeneration"] is False for call in gateway.calls)
    assert all(call["request"].metadata["run_id"] == str(run_id) for call in gateway.calls)
    assert gateway.calls[0]["request"].metadata["schema_version"] == (ATOMIC_PROPOSITION_SCHEMA_V3)
    assert gateway.calls[0]["request"].metadata["prompt_version"] == (
        ATOMIC_DECOMPOSITION_PROMPT_V3
    )
    snapshots = [
        value
        for uri, value in artifacts.values.items()
        if uri.startswith(f"runs/{run_id}/semantic-shadow/")
        and "/attempts/" not in uri
        and "/receipts/" not in uri
        and "/work-products/" not in uri
        and "/work-products/" not in uri
    ]
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["schema_version"] == "semantic-shadow-verification-snapshot-v2"
    assert snapshot["claim_refs"] == [str(persisted.claim_receipts[0].claim_id)]
    assert snapshot["evidence_refs"] == [str(persisted.claim_receipts[0].evidence_ids[0])]
    assert snapshot["proposition_refs"]
    assessment = direction_results[0].assessments[0]
    assert [item["relation"] for item in snapshot["proposition_evidence_links"]] == ["supports"], {
        "diagnostics": snapshot["diagnostics"],
        "formation": snapshot["formation_assessments"],
        "direction": json.dumps(assessment.model_dump(mode="json"), indent=2),
        "target": next(iter(first._target_assertions.values())).model_dump(mode="json"),
    }
    assert assessment.a2_comparability == "COMPARABLE"
    assert assessment.grounding_usability.value == "SUFFICIENT"
    assert "GROUNDING_UNUSABLE" not in {item.value for item in assessment.reason_codes}
    target_assertion = next(iter(first._target_assertions.values()))
    assert target_assertion.grounding_usability.value == "SUFFICIENT"
    assert snapshot["formation_assessments"][0]["formation_status"] == "FORMATION_SUFFICIENT"
    alignment_products = [
        value
        for uri, value in artifacts.values.items()
        if "/work-products/identity-alignment-" in uri
    ]
    assert len(alignment_products) == 1
    assert all(item["alignment_status"] == "SAME" for item in alignment_products[0]["facts"])
    source_product = next(
        value for uri, value in artifacts.values.items() if "/work-products/source-" in uri
    )
    polarity_grounding = next(
        item
        for item in source_product["assertions"][0]["field_grounding"]
        if item["field"] == "POLARITY"
    )
    assert polarity_grounding["textual_grounding"] == "VALID"
    assert polarity_grounding["semantic_support"] == "VALID"
    assert polarity_grounding["structural_grounding"] == "VALID"
    assert polarity_grounding["reason_codes"] == [
        "ASSERTION_LEVEL_POLARITY_BASIS",
        "V33_SEMANTIC_BINDING_VALIDATED",
    ]
    assert snapshot["corroboration_assessments"][0]["support_evidence_count"] == 1
    assert snapshot["verification_assessments"][0]["positive_support"] != "NONE"
    assert snapshot["verification_assessments"][0]["verification_status"] == "SUPPORTED"
    assert snapshot["proposition_group_refs"]
    assert snapshot["corroboration_assessments"]
    assert snapshot["conflicts"]
    assert snapshot["verification_assessments"]
    assert all(
        item["verification_status"] != "VERIFIED" for item in snapshot["verification_assessments"]
    )
    assert "设备成功率为12.4%。" not in json.dumps(snapshot)

    replay = pipeline()
    _run_immediate(replay.process_persisted_page(run_id, persisted))
    assert len(gateway.calls) == 4
    assert replay.budget_state.provider_ap_decomposition_calls == 1
    assert replay.budget_state.provider_source_semantic_extraction_calls == 1
    assert replay.budget_state.provider_identity_resolution_calls == 1
    assert replay.budget_state.provider_formation_validation_calls == 1


def test_v34_extraction_gateway_failure_cannot_form_false_semantic_success(
    monkeypatch: Any,
) -> None:
    run_id, persisted = _receipt()
    gateway = FakeGateway(fail_task_kind="source_semantic_extraction_v3_4")
    source_calls = 0
    actual_runtime = shadow_module.SourceSemanticExtractionV34Runtime

    class CountedSourceRuntime(actual_runtime):
        async def extract(self, runtime_input: Any) -> Any:
            nonlocal source_calls
            source_calls += 1
            return await super().extract(runtime_input)

    monkeypatch.setattr(shadow_module, "SourceSemanticExtractionV34Runtime", CountedSourceRuntime)
    artifacts = MemoryArtifactStore()
    pipeline = SemanticResearchShadowPipeline(
        repository=SimpleNamespace(),
        bindings=FakeBindings(run_id),
        cipher=FakeCipher(),
        gateway=gateway,
        artifacts=artifacts,
        budget=_budget(),
    )

    _run_immediate(pipeline.process_persisted_page(run_id, persisted))

    source_calls_made = [
        call
        for call in gateway.calls
        if call["request"].task_kind == "source_semantic_extraction_v3_4"
    ]
    assert source_calls == 1
    assert len(source_calls_made) == 1
    source_products = [
        value for uri, value in artifacts.values.items() if "/work-products/source-" in uri
    ]
    assert len(source_products) == 1
    assert source_products[0]["runtime_status"] == "FAILED"
    assert source_products[0]["assertions"] == []
    assert source_products[0].get("typed_value_grounding_assessments", []) == []

    snapshots = [
        value
        for uri, value in artifacts.values.items()
        if uri.startswith(f"runs/{run_id}/semantic-shadow/")
        and "/attempts/" not in uri
        and "/receipts/" not in uri
        and "/work-products/" not in uri
    ]
    assert len(snapshots) == 1
    assert snapshots[0]["proposition_evidence_links"] == []
    assert all(
        item["verification_status"] != "VERIFIED"
        for item in snapshots[0]["verification_assessments"]
    )


def test_shadow_negative_source_can_form_contradiction_after_target_formation() -> None:
    run_id, persisted = _receipt(evidence_text="设备成功率不是12.4%。")
    gateway = FakeGateway(source_text="设备成功率不是12.4%。", source_polarity="negated")
    artifacts = MemoryArtifactStore()
    pipeline = SemanticResearchShadowPipeline(
        repository=SimpleNamespace(),
        bindings=FakeBindings(run_id),
        cipher=FakeCipher(),
        gateway=gateway,
        artifacts=artifacts,
        budget=_budget(),
    )

    _run_immediate(pipeline.process_persisted_page(run_id, persisted))

    snapshots = [
        value
        for uri, value in artifacts.values.items()
        if uri.startswith(f"runs/{run_id}/semantic-shadow/")
        and "/attempts/" not in uri
        and "/receipts/" not in uri
        and "/work-products/" not in uri
    ]
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert [item["relation"] for item in snapshot["proposition_evidence_links"]] == ["contradicts"]
    assert snapshot["verification_assessments"]
    source_product = next(
        value for uri, value in artifacts.values.items() if "/work-products/source-" in uri
    )
    negative_polarity = next(
        item
        for item in source_product["assertions"][0]["field_grounding"]
        if item["field"] == "POLARITY"
    )
    assert negative_polarity["textual_grounding"] == "VALID"
    assert negative_polarity["semantic_support"] == "VALID"
    assert negative_polarity["structural_grounding"] == "VALID"


def test_cross_claim_b1_fails_closed_without_frozen_same_identity_inputs() -> None:
    run_id, first = _receipt()
    _other_run_id, second = _receipt()
    combined = PagePersistenceResult(
        inserted_count=first.inserted_count + second.inserted_count,
        accepted_count=first.accepted_count + second.accepted_count,
        claim_receipts=first.claim_receipts + second.claim_receipts,
    )
    gateway = FakeGateway()
    artifacts = MemoryArtifactStore()
    pipeline = SemanticResearchShadowPipeline(
        repository=SimpleNamespace(),
        bindings=FakeBindings(run_id),
        cipher=FakeCipher(),
        gateway=gateway,
        artifacts=artifacts,
        budget=_budget(),
    )

    _run_immediate(pipeline.process_persisted_page(run_id, combined))

    snapshot = next(
        value
        for uri, value in artifacts.values.items()
        if uri.startswith(f"runs/{run_id}/semantic-shadow/")
        and "/attempts/" not in uri
        and "/receipts/" not in uri
        and "/work-products/" not in uri
    )
    links = snapshot["proposition_evidence_links"]
    assert len(links) == 2
    assert {item["relation"] for item in links} == {"supports"}
    # Current target assertions do not receive comparison-local SAME identity
    # from source alignment, so frozen A2/A3 conservatively keep two singletons.
    # This proves fail-closed behavior, and records the remaining cross-claim gate.
    assert len(snapshot["proposition_group_refs"]) == 2
    assert [item["support_evidence_count"] for item in snapshot["corroboration_assessments"]] == [
        1,
        1,
    ]
    assert snapshot["verification_assessments"]
    assert len(gateway.calls) == 8


def test_c1_c2_consume_support_and_contradiction_links_from_v34_runtime() -> None:
    run_id, positive = _receipt()
    _other_run_id, negative = _receipt(evidence_text="设备成功率不是12.4%。")
    first_claim = positive.claim_receipts[0]
    second_claim = negative.claim_receipts[0]
    positive_evidence = first_claim.evidence_receipts[0]
    negative_evidence = replace(second_claim.evidence_receipts[0], claim_id=first_claim.claim_id)
    combined_claim = replace(
        first_claim,
        evidence_ids=(positive_evidence.evidence_id, negative_evidence.evidence_id),
        source_ids=(positive_evidence.source_id, negative_evidence.source_id),
        snapshot_ids=(positive_evidence.snapshot_id, negative_evidence.snapshot_id),
        source_provenance_refs=(
            *first_claim.source_provenance_refs,
            f"evidence:{negative_evidence.evidence_id}",
            f"source:{negative_evidence.source_id}",
            f"snapshot:{negative_evidence.snapshot_id}",
        ),
        evidence_receipts=(positive_evidence, negative_evidence),
    )
    combined = PagePersistenceResult(2, 2, (combined_claim,))
    gateway = FakeGateway(infer_fixture_polarity_from_text=True)
    artifacts = MemoryArtifactStore()
    pipeline = SemanticResearchShadowPipeline(
        repository=SimpleNamespace(),
        bindings=FakeBindings(run_id),
        cipher=FakeCipher(),
        gateway=gateway,
        artifacts=artifacts,
        budget=_budget(),
    )

    _run_immediate(pipeline.process_persisted_page(run_id, combined))
    snapshot = next(
        value
        for uri, value in artifacts.values.items()
        if uri.startswith(f"runs/{run_id}/semantic-shadow/")
        and "/attempts/" not in uri
        and "/receipts/" not in uri
        and "/work-products/" not in uri
    )
    assert {item["relation"] for item in snapshot["proposition_evidence_links"]} == {
        "supports",
        "contradicts",
    }
    assert snapshot["corroboration_assessments"]
    assert snapshot["conflicts"]["group_summaries"]
    assert any(
        item["counterevidence"]["contradicting_evidence_refs"]
        for item in snapshot["conflicts"]["group_summaries"]
    )
    assert snapshot["verification_assessments"]
    assert snapshot["verification_assessments"][0]["verification_status"] == "SUPPORTED"
    assert snapshot["verification_assessments"][0]["conflict_state"] == "UNRESOLVED"
    assert len(gateway.calls) == 6


def test_resume_restores_durable_ap_source_and_alignment_products_without_calls() -> None:
    run_id, persisted = _receipt()
    gateway = FakeGateway()
    artifacts = MemoryArtifactStore()
    budget = _budget()

    def pipeline() -> SemanticResearchShadowPipeline:
        return SemanticResearchShadowPipeline(
            repository=SimpleNamespace(),
            bindings=FakeBindings(run_id),
            cipher=FakeCipher(),
            gateway=gateway,
            artifacts=artifacts,
            budget=budget,
        )

    first = pipeline()
    _run_immediate(first.process_persisted_page(run_id, persisted))
    pointer_uri = next(
        uri for uri in artifacts.values if "/receipts/" in uri and uri.endswith("snapshot.json")
    )
    snapshot_ref = artifacts.values[pointer_uri]["artifact_ref"]
    del artifacts.values[pointer_uri]
    del artifacts.values[snapshot_ref]
    assert pointer_uri not in artifacts.values and snapshot_ref not in artifacts.values
    assert any("/work-products/ap-" in uri for uri in artifacts.values)

    resumed = pipeline()
    _run_immediate(resumed.process_persisted_page(run_id, persisted))
    assert resumed._claims
    assert len(gateway.calls) == 4
    assert resumed.budget_state.provider_ap_decomposition_calls == 1, {
        "diagnostics": [item.model_dump(mode="json") for item in resumed._diagnostics],
        "uris": list(artifacts.values),
    }
    assert resumed.budget_state.provider_source_semantic_extraction_calls == 1
    assert resumed.budget_state.provider_identity_resolution_calls == 1
    assert resumed.budget_state.provider_formation_validation_calls == 1
    source_product = next(
        value for uri, value in artifacts.values.items() if "/work-products/source-" in uri
    )
    resumed_polarity = next(
        item
        for item in source_product["assertions"][0]["field_grounding"]
        if item["field"] == "POLARITY"
    )
    assert resumed_polarity["textual_grounding"] == "VALID"
    assert resumed_polarity["semantic_support"] == "VALID"
    assert resumed_polarity["structural_grounding"] == "VALID"


def test_partial_resume_missing_identity_product_calls_only_identity_stage() -> None:
    run_id, persisted = _receipt()
    gateway = FakeGateway()
    artifacts = MemoryArtifactStore()
    budget = _budget()

    def pipeline() -> SemanticResearchShadowPipeline:
        return SemanticResearchShadowPipeline(
            repository=SimpleNamespace(),
            bindings=FakeBindings(run_id),
            cipher=FakeCipher(),
            gateway=gateway,
            artifacts=artifacts,
            budget=budget,
        )

    first = pipeline()
    _run_immediate(first.process_persisted_page(run_id, persisted))
    pointer_uri = next(
        uri for uri in artifacts.values if "/receipts/" in uri and uri.endswith("snapshot.json")
    )
    snapshot_ref = artifacts.values[pointer_uri]["artifact_ref"]
    del artifacts.values[pointer_uri]
    del artifacts.values[snapshot_ref]
    for uri in tuple(artifacts.values):
        if (
            "/work-products/identity-alignment-" in uri
            or "/attempts/alignment-" in uri
            or uri.endswith("/index/derived-state-v1.json")
        ):
            del artifacts.values[uri]

    resumed = pipeline()
    _run_immediate(resumed.process_persisted_page(run_id, persisted))
    assert len(gateway.calls) == 5, {
        "uris": list(artifacts.values),
        "diagnostics": [item.model_dump(mode="json") for item in resumed._diagnostics],
    }
    assert gateway.calls[-1]["request"].task_kind == "semantic_identity_alignment_v1"
    assert resumed.budget_state.provider_ap_decomposition_calls == 1
    assert resumed.budget_state.provider_source_semantic_extraction_calls == 1
    assert resumed.budget_state.provider_identity_resolution_calls == 1


def test_identity_budget_exhaustion_does_not_dispatch_or_claim_semantic_unknown() -> None:
    run_id, persisted = _receipt()
    gateway = FakeGateway()
    artifacts = MemoryArtifactStore()
    budget = _budget().model_copy(update={"max_identity_resolution_calls": 0})
    pipeline = SemanticResearchShadowPipeline(
        repository=SimpleNamespace(),
        bindings=FakeBindings(run_id),
        cipher=FakeCipher(),
        gateway=gateway,
        artifacts=artifacts,
        budget=budget,
    )
    _run_immediate(pipeline.process_persisted_page(run_id, persisted))
    assert [call["request"].task_kind for call in gateway.calls] == [
        "atomic_proposition_shadow_decomposition",
        "target_proposition_formation_validation_v1",
        "source_semantic_extraction_v3_4",
    ]
    assert pipeline.budget_state.provider_identity_resolution_calls == 0
    assert any(
        item.reason_code == "IDENTITY_ALIGNMENT_PROVIDER_BUDGET_EXHAUSTED"
        and item.status == "NOT_PROCESSED_BUDGET"
        for item in pipeline._diagnostics
    )


def test_source_budget_exhaustion_does_not_dispatch_reader_or_identity_work() -> None:
    run_id, persisted = _receipt()
    gateway = FakeGateway()
    artifacts = MemoryArtifactStore()
    budget = _budget().model_copy(update={"max_source_semantic_extraction_calls": 0})
    pipeline = SemanticResearchShadowPipeline(
        repository=SimpleNamespace(),
        bindings=FakeBindings(run_id),
        cipher=FakeCipher(),
        gateway=gateway,
        artifacts=artifacts,
        budget=budget,
    )
    _run_immediate(pipeline.process_persisted_page(run_id, persisted))
    assert [call["request"].task_kind for call in gateway.calls] == [
        "atomic_proposition_shadow_decomposition",
        "target_proposition_formation_validation_v1",
    ]
    assert pipeline.budget_state.provider_source_semantic_extraction_calls == 0
    assert pipeline.budget_state.provider_identity_resolution_calls == 0
    assert any(
        item.reason_code == "SOURCE_SEMANTIC_PROVIDER_BUDGET_EXHAUSTED"
        and item.status == "NOT_PROCESSED_BUDGET"
        for item in pipeline._diagnostics
    )


def test_snapshot_marker_creation_is_atomic_at_most_once() -> None:
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
        destination = Path(temp_dir) / "attempts" / "one.json"
        first = LocalArtifactStore._create_once(destination, '{"state":"RESERVED"}')
        second = LocalArtifactStore._create_once(destination, '{"state":"RESERVED"}')
    assert (first, second) == (True, False)


def test_missing_shadow_budget_fails_closed_without_gateway_call() -> None:
    run_id, persisted = _receipt()
    gateway = FakeGateway()
    pipeline = SemanticResearchShadowPipeline(
        repository=SimpleNamespace(),
        bindings=FakeBindings(run_id),
        cipher=FakeCipher(),
        gateway=gateway,
        artifacts=MemoryArtifactStore(),
        budget=None,
    )
    _run_immediate(pipeline.process_persisted_page(run_id, persisted))
    assert gateway.calls == []
    assert pipeline.budget_state.provider_ap_decomposition_calls == 0


def test_provider_failure_is_isolated_and_persisted_receipt_is_not_replayed() -> None:
    run_id, persisted = _receipt()

    class FailingGateway:
        def __init__(self) -> None:
            self.calls = 0

        async def generate_structured(self, **_kwargs: Any) -> Any:
            self.calls += 1
            raise RuntimeError("sensitive provider response must not be serialized")

    gateway = FailingGateway()
    artifacts = MemoryArtifactStore()
    pipeline = SemanticResearchShadowPipeline(
        repository=SimpleNamespace(),
        bindings=FakeBindings(run_id),
        cipher=FakeCipher(),
        gateway=gateway,
        artifacts=artifacts,
        budget=_budget(),
    )

    _run_immediate(pipeline.process_persisted_page(run_id, persisted))
    _run_immediate(pipeline.process_persisted_page(run_id, persisted))

    assert gateway.calls == 1
    assert pipeline.budget_state.provider_ap_decomposition_calls == 1
    serialized_artifacts = json.dumps(artifacts.values, sort_keys=True)
    assert "sensitive provider response" not in serialized_artifacts
    snapshots = [
        value
        for uri, value in artifacts.values.items()
        if uri.startswith(f"runs/{run_id}/semantic-shadow/")
        and "/attempts/" not in uri
        and "/receipts/" not in uri
        and "/work-products/" not in uri
        and "/work-products/" not in uri
    ]
    assert len(snapshots) == 1
    assert any(
        diagnostic["stage"] == "AP_FORMATION"
        and diagnostic["reason_code"] == "AP_DECOMPOSITION_UNRESOLVED_OR_FAILED"
        for diagnostic in snapshots[0]["diagnostics"]
    )


def test_provider_budget_exhaustion_is_reported_without_outbound() -> None:
    run_id, persisted = _receipt()
    gateway = FakeGateway()
    budget = _budget().model_copy(update={"max_ap_decomposition_calls": 0})
    artifacts = MemoryArtifactStore()
    pipeline = SemanticResearchShadowPipeline(
        repository=SimpleNamespace(),
        bindings=FakeBindings(run_id),
        cipher=FakeCipher(),
        gateway=gateway,
        artifacts=artifacts,
        budget=budget,
    )

    _run_immediate(pipeline.process_persisted_page(run_id, persisted))

    assert gateway.calls == []
    assert pipeline.budget_state.provider_ap_decomposition_calls == 0
    snapshot = next(
        value
        for uri, value in artifacts.values.items()
        if uri.startswith(f"runs/{run_id}/semantic-shadow/")
        and "/attempts/" not in uri
        and "/receipts/" not in uri
        and "/work-products/" not in uri
    )
    assert any(
        diagnostic["reason_code"] == "AP_DECOMPOSITION_PROVIDER_BUDGET_EXHAUSTED"
        for diagnostic in snapshot["diagnostics"]
    )


def test_research_loop_off_uses_receipt_persistence_without_shadow_work() -> None:
    run_id, persisted = _receipt()
    events: list[str] = []

    class Repository:
        async def record_page(self, _run_id: Any, **_kwargs: Any) -> tuple[int, int]:
            events.append("legacy_persist")
            return (1, 1)

        async def record_page_with_receipts(self, *_args: Any, **_kwargs: Any) -> Any:
            events.append("receipt_persist")
            return persisted

    class Shadow:
        async def process_persisted_page(self, *_args: Any) -> None:
            events.append("shadow")

    service = ResearchLoopService(
        Repository(),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        semantic_shadow_mode=SemanticShadowMode.OFF,
        semantic_shadow=Shadow(),  # type: ignore[arg-type]
    )
    counts = _run_immediate(service._record_page_with_shadow(run_id, sentinel="value"))

    expected_evidence_refs = tuple(
        evidence.evidence_id
        for receipt in persisted.claim_receipts
        for evidence in receipt.evidence_receipts
        if evidence.accepted_and_persisted
    )
    assert counts == (1, 1, expected_evidence_refs)
    assert events == ["receipt_persist"]


def test_research_loop_shadow_runs_after_receipt_persistence_and_is_non_authoritative() -> None:
    run_id, persisted = _receipt()
    events: list[str] = []

    class Repository:
        async def record_page(self, *_args: Any, **_kwargs: Any) -> tuple[int, int]:
            raise AssertionError("SHADOW must preserve receipt persistence")

        async def record_page_with_receipts(self, *_args: Any, **_kwargs: Any) -> Any:
            events.append("committed_receipt")
            return persisted

    class FailingShadow:
        async def process_persisted_page(self, *_args: Any) -> None:
            events.append("shadow")
            raise RuntimeError("isolated shadow failure")

    service = ResearchLoopService(
        Repository(),  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        semantic_shadow_mode=SemanticShadowMode.SHADOW,
        semantic_shadow=FailingShadow(),  # type: ignore[arg-type]
    )
    counts = _run_immediate(service._record_page_with_shadow(run_id, sentinel="value"))

    expected_evidence_refs = tuple(
        evidence.evidence_id
        for receipt in persisted.claim_receipts
        for evidence in receipt.evidence_receipts
        if evidence.accepted_and_persisted
    )
    assert counts == (
        persisted.inserted_count,
        persisted.accepted_count,
        expected_evidence_refs,
    )
    assert events == ["committed_receipt", "shadow"]


def test_one_evidence_failure_preserves_successful_sibling_result(
    monkeypatch: Any,
) -> None:
    run_id, persisted = _receipt()
    claim = persisted.claim_receipts[0]
    original_evidence = claim.evidence_receipts[0]
    sibling = replace(original_evidence, evidence_id=uuid4())
    evidence_items = tuple(
        sorted((original_evidence, sibling), key=lambda item: str(item.evidence_id))
    )
    failing_evidence, successful_evidence = evidence_items
    claim_with_sibling = replace(
        claim,
        evidence_ids=tuple(item.evidence_id for item in evidence_items),
        evidence_receipts=evidence_items,
    )
    persisted_with_sibling = replace(
        persisted,
        claim_receipts=(claim_with_sibling,),
    )
    gateway = FakeGateway()
    artifacts = MemoryArtifactStore()
    pipeline = SemanticResearchShadowPipeline(
        repository=SimpleNamespace(),
        bindings=FakeBindings(run_id),
        cipher=FakeCipher(),
        gateway=gateway,
        artifacts=artifacts,
        budget=_budget(),
    )
    original_process = pipeline._process_evidence_receipt

    async def fail_first_evidence(
        actual_run_id: Any,
        actual_claim: Any,
        actual_evidence: Any,
        decomposition: Any,
    ) -> None:
        if actual_evidence.evidence_id == failing_evidence.evidence_id:
            raise RuntimeError("isolated evidence fixture failure")
        await original_process(actual_run_id, actual_claim, actual_evidence, decomposition)

    monkeypatch.setattr(pipeline, "_process_evidence_receipt", fail_first_evidence)
    _run_immediate(pipeline.process_persisted_page(run_id, persisted_with_sibling))

    snapshot = next(
        value
        for uri, value in artifacts.values.items()
        if uri.startswith(f"runs/{run_id}/semantic-shadow/")
        and "/attempts/" not in uri
        and "/receipts/" not in uri
        and "/work-products/" not in uri
    )
    assert snapshot["evidence_refs"] == [str(successful_evidence.evidence_id)]
    assert any(
        diagnostic["evidence_ref"] == str(failing_evidence.evidence_id)
        and diagnostic["status"] == "FAILED_ISOLATED"
        for diagnostic in snapshot["diagnostics"]
    )
