from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace
from typing import Any

import httpx
from app.domain.grounding_requirements import GroundingNodeUsability
from app.domain.source_context_grounding_v2 import (
    GroundingDimensionStatus,
    GroundingStatusV2,
    SourceAssertionV2,
    SourceContextGroundingResultV2,
    SourceContextV2,
    SourceEvidenceSpan,
)
from app.llm.canary_semantic_pipeline_guard import (
    CanarySemanticPipelineGuard,
    SemanticCallPurpose,
)

from scripts.run_v2_1r_d4b_hybrid_canary import (
    EXPECTED_CLAIMS,
    EXPECTED_MANIFEST_HASH,
    _build_requirement_results,
    _CountingPostClient,
    _node_inputs,
    run_preflight,
)


def test_d4b_zero_call_preflight_rechecks_frozen_cohort_and_contract() -> None:
    result = run_preflight()

    assert result["claim_ids"] == list(EXPECTED_CLAIMS)
    assert result["selection_manifest_normalized_sha256"] == EXPECTED_MANIFEST_HASH
    assert result["claim_hashes_verified"] is True
    assert result["grounding_requirement_contract"] == "grounding-requirements.generic.v2"
    assert result["provider_calls"] == 0
    assert result["external_http_outbound"] == 0
    assert result["database_writes"] == 0
    assert result["holdout_accessed"] is False


def test_requirement_projection_does_not_invent_missing_context_binding() -> None:
    span = SourceEvidenceSpan(
        evidence_ref="E1",
        exact_text="The value was 12.",
        occurrence_index=0,
        start_offset=0,
        end_offset=18,
        claim_hash="a" * 64,
        textual_grounding_status=GroundingDimensionStatus.VALID,
    )
    extraction = SourceContextGroundingResultV2(
        claim_ref="claim-fixture",
        claim_hash="a" * 64,
        source_assertions=(
            SourceAssertionV2(
                source_assertion_ref="A1",
                evidence_refs=("E1",),
                assertion_status="ASSERTED",
            ),
        ),
        source_contexts=(
            SourceContextV2(
                context_ref="C1",
                context_type="MEASUREMENT",
                semantic_value={"metric": "rate", "value": 12, "unit": "%"},
                lexical_value="12%",
            ),
        ),
        evidence_spans=(span,),
        status=GroundingStatusV2.VALID,
    )
    assert {node.node_ref for node in _node_inputs(extraction)} == {"A1", "C1"}

    nodes, obligations = _build_requirement_results(extraction, SimpleNamespace(bindings=[]))

    context = next(node for node in nodes if node["node_ref"] == "C1")
    assert context["usability"] != GroundingNodeUsability.GROUNDED_USABLE.value
    assert any(
        row["node_ref"] == "C1"
        and row["status"] == "MISSING_REQUIRED_BINDING"
        and row["requirement_level"] == "REQUIRED_FOR_NODE_SEMANTICS"
        for row in obligations
    )


def test_guard_blocks_a_second_provider_post_before_counted_transport() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        async def post(self, url: str, *, headers: Any, json: Any) -> httpx.Response:
            self.calls += 1
            return httpx.Response(200, json={"ok": True})

    async def run() -> tuple[int, int, int]:
        guard = CanarySemanticPipelineGuard()
        fake = FakeClient()
        counted = _CountingPostClient(fake)  # type: ignore[arg-type]
        scoped = guard.client_for("claim-fixture", SemanticCallPurpose.EXTRACTOR, counted)
        await scoped.post("https://unit.test/", headers={}, json={})
        with suppress(Exception):
            await scoped.post("https://unit.test/", headers={}, json={})
        snapshot = guard.snapshot("claim-fixture")
        return fake.calls, counted.external_http_outbound_count, snapshot["blocked_call_count"]

    assert asyncio.run(run()) == (1, 1, 1)
