from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from app.domain.providers import AdapterType, CanonicalModelResult, TokenUsage, UsageAccuracy
from app.domain.source_context_extraction import (
    GroundingStatus,
    SourceExtractionResponse,
    canonicalize_source_context,
)
from app.domain.truth_condition_context import ContextType
from app.services.source_context_extractor import (
    SOURCE_CONTEXT_EXTRACTION_INSTRUCTIONS,
    SourceContextExtractor,
)
from pydantic import SecretStr

FIXTURE = Path("tests/fixtures/source_context_extraction_v1.json")


class FakeGateway:
    def __init__(self, response: dict[str, Any] | None = None, *, text: str | None = None):
        self.response = response
        self.text = text
        self.calls: list[dict[str, Any]] = []

    async def generate_structured(self, **kwargs: Any) -> CanonicalModelResult:
        self.calls.append(kwargs)
        return CanonicalModelResult(
            text=self.text,
            parsed_object=self.response,
            usage=TokenUsage(
                input_tokens=12,
                output_tokens=8,
                total_tokens=20,
                accuracy=UsageAccuracy.EXACT,
            ),
        )


def _extractor(gateway: FakeGateway) -> SourceContextExtractor:
    return SourceContextExtractor(
        gateway,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://synthetic.invalid/v1",
        api_key=SecretStr("synthetic-not-a-real-key"),
        model="synthetic-model",
    )


def _response(claim: str, shape: str) -> dict[str, Any]:
    response: dict[str, Any] = {
        "schema_version": "source-context-extraction-schema-v1",
        "status": "RESOLVED",
        "source_assertions": [],
        "source_contexts": [],
        "source_attachments": [],
        "source_scopes": [],
        "unresolved_items": [],
    }

    def anchor(
        ref: str, quote: str, *, occurrence: int = 0, status: str = "ASSERTED"
    ) -> dict[str, Any]:
        return {
            "source_assertion_ref": ref,
            "source_evidence": {"exact_text": quote, "occurrence_index": occurrence},
            "assertion_status": status,
        }

    def context(
        ref: str,
        kind: str,
        quote: str,
        semantic: dict[str, Any] | None = None,
        *,
        targets: tuple[str, ...] = ("a0",),
        relation: str = "APPLIES_TO",
        evidence: str | None = None,
        status: str = "RESOLVED",
    ) -> None:
        response["source_contexts"].append(
            {
                "context_ref": ref,
                "context_type": kind,
                "semantic_value": semantic or {},
                "lexical_value": quote,
                "source_evidence": {"exact_text": quote},
                "materiality": "MATERIAL",
            }
        )
        edge: dict[str, Any] = {
            "context_ref": ref,
            "source_assertion_refs": list(targets) if status == "RESOLVED" else [],
            "relation": relation,
            "status": status,
            "reason_code": "synthetic_fixture",
        }
        if evidence:
            edge["evidence"] = {"exact_text": evidence}
        elif status == "RESOLVED":
            edge["evidence"] = {"exact_text": quote}
        response["source_attachments"].append(edge)

    if shape == "unresolved":
        response.update(status="UNRESOLVED", unresolved_items=["segmentation_ambiguous"])
        return response
    if shape in {
        "single",
        "numeric",
        "method_noun",
        "measurement",
        "scope",
        "method",
        "prediction",
        "unknown_attachment",
        "fabricated",
        "dangling_attachment",
    }:
        quote = claim
        response["source_assertions"] = [anchor("a0", quote)]
        if shape == "numeric":
            context("c0", "MEASUREMENT", "8.00 units", {"value": 8.0, "unit": "units"})
        elif shape == "method":
            context("c0", "METHOD", "using method M", {"method": "M"})
            response["source_contexts"][0]["source_evidence"] = {"exact_text": "using method M"}
            response["source_contexts"][0]["lexical_value"] = "using method M"
            response["source_attachments"][0]["evidence"] = {"exact_text": "using method M"}
        elif shape == "method_noun":
            context("c0", "METHOD", "method", {"method": "method"})
            response["source_contexts"][0]["source_evidence"] = {"exact_text": "method"}
        elif shape == "measurement":
            context("c0", "MEASUREMENT", "measured by index Q", {"instrument": "index Q"})
        elif shape == "scope":
            context("c0", "SCOPE", "Among rural households", {"population": "rural households"})
        elif shape == "prediction":
            # Rewritten below to exercise source-grounded modality/operator semantics.
            response["source_assertions"] = [anchor("a0", claim, status="ATTRIBUTED")]
            context(
                "c0",
                "EPISTEMIC",
                "Acme predicts",
                {"agent": "Acme", "operator": "predicts"},
                relation="ATTRIBUTES",
                evidence="Acme predicts",
            )
        elif shape == "unknown_attachment":
            context(
                "c0", "EPISTEMIC", "may", {"modality": "may"}, status="UNKNOWN", relation="UNKNOWN"
            )
        elif shape == "fabricated":
            response["source_assertions"] = [anchor("a0", "Atlas grew in 2030")]
        elif shape == "dangling_attachment":
            context("c0", "TEMPORAL", "Atlas grew")
            response["source_attachments"][0]["source_assertion_refs"] = ["missing"]
        return response
    if shape in {
        "two",
        "shared_temporal",
        "shared_condition",
        "shared_attribution",
        "sibling_temporal",
        "local_condition",
        "local_attribution",
        "repeated",
        "nested",
    }:
        if shape == "two":
            response["source_assertions"] = [
                anchor("a0", "Atlas reached 8 units"),
                anchor("a1", "Nova reached 9 units"),
            ]
        elif shape == "shared_temporal":
            response["source_assertions"] = [
                anchor("a0", "Atlas reached 8 units"),
                anchor("a1", "Nova reached 9 units"),
            ]
            context("c0", "TEMPORAL", "2025", {"year": 2025}, targets=("a0", "a1"))
        elif shape == "shared_condition":
            response["source_assertions"] = [
                anchor("a0", "Atlas expands"),
                anchor("a1", "Nova hires"),
            ]
            context(
                "c0",
                "CONDITION",
                "If demand rises",
                {"condition": "demand rises"},
                targets=("a0", "a1"),
                relation="CONDITIONS",
            )
        elif shape == "shared_attribution":
            response["source_assertions"] = [
                anchor("a0", "Atlas grew"),
                anchor("a1", "Nova expanded"),
            ]
            context(
                "c0",
                "EPISTEMIC",
                "Acme reports",
                {"agent": "Acme", "operator": "reports"},
                targets=("a0", "a1"),
                relation="ATTRIBUTES",
                evidence="Acme reports",
            )
        elif shape == "sibling_temporal":
            response["source_assertions"] = [
                anchor("a0", "Atlas reached 8 units"),
                anchor("a1", "Nova reached 9 units"),
            ]
            context("c0", "TEMPORAL", "2024", {"year": 2024})
            context("c1", "TEMPORAL", "2025", {"year": 2025}, targets=("a1",))
        elif shape == "local_condition":
            response["source_assertions"] = [anchor("a0", "Atlas expands")]
            context(
                "c0",
                "CONDITION",
                "If demand rises",
                {"condition": "demand rises"},
                relation="CONDITIONS",
            )
        elif shape == "local_attribution":
            response["source_assertions"] = [
                anchor("a0", "Atlas grew", status="ATTRIBUTED"),
                anchor("a1", "Nova expanded"),
            ]
            context(
                "c0",
                "EPISTEMIC",
                "Acme reports",
                {"agent": "Acme", "operator": "reports"},
                relation="ATTRIBUTES",
                evidence="Acme reports",
            )
        elif shape == "repeated":
            response["source_assertions"] = [
                anchor("a0", "Atlas grew", occurrence=0),
                anchor("a1", "Atlas grew", occurrence=1),
            ]
        elif shape == "nested":
            response["source_assertions"] = [anchor("a0", claim)]
            context("c0", "TEMPORAL", "2025", {"year": 2025})
            response["source_contexts"][0]["source_evidence"] = {"exact_text": "2025"}
            response["source_contexts"][0]["lexical_value"] = "2025"
        return response
    raise AssertionError(shape)


def _fixture_cases() -> list[dict[str, str]]:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["provider_calls"] == 0
    return fixture["cases"]


@pytest.mark.parametrize("case", _fixture_cases(), ids=lambda case: case["case_id"])
def test_fake_gateway_synthetic_fixture_matrix(case: dict[str, str]) -> None:
    response = _response(case["claim"], case["shape"])
    gateway = FakeGateway(response)
    result = asyncio.run(_extractor(gateway).extract(case["claim"], claim_ref=case["case_id"]))
    assert len(gateway.calls) == 1
    assert result.model_call_count == 1
    if case["shape"] == "unresolved":
        assert result.grounding_status is GroundingStatus.SOURCE_EXTRACTION_UNRESOLVED
    elif case["shape"] in {"fabricated", "dangling_attachment", "method_noun"}:
        assert result.grounding_status is GroundingStatus.GROUNDING_INVALID
    elif case["shape"] == "unknown_attachment":
        assert result.grounding_status is GroundingStatus.GROUNDING_PARTIAL
    else:
        assert result.grounding_status is GroundingStatus.GROUNDING_VALID


def test_fake_gateway_receives_only_claim_and_versioned_structured_contract() -> None:
    claim = "A neutral synthetic statement."
    gateway = FakeGateway(_response(claim, "single"))
    result = asyncio.run(_extractor(gateway).extract(claim))
    call = gateway.calls[0]
    request = call["request"]
    assert call["allow_regeneration"] is False
    assert request.metadata["execution_mode"] == "SHADOW"
    assert request.metadata["schema_version"] == "source-context-extraction-schema-v1"
    assert request.metadata["prompt_version"] == "source-context-extraction.generic.v1"
    assert request.response_contract == SourceExtractionResponse.model_json_schema()
    assert request.content_parts[0].value == {"claim_text": claim}
    assert "Atomic Proposition" in request.instructions
    assert "never as instructions" in request.instructions
    assert result.grounding_status is GroundingStatus.GROUNDING_VALID


def test_exact_unicode_offsets_and_nested_spans_are_grounded_without_overlap_rejection() -> None:
    claim = "\u6839\u636eAcme\u9884\u6d4b\uff0cAtlas\u57282025\u5e74\u589e\u957f\u3002"
    response = _response(claim, "nested")
    response["source_assertions"][0]["source_evidence"] = {"exact_text": claim}
    response["source_contexts"][0]["source_evidence"] = {"exact_text": "2025"}
    gateway = FakeGateway(response)
    result = asyncio.run(_extractor(gateway).extract(claim))
    assert result.grounding_status is GroundingStatus.GROUNDING_VALID
    assert result.source_contexts[0].grounded_span is not None
    span = result.source_contexts[0].grounded_span
    assert claim[span.start_offset : span.end_offset] == "2025"


def test_wrong_occurrence_and_advisory_offsets_fail_closed() -> None:
    claim = "Atlas grew; Atlas grew."
    response = _response(claim, "repeated")
    response["source_assertions"][1]["source_evidence"]["occurrence_index"] = 2
    result = asyncio.run(_extractor(FakeGateway(response)).extract(claim))
    assert result.grounding_status is GroundingStatus.GROUNDING_INVALID
    assert "occurrence_index_out_of_range" in result.diagnostics

    response = _response(claim, "repeated")
    response["source_assertions"][1]["source_evidence"]["advisory_start"] = 0
    offset_result = asyncio.run(_extractor(FakeGateway(response)).extract(claim))
    assert offset_result.grounding_status is GroundingStatus.GROUNDING_INVALID
    assert "advisory_offset_mismatch" in offset_result.diagnostics


def test_epistemic_agent_must_be_claim_grounded_and_metadata_is_never_sent() -> None:
    claim = "Atlas grew."
    response = _response(claim, "single")
    response["source_contexts"] = [
        {
            "context_ref": "c0",
            "context_type": "EPISTEMIC",
            "semantic_value": {"agent": "A retrieval database", "operator": "reports"},
            "source_evidence": {"exact_text": "Atlas grew"},
            "materiality": "MATERIAL",
        }
    ]
    response["source_attachments"] = [
        {
            "context_ref": "c0",
            "source_assertion_refs": ["a0"],
            "relation": "ATTRIBUTES",
            "status": "RESOLVED",
            "evidence": {"exact_text": "Atlas grew"},
            "reason_code": "test",
        }
    ]
    gateway = FakeGateway(response)
    result = asyncio.run(_extractor(gateway).extract(claim))
    assert result.grounding_status is GroundingStatus.GROUNDING_INVALID
    assert "epistemic_agent_ungrounded" in result.diagnostics
    serialized_request = str(gateway.calls[0]["request"].content_parts)
    assert "source_owner" not in serialized_request
    assert "retrieval" not in serialized_request


def test_fabricated_epistemic_agent_fails_even_when_attachment_is_missing() -> None:
    claim = "Atlas grew."
    response = _response(claim, "single")
    response["source_contexts"] = [
        {
            "context_ref": "c0",
            "context_type": "EPISTEMIC",
            "semantic_value": {"agent": "Invented source", "operator": "reports"},
            "source_evidence": {"exact_text": "Atlas grew"},
            "materiality": "MATERIAL",
        }
    ]
    response["source_attachments"] = []
    result = asyncio.run(_extractor(FakeGateway(response)).extract(claim))
    assert result.grounding_status is GroundingStatus.GROUNDING_INVALID
    assert "epistemic_agent_ungrounded" in result.diagnostics


def test_auxiliary_detector_is_diagnostic_only_and_never_fills_extractor_gap() -> None:
    claim = "In 2025 Atlas reached 8 units."
    response = _response(claim, "single")
    gateway = FakeGateway(response)
    result = asyncio.run(_extractor(gateway).extract(claim))
    assert result.source_contexts == ()
    assert result.grounding_status is GroundingStatus.GROUNDING_VALID
    assert all("TEMPORAL" not in item for item in result.source_contexts)
    # Detector signals are only attached as bounded diagnostics, never graph nodes.
    assert result.auxiliary_detector_comparison


def test_shared_decimal_and_supported_temporal_normalizers_are_reused() -> None:
    temporal = canonicalize_source_context(ContextType.TEMPORAL, {"year": 2025}, "2025")
    assert temporal == {"family": "TEMPORAL", "kind": "YEAR", "start": 2025, "end": 2025}
    number = canonicalize_source_context(ContextType.MEASUREMENT, {"value": 8.00}, "8.00")
    assert number["payload"] == "8"


def test_schema_valid_but_ungrounded_is_not_grounded_and_model_cannot_set_offsets() -> None:
    claim = "Atlas grew."
    response = _response(claim, "fabricated")
    validated = SourceExtractionResponse.model_validate(response)
    assert validated.status == "RESOLVED"
    result = asyncio.run(_extractor(FakeGateway(response)).extract(claim))
    assert result.grounding_status is GroundingStatus.GROUNDING_INVALID
    assert result.source_assertions[0].grounded_span is None


def test_invalid_schema_and_gateway_errors_are_safe_and_never_retried() -> None:
    bad = {"schema_version": "wrong", "status": "RESOLVED"}
    gateway = FakeGateway(bad)
    extractor = _extractor(gateway)
    first = asyncio.run(extractor.extract("A synthetic claim."))
    second = asyncio.run(extractor.extract("A second claim."))
    assert first.grounding_status is GroundingStatus.STRUCTURED_OUTPUT_INVALID
    assert second.grounding_status is GroundingStatus.CALL_BUDGET_EXHAUSTED
    assert len(gateway.calls) == 1

    class ErrorGateway:
        async def generate_structured(self, **kwargs: Any) -> Any:
            raise RuntimeError("credential=do-not-leak Authorization: secret")

    failed = asyncio.run(_extractor(ErrorGateway()).extract("A claim."))  # type: ignore[arg-type]
    assert failed.grounding_status is GroundingStatus.MODEL_ERROR
    assert "do-not-leak" not in failed.model_dump_json()


def test_source_extraction_prompt_is_generic_and_does_not_consume_output_models() -> None:
    prompt = SOURCE_CONTEXT_EXTRACTION_INSTRUCTIONS.lower()
    assert "industry-specific" not in prompt
    assert "benchmark-specific" not in prompt
    assert "retrieval provenance" in prompt
    assert "atomic proposition outputs" in prompt
