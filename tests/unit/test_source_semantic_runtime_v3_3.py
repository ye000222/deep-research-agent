from __future__ import annotations

import hashlib
from typing import Any

import pytest
from app.domain.document_evidence_context import (
    CanonicalEvidenceSpan,
    ContextWindow,
    SourceLocator,
    SourceLocatorType,
)
from app.domain.providers import AdapterType, CanonicalModelResult, TokenUsage, UsageAccuracy
from app.domain.source_atomic_assertion import (
    SourceAssertionEligibility,
    SourceAssertionField,
    SourceFieldGrounding,
    SourceGroundingState,
)
from app.services.source_semantic_runtime_v3_3 import (
    SourceSemanticExtractionV33Runtime,
    SourceSemanticRuntimeInput,
    SourceSemanticRuntimeStatus,
)
from pydantic import SecretStr


class FakeGateway:
    def __init__(self, payload: dict[str, Any] | Exception) -> None:
        self.payload = payload
        self.calls = 0
        self.request: Any = None
        self.allow_regeneration: bool | None = None

    async def generate_structured(self, **kwargs: Any) -> CanonicalModelResult:
        self.calls += 1
        self.request = kwargs["request"]
        self.allow_regeneration = kwargs["allow_regeneration"]
        if isinstance(self.payload, Exception):
            raise self.payload
        return CanonicalModelResult(
            parsed_object=self.payload,
            usage=TokenUsage(
                input_tokens=14,
                output_tokens=7,
                total_tokens=21,
                accuracy=UsageAccuracy.EXACT,
            ),
        )


def _run_immediate(coroutine: Any) -> Any:
    """Drive the no-I/O Fake Gateway coroutine without creating an event loop."""
    try:
        coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    raise AssertionError("Fake Gateway unexpectedly suspended on external work")


def _canonical_input(text: str = "Acme reported 42 units.") -> SourceSemanticRuntimeInput:
    snapshot = hashlib.sha256(b"snapshot").hexdigest()
    document = hashlib.sha256(b"document").hexdigest()
    span_ref = hashlib.sha256(b"span").hexdigest()
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    span = CanonicalEvidenceSpan(
        span_ref=span_ref,
        document_ref=document,
        source_ref="source:example",
        snapshot_ref=snapshot,
        char_start=0,
        char_end=len(text),
        exact_text=text,
        content_sha256=content_hash,
        source_locator=SourceLocator(locator_type=SourceLocatorType.UNKNOWN),
    )
    window = ContextWindow(
        context_ref=hashlib.sha256(b"window").hexdigest(),
        document_ref=document,
        window_start=0,
        window_end=len(text),
        evidence_span_ref=span_ref,
        text=text,
    )
    return SourceSemanticRuntimeInput(
        source_ref="source:example",
        snapshot_ref=snapshot,
        canonical_evidence_span=span,
        context_window=window,
    )


def _response(*, unresolved_identity: bool = False) -> dict[str, Any]:
    identity = (
        {"status": "UNRESOLVED"}
        if unresolved_identity
        else {
            "status": "RESOLVED",
            "key": "entity:acme",
            "namespace": "test",
            "normalization_version": "test-v1",
            "basis": "EXPLICIT_CANONICAL_REFERENCE",
            "provenance_ref": "ref:acme",
        }
    )
    return {
        "schema_version": "source-semantic-runtime-response-v1",
        "status": "RESOLVED",
        "source_extraction": {
            "schema_version": "source-context-extraction-schema-v3.3",
            "status": "RESOLVED",
            "source_assertions": [
                {
                    "source_assertion_ref": "A1",
                    "evidence_refs": ["e1"],
                    "assertion_status": "ASSERTED",
                }
            ],
            "source_contexts": [
                {
                    "context_ref": "T1",
                    "context_type": "EPISTEMIC",
                    "semantic_value": {"operator": "REPORTED", "agent": "Acme"},
                }
            ],
            "evidence_spans": [
                {"evidence_ref": "e1", "exact_text": "Acme reported 42 units."}
            ],
            "evidence_bindings": [
                {
                    "binding_ref": "b-subject",
                    "evidence_ref": "e1",
                    "semantic_role": "ASSERTION_EVIDENCE",
                    "target_kind": "ASSERTION",
                    "target_ref": "A1",
                },
                {
                    "binding_ref": "b-predicate",
                    "evidence_ref": "e1",
                    "semantic_role": "RELATION_OPERATOR",
                    "target_kind": "ASSERTION",
                    "target_ref": "A1",
                },
                {
                    "binding_ref": "b-value",
                    "evidence_ref": "e1",
                    "semantic_role": "ASSERTION_EVIDENCE",
                    "target_kind": "ASSERTION",
                    "target_ref": "A1",
                },
                {
                    "binding_ref": "b-polarity",
                    "evidence_ref": "e1",
                    "semantic_role": "ASSERTION_EVIDENCE",
                    "target_kind": "ASSERTION",
                    "target_ref": "A1",
                },
                {
                    "binding_ref": "b-attribution",
                    "evidence_ref": "e1",
                    "semantic_role": "EPISTEMIC_AGENT",
                    "target_kind": "CONTEXT",
                    "target_ref": "T1",
                },
                {
                    "binding_ref": "b-modality",
                    "evidence_ref": "e1",
                    "semantic_role": "MODALITY",
                    "target_kind": "CONTEXT",
                    "target_ref": "T1",
                },
            ],
        },
        "assertion_candidates": [
            {
                "local_ref": "c1",
                "source_assertion_ref": "A1",
                "evidence_refs": ["e1"],
                "subject": {
                    "source_representation": "Acme",
                    "semantic_identity": identity,
                },
                "predicate": {
                    "source_representation": "reported",
                    "semantic_identity": identity,
                },
                "value": {"kind": "text", "text": "42 units"},
                "polarity": "asserted",
                "field_evidence_refs": {
                    "SUBJECT": ["e1"],
                    "PREDICATE": ["e1"],
                    "VALUE": ["e1"],
                    "POLARITY": ["e1"],
                    "ATTRIBUTION": ["e1"],
                    "MODALITY": ["e1"],
                },
                "truth_condition_context_refs": ["T1"],
                "verification_eligibility": "INDEPENDENTLY_ASSERTED",
            }
        ],
        "truth_condition_contexts": [
            {
                "context_ref": "T1",
                "semantic_signature": "pending",
                "context_type": "EPISTEMIC",
                "semantic_value": {"operator": "REPORTED", "agent": "Acme"},
                "lexical_value": "Acme reported",
                "source_span": "Acme reported",
                "source_start": 0,
                "source_end": 13,
                "materiality": "MATERIAL",
                "scope": {"kind": "PROPOSITION", "target_refs": ["c1"]},
                "attachments": [
                    {
                        "context_ref": "T1",
                        "target_ref": "c1",
                        "target_kind": "ASSERTION",
                        "relation": "ATTRIBUTES",
                        "status": "RESOLVED",
                        "reason_code": "fixture",
                    }
                ],
                "origin": "SOURCE",
            }
        ],
    }


def _runtime(fake: FakeGateway) -> SourceSemanticExtractionV33Runtime:
    return SourceSemanticExtractionV33Runtime(
        fake,
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://example.invalid/v1",
        api_key=SecretStr("fake-secret"),
        model="fake-model",
    )


def test_fake_gateway_runs_complete_v33_parser_and_builds_source_assertion() -> None:
    fake = FakeGateway(_response())
    runtime_input = _canonical_input()
    original_input = runtime_input.model_dump(mode="json")
    result = _run_immediate(_runtime(fake).extract(runtime_input))

    assert fake.calls == 1
    assert result.model_call_count == 1
    assert result.runtime_status == SourceSemanticRuntimeStatus.COMPLETED
    assert len(result.assertions) == 1
    assertion = result.assertions[0]
    assert assertion.source_ref == "source:example"
    assert (
        assertion.canonical_evidence_span_ref
        == _canonical_input().canonical_evidence_span.span_ref
    )
    assert assertion.evidence_span_refs == ("e1",)
    assert assertion.subject.source_representation == "Acme"
    assert assertion.predicate.source_representation == "reported"
    assert assertion.value.text == "42 units"
    assert assertion.polarity.value == "asserted"
    assert assertion.truth_condition_context_refs == ("T1",)
    assert assertion.attribution_context_refs == ("T1",)
    assert assertion.modality_context_refs == ("T1",)
    subject_grounding = next(
        item for item in assertion.field_grounding if item.field is SourceAssertionField.SUBJECT
    )
    assert subject_grounding.textual_grounding is SourceGroundingState.VALID
    assert subject_grounding.semantic_support is SourceGroundingState.VALID
    assert len(result.anchors) == 1
    assert len(result.semantic_evidence_bindings) == 6
    assert result.input_tokens == 14 and result.output_tokens == 7
    assert fake.allow_regeneration is False
    assert runtime_input.model_dump(mode="json") == original_input
    assert fake.request.metadata["source_schema_version"] == "source-context-extraction-schema-v3.3"
    assert (
        fake.request.metadata["source_prompt_version"]
        == "source-context-extraction.generic.v3.3"
    )


@pytest.mark.parametrize(
    ("claim_text", "predicate", "polarity"),
    (
        ("Acme reported 42 units.", "reported", "asserted"),
        ("Acme did not report 42 units.", "did not report", "negated"),
    ),
)
def test_polarity_grounding_uses_assertion_and_relation_basis(
    claim_text: str, predicate: str, polarity: str
) -> None:
    payload = _response()
    payload["source_extraction"]["evidence_spans"][0]["exact_text"] = claim_text
    payload["assertion_candidates"][0]["predicate"]["source_representation"] = predicate
    payload["assertion_candidates"][0]["polarity"] = polarity
    fake = FakeGateway(payload)

    result = _run_immediate(_runtime(fake).extract(_canonical_input(claim_text)))

    assert result.runtime_status is SourceSemanticRuntimeStatus.COMPLETED
    grounding = next(
        item
        for item in result.assertions[0].field_grounding
        if item.field is SourceAssertionField.POLARITY
    )
    assert grounding.textual_grounding is SourceGroundingState.VALID
    assert grounding.semantic_support is SourceGroundingState.VALID
    assert grounding.structural_grounding is SourceGroundingState.VALID
    assert "ASSERTION_LEVEL_POLARITY_BASIS" in grounding.reason_codes


def test_unknown_polarity_remains_ungrounded() -> None:
    payload = _response()
    payload["assertion_candidates"][0]["polarity"] = "unknown"
    result = _run_immediate(_runtime(FakeGateway(payload)).extract(_canonical_input()))

    grounding = next(
        item
        for item in result.assertions[0].field_grounding
        if item.field is SourceAssertionField.POLARITY
    )
    assert grounding.textual_grounding is SourceGroundingState.UNKNOWN
    assert grounding.semantic_support is SourceGroundingState.UNKNOWN
    assert grounding.structural_grounding is SourceGroundingState.VALID


def test_negative_polarity_without_explicit_predicate_basis_remains_unknown() -> None:
    payload = _response()
    payload["source_extraction"]["evidence_spans"][0]["exact_text"] = (
        "Acme did not report 42 units."
    )
    payload["assertion_candidates"][0]["polarity"] = "negated"
    # The predicate binding points only to the affirmative relation label.
    # A distant or otherwise unbound negation must not validate polarity.
    result = _run_immediate(
        _runtime(FakeGateway(payload)).extract(
            _canonical_input("Acme did not report 42 units.")
        )
    )

    grounding = next(
        item
        for item in result.assertions[0].field_grounding
        if item.field is SourceAssertionField.POLARITY
    )
    assert grounding.textual_grounding is SourceGroundingState.UNKNOWN
    assert grounding.semantic_support is SourceGroundingState.UNKNOWN


def test_polarity_wrong_nonoverlapping_span_fails_closed() -> None:
    claim_text = "Acme reported 42 units. unrelated note."
    payload = _response()
    extraction = payload["source_extraction"]
    extraction["source_assertions"][0]["evidence_refs"].append("e2")
    extraction["evidence_spans"].append(
        {"evidence_ref": "e2", "exact_text": "unrelated note"}
    )
    extraction["evidence_bindings"].append(
        {
            "binding_ref": "b-polarity-unrelated",
            "evidence_ref": "e2",
            "semantic_role": "ASSERTION_EVIDENCE",
            "target_kind": "ASSERTION",
            "target_ref": "A1",
        }
    )
    candidate = payload["assertion_candidates"][0]
    candidate["evidence_refs"].append("e2")
    candidate["field_evidence_refs"]["POLARITY"] = ["e2"]

    result = _run_immediate(_runtime(FakeGateway(payload)).extract(_canonical_input(claim_text)))

    assert result.runtime_status is SourceSemanticRuntimeStatus.COMPLETED_WITH_UNRESOLVED
    grounding = next(
        item
        for item in result.assertions[0].field_grounding
        if item.field is SourceAssertionField.POLARITY
    )
    assert grounding.textual_grounding is SourceGroundingState.UNKNOWN
    assert grounding.semantic_support is SourceGroundingState.UNKNOWN
    assert grounding.structural_grounding is SourceGroundingState.VALID


def test_unresolved_semantic_identity_is_retained_not_rejected() -> None:
    fake = FakeGateway(_response(unresolved_identity=True))
    result = _run_immediate(_runtime(fake).extract(_canonical_input()))
    assert result.runtime_status == SourceSemanticRuntimeStatus.COMPLETED_WITH_UNRESOLVED
    assert result.assertions[0].subject.semantic_identity.status.value == "UNRESOLVED"


def test_second_outbound_is_blocked_before_gateway() -> None:
    fake = FakeGateway(_response())
    runtime = _runtime(fake)
    runtime_input = _canonical_input()
    _run_immediate(runtime.extract(runtime_input))
    second = _run_immediate(runtime.extract(runtime_input))
    assert fake.calls == 1
    assert second.runtime_status == SourceSemanticRuntimeStatus.NOT_PROCESSED
    assert "CALL_LIMIT_REACHED" in second.reason_codes


def test_gateway_exception_is_failed_without_retry() -> None:
    fake = FakeGateway(RuntimeError("private detail"))
    result = _run_immediate(_runtime(fake).extract(_canonical_input()))
    assert fake.calls == 1
    assert result.runtime_status == SourceSemanticRuntimeStatus.FAILED
    assert result.reason_codes == ("GATEWAY_ERROR",)


def test_malformed_structured_output_is_invalid_not_empty() -> None:
    fake = FakeGateway({"unexpected": True})
    result = _run_immediate(_runtime(fake).extract(_canonical_input()))
    assert fake.calls == 1
    assert result.runtime_status == SourceSemanticRuntimeStatus.INVALID
    assert result.reason_codes == ("SCHEMA_INVALID",)


def test_valid_empty_extraction_is_empty_not_failed() -> None:
    payload = {
        "schema_version": "source-semantic-runtime-response-v1",
        "status": "UNRESOLVED",
        "unresolved_items": ["NO_SOURCE_ASSERTION_FORMED"],
        "source_extraction": {
            "schema_version": "source-context-extraction-schema-v3.3",
            "status": "UNRESOLVED",
            "unresolved_items": ["NO_ASSERTION"],
        },
    }
    fake = FakeGateway(payload)
    result = _run_immediate(_runtime(fake).extract(_canonical_input()))
    assert fake.calls == 1
    assert result.runtime_status == SourceSemanticRuntimeStatus.EMPTY
    assert result.reason_codes == ("SOURCE_SEMANTICS_EXTRACTED",)


def test_measurement_admissibility_is_invoked_on_v33_response() -> None:
    payload = _response()
    source = payload["source_extraction"]
    payload["truth_condition_contexts"] = []
    candidate = payload["assertion_candidates"][0]
    candidate["truth_condition_context_refs"] = []
    candidate["field_evidence_refs"].pop("ATTRIBUTION")
    candidate["field_evidence_refs"].pop("MODALITY")
    source["source_contexts"] = []
    source["evidence_bindings"] = [
        item
        for item in source["evidence_bindings"]
        if item["target_kind"] != "CONTEXT"
    ]
    source["evidence_spans"][0]["exact_text"] = "AUC is 99.6%."
    source["source_contexts"] = [
        {
            "context_ref": "M1",
            "context_type": "MEASUREMENT",
            "semantic_value": {
                "measurement_identity": {
                    "name": "AUC",
                    "identity_type": "SCORE_OR_INDEX",
                    "source_span": {"evidence_ref": "e2"},
                },
                "observed_value": {
                    "kind": "DECIMAL",
                    "raw_value": "99.6",
                    "value": "99.6",
                },
            },
        }
    ]
    source["evidence_spans"].extend(
        [
            {"evidence_ref": "e2", "exact_text": "AUC"},
            {"evidence_ref": "e3", "exact_text": "99.6%"},
        ]
    )
    source["evidence_bindings"].extend(
        [
            {
                "binding_ref": "b-measurement-identity",
                "evidence_ref": "e2",
                "semantic_role": "MEASUREMENT_IDENTITY",
                "target_kind": "CONTEXT",
                "target_ref": "M1",
            },
            {
                "binding_ref": "b-measurement-value",
                "evidence_ref": "e3",
                "semantic_role": "MEASUREMENT_VALUE",
                "target_kind": "CONTEXT",
                "target_ref": "M1",
            },
        ]
    )
    fake = FakeGateway(payload)
    result = _run_immediate(_runtime(fake).extract(_canonical_input("AUC is 99.6%.")))
    assert fake.calls == 1
    assert result.measurement_admissibility["M1"].value == "ADMISSIBLE"


def test_one_assertion_can_reference_multiple_exact_source_spans() -> None:
    payload = _response()
    source = payload["source_extraction"]
    source["evidence_spans"].append({"evidence_ref": "e2", "exact_text": "reported"})
    source["source_assertions"][0]["evidence_refs"].append("e2")
    payload["assertion_candidates"][0]["evidence_refs"].append("e2")
    result = _run_immediate(
        _runtime(FakeGateway(payload)).extract(_canonical_input())
    )
    assert result.assertions[0].source_evidence_refs == ("e1", "e2")
    assert len(result.anchors) == 2


def test_one_span_can_form_multiple_assertions_and_deduplicate_exact_duplicates() -> None:
    payload = _response()
    payload["truth_condition_contexts"] = []
    source = payload["source_extraction"]
    source["source_contexts"] = []
    source["evidence_bindings"] = [
        binding for binding in source["evidence_bindings"] if binding["target_kind"] != "CONTEXT"
    ]
    first = payload["assertion_candidates"][0]
    first["truth_condition_context_refs"] = []
    first["field_evidence_refs"].pop("ATTRIBUTION")
    first["field_evidence_refs"].pop("MODALITY")
    duplicate = {**first, "local_ref": "c2"}
    payload["assertion_candidates"].append(duplicate)
    duplicate_result = _run_immediate(
        _runtime(FakeGateway(payload)).extract(_canonical_input())
    )
    assert len(duplicate_result.assertions) == 1

    distinct_payload = _response()
    distinct_payload["truth_condition_contexts"] = []
    distinct_source = distinct_payload["source_extraction"]
    distinct_source["source_contexts"] = []
    distinct_source["evidence_bindings"] = [
        binding
        for binding in distinct_source["evidence_bindings"]
        if binding["target_kind"] != "CONTEXT"
    ]
    first = distinct_payload["assertion_candidates"][0]
    first["truth_condition_context_refs"] = []
    first["field_evidence_refs"].pop("ATTRIBUTION")
    first["field_evidence_refs"].pop("MODALITY")
    second = {
        **first,
        "local_ref": "c2",
        "source_assertion_ref": "A2",
        "value": {"kind": "text", "text": "43 units"},
    }
    distinct_payload["assertion_candidates"].append(second)
    distinct_source["source_assertions"].append(
        {"source_assertion_ref": "A2", "evidence_refs": ["e1"], "assertion_status": "ASSERTED"}
    )
    for binding in tuple(distinct_source["evidence_bindings"]):
        if binding["target_ref"] == "A1":
            distinct_source["evidence_bindings"].append(
                {**binding, "binding_ref": f"{binding['binding_ref']}-2", "target_ref": "A2"}
            )
    distinct_source["evidence_spans"][0]["exact_text"] = "Acme reported 42 units and 43 units."
    distinct = _run_immediate(
        _runtime(FakeGateway(distinct_payload)).extract(
            _canonical_input("Acme reported 42 units and 43 units.")
        )
    )
    assert len(distinct.assertions) == 2


def test_same_text_at_different_canonical_positions_is_not_deduplicated() -> None:
    payload = _response()
    payload["truth_condition_contexts"] = []
    source = payload["source_extraction"]
    source["source_contexts"] = []
    source["evidence_bindings"] = [
        binding for binding in source["evidence_bindings"] if binding["target_kind"] != "CONTEXT"
    ]
    first = payload["assertion_candidates"][0]
    first["truth_condition_context_refs"] = []
    first["field_evidence_refs"].pop("ATTRIBUTION")
    first["field_evidence_refs"].pop("MODALITY")
    quote = "Acme reported 42 units."
    canonical_text = f"{quote} {quote}"
    source["evidence_spans"][0]["exact_text"] = quote
    source["source_assertions"][0]["evidence_refs"] = ["e1"]
    source["evidence_spans"].append(
        {"evidence_ref": "e2", "exact_text": quote, "occurrence_index": 1}
    )
    source["source_assertions"].append(
        {"source_assertion_ref": "A2", "evidence_refs": ["e2"], "assertion_status": "ASSERTED"}
    )
    second = {
        **first,
        "local_ref": "c2",
        "source_assertion_ref": "A2",
        "evidence_refs": ["e2"],
        "field_evidence_refs": {key: ["e2"] for key in first["field_evidence_refs"]},
    }
    payload["assertion_candidates"].append(second)
    for binding in tuple(source["evidence_bindings"]):
        if binding["target_ref"] == "A1":
            source["evidence_bindings"].append(
                {
                    **binding,
                    "binding_ref": f"{binding['binding_ref']}-2",
                    "evidence_ref": "e2",
                    "target_ref": "A2",
                }
            )
    result = _run_immediate(
        _runtime(FakeGateway(payload)).extract(_canonical_input(canonical_text))
    )
    assert len(result.assertions) == 2
    assert {item.source_evidence_refs for item in result.assertions} == {("e1",), ("e2",)}


def test_structured_or_composition_is_resolved_to_source_assertion_refs() -> None:
    payload = _response()
    payload["truth_condition_contexts"] = []
    source = payload["source_extraction"]
    source["source_contexts"] = []
    source["evidence_bindings"] = [
        binding for binding in source["evidence_bindings"] if binding["target_kind"] != "CONTEXT"
    ]
    first = payload["assertion_candidates"][0]
    first["truth_condition_context_refs"] = []
    first["field_evidence_refs"].pop("ATTRIBUTION")
    first["field_evidence_refs"].pop("MODALITY")
    second = {
        **first,
        "local_ref": "c2",
        "source_assertion_ref": "A2",
        "value": {"kind": "text", "text": "another alternative"},
    }
    payload["assertion_candidates"].append(second)
    source["source_assertions"].append(
        {"source_assertion_ref": "A2", "evidence_refs": ["e1"], "assertion_status": "ASSERTED"}
    )
    for binding in tuple(source["evidence_bindings"]):
        if binding["target_ref"] == "A1":
            source["evidence_bindings"].append(
                {**binding, "binding_ref": f"{binding['binding_ref']}-2", "target_ref": "A2"}
            )
    payload["composition"] = {
        "root_members": [{"position": 0, "group_ref": "g1"}],
        "groups": [
            {
                "group_ref": "g1",
                "operator": "OR",
                "members": [
                    {
                        "position": 0,
                        "assertion_ref": "c1",
                        "eligibility": "INDEPENDENTLY_ASSERTED",
                    },
                    {
                        "position": 1,
                        "assertion_ref": "c2",
                        "eligibility": "INDEPENDENTLY_ASSERTED",
                    },
                ],
            }
        ],
    }
    result = _run_immediate(_runtime(FakeGateway(payload)).extract(_canonical_input()))
    assert len(result.assertions) == 2
    assert all(
        item.verification_eligibility is SourceAssertionEligibility.NOT_INDEPENDENTLY_ASSERTED
        for item in result.assertions
    )
    assert all(
        member.eligibility is SourceAssertionEligibility.NOT_INDEPENDENTLY_ASSERTED
        for member in result.composition.groups[0].members
    )


def test_wrong_snapshot_fails_before_gateway_call() -> None:
    payload = _canonical_input().model_dump(mode="python")
    payload["snapshot_ref"] = "0" * 64
    with pytest.raises(ValueError, match="canonical_span_source_or_snapshot_mismatch"):
        SourceSemanticRuntimeInput.model_validate(payload)


def test_context_window_is_not_promoted_to_source_evidence_span() -> None:
    complete = "Acme reported 42 units."
    evidence = "Acme reported"
    original = _canonical_input(complete)
    snapshot = original.snapshot_ref
    span = CanonicalEvidenceSpan(
        span_ref=hashlib.sha256(b"short-canonical-span").hexdigest(),
        document_ref=original.canonical_evidence_span.document_ref,
        source_ref=original.source_ref,
        snapshot_ref=snapshot,
        char_start=0,
        char_end=len(evidence),
        exact_text=evidence,
        content_sha256=hashlib.sha256(evidence.encode()).hexdigest(),
        source_locator=SourceLocator(locator_type=SourceLocatorType.UNKNOWN),
    )
    runtime_input = SourceSemanticRuntimeInput(
        source_ref=original.source_ref,
        snapshot_ref=snapshot,
        canonical_evidence_span=span,
        context_window=original.context_window.model_copy(
            update={"evidence_span_ref": span.span_ref}
        ),
    )
    payload = _response()
    payload["source_extraction"]["evidence_spans"][0]["exact_text"] = evidence
    payload["assertion_candidates"][0]["value"] = {"kind": "text", "text": "reported"}
    fake = FakeGateway(payload)
    result = _run_immediate(_runtime(fake).extract(runtime_input))
    sent = fake.request.content_parts[0].value
    assert sent["canonical_span"]["exact_text"] == evidence
    assert sent["context_window"]["text"] == complete
    assert result.source_evidence_spans[0].exact_text == evidence


def test_or_and_xor_unknown_composition_do_not_leak_independent_leaf_eligibility() -> None:
    from app.domain.atomic_proposition import PropositionValue
    from app.domain.source_atomic_assertion import (
        SourceAssertionComposition,
        SourceAssertionCompositionGroup,
        SourceAssertionCompositionMember,
        SourceAtomicAssertion,
        SourceSemanticTerm,
    )
    from app.services.source_semantic_runtime_v3_3 import _apply_composition_eligibility

    def assertion(ref: str) -> SourceAtomicAssertion:
        source = SourceSemanticTerm(source_representation=ref)
        grounding = tuple(
            SourceFieldGrounding(
                field=field,
                evidence_refs=("e1",),
                textual_grounding="VALID",
                semantic_support="VALID",
                structural_grounding="VALID",
            )
            for field in sorted(SourceAssertionField, key=lambda item: item.value)
        )
        return SourceAtomicAssertion(
            assertion_ref=ref * 64,
            source_ref="s",
            snapshot_ref="a" * 64,
            canonical_evidence_span_ref="b" * 64,
            source_assertion_anchor_refs=("A1",),
            evidence_span_refs=("e1",),
            source_evidence_refs=("e1",),
            subject=source,
            predicate=source,
            value=PropositionValue(kind="unknown"),
            polarity="asserted",
            verification_eligibility="INDEPENDENTLY_ASSERTED",
            field_grounding=grounding,
        )

    for operator, expected in (
        ("AND", SourceAssertionEligibility.INDEPENDENTLY_ASSERTED),
        ("OR", SourceAssertionEligibility.NOT_INDEPENDENTLY_ASSERTED),
        ("XOR", SourceAssertionEligibility.NOT_INDEPENDENTLY_ASSERTED),
        ("UNKNOWN", SourceAssertionEligibility.UNKNOWN),
    ):
        composition = SourceAssertionComposition(
            root_members=(SourceAssertionCompositionMember(position=0, group_ref="g"),),
            groups=(
                SourceAssertionCompositionGroup(
                    group_ref="g",
                    operator=operator,
                    members=(
                        SourceAssertionCompositionMember(
                            position=0, assertion_ref="1" * 64, eligibility="INDEPENDENTLY_ASSERTED"
                        ),
                        SourceAssertionCompositionMember(
                            position=1, assertion_ref="2" * 64, eligibility="INDEPENDENTLY_ASSERTED"
                        ),
                    ),
                ),
            ),
        )
        result = _apply_composition_eligibility([assertion("1"), assertion("2")], composition)
        assert all(item.verification_eligibility is expected for item in result)
