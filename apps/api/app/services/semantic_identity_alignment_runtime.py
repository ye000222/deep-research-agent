"""One-call, structured, fail-closed comparison-local identity alignment runtime."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from app.domain.atomic_proposition import AtomicProposition
from app.domain.providers import AdapterType, CanonicalModelRequest, ContentPart
from app.domain.semantic_identity_alignment import (
    SEMANTIC_IDENTITY_ALIGNMENT_RUNTIME_VERSION,
    SemanticAlignmentStatus,
    SemanticIdentityAlignmentFact,
    SemanticIdentityAlignmentReasonCode,
    SemanticIdentityFieldRole,
)
from app.domain.source_atomic_assertion import (
    SourceAssertionField,
    SourceAtomicAssertion,
    SourceGroundingState,
)
from app.domain.truth_condition_context import TruthConditionContext

SEMANTIC_IDENTITY_ALIGNMENT_SCHEMA = "semantic-identity-alignment-response-v1"
SEMANTIC_IDENTITY_ALIGNMENT_PROMPT = "semantic-identity-alignment.generic.v1"


class AlignmentGateway(Protocol):
    async def generate_structured(
        self,
        *,
        adapter_type: AdapterType,
        base_url: str,
        api_key: SecretStr,
        request: CanonicalModelRequest,
        allow_regeneration: bool = True,
    ) -> Any: ...


class IdentityAlignmentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pair_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    subject_relation: SemanticAlignmentStatus
    predicate_relation: SemanticAlignmentStatus
    reason_code: SemanticIdentityAlignmentReasonCode


class IdentityAlignmentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SEMANTIC_IDENTITY_ALIGNMENT_SCHEMA
    status: str = Field(pattern=r"^(RESOLVED|UNRESOLVED)$")
    alignments: tuple[IdentityAlignmentPayload, ...] = Field(max_length=128)
    unresolved_reason_codes: tuple[SemanticIdentityAlignmentReasonCode, ...] = ()

    @model_validator(mode="after")
    def validate_shape(self) -> IdentityAlignmentResponse:
        if self.status == "RESOLVED" and not self.alignments:
            raise ValueError("resolved_alignment_requires_facts")
        if self.status == "UNRESOLVED" and not self.unresolved_reason_codes:
            raise ValueError("unresolved_alignment_requires_reason")
        if self.status == "UNRESOLVED" and self.alignments:
            raise ValueError("unresolved_alignment_must_not_contain_facts")
        return self


class AlignmentRuntimeStatus(StrEnum):
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_UNKNOWN = "COMPLETED_WITH_UNKNOWN"
    INVALID = "INVALID"
    FAILED = "FAILED"
    NOT_PROCESSED_BUDGET = "NOT_PROCESSED_BUDGET"


class IdentityAlignmentRuntimeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_version: str = SEMANTIC_IDENTITY_ALIGNMENT_RUNTIME_VERSION
    status: AlignmentRuntimeStatus
    facts: tuple[SemanticIdentityAlignmentFact, ...] = ()
    reason_codes: tuple[str, ...] = ()
    model_call_count: int = Field(default=0, ge=0, le=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)


class IdentityAlignmentPair(BaseModel):
    """Only the formal pair fields needed by one claim-local comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    target: AtomicProposition
    source: SourceAtomicAssertion
    target_proposition_ref: str
    source_grounding_refs: tuple[str, ...]
    context_refs: tuple[str, ...] = ()
    target_context_payload: tuple[dict[str, Any], ...] = ()


def deterministic_pair_ref(target_ref: str, source_ref: str, version: str) -> str:
    raw = json.dumps([target_ref, source_ref, version], separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def alignment_cache_ref(pairs: tuple[IdentityAlignmentPair, ...]) -> str:
    identity = sorted((item.target_proposition_ref, item.source.assertion_ref) for item in pairs)
    return hashlib.sha256(
        json.dumps([SEMANTIC_IDENTITY_ALIGNMENT_SCHEMA, identity], separators=(",", ":")).encode()
    ).hexdigest()


def _semantic_ref(pair_ref: str, role: SemanticIdentityFieldRole, side: str) -> str:
    return hashlib.sha256(f"{pair_ref}:{role.value}:{side}".encode()).hexdigest()


def _grounding_refs(
    source: SourceAtomicAssertion, role: SemanticIdentityFieldRole
) -> tuple[str, ...]:
    field = (
        SourceAssertionField.SUBJECT
        if role is SemanticIdentityFieldRole.SUBJECT
        else SourceAssertionField.PREDICATE
    )
    grounded = next((item for item in source.field_grounding if item.field is field), None)
    if grounded is None or any(
        state is not SourceGroundingState.VALID
        for state in (
            grounded.textual_grounding,
            grounded.semantic_support,
            grounded.structural_grounding,
        )
    ):
        return ()
    return grounded.evidence_refs


def _context_payload(
    contexts: Mapping[str, TruthConditionContext], refs: tuple[str, ...]
) -> list[dict[str, Any]]:
    result = []
    for ref in sorted(set(refs)):
        item = contexts.get(ref)
        if item is not None and item.materiality != "DESCRIPTIVE":
            result.append(
                {
                    "context_ref": ref,
                    "context_type": item.context_type.value,
                    "semantic_value": item.semantic_value,
                }
            )
    return result


def build_alignment_request_payload(
    pairs: tuple[IdentityAlignmentPair, ...],
    contexts: Mapping[str, TruthConditionContext],
) -> dict[str, Any]:
    payload_pairs: list[dict[str, Any]] = []
    for pair in sorted(
        pairs, key=lambda item: (item.target_proposition_ref, item.source.assertion_ref)
    ):
        pair_ref = deterministic_pair_ref(
            pair.target_proposition_ref,
            pair.source.assertion_ref,
            SEMANTIC_IDENTITY_ALIGNMENT_SCHEMA,
        )
        target_subject = pair.target.subject
        target_predicate = pair.target.predicate
        source_subject = pair.source.subject.source_representation
        source_predicate = pair.source.predicate.source_representation
        if target_subject is None or target_predicate is None:
            continue
        payload_pairs.append(
            {
                "pair_ref": pair_ref,
                "target_proposition_ref": pair.target_proposition_ref,
                "source_assertion_ref": pair.source.assertion_ref,
                "target_subject": target_subject,
                "target_predicate": target_predicate,
                "target_typed_value": pair.target.value.model_dump(mode="json"),
                "target_contexts": list(pair.target_context_payload),
                "target_modality": pair.target.modality,
                "target_attribution": pair.target.attribution,
                "target_qualifiers": [
                    item.model_dump(mode="json") for item in pair.target.qualifiers
                ],
                "source_subject": source_subject,
                "source_predicate": source_predicate,
                "source_typed_value": pair.source.value.model_dump(mode="json"),
                "source_contexts": _context_payload(contexts, pair.context_refs),
                "source_modality_context_refs": list(pair.source.modality_context_refs),
                "source_attribution_context_refs": list(pair.source.attribution_context_refs),
                "source_grounding_refs": {
                    role.value: list(_grounding_refs(pair.source, role))
                    for role in SemanticIdentityFieldRole
                },
                "source_span_refs": list(pair.source.evidence_span_refs),
                "source_ref": pair.source.source_ref,
                "snapshot_ref": pair.source.snapshot_ref,
            }
        )
    return {"schema_version": SEMANTIC_IDENTITY_ALIGNMENT_SCHEMA, "pairs": payload_pairs}


def parse_alignment_response(
    raw: Any,
    pairs: tuple[IdentityAlignmentPair, ...],
) -> tuple[SemanticIdentityAlignmentFact, ...]:
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump(mode="json")
    response = IdentityAlignmentResponse.model_validate(raw)
    expected = {
        deterministic_pair_ref(
            item.target_proposition_ref,
            item.source.assertion_ref,
            SEMANTIC_IDENTITY_ALIGNMENT_SCHEMA,
        ): item
        for item in pairs
    }
    if response.status == "UNRESOLVED":
        returned = {
            pair_ref: IdentityAlignmentPayload(
                pair_ref=pair_ref,
                subject_relation=SemanticAlignmentStatus.UNKNOWN,
                predicate_relation=SemanticAlignmentStatus.UNKNOWN,
                reason_code=response.unresolved_reason_codes[0],
            )
            for pair_ref in expected
        }
    else:
        returned = {item.pair_ref: item for item in response.alignments}
    if len(returned) != len(response.alignments) and response.status != "UNRESOLVED":
        raise ValueError("ALIGNMENT_PAIR_SET_MISMATCH")
    if set(returned) != set(expected):
        raise ValueError("ALIGNMENT_PAIR_SET_MISMATCH")
    facts: list[SemanticIdentityAlignmentFact] = []
    for pair_ref in sorted(expected):
        pair = expected[pair_ref]
        alignment = returned[pair_ref]
        for role, status in (
            (SemanticIdentityFieldRole.SUBJECT, alignment.subject_relation),
            (SemanticIdentityFieldRole.PREDICATE, alignment.predicate_relation),
        ):
            grounded_refs = _grounding_refs(pair.source, role)
            # Provider claims cannot overcome missing/invalid source-field grounding.
            effective_status = status if grounded_refs else SemanticAlignmentStatus.UNKNOWN
            effective_reason = alignment.reason_code
            if not grounded_refs:
                effective_reason = SemanticIdentityAlignmentReasonCode.SOURCE_GROUNDING_UNAVAILABLE
            elif (
                (
                    status is SemanticAlignmentStatus.SAME
                    and effective_reason is not SemanticIdentityAlignmentReasonCode.IDENTITY_ALIGNED
                )
                or (
                    status is SemanticAlignmentStatus.DIFFERENT
                    and effective_reason
                    is not SemanticIdentityAlignmentReasonCode.IDENTITY_DISTINCT
                )
                or (
                    status is SemanticAlignmentStatus.UNKNOWN
                    and effective_reason
                    not in {
                        SemanticIdentityAlignmentReasonCode.AMBIGUOUS,
                        SemanticIdentityAlignmentReasonCode.INSUFFICIENT_CONTEXT,
                    }
                )
            ):
                raise ValueError("ALIGNMENT_REASON_STATUS_MISMATCH")
            facts.append(
                SemanticIdentityAlignmentFact(
                    alignment_ref=hashlib.sha256(
                        f"{pair_ref}:{role.value}:{effective_status.value}".encode()
                    ).hexdigest(),
                    target_proposition_ref=pair.target_proposition_ref,
                    source_assertion_ref=pair.source.assertion_ref,
                    field_role=role,
                    alignment_status=effective_status,
                    reason_code=effective_reason,
                    target_semantic_ref=_semantic_ref(pair_ref, role, "target"),
                    source_semantic_ref=_semantic_ref(pair_ref, role, "source"),
                    source_grounding_refs=grounded_refs,
                    basis_refs=tuple(
                        sorted((*grounded_refs, pair.source.canonical_evidence_span_ref))
                    ),
                )
            )
    return tuple(facts)


class SemanticIdentityAlignmentRuntime:
    def __init__(
        self,
        gateway: AlignmentGateway,
        *,
        adapter_type: AdapterType,
        base_url: str,
        api_key: SecretStr,
        model: str,
    ) -> None:
        self._gateway = gateway
        self._adapter_type = adapter_type
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._called = False

    async def align(
        self,
        pairs: tuple[IdentityAlignmentPair, ...],
        contexts: Mapping[str, TruthConditionContext],
    ) -> IdentityAlignmentRuntimeResult:
        if self._called or not pairs:
            return IdentityAlignmentRuntimeResult(
                status=AlignmentRuntimeStatus.NOT_PROCESSED_BUDGET,
                reason_codes=("ALIGNMENT_CALL_LIMIT_REACHED",),
            )
        self._called = True
        payload = build_alignment_request_payload(pairs, contexts)
        if not payload["pairs"]:
            return IdentityAlignmentRuntimeResult(
                status=AlignmentRuntimeStatus.INVALID, reason_codes=("ALIGNMENT_INPUT_UNAVAILABLE",)
            )
        request = CanonicalModelRequest(
            task_kind="semantic_identity_alignment_v1",
            role="semantic_identity_alignment_runtime_v1",
            model=self._model,
            instructions="\n".join(
                (
                    "Return only the strict structured semantic alignment response.",
                    "Judge whether each target/source subject and predicate have the same",
                    "referent or relation in the supplied context. Use DIFFERENT only when",
                    "explicit and unambiguous; otherwise use UNKNOWN.",
                    "Never infer identity from string similarity, aliases, domain knowledge,",
                    "or arbitrary canonical IDs. Preserve attribution, modality, causal,",
                    "and method distinctions. Values are context only, not alignment targets.",
                    "Return no explanation beyond closed reason codes.",
                )
            ),
            content_parts=(ContentPart(kind="json", value=payload),),
            response_contract=IdentityAlignmentResponse.model_json_schema(),
            generation_parameters={"temperature": 0.0, "reasoning_enabled": False},
            max_output_tokens=4_000,
            context_manifest_id=uuid4(),
            metadata={
                "schema_version": SEMANTIC_IDENTITY_ALIGNMENT_SCHEMA,
                "prompt_version": SEMANTIC_IDENTITY_ALIGNMENT_PROMPT,
            },
        )
        try:
            result = await self._gateway.generate_structured(
                adapter_type=self._adapter_type,
                base_url=self._base_url,
                api_key=self._api_key,
                request=request,
                allow_regeneration=False,
            )
            facts = parse_alignment_response(
                getattr(result, "parsed_object", None) or getattr(result, "text", None), pairs
            )
        except Exception:
            return IdentityAlignmentRuntimeResult(
                status=AlignmentRuntimeStatus.FAILED,
                reason_codes=("ALIGNMENT_RESPONSE_INVALID_OR_FAILED",),
                model_call_count=1,
            )
        usage = getattr(result, "usage", None)
        unknown = any(fact.alignment_status is SemanticAlignmentStatus.UNKNOWN for fact in facts)
        return IdentityAlignmentRuntimeResult(
            status=AlignmentRuntimeStatus.COMPLETED_WITH_UNKNOWN
            if unknown
            else AlignmentRuntimeStatus.COMPLETED,
            facts=facts,
            model_call_count=1,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            latency_ms=getattr(result, "latency_ms", 0) or 0,
        )
