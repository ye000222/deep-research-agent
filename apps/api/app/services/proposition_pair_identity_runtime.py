"""Bounded structured AP/AP alignment, independent of source identity alignment."""

from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import combinations, islice
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from app.domain.proposition_pair_identity import (
    PAIR_IDENTITY_PROMPT,
    PAIR_IDENTITY_RUNTIME,
    PAIR_IDENTITY_SCHEMA,
    PairFieldRole,
    PairIdentityReason,
    PairIdentityStatus,
    PropositionPairBinding,
    PropositionPairInput,
    PropositionPairSemanticAlignmentFact,
    PropositionPairTarget,
    bind_pair,
    content_hash,
    project_pair_views,
)
from app.domain.providers import AdapterType, CanonicalModelRequest, ContentPart
from app.services.semantic_identity_alignment_runtime import AlignmentGateway


class PairJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pair_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    left_proposition_ref: UUID
    right_proposition_ref: UUID
    field_role: PairFieldRole
    alignment_status: PairIdentityStatus
    reason_code: PairIdentityReason


class PairAlignmentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["proposition-pair-identity-v1"] = PAIR_IDENTITY_SCHEMA
    run_ref: UUID
    judgments: tuple[PairJudgment, ...] = Field(max_length=128)


def request_payload(pairs: tuple[PropositionPairInput, ...]) -> dict[str, Any]:
    return {
        "schema_version": PAIR_IDENTITY_SCHEMA,
        "run_ref": str(pairs[0].binding.run_ref) if pairs else None,
        "pairs": [item.provider_payload() for item in pairs],
    }


def parse_pair_response(
    raw: Any, pairs: tuple[PropositionPairInput, ...]
) -> tuple[PropositionPairSemanticAlignmentFact, ...]:
    response = PairAlignmentResponse.model_validate(raw)
    expected = {pair.binding.pair_ref: pair for pair in pairs}
    if not expected or len(expected) != len(pairs):
        raise ValueError("PAIR_REQUEST_SET_INVALID")
    if any(pair.binding.run_ref != response.run_ref for pair in pairs):
        raise ValueError("PAIR_RESPONSE_RUN_MISMATCH")
    seen: set[tuple[str, PairFieldRole]] = set()
    facts: list[PropositionPairSemanticAlignmentFact] = []
    for judgment in response.judgments:
        pair = expected.get(judgment.pair_ref)
        if pair is None or (
            judgment.left_proposition_ref != pair.binding.left_proposition_ref
            or judgment.right_proposition_ref != pair.binding.right_proposition_ref
        ):
            raise ValueError("PAIR_RESPONSE_REF_MISMATCH")
        key = (judgment.pair_ref, judgment.field_role)
        if key in seen:
            raise ValueError("PAIR_RESPONSE_DUPLICATE_FIELD")
        seen.add(key)
        facts.append(
            PropositionPairSemanticAlignmentFact(
                alignment_ref=content_hash(
                    [
                        pair.binding.model_dump(mode="json"),
                        judgment.field_role.value,
                        judgment.alignment_status.value,
                        judgment.reason_code.value,
                    ]
                ),
                binding=pair.binding,
                field_role=judgment.field_role,
                alignment_status=judgment.alignment_status,
                reason_code=judgment.reason_code,
            )
        )
    if seen != {(ref, role) for ref in expected for role in PairFieldRole}:
        raise ValueError("PAIR_RESPONSE_FIELD_SET_MISMATCH")
    return tuple(sorted(facts, key=lambda item: (item.binding.pair_ref, item.field_role.value)))


class PairCandidatePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    pairs: tuple[PropositionPairInput, ...] = ()
    batches: tuple[tuple[PropositionPairInput, ...], ...] = ()
    enumerated_pair_count: int = 0
    oversized_pair_refs: tuple[str, ...] = ()


def plan_proposition_pairs(
    run_ref: UUID,
    targets: Sequence[PropositionPairTarget],
    *,
    max_pairs: int,
    batch_size: int,
    max_request_bytes: int,
) -> PairCandidatePlan:
    """Bound enumeration itself; never materialize an unbounded Cartesian product."""
    if max_pairs <= 0 or batch_size <= 0 or max_request_bytes <= 0:
        return PairCandidatePlan()
    if batch_size > 64:
        raise ValueError("PAIR_BATCH_EXCEEDS_RESPONSE_CONTRACT")
    eligible = sorted(
        (item for item in targets if item.eligible),
        key=lambda item: str(item.proposition.proposition_id),
    )
    if len({item.proposition.proposition_id for item in eligible}) != len(eligible):
        raise ValueError("DUPLICATE_PAIR_TARGET")
    pairs: list[PropositionPairInput] = []
    batches: list[tuple[PropositionPairInput, ...]] = []
    pending: tuple[PropositionPairInput, ...] = ()
    oversized: list[str] = []
    enumerated = 0
    for left, right in islice(combinations(eligible, 2), max_pairs):
        enumerated += 1
        if not any(a != b for a in left.claim_refs for b in right.claim_refs):
            continue
        pair = PropositionPairInput(binding=bind_pair(run_ref, left, right), left=left, right=right)
        proposed = (*pending, pair)
        if pending and (len(proposed) > batch_size or _request_size(proposed) > max_request_bytes):
            batches.append(pending)
            pending = ()
        if _request_size((pair,)) > max_request_bytes:
            oversized.append(pair.binding.pair_ref)
            continue
        pending = (*pending, pair)
        pairs.append(pair)
    if pending:
        batches.append(pending)
    return PairCandidatePlan(
        pairs=tuple(pairs),
        batches=tuple(batches),
        enumerated_pair_count=enumerated,
        oversized_pair_refs=tuple(oversized),
    )


def _request_size(pairs: tuple[PropositionPairInput, ...]) -> int:
    return len(
        json.dumps(
            request_payload(pairs),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    )


class PairAlignmentRuntimeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_version: Literal["proposition-pair-identity-runtime-v1"] = PAIR_IDENTITY_RUNTIME
    status: Literal["COMPLETED", "INVALID", "FAILED"]
    bindings: tuple[PropositionPairBinding, ...]
    facts: tuple[PropositionPairSemanticAlignmentFact, ...] = ()
    model_call_count: int = Field(ge=0, le=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_facts(self) -> PairAlignmentRuntimeResult:
        expected = {(binding.pair_ref, role) for binding in self.bindings for role in PairFieldRole}
        actual = {(item.binding.pair_ref, item.field_role) for item in self.facts}
        if self.status == "COMPLETED":
            if len(self.facts) != len(expected) or actual != expected:
                raise ValueError("PAIR_PRODUCT_FIELD_SET_MISMATCH")
            by_ref = {item.pair_ref: item for item in self.bindings}
            if any(item.binding != by_ref[item.binding.pair_ref] for item in self.facts):
                raise ValueError("PAIR_PRODUCT_BINDING_MISMATCH")
        elif self.facts:
            raise ValueError("FAILED_PAIR_PRODUCT_CANNOT_HAVE_FACTS")
        return self


def validate_pair_product(
    result: PairAlignmentRuntimeResult, pairs: tuple[PropositionPairInput, ...]
) -> None:
    if result.bindings != tuple(pair.binding for pair in pairs):
        raise ValueError("PAIR_PRODUCT_INPUT_MISMATCH")
    if result.request_hash != content_hash(request_payload(pairs)):
        raise ValueError("PAIR_PRODUCT_REQUEST_MISMATCH")
    if result.status == "COMPLETED":
        for pair in pairs:
            project_pair_views(
                pair, tuple(fact for fact in result.facts if fact.binding == pair.binding)
            )


class PropositionPairSemanticAlignmentRuntime:
    def __init__(
        self,
        gateway: AlignmentGateway,
        *,
        adapter_type: AdapterType,
        base_url: str,
        api_key: SecretStr,
        model: str,
        max_request_bytes: int,
    ) -> None:
        self._gateway = gateway
        self._adapter_type = adapter_type
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._max_request_bytes = max_request_bytes
        self._called = False

    async def align(self, pairs: tuple[PropositionPairInput, ...]) -> PairAlignmentRuntimeResult:
        if self._called or not pairs or len(pairs) > 64:
            raise ValueError("PAIR_RUNTIME_CALL_BOUND")
        self._called = True
        payload = request_payload(pairs)
        request = CanonicalModelRequest(
            task_kind="proposition_pair_identity_alignment_v1",
            role="proposition_pair_identity_alignment_runtime_v1",
            model=self._model,
            instructions=(
                "Return the strict pair-field judgment schema only. For each supplied pair, "
                "judge SUBJECT referent identity and PREDICATE relation identity independently. "
                "Use SAME, DIFFERENT, or UNKNOWN and the matching closed reason code. "
                "Ambiguous pronouns or insufficient context require UNKNOWN even for identical "
                "wording. Do not invent aliases or canonical entities. Never judge whole "
                "proposition equality, values, polarity relation or context compatibility. "
                "Preserve modality, attribution, scope, conditions and composition as context. "
                "Return exactly both field roles for every pair with the supplied refs."
            ),
            content_parts=(ContentPart(kind="json", value=payload),),
            response_contract=PairAlignmentResponse.model_json_schema(),
            generation_parameters={"temperature": 0.0, "reasoning_enabled": False},
            max_output_tokens=8_000,
            context_manifest_id=uuid4(),
            metadata={
                "schema_version": PAIR_IDENTITY_SCHEMA,
                "prompt_version": PAIR_IDENTITY_PROMPT,
            },
        )
        if (
            len(json.dumps(request.model_dump(mode="json"), ensure_ascii=False).encode())
            > self._max_request_bytes
        ):
            return PairAlignmentRuntimeResult(
                status="INVALID",
                bindings=tuple(pair.binding for pair in pairs),
                model_call_count=0,
                request_hash=content_hash(payload),
            )
        facts: tuple[PropositionPairSemanticAlignmentFact, ...] = ()
        status: Literal["COMPLETED", "INVALID", "FAILED"] = "FAILED"
        usage: Any = None
        latency = 0
        try:
            result = await self._gateway.generate_structured(
                adapter_type=self._adapter_type,
                base_url=self._base_url,
                api_key=self._api_key,
                request=request,
                allow_regeneration=False,
            )
            usage = getattr(result, "usage", None)
            latency = getattr(result, "latency_ms", 0) or 0
            status = "INVALID"
            facts = parse_pair_response(getattr(result, "parsed_object", None), pairs)
            status = "COMPLETED"
        except Exception:
            # Persist only closed execution states, never provider exception text.
            pass
        return PairAlignmentRuntimeResult(
            status=status,
            bindings=tuple(pair.binding for pair in pairs),
            facts=facts,
            model_call_count=1,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            latency_ms=latency,
            request_hash=content_hash(payload),
        )
