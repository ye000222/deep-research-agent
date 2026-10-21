"""Independent, bounded Claim-to-AtomicProposition formation validation."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from time import perf_counter
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from app.domain.atomic_proposition import AtomicProposition, ClaimPropositionLink
from app.domain.providers import AdapterType, CanonicalModelRequest, ContentPart

TARGET_FORMATION_SCHEMA_VERSION = "target-proposition-formation-response-v1"
TARGET_FORMATION_VALIDATOR_VERSION = "target-proposition-formation-validator.v1"
TARGET_FORMATION_PROMPT_VERSION = "target-proposition-formation.generic.v1"


class FormationFieldStatus(StrEnum):
    PRESERVED = "PRESERVED"
    NOT_PRESERVED = "NOT_PRESERVED"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class FormationStatus(StrEnum):
    FORMATION_SUFFICIENT = "FORMATION_SUFFICIENT"
    FORMATION_NOT_PRESERVED = "FORMATION_NOT_PRESERVED"
    FORMATION_UNKNOWN = "FORMATION_UNKNOWN"


class FormationExecutionStatus(StrEnum):
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_UNKNOWN = "COMPLETED_WITH_UNKNOWN"
    INVALID = "INVALID"
    FAILED = "FAILED"
    NOT_PROCESSED_BUDGET = "NOT_PROCESSED_BUDGET"


class FormationReasonCode(StrEnum):
    SUBJECT_NOT_PRESERVED = "SUBJECT_NOT_PRESERVED"
    PREDICATE_NOT_PRESERVED = "PREDICATE_NOT_PRESERVED"
    VALUE_NOT_PRESERVED = "VALUE_NOT_PRESERVED"
    POLARITY_NOT_PRESERVED = "POLARITY_NOT_PRESERVED"
    MATERIAL_CONTEXT_NOT_PRESERVED = "MATERIAL_CONTEXT_NOT_PRESERVED"
    ATTRIBUTION_NOT_PRESERVED = "ATTRIBUTION_NOT_PRESERVED"
    MODALITY_NOT_PRESERVED = "MODALITY_NOT_PRESERVED"
    COMPOSITION_NOT_PRESERVED = "COMPOSITION_NOT_PRESERVED"
    SCOPE_NOT_PRESERVED = "SCOPE_NOT_PRESERVED"
    FIELD_UNKNOWN = "FIELD_UNKNOWN"
    MISSING_ASSESSMENT = "MISSING_ASSESSMENT"
    INVALID_REFERENCE = "INVALID_REFERENCE"
    DUPLICATE_ASSESSMENT = "DUPLICATE_ASSESSMENT"
    SCHEMA_INVALID = "SCHEMA_INVALID"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    DECOMPOSITION_UNRESOLVED = "DECOMPOSITION_UNRESOLVED"
    COMPOSITION_INELIGIBLE = "COMPOSITION_INELIGIBLE"


class FormationFields(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subject: FormationFieldStatus
    predicate: FormationFieldStatus
    typed_value: FormationFieldStatus
    polarity: FormationFieldStatus
    material_contexts: FormationFieldStatus
    attribution: FormationFieldStatus
    modality: FormationFieldStatus
    composition: FormationFieldStatus
    scope_eligibility: FormationFieldStatus


class FormationJudgmentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_ref: UUID
    proposition_ref: UUID
    claim_proposition_link_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    fields: FormationFields
    reason_codes: tuple[FormationReasonCode, ...] = ()


class FormationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(pattern=r"^target-proposition-formation-response-v1$")
    judgments: tuple[FormationJudgmentPayload, ...] = Field(max_length=64)


class TargetPropositionFormationAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    assessment_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_ref: UUID
    proposition_ref: UUID
    claim_proposition_link_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    formation_schema_version: str = TARGET_FORMATION_SCHEMA_VERSION
    validator_version: str = TARGET_FORMATION_VALIDATOR_VERSION
    execution_status: FormationExecutionStatus
    formation_status: FormationStatus
    fields: FormationFields
    reason_codes: tuple[FormationReasonCode, ...] = ()


class TargetFormationRuntimeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_version: str = TARGET_FORMATION_VALIDATOR_VERSION
    execution_status: FormationExecutionStatus
    assessments: tuple[TargetPropositionFormationAssessment, ...] = ()
    reason_codes: tuple[FormationReasonCode, ...] = ()
    model_call_count: int = Field(default=0, ge=0, le=1)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)


class FormationGateway(Protocol):
    async def generate_structured(
        self,
        *,
        adapter_type: AdapterType,
        base_url: str,
        api_key: SecretStr,
        request: CanonicalModelRequest,
        allow_regeneration: bool = True,
    ) -> Any: ...


_FIELDS = tuple(FormationFields.model_fields)
_FIELD_REASON = {
    "subject": FormationReasonCode.SUBJECT_NOT_PRESERVED,
    "predicate": FormationReasonCode.PREDICATE_NOT_PRESERVED,
    "typed_value": FormationReasonCode.VALUE_NOT_PRESERVED,
    "polarity": FormationReasonCode.POLARITY_NOT_PRESERVED,
    "material_contexts": FormationReasonCode.MATERIAL_CONTEXT_NOT_PRESERVED,
    "attribution": FormationReasonCode.ATTRIBUTION_NOT_PRESERVED,
    "modality": FormationReasonCode.MODALITY_NOT_PRESERVED,
    "composition": FormationReasonCode.COMPOSITION_NOT_PRESERVED,
    "scope_eligibility": FormationReasonCode.SCOPE_NOT_PRESERVED,
}
_VALIDATOR_INSTRUCTIONS = (
    "Independently judge whether each supplied Atomic Proposition faithfully expresses its "
    "parent Claim. This is formation fidelity only, not factual verification. Do not use "
    "external evidence. Return exactly one judgment per proposition using the supplied "
    "claim_ref, proposition_ref, and claim_proposition_link_ref. Judge subject, "
    "predicate/relation (including attribution and modality force), typed value and bounds, "
    "polarity, material truth-condition contexts, attribution, modality, composition and "
    "scope/eligibility. Use UNKNOWN when uncertain. NOT_APPLICABLE is allowed only when the "
    "dimension is genuinely absent from the Claim and AP. Never use confidence scores, "
    "free-form reasoning, or source evidence. Preserve OR/XOR/UNKNOWN and do not make "
    "descendants independently asserted. Do not assess decomposition completeness as per-AP "
    "fidelity. Output JSON only under the requested schema."
)


def claim_proposition_link_ref(
    claim_id: UUID, proposition: AtomicProposition, link: ClaimPropositionLink
) -> str:
    identity = {
        "claim_id": str(claim_id),
        "proposition_id": str(proposition.proposition_id),
        "relation": link.relation,
        "proposition_index": link.proposition_index,
        "claim_span": link.claim_span,
        "decomposition_status": link.decomposition_status.value,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _assessment_ref(
    claim_id: UUID, proposition_id: UUID, link_ref: str, fields: FormationFields
) -> str:
    payload = json.dumps(
        [str(claim_id), str(proposition_id), link_ref, fields.model_dump(mode="json"),
         TARGET_FORMATION_VALIDATOR_VERSION],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _overall(fields: FormationFields) -> FormationStatus:
    values = tuple(getattr(fields, name) for name in _FIELDS)
    if FormationFieldStatus.NOT_PRESERVED in values:
        return FormationStatus.FORMATION_NOT_PRESERVED
    if FormationFieldStatus.UNKNOWN in values:
        return FormationStatus.FORMATION_UNKNOWN
    return FormationStatus.FORMATION_SUFFICIENT


def _derived_reasons(fields: FormationFields) -> tuple[FormationReasonCode, ...]:
    reasons = {
        _FIELD_REASON[name]
        for name in _FIELDS
        if getattr(fields, name) is FormationFieldStatus.NOT_PRESERVED
    }
    if any(getattr(fields, name) is FormationFieldStatus.UNKNOWN for name in _FIELDS):
        reasons.add(FormationReasonCode.FIELD_UNKNOWN)
    return tuple(sorted(reasons, key=lambda item: item.value))


def _assessment(
    claim_id: UUID,
    proposition: AtomicProposition,
    link: ClaimPropositionLink,
    fields: FormationFields,
    execution: FormationExecutionStatus,
    extra_reasons: tuple[FormationReasonCode, ...] = (),
) -> TargetPropositionFormationAssessment:
    link_ref = claim_proposition_link_ref(claim_id, proposition, link)
    status = _overall(fields)
    reasons = tuple(sorted(set((*_derived_reasons(fields), *extra_reasons)), key=lambda x: x.value))
    return TargetPropositionFormationAssessment(
        assessment_ref=_assessment_ref(claim_id, proposition.proposition_id, link_ref, fields),
        claim_ref=claim_id,
        proposition_ref=proposition.proposition_id,
        claim_proposition_link_ref=link_ref,
        execution_status=execution,
        formation_status=status,
        fields=fields,
        reason_codes=reasons,
    )


def _unknown_fields() -> FormationFields:
    return FormationFields(**{name: FormationFieldStatus.UNKNOWN for name in _FIELDS})


def assess_response(
    raw: Any,
    *,
    claim_id: UUID,
    propositions: tuple[AtomicProposition, ...],
    links: tuple[ClaimPropositionLink, ...],
    composition_eligibility: dict[UUID, str],
) -> tuple[TargetPropositionFormationAssessment, ...]:
    """Validate refs and derive overall status; provider overall claims are not accepted."""
    if len(propositions) != len(links) or any(
        link.claim_id != claim_id or link.proposition_id != proposition.proposition_id
        for proposition, link in zip(propositions, links, strict=True)
    ):
        raise ValueError("claim_proposition_provenance_mismatch")
    response = FormationResponse.model_validate(raw)
    expected = {
        item.proposition_id: (item, link)
        for item, link in zip(propositions, links, strict=True)
    }
    judgments: dict[UUID, FormationJudgmentPayload] = {}
    for returned_judgment in response.judgments:
        if (
            returned_judgment.claim_ref != claim_id
            or returned_judgment.proposition_ref not in expected
        ):
            raise ValueError("invalid_formation_reference")
        proposition, link = expected[returned_judgment.proposition_ref]
        if returned_judgment.claim_proposition_link_ref != claim_proposition_link_ref(
            claim_id, proposition, link
        ):
            raise ValueError("formation_link_reference_mismatch")
        if returned_judgment.proposition_ref in judgments:
            raise ValueError("duplicate_formation_assessment")
        judgments[returned_judgment.proposition_ref] = returned_judgment

    results = []
    for proposition_id, (proposition, link) in expected.items():
        judgment = judgments.get(proposition_id)
        if judgment is None:
            results.append(
                _assessment(
                    claim_id, proposition, link, _unknown_fields(),
                    FormationExecutionStatus.COMPLETED_WITH_UNKNOWN,
                    (FormationReasonCode.MISSING_ASSESSMENT,),
                )
            )
            continue
        fields = judgment.fields
        # Composition eligibility is derived by the frozen composition contract.
        eligibility = composition_eligibility.get(proposition_id, "UNKNOWN")
        if eligibility == "INELIGIBLE":
            fields = fields.model_copy(
                update={"composition": FormationFieldStatus.NOT_PRESERVED,
                        "scope_eligibility": FormationFieldStatus.NOT_PRESERVED}
            )
        elif eligibility != "ELIGIBLE":
            fields = fields.model_copy(
                update={"composition": FormationFieldStatus.UNKNOWN,
                        "scope_eligibility": FormationFieldStatus.UNKNOWN}
            )
        results.append(
            _assessment(
                claim_id, proposition, link, fields,
                FormationExecutionStatus.COMPLETED,
                judgment.reason_codes,
            )
        )
    return tuple(results)


class TargetPropositionFormationValidator:
    """At most one independent structured validation call per Claim/AP set."""

    def __init__(
        self,
        gateway: FormationGateway,
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

    async def validate_claim(
        self,
        *,
        claim_id: UUID,
        claim_text: str,
        propositions: tuple[AtomicProposition, ...],
        links: tuple[ClaimPropositionLink, ...],
        composition: Any,
        composition_eligibility: dict[UUID, str],
    ) -> TargetFormationRuntimeResult:
        if self._called:
            return _unknown_result(
                claim_id, propositions, links, FormationExecutionStatus.INVALID,
                FormationReasonCode.INVALID_REFERENCE,
            )
        self._called = True
        if len(propositions) != len(links) or not propositions:
            return _unknown_result(
                claim_id, propositions, links, FormationExecutionStatus.INVALID,
                FormationReasonCode.INVALID_REFERENCE,
            )
        expected_refs = tuple(
            claim_proposition_link_ref(claim_id, proposition, link)
            for proposition, link in zip(propositions, links, strict=True)
        )
        payload = {
            "schema_version": TARGET_FORMATION_SCHEMA_VERSION,
            "claim_ref": str(claim_id),
            "claim_text": claim_text,
            "claim_propositions": [
                {
                    "proposition_ref": str(proposition.proposition_id),
                    "claim_proposition_link_ref": link_ref,
                    "link_relation": link.relation,
                    "proposition_index": link.proposition_index,
                    "claim_span": link.claim_span,
                    "decomposition_status": link.decomposition_status.value,
                    "atomic_proposition": proposition.model_dump(mode="json"),
                    "composition_eligibility": composition_eligibility.get(
                        proposition.proposition_id, "UNKNOWN"
                    ),
                }
                for proposition, link, link_ref in zip(
                    propositions, links, expected_refs, strict=True
                )
            ],
            "assertion_composition": (
                composition.model_dump(mode="json") if composition is not None else None
            ),
            "truth_condition_contexts": [],
            "truth_condition_contexts_status": "NOT_AVAILABLE_IN_TARGET_CLAIM_RECEIPT",
        }
        request = CanonicalModelRequest(
            task_kind="target_proposition_formation_validation_v1",
            role="target_proposition_formation_validator",
            model=self._model,
            instructions=_VALIDATOR_INSTRUCTIONS,
            content_parts=(ContentPart(kind="json", value=payload),),
            response_contract=FormationResponse.model_json_schema(),
            generation_parameters={"temperature": 0.0, "reasoning_enabled": False},
            max_output_tokens=8_000,
            context_manifest_id=UUID(int=0),
            metadata={
                "schema_version": TARGET_FORMATION_SCHEMA_VERSION,
                "prompt_version": TARGET_FORMATION_PROMPT_VERSION,
            },
        )
        started = perf_counter()
        try:
            result = await self._gateway.generate_structured(
                adapter_type=self._adapter_type,
                base_url=self._base_url,
                api_key=self._api_key,
                request=request,
                allow_regeneration=False,
            )
        except Exception:
            return _unknown_result(
                claim_id, propositions, links, FormationExecutionStatus.FAILED,
                FormationReasonCode.PROVIDER_FAILURE,
                latency_ms=round((perf_counter() - started) * 1000),
            )
        raw = getattr(result, "parsed_object", None) or getattr(result, "text", None)
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = None
        try:
            assessments = assess_response(
                raw,
                claim_id=claim_id,
                propositions=propositions,
                links=links,
                composition_eligibility=composition_eligibility,
            )
        except (ValidationError, ValueError, TypeError):
            return _unknown_result(
                claim_id, propositions, links, FormationExecutionStatus.INVALID,
                FormationReasonCode.SCHEMA_INVALID,
                model_call_count=1,
                latency_ms=round((perf_counter() - started) * 1000),
                usage=getattr(result, "usage", None),
            )
        usage = getattr(result, "usage", None)
        unknown = any(
            item.formation_status is FormationStatus.FORMATION_UNKNOWN
            for item in assessments
        )
        return TargetFormationRuntimeResult(
            execution_status=(
                FormationExecutionStatus.COMPLETED_WITH_UNKNOWN
                if unknown else FormationExecutionStatus.COMPLETED
            ),
            assessments=assessments,
            model_call_count=1,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            latency_ms=round((perf_counter() - started) * 1000),
        )


def _unknown_result(
    claim_id: UUID,
    propositions: tuple[AtomicProposition, ...],
    links: tuple[ClaimPropositionLink, ...],
    execution_status: FormationExecutionStatus,
    reason: FormationReasonCode,
    *,
    model_call_count: int = 0,
    latency_ms: int = 0,
    usage: Any = None,
) -> TargetFormationRuntimeResult:
    if len(propositions) != len(links):
        return TargetFormationRuntimeResult(
            execution_status=FormationExecutionStatus.INVALID,
            reason_codes=(FormationReasonCode.INVALID_REFERENCE,),
            model_call_count=model_call_count,
            latency_ms=latency_ms,
        )
    assessments = tuple(
        _assessment(
            claim_id, proposition, link, _unknown_fields(), execution_status, (reason,)
        )
        for proposition, link in zip(propositions, links, strict=True)
    )
    return TargetFormationRuntimeResult(
        execution_status=execution_status,
        assessments=assessments,
        reason_codes=(reason,),
        model_call_count=model_call_count,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
        latency_ms=latency_ms,
    )


def unknown_formation_result(
    claim_id: UUID,
    propositions: tuple[AtomicProposition, ...],
    links: tuple[ClaimPropositionLink, ...],
    execution_status: FormationExecutionStatus,
    reason: FormationReasonCode,
) -> TargetFormationRuntimeResult:
    return _unknown_result(claim_id, propositions, links, execution_status, reason)
