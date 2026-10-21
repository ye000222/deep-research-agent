"""Shadow-only deterministic routing and independent semantic-support verification."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from time import perf_counter
from typing import Any, Protocol
from uuid import uuid4

from pydantic import SecretStr, ValidationError

from app.domain.providers import AdapterType, CanonicalModelRequest, ContentPart
from app.domain.semantic_support_verification import (
    SEMANTIC_SUPPORT_VERIFIER_PROMPT,
    SEMANTIC_SUPPORT_VERIFIER_PROMPT_VERSION,
    SEMANTIC_SUPPORT_VERIFIER_SCHEMA_VERSION,
    BindingMateriality,
    BindingMaterialityRecord,
    HybridSemanticSupportShadowResult,
    SemanticSupportBindingResult,
    SemanticSupportRoute,
    SemanticSupportRouterDecision,
    SemanticSupportSource,
    SemanticVerificationBatchResponse,
    SemanticVerificationItem,
    SemanticVerificationRequest,
    SemanticVerificationVerdictItem,
    SemanticVerifierVerdict,
    VerifierExecutionDiagnostics,
    VerifierFailureStatus,
)
from app.domain.source_context_grounding_v2 import (
    EvidenceTargetKind,
    GroundingDimensionStatus,
    SemanticEvidenceRole,
    SourceContextGroundingResultV2,
    SourceEvidenceSpan,
)

SEMANTIC_SUPPORT_VERIFIER_MAX_OUTPUT_TOKENS = 4_000
_DECIMAL = re.compile(r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?![\w.])")
_LEXICAL_VALUE_ROLES = frozenset(
    {
        SemanticEvidenceRole.CONTEXT_VALUE,
        SemanticEvidenceRole.METHOD_VALUE,
        SemanticEvidenceRole.CONFIGURATION_VALUE,
        SemanticEvidenceRole.SCOPE_VALUE,
        SemanticEvidenceRole.POPULATION_VALUE,
        SemanticEvidenceRole.LOCATION_VALUE,
        SemanticEvidenceRole.SCENARIO_VALUE,
        SemanticEvidenceRole.TEMPORAL_VALUE,
        SemanticEvidenceRole.VERSION_VALUE,
        SemanticEvidenceRole.SAMPLE_SCOPE_VALUE,
        SemanticEvidenceRole.MEASUREMENT_UNIT,
    }
)
_INFERENCE_ROLES = frozenset(
    {
        SemanticEvidenceRole.ATTACHMENT,
        SemanticEvidenceRole.METHOD_RELATION,
        SemanticEvidenceRole.EPISTEMIC_OPERATOR,
        SemanticEvidenceRole.EPISTEMIC_AGENT,
        SemanticEvidenceRole.MODALITY,
        SemanticEvidenceRole.CONDITION,
        SemanticEvidenceRole.RELATION_OPERATOR,
        SemanticEvidenceRole.MEASUREMENT_DEFINITION,
        SemanticEvidenceRole.MEASUREMENT_COMPARATOR,
        SemanticEvidenceRole.MEASUREMENT_VALUE,
    }
)


class StructuredSemanticVerifierGateway(Protocol):
    async def generate_structured(
        self,
        *,
        adapter_type: AdapterType,
        base_url: str,
        api_key: SecretStr,
        request: CanonicalModelRequest,
        allow_regeneration: bool = True,
    ) -> Any: ...


def _scalars(value: object) -> list[str]:
    if isinstance(value, dict):
        return [item for nested in value.values() for item in _scalars(nested)]
    if isinstance(value, (tuple, list)):
        return [item for nested in value for item in _scalars(nested)]
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return [str(value)]
    return []


def _decimals(value: str) -> set[Decimal]:
    result: set[Decimal] = set()
    for match in _DECIMAL.findall(value):
        try:
            result.add(Decimal(match))
        except InvalidOperation:
            continue
    return result


def route_semantic_support(
    binding_ref: str,
    *,
    role: SemanticEvidenceRole,
    target_kind: EvidenceTargetKind,
    target_ref: str,
    evidence_span: SourceEvidenceSpan | None,
    textual_status: GroundingDimensionStatus,
    structural_status: GroundingDimensionStatus,
    target_payload: dict[str, Any],
    scope_membership_valid: bool = False,
    assertion_membership_valid: bool = False,
    materiality: BindingMateriality | None = None,
) -> SemanticSupportRouterDecision:
    """Route only exact/mechanical contracts; natural-language entailment abstains."""
    if textual_status is not GroundingDimensionStatus.VALID:
        return SemanticSupportRouterDecision(
            binding_ref=binding_ref,
            route=SemanticSupportRoute.INSUFFICIENT_EVIDENCE,
            rule_id="textual_precondition_not_valid",
            diagnostics=("verifier_not_eligible",),
        )
    if structural_status is not GroundingDimensionStatus.VALID:
        return SemanticSupportRouterDecision(
            binding_ref=binding_ref,
            route=SemanticSupportRoute.INSUFFICIENT_EVIDENCE,
            rule_id="structural_precondition_not_valid",
            diagnostics=("verifier_not_eligible",),
        )
    if evidence_span is None:
        return SemanticSupportRouterDecision(
            binding_ref=binding_ref,
            route=SemanticSupportRoute.INSUFFICIENT_EVIDENCE,
            rule_id="evidence_span_unavailable",
            diagnostics=("verifier_not_eligible",),
        )
    if materiality is BindingMateriality.STRUCTURAL_SUPPORT_ONLY:
        return SemanticSupportRouterDecision(
            binding_ref=binding_ref,
            route=SemanticSupportRoute.NOT_APPLICABLE,
            rule_id="structural_support_only_contract",
        )
    if role is SemanticEvidenceRole.ASSERTION_EVIDENCE:
        if assertion_membership_valid:
            return SemanticSupportRouterDecision(
                binding_ref=binding_ref,
                route=SemanticSupportRoute.DETERMINISTICALLY_SUPPORTED,
                rule_id="explicit_assertion_evidence_membership",
            )
        return SemanticSupportRouterDecision(
            binding_ref=binding_ref,
            route=SemanticSupportRoute.DETERMINISTICALLY_REJECTED,
            rule_id="assertion_evidence_not_a_member",
        )
    if role is SemanticEvidenceRole.SCOPE_STRUCTURE and scope_membership_valid:
        return SemanticSupportRouterDecision(
            binding_ref=binding_ref,
            route=SemanticSupportRoute.DETERMINISTICALLY_SUPPORTED,
            rule_id="explicit_scope_graph_membership",
        )

    expected = _scalars(target_payload.get("semantic_value"))
    lexical_value = target_payload.get("lexical_value")
    if not expected and isinstance(lexical_value, str) and lexical_value:
        expected.append(lexical_value)
    evidence = evidence_span.exact_text
    if role in _LEXICAL_VALUE_ROLES:
        if len(expected) == 1 and expected[0] == evidence:
            return SemanticSupportRouterDecision(
                binding_ref=binding_ref,
                route=SemanticSupportRoute.DETERMINISTICALLY_SUPPORTED,
                rule_id="exact_lexical_value_identity",
            )
        expected_numbers = (
            set().union(*(_decimals(value) for value in expected)) if len(expected) == 1 else set()
        )
        evidence_numbers = _decimals(evidence)
        if expected_numbers and evidence_numbers:
            if expected_numbers == evidence_numbers:
                return SemanticSupportRouterDecision(
                    binding_ref=binding_ref,
                    route=SemanticSupportRoute.DETERMINISTICALLY_SUPPORTED,
                    rule_id="exact_decimal_value_equivalence",
                )
            if (
                role is SemanticEvidenceRole.CONTEXT_VALUE
                and len(evidence_numbers) == 1
                and evidence.strip() == next(iter(evidence_numbers)).to_eng_string()
            ):
                return SemanticSupportRouterDecision(
                    binding_ref=binding_ref,
                    route=SemanticSupportRoute.DETERMINISTICALLY_REJECTED,
                    rule_id="exact_decimal_value_mismatch",
                    diagnostics=("numeric_identity_contract",),
                )
    if role in _INFERENCE_ROLES:
        return SemanticSupportRouterDecision(
            binding_ref=binding_ref,
            route=SemanticSupportRoute.REQUIRES_SEMANTIC_INFERENCE,
            rule_id="natural_language_relation_requires_verification",
        )
    return SemanticSupportRouterDecision(
        binding_ref=binding_ref,
        route=SemanticSupportRoute.REQUIRES_SEMANTIC_INFERENCE,
        rule_id="no_safe_mechanical_support_rule",
        diagnostics=(f"target={target_kind.value}:{target_ref}",),
    )


def _span_map(result: SourceContextGroundingResultV2) -> dict[str, SourceEvidenceSpan]:
    return {span.evidence_ref: span for span in result.evidence_spans}


def _context_payload(result: SourceContextGroundingResultV2, binding: Any) -> dict[str, Any]:
    if binding.target_kind is EvidenceTargetKind.CONTEXT:
        context = next(
            (c for c in result.source_contexts if c.context_ref == binding.target_ref), None
        )
        if context is not None:
            return {
                "context_type": context.context_type.value,
                "semantic_value": context.semantic_value,
                "lexical_value": context.lexical_value,
            }
    if binding.target_kind is EvidenceTargetKind.ATTACHMENT:
        edge = next(
            (x for x in result.source_attachments if x.attachment_ref == binding.target_ref), None
        )
        if edge is not None:
            context = next(
                (c for c in result.source_contexts if c.context_ref == edge.context_ref), None
            )
            return {
                "relation": edge.relation.value,
                "status": edge.status,
                "context": (
                    {
                        "context_type": context.context_type.value,
                        "semantic_value": context.semantic_value,
                        "lexical_value": context.lexical_value,
                    }
                    if context is not None
                    else {}
                ),
                "target_assertion_refs": edge.source_assertion_refs,
            }
    if binding.target_kind is EvidenceTargetKind.ASSERTION:
        assertion = next(
            (a for a in result.source_assertions if a.source_assertion_ref == binding.target_ref),
            None,
        )
        if assertion is not None:
            return {"assertion_status": assertion.assertion_status}
    if binding.target_kind is EvidenceTargetKind.SCOPE:
        scope = next(
            (s for s in result.source_scopes if s.source_scope_ref == binding.target_ref), None
        )
        if scope is not None:
            return {"scope_kind": scope.kind.value, "assertion_refs": scope.source_assertion_refs}
    return {}


def _scope_membership_valid(result: SourceContextGroundingResultV2, binding: Any) -> bool:
    scope = next(
        (item for item in result.source_scopes if item.source_scope_ref == binding.target_ref),
        None,
    )
    if (
        scope is None
        or not scope.source_assertion_refs
        or binding.evidence_ref not in scope.evidence_refs
    ):
        return False
    assertion_refs = {item.source_assertion_ref for item in result.source_assertions}
    span_refs = {item.evidence_ref for item in result.evidence_spans}
    return (
        set(scope.source_assertion_refs) <= assertion_refs and set(scope.evidence_refs) <= span_refs
    )


def _text_for_refs(refs: tuple[str, ...], spans: dict[str, SourceEvidenceSpan]) -> list[str]:
    return [spans[ref].exact_text for ref in refs if ref in spans]


def build_semantic_verification_item(
    result: SourceContextGroundingResultV2,
    binding: Any,
    source_text: str,
) -> SemanticVerificationItem | None:
    """Build bounded source-local context; never forwards whole-run state."""
    spans = _span_map(result)
    span = spans.get(binding.evidence_ref)
    if span is None or span.start_offset is None or span.end_offset is None:
        return None
    if source_text[span.start_offset : span.end_offset] != span.exact_text:
        return None

    assertion_refs: list[str] = []
    scope_refs: list[str] = []
    if binding.target_kind is EvidenceTargetKind.ASSERTION:
        assertion_refs.append(binding.target_ref)
    elif binding.target_kind is EvidenceTargetKind.ATTACHMENT:
        edge = next(
            (a for a in result.source_attachments if a.attachment_ref == binding.target_ref), None
        )
        if edge is not None:
            assertion_refs.extend(edge.source_assertion_refs)
            if edge.source_scope_ref:
                scope_refs.append(edge.source_scope_ref)
    elif binding.target_kind is EvidenceTargetKind.CONTEXT:
        for edge in result.source_attachments:
            if edge.context_ref == binding.target_ref:
                assertion_refs.extend(edge.source_assertion_refs)
                if edge.source_scope_ref:
                    scope_refs.append(edge.source_scope_ref)
    elif binding.target_kind is EvidenceTargetKind.SCOPE:
        scope_refs.append(binding.target_ref)
        scope = next(
            (s for s in result.source_scopes if s.source_scope_ref == binding.target_ref), None
        )
        if scope is not None:
            assertion_refs.extend(scope.source_assertion_refs)

    assertions = {item.source_assertion_ref: item for item in result.source_assertions}
    assertion_refs = list(dict.fromkeys(ref for ref in assertion_refs if ref in assertions))
    scope_map = {item.source_scope_ref: item for item in result.source_scopes}
    scope_refs = list(dict.fromkeys(ref for ref in scope_refs if ref in scope_map))
    assertion_texts = [
        text
        for ref in assertion_refs
        for text in _text_for_refs(assertions[ref].evidence_refs, spans)
    ]
    scope_texts = [
        text for ref in scope_refs for text in _text_for_refs(scope_map[ref].evidence_refs, spans)
    ]
    pieces = list(dict.fromkeys([span.exact_text, *assertion_texts, *scope_texts]))
    local_context = "\n".join(pieces)[:8_000]
    if not local_context:
        return None
    target_assertion_text = "\n".join(assertion_texts)[:4_000] or None
    scope_text = "\n".join(scope_texts)[:4_000] or None
    return SemanticVerificationItem(
        binding_ref=binding.binding_ref,
        semantic_role=binding.semantic_role,
        target_kind=binding.target_kind,
        target_ref=binding.target_ref,
        canonical_target_payload=_context_payload(result, binding),
        exact_evidence_span=span.exact_text,
        source_local_context=local_context,
        target_source_assertion=target_assertion_text,
        source_scope_or_group=scope_text,
        neighboring_source_assertions=tuple(assertion_texts[1:9]),
    )


def _derive_status(
    route: SemanticSupportRoute, verdict: SemanticVerifierVerdict | None
) -> GroundingDimensionStatus:
    if route is SemanticSupportRoute.DETERMINISTICALLY_SUPPORTED:
        return GroundingDimensionStatus.VALID
    if route is SemanticSupportRoute.DETERMINISTICALLY_REJECTED:
        return GroundingDimensionStatus.INVALID
    if route is SemanticSupportRoute.INSUFFICIENT_EVIDENCE:
        return GroundingDimensionStatus.MISSING_EVIDENCE
    if route is SemanticSupportRoute.NOT_APPLICABLE:
        return GroundingDimensionStatus.NOT_EVALUATED
    return {
        SemanticVerifierVerdict.SUPPORTED: GroundingDimensionStatus.VALID,
        SemanticVerifierVerdict.NOT_SUPPORTED: GroundingDimensionStatus.INVALID,
        SemanticVerifierVerdict.AMBIGUOUS: GroundingDimensionStatus.AMBIGUOUS,
        SemanticVerifierVerdict.INSUFFICIENT_EVIDENCE: GroundingDimensionStatus.MISSING_EVIDENCE,
        None: GroundingDimensionStatus.AMBIGUOUS,
    }[verdict]


def _safe_result_payload(value: Any) -> dict[str, Any] | None:
    parsed = getattr(value, "parsed_object", None)
    if isinstance(parsed, dict):
        return parsed
    text = getattr(value, "text", None)
    if isinstance(text, str):
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            return None
        return data if isinstance(data, dict) else None
    if isinstance(value, dict):
        return value
    return None


def _usage_value(value: Any, name: str) -> int | None:
    usage = getattr(value, "usage", None)
    raw = getattr(usage, name, None) if usage is not None else None
    return raw if isinstance(raw, int) and raw >= 0 else None


async def evaluate_hybrid_semantic_support(
    result: SourceContextGroundingResultV2,
    source_text: str,
    *,
    gateway: StructuredSemanticVerifierGateway | None = None,
    adapter_type: AdapterType | None = None,
    base_url: str | None = None,
    api_key: SecretStr | None = None,
    model: str | None = None,
    provider_identity: str | None = None,
    model_identity: str | None = None,
    materiality: tuple[BindingMaterialityRecord, ...] = (),
) -> HybridSemanticSupportShadowResult:
    """Evaluate one Claim in shadow; at most one structured verifier call."""
    claim_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    span_map = _span_map(result)
    materiality_map = {item.binding_ref: item for item in materiality}
    decisions: dict[str, SemanticSupportRouterDecision] = {}
    proposals = {item.binding_ref: item for item in result.evidence_bindings}
    inference_items: list[SemanticVerificationItem] = []
    diagnostics: list[str] = []

    for binding in result.evidence_bindings:
        payload = _context_payload(result, binding)
        target_assertion = next(
            (
                item
                for item in result.source_assertions
                if item.source_assertion_ref == binding.target_ref
            ),
            None,
        )
        assertion_membership_valid = bool(
            binding.target_kind is EvidenceTargetKind.ASSERTION
            and target_assertion is not None
            and binding.evidence_ref in target_assertion.evidence_refs
        )
        scope_valid = (
            _scope_membership_valid(result, binding)
            if binding.semantic_role is SemanticEvidenceRole.SCOPE_STRUCTURE
            else False
        )
        decision = route_semantic_support(
            binding.binding_ref,
            role=binding.semantic_role,
            target_kind=binding.target_kind,
            target_ref=binding.target_ref,
            evidence_span=span_map.get(binding.evidence_ref),
            textual_status=binding.dimensions.textual_grounding,
            structural_status=binding.dimensions.structural_grounding,
            target_payload=payload,
            scope_membership_valid=scope_valid,
            assertion_membership_valid=assertion_membership_valid,
            materiality=(
                materiality_map[binding.binding_ref].materiality
                if binding.binding_ref in materiality_map
                else None
            ),
        )
        decisions[binding.binding_ref] = decision
        if decision.route is SemanticSupportRoute.REQUIRES_SEMANTIC_INFERENCE:
            item = build_semantic_verification_item(result, binding, source_text)
            if item is None:
                decision = decision.model_copy(
                    update={
                        "route": SemanticSupportRoute.INSUFFICIENT_EVIDENCE,
                        "rule_id": "bounded_local_context_unavailable",
                        "diagnostics": ("verifier_not_eligible",),
                    }
                )
                decisions[binding.binding_ref] = decision
            else:
                inference_items.append(item)

    verdicts: dict[str, SemanticVerificationVerdictItem] = {}
    failure = VerifierFailureStatus.NONE
    call_count = 0
    elapsed_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    refs = tuple(item.binding_ref for item in inference_items)
    if inference_items:
        if (
            gateway is None
            or adapter_type is None
            or base_url is None
            or api_key is None
            or not model
        ):
            failure = VerifierFailureStatus.GATEWAY_UNAVAILABLE
            diagnostics.append("verifier_gateway_not_configured")
        else:
            try:
                verification_request = SemanticVerificationRequest(items=tuple(inference_items))
                request = CanonicalModelRequest(
                    task_kind="semantic_support_verification_shadow_v1",
                    role="semantic_support_verifier_v1",
                    model=model,
                    instructions=SEMANTIC_SUPPORT_VERIFIER_PROMPT,
                    content_parts=(
                        ContentPart(
                            kind="json", value=verification_request.model_dump(mode="json")
                        ),
                    ),
                    response_contract=SemanticVerificationBatchResponse.model_json_schema(),
                    generation_parameters={"temperature": 0.0, "reasoning_enabled": False},
                    max_output_tokens=SEMANTIC_SUPPORT_VERIFIER_MAX_OUTPUT_TOKENS,
                    context_manifest_id=uuid4(),
                    metadata={
                        "prompt_version": SEMANTIC_SUPPORT_VERIFIER_PROMPT_VERSION,
                        "schema_version": SEMANTIC_SUPPORT_VERIFIER_SCHEMA_VERSION,
                        "execution_mode": "SHADOW",
                        "max_calls": "1",
                        "retry": "0",
                        "regeneration": "false",
                        "fallback": "none",
                        "repair": "none",
                    },
                )
                call_count = 1
                started = perf_counter()
                response = await gateway.generate_structured(
                    adapter_type=adapter_type,
                    base_url=base_url.rstrip("/"),
                    api_key=api_key,
                    request=request,
                    allow_regeneration=False,
                )
                elapsed_ms = int((perf_counter() - started) * 1000)
                input_tokens = _usage_value(response, "input_tokens")
                output_tokens = _usage_value(response, "output_tokens")
                total_tokens = _usage_value(response, "total_tokens")
                response_payload = _safe_result_payload(response)
                if response_payload is None:
                    failure = VerifierFailureStatus.SCHEMA_INVALID
                    diagnostics.append("verifier_response_not_structured_json")
                else:
                    batch = SemanticVerificationBatchResponse.model_validate(response_payload)
                    returned_refs = [item.binding_ref for item in batch.verdicts]
                    counts = Counter(returned_refs)
                    if any(ref not in set(refs) for ref in returned_refs) or any(
                        count > 1 for count in counts.values()
                    ):
                        failure = VerifierFailureStatus.BATCH_INVALID
                        diagnostics.append("unknown_or_duplicate_binding_ref")
                    else:
                        verdicts = {item.binding_ref: item for item in batch.verdicts}
                        if any(ref not in verdicts for ref in refs):
                            failure = VerifierFailureStatus.RESPONSE_OMITTED
                            diagnostics.append("requested_binding_verdict_omitted")
            except ValidationError:
                failure = VerifierFailureStatus.SCHEMA_INVALID
                diagnostics.append("verifier_schema_invalid")
            except Exception:
                failure = VerifierFailureStatus.GATEWAY_ERROR
                diagnostics.append("verifier_gateway_error")

    output_bindings: list[SemanticSupportBindingResult] = []
    for binding_ref, decision in decisions.items():
        proposal = proposals[binding_ref]
        verifier_result = verdicts.get(binding_ref)
        verifier_verdict = verifier_result.verdict if verifier_result is not None else None
        local_diagnostics = list(decision.diagnostics)
        source = SemanticSupportSource.DETERMINISTIC_ROUTER
        if decision.route is SemanticSupportRoute.REQUIRES_SEMANTIC_INFERENCE:
            source = SemanticSupportSource.INDEPENDENT_VERIFIER
            if failure is not VerifierFailureStatus.NONE:
                verifier_verdict = (
                    SemanticVerifierVerdict.INSUFFICIENT_EVIDENCE
                    if failure is VerifierFailureStatus.RESPONSE_OMITTED
                    else SemanticVerifierVerdict.AMBIGUOUS
                )
                local_diagnostics.append(f"verifier_fail_closed:{failure.value}")
            elif verifier_verdict is None:
                verifier_verdict = SemanticVerifierVerdict.AMBIGUOUS
                local_diagnostics.append("verifier_verdict_missing_fail_closed")
            support = _derive_status(decision.route, verifier_verdict)
        else:
            support = _derive_status(decision.route, None)
        if proposal.dimensions.textual_grounding is not GroundingDimensionStatus.VALID:
            support = GroundingDimensionStatus.NOT_EVALUATED
            local_diagnostics.append("textual_precondition_prevents_support_evaluation")
        elif proposal.dimensions.structural_grounding is not GroundingDimensionStatus.VALID:
            support = GroundingDimensionStatus.NOT_EVALUATED
            local_diagnostics.append("structural_precondition_prevents_support_evaluation")
        materiality_record = materiality_map.get(binding_ref)
        output_bindings.append(
            SemanticSupportBindingResult(
                binding_ref=binding_ref,
                support_source=source,
                router_decision=decision,
                verifier_verdict=verifier_verdict,
                verifier_result=verifier_result,
                semantic_support=support,
                materiality=materiality_record.materiality if materiality_record else None,
                materiality_contract_version=(
                    materiality_record.contract_version if materiality_record else None
                ),
                diagnostics=tuple(local_diagnostics),
            )
        )

    return HybridSemanticSupportShadowResult(
        claim_ref=result.claim_ref,
        claim_hash=claim_hash,
        bindings=tuple(output_bindings),
        verifier=VerifierExecutionDiagnostics(
            provider_identity=provider_identity,
            model_identity=model_identity,
            input_binding_refs=refs,
            call_count=call_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=elapsed_ms,
            failure_status=failure,
            diagnostics=tuple(diagnostics),
        ),
        status="SHADOW_EVALUATED",
    )
