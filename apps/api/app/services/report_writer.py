"""Evidence-only report writing, citation assembly, and deterministic verification."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, ValidationError

from app.context.manager import (
    ContextBudgetInsufficientError,
    ContextBudgetManager,
    allocate_budget,
)
from app.core.stage_timing import stage_timer, timed_call
from app.domain.context import ContextCandidate, ContextEnvelope, ContextItemType
from app.domain.event_coverage import (
    MISSING_EVENT_TEXT,
    event_types,
    project_event_coverage,
    temporal_support,
)
from app.domain.providers import CanonicalModelRequest, ContentPart, TokenUsage, UsageAccuracy
from app.domain.research_budget import estimate_writer_reserve_tokens
from app.evaluation.report_verifier import verify_citation_integrity, verify_evidence_support
from app.infrastructure.db.report_draft_cache import ReportDraftCacheRepository
from app.infrastructure.db.reports import (
    PersistedCitation,
    PersistedSection,
    ReportContext,
    ReportEvidenceCard,
    ReportRepository,
)
from app.infrastructure.db.run_providers import RunProviderBindingRepository
from app.llm.adapters import LLMGateway, ModelGatewayError
from app.security.secrets import SecretCipher
from app.services.writer_context import compact_json, pack_writer_context

_MAX_WRITER_EVIDENCE = 20
_MAX_QUOTE_CHARS = 900
_WRITER_TOKEN_RESERVE = 15_000
_CITATION_PATTERN = re.compile(r"\[\d+\]")
_BUDGET_STOP_REASONS = frozenset(
    {
        "research_budget_exhausted",
        "page_budget_exhausted",
        "search_budget_exhausted",
        "token_budget_exhausted",
        "iteration_budget_exhausted",
        "deadline_exhausted",
        "logical_query_budget_exhausted",
        "provider_request_budget_exhausted",
        "fetched_page_budget_exhausted",
        "extracted_page_budget_exhausted",
        "action_budget_exhausted",
    }
)

WRITER_INSTRUCTIONS = """你是 Deep Research Agent 的报告 Writer。
你只能使用输入中的 Evidence Cards,不得加入模型记忆、常识或未经证据支持的事实。
每个段落必须列出直接支持它的 evidence_ids;没有支持证据就不要写该段落。
不要自己生成 [1] 等引用编号,系统会把 Evidence ID 映射为稳定引用。
避免夸大因果关系;证据不足、单一来源、预算耗尽或定义不一致必须写入 limitations。
输出中文、结构清晰、适合 Markdown 报告,并严格满足 JSON Schema。"""
WRITER_INSTRUCTIONS += (
    "\ncoverage_targets 必须分别回答; section.question_id 使用 target_id。"
    "事件不可互相替代。缺失节点写入 limitations; 无年份的月日不能作为完整历史日期。"
)


@dataclass(slots=True)
class WriterAttempt:
    gateway_dispatched: bool = False
    response_received: bool = False
    usage: TokenUsage | None = None
    input_evidence_ids: set[str] = field(default_factory=set)
    packing: dict[str, Any] = field(default_factory=dict)


class WriterStageError(RuntimeError):
    pass


def _writer_failure(exc: Exception, attempt: WriterAttempt) -> tuple[str, str]:
    if isinstance(exc, ContextBudgetInsufficientError):
        return "WRITER_CONTEXT_BUDGET_INSUFFICIENT", str(exc)
    if isinstance(exc, WriterStageError):
        return "WRITER_OUTPUT_EMPTY", "empty_provider_output"
    if not attempt.gateway_dispatched:
        return "WRITER_CALL_NOT_DISPATCHED", type(exc).__name__
    if isinstance(exc, ModelGatewayError):
        length = _safe_int(exc.diagnostics.get("response_length", 0))
        if exc.code == "UNSUPPORTED_ADAPTER":
            return "WRITER_CALL_NOT_DISPATCHED", exc.code
        if "PARSE_EMPTY" in (exc.detail_code or ""):
            return "WRITER_OUTPUT_EMPTY", exc.code
        if length > 0 and ("OUTPUT" in exc.code or "PARSE_" in (exc.detail_code or "")):
            return "WRITER_SCHEMA_INVALID", exc.code
        return "WRITER_PROVIDER_ERROR", exc.code
    if isinstance(exc, ValidationError) and attempt.response_received:
        return "WRITER_SCHEMA_INVALID", "returned_content_failed_schema"
    return "WRITER_PROVIDER_ERROR", "timeout_outcome_unknown" if isinstance(
        exc, TimeoutError
    ) else type(exc).__name__


class DraftParagraph(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    evidence_ids: list[str] = Field(min_length=1, max_length=6)


class DraftSection(BaseModel):
    question_id: str = Field(min_length=1, max_length=50)
    title: str = Field(min_length=1, max_length=300)
    paragraphs: list[DraftParagraph] = Field(min_length=1, max_length=8)


class ReportDraft(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    executive_summary: list[DraftParagraph] = Field(min_length=1, max_length=6)
    sections: list[DraftSection] = Field(min_length=1, max_length=20)
    limitations: list[str] = Field(default_factory=list, max_length=12)


@dataclass(frozen=True, slots=True)
class AssembledReport:
    title: str
    final_markdown: str
    limitations: list[str]
    verification_result: dict[str, Any]
    sections: list[PersistedSection]
    citations: list[PersistedCitation]


class ReportWriterService:
    def __init__(
        self,
        repository: ReportRepository,
        bindings: RunProviderBindingRepository,
        cipher: SecretCipher,
        gateway: LLMGateway,
        contexts: ContextBudgetManager | None = None,
        draft_cache: ReportDraftCacheRepository | None = None,
        budget_repository: Any | None = None,
    ) -> None:
        self._repository = repository
        self._bindings = bindings
        self._cipher = cipher
        self._gateway = gateway
        self._contexts = contexts
        self._draft_cache = draft_cache
        self._budget_repository = budget_repository

    async def write(self, run_id: UUID, *, worker_task_id: str) -> str:
        context = await self._repository.load_for_writing(run_id, worker_task_id=worker_task_id)
        if not context.evidence:
            await self._repository.fail_no_evidence(run_id, worker_task_id=worker_task_id)
            return "failed:no_accepted_evidence"

        selected = select_writer_evidence(context)
        draft: ReportDraft | None = None
        usage: TokenUsage | None = None
        fallback_reason: str | None = None
        fallback_detail: str | None = None
        attempt = WriterAttempt()
        draft_cache_hit = False
        cache_key: dict[str, str] | None = None
        if (
            self._draft_cache is not None
            and context.budget_snapshot.get("report_draft_cache_enabled") is True
        ):
            cache_key = {
                "context_hash": _draft_context_hash(context, selected),
                "writer_version": _WRITER_VERSION,
            }
            cached_draft = await self._draft_cache.get(run_id, **cache_key)
            if cached_draft is not None:
                try:
                    cached_candidate = ReportDraft.model_validate(cached_draft)
                    if _draft_rejection_reason(cached_candidate, selected, context=context) is None:
                        draft = cached_candidate
                        draft_cache_hit = True
                except ValidationError:
                    draft = None  # a stale/invalid cache entry must not break writing
        if _writer_budget_available(context):
            if draft is None:
                writer_limit = _writer_token_limit(context)
                writer_output_reserve = _writer_output_reserve(writer_limit)
                reservation = None
                reservation_id = uuid4()
                if self._budget_repository is not None:
                    reservation = await self._budget_repository.reserve_model_tokens(
                        run_id,
                        worker_task_id=worker_task_id,
                        question_id="__writer__",
                        node="report_writer",
                        attempt_id=reservation_id,
                        estimated_input=max(1, writer_limit - writer_output_reserve),
                        max_output=writer_output_reserve,
                    )
                    if not reservation.granted:
                        fallback_reason = "WRITER_SKIPPED_RESEARCH_BUDGET_EXHAUSTED"
                if fallback_reason is None:
                    try:
                        candidate, usage = await asyncio.wait_for(
                            self._generate_draft(
                                context,
                                selected,
                                max_total_tokens=writer_limit,
                                attempt=attempt,
                            ),
                            timeout=_context_deadline_remaining(context),
                        )
                        assert usage is not None
                        if reservation is not None and self._budget_repository is not None:
                            await self._budget_repository.settle_model_reservation(
                                run_id,
                                worker_task_id=worker_task_id,
                                attempt_id=reservation_id,
                                actual_total=usage.total_tokens,
                                usage_estimated=usage.accuracy != UsageAccuracy.EXACT,
                            )
                        dispatched_cards = [
                            c for c in selected if str(c.evidence_id) in attempt.input_evidence_ids
                        ]
                        fallback_detail = _draft_rejection_reason(
                            candidate, dispatched_cards, context=context
                        )
                        fallback_reason = "WRITER_VALIDATION_FAILED" if fallback_detail else None
                        if fallback_reason is None:
                            draft = candidate
                            if cache_key is not None and self._draft_cache is not None:
                                await self._draft_cache.put(
                                    run_id,
                                    **cache_key,
                                    draft=draft.model_dump(mode="json"),
                                )
                    except (
                        ContextBudgetInsufficientError,
                        ModelGatewayError,
                        ValidationError,
                        TimeoutError,
                        WriterStageError,
                        LookupError,
                        ValueError,
                    ) as exc:
                        if isinstance(exc, ModelGatewayError) and (
                            exc.usage is not None or "response_length" in exc.diagnostics
                        ):
                            attempt.response_received = True
                        if reservation is not None and self._budget_repository is not None:
                            known_usage = (
                                exc.usage if isinstance(exc, ModelGatewayError) else attempt.usage
                            )
                            if known_usage is not None:
                                usage = known_usage
                                await self._budget_repository.settle_model_reservation(
                                    run_id,
                                    worker_task_id=worker_task_id,
                                    attempt_id=reservation_id,
                                    actual_total=known_usage.total_tokens,
                                    usage_estimated=known_usage.accuracy != UsageAccuracy.EXACT,
                                )
                            elif not attempt.gateway_dispatched:
                                await self._budget_repository.release_model_reservation(
                                    run_id,
                                    worker_task_id=worker_task_id,
                                    attempt_id=reservation_id,
                                )
                            else:
                                await self._budget_repository.mark_model_reservation_uncertain(
                                    run_id,
                                    worker_task_id=worker_task_id,
                                    attempt_id=reservation_id,
                                )
                        fallback_reason, fallback_detail = _writer_failure(exc, attempt)
        elif draft is None:
            fallback_reason = "WRITER_SKIPPED_RESEARCH_BUDGET_EXHAUSTED"

        with stage_timer("report.assemble_and_verify", run_id=run_id):
            assembled = assemble_report(
                context,
                selected,
                draft=draft,
                fallback_reason=fallback_reason,
            )
        assembled.verification_result["writer"] = {
            "gateway_dispatched": attempt.gateway_dispatched,
            "response_received": attempt.response_received,
            "dispatch_status": (
                "RESPONSE_RECEIVED"
                if attempt.response_received
                else "GATEWAY_DISPATCHED_OUTCOME_UNKNOWN"
                if attempt.gateway_dispatched
                else "NOT_DISPATCHED"
            ),
            "fallback_reason": fallback_reason,
            "fallback_detail": fallback_detail,
            "fallback_type": "evidence_template" if fallback_reason else None,
            "cache_hit": draft_cache_hit,
            "context_packing": attempt.packing,
        }
        # The report may contain implementation-level notes (for example a
        # Writer schema fallback) even when the research quality gate was met.
        # Those notes must not downgrade an otherwise complete research run.
        completed_with_limitations = (
            _research_completion_has_limitations(context)
            or not assembled.verification_result.get("event_coverage", {}).get("complete", True)
            or not assembled.verification_result["verified"]
        )
        report = await timed_call(
            "report.persistence",
            self._repository.save(
                run_id,
                worker_task_id=worker_task_id,
                title=assembled.title,
                final_markdown=assembled.final_markdown,
                limitations=assembled.limitations,
                verification_result=assembled.verification_result,
                sections=assembled.sections,
                citations=assembled.citations,
                usage=usage,
                writer_mode=(
                    "model"
                    if draft is not None and not draft_cache_hit
                    else "cache"
                    if draft_cache_hit
                    else "deterministic_fallback"
                ),
                completed_with_limitations=completed_with_limitations,
            ),
            run_id=run_id,
        )
        return f"report_completed:{report.status}:citations={len(report.citations)}"

    async def _generate_draft(
        self,
        context: ReportContext,
        evidence: list[ReportEvidenceCard],
        *,
        max_total_tokens: int | None = None,
        attempt: WriterAttempt | None = None,
    ) -> tuple[ReportDraft, TokenUsage]:
        attempt = attempt or WriterAttempt()
        binding = await self._bindings.get(context.run_id)
        api_key = self._cipher.decrypt(
            binding.encrypted_secret,
            credential_id=binding.credential_id,
            adapter_type=binding.adapter_type.value,
            credential_version=binding.credential_version,
        )
        output_tokens = min(5000, binding.max_output_tokens or 5000)
        input_token_cap: int | None = None
        if max_total_tokens is not None:
            output_tokens = min(output_tokens, _writer_output_reserve(max_total_tokens))
            input_token_cap = max_total_tokens - output_tokens - 512
        allocation = allocate_budget(
            context_window=binding.context_window,
            requested_output_tokens=output_tokens,
            provider_max_output_tokens=binding.max_output_tokens,
            max_input_tokens=input_token_cap,
        )
        payload = pack_writer_context(
            context,
            evidence,
            instructions=WRITER_INSTRUCTIONS,
            schema=ReportDraft.model_json_schema(),
            input_budget=allocation.input_budget,
        )
        evidence_payloads = {c["evidence_id"]: c for c in payload["evidence_cards"]}
        attempt.input_evidence_ids = set(evidence_payloads)
        attempt.packing = {
            "input_budget": allocation.input_budget,
            "output_reserve": output_tokens,
            "available_cards": len(evidence),
            "packed_cards": len(evidence_payloads),
            "source_count": len(payload["sources"]),
        }
        task_payload = {k: v for k, v in payload.items() if k != "evidence_cards"}
        manifest_id = uuid4()
        if self._contexts is not None:
            envelope = await self._contexts.build(
                run_id=context.run_id,
                node_name="report_writer",
                provider_adapter=binding.adapter_type.value,
                model=binding.model,
                candidates=(
                    ContextCandidate(
                        item_type=ContextItemType.INSTRUCTION,
                        content=WRITER_INSTRUCTIONS,
                        rank_score=1.0,
                        protected=True,
                        selected_reason_code="node_policy_required",
                    ),
                    ContextCandidate(
                        item_type=ContextItemType.TASK_BRIEF,
                        content=json.dumps(task_payload, ensure_ascii=False, separators=(",", ":")),
                        rank_score=1.0,
                        protected=True,
                        selected_reason_code="user_goal_required",
                    ),
                    *[
                        ContextCandidate(
                            item_type=ContextItemType.EVIDENCE_CARD,
                            content=json.dumps(card, ensure_ascii=False, separators=(",", ":")),
                            rank_score=1.0,
                            source_ref_type="evidence",
                            source_ref_id=evidence_id,
                            protected=True,
                            selected_reason_code="evidence_quality_rank",
                        )
                        for evidence_id, card in evidence_payloads.items()
                    ],
                    ContextCandidate(
                        item_type=ContextItemType.OUTPUT_SCHEMA,
                        content=compact_json(ReportDraft.model_json_schema()),
                        rank_score=1.0,
                        protected=True,
                        selected_reason_code="output_contract_required",
                    ),
                ),
                requested_output_tokens=output_tokens,
                context_window=binding.context_window,
                provider_max_output_tokens=binding.max_output_tokens,
                prompt_template_version="report_writer.v1",
                max_input_tokens=input_token_cap,
            )
            manifest_id = envelope.manifest_id
            _selected_writer_evidence_ids(envelope)
        request = CanonicalModelRequest(
            task_kind="report_writing",
            role="writer",
            model=binding.model,
            instructions=WRITER_INSTRUCTIONS,
            content_parts=(
                ContentPart(
                    kind="json",
                    value=payload,
                ),
            ),
            response_contract=ReportDraft.model_json_schema(),
            generation_parameters={"temperature": 0.1},
            max_output_tokens=output_tokens,
            context_manifest_id=manifest_id,
            metadata={"run_id": str(context.run_id), "node": "report_writer"},
        )
        attempt.gateway_dispatched = True
        result = await timed_call(
            "llm.report_request",
            self._gateway.generate_structured(
                adapter_type=binding.adapter_type,
                base_url=binding.base_url,
                api_key=api_key,
                request=request,
            ),
            run_id=context.run_id,
        )
        attempt.response_received = True
        attempt.usage = result.usage
        if result.parsed_object is None and not getattr(result, "text", None):
            raise WriterStageError("WRITER_OUTPUT_EMPTY")
        return ReportDraft.model_validate(result.parsed_object), result.usage


def select_writer_evidence(context: ReportContext) -> list[ReportEvidenceCard]:
    """Keep at least one high-score card per question, then fill by score."""
    selected: list[ReportEvidenceCard] = []
    seen: set[UUID] = set()
    for question in context.questions:
        candidate = next(
            (card for card in context.evidence if card.question_id == question.question_id),
            None,
        )
        if candidate is not None and candidate.evidence_id not in seen:
            selected.append(candidate)
            seen.add(candidate.evidence_id)
    for card in sorted(context.evidence, key=lambda item: item.evidence_score, reverse=True):
        if len(selected) >= _MAX_WRITER_EVIDENCE:
            break
        if card.evidence_id not in seen:
            selected.append(card)
            seen.add(card.evidence_id)
    return selected[:_MAX_WRITER_EVIDENCE]


def _selected_writer_evidence_ids(envelope: ContextEnvelope) -> set[str]:
    """Require at least one card after context budgeting.

    A valid run can have accepted cards while the context manager rejects all
    of them under a small provider window. Treat that as a local budget
    fallback, rather than sending an empty card set to the model.
    """

    selected_ids = {
        candidate.source_ref_id
        for candidate in envelope.selected_by_type(ContextItemType.EVIDENCE_CARD)
        if candidate.source_ref_id is not None
    }
    if not selected_ids:
        raise ContextBudgetInsufficientError("REPORT_EVIDENCE_CARDS_PRUNED")
    return selected_ids


def assemble_report(
    context: ReportContext,
    selected: list[ReportEvidenceCard],
    *,
    draft: ReportDraft | None,
    fallback_reason: str | None,
) -> AssembledReport:
    event_coverage = project_event_coverage(context.goal, context.evidence)
    card_by_id = {str(card.evidence_id): card for card in selected}
    card_by_uuid = {card.evidence_id: card for card in selected}
    citation_numbers: dict[UUID, int] = {}
    citations: list[PersistedCitation] = []

    def render(paragraph: DraftParagraph) -> str | None:
        cards: list[ReportEvidenceCard] = []
        for evidence_id in paragraph.evidence_ids:
            card = card_by_id.get(evidence_id)
            if card is not None and card not in cards:
                cards.append(card)
        text = _clean_paragraph(paragraph.text)
        if not cards or not text or not temporal_support(text):
            return None
        markers: list[str] = []
        for card in cards:
            number = citation_numbers.get(card.evidence_id)
            if number is None:
                number = len(citation_numbers) + 1
                citation_numbers[card.evidence_id] = number
                citations.append(
                    PersistedCitation(
                        citation_number=number,
                        evidence_id=card.evidence_id,
                        claim_id=card.claim_id,
                        snapshot_id=card.snapshot_id,
                        chunk_id=card.chunk_id,
                        source_content_hash=card.source_content_hash,
                        url=card.source_url,
                        accessed_at=card.fetched_at,
                        analysis_artifact_id=card.analysis_artifact_id,
                    )
                )
            markers.append(f"[{number}]")
        return f"{text} {''.join(markers)}"

    def render_many(items: list[DraftParagraph]) -> list[str]:
        rendered: list[str] = []
        for item in items:
            line = render(item)
            if line is not None:
                rendered.append(line)
        return rendered

    top_cards = sorted(selected, key=lambda item: item.evidence_score, reverse=True)
    summary_drafts = draft.executive_summary if draft is not None else []
    summary_lines = render_many(summary_drafts)
    if not summary_lines:
        summary_lines = render_many(
            [
                DraftParagraph(text=card.claim, evidence_ids=[str(card.evidence_id)])
                for card in top_cards
                if temporal_support(card.claim) and temporal_support(card.exact_quote)
            ][:3]
        )

    persisted_sections: list[PersistedSection] = []
    markdown_parts = [
        f"# {(draft.title if draft else context.goal).strip()}",
        "",
        "## 执行摘要",
        "",
    ]
    markdown_parts.extend(summary_lines)
    summary_markdown = "\n\n".join(summary_lines)
    persisted_sections.append(
        PersistedSection(
            section_key="executive_summary",
            title="执行摘要",
            markdown=summary_markdown,
            verification_result=_section_verification(summary_lines),
        )
    )

    draft_sections = {section.question_id: section for section in (draft.sections if draft else [])}
    section_questions = context.questions if not event_coverage["targets"] else ()
    for question in section_questions:
        section = draft_sections.get(question.question_id)
        lines = render_many(section.paragraphs if section else [])
        if not lines:
            cards = [card for card in selected if card.question_id == question.question_id]
            lines = render_many(
                [
                    DraftParagraph(text=card.claim, evidence_ids=[str(card.evidence_id)])
                    for card in cards[:3]
                ]
            )
        if not lines:
            continue
        title = section.title.strip() if section else question.question
        section_markdown = "\n\n".join(lines)
        markdown_parts.extend(["", f"## {title}", "", section_markdown])
        persisted_sections.append(
            PersistedSection(
                section_key=question.question_id,
                title=title,
                markdown=section_markdown,
                verification_result=_section_verification(lines),
            )
        )

    # User-requested events own their sections. A source about opening may be
    # cited for opening, but never fills a construction/completion/operation gap.
    for target in event_coverage["targets"]:
        ids = set(target["temporally_complete_evidence_ids"])
        cards = [card for card in selected if str(card.evidence_id) in ids]
        section = draft_sections.get(target["target_id"])
        paragraphs = section.paragraphs if section else []
        paragraphs = [
            p
            for p in paragraphs
            if p.evidence_ids
            and set(p.evidence_ids) <= ids
            and target["target_id"] in event_types(p.text)
        ]
        lines = render_many(paragraphs) or render_many(
            [
                DraftParagraph(
                    text=(
                        card.claim
                        if target["target_id"] in event_types(card.claim)
                        else card.exact_quote
                    ),
                    evidence_ids=[str(card.evidence_id)],
                )
                for card in cards[:3]
            ]
        )
        title = target["label"]
        section_markdown = "\n\n".join(lines)
        if target["coverage_status"] != "SUPPORTED" or not lines:
            section_markdown = (section_markdown + "\n\n" + MISSING_EVENT_TEXT).strip()
        verification_section = (
            _section_verification(lines)
            if lines
            else {
                "verified": True,
                "factual_paragraphs": 0,
            }
        )
        verification_section["coverage_status"] = target["coverage_status"]
        markdown_parts.extend(["", f"## {title}", "", section_markdown])
        persisted_sections.append(
            PersistedSection(
                section_key=target["target_id"],
                title=title,
                markdown=section_markdown,
                verification_result=verification_section,
            )
        )

    limitations = _limitations(context, draft, fallback_reason)
    if limitations:
        limitations_markdown = "\n".join(f"- {item}" for item in limitations)
        markdown_parts.extend(["", "## 研究限制", "", limitations_markdown])
        persisted_sections.append(
            PersistedSection(
                section_key="limitations",
                title="研究限制",
                markdown=limitations_markdown,
                verification_result={"verified": True, "factual_paragraphs": 0},
            )
        )

    markdown_parts.extend(["", "## 参考资料", ""])
    for citation in citations:
        card = next(item for item in selected if item.evidence_id == citation.evidence_id)
        markdown_parts.append(
            f"{citation.citation_number}. [{card.source_title}]({card.source_url}) "
            f"({card.source_domain},快照 {card.source_content_hash[:12]})"
        )

    factual_lines = summary_lines + [
        line
        for section in persisted_sections
        if section.section_key not in {"executive_summary", "limitations"}
        for line in section.markdown.split("\n\n")
        if line.strip() and line != MISSING_EVENT_TEXT
    ]
    numeric_lines = [line for line in factual_lines if any(char.isdigit() for char in line)]
    citation_complete = sum(bool(_CITATION_PATTERN.search(line)) for line in factual_lines)
    numeric_complete = sum(bool(_CITATION_PATTERN.search(line)) for line in numeric_lines)
    verification: dict[str, float | int | bool | list[int]] = {
        "verified": bool(factual_lines) and citation_complete == len(factual_lines),
        "factual_paragraphs": len(factual_lines),
        "citation_count": len(citations),
        "unresolved_citations": 0,
        "citation_completeness": (citation_complete / len(factual_lines) if factual_lines else 0.0),
        "numeric_citation_rate": (numeric_complete / len(numeric_lines) if numeric_lines else 1.0),
    }
    verification.update(
        verify_citation_integrity(
            factual_lines,
            numeric_lines,
            (item.citation_number for item in citations),
            evidence_ids=(item.evidence_id for item in citations),
        )
    )
    verification.update(
        verify_evidence_support(
            factual_lines,
            {
                citation.citation_number: (
                    card_by_uuid[citation.evidence_id].claim,
                    card_by_uuid[citation.evidence_id].exact_quote,
                )
                for citation in citations
            },
        )
    )
    verification["analysis_artifact_citations"] = sum(
        citation.analysis_artifact_id is not None for citation in citations
    )
    verification["event_coverage"] = event_coverage
    # Full technical success is established by the repository only after the
    # report and citations join the Run's persisted acquisition evidence.
    verification["technical_e2e_success"] = False
    verification["research_complete"] = (
        bool(verification["verified"])
        and event_coverage["complete"]
        and _research_quality_gate_met(context.quality_snapshot)
        and not _research_completion_has_limitations(context)
    )
    return AssembledReport(
        title=(draft.title if draft else context.goal).strip(),
        final_markdown="\n".join(markdown_parts).strip() + "\n",
        limitations=limitations,
        verification_result=verification,
        sections=persisted_sections,
        citations=citations,
    )


_WRITER_VERSION = "report_writer.v3"


def _draft_rejection_reason(
    draft: ReportDraft,
    selected: list[ReportEvidenceCard],
    *,
    context: ReportContext | None = None,
) -> str | None:
    """Reject structurally valid Writer output that contradicts its evidence input."""

    available_ids = {str(card.evidence_id) for card in selected}
    referenced_ids = {
        evidence_id
        for paragraph in draft.executive_summary
        for evidence_id in paragraph.evidence_ids
    }
    referenced_ids.update(
        evidence_id
        for section in draft.sections
        for paragraph in section.paragraphs
        for evidence_id in paragraph.evidence_ids
    )
    if not available_ids.intersection(referenced_ids):
        return "WRITER_DRAFT_NO_VALID_EVIDENCE_REFERENCES"
    if referenced_ids - available_ids:
        return "WRITER_DRAFT_UNKNOWN_EVIDENCE_REFERENCES"
    if context is not None and event_types(context.goal):
        coverage = project_event_coverage(context.goal, selected)
        target_ids = {
            t["target_id"]: set(t["temporally_complete_evidence_ids"]) for t in coverage["targets"]
        }
        for section in draft.sections:
            if section.question_id in target_ids and any(
                not set(p.evidence_ids) <= target_ids[section.question_id]
                for p in section.paragraphs
            ):
                return "WRITER_EVENT_BOUNDARY_MISMATCH"
        if any(
            not temporal_support(p.text)
            for p in draft.executive_summary + [p for s in draft.sections for p in s.paragraphs]
        ):
            return "WRITER_PARTIAL_TEMPORAL_SUPPORT"
        by_id = {str(c.evidence_id): c for c in selected}
        for paragraph in draft.executive_summary + [
            p for s in draft.sections for p in s.paragraphs
        ]:
            requested = event_types(paragraph.text) & event_types(context.goal)
            supported = set().union(
                *(event_types(by_id[i].exact_quote) for i in paragraph.evidence_ids)
            )
            if requested - supported:
                return "WRITER_EVENT_BOUNDARY_MISMATCH"
            years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", paragraph.text))
            supported_years = set().union(
                *(
                    set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", by_id[i].exact_quote))
                    for i in paragraph.evidence_ids
                )
            )
            if years - supported_years:
                return "WRITER_PARTIAL_TEMPORAL_SUPPORT"

    refusal_text = " ".join((draft.title, *draft.limitations)).casefold()
    compact_refusal = re.sub(r"[\W_]+", "", refusal_text)
    false_no_evidence_markers = (
        "无法生成研究报告",
        "未提供evidencecards",
        "未提供证据卡",
        "没有提供evidencecards",
        "没有提供证据卡",
        "evidencecardsnotprovided",
        "noevidencecardsprovided",
        "cannotgeneratereport",
    )
    if available_ids and any(marker in compact_refusal for marker in false_no_evidence_markers):
        return "WRITER_DRAFT_FALSE_NO_EVIDENCE_REFUSAL"
    return None


def _draft_context_hash(
    context: ReportContext,
    selected: list[ReportEvidenceCard],
) -> str:
    """Hash the exact assembled writing context; any change invalidates the cache."""
    payload = {
        "goal": context.goal,
        "stop_reason": context.stop_reason,
        "quality_snapshot": context.quality_snapshot,
        "questions": [
            {"id": q.question_id, "question": q.question, "priority": q.priority}
            for q in context.questions
        ],
        "evidence": [
            {
                "id": str(card.evidence_id),
                "question_id": card.question_id,
                "claim": card.claim,
                "quote": card.exact_quote,
                "source_url": card.source_url,
                "source_content_hash": card.source_content_hash,
                "score": card.evidence_score,
            }
            for card in selected
        ],
    }
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _writer_budget_available(context: ReportContext) -> bool:
    maximum = int(context.budget_snapshot.get("max_tokens", 0) or 0)
    planner = context.usage_snapshot.get("planner", {})
    planner_tokens = int(planner.get("total_tokens", 0)) if isinstance(planner, dict) else 0
    used = planner_tokens + int(context.usage_snapshot.get("evidence_total_tokens", 0) or 0)
    if maximum <= 0:
        return True
    reserve = _writer_token_limit(context)
    return maximum - used >= reserve


def _writer_token_limit(context: ReportContext) -> int:
    allocation = context.budget_snapshot.get("allocation", {})
    allocated_writer = (
        _safe_int(allocation.get("writer_tokens_initial", 0)) if isinstance(allocation, dict) else 0
    )
    return allocated_writer or estimate_writer_reserve_tokens(
        question_count=len(context.questions),
        evidence_count=len(context.evidence),
        maximum=_WRITER_TOKEN_RESERVE,
    )


def _writer_output_reserve(total_tokens: int) -> int:
    return min(5_000, max(768, max(1, total_tokens) // 3))


def _limitations(
    context: ReportContext,
    draft: ReportDraft | None,
    fallback_reason: str | None,
) -> list[str]:
    items: list[str] = []
    quality = context.quality_snapshot
    if context.stop_reason in _BUDGET_STOP_REASONS:
        items.append("研究预算已耗尽,报告仅综合预算停止前已验证的证据。")
    if context.stop_reason == "sources_exhausted":
        items.append("计划已遍历,但部分研究问题仍未满足证据验收条件。")
    if context.stop_reason == "stagnation":
        items.append("连续两轮边际信息增益过低,系统停止继续扩展检索。")
    if float(quality.get("cross_validation", 0.0) or 0.0) < 0.70:
        items.append("部分研究问题尚未达到多来源交叉验证阈值。")
    if float(quality.get("source_quality", 0.0) or 0.0) < 0.75:
        items.append("当前来源整体质量低于目标阈值,应补充官方、政府或论文来源。")
    projection = project_event_coverage(context.goal, context.evidence)
    for target in projection["targets"]:
        if target["coverage_status"] != "SUPPORTED":
            items.append(f"{target['label']}: {MISSING_EVENT_TEXT}")
    if any(
        not temporal_support(card.claim) or not temporal_support(card.exact_quote)
        for card in context.evidence
    ):
        items.append("部分日期缺少可追溯的年份, 未作为完整历史时间点输出。")
    if fallback_reason is not None:
        items.append(f"Writer 使用证据模板降级生成({fallback_reason})。")
    if draft is not None:
        items.extend(_clean_paragraph(item) for item in draft.limitations if item.strip())
    return list(dict.fromkeys(item for item in items if item))


def _research_quality_gate_met(quality: dict[str, object]) -> bool:
    """Mirror the research evaluator's terminal quality gate for Writer."""

    return (
        _safe_float(quality.get("coverage", 0.0)) >= 0.85
        and (
            not isinstance(quality.get("event_coverage"), dict)
            or quality["event_coverage"].get("complete") is True
        )
        and _safe_float(quality.get("priority_one_coverage", 0.0)) >= 0.80
        and _safe_float(quality.get("source_quality", 0.0)) >= 0.75
        and _safe_float(quality.get("cross_validation", 0.0)) >= 0.70
        and _safe_int(quality.get("critical_gaps", 0)) == 0
        and _safe_int(quality.get("unresolved_high_risk_conflicts", 0)) == 0
        and (
            "numeric_scope_consistency" not in quality
            or _safe_float(quality.get("numeric_scope_consistency")) >= 1.0
        )
        and ("source_role_fit" not in quality or _safe_float(quality.get("source_role_fit")) >= 1.0)
        and ("freshness" not in quality or _safe_float(quality.get("freshness")) >= 0.70)
    )


def _research_completion_has_limitations(context: ReportContext) -> bool:
    """Classify limitations from research state, never from Writer mechanics."""

    if context.stop_reason == "quality_met":
        return not _research_quality_gate_met(context.quality_snapshot)
    # A missing or unknown stop reason is not proof that every quality gate
    # passed.  Preserve the conservative classification in that case.
    return True


def _clean_paragraph(value: str) -> str:
    return " ".join(_CITATION_PATTERN.sub("", value).split()).strip()


def _safe_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, bytes, bytearray)):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, bytes, bytearray)):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _context_deadline_remaining(context: ReportContext) -> float | None:
    raw = context.budget_snapshot.get("deadline_at")
    if not isinstance(raw, str):
        return None
    try:
        deadline = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    return max(0.0, (deadline - datetime.now(UTC)).total_seconds())


def _section_verification(lines: list[str]) -> dict[str, float | int | bool]:
    resolved = sum(bool(_CITATION_PATTERN.search(line)) for line in lines)
    return {
        "verified": bool(lines) and resolved == len(lines),
        "factual_paragraphs": len(lines),
        "citation_completeness": resolved / len(lines) if lines else 0.0,
    }
