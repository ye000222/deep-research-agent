"""Actual Writer packing/context/reservation/dispatch, with only model IO faked."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from app.context.manager import ContextBudgetManager, estimate_tokens
from app.domain.providers import CanonicalModelResult, TokenUsage, UsageAccuracy
from app.llm.adapters import ModelGatewayError
from app.services.planner import PlannerService
from app.services.report_writer import WRITER_INSTRUCTIONS, ReportDraft, ReportWriterService
from app.services.writer_context import compact_json

from tests.integration.test_v3_deterministic_e2e import (
    _Bindings,
    _Cipher,
    _new_store,
    _seed_run,
    _sqlite_utc_datetime_compatibility,  # noqa: F401
)
from tests.unit.test_event_coverage import GOAL
from tests.unit.test_report_writer import _card, _context


def draft_for(request):
    payload = request.content_parts[0].value
    card = payload["evidence_cards"][0]
    paragraph = {"text": card["claim"], "evidence_ids": [card["evidence_id"]]}
    return {
        "title": "离线证据报告",
        "executive_summary": [paragraph],
        "sections": [
            {"question_id": card["question_id"], "title": "可验证发现", "paragraphs": [paragraph]}
        ],
    }


@pytest.mark.asyncio
async def test_planner_receives_required_targets_without_changing_outline_slots():
    run_id = uuid4()
    binding = _Bindings(run_id)
    binding.binding.goal = GOAL
    result = CanonicalModelResult(
        parsed_object={
            "scope_summary": GOAL,
            "questions": [
                {
                    "question": "Atlas系统何时正式开通服务?",
                    "priority": 1,
                    "rationale": "核实核心日期和原始证据。",
                    "evidence_requirements": ["官方开通日期原文"],
                    "search_hints": ["Atlas 正式开通 日期"],
                }
            ],
            "completion_criteria": ["各事件独立覆盖", "引用原始证据"],
        },
        usage=TokenUsage(
            input_tokens=20, output_tokens=30, total_tokens=50, accuracy=UsageAccuracy.EXACT
        ),
    )
    gateway = SimpleNamespace(generate_structured=AsyncMock(return_value=result))
    plan, _usage = await PlannerService(binding, _Cipher(), gateway).generate(run_id)
    request = gateway.generate_structured.await_args.kwargs["request"]
    text = request.content_parts[0].value
    assert GOAL in text
    for target in [
        "START_OF_CONSTRUCTION",
        "COMPLETION",
        "OFFICIAL_SERVICE_OPENING",
        "OPERATIONAL_ENTRY",
    ]:
        assert target in text
    assert len(plan.questions) == 1 and plan.questions[0].id == "q1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "five",
        "ten",
        "small_context",
        "provider",
        "timeout",
        "empty",
        "schema",
        "validation",
        "binding",
        "provider_empty",
        "provider_schema",
    ],
)
async def test_writer_actual_dispatch_and_stage_classification(tmp_path, mode):
    factory, _ = _new_store(tmp_path)
    run_id = uuid4()
    _seed_run(factory, run_id, "离线研究")
    cards = [_card(i, f"q{1 + i % 2}") for i in range(1, 11 if mode == "ten" else 6)]
    context = replace(
        _context(*cards),
        run_id=run_id,
        budget_snapshot={
            "max_tokens": 30000,
            "allocation": {"writer_tokens_initial": 900 if mode == "small_context" else 4500},
        },
        usage_snapshot={"evidence_total_tokens": 1000},
        quality_snapshot={"risk_state": {"redundant": "state" * 20000}},
    )
    repository = SimpleNamespace(
        load_for_writing=AsyncMock(return_value=context),
        save=AsyncMock(return_value=SimpleNamespace(status="verified", citations=[])),
    )
    budget = SimpleNamespace(
        reserve_model_tokens=AsyncMock(return_value=SimpleNamespace(granted=True)),
        settle_model_reservation=AsyncMock(),
        release_model_reservation=AsyncMock(),
        mark_model_reservation_uncertain=AsyncMock(),
    )
    calls = []

    async def generate_structured(**kwargs):
        request = kwargs["request"]
        calls.append(request)
        assert budget.reserve_model_tokens.await_count == 1
        if mode == "provider":
            raise ModelGatewayError("MODEL_PROVIDER_UNAVAILABLE", retryable=True)
        if mode == "timeout":
            raise TimeoutError()
        if mode in {"provider_empty", "provider_schema"}:
            raise ModelGatewayError(
                "MODEL_OUTPUT_INVALID",
                retryable=False,
                detail_code="PARSE_EMPTY_RESPONSE"
                if mode == "provider_empty"
                else "PARSE_INVALID_JSON",
                diagnostics={"response_length": 0 if mode == "provider_empty" else 12},
                usage=TokenUsage(
                    input_tokens=500,
                    output_tokens=100,
                    total_tokens=600,
                    accuracy=UsageAccuracy.EXACT,
                ),
            )
        payload = draft_for(request)
        if mode == "empty":
            payload = None
        if mode == "schema":
            payload = {"unexpected": "returned nonempty content"}
        if mode == "validation":
            payload["executive_summary"][0]["evidence_ids"] = [str(uuid4())]
        return CanonicalModelResult(
            parsed_object=payload,
            usage=TokenUsage(
                input_tokens=500, output_tokens=100, total_tokens=600, accuracy=UsageAccuracy.EXACT
            ),
        )

    gateway = SimpleNamespace(generate_structured=AsyncMock(side_effect=generate_structured))
    bindings = _Bindings(run_id)
    if mode == "binding":
        bindings.get = AsyncMock(side_effect=LookupError("safe missing binding"))
    contexts = ContextBudgetManager(factory)
    writer = ReportWriterService(
        repository, bindings, _Cipher(), gateway, contexts, budget_repository=budget
    )
    await writer.write(run_id, worker_task_id="offline")
    saved = repository.save.await_args.kwargs
    diagnostic = saved["verification_result"]["writer"]
    expected = {
        "small_context": "WRITER_CONTEXT_BUDGET_INSUFFICIENT",
        "binding": "WRITER_CALL_NOT_DISPATCHED",
        "provider": "WRITER_PROVIDER_ERROR",
        "timeout": "WRITER_PROVIDER_ERROR",
        "empty": "WRITER_OUTPUT_EMPTY",
        "schema": "WRITER_SCHEMA_INVALID",
        "validation": "WRITER_VALIDATION_FAILED",
        "provider_empty": "WRITER_OUTPUT_EMPTY",
        "provider_schema": "WRITER_SCHEMA_INVALID",
    }.get(mode)
    assert diagnostic["fallback_reason"] == expected
    dispatched = mode not in {"small_context", "binding"}
    assert diagnostic["gateway_dispatched"] is dispatched
    assert gateway.generate_structured.await_count == int(dispatched)
    assert diagnostic["fallback_type"] == ("evidence_template" if expected else None)
    if not dispatched:
        budget.release_model_reservation.assert_awaited_once()
        budget.mark_model_reservation_uncertain.assert_not_awaited()
    elif mode in {"provider", "timeout"}:
        budget.mark_model_reservation_uncertain.assert_awaited_once()
        budget.release_model_reservation.assert_not_awaited()
    else:
        budget.settle_model_reservation.assert_awaited_once()
        budget.release_model_reservation.assert_not_awaited()
    if mode in {"five", "ten"}:
        assert saved["writer_mode"] == "model"
        request = calls[0]
        payload = request.content_parts[0].value
        assert "quality_snapshot" not in payload and "risk_state" not in compact_json(payload)
        assert payload["goal"] == context.goal
        assert len(payload["evidence_cards"]) >= 5
        by_id = {str(c.evidence_id): c for c in cards}
        for packed in payload["evidence_cards"]:
            assert packed["exact_quote"] == by_id[packed["evidence_id"]].exact_quote
            assert (
                payload["sources"][packed["source_ref"]]["url"]
                == by_id[packed["evidence_id"]].source_url
            )
        token_estimate = (
            estimate_tokens(WRITER_INSTRUCTIONS)
            + estimate_tokens(compact_json(ReportDraft.model_json_schema()))
            + estimate_tokens(compact_json(payload))
        )
        assert token_estimate <= 2488 and request.max_output_tokens == 1500
        metrics = await contexts.list_metrics(run_id)
        assert len(metrics) == 1 and metrics[0].token_after <= metrics[0].input_budget
