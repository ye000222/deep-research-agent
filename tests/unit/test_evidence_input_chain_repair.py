"""Offline Reader -> ranked verbatim passages -> real extraction validation."""
# ruff: noqa: RUF001 -- fixtures intentionally preserve real-world Unicode punctuation

import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from app.context.source_selector import SourceWindow, pack_source_windows, select_relevant_blocks
from app.domain.adaptive_scheduler import cheap_triage
from app.domain.document_evidence_context import LocalizationStatus
from app.domain.providers import CanonicalModelResult, TokenUsage, UsageAccuracy
from app.domain.research_tools import ReadPage
from app.domain.source_text_quality import assess_text_quality, decode_html
from app.llm.adapters import ModelGatewayError
from app.services.evidence_extractor import EvidenceExtractorService
from app.tools import web_reader
from app.tools.errors import ToolExecutionError

from tests.unit.test_planner_service import FakeBindingRepository, dependencies

QUESTION = "北斗三号全球卫星导航系统何时正式开通服务？"
ANSWER = "北斗三号全球卫星导航系统于2020年7月31日正式开通服务。"
LAUNCH = "The final BeiDou-3 satellite was successfully launched on June 23, 2020."
BACKGROUND = "背景介绍与导航菜单。这段文字介绍设备和历史背景，不提供事件日期。" * 20


@pytest.mark.parametrize(
    ("encoding", "header", "meta", "body"),
    [
        ("utf-8", "utf-8", "utf-8", ANSWER),
        ("utf-8", "iso-8859-1", "utf-8", ANSWER),
        ("utf-8", "gbk", "gbk", ANSWER),
        ("gbk", "utf-8", "gbk", ANSWER),
        ("gb18030", "utf-8", "gb18030", ANSWER + "𠀀"),
        ("cp1252", "windows-1252", "windows-1252", "The café’s official announcement."),
    ],
)
def test_lossless_charset_selection(encoding, header, meta, body):
    html = f'<html><meta charset="{meta}"><body>{body}</body></html>'
    decoded, metrics = decode_html(html.encode(encoding), header)
    assert decoded == html
    assert metrics["decode_quality_flag"] != "lossy_fallback"


@pytest.mark.parametrize(
    ("title", "body", "rejected"),
    [
        ("有效公告", ANSWER * 12, False),
        ("\ufffd" * 20, ANSWER * 12, False),
        ("\ufffd" * 20, "\ufffd" * 300 + "Ã©" * 150, True),
        ("公告", "\ufffd" * 400, True),
        ("实体", ("官方 &amp; 来源 &quot;公告&quot;：" + ANSWER) * 12, False),
    ],
)
async def test_reader_checks_body_not_title_and_never_calls_public_dns(
    monkeypatch, title, body, rejected
):
    monkeypatch.setattr(web_reader, "_require_public_destination", AsyncMock())
    html = (
        f"<html><head><title>{title}</title></head>"
        f"<body><article><p>{body}</p></article></body></html>"
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, content=html.encode(), headers={"Content-Type": "text/html"}
            )
        )
    ) as client:
        reader = web_reader.PublicWebReader(client)
        if rejected:
            with pytest.raises(ToolExecutionError, match="WEBPAGE_TEXT_QUALITY_REJECTED"):
                await reader.read("https://public.example.org/article")
            assert not any(e["event_type"] == "source.readable" for e in reader.events)
        else:
            page = await reader.read("https://public.example.org/article")
            assert ANSWER in page.clean_text
            if title == "实体":
                assert '官方 & 来源 "公告"' in page.clean_text


@pytest.mark.parametrize(
    ("question", "text", "accept"),
    [
        ("最后一颗北斗三号组网卫星何时成功发射？", LAUNCH * 8, True),
        (
            "最后一颗北斗三号组网卫星何时成功发射？",
            "The final BeiDou satellite arrived at the launch site "
            "and is planned to launch in May. " * 8,
            False,
        ),
        (QUESTION, ANSWER * 12, True),
        (
            QUESTION,
            "北斗三号原子钟性能研究，描述授时精度、频率和参数，不涉及服务日期。" * 20,
            False,
        ),
        (QUESTION, "\ufffd" * 500, False),
    ],
)
def test_current_question_triage_not_final_goal_or_title_truth(question, text, accept):
    result = cheap_triage(
        question=question,
        criteria=(),
        text=text,
        title="公告",
        url="https://publisher.example/article",
    )
    assert result.accepted is accept


@pytest.mark.parametrize("position", [0, 0.5, 0.95])
@pytest.mark.parametrize("paragraphs", [True, False])
def test_answer_anywhere_in_eighty_thousand_chars_keeps_exact_offsets(position, paragraphs):
    sep = "\n\n" if paragraphs else ""
    background = sep.join([BACKGROUND] * 150)
    cut = int(len(background) * position)
    text = background[:cut] + sep + ANSWER + sep + background[cut:]
    windows = pack_source_windows(
        select_relevant_blocks(text, (QUESTION,), ranked=True), max_tokens=1000
    )
    assert any(ANSWER in w.content for w in windows)
    assert sum((len(w.content) + 2) // 3 + 1 for w in windows) <= 1000
    assert all(text[w.start : w.end] == w.content for w in windows)


def test_completed_vs_planned_and_construction_vs_opening_are_not_equivalent():
    prefix = "北斗三号最后一颗卫星计划于五月发射。" + BACKGROUND
    text = prefix + "\n\n" + BACKGROUND * 100 + "\n\n" + LAUNCH
    chosen = select_relevant_blocks(text, ("最后一颗北斗三号卫星何时成功发射？",), ranked=True)
    assert LAUNCH in chosen[0].content
    text = "北斗三号系统建成。" + BACKGROUND * 100 + "\n\n" + ANSWER
    assert ANSWER in select_relevant_blocks(text, (QUESTION,), ranked=True)[0].content


def test_top_two_sections_are_packed_without_fabricated_contiguous_quote():
    text = ANSWER + "\n\n" + BACKGROUND * 20 + "\n\n" + LAUNCH
    windows = pack_source_windows(
        select_relevant_blocks(text, (QUESTION, "最后一颗北斗三号卫星何时成功发射？"), ranked=True),
        max_tokens=800,
    )
    assert any(ANSWER in w.content for w in windows)
    assert any(LAUNCH in w.content for w in windows)
    assert all(text[w.start : w.end] == w.content for w in windows)


class Gateway:
    def __init__(self, mode):
        self.mode, self.calls = mode, []

    async def generate_structured(self, *, request, **_):
        self.calls.append(request)
        text = str(request.content_parts[0].value)
        if self.mode == "malformed":
            payload = {"items": [{"claim": "invalid"}]}
        elif self.mode == "empty" or ANSWER not in text:
            payload = {"items": []}
        else:
            payload = {
                "items": [
                    {
                        "claim": "北斗三号于2019年正式开通服务。"
                        if self.mode == "wrong_claim"
                        else ANSWER,
                        "exact_quote": ANSWER,
                        "relation": "supports",
                        "relevance": 0.6 if self.mode == "low_relevance" else 0.99,
                        "confidence": 0.99,
                    }
                ]
            }
        return CanonicalModelResult(
            parsed_object=payload,
            usage=TokenUsage(
                input_tokens=50, output_tokens=10, total_tokens=60, accuracy=UsageAccuracy.EXACT
            ),
        )


@pytest.mark.parametrize("mode", ["accepted", "empty", "wrong_claim", "low_relevance", "malformed"])
async def test_real_extractor_candidate_validation_and_selected_span(mode):
    binding, cipher = dependencies()
    binding = replace(binding, relevant_chunks_enabled=True)
    gateway = Gateway(mode)
    text = BACKGROUND * 150 + "\n\n" + ANSWER
    page = ReadPage(
        final_url="https://agency.gov/announcement",
        title="官方公告",
        clean_text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        fetched_at=datetime.now(UTC),
        document_type="HTML",
        available_char_count=len(text),
    )
    service = EvidenceExtractorService(FakeBindingRepository(binding), cipher, gateway)
    if mode == "malformed":
        with pytest.raises(ModelGatewayError, match="EVIDENCE_OUTPUT_SCHEMA_INVALID"):
            await service.extract(
                binding.run_id, question=QUESTION, page=page, max_total_tokens=7000
            )
        return
    evidence, _, metrics = await service.extract(
        binding.run_id, question=QUESTION, page=page, max_total_tokens=4000, source_id=uuid4()
    )
    assert ANSWER in str(gateway.calls[0].content_parts[0].value)
    assert metrics["extraction_output_parse_status"] == "schema_valid"
    if mode == "accepted":
        assert evidence[0].accepted
        assert evidence[0].localization_status is LocalizationStatus.LOCATED
        span = evidence[0].canonical_span
        assert span is not None and text[span.char_start : span.char_end] == ANSWER
    elif mode == "empty":
        assert not evidence and metrics["zero_candidate_reason"] == "no_claim_found"
    else:
        assert not evidence[0].accepted
        assert evidence[0].rejection_reason == (
            "claim_quote_entailment_failed"
            if mode == "wrong_claim"
            else "evidence_relevance_below_threshold"
        )


@pytest.mark.parametrize("text", ["�" * 300, "\x01" * 300, "Ã©" * 300])
def test_corrupt_body_quality_signals(text):
    quality = assess_text_quality(text)
    assert not quality.accepted
    assert quality.quality_flag == "corrupt_text"
    assert 0 <= quality.substantive_ratio <= 1


def test_top_one_without_answer_does_not_exclude_top_two():
    first = "北斗三号正式开通服务公告索引, 没有给出日期。"
    text = first + "\n\n" + ANSWER
    start = len(first) + 2
    windows = pack_source_windows(
        [
            SourceWindow(0, len(first), first, 1.0),
            SourceWindow(start, len(text), ANSWER, 0.95),
        ],
        max_tokens=400,
    )
    assert len(windows) == 2 and ANSWER in windows[1].content
    assert all(text[w.start : w.end] == w.content for w in windows)


@pytest.mark.parametrize(
    ("code", "detail", "reason"),
    [
        ("MODEL_OUTPUT_INVALID", "PARSE_EMPTY_RESPONSE", "empty_model_output"),
        ("MODEL_OUTPUT_INVALID", "PARSE_BAD_JSON", "structured_output_parse_failed"),
        ("MODEL_OUTPUT_TRUNCATED", None, "model_output_truncated"),
        ("MODEL_TIMEOUT", None, "unknown"),
    ],
)
async def test_safe_model_output_failure_classification(code, detail, reason):
    binding, cipher = dependencies()

    class FailingGateway:
        async def generate_structured(self, **kwargs):
            raise ModelGatewayError(code, retryable=False, detail_code=detail)

    service = EvidenceExtractorService(FakeBindingRepository(binding), cipher, FailingGateway())
    from app.domain.providers import CanonicalModelRequest

    request = CanonicalModelRequest(
        task_kind="evidence_extraction",
        role="extractor",
        model=binding.model,
        instructions="offline",
        max_output_tokens=100,
        content_parts=(),
        context_manifest_id=uuid4(),
    )
    with pytest.raises(ModelGatewayError) as error:
        await service._timed_model_request(request=request)
    assert error.value.diagnostics["zero_candidate_reason"] == reason


async def test_five_actual_model_calls_can_all_fail_validation_without_fake_success():
    binding, cipher = dependencies()
    binding = replace(binding, relevant_chunks_enabled=True)
    gateway = Gateway("low_relevance")
    service = EvidenceExtractorService(FakeBindingRepository(binding), cipher, gateway)
    text = BACKGROUND + ANSWER
    page = ReadPage(
        final_url="https://agency.gov/report",
        title="公告",
        clean_text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        fetched_at=datetime.now(UTC),
    )
    for _ in range(5):
        evidence, _, _ = await service.extract(
            binding.run_id,
            question=QUESTION,
            page=page,
            max_total_tokens=4000,
        )
        assert evidence and not any(item.accepted for item in evidence)
    assert len(gateway.calls) == 5


async def test_readable_input_without_answer_never_forms_accepted_evidence():
    binding, cipher = dependencies()
    binding = replace(binding, relevant_chunks_enabled=True)
    gateway = Gateway("accepted")
    text = BACKGROUND * 150
    page = ReadPage(
        final_url="https://agency.gov/background",
        title="背景",
        clean_text=text,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        fetched_at=datetime.now(UTC),
    )
    evidence, _, metrics = await EvidenceExtractorService(
        FakeBindingRepository(binding),
        cipher,
        gateway,
    ).extract(binding.run_id, question=QUESTION, page=page, max_total_tokens=4000)
    assert not evidence and metrics["zero_candidate_reason"] == "no_claim_found"
    assert all(ANSWER not in str(r.content_parts[0].value) for r in gateway.calls)
