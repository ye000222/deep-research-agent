from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from app.domain.evidence_extraction_request import EvidenceExtractionRequestFactory
from app.domain.evidence_extractor import (
    EvidenceExtractionEventType,
    EvidenceExtractionStatus,
    EvidenceExtractor,
    EvidenceExtractorProviderResult,
)
from app.domain.reader_executor import (
    ReaderExecutionResult,
    ReaderExecutionStatus,
    ReaderParseStatus,
    ReaderQualityStatus,
)

READER_EXECUTION_ID = UUID("00000000-0000-0000-0000-000000000033")
ROUTING_ID = UUID("00000000-0000-0000-0000-000000000034")
ALIGNMENT_ID = UUID("00000000-0000-0000-0000-000000000035")
RUN_ID = UUID("00000000-0000-0000-0000-000000000036")
GAP_ID = UUID("00000000-0000-0000-0000-000000000037")


class FakeExtractor:
    def __init__(
        self,
        result: EvidenceExtractorProviderResult | None = None,
        error: Exception | None = None,
    ):
        self.result = result
        self.error = error
        self.calls: list[str] = []

    def extract(self, content: str) -> EvidenceExtractorProviderResult:
        self.calls.append(content)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


def _reader_result(status: ReaderExecutionStatus) -> ReaderExecutionResult:
    return ReaderExecutionResult(
        execution_id=READER_EXECUTION_ID,
        routing_id=ROUTING_ID,
        alignment_id=ALIGNMENT_ID,
        source_id="https://industry-report.example/report",
        run_id=RUN_ID,
        question_id="q7",
        gap_id=GAP_ID,
        dimension_key="q7:d2",
        status=status,
        content_available=status is ReaderExecutionStatus.SUCCESS,
        content_length=24 if status is ReaderExecutionStatus.SUCCESS else 0,
        parse_status=ReaderParseStatus.SUCCESS,
        quality_status=ReaderQualityStatus.NOT_EVALUATED,
        failure_reason=None if status is ReaderExecutionStatus.SUCCESS else "reader_failure",
    )


def _request(reader_result: ReaderExecutionResult, content: str = "readable source body"):
    return EvidenceExtractionRequestFactory.create(
        reader_result,
        extraction_request_id=UUID("00000000-0000-0000-0000-000000000038"),
        requirement_type="independent_source",
        content_reference=content,
        execution_reason="Reader content available",
    )


def test_successful_extraction_normalizes_candidate_evidence() -> None:
    reader_result = _reader_result(ReaderExecutionStatus.SUCCESS)
    extractor = FakeExtractor(
        EvidenceExtractorProviderResult(
            extracted_claims=(" claim one ",),
            evidence_items=(" evidence one ", "", " evidence one ", "evidence two"),
        )
    )

    result = EvidenceExtractor.execute(
        _request(reader_result),
        reader_result=reader_result,
        provider_adapter=extractor,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert result.status is EvidenceExtractionStatus.SUCCESS
    assert result.candidate_evidence_count == 2
    assert result.extracted_claims == ("claim one",)
    assert result.evidence_items == ("evidence one", "evidence two")
    assert result.events[0].event_type is EvidenceExtractionEventType.STARTED
    assert result.events[-1].event_type is EvidenceExtractionEventType.COMPLETED
    assert extractor.calls == ["readable source body"]


def test_reader_failure_blocks_extractor() -> None:
    reader_result = _reader_result(ReaderExecutionStatus.FAILED)
    extractor = FakeExtractor(
        EvidenceExtractorProviderResult(("claim",), ("evidence",))
    )

    result = EvidenceExtractor.execute(
        _request(reader_result),
        reader_result=reader_result,
        provider_adapter=extractor,
    )

    assert result.status is EvidenceExtractionStatus.BLOCKED
    assert result.failure_reason == "reader_result_not_success"
    assert extractor.calls == []


def test_empty_extraction_returns_empty() -> None:
    reader_result = _reader_result(ReaderExecutionStatus.SUCCESS)
    extractor = FakeExtractor(EvidenceExtractorProviderResult((), ()))

    result = EvidenceExtractor.execute(
        _request(reader_result),
        reader_result=reader_result,
        provider_adapter=extractor,
    )

    assert result.status is EvidenceExtractionStatus.EMPTY
    assert result.candidate_evidence_count == 0
    assert result.failure_reason == "no_candidate_evidence"


def test_extraction_exception_returns_failed() -> None:
    reader_result = _reader_result(ReaderExecutionStatus.SUCCESS)
    extractor = FakeExtractor(error=TimeoutError("extractor timeout"))

    result = EvidenceExtractor.execute(
        _request(reader_result),
        reader_result=reader_result,
        provider_adapter=extractor,
    )

    assert result.status is EvidenceExtractionStatus.FAILED
    assert result.failure_reason == "timeout"
    assert result.events[-1].event_type is EvidenceExtractionEventType.FAILED


def test_extraction_chain_preserves_reader_gap_and_dimension_links() -> None:
    reader_result = _reader_result(ReaderExecutionStatus.SUCCESS)
    extractor = FakeExtractor(EvidenceExtractorProviderResult(("claim",), ("evidence",)))

    result = EvidenceExtractor.execute(
        _request(reader_result),
        reader_result=reader_result,
        provider_adapter=extractor,
    )

    assert result.reader_execution_id == READER_EXECUTION_ID
    assert result.gap_id == GAP_ID
    assert result.dimension_key == "q7:d2"
    assert result.routing_id == ROUTING_ID
    assert result.alignment_id == ALIGNMENT_ID
    assert result.extraction_id == result.events[0].extraction_id
