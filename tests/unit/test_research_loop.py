import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from app.domain.providers import TokenUsage, UsageAccuracy
from app.domain.research_tools import ReadPage, ReusablePageRef, SearchResult
from app.domain.source_policy import normalize_source_url
from app.infrastructure.db.research_tools import (
    _NON_CONSUMING_ITERATION_OUTCOMES,
    EvidenceModelBudget,
    IterationEvaluation,
    PageBudgetReservation,
    ResearchTarget,
    _budget_exhaustion_reason,
    _executed_query_families,
    _fair_provider_request_allowance,
    _mark_query_family_executed,
    _p1_variant_schedule_key,
    _protected_page_schedule_key,
    _quality_enrichment_needed,
    _quality_gate_met_from_snapshot,
    _quality_repair_targets,
    _query_family_capacity_remaining,
    _query_family_order,
    _query_space_exhausted_for_freeze,
    _question_low_gain_streak,
    _requirements_satisfied,
    _requires_independent_sources,
    _research_attempt_consumes_iteration,
    _search_acquisition_budget_exhausted,
    _search_query_for_attempt,
    _select_unmet_requirement,
    _source_failure_feedback,
    _source_hint_for_requirement,
    _zero_yield_retry_deprioritized,
)
from app.llm.adapters import ModelGatewayError
from app.services.research_loop import (
    ResearchLoopService,
    _deduplicate_search_results,
    _is_read_candidate,
    _prioritize_search_results,
    _topic_relevance_ok,
)
from app.tools.errors import ToolExecutionError


def test_source_failure_feedback_is_run_wide_but_domain_exclusion_is_bounded() -> None:
    failed_urls, excluded_owners = _source_failure_feedback(
        [
            (
                "source.rejected",
                {
                    "url": "https://papers.example.org/a",
                    "requested_url": "https://doi.org/10.1000/a",
                    "error_code": "WEBPAGE_REQUEST_REJECTED",
                },
            ),
            (
                "source.triage_rejected",
                {
                    "url": "https://www.example.org/b",
                    "reason": "body_too_short",
                },
            ),
            (
                "source.triage_rejected",
                {
                    "url": "https://useful.example.net/c",
                    "reason": "topic_mismatch",
                },
            ),
            (
                "source.rejected",
                {
                    "url": "https://transient.example.com/d",
                    "error_code": "WEBPAGE_NETWORK_ERROR",
                },
            ),
        ]
    )

    assert normalize_source_url("https://papers.example.org/a") in failed_urls
    assert normalize_source_url("https://doi.org/10.1000/a") in failed_urls
    assert normalize_source_url("https://www.example.org/b") in failed_urls
    assert normalize_source_url("https://useful.example.net/c") not in failed_urls
    assert normalize_source_url("https://transient.example.com/d") in failed_urls
    assert excluded_owners == ("example.org",)


def test_p1_variant_schedule_prioritizes_zero_coverage_over_productive_retry() -> None:
    missing_p1 = _p1_variant_schedule_key(
        attempts=2,
        coverage=0.0,
        priority=1,
        question_id="q3",
    )
    partially_covered_p1 = _p1_variant_schedule_key(
        attempts=1,
        coverage=0.5,
        priority=1,
        question_id="q2",
    )

    assert missing_p1 < partially_covered_p1


def test_quality_gate_snapshot_requires_all_hard_thresholds() -> None:
    snapshot = {
        "coverage": 0.90,
        "priority_one_coverage": 0.90,
        "source_quality": 0.80,
        "cross_validation": 0.80,
        "critical_gaps": 0,
    }
    assert _quality_gate_met_from_snapshot(snapshot) is True
    snapshot["critical_gaps"] = 1
    assert _quality_gate_met_from_snapshot(snapshot) is False


def test_quality_gate_snapshot_does_not_treat_empty_metrics_as_success() -> None:
    assert _quality_gate_met_from_snapshot({}) is False


def test_provider_requests_are_shared_across_untouched_questions() -> None:
    assert _fair_provider_request_allowance(24, 8) == 3
    assert _fair_provider_request_allowance(2, 8) == 1
    # Once every question has had a first pass, a retry is still bounded.
    assert _fair_provider_request_allowance(16, 0) == 3
    # The Standard 48-attempt pool leaves half its capacity for P1 variants
    # after eight questions receive one three-provider first pass.
    assert 48 - (8 * _fair_provider_request_allowance(48, 8)) == 24


def test_search_attempt_limit_allows_cached_candidate_drain() -> None:
    assert _search_acquisition_budget_exhausted(
        "provider_request_budget_exhausted"
    )
    assert _search_acquisition_budget_exhausted("logical_query_budget_exhausted")
    assert not _search_acquisition_budget_exhausted("token_budget_exhausted")


def test_protected_page_mode_keeps_untouched_questions_ahead_of_retries() -> None:
    untouched = _protected_page_schedule_key(
        attempts=0,
        coverage=0.0,
        priority=3,
        question_id="q8",
    )
    retried_p1 = _protected_page_schedule_key(
        attempts=4,
        coverage=0.0,
        priority=1,
        question_id="q1",
    )
    assert untouched < retried_p1


def test_zero_yield_p1_is_deprioritized_but_not_made_terminal() -> None:
    assert _zero_yield_retry_deprioritized(
        search_budget_exhausted=False,
        attempts=2,
        coverage=0.0,
        accepted_evidence=0,
        is_corroboration_target=False,
    )
    assert not _zero_yield_retry_deprioritized(
        search_budget_exhausted=False,
        attempts=2,
        coverage=0.5,
        accepted_evidence=1,
        is_corroboration_target=True,
    )
    # Cached candidates must remain drainable after search acquisition stops.
    assert not _zero_yield_retry_deprioritized(
        search_budget_exhausted=True,
        attempts=2,
        coverage=0.0,
        accepted_evidence=0,
        is_corroboration_target=False,
    )


def test_zero_yield_question_is_not_frozen_with_untried_query_families() -> None:
    assert not _query_space_exhausted_for_freeze(
        executed_families={"scope", "authoritative"},
        question="A question",
    )
    assert _query_space_exhausted_for_freeze(
        executed_families={"scope", "authoritative", "alternate", "contradiction"},
        question="A question",
    )


def test_p1_protected_lane_uses_all_four_query_families() -> None:
    assert _query_family_capacity_remaining(
        question="A question",
        attempted_families=3,
    )
    assert not _query_family_capacity_remaining(
        question="A question",
        attempted_families=4,
    )


def test_low_gain_streak_does_not_leak_between_questions() -> None:
    snapshot = {
        "low_information_gain_streak": 2,
        "low_information_gain_streak_by_question": {"q1": 2},
    }
    assert _question_low_gain_streak(snapshot, "q1") == 2
    assert _question_low_gain_streak(snapshot, "q2") == 0
    assert _question_low_gain_streak({"low_information_gain_streak": 1}, "q1") == 1


def test_source_quality_or_cross_validation_gap_requires_enrichment() -> None:
    assert _quality_enrichment_needed(source_quality=0.74, cross_validation=1.0) is True
    assert _quality_enrichment_needed(source_quality=0.80, cross_validation=0.0) is True
    assert _quality_enrichment_needed(source_quality=0.75, cross_validation=0.70) is False


def test_quality_repair_targets_only_dimensions_that_can_improve_gate() -> None:
    coverage_map = [
        {
            "dimension_key": "q1",
            "question": "Needs corroboration",
            "priority": 1,
            "coverage": 0.5,
            "accepted_evidence": 1,
            "independent_sources": 1,
            "acceptance_criteria": ["two sources"],
            "missing_reasons": [],
            "requirement_statuses": [
                {
                    "dimension_key": "q1:d1",
                    "accepted_evidence": 1,
                    "independent_sources": 1,
                    "required_sources": 2,
                    "max_source_reliability": 0.70,
                }
            ],
        },
        {
            "dimension_key": "q3",
            "question": "Already complete",
            "priority": 1,
            "coverage": 1.0,
            "accepted_evidence": 15,
            "independent_sources": 7,
            "acceptance_criteria": ["one source"],
            "missing_reasons": [],
            "requirement_statuses": [
                {
                    "dimension_key": "q3:d1",
                    "accepted_evidence": 15,
                    "independent_sources": 7,
                    "required_sources": 1,
                    "max_source_reliability": 0.90,
                }
            ],
        },
    ]

    targets = _quality_repair_targets(  # type: ignore[arg-type]
        coverage_map, source_quality=0.70, cross_validation=0.50
    )

    assert targets == {
        "q1": ["cross_validation:q1:d1", "source_quality:q1:d1"]
    }


@pytest.mark.parametrize(
    "criterion",
    ["市场规模预测", "至少两个独立来源", "market share forecast"],
)
def test_high_risk_claims_require_independent_sources(criterion: str) -> None:
    assert _requires_independent_sources(criterion) is True


def test_ordinary_requirements_do_not_force_an_unplanned_second_source() -> None:
    assert (
        _requirements_satisfied(
            "q1",
            ["Describe one documented deployment"],
            {"q1:d1": (2, 1)},
            accepted_for_question=2,
        )
        is True
    )


def test_high_risk_requirement_still_requires_independent_sources() -> None:
    assert (
        _requirements_satisfied(
            "q1",
            ["Compare market share forecast"],
            {"q1:d1": (2, 1)},
            accepted_for_question=2,
        )
        is False
    )


def test_discovered_high_risk_claim_prevents_single_source_completion() -> None:
    assert (
        _requirements_satisfied(
            "q1",
            ["Describe one documented deployment"],
            {"q1:d1": (2, 1)},
            accepted_for_question=2,
            high_risk_dimension_keys={"q1:d1"},
        )
        is False
    )


def test_retry_query_targets_unmet_requirement_and_changes_shape() -> None:
    first = _search_query_for_attempt(
        "Which industrial inspection routes are mainstream?",
        ["industrial inspection technology", "machine vision inspection"],
        ["Compare the core principle and applicable scenario"],
        attempt_index=0,
    )
    second = _search_query_for_attempt(
        "Which industrial inspection routes are mainstream?",
        ["industrial inspection technology", "machine vision inspection"],
        ["Compare the core principle and applicable scenario"],
        attempt_index=1,
        unmet_criterion="Compare the core principle and applicable scenario",
    )
    third = _search_query_for_attempt(
        "Which industrial inspection routes are mainstream?",
        ["industrial inspection technology", "machine vision inspection"],
        ["Compare the core principle and applicable scenario"],
        attempt_index=2,
        unmet_criterion="Compare the core principle and applicable scenario",
    )

    assert first == "industrial inspection technology"
    assert second != first
    assert "applicable scenario" in second
    assert "official technical paper" in second
    assert third not in {first, second}
    assert "case study" in third


def test_quality_enrichment_query_explicitly_targets_authoritative_sources() -> None:
    query = _search_query_for_attempt(
        "工业视觉缺陷检测趋势",
        ["工业视觉缺陷检测趋势"],
        ["趋势预测"],
        attempt_index=0,
        require_authoritative_source=True,
    )

    assert "官方统计" in query
    assert "行业协会" in query


def test_query_variants_switch_core_angle_instead_of_cycling_suffixes() -> None:
    first_suffix = _search_query_for_attempt(
        "工业视觉表面缺陷检测",
        ["工业视觉表面缺陷检测"],
        ["比较公开基准性能"],
        attempt_index=2,
    )
    after_suffixes = _search_query_for_attempt(
        "工业视觉表面缺陷检测",
        ["工业视觉表面缺陷检测"],
        ["比较公开基准性能"],
        attempt_index=9,
    )

    assert "综述 技术论文" in first_suffix
    assert "基准 对比 指标 数据" in after_suffixes
    assert after_suffixes != first_suffix


def test_metric_requirement_targets_benchmark_sources() -> None:
    hint = _source_hint_for_requirement("需给出公开数据集上的性能指标或对比结果")

    assert "公开基准" in hint
    assert "数据集" in hint


def test_vendor_requirement_targets_manufacturer_product_pages() -> None:
    hint = _source_hint_for_requirement("原文列出的厂商名称及其缺陷检测产品")

    assert "厂商官网" in hint
    assert "产品页" in hint


def test_unmet_requirement_prioritizes_zero_coverage_before_corroboration() -> None:
    statuses: list[object] = [
        {
            "dimension_key": "q1:d1",
            "criterion": "已有一个来源, 仍需交叉验证",
            "coverage": 0.5,
            "accepted_evidence": 2,
            "independent_sources": 1,
        },
        {
            "dimension_key": "q1:d2",
            "criterion": "尚无证据的维度",
            "coverage": 0.0,
            "accepted_evidence": 0,
            "independent_sources": 0,
        },
    ]

    assert _select_unmet_requirement(statuses) == ("q1:d2", "尚无证据的维度")


def test_forecast_requirement_targets_authoritative_roadmaps() -> None:
    hint = _source_hint_for_requirement("需引用机构或论文的趋势预测")

    assert "行业协会" in hint
    assert "机构预测" in hint


@pytest.mark.parametrize(
    "outcome",
    [
        "zero_results",
        "unreadable",
        "no_evidence",
        "provider_error",
        "yield_question",
        "budget_exhausted",
    ],
)
def test_non_productive_attempt_does_not_consume_research_round(outcome: str) -> None:
    assert outcome in _NON_CONSUMING_ITERATION_OUTCOMES
    assert _research_attempt_consumes_iteration(outcome) is False


def test_evidence_attempt_consumes_research_round() -> None:
    assert _research_attempt_consumes_iteration("evidence_gained") is True
    assert _research_attempt_consumes_iteration("provider_error", technical_outcome=True) is False


class FakeRepository:
    def __init__(self) -> None:
        self.target: ResearchTarget | None = ResearchTarget(
            plan_version=1,
            question_id="q1",
            question="Which routes are used for industrial inspection?",
            query="industrial inspection technology routes",
            gap_id=uuid4(),
            tool_call_id=uuid4(),
            source_id_seed=uuid4(),
        )
        self.extraction_failures: list[str] = []
        self.finished = False
        self.finish_calls = 0
        self.stop_after = 1
        self.last_attempt_outcome: str | None = None
        self.allow_search_failure = False
        self.model_budget_allowed = True
        self.model_budget_outcome = "execute"
        self.model_budget_max_call = 18_000
        self.minimum_call_seen: int | None = None
        self.page_budget_granted = 3
        self.record_page_result = (1, 1)
        self.released_page_slots = 0
        self.page_failures = 0
        self.fetch_budget_settlements = 0
        self.duplicate_page = False
        self.selection_events: list[tuple[str, str | None]] = []
        self.extraction_slot_allowed = True
        self.token_reservations: dict[UUID, int] = {}
        self.token_reservation_granted = True
        self.settled_reservations: list[tuple[UUID, int]] = []
        self.uncertain_reservations: list[UUID] = []
        self.released_reservations: list[UUID] = []
        self.research_still_active = False

    async def prepare_target(
        self, *args: object, **kwargs: object
    ) -> ResearchTarget | None:
        return self.target

    async def research_phase_active(self, *args: object, **kwargs: object) -> bool:
        return self.research_still_active

    async def record_search_results(self, *args: object, **kwargs: object) -> None:
        return None

    async def evidence_model_budget(self, *args: object, **kwargs: object) -> EvidenceModelBudget:
        self.minimum_call_seen = kwargs.get("minimum_call")
        return EvidenceModelBudget(
            allowed=self.model_budget_allowed,
            max_call_tokens=self.model_budget_max_call,
            remaining_tokens=25_000,
            writer_reserve_tokens=10_000,
            outcome=self.model_budget_outcome,
        )

    async def reserve_page_slots(self, *args: object, **kwargs: object) -> PageBudgetReservation:
        return PageBudgetReservation(granted=self.page_budget_granted, remaining=0)

    async def record_page_fetched(self, *args: object, **kwargs: object) -> None:
        self.fetch_budget_settlements += 1
        return None

    async def reserve_extraction_slot(self, *args: object, **kwargs: object) -> bool:
        return self.extraction_slot_allowed

    async def release_extraction_slot(self, *args: object, **kwargs: object) -> None:
        return None

    async def record_triage_rejection(self, *args: object, **kwargs: object) -> None:
        return None

    async def reserve_model_tokens(self, *args: object, **kwargs: object) -> object:
        from app.infrastructure.db.research_tools import ModelTokenReservation

        if not self.token_reservation_granted:
            return ModelTokenReservation(False, 0, "insufficient_budget")
        attempt_id = kwargs["attempt_id"]  # type: ignore[index]
        assert attempt_id not in self.token_reservations, "attempt reserved twice"
        self.token_reservations[attempt_id] = int(kwargs["estimated_input"]) + int(  # type: ignore[index]
            kwargs["max_output"]
        )
        return ModelTokenReservation(True, self.token_reservations[attempt_id], "reserved")

    async def settle_model_reservation(self, *args: object, **kwargs: object) -> bool:
        self.settled_reservations.append(
            (kwargs["attempt_id"], int(kwargs["actual_total"]))  # type: ignore[index]
        )
        return True

    async def mark_model_reservation_uncertain(self, *args: object, **kwargs: object) -> bool:
        self.uncertain_reservations.append(kwargs["attempt_id"])  # type: ignore[index]
        return True

    async def release_model_reservation(self, *args: object, **kwargs: object) -> bool:
        self.released_reservations.append(kwargs["attempt_id"])  # type: ignore[index]
        return True

    async def release_page_slots(self, *args: object, **kwargs: object) -> None:
        self.released_page_slots += int(kwargs.get("count", 0))

    async def record_tool_failure(self, *args: object, **kwargs: object) -> None:
        if not self.allow_search_failure:
            raise AssertionError("search must not fail")

    async def record_extraction_started(self, *args: object, **kwargs: object) -> None:
        return None

    async def record_evidence_selection_event(self, *args: object, **kwargs: object) -> None:
        self.selection_events.append(
            (str(kwargs["stage"]), kwargs.get("reason"))
        )

    async def page_already_processed(self, *args: object, **kwargs: object) -> bool:
        return self.duplicate_page

    async def record_page_failure(self, *args: object, **kwargs: object) -> None:
        self.page_failures += 1
        self.fetch_budget_settlements += 1

    async def record_extraction_failure(self, *args: object, **kwargs: object) -> None:
        self.extraction_failures.append(str(kwargs["error_code"]))
        attempt_id = kwargs.get("attempt_id")
        if kwargs.get("usage") is None:
            if attempt_id is not None:
                self.uncertain_reservations.append(attempt_id)
        elif attempt_id is not None:
            self.settled_reservations.append((attempt_id, kwargs["usage"].total_tokens))

    async def record_page(self, *args: object, **kwargs: object) -> tuple[int, int]:
        attempt_id = kwargs.get("attempt_id")
        if attempt_id is not None:
            self.settled_reservations.append((attempt_id, kwargs["usage"].total_tokens))
        return self.record_page_result

    async def finish_iteration(self, *args: object, **kwargs: object) -> IterationEvaluation:
        self.last_attempt_outcome = str(kwargs.get("attempt_outcome"))
        self.finished = True
        self.finish_calls += 1
        should_continue = self.finish_calls < self.stop_after
        return IterationEvaluation(
            continue_research=should_continue,
            decision="continue_plan" if should_continue else "ready_to_write",
            stop_reason=None if should_continue else "writer_not_implemented",
            question_status="researched",
        )


class FakeSearch:
    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: str, *, limit: int) -> list[SearchResult]:
        self.calls += 1
        return [
            SearchResult(title=f"Source {rank}", url=f"https://example.com/{rank}", rank=rank)
            for rank in (1, 2, 3)
        ]


class FailingSearch:
    async def search(self, query: str, *, limit: int) -> list[SearchResult]:
        raise ToolExecutionError("SEARCH_PROVIDER_DEGRADED", retryable=True)


class FakeReader:
    def __init__(self) -> None:
        self.calls = 0

    async def read(self, url: str) -> ReadPage:
        self.calls += 1
        return ReadPage(
            final_url=url,
            title=url,
            clean_text="Evidence-bearing public page content. " * 5,
            content_hash="a" * 64,
            fetched_at=datetime.now(UTC),
        )


class FirstThreePagesFailReader(FakeReader):
    async def read(self, url: str) -> ReadPage:
        self.calls += 1
        if self.calls <= 3:
            raise ToolExecutionError("WEBPAGE_PROVIDER_UNAVAILABLE", retryable=False)
        return ReadPage(
            final_url=url,
            title=url,
            clean_text="Evidence-bearing public page content. " * 5,
            content_hash="b" * 64,
            fetched_at=datetime.now(UTC),
        )


class FiveResultSearch:
    async def search(self, query: str, *, limit: int) -> list[SearchResult]:
        return [
            SearchResult(title=f"Source {rank}", url=f"https://example.com/{rank}", rank=rank)
            for rank in range(1, 6)
        ]


class FakeExtractor:
    def __init__(self) -> None:
        self.calls = 0
        self.error_code = "EVIDENCE_OUTPUT_SCHEMA_INVALID"
        self.source_ids: list[object] = []
        self.minimum_total = 4_000

    def estimate_minimum_request_tokens(self, **kwargs: object) -> object:
        from app.domain.research_budget import MinimumCallEstimate

        return MinimumCallEstimate(
            fixed_tokens=1_000,
            min_source_tokens=1_000,
            min_output_tokens=768,
            safety_margin_tokens=1_232,
            total_tokens=self.minimum_total,
        )

    async def extract(self, *args: object, **kwargs: object) -> tuple[object, ...]:
        self.calls += 1
        self.source_ids.append(kwargs["source_id"])
        if self.calls == 2:
            raise ModelGatewayError(
                self.error_code,
                retryable=self.error_code != "EVIDENCE_OUTPUT_SCHEMA_INVALID",
            )
        return (
            [],
            TokenUsage(
                input_tokens=1,
                output_tokens=1,
                total_tokens=2,
                accuracy=UsageAccuracy.EXACT,
            ),
            {"source_chars": 200, "selected_chars": 200, "truncated": False},
        )


class AlwaysInvalidExtractor:
    def estimate_minimum_request_tokens(self, **kwargs: object) -> object:
        from app.domain.research_budget import MinimumCallEstimate

        return MinimumCallEstimate(1000, 1000, 768, 1232, 4000)

    async def extract(self, *args: object, **kwargs: object) -> tuple[object, ...]:
        raise ModelGatewayError(
            "MODEL_OUTPUT_INVALID",
            retryable=False,
            detail_code="OUTPUT_INVALID_PROMPT_JSON_FINISH_LENGTH_CHARS_6000",
        )


class FakeArtifacts:
    async def save_page(self, run_id: object, source_id: object, text: str) -> str:
        return "runs/test/source.txt"

    async def read_text(self, artifact_uri: str) -> str:
        return "Previously fetched evidence-bearing page content. " * 5


class FailingArtifacts(FakeArtifacts):
    async def save_page(self, run_id: object, source_id: object, text: str) -> str:
        raise OSError("artifact unavailable")


@pytest.mark.asyncio
async def test_empty_scheduler_pass_reenters_research_when_phase_is_active() -> None:
    repository = FakeRepository()
    repository.target = None
    repository.research_still_active = True
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        FakeReader(),
        FakeExtractor(),
        FakeArtifacts(),
    )

    result = await service.run_one_iteration("run-1", worker_task_id="w1")

    assert result.continue_research is True
    assert result.decision == "scheduler_advanced"
    assert result.pages_read == 0


@pytest.mark.asyncio
async def test_empty_scheduler_pass_enters_writer_after_durable_phase_change() -> None:
    repository = FakeRepository()
    repository.target = None
    repository.research_still_active = False
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        FakeReader(),
        FakeExtractor(),
        FakeArtifacts(),
    )

    result = await service.run_one_iteration("run-1", worker_task_id="w1")

    assert result.continue_research is False
    assert result.decision == "no_pending_question"
    assert result.pages_read == 0


@pytest.mark.asyncio
async def test_minimum_call_guard_prevents_unaffordable_extraction() -> None:
    repository = FakeRepository()
    repository.model_budget_max_call = 2_000  # below the assembled minimum
    extractor = FakeExtractor()
    extractor.minimum_total = 4_000
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        FakeReader(),
        extractor,
        FakeArtifacts(),
    )
    result = await service.run_one_iteration("run-1", worker_task_id="w1")

    # The assembled request could not fit the call allowance, so no page was
    # read and no model call was issued or reserved.
    assert repository.minimum_call_seen == 4_000
    assert extractor.calls == 0
    assert not repository.token_reservations
    assert result.pages_read == 0


@pytest.mark.asyncio
async def test_artifact_failure_releases_model_and_page_reservations() -> None:
    repository = FakeRepository()
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        FakeReader(),
        FakeExtractor(),
        FailingArtifacts(),
    )

    with pytest.raises(OSError, match="artifact unavailable"):
        await service.run_one_iteration("run-1", worker_task_id="w1")

    assert len(repository.released_reservations) == 1
    assert repository.uncertain_reservations == []
    assert repository.released_page_slots == repository.page_budget_granted


@pytest.mark.asyncio
async def test_question_budget_yield_is_explicit_and_does_not_reenter_search() -> None:
    repository = FakeRepository()
    repository.model_budget_allowed = False
    repository.model_budget_outcome = "yield_question"
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        FakeReader(),
        FakeExtractor(),
        FakeArtifacts(),
    )

    await service.run_one_iteration("run-1", worker_task_id="w1")

    assert repository.last_attempt_outcome == "yield_question"
    assert repository.finish_calls == 1


@pytest.mark.asyncio
async def test_minimum_call_estimate_is_forwarded_under_normal_budget() -> None:
    repository = FakeRepository()
    extractor = FakeExtractor()
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        FakeReader(),
        extractor,
        FakeArtifacts(),
    )
    result = await service.run_one_iteration("run-1", worker_task_id="w1")

    assert repository.minimum_call_seen == 4_000
    assert extractor.calls >= 1
    assert result.accepted_evidence >= 0
    assert ("selection_started", None) in repository.selection_events
    assert ("selection_selected", None) in repository.selection_events


@pytest.mark.asyncio
async def test_zero_candidate_page_rotates_without_reading_second_page() -> None:
    repository = FakeRepository()
    repository.record_page_result = (0, 0)
    extractor = FakeExtractor()
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        FakeReader(),
        extractor,
        FakeArtifacts(),
    )

    result = await service.run_one_iteration(uuid4(), worker_task_id="worker-1")

    assert result.pages_read == 1
    assert result.accepted_evidence == 0
    assert extractor.calls == 1
    assert repository.released_page_slots == 2


@pytest.mark.asyncio
async def test_first_pass_uses_one_page_before_giving_other_questions_a_turn() -> None:
    repository = FakeRepository()
    repository.target = replace(repository.target, first_pass=True)
    extractor = FakeExtractor()
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        FakeReader(),
        extractor,
        FakeArtifacts(),
    )
    await service.run_one_iteration(uuid4(), worker_task_id="worker-1")
    assert extractor.calls == 1
    assert repository.released_page_slots == 2


@pytest.mark.asyncio
async def test_priority_one_first_pass_can_read_second_candidate() -> None:
    repository = FakeRepository()
    repository.target = replace(repository.target, first_pass=True, priority=1)
    extractor = FakeExtractor()
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        FakeReader(),
        extractor,
        FakeArtifacts(),
    )

    await service.run_one_iteration(uuid4(), worker_task_id="worker-1")

    assert extractor.calls == 2
    assert repository.released_page_slots == 1


@pytest.mark.asyncio
async def test_schema_failure_is_isolated_to_page_and_iteration_finishes() -> None:
    repository = FakeRepository()
    extractor = FakeExtractor()
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        FakeReader(),
        extractor,
        FakeArtifacts(),
    )

    outcome = await service.run_iteration(uuid4(), worker_task_id="worker-1")

    assert outcome == "research_stopped:ready_to_write:iterations=1:pages=2:accepted=1"
    assert repository.extraction_failures == ["EVIDENCE_OUTPUT_SCHEMA_INVALID"]
    assert len(extractor.source_ids) == 2
    assert all(isinstance(source_id, UUID) for source_id in extractor.source_ids)
    assert repository.finished is True


@pytest.mark.asyncio
async def test_iteration_clamps_page_batch_to_reserved_budget() -> None:
    repository = FakeRepository()
    repository.page_budget_granted = 1
    reader = FakeReader()
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        reader,
        FakeExtractor(),
        FakeArtifacts(),
    )

    outcome = await service.run_iteration(uuid4(), worker_task_id="worker-1")

    assert outcome == "research_stopped:ready_to_write:iterations=1:pages=1:accepted=1"
    assert reader.calls == 1


@pytest.mark.asyncio
async def test_unreadable_top_results_do_not_block_later_readable_candidate() -> None:
    repository = FakeRepository()
    repository.page_budget_granted = 1
    reader = FirstThreePagesFailReader()
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FiveResultSearch(),
        reader,
        FakeExtractor(),
        FakeArtifacts(),
    )

    outcome = await service.run_iteration(uuid4(), worker_task_id="worker-1")

    assert outcome == "research_stopped:ready_to_write:iterations=1:pages=1:accepted=1"
    assert reader.calls == 4
    assert repository.page_failures == 3


@pytest.mark.asyncio
async def test_evaluator_continues_without_manual_resume() -> None:
    repository = FakeRepository()
    repository.stop_after = 2
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        FakeReader(),
        FakeExtractor(),
        FakeArtifacts(),
    )

    outcome = await service.run_iteration(uuid4(), worker_task_id="worker-1")

    assert outcome == "research_stopped:ready_to_write:iterations=2:pages=4:accepted=3"
    assert repository.finish_calls == 2


@pytest.mark.asyncio
async def test_retryable_model_failure_preserves_run_for_durable_retry() -> None:
    repository = FakeRepository()
    extractor = FakeExtractor()
    extractor.error_code = "MODEL_TIMEOUT"
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        FakeReader(),
        extractor,
        FakeArtifacts(),
    )

    with pytest.raises(ModelGatewayError) as raised:
        await service.run_iteration(uuid4(), worker_task_id="worker-1")

    assert raised.value.code == "MODEL_TIMEOUT"
    assert repository.extraction_failures == ["MODEL_TIMEOUT"]
    assert repository.released_page_slots == 1
    assert repository.finished is False


@pytest.mark.asyncio
async def test_repeated_structured_extraction_failure_opens_capability_circuit() -> None:
    repository = FakeRepository()
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        FakeReader(),
        AlwaysInvalidExtractor(),
        FakeArtifacts(),
    )

    with pytest.raises(ModelGatewayError) as raised:
        await service.run_one_iteration(uuid4(), worker_task_id="worker-1")

    assert raised.value.code == "MODEL_CAPABILITY_INSUFFICIENT"
    assert raised.value.detail_code == "EVIDENCE_EXTRACTOR_CIRCUIT_OPEN_AFTER_2_SOURCES"
    assert repository.extraction_failures == [
        "MODEL_OUTPUT_INVALID",
        "MODEL_OUTPUT_INVALID",
    ]
    assert repository.finished is False


def test_open_pdfs_are_read_before_paywalls_after_pdf_reader_is_enabled() -> None:
    results = [
        SearchResult(title="Publisher", url="https://www.sciencedirect.com/article/1", rank=1),
        SearchResult(title="PDF", url="https://example.com/paper.pdf", rank=2),
        SearchResult(title="Arxiv", url="https://arxiv.org/abs/2109.11304", rank=3),
        SearchResult(title="Public page", url="https://example.org/article", rank=4),
    ]

    ordered = _prioritize_search_results(results)

    assert [item.title for item in ordered] == [
        "Arxiv",
        "PDF",
        "Public page",
        "Publisher",
    ]


def test_readable_public_result_beats_equally_relevant_restricted_academic() -> None:
    results = [
        SearchResult(
            title="Industrial defect detection review",
            url="https://link.springer.com/article/review",
            snippet="Industrial defect detection methods review.",
            rank=1,
        ),
        SearchResult(
            title="Industrial defect detection review",
            url="https://source.asnt.org/industrial-defect-detection-review/",
            snippet="Industrial defect detection methods review.",
            rank=2,
        ),
    ]

    ordered = _prioritize_search_results(
        results,
        query="industrial defect detection methods review",
        acceptance_criteria=("原文描述工业缺陷检测方法",),
    )

    assert ordered[0].url.startswith("https://source.asnt.org/")


def test_vendor_requirement_prioritizes_official_manufacturer_over_news() -> None:
    results = [
        SearchResult(
            title="工业视觉缺陷检测厂商与产品盘点",
            url="https://it.sohu.com/a/123",
            snippet="介绍工业视觉缺陷检测厂商及产品平台。",
            rank=1,
        ),
        SearchResult(
            title="DeepV AI Defect Detection Platform",
            url="https://www.deepvai.com/products/inspection",
            snippet="Manufacturer product platform for industrial defect inspection.",
            rank=7,
        ),
    ]

    ordered = _prioritize_search_results(
        results,
        query="工业视觉缺陷检测 厂商 产品 平台",
        alternate_query="industrial defect inspection vendors products platform",
        acceptance_criteria=("原文列出厂商名称与其缺陷检测产品名称",),
    )

    assert ordered[0].url.startswith("https://www.deepvai.com/")


def test_source_owner_diversity_beats_a_repeated_domain() -> None:
    results = [
        SearchResult(title="Repeated", url="https://www.example.org/second", rank=1),
        SearchResult(title="Independent", url="https://public.example.net/report", rank=2),
    ]

    ordered = _prioritize_search_results(
        results,
        query="industrial inspection report",
        used_owner_keys={"example.org"},
    )

    assert ordered[0].title == "Independent"


def test_source_owner_diversity_limits_one_domain_in_first_batch() -> None:
    results = [
        SearchResult(title=f"Repeated {rank}", url=f"https://example.org/{rank}", rank=rank)
        for rank in range(1, 5)
    ]
    results.append(
        SearchResult(title="Independent", url="https://nist.gov/report", rank=5)
    )

    ordered = _prioritize_search_results(results)

    assert [result.title for result in ordered[:3]] == [
        "Independent",
        "Repeated 1",
        "Repeated 2",
    ]


def test_weakly_related_academic_result_does_not_displace_direct_match() -> None:
    results = [
        SearchResult(
            title="Vision navigation for autonomous aircraft",
            url="https://arxiv.org/abs/1234.5678",
            snippet="A general vision survey.",
            rank=1,
        ),
        SearchResult(
            title="Machine vision market outlook and adoption roadmap",
            url="https://example.org/machine-vision-roadmap",
            snippet="Machine vision inspection forecast for 2026 to 2029.",
            rank=2,
        ),
    ]

    ordered = _prioritize_search_results(
        results,
        query="machine vision inspection market forecast 2026 2029 roadmap",
    )

    assert ordered[0].url == "https://example.org/machine-vision-roadmap"


def test_equally_relevant_academic_source_ranks_before_unknown_com_source() -> None:
    results = [
        SearchResult(
            title="Industrial defect detection transformer",
            url="https://example.com/transformer",
            snippet="Industrial defect detection transformer benchmark.",
            rank=1,
        ),
        SearchResult(
            title="Industrial defect detection transformer",
            url="https://arxiv.org/abs/2601.00001",
            snippet="Industrial defect detection transformer benchmark.",
            rank=2,
        ),
    ]

    ordered = _prioritize_search_results(results, query="industrial defect detection transformer")

    assert ordered[0].url.startswith("https://arxiv.org/")


def test_duplicate_search_urls_are_read_once() -> None:
    results = [
        SearchResult(title="First", url="https://example.com/result", rank=1),
        SearchResult(title="Duplicate", url="https://example.com/result", rank=2),
        SearchResult(title="Other", url="https://example.org/result", rank=3),
    ]

    unique = _deduplicate_search_results(results)

    assert [item.title for item in unique] == ["First", "Other"]


def test_http_https_and_tracking_variants_share_one_source_identity() -> None:
    first = "http://arxiv.org/abs/2012.03586v2?utm_source=test#section"
    second = "https://arxiv.org/abs/2012.03586v2"

    assert normalize_source_url(first) == normalize_source_url(second)
    unique = _deduplicate_search_results(
        [
            SearchResult(title="HTTP", url=first, rank=1),
            SearchResult(title="HTTPS", url=second, rank=2),
        ]
    )
    assert [item.title for item in unique] == ["HTTP"]


def test_industrial_control_anomaly_paper_fails_visual_topic_precheck() -> None:
    result = SearchResult(
        title="Anomaly Detection for Industrial Control Systems",
        url="https://arxiv.org/abs/2012.03586v2",
        snippet="Network intrusion and process anomaly detection in industrial control systems.",
        rank=1,
    )

    assert (
        _topic_relevance_ok("industrial machine vision surface defect detection", result)
        is False
    )


@pytest.mark.asyncio
async def test_duplicate_final_page_consumes_fetch_but_not_extraction_budget() -> None:
    repository = FakeRepository()
    repository.duplicate_page = True
    reader = FakeReader()
    extractor = FakeExtractor()
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        reader,
        extractor,
        FakeArtifacts(),
    )

    result = await service.run_one_iteration(uuid4(), worker_task_id="worker-1")

    assert reader.calls == 3
    assert repository.fetch_budget_settlements == 3
    assert extractor.calls == 0
    assert result.pages_read == 0
    assert repository.released_page_slots == 0
    assert repository.selection_events
    assert all(
        event == ("selection_skipped", "already_processed")
        for event in repository.selection_events
        if event[0] == "selection_skipped"
    )


def test_query_family_exhaustion_uses_actual_fresh_searches_only() -> None:
    usage: dict[str, object] = {}

    assert _executed_query_families(usage, "q1") == set()
    _mark_query_family_executed(usage, question_id="q1", family="scope")

    # Cache-only/gap iterations do not call the marker and therefore cannot
    # advance source-space exhaustion.
    assert _executed_query_families(usage, "q1") == {"scope"}
    assert next(iter(_query_family_order(prefer_authoritative=True))).value == "authoritative"

    for family in ("authoritative", "alternate", "contradiction"):
        _mark_query_family_executed(usage, question_id="q1", family=family)
    assert _executed_query_families(usage, "q1") == {
        "scope",
        "authoritative",
        "alternate",
        "contradiction",
    }


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"pages": 30}, "page_budget_exhausted"),
        ({"searches": 20}, "search_budget_exhausted"),
        ({"evidence_total_tokens": 100_000}, "token_budget_exhausted"),
        ({"iterations": 20}, "iteration_budget_exhausted"),
    ],
)
def test_budget_stop_reason_identifies_the_actual_limit(
    usage: dict[str, int], expected: str
) -> None:
    run = SimpleNamespace(
        budget_snapshot={
            "max_pages": 30,
            "max_searches": 20,
            "max_tokens": 100_000,
            "max_iterations": 20,
        },
        usage_snapshot=usage,
    )

    assert _budget_exhaustion_reason(run) == expected  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_retryable_search_failure_becomes_an_evaluable_provider_error() -> None:
    repository = FakeRepository()
    repository.allow_search_failure = True
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FailingSearch(),
        FakeReader(),
        FakeExtractor(),
        FakeArtifacts(),
    )

    result = await service.run_one_iteration(uuid4(), worker_task_id="worker-1")

    assert result.decision == "ready_to_write"
    assert result.continue_research is False
    assert repository.last_attempt_outcome == "provider_error"
    assert repository.finished is True


@pytest.mark.asyncio
async def test_research_reuses_unread_candidates_without_new_search_call() -> None:
    repository = FakeRepository()
    repository.target = replace(
        repository.target,
        reusable_results=(
            SearchResult(title="Cached source", url="https://example.com/cached", rank=1),
        ),
    )
    search = FakeSearch()
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        search,
        FakeReader(),
        FakeExtractor(),
        FakeArtifacts(),
    )

    await service.run_one_iteration(uuid4(), worker_task_id="worker-1")

    assert search.calls == 0


@pytest.mark.asyncio
async def test_quality_retry_forces_fresh_search_after_cached_pass() -> None:
    repository = FakeRepository()
    repository.target = replace(
        repository.target,
        gap_attempt_index=1,
        reusable_results=(
            SearchResult(title="Cached source", url="https://example.com/cached", rank=1),
        ),
    )
    search = FakeSearch()
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        search,
        FakeReader(),
        FakeExtractor(),
        FakeArtifacts(),
    )

    await service.run_one_iteration(uuid4(), worker_task_id="worker-1")

    assert search.calls == 1


@pytest.mark.asyncio
async def test_even_retry_reuses_cached_candidates_between_fresh_searches() -> None:
    repository = FakeRepository()
    repository.target = replace(
        repository.target,
        gap_attempt_index=2,
        reusable_results=(
            SearchResult(title="Cached source", url="https://example.com/cached", rank=1),
        ),
    )
    search = FakeSearch()
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        search,
        FakeReader(),
        FakeExtractor(),
        FakeArtifacts(),
    )

    await service.run_one_iteration(uuid4(), worker_task_id="worker-1")

    assert search.calls == 0


@pytest.mark.parametrize(
    "url",
    [
        "https://www.bilibili.com/video/BV1x",
        "https://accounts.google.com/signin",
        "https://www.renrendoc.com/paper/123.html",
        "https://blog.csdn.net/example/article/details/1",
        "https://max.book118.com/html/2026/report.shtml",
        "https://www.google.com/search?q=inspection",
        "https://example.com/login",
    ],
)
def test_non_attributable_or_login_pages_are_filtered_before_read(url: str) -> None:
    assert _is_read_candidate(url) is False


def test_public_article_remains_readable_candidate() -> None:
    assert _is_read_candidate("https://www.nist.gov/publications/inspection-report") is True


@pytest.mark.asyncio
async def test_research_reuses_snapshot_artifact_without_http_read() -> None:
    repository = FakeRepository()
    reader = FakeReader()
    repository.target = replace(
        repository.target,
        reusable_pages=(
            ReusablePageRef(
                source_id=uuid4(),
                final_url="https://example.com/snapshot",
                title="Industrial inspection technology routes snapshot",
                artifact_uri="runs/test/snapshot.txt",
                content_hash=hashlib.sha256(
                    ("Previously fetched evidence-bearing page content. " * 5).encode("utf-8")
                ).hexdigest(),
                fetched_at=datetime.now(UTC),
            ),
        ),
    )
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        reader,
        FakeExtractor(),
        FakeArtifacts(),
    )

    await service.run_one_iteration(uuid4(), worker_task_id="worker-1")

    assert reader.calls == 0
    assert repository.last_attempt_outcome == "evidence_gained"


@pytest.mark.asyncio
async def test_model_budget_guard_stops_before_page_read_or_extraction() -> None:
    repository = FakeRepository()
    repository.model_budget_allowed = False
    extractor = FakeExtractor()
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        FakeReader(),
        extractor,
        FakeArtifacts(),
    )

    result = await service.run_one_iteration(uuid4(), worker_task_id="worker-1")

    assert result.pages_read == 0
    assert extractor.calls == 0
    assert repository.finished is True


@pytest.mark.asyncio
async def test_parallel_reads_isolates_prefetch_failures_and_skips_extraction() -> None:
    repository = FakeRepository()
    extractor = FakeExtractor()
    reader = FirstThreePagesFailReader()
    service = ResearchLoopService(  # type: ignore[arg-type]
        repository,
        FakeSearch(),
        reader,
        extractor,
        FakeArtifacts(),
        parallel_reads_enabled=True,
    )

    result = await service.run_one_iteration(uuid4(), worker_task_id="worker-1")

    # Every prefetched page failed, so no model extraction or reservation happened.
    assert repository.page_failures >= 1
    assert repository.fetch_budget_settlements == reader.calls
    assert extractor.calls == 0
    assert result.pages_read == 0
