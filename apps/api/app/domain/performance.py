"""Performance projection from existing run and per-call audit records."""

from datetime import UTC, datetime

from app.domain.research_runs import ResearchRunView


def summarize_performance(
    run: ResearchRunView,
    calls: list[dict[str, object]],
) -> dict[str, object]:
    def number(value: object) -> int:
        return max(0, value) if isinstance(value, int) and not isinstance(value, bool) else 0

    known_tokens = 0
    estimated_calls = 0
    unavailable_calls = 0
    retry_tokens = 0
    model_latency = 0
    for call in calls:
        raw = call.get("usage")
        usage = raw if isinstance(raw, dict) else {}
        accuracy = usage.get("accuracy", "unavailable")
        unavailable_calls += int(accuracy not in {"exact", "estimated"})
        estimated_calls += int(accuracy == "estimated")
        total = number(usage.get("total_tokens"))
        known_tokens += total
        if call.get("retry_mode") not in {None, "", "none"}:
            retry_tokens += total
        model_latency += number(call.get("latency_ms"))
    ended = run.finished_at or datetime.now(UTC)
    latency_samples = sorted(
        number(call.get("latency_ms")) for call in calls if number(call.get("latency_ms")) > 0
    )

    def percentile(values: list[int], fraction: float) -> int | None:
        if not values:
            return None
        index = min(len(values) - 1, max(0, int((len(values) - 1) * fraction)))
        return values[index]

    first_evidence_ms: int | None = None
    elapsed_to_evidence = 0
    for call in calls:
        elapsed_to_evidence += number(call.get("latency_ms"))
        if number(call.get("accepted_evidence", call.get("accepted", 0))) > 0:
            first_evidence_ms = elapsed_to_evidence
            break
    usage = run.usage_snapshot
    pages_fetched = number(usage.get("pages_fetched", usage.get("pages", 0)))
    pages_extracted = number(usage.get("pages_extracted", usage.get("pages", 0)))
    logical_queries = number(usage.get("logical_queries", usage.get("searches", 0)))
    provider_requests = number(usage.get("search_provider_requests", 0))
    healthy_provider_responses = number(usage.get("search_provider_healthy_responses", 0))
    productive_provider_responses = number(
        usage.get("search_provider_productive_responses", 0)
    )
    unresponsive_provider_responses = number(
        usage.get("search_provider_unresponsive_responses", 0)
    )
    zero_yield = number(usage.get("zero_yield_pages", 0))
    accepted_evidence = number(usage.get("accepted_evidence", 0))
    quality = run.quality_snapshot
    elapsed_ms = max(0, int((ended - run.created_at).total_seconds() * 1000))
    elapsed_samples = usage.get("completion_elapsed_samples_ms", [])
    if not isinstance(elapsed_samples, list) or not elapsed_samples:
        elapsed_samples = [elapsed_ms]
    elapsed_samples = sorted(number(value) for value in elapsed_samples if number(value) > 0)
    first_samples = usage.get("first_evidence_latency_samples_ms", [])
    if not isinstance(first_samples, list) or not first_samples:
        first_samples = [first_evidence_ms] if first_evidence_ms is not None else []
    first_samples = sorted(number(value) for value in first_samples if number(value) > 0)
    technical_failures = number(usage.get("technical_retries")) + number(
        usage.get("evidence_extraction_failures")
    )
    rate_limited = number(usage.get("model_rate_limited")) + number(
        usage.get("search_rate_limited")
    )
    return {
        "run_id": str(run.run_id),
        "status": run.status.value,
        "elapsed_ms": elapsed_ms,
        "completion_time_p50_ms": percentile(elapsed_samples, 0.50),
        "completion_time_p95_ms": percentile(elapsed_samples, 0.95),
        "model_call_latency_sum_ms": model_latency,
        "active_ms": None,
        "dependency_wait_ms": None,
        "audited_total_tokens": known_tokens,
        "budget_accounted_tokens": run.usage_snapshot.get("model_tokens"),
        "estimated_calls": estimated_calls,
        "unavailable_usage_calls": unavailable_calls,
        "retry_tokens": retry_tokens,
        "model_calls": len(calls),
        "latency_p50_ms": percentile(latency_samples, 0.50),
        "latency_p95_ms": percentile(latency_samples, 0.95),
        "first_evidence_ms": first_evidence_ms,
        "first_evidence_p50_ms": percentile(first_samples, 0.50),
        "first_evidence_p95_ms": percentile(first_samples, 0.95),
        "pages_fetched": pages_fetched,
        "pages_extracted": pages_extracted,
        "logical_queries": logical_queries,
        "provider_requests": provider_requests,
        "healthy_provider_responses": healthy_provider_responses,
        "productive_provider_responses": productive_provider_responses,
        "unresponsive_provider_responses": unresponsive_provider_responses,
        "productive_provider_response_rate": (
            round(productive_provider_responses / provider_requests, 4)
            if provider_requests
            else None
        ),
        "pages_per_logical_query": round(pages_fetched / logical_queries, 4)
        if logical_queries
        else None,
        "pages_per_provider_request": round(pages_fetched / provider_requests, 4)
        if provider_requests
        else None,
        "zero_yield_pages": zero_yield,
        "zero_yield_rate": round(zero_yield / pages_fetched, 4) if pages_fetched else None,
        "accepted_evidence": accepted_evidence,
        "tokens_per_accepted_evidence": (
            round(known_tokens / accepted_evidence, 4) if accepted_evidence else None
        ),
        "extracted_pages_per_accepted_evidence": (
            round(pages_extracted / accepted_evidence, 4) if accepted_evidence else None
        ),
        "coverage": quality.get("coverage"),
        "priority_one_coverage": quality.get("priority_one_coverage"),
        "citation_accuracy": quality.get("citation_support"),
        "dual_source_rate": quality.get("high_risk_claim_dual_source_rate"),
        "conflict_count": quality.get("conflict_count"),
        "unresolved_conflict_count": quality.get("conflict_count"),
        "technical_failure_count": technical_failures,
        "rate_limited_count": rate_limited,
        "completed_with_limitations": run.status.value == "completed_with_limitations",
        "limitations_ratio": 1.0 if run.status.value == "completed_with_limitations" else 0.0,
        "limitations_count": len(quality.get("limitations", []))
        if isinstance(quality.get("limitations"), list)
        else None,
        "stop_reason": run.termination_reason,
        "policy_version": run.budget_snapshot.get("performance_policy_version", "legacy"),
        "audited_usage_complete": bool(calls) and unavailable_calls == 0,
        "measurement_complete": False,
    }
