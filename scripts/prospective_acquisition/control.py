"""Fail-closed acquisition admission and pre-dispatch budget primitives.

This module is deliberately isolated from the production ResearchLoop. It
provides deterministic control-plane behavior for offline/fake harness tests;
it does not create runs, call a Provider, or persist database records.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TypeVar

import httpx

T = TypeVar("T")


class AcquisitionBlocked(RuntimeError):
    """Raised before an outbound call when the frozen budget is exhausted."""


@dataclass
class CallBudget:
    maximum: int
    dispatched: int = 0
    parent: CallBudget | None = None
    blocked_attempts: int = 0

    def __post_init__(self) -> None:
        if self.maximum < 0 or self.dispatched < 0 or self.dispatched > self.maximum:
            raise ValueError("invalid call budget")

    @property
    def remaining(self) -> int:
        local_remaining = self.maximum - self.dispatched
        if self.parent is None:
            return local_remaining
        return min(local_remaining, self.parent.remaining)

    def reserve(self) -> None:
        """Atomically reserve this scope and every enclosing budget."""
        if self.maximum - self.dispatched <= 0:
            self.blocked_attempts += 1
            raise AcquisitionBlocked("provider_call_budget_exhausted")
        if self.parent is not None:
            try:
                self.parent.reserve()
            except AcquisitionBlocked:
                self.blocked_attempts += 1
                raise
        self.dispatched += 1

    def dispatch(self, outbound: Callable[..., T], *args: object, **kwargs: object) -> T:
        """Reserve before transport invocation; failures still consume a call."""
        self.reserve()
        return outbound(*args, **kwargs)


class BudgetedAsyncTransport(httpx.AsyncBaseTransport):
    """Count each HTTP request before delegating to the underlying transport."""

    def __init__(self, budget: CallBudget, transport: httpx.AsyncBaseTransport) -> None:
        self._budget = budget
        self._transport = transport

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self._budget.reserve()
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self._transport.aclose()


def build_guarded_provider_client(
    budget: CallBudget,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Create a no-implicit-retry model client with a pre-transport guard."""
    return build_guarded_http_client(budget, transport=transport)


def build_guarded_http_client(
    budget: CallBudget,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Count and reject the N+1 HTTP request before an underlying transport."""
    inner = transport or httpx.AsyncHTTPTransport(retries=0)
    return httpx.AsyncClient(
        transport=BudgetedAsyncTransport(budget, inner),
        follow_redirects=False,
        trust_env=False,
    )


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    run_id: str
    claims: tuple[ClaimRecord, ...]


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    created_at: str
    source_refs: tuple[str, ...]
    source_key: str | None
    owner_key: str | None
    content_hash: str
    provenance_complete: bool
    identity_isolation_attested: bool
    evidence_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    snapshot_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdmissionDecision:
    claim_id: str
    admitted: bool
    reason: str
    attestation_hash: str | None = None


@dataclass(frozen=True)
class HarnessResult:
    task_results: tuple[TaskResult, ...]
    admissions: tuple[AdmissionDecision, ...]
    task_failures: tuple[tuple[str, str], ...]


def _stable_key(batch_id: str, claim_id: str) -> str:
    return hashlib.sha256(f"{batch_id}\0{claim_id}".encode()).hexdigest()


def _attestation(
    claim: ClaimRecord,
    *,
    population_id: str,
    batch_id: str,
    run_id: str,
    protocol_version: str,
    admitted_at: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "claim_id": claim.claim_id,
        "population_id": population_id,
        "batch_id": batch_id,
        "run_id": run_id,
        "created_at": claim.created_at,
        "development_only": True,
        "protocol_version": protocol_version,
        "source_provenance_refs": list(claim.source_refs),
        "source_diversity_key": claim.source_key,
        "owner_key": claim.owner_key,
        "content_hash": claim.content_hash,
        "identity_isolation_attested": claim.identity_isolation_attested,
        "initial_exposure_state": "UNINSPECTED",
        "admission_timestamp": admitted_at,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["attestation_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


class ProspectiveAcquisitionHarness:
    """Run each frozen task once, then admit claims using provenance only."""

    def __init__(
        self,
        *,
        population_id: str,
        batch_id: str,
        protocol_version: str,
        provider_call_budget: CallBudget | None = None,
        source_cap: int = 5,
        owner_cap: int = 10,
    ) -> None:
        if not population_id or not batch_id or not protocol_version:
            raise ValueError("population, batch, and protocol identities are required")
        if source_cap <= 0 or owner_cap <= 0:
            raise ValueError("caps must be positive")
        self.population_id = population_id
        self.batch_id = batch_id
        self.protocol_version = protocol_version
        self.provider_call_budget = provider_call_budget or CallBudget(maximum=0)
        self.source_cap = source_cap
        self.owner_cap = owner_cap

    def provider_dispatch(self, outbound: Callable[..., T], *args: object, **kwargs: object) -> T:
        """The fake/runtime adapter must route every generation request here."""
        return self.provider_call_budget.dispatch(outbound, *args, **kwargs)

    def execute(
        self,
        task_ids: tuple[str, ...],
        run_once: Callable[[str], TaskResult],
        *,
        attestation_writer: Callable[[dict[str, object]], None],
        artifact_schema_valid: Callable[[dict[str, object]], bool],
        admitted_at: str,
    ) -> HarnessResult:
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError("frozen task IDs must be non-empty and unique")

        completed: list[TaskResult] = []
        failures: list[tuple[str, str]] = []
        for task_id in task_ids:
            try:
                result = run_once(task_id)
                if result.task_id != task_id:
                    failures.append((task_id, "task_identity_mismatch"))
                    continue
                completed.append(result)
            except Exception as exc:  # one task failure never triggers replacement/retry
                failures.append((task_id, type(exc).__name__))

        candidates: list[tuple[ClaimRecord, str]] = []
        for result in completed:
            candidates.extend((claim, result.run_id) for claim in result.claims)
        candidates.sort(key=lambda item: _stable_key(self.batch_id, item[0].claim_id))

        seen: set[str] = set()
        source_counts: dict[str, int] = {}
        owner_counts: dict[str, int] = {}
        decisions: list[AdmissionDecision] = []
        for claim, run_id in candidates:
            if claim.claim_id in seen:
                decisions.append(
                    AdmissionDecision(claim.claim_id, False, "duplicate_claim_identity")
                )
                continue
            seen.add(claim.claim_id)
            if not claim.provenance_complete or not claim.source_refs or not claim.source_key:
                decisions.append(
                    AdmissionDecision(claim.claim_id, False, "incomplete_source_provenance")
                )
                continue
            if not claim.content_hash:
                decisions.append(AdmissionDecision(claim.claim_id, False, "missing_content_hash"))
                continue
            if not claim.created_at or not claim.identity_isolation_attested:
                decisions.append(
                    AdmissionDecision(claim.claim_id, False, "identity_attestation_incomplete")
                )
                continue
            if source_counts.get(claim.source_key, 0) >= self.source_cap:
                decisions.append(AdmissionDecision(claim.claim_id, False, "source_cap_exceeded"))
                continue
            if claim.owner_key and owner_counts.get(claim.owner_key, 0) >= self.owner_cap:
                decisions.append(AdmissionDecision(claim.claim_id, False, "owner_cap_exceeded"))
                continue

            attestation = _attestation(
                claim,
                population_id=self.population_id,
                batch_id=self.batch_id,
                run_id=run_id,
                protocol_version=self.protocol_version,
                admitted_at=admitted_at,
            )
            if not artifact_schema_valid(attestation):
                decisions.append(
                    AdmissionDecision(claim.claim_id, False, "schema_invalid_attestation")
                )
                continue
            try:
                attestation_writer(attestation)
            except Exception:
                decisions.append(
                    AdmissionDecision(claim.claim_id, False, "attestation_write_failed")
                )
                continue

            source_counts[claim.source_key] = source_counts.get(claim.source_key, 0) + 1
            if claim.owner_key:
                owner_counts[claim.owner_key] = owner_counts.get(claim.owner_key, 0) + 1
            decisions.append(
                AdmissionDecision(
                    claim.claim_id,
                    True,
                    "admitted",
                    str(attestation["attestation_hash"]),
                )
            )

        return HarnessResult(tuple(completed), tuple(decisions), tuple(failures))


def execute_frozen_tasks(
    task_ids: Iterable[str],
    run_once: Callable[[str], TaskResult],
) -> tuple[TaskResult, ...]:
    """Minimal deterministic task runner; each supplied task is invoked once."""
    results: list[TaskResult] = []
    for task_id in tuple(task_ids):
        results.append(run_once(task_id))
    return tuple(results)
