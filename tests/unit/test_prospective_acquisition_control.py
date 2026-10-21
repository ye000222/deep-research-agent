from __future__ import annotations

import unittest

from scripts.prospective_acquisition.control import (
    AcquisitionBlocked,
    CallBudget,
    ClaimRecord,
    ProspectiveAcquisitionHarness,
    TaskResult,
)


def claim(
    claim_id: str,
    *,
    source: str = "source-a",
    owner: str | None = "owner-a",
    complete: bool = True,
) -> ClaimRecord:
    return ClaimRecord(
        claim_id=claim_id,
        created_at="2026-10-01T00:00:00Z",
        source_refs=(f"source-ref-{claim_id}",) if complete else (),
        source_key=source if complete else None,
        owner_key=owner,
        content_hash=f"hash-{claim_id}" if complete else "",
        provenance_complete=complete,
        identity_isolation_attested=True,
    )


def harness(*, source_cap: int = 5, owner_cap: int = 10) -> ProspectiveAcquisitionHarness:
    return ProspectiveAcquisitionHarness(
        population_id="population-test",
        batch_id="batch-test",
        protocol_version="prospective-development-population.v1",
        source_cap=source_cap,
        owner_cap=owner_cap,
    )


class CallBudgetTests(unittest.TestCase):
    def test_zero_remaining_blocks_before_outbound(self) -> None:
        invoked: list[bool] = []
        budget = CallBudget(maximum=1, dispatched=1)
        with self.assertRaises(AcquisitionBlocked):
            budget.dispatch(lambda: invoked.append(True))
        self.assertEqual(invoked, [])

    def test_transport_failure_consumes_reserved_call_and_is_not_retried(self) -> None:
        calls = 0
        budget = CallBudget(maximum=1)

        def fail() -> None:
            nonlocal calls
            calls += 1
            raise TimeoutError("fake timeout")

        with self.assertRaises(TimeoutError):
            budget.dispatch(fail)
        self.assertEqual(calls, 1)
        self.assertEqual(budget.remaining, 0)


class AcquisitionHarnessTests(unittest.TestCase):
    def execute(self, result: TaskResult, **kwargs: object):
        written: list[dict[str, object]] = []
        h = harness(**kwargs)
        output = h.execute(
            ("task-1",),
            lambda task_id: result,
            attestation_writer=written.append,
            artifact_schema_valid=lambda _: True,
            admitted_at="2026-10-02T00:00:00Z",
        )
        return output, written

    def test_successful_run_admits_multiple_claims(self) -> None:
        candidates = (claim("c1"), claim("c2", source="source-b"))
        result, written = self.execute(TaskResult("task-1", "run-1", candidates))
        self.assertEqual(sum(item.admitted for item in result.admissions), 2)
        self.assertEqual(len(written), 2)
        self.assertTrue(all(x["initial_exposure_state"] == "UNINSPECTED" for x in written))

    def test_zero_claim_run_is_valid_empty_result(self) -> None:
        result, written = self.execute(TaskResult("task-1", "run-1", ()))
        self.assertEqual(result.admissions, ())
        self.assertEqual(written, [])

    def test_incomplete_provenance_fails_closed(self) -> None:
        incomplete = (claim("c1", complete=False),)
        result, written = self.execute(TaskResult("task-1", "run-1", incomplete))
        self.assertEqual(result.admissions[0].reason, "incomplete_source_provenance")
        self.assertEqual(written, [])

    def test_unknown_owner_does_not_block_source_complete_claim(self) -> None:
        result, written = self.execute(TaskResult("task-1", "run-1", (claim("c1", owner=None),)))
        self.assertTrue(result.admissions[0].admitted)
        self.assertIsNone(written[0]["owner_key"])

    def test_source_cap_overflow_is_provenance_only(self) -> None:
        claims = tuple(claim(f"c{i}") for i in range(1, 4))
        result, _ = self.execute(TaskResult("task-1", "run-1", claims), source_cap=2)
        self.assertEqual(sum(x.admitted for x in result.admissions), 2)
        self.assertEqual(sum(x.reason == "source_cap_exceeded" for x in result.admissions), 1)

    def test_owner_cap_overflow_is_provenance_only(self) -> None:
        claims = tuple(claim(f"c{i}", source=f"s{i}") for i in range(1, 4))
        result, _ = self.execute(TaskResult("task-1", "run-1", claims), owner_cap=2)
        self.assertEqual(sum(x.admitted for x in result.admissions), 2)
        self.assertEqual(sum(x.reason == "owner_cap_exceeded" for x in result.admissions), 1)

    def test_duplicate_identity_excluded(self) -> None:
        duplicates = (claim("same"), claim("same", source="source-b"))
        result, _ = self.execute(TaskResult("task-1", "run-1", duplicates))
        self.assertEqual(sum(x.admitted for x in result.admissions), 1)
        self.assertIn("duplicate_claim_identity", {x.reason for x in result.admissions})

    def test_attestation_write_failure_excludes_claim(self) -> None:
        result = harness().execute(
            ("task-1",),
            lambda task_id: TaskResult(task_id, "run-1", (claim("c1"),)),
            attestation_writer=lambda _: (_ for _ in ()).throw(OSError("fake")),
            artifact_schema_valid=lambda _: True,
            admitted_at="2026-10-02T00:00:00Z",
        )
        self.assertEqual(result.admissions[0].reason, "attestation_write_failed")

    def test_schema_invalid_attestation_excludes_claim(self) -> None:
        result = harness().execute(
            ("task-1",),
            lambda task_id: TaskResult(task_id, "run-1", (claim("c1"),)),
            attestation_writer=lambda _: self.fail("invalid artifact must not be written"),
            artifact_schema_valid=lambda _: False,
            admitted_at="2026-10-02T00:00:00Z",
        )
        self.assertEqual(result.admissions[0].reason, "schema_invalid_attestation")

    def test_task_failure_is_not_retried_or_replaced(self) -> None:
        calls: list[str] = []

        def run_once(task_id: str) -> TaskResult:
            calls.append(task_id)
            raise ConnectionError("fake transport failure")

        result = harness().execute(
            ("task-1",), run_once,
            attestation_writer=lambda _: None,
            artifact_schema_valid=lambda _: True,
            admitted_at="2026-10-02T00:00:00Z",
        )
        self.assertEqual(calls, ["task-1"])
        self.assertEqual(result.task_failures, (("task-1", "ConnectionError"),))

    def test_harness_provider_dispatch_blocks_before_fake_transport_at_zero_budget(self) -> None:
        h = ProspectiveAcquisitionHarness(
            population_id="population-test",
            batch_id="batch-test",
            protocol_version="prospective-development-population.v1",
            provider_call_budget=CallBudget(maximum=0),
        )
        outbound: list[bool] = []
        with self.assertRaises(AcquisitionBlocked):
            h.provider_dispatch(lambda: outbound.append(True))
        self.assertEqual(outbound, [])


if __name__ == "__main__":
    unittest.main()
