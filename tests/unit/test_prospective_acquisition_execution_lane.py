from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from collections.abc import Coroutine
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID, uuid4, uuid5

from app.domain.research_runs import RunKind
from app.domain.research_tools import (
    EvidenceCandidate,
    EvidenceRelation,
    ReadPage,
    ScoredEvidence,
)
from app.infrastructure.db.outbox import TaskDispatchOutboxRepository
from app.infrastructure.db.research_runs import (
    ResearchRunNotFoundError,
    ResearchRunRepository,
)
from app.infrastructure.db.research_tools import (
    ClaimPersistenceReceipt,
    EvidencePersistenceReceipt,
    PagePersistenceResult,
    ResearchTarget,
)
from app.infrastructure.db.run_models import ResearchRunRow, TaskDispatchOutboxRow
from app.security.client_sessions import ClientSession
from app.services.prospective_acquisition_execution_identity import (
    FrozenProviderBinding,
    create_execution_context,
)
from app.services.provider_fingerprint_contract import safe_provider_fingerprint

from scripts.prospective_acquisition.control import (
    CallBudget,
    ClaimRecord,
    ProspectiveAcquisitionHarness,
    TaskResult,
)
from scripts.prospective_acquisition.execution_lane import (
    _RUN_NAMESPACE,
    AcquisitionLaneBlocked,
    FrozenAcquisitionTask,
    ProspectiveAcquisitionExecutionLane,
    VerifiedTaskBatch,
    task_execution_identity,
)
from scripts.prospective_acquisition.execution_ledger import (
    ExecutionLedgerError,
    TaskExecutionLedger,
)
from scripts.prospective_acquisition.production import (
    ExtractedEvidenceBatch,
    ProductionClaimFormationComponent,
)
from scripts.prospective_acquisition.profile import (
    BoundedProspectiveAcquisitionProfile,
    BoundedTaskExecution,
)
from scripts.prospective_acquisition.verify_frozen_manifest import (
    ATTEMPT_ID,
    AUTHORIZATION_ID,
    BATCH_ID,
    MANIFEST_SHA256,
    POPULATION_ID,
)

OWNER = "a" * 64
PROFILE_ID = uuid4()
CREDENTIAL_ID = uuid4()


def _run_immediate(coroutine: Coroutine[object, object, object]) -> object:
    """Drive async repository code whose fake awaits all complete immediately."""
    try:
        yielded = coroutine.send(None)
    except StopIteration as result:
        return result.value
    coroutine.close()
    raise AssertionError(f"fake repository unexpectedly yielded: {yielded!r}")


class _FakeResult:
    def __init__(self, profile=None, credential=None, rows=None) -> None:
        self.profile = profile
        self.credential = credential
        self.rows = list(rows or [])

    def tuples(self):
        return self

    def first(self):
        return (self.profile, self.credential) if self.profile is not None else None

    def all(self):
        return self.rows


class _FakeSession:
    def __init__(self, profile=None, credential=None) -> None:
        self.profile = profile
        self.credential = credential
        self.rows: list[object] = []
        self.scalar_value = None
        self.statements: list[object] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        return None

    def begin(self):
        return self

    async def scalar(self, _statement):
        self.statements.append(_statement)
        return self.scalar_value

    async def execute(self, _statement):
        return _FakeResult(self.profile, self.credential)

    async def scalars(self, statement):
        self.statements.append(statement)
        rows = [row for row in self.rows if isinstance(row, TaskDispatchOutboxRow)]
        return _FakeResult(rows=rows)

    def add(self, row) -> None:
        self.rows.append(row)

    async def flush(self) -> None:
        now = datetime.now(UTC)
        for row in self.rows:
            if isinstance(row, ResearchRunRow):
                row.state_version = row.state_version or 1
                row.plan_version = row.plan_version or 0
                row.next_event_seq = row.next_event_seq or 1
                row.quality_snapshot = row.quality_snapshot or {}
                row.usage_snapshot = row.usage_snapshot or {}
                row.created_at = row.created_at or now
                row.updated_at = row.updated_at or now
            elif isinstance(row, TaskDispatchOutboxRow):
                row.attempt_count = row.attempt_count or 0
                row.next_attempt_at = row.next_attempt_at or now
                row.created_at = row.created_at or now


class _FakeSessions:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session

    def __call__(self):
        return self.session


def _context_and_profile():
    profile = SimpleNamespace(
        id=PROFILE_ID,
        owner_hash=OWNER,
        status="active",
        deleted_at=None,
        adapter_type="openai_compatible_chat",
        normalized_base_url="https://provider.example/v1",
        endpoint_host="provider.example",
        auth_type="bearer",
        non_secret_settings={"supports_json_mode": True},
        model="frozen-model",
        version=4,
    )
    credential = SimpleNamespace(
        id=CREDENTIAL_ID,
        profile_id=PROFILE_ID,
        credential_version=3,
        ciphertext=b"fixture-only",
        revoked_at=None,
        deleted_at=None,
    )
    _, fingerprint = safe_provider_fingerprint(
        {
            "profile_id": str(PROFILE_ID),
            "adapter": profile.adapter_type,
            "model": profile.model,
            "base_url": profile.normalized_base_url,
            "auth_type": profile.auth_type,
            "profile_version": profile.version,
            "credential": {
                "id": str(CREDENTIAL_ID),
                "version": credential.credential_version,
                "present": True,
            },
            "non_secret_settings": profile.non_secret_settings,
        }
    )
    context = create_execution_context(
        ClientSession(uuid4(), OWNER, False),
        population_id=POPULATION_ID,
        batch_id=BATCH_ID,
        protocol_version="prospective-development-population.v1",
    )
    context = replace(
        context, execution_context_id="8862f286-e707-461e-876f-ec57207802e8"
    )
    context = context.with_provider_binding(
        FrozenProviderBinding(
            profile_id=str(PROFILE_ID),
            adapter=profile.adapter_type,
            model=profile.model,
            endpoint_family="https://provider.example",
            profile_version=profile.version,
            credential_version_id=str(CREDENTIAL_ID),
            credential_version=credential.credential_version,
            credential_present=True,
            fingerprint_contract="provider-safe-config-fingerprint.v1",
            provider_safe_config_fingerprint=fingerprint,
        )
    )
    return context, profile, credential


def _frozen_tasks() -> tuple[FrozenAcquisitionTask, ...]:
    return tuple(
        FrozenAcquisitionTask(
            task_id=f"task-{index:03d}",
            task_text=f"opaque frozen text {index}",
            ordinal=index,
            task_execution_id=task_execution_identity(
                task_id=f"task-{index:03d}", ordinal=index
            ),
            run_id=uuid5(
                _RUN_NAMESPACE,
                task_execution_identity(task_id=f"task-{index:03d}", ordinal=index),
            ),
        )
        for index in range(1, 41)
    )


class ExecutionIdentityTests(unittest.TestCase):
    def test_40_task_execution_identities_are_stable_and_unique(self) -> None:
        first = _frozen_tasks()
        second = _frozen_tasks()
        self.assertEqual(
            [item.task_execution_id for item in first],
            [item.task_execution_id for item in second],
        )
        self.assertEqual(len({item.task_execution_id for item in first}), 40)
        self.assertEqual(len({item.run_id for item in first}), 40)
        self.assertNotEqual(
            first[0].task_execution_id,
            task_execution_identity(task_id="task-001", ordinal=2),
        )

    def test_identity_rejects_attempt_or_manifest_drift(self) -> None:
        normal = task_execution_identity(task_id="task-001", ordinal=1)
        drifted = task_execution_identity(
            task_id="task-001", ordinal=1, manifest_sha256="0" * 64
        )
        self.assertEqual(len(normal), 64)
        self.assertNotEqual(normal, drifted)
        self.assertEqual(
            ATTEMPT_ID,
            "4f05b3509cbb75d2481dcf0fa46b0683e9278d5c66021b03c59f6ebc9eb69ef0",
        )
        self.assertEqual(
            AUTHORIZATION_ID, "213e94e8-5f75-4d05-87b9-eb41afe40ba8"
        )

    def test_one_task_failure_never_substitutes_or_retries_a_frozen_task(self) -> None:
        calls: list[str] = []
        harness = ProspectiveAcquisitionHarness(
            population_id=POPULATION_ID,
            batch_id=BATCH_ID,
            protocol_version="prospective-development-population.v1",
            provider_call_budget=CallBudget(maximum=0),
        )

        def run_once(task_id: str) -> TaskResult:
            calls.append(task_id)
            if task_id == "task-001":
                raise RuntimeError("fixture failure")
            return TaskResult(task_id, f"run-{task_id}", ())

        result = harness.execute(
            ("task-001", "task-002"),
            run_once,
            attestation_writer=lambda _document: None,
            artifact_schema_valid=lambda _document: True,
            admitted_at="2026-10-13T00:00:00Z",
        )
        self.assertEqual(calls, ["task-001", "task-002"])
        self.assertEqual(result.task_failures, (("task-001", "RuntimeError"),))
        self.assertEqual(len(result.task_results), 1)


class ExecutionLedgerTests(unittest.TestCase):
    def test_success_skips_and_terminal_receipt_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            ledger = TaskExecutionLedger(Path(temp))
            task_id = "a" * 64
            self.assertEqual(ledger.state(task_id), "NOT_STARTED")
            self.assertEqual(
                ledger.prepare(task_id, task_id="task-001", task_ordinal=1), "PREPARED"
            )
            ledger.mark_external_effect_started(task_id)
            receipt = ledger.finish(
                task_id,
                terminal_state="SUCCEEDED",
                receipt={"run_id": "run-safe", "budget_counters": {"provider": 1}},
            )
            self.assertEqual(ledger.state(task_id), "SUCCEEDED")
            self.assertTrue(receipt.is_file())
            with self.assertRaises(ExecutionLedgerError):
                ledger.finish(task_id, terminal_state="SUCCEEDED", receipt={})

    def test_pre_effect_failure_resumes_but_post_effect_states_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            ledger = TaskExecutionLedger(Path(temp))
            resumable = "b" * 64
            ledger.prepare(resumable, task_id="task-002", task_ordinal=2)
            ledger.fail_pre_effect(
                resumable,
                safe_failure_code="SAFE_PREFLIGHT_FAILURE",
                counters={
                    "search": 0,
                    "reader": 0,
                    "provider": 0,
                    "http": 0,
                    "production_db_writes": 0,
                },
            )
            self.assertEqual(
                ledger.prepare(resumable, task_id="task-002", task_ordinal=2), "PREPARED"
            )

            ambiguous = "c" * 64
            ledger.prepare(ambiguous, task_id="task-003", task_ordinal=3)
            ledger.mark_external_effect_started(ambiguous)
            self.assertEqual(ledger.state(ambiguous), "EXTERNAL_EFFECT_STARTED")
            with self.assertRaises(ExecutionLedgerError):
                ledger.prepare(ambiguous, task_id="task-003", task_ordinal=3)

            failed = "d" * 64
            ledger.prepare(failed, task_id="task-004", task_ordinal=4)
            ledger.mark_external_effect_started(failed)
            ledger.finish(
                failed,
                terminal_state="FAILED_AFTER_EFFECT",
                receipt={"safe_failure_code": "SAFE_FAILURE"},
            )
            with self.assertRaises(ExecutionLedgerError):
                ledger.prepare(failed, task_id="task-004", task_ordinal=4)

    def test_hash_chain_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            root = Path(temp)
            ledger = TaskExecutionLedger(root)
            ledger.prepare("e" * 64, task_id="task-005", task_ordinal=5)
            path = root / "execution_ledger.jsonl"
            event = json.loads(path.read_text(encoding="utf-8"))
            event["details"]["task_id"] = "tampered"
            path.write_text(json.dumps(event) + "\n", encoding="utf-8")
            with self.assertRaises(ExecutionLedgerError):
                ledger.state("e" * 64)


class RunKindIsolationTests(unittest.TestCase):
    def test_acquisition_run_has_no_outbox_and_worker_cannot_claim_it(self) -> None:
        context, profile, credential = _context_and_profile()
        session = _FakeSession(profile, credential)
        repo = ResearchRunRepository(_FakeSessions(session))
        task = _frozen_tasks()[0]
        view, created = _run_immediate(repo.create_for_prospective_acquisition(
            context,
            run_id=task.run_id,
            worker_task_id="worker-acquisition-test",
            idempotency_key=f"prospective-acquisition:{task.task_execution_id}",
            task_text=task.task_text,
            task_ordinal=task.ordinal,
            task_execution_id=task.task_execution_id,
            budget_snapshot={"max_provider_requests": 1, "max_logical_queries": 1},
        ))
        self.assertTrue(created)
        self.assertEqual(view.status.value, "running")
        run = next(row for row in session.rows if isinstance(row, ResearchRunRow))
        self.assertEqual(run.run_kind, RunKind.PROSPECTIVE_ACQUISITION_TASK.value)
        self.assertEqual(sum(isinstance(row, TaskDispatchOutboxRow) for row in session.rows), 0)
        claimed = _run_immediate(TaskDispatchOutboxRepository(_FakeSessions(session)).claim_batch())
        self.assertEqual(claimed, [])
        session.scalar_value = run
        self.assertFalse(
            _run_immediate(repo.acquire_for_execution(task.run_id, worker_task_id="celery-task"))
        )

    def test_normal_run_creation_retains_outbox_and_research_kind(self) -> None:
        _context, profile, credential = _context_and_profile()
        session = _FakeSession(profile, credential)
        repo = ResearchRunRepository(_FakeSessions(session))
        view, created = _run_immediate(repo.create(
            OWNER,
            idempotency_key="normal-run-test",
            original_query="ordinary question",
            normalized_goal="ordinary question",
            credential_version_id=CREDENTIAL_ID,
            budget_snapshot={"tier": "test", "allocation": {}},
        ))
        self.assertTrue(created)
        self.assertEqual(view.status.value, "queued")
        run = next(row for row in session.rows if isinstance(row, ResearchRunRow))
        self.assertEqual(run.run_kind, RunKind.RESEARCH.value)
        self.assertEqual(sum(isinstance(row, TaskDispatchOutboxRow) for row in session.rows), 1)
        claimed = _run_immediate(TaskDispatchOutboxRepository(_FakeSessions(session)).claim_batch())
        self.assertEqual(len(claimed), 1)

    def test_user_facing_list_and_get_filter_by_typed_run_kind(self) -> None:
        session = _FakeSession()
        repo = ResearchRunRepository(_FakeSessions(session))
        self.assertEqual(_run_immediate(repo.list_recent(OWNER, limit=10)), [])
        with self.assertRaises(ResearchRunNotFoundError):
            _run_immediate(repo.get(OWNER, uuid4()))
        sql = "\n".join(str(statement) for statement in session.statements)
        self.assertGreaterEqual(sql.count("run_kind"), 2)


class MigrationContractTests(unittest.TestCase):
    def test_single_narrow_run_kind_migration_defaults_existing_rows(self) -> None:
        path = Path("apps/api/alembic/versions/20261013_0026_research_run_kind.py")
        spec = importlib.util.spec_from_file_location("run_kind_migration_test", path)
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.object(module.op, "add_column") as add_column:
            module.upgrade()
        args = add_column.call_args.args
        self.assertEqual(args[0], "research_runs")
        column = args[1]
        self.assertEqual(column.name, "run_kind")
        self.assertFalse(column.nullable)
        self.assertEqual(str(column.server_default.arg), "research")
        with patch.object(module.op, "drop_column") as drop_column:
            module.downgrade()
        drop_column.assert_called_once_with("research_runs", "run_kind")


class PreflightTests(unittest.TestCase):
    def test_wrong_manifest_is_rejected_before_any_external_dependency(self) -> None:
        context, _profile, _credential = _context_and_profile()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            lane = ProspectiveAcquisitionExecutionLane(
                repository=SimpleNamespace(),
                ledger=TaskExecutionLedger(Path(temp)),
                profile=SimpleNamespace(
                    profile_version="prospective-acquisition-profile.v1",
                    provider_outbound_max_per_task=1,
                    search_outbound_max_per_task=1,
                    search_results_per_task=1,
                    pages_read_per_task=1,
                    reader_outbound_max_per_task=4,
                ),
                profile_service=SimpleNamespace(),
                profile_repository=SimpleNamespace(),
                execution_context=context,
                components_ready=False,
            )
            batch = VerifiedTaskBatch(
                "0" * 64,
                _frozen_tasks(),
                {"status": "invalid"},
                Path(temp) / "manifest.json",
            )
            with self.assertRaises(AcquisitionLaneBlocked):
                _run_immediate(lane.preflight(batch))
            self.assertEqual(lane._preflight_task_ids, frozenset())

    def test_incomplete_production_readiness_blocks_before_profile_resolution(self) -> None:
        context, _profile, _credential = _context_and_profile()

        class NeverCalled:
            async def list_profiles(self, _owner_hash):
                raise AssertionError("provider identity lookup must not run after failed readiness")

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            lane = ProspectiveAcquisitionExecutionLane(
                repository=SimpleNamespace(),
                ledger=TaskExecutionLedger(Path(temp)),
                profile=SimpleNamespace(
                    profile_version="prospective-acquisition-profile.v1",
                    provider_outbound_max_per_task=1,
                    search_outbound_max_per_task=1,
                    search_results_per_task=1,
                    pages_read_per_task=1,
                    reader_outbound_max_per_task=4,
                ),
                profile_service=NeverCalled(),
                profile_repository=NeverCalled(),
                execution_context=context,
                components_ready=False,
            )
            batch = VerifiedTaskBatch(
                MANIFEST_SHA256,
                _frozen_tasks(),
                {"status": "verified"},
                Path(temp) / "manifest.json",
            )
            with self.assertRaisesRegex(
                AcquisitionLaneBlocked, "FROZEN_MANIFEST_VERIFICATION_INCOMPLETE"
            ):
                _run_immediate(lane.preflight(batch))
            self.assertEqual(lane._preflight_task_ids, frozenset())


class BoundedProfileContractTests(unittest.TestCase):
    def test_frozen_per_task_caps_remain_exact(self) -> None:
        self.assertEqual(BoundedProspectiveAcquisitionProfile.provider_outbound_max_per_task, 1)
        self.assertEqual(BoundedProspectiveAcquisitionProfile.search_outbound_max_per_task, 1)
        self.assertEqual(BoundedProspectiveAcquisitionProfile.search_results_per_task, 1)
        self.assertEqual(BoundedProspectiveAcquisitionProfile.pages_read_per_task, 1)
        self.assertEqual(BoundedProspectiveAcquisitionProfile.reader_outbound_max_per_task, 4)


class FakeLaneEntryTests(unittest.TestCase):
    def _make_lane(self, temp: str, status: str):
        context, _profile, _credential = _context_and_profile()
        task = _frozen_tasks()[0]
        ledger = TaskExecutionLedger(Path(temp) / "execution")

        class FakeRepository:
            def __init__(self) -> None:
                self.create_calls = 0
                self.finalize_calls: list[bool] = []

            async def create_for_prospective_acquisition(self, _context, **_kwargs):
                self.create_calls += 1
                if ledger.state(task.task_execution_id) != "EXTERNAL_EFFECT_STARTED":
                    raise AssertionError("effect marker must precede ResearchRun creation")
                return object(), True

            async def finalize_prospective_acquisition_task(
                self, _run_id, *, worker_task_id, success, safe_failure_code=None
            ):
                self.finalize_calls.append(success)
                return worker_task_id == f"pa-{task.task_execution_id[:48]}"

        repository = FakeRepository()
        lane = ProspectiveAcquisitionExecutionLane(
            repository=repository,
            ledger=ledger,
            profile=SimpleNamespace(),
            profile_service=SimpleNamespace(),
            profile_repository=SimpleNamespace(),
            execution_context=context,
            components_ready=True,
        )
        lane._preflight_task_ids = frozenset({task.task_execution_id})
        claim = ClaimRecord(
            claim_id="claim-safe-id",
            created_at="2026-10-13T00:00:00Z",
            source_refs=("source-ref",),
            source_key="source-diversity-hash",
            owner_key="source-owner-hash",
            content_hash="f" * 64,
            provenance_complete=True,
            identity_isolation_attested=True,
            evidence_ids=("evidence-safe-id",),
            source_ids=("source-safe-id",),
            snapshot_ids=("snapshot-safe-id",),
        )
        execution = BoundedTaskExecution(
            task_id=task.task_id,
            run_id=str(task.run_id),
            status=status,
            result=TaskResult(task.task_id, str(task.run_id), (claim,)),
            provider_outbound_count=1,
            search_outbound_count=1,
            reader_outbound_count=1,
            blocked_extra_outbound_count=0,
        )

        async def fake_execute_with_heartbeat(_task):
            return execution

        lane._execute_with_heartbeat = fake_execute_with_heartbeat
        return lane, task, repository, ledger

    def test_successful_task_runs_once_then_skips_and_retains_receipt_reference(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            lane, task, repository, ledger = self._make_lane(temp, "CLAIM_RESULT")
            async def no_drift(*_args):
                return None

            with patch(
                "scripts.prospective_acquisition.execution_lane.assert_provider_binding_unchanged",
                new=no_drift,
            ):
                result = _run_immediate(lane.execute_task(task))
                skipped = _run_immediate(lane.execute_task(task))
            self.assertEqual(result["state"], "SUCCEEDED")
            self.assertEqual(skipped["skipped"], True)
            self.assertEqual(repository.create_calls, 1)
            self.assertEqual(repository.finalize_calls, [True])
            self.assertEqual(ledger.state(task.task_execution_id), "SUCCEEDED")
            receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(receipt["persistence_receipt_reference"]), 1)
            self.assertIn("persistence_receipt_sha256", receipt["persistence_receipt_reference"][0])
            self.assertNotIn("source_owner_key", receipt["persistence_receipt_reference"][0])

    def test_failed_task_is_terminal_and_is_never_retried_or_replaced(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
            lane, task, repository, ledger = self._make_lane(temp, "FAILED:TimeoutError")
            async def no_drift(*_args):
                return None

            with patch(
                "scripts.prospective_acquisition.execution_lane.assert_provider_binding_unchanged",
                new=no_drift,
            ):
                result = _run_immediate(lane.execute_task(task))
                with self.assertRaises(ExecutionLedgerError):
                    _run_immediate(lane.execute_task(task))
            self.assertEqual(result["state"], "FAILED_AFTER_EFFECT")
            self.assertEqual(repository.create_calls, 1)
            self.assertEqual(repository.finalize_calls, [False])
            self.assertEqual(ledger.state(task.task_execution_id), "FAILED_AFTER_EFFECT")


class PersistencePassthroughTests(unittest.TestCase):
    def test_existing_production_receipt_ids_are_passed_through_unchanged(self) -> None:
        claim_id, evidence_id, source_id, snapshot_id = uuid4(), uuid4(), uuid4(), uuid4()
        receipt = ClaimPersistenceReceipt(
            claim_id=claim_id,
            evidence_ids=(evidence_id,),
            source_ids=(source_id,),
            snapshot_ids=(snapshot_id,),
            source_provenance_refs=("source-ref-safe",),
            source_diversity_key="domain-hash",
            source_owner_key="owner-hash",
            claim_content_hash="c" * 64,
            claim_text="frozen claim",
            evidence_receipts=(
                EvidencePersistenceReceipt(
                    claim_id=claim_id,
                    evidence_id=evidence_id,
                    source_id=source_id,
                    snapshot_id=snapshot_id,
                    snapshot_content_hash="d" * 64,
                    question_id="task-001",
                    dimension_key="task-001:d1",
                    source_owner_key="owner-hash",
                    canonical_span=None,
                    accepted_and_persisted=True,
                ),
            ),
            created_at=datetime.now(UTC),
            accepted_and_persisted=True,
        )
        persistence = PagePersistenceResult(1, 1, (receipt,))

        class FakeRepository:
            async def record_page_with_receipts(self, *_args, **_kwargs):
                return persistence

        class FakeArtifactStore:
            async def save_page(self, *_args):
                return "runs/fake/source.txt"

        target = ResearchTarget(
            plan_version=1,
            question_id="task-001",
            question="frozen text",
            query="frozen text",
            gap_id=uuid4(),
            tool_call_id=uuid4(),
            source_id_seed=source_id,
        )
        task = SimpleNamespace(
            task_id="task-001",
            run_id=str(uuid4()),
            task_text="frozen text",
            execution_context_id="context-id",
            research_target=target,
            worker_task_id="worker-id",
        )
        component = ProductionClaimFormationComponent(
            FakeRepository(),
            FakeArtifactStore(),
            task=task,
            target=target,
            run_id=UUID(task.run_id),
            worker_task_id=task.worker_task_id,
        )
        page = ReadPage(
            final_url="https://example.test/",
            title="Fixture",
            clean_text="x" * 100,
            content_hash="a" * 64,
            fetched_at=datetime.now(UTC),
        )
        evidence = ExtractedEvidenceBatch(
            evidence=(
                ScoredEvidence(
                    candidate=EvidenceCandidate(
                        claim="A fixture claim that satisfies the minimum text length.",
                        exact_quote="A fixture exact quote that satisfies the minimum length.",
                        relation=EvidenceRelation.SUPPORTS,
                        relevance=0.8,
                        confidence=0.8,
                    ),
                    source_reliability=0.9,
                    evidence_score=0.8,
                    accepted=True,
                ),
            ),
            usage=SimpleNamespace(),
            context_manifest={},
        )
        claims = _run_immediate(component.record_page(task, page, evidence))
        claim = claims[0]
        self.assertEqual(claim.claim_id, str(claim_id))
        self.assertEqual(claim.evidence_ids, (str(evidence_id),))
        self.assertEqual(claim.source_ids, (str(source_id),))
        self.assertEqual(claim.snapshot_ids, (str(snapshot_id),))
        self.assertEqual(claim.source_key, "domain-hash")
        self.assertEqual(claim.owner_key, "owner-hash")


if __name__ == "__main__":
    unittest.main()
