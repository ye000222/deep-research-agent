from __future__ import annotations

import unittest
from dataclasses import fields
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import httpx
from app.domain.provider_profiles import ProviderProfileView
from app.domain.providers import AdapterType
from app.infrastructure.db.models import CredentialVersionRow
from app.security.client_sessions import ClientSession

from scripts.prospective_acquisition.control import (
    ProspectiveAcquisitionHarness,
)
from scripts.prospective_acquisition.execution_identity import (
    ExecutionProviderBindingChanged,
    create_execution_context,
    resolve_provider_binding,
)
from scripts.prospective_acquisition.profile import (
    AcquisitionComponents,
    AcquisitionSource,
    AcquisitionTask,
    BoundedProspectiveAcquisitionProfile,
)
from tests.unit.test_prospective_acquisition_control import claim


class FakeSearch:
    def __init__(self, results: list[AcquisitionSource]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, *, limit: int) -> list[AcquisitionSource]:
        self.calls.append((query, limit))
        return self.results


class FakeReader:
    def __init__(self, page: object) -> None:
        self.page = page
        self.calls: list[str] = []

    async def read(self, url: str) -> object:
        self.calls.append(url)
        return self.page


class FakeExtractor:
    def __init__(self, client: httpx.AsyncClient, *, nested_call: bool = False) -> None:
        self.client = client
        self.nested_call = nested_call
        self.calls: list[tuple[str, object]] = []

    async def extract(self, query: str, page: object) -> list[object]:
        self.calls.append((query, page))
        await self.client.post("https://provider.invalid/extract", json={"query": query})
        if self.nested_call:
            await self.client.post("https://provider.invalid/nested", json={"query": query})
        return ["fake-evidence"]


class FakeClaimFormation:
    def __init__(self, claims: tuple[object, ...]) -> None:
        self.claims = claims
        self.calls = 0

    async def record_page(
        self,
        task: AcquisitionTask,
        page: object,
        evidence: list[object],
    ) -> tuple[object, ...]:
        self.calls += 1
        return self.claims


class BoundedProfileTests(unittest.IsolatedAsyncioTestCase):
    async def _context(self):
        context = create_execution_context(
            ClientSession(uuid4(), "a" * 64, False),
            population_id="population-test",
            batch_id="batch-test",
            protocol_version="prospective-development-population.v1",
        )
        now = datetime(2026, 10, 2, tzinfo=UTC)
        profile_id = uuid4()
        credential_id = uuid4()
        view = ProviderProfileView(
            profile_id=profile_id,
            name="fake",
            adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
            base_url="https://provider.test/v1",
            endpoint_host="provider.test",
            model="test-model",
            status="active",
            version=1,
            is_default=True,
            credential_version_id=credential_id,
            credential_version=1,
            credential_last_four="0000",
            credential_fingerprint="fake-safe",
            created_at=now,
            updated_at=now,
        )

        class FakeProfileService:
            def __init__(self) -> None:
                self.views = [view]

            async def list_profiles(self, owner_hash: str):
                assert owner_hash == context.owner_hash
                return self.views

        class FakeProfileRepository:
            async def get_active(self, owner_hash: str, requested_profile_id):
                assert owner_hash == context.owner_hash
                assert requested_profile_id == profile_id
                row = SimpleNamespace(
                    id=profile_id,
                    owner_hash=owner_hash,
                    status="active",
                    is_default=True,
                    deleted_at=None,
                    version=1,
                    adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT.value,
                    model="test-model",
                    normalized_base_url="https://provider.test/v1",
                    auth_type="bearer",
                    non_secret_settings={},
                )
                credential = CredentialVersionRow(
                    id=credential_id,
                    profile_id=profile_id,
                    scope="saved_profile",
                    credential_version=1,
                    ciphertext=b"fixture-ciphertext-not-a-secret",
                    nonce=b"0" * 12,
                    key_version=1,
                    aad_version=1,
                    hmac_fingerprint="d" * 64,
                    last_four="0000",
                    revoked_at=None,
                    deleted_at=None,
                )
                return row, credential

        profile_service = FakeProfileService()
        profile_repository = FakeProfileRepository()
        binding = await resolve_provider_binding(context, profile_service, profile_repository)
        return (
            context.with_provider_binding(binding),
            profile_service,
            profile_repository,
        )

    def _profile(
        self,
        *,
        claims: tuple[object, ...] = (),
        results: list[AcquisitionSource] | None = None,
        nested_call: bool = False,
        transport_error: Exception | None = None,
    ):
        trace: dict[str, object] = {"outbound": []}

        def transport_factory() -> httpx.AsyncBaseTransport:
            def handler(request: httpx.Request) -> httpx.Response:
                cast_list = trace["outbound"]
                assert isinstance(cast_list, list)
                cast_list.append(str(request.url))
                if transport_error is not None:
                    raise transport_error
                return httpx.Response(200, json={"ok": True})

            return httpx.MockTransport(handler)

        def component_factory(
            task: AcquisitionTask, context, binding, clients
        ) -> AcquisitionComponents:
            self.assertEqual(task.execution_context_id, context.execution_context_id)
            self.assertEqual(binding, context.provider_binding)
            return AcquisitionComponents(
                search=FakeSearch(
                    [AcquisitionSource("https://source.test/a", "A")]
                    if results is None
                    else results
                ),
                reader=FakeReader({"content": "fake page"}),
                extractor=FakeExtractor(clients.provider, nested_call=nested_call),
                claim_formation=FakeClaimFormation(claims),
            )

        profile = BoundedProspectiveAcquisitionProfile(
            component_factory,
            expected_task_count=1,
            transport_factory=transport_factory,
        )
        return profile, trace

    async def test_fixed_path_reaches_claim_admission_and_uninspected_attestation(self) -> None:
        profile, trace = self._profile(claims=(claim("c1"),))
        context, profile_service, profile_repository = await self._context()
        task = _task("task-001", "run-001", context)
        result = await profile.execute(
            (task,),
            execution_context=context,
            profile_service=profile_service,
            profile_repository=profile_repository,
        )
        self.assertEqual(result.executions[0].status, "CLAIM_RESULT")
        self.assertEqual(result.batch_provider_outbound_count, 1)
        self.assertEqual(len(trace["outbound"]), 1)

        written: list[dict[str, object]] = []
        admissions = ProspectiveAcquisitionHarness(
            population_id="population-test",
            batch_id="batch-test",
            protocol_version="prospective-development-population.v1",
        ).execute(
            (task.task_id,),
            lambda _: result.executions[0].result,
            attestation_writer=written.append,
            artifact_schema_valid=lambda _: True,
            admitted_at="2026-10-02T00:00:00Z",
        )
        self.assertTrue(admissions.admissions[0].admitted)
        self.assertEqual(written[0]["initial_exposure_state"], "UNINSPECTED")

    async def test_zero_claim_task_stops_after_fixed_path(self) -> None:
        profile, trace = self._profile()
        context, profile_service, profile_repository = await self._context()
        result = await profile.execute(
            (_task("t1", "r1", context),),
            execution_context=context,
            profile_service=profile_service,
            profile_repository=profile_repository,
        )
        self.assertEqual(result.executions[0].status, "ZERO_CLAIM")
        self.assertEqual(result.executions[0].result.claims, ())
        self.assertEqual(len(trace["outbound"]), 1)

    async def test_multiple_claims_from_one_task_are_returned_without_extra_work(self) -> None:
        profile, trace = self._profile(claims=(claim("c1"), claim("c2", source="source-b")))
        context, profile_service, profile_repository = await self._context()
        result = await profile.execute(
            (_task("t1", "r1", context),),
            execution_context=context,
            profile_service=profile_service,
            profile_repository=profile_repository,
        )
        self.assertEqual(len(result.executions[0].result.claims), 2)
        self.assertEqual(result.executions[0].provider_outbound_count, 1)
        self.assertEqual(len(trace["outbound"]), 1)

    async def test_nested_provider_call_is_blocked_before_second_transport_dispatch(self) -> None:
        profile, trace = self._profile(nested_call=True)
        context, profile_service, profile_repository = await self._context()
        result = await profile.execute(
            (_task("t1", "r1", context),),
            execution_context=context,
            profile_service=profile_service,
            profile_repository=profile_repository,
        )
        execution = result.executions[0]
        self.assertTrue(execution.status.startswith("FAILED:"))
        self.assertEqual(execution.provider_outbound_count, 1)
        self.assertEqual(execution.blocked_extra_outbound_count, 1)
        self.assertEqual(len(trace["outbound"]), 1)

    async def test_transport_failure_consumes_only_one_attempt_without_retry(self) -> None:
        profile, trace = self._profile(transport_error=httpx.ConnectError("fake failure"))
        context, profile_service, profile_repository = await self._context()
        result = await profile.execute(
            (_task("t1", "r1", context),),
            execution_context=context,
            profile_service=profile_service,
            profile_repository=profile_repository,
        )
        execution = result.executions[0]
        self.assertEqual(execution.provider_outbound_count, 1)
        self.assertEqual(execution.blocked_extra_outbound_count, 0)
        self.assertEqual(len(trace["outbound"]), 1)
        self.assertTrue(execution.status.startswith("FAILED:"))

    async def test_search_empty_stops_before_reader_and_model_outbound(self) -> None:
        profile, trace = self._profile(results=[])
        context, profile_service, profile_repository = await self._context()
        result = await profile.execute(
            (_task("t1", "r1", context),),
            execution_context=context,
            profile_service=profile_service,
            profile_repository=profile_repository,
        )
        self.assertEqual(result.executions[0].status, "NO_SEARCH_RESULT")
        self.assertEqual(result.batch_provider_outbound_count, 0)
        self.assertEqual(trace["outbound"], [])

    async def test_task_manifest_count_is_exact_and_not_topped_up(self) -> None:
        profile, _ = self._profile()
        context, profile_service, profile_repository = await self._context()
        with self.assertRaises(ValueError):
            await profile.execute(
                (),
                execution_context=context,
                profile_service=profile_service,
                profile_repository=profile_repository,
            )

    async def test_mismatched_task_context_fails_before_factory_or_budget(self) -> None:
        profile, trace = self._profile()
        context, profile_service, profile_repository = await self._context()
        task = AcquisitionTask("t1", "r1", "task", "another-context")
        with self.assertRaisesRegex(ValueError, "does not match"):
            await profile.execute(
                (task,),
                execution_context=context,
                profile_service=profile_service,
                profile_repository=profile_repository,
            )
        self.assertEqual(trace["outbound"], [])

    async def test_provider_drift_fails_before_component_or_outbound(self) -> None:
        profile, trace = self._profile()
        context, profile_service, profile_repository = await self._context()
        profile_service.views = []

        with self.assertRaises(ExecutionProviderBindingChanged):
            await profile.execute(
                (_task("t1", "r1", context),),
                execution_context=context,
                profile_service=profile_service,
                profile_repository=profile_repository,
            )
        self.assertEqual(trace["outbound"], [])

    async def test_report_and_gap_closure_are_not_profile_components(self) -> None:
        names = {field.name for field in fields(AcquisitionComponents)}
        self.assertEqual(names, {"search", "reader", "extractor", "claim_formation"})
        self.assertNotIn("report_writer", names)
        self.assertNotIn("gap_closure", names)

    async def test_search_and_reader_dispatches_are_counted_at_http_transport(self) -> None:
        context, profile_service, profile_repository = await self._context()
        dispatched = {"search": 0, "reader": 0, "provider": 0}

        def transport(name: str) -> httpx.MockTransport:
            def handler(request: httpx.Request) -> httpx.Response:
                dispatched[name] += 1
                return httpx.Response(200, json={"ok": True})

            return httpx.MockTransport(handler)

        class Search:
            def __init__(self, client: httpx.AsyncClient) -> None:
                self.client = client

            async def search(self, query: str, *, limit: int):
                await self.client.get("https://search.test/query")
                return [AcquisitionSource("https://source.test/a", "A")]

        class Reader:
            def __init__(self, client: httpx.AsyncClient) -> None:
                self.client = client

            async def read(self, url: str):
                await self.client.get(url)
                return {"content": "page"}

        class Extractor:
            def __init__(self, client: httpx.AsyncClient) -> None:
                self.client = client

            async def extract(self, query: str, page: object):
                await self.client.post("https://provider.test/extract")
                return ["candidate"]

        class Formation:
            async def record_page(self, task, page, evidence):
                return (claim("accepted"),)

        def factory(task, execution_context, binding, clients):
            return AcquisitionComponents(
                Search(clients.search),
                Reader(clients.reader),
                Extractor(clients.provider),
                Formation(),
            )

        profile = BoundedProspectiveAcquisitionProfile(
            factory,
            expected_task_count=1,
            transport_factory=lambda: transport("provider"),
            search_transport_factory=lambda: transport("search"),
            reader_transport_factory=lambda: transport("reader"),
        )
        result = await profile.execute(
            (_task("t1", "r1", context),),
            execution_context=context,
            profile_service=profile_service,
            profile_repository=profile_repository,
        )
        execution = result.executions[0]
        self.assertEqual(execution.status, "CLAIM_RESULT")
        self.assertEqual(dispatched, {"search": 1, "reader": 1, "provider": 1})
        self.assertEqual(execution.search_outbound_count, 1)
        self.assertEqual(execution.reader_outbound_count, 1)
        self.assertEqual(execution.provider_outbound_count, 1)

    async def test_search_n_plus_one_is_blocked_before_transport_and_claim_result(self) -> None:
        context, profile_service, profile_repository = await self._context()
        dispatched = 0
        claims_written = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal dispatched
            dispatched += 1
            return httpx.Response(200)

        class Search:
            def __init__(self, client: httpx.AsyncClient) -> None:
                self.client = client

            async def search(self, query: str, *, limit: int):
                await self.client.get("https://search.test/first")
                await self.client.get("https://search.test/second")
                return [AcquisitionSource("https://source.test/a", "A")]

        class Formation:
            async def record_page(self, task, page, evidence):
                nonlocal claims_written
                claims_written += 1
                return (claim("must-not-persist"),)

        def factory(task, execution_context, binding, clients):
            return AcquisitionComponents(
                Search(clients.search),
                FakeReader({}),
                FakeExtractor(clients.provider),
                Formation(),
            )

        profile = BoundedProspectiveAcquisitionProfile(
            factory,
            expected_task_count=1,
            search_transport_factory=lambda: httpx.MockTransport(handler),
        )
        result = await profile.execute(
            (_task("t1", "r1", context),),
            execution_context=context,
            profile_service=profile_service,
            profile_repository=profile_repository,
        )
        execution = result.executions[0]
        self.assertTrue(execution.status.startswith("FAILED:AcquisitionBlocked"))
        self.assertEqual(dispatched, 1)
        self.assertEqual(execution.search_outbound_count, 1)
        self.assertEqual(execution.blocked_extra_outbound_count, 1)
        self.assertEqual(claims_written, 0)

    async def test_reader_redirect_budget_blocks_fifth_physical_fetch(self) -> None:
        context, profile_service, profile_repository = await self._context()
        dispatched = 0
        extraction_calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal dispatched
            dispatched += 1
            return httpx.Response(200)

        class Reader:
            def __init__(self, client: httpx.AsyncClient) -> None:
                self.client = client

            async def read(self, url: str):
                for _ in range(5):
                    await self.client.get("https://source.test/page")
                return {"content": "unreachable"}

        class Extractor:
            async def extract(self, query: str, page: object):
                nonlocal extraction_calls
                extraction_calls += 1
                return []

        def factory(task, execution_context, binding, clients):
            return AcquisitionComponents(
                FakeSearch([AcquisitionSource("https://source.test/a", "A")]),
                Reader(clients.reader),
                Extractor(),
                FakeClaimFormation(()),
            )

        profile = BoundedProspectiveAcquisitionProfile(
            factory,
            expected_task_count=1,
            reader_transport_factory=lambda: httpx.MockTransport(handler),
        )
        result = await profile.execute(
            (_task("t1", "r1", context),),
            execution_context=context,
            profile_service=profile_service,
            profile_repository=profile_repository,
        )
        execution = result.executions[0]
        self.assertTrue(execution.status.startswith("FAILED:AcquisitionBlocked"))
        self.assertEqual(dispatched, 4)
        self.assertEqual(execution.reader_outbound_count, 4)
        self.assertEqual(execution.blocked_extra_outbound_count, 1)
        self.assertEqual(extraction_calls, 0)


if __name__ == "__main__":
    unittest.main()


def _task(task_id: str, run_id: str, context) -> AcquisitionTask:
    return AcquisitionTask(task_id, run_id, "task", context.execution_context_id)
