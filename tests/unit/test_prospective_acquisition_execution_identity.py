from __future__ import annotations

import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from app.domain.provider_profiles import ProviderProfileView
from app.domain.providers import AdapterType
from app.infrastructure.db.models import CredentialVersionRow
from app.security.client_sessions import ClientSession

from scripts.prospective_acquisition.execution_identity import (
    ExecutionContextMismatch,
    ExecutionIdentityError,
    ExecutionProviderBindingChanged,
    FrozenProviderBinding,
    OwnerScopeRequired,
    ProviderCardinalityError,
    ProviderCredentialVersionInvalid,
    assert_provider_binding_unchanged,
    bind_task_to_context,
    create_bound_research_run,
    create_execution_context,
    operator_probe_reference,
    resolve_provider_binding,
    verify_operator_probe_reference,
)

OWNER = "a" * 64
OTHER_OWNER = "b" * 64
PROFILE_ID = uuid4()
CREDENTIAL_ID = uuid4()
_UNSET = object()


def credential_row(
    view: ProviderProfileView, credential_version: object
) -> CredentialVersionRow:
    return CredentialVersionRow(
        id=view.credential_version_id,
        profile_id=view.profile_id,
        scope="saved_profile",
        credential_version=credential_version,
        ciphertext=b"fixture-ciphertext-not-a-secret",
        nonce=b"0" * 12,
        key_version=1,
        aad_version=1,
        hmac_fingerprint="d" * 64,
        last_four="0000",
    )


def profile_view(
    *,
    profile_id=PROFILE_ID,
    version: int = 4,
    credential_version: int = 3,
    credential_id=CREDENTIAL_ID,
    is_default: bool = True,
    status: str = "active",
) -> ProviderProfileView:
    now = datetime(2026, 10, 2, tzinfo=UTC)
    return ProviderProfileView(
        profile_id=profile_id,
        name="test",
        adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
        base_url="https://provider.example/v1",
        endpoint_host="provider.example",
        model="frozen-model",
        status=status,
        version=version,
        is_default=is_default,
        credential_version_id=credential_id,
        credential_version=credential_version,
        credential_last_four="0000",
        credential_fingerprint="safe-fingerprint",
        created_at=now,
        updated_at=now,
    )


class FakeProfileService:
    def __init__(self, views: list[ProviderProfileView]) -> None:
        self.views = views
        self.owner_calls: list[str] = []

    async def list_profiles(self, owner_hash: str) -> list[ProviderProfileView]:
        self.owner_calls.append(owner_hash)
        return list(self.views)


class FakeProfileRepository:
    def __init__(self, *, owner_hash: str = OWNER, settings=None) -> None:
        self.owner_hash = owner_hash
        self.settings = settings or {"supports_json_mode": True}
        self.calls: list[tuple[str, object]] = []
        self.view = profile_view()
        self.credential_version_override: object = _UNSET
        self.credential_shape_override: object | None = None

    async def get_active(self, owner_hash: str, profile_id):
        self.calls.append((owner_hash, profile_id))
        view = self.view
        row = SimpleNamespace(
            id=view.profile_id,
            owner_hash=self.owner_hash,
            status="active",
            is_default=True,
            deleted_at=None,
            version=view.version,
            adapter_type=view.adapter_type.value,
            model=view.model,
            normalized_base_url=view.base_url,
            auth_type="bearer",
            non_secret_settings=self.settings,
        )
        if self.credential_shape_override is not None:
            credential = self.credential_shape_override
        else:
            version = (
                view.credential_version
                if self.credential_version_override is _UNSET
                else self.credential_version_override
            )
            credential = credential_row(view, version)
        return row, credential


def context(session: ClientSession | None = None):
    return create_execution_context(
        session or ClientSession(uuid4(), OWNER, False),
        population_id="population-dev",
        batch_id="batch-frozen",
        protocol_version="prospective-development-population.v1",
        created_at=datetime(2026, 10, 2, tzinfo=UTC),
    )


class ExecutionContextTests(unittest.TestCase):
    def test_authenticated_session_creates_immutable_context(self) -> None:
        execution = context()
        self.assertEqual(execution.owner_hash, OWNER)
        self.assertEqual(execution.owner_scope_source, "AUTHENTICATED_CLIENT_SESSION")
        with self.assertRaises((AttributeError, TypeError)):
            execution.owner_hash = OTHER_OWNER  # type: ignore[misc]

    def test_caller_cannot_override_scope_and_missing_scope_fails_closed(self) -> None:
        with self.assertRaises(OwnerScopeRequired):
            create_execution_context(
                None,
                population_id="population-dev",
                batch_id="batch-frozen",
                protocol_version="v1",
            )
        with self.assertRaises(OwnerScopeRequired):
            create_execution_context(
                ClientSession(uuid4(), "not-an-owner-hash", False),
                population_id="population-dev",
                batch_id="batch-frozen",
                protocol_version="v1",
            )

    def test_task_references_context_without_copying_raw_scope(self) -> None:
        execution = context()
        execution = execution.with_provider_binding(
            FrozenProviderBinding(
                profile_id="profile",
                adapter="openai_compatible_chat",
                model="model",
                endpoint_family="https://provider.example",
                profile_version=1,
                credential_version_id="credential",
                credential_version=1,
                credential_present=True,
                fingerprint_contract="provider-safe-config-fingerprint.v1",
                provider_safe_config_fingerprint="c" * 64,
            )
        )
        task = bind_task_to_context(
            task_id="task-1", run_id="run-1", task_text="fixed", context=execution
        )
        self.assertEqual(task["execution_context_id"], execution.execution_context_id)
        self.assertNotIn("owner_hash", task)


class ProviderBindingTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_resolution_uses_exact_owner_and_single_default(self) -> None:
        execution = context()
        service = FakeProfileService([profile_view()])
        repository = FakeProfileRepository()
        binding = await resolve_provider_binding(execution, service, repository)
        self.assertIsInstance(credential_row(profile_view(), 3), CredentialVersionRow)
        self.assertEqual(service.owner_calls, [OWNER])
        self.assertEqual(repository.calls, [(OWNER, PROFILE_ID)])
        self.assertEqual(binding.profile_id, str(PROFILE_ID))
        self.assertEqual(binding.credential_version_id, str(CREDENTIAL_ID))
        self.assertEqual(binding.credential_version, 3)
        self.assertEqual(binding.fingerprint_contract, "provider-safe-config-fingerprint.v1")
        safe_binding = binding.safe_dict()
        self.assertEqual(safe_binding["credential_version"], 3)
        self.assertNotIn("ciphertext", str(safe_binding))
        self.assertNotIn("fixture-ciphertext", str(safe_binding))

    async def test_actual_credential_row_version_must_match_profile_view(self) -> None:
        repository = FakeProfileRepository()
        repository.credential_version_override = 8
        with self.assertRaises(ProviderCredentialVersionInvalid):
            await resolve_provider_binding(
                context(), FakeProfileService([profile_view()]), repository
            )

    async def test_missing_and_invalid_credential_versions_fail_closed(self) -> None:
        for invalid_version in (None, True, "3", 3.5):
            with self.subTest(invalid_version=invalid_version):
                repository = FakeProfileRepository()
                repository.credential_version_override = invalid_version
                with self.assertRaises(ProviderCredentialVersionInvalid):
                    await resolve_provider_binding(
                        context(), FakeProfileService([profile_view()]), repository
                    )

    async def test_credential_row_without_canonical_field_fails_closed(self) -> None:
        repository = FakeProfileRepository()
        repository.credential_shape_override = SimpleNamespace(
            id=CREDENTIAL_ID, revoked_at=None, deleted_at=None
        )
        with self.assertRaises(ProviderCredentialVersionInvalid):
            await resolve_provider_binding(
                context(), FakeProfileService([profile_view()]), repository
            )

    async def test_zero_defaults_fails_closed(self) -> None:
        with self.assertRaisesRegex(ProviderCardinalityError, "NO_DEFAULT_PROFILE"):
            await resolve_provider_binding(
                context(), FakeProfileService([]), FakeProfileRepository()
            )

    async def test_multiple_same_scope_defaults_fails_closed(self) -> None:
        two = [profile_view(), profile_view(profile_id=uuid4())]
        with self.assertRaisesRegex(ProviderCardinalityError, "MULTIPLE_DEFAULT"):
            await resolve_provider_binding(
                context(), FakeProfileService(two), FakeProfileRepository()
            )

    async def test_defaults_in_other_owner_scope_are_irrelevant(self) -> None:
        execution = context()
        # The production service is called for only the bound owner. Its result
        # excludes defaults belonging to all other scopes.
        service = FakeProfileService([profile_view()])
        binding = await resolve_provider_binding(execution, service, FakeProfileRepository())
        self.assertEqual(service.owner_calls, [OWNER])
        self.assertNotEqual(execution.owner_hash, OTHER_OWNER)
        self.assertTrue(binding.profile_id)

    async def test_historical_profile_id_cannot_select_provider(self) -> None:
        execution = context()
        service = FakeProfileService([profile_view()])
        repository = FakeProfileRepository()
        binding = await resolve_provider_binding(execution, service, repository)
        self.assertEqual(binding.profile_id, str(PROFILE_ID))
        self.assertEqual(repository.calls[0], (OWNER, PROFILE_ID))

    async def test_repository_scope_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ExecutionIdentityError, "changed_during_resolution"):
            await resolve_provider_binding(
                context(),
                FakeProfileService([profile_view()]),
                FakeProfileRepository(owner_hash=OTHER_OWNER),
            )

    async def test_profile_id_drift_detected(self) -> None:
        execution = context()
        service = FakeProfileService([profile_view()])
        repo = FakeProfileRepository()
        binding = await resolve_provider_binding(execution, service, repo)
        frozen = execution.with_provider_binding(binding)
        repo.view = profile_view(profile_id=uuid4())
        service.views = [repo.view]
        with self.assertRaises(ExecutionProviderBindingChanged):
            await assert_provider_binding_unchanged(frozen, service, repo)

    async def test_profile_version_drift_detected(self) -> None:
        execution = context()
        service = FakeProfileService([profile_view()])
        repo = FakeProfileRepository()
        frozen = execution.with_provider_binding(
            await resolve_provider_binding(execution, service, repo)
        )
        repo.view = profile_view(version=5)
        service.views = [repo.view]
        with self.assertRaises(ExecutionProviderBindingChanged):
            await assert_provider_binding_unchanged(frozen, service, repo)

    async def test_credential_version_drift_detected(self) -> None:
        execution = context()
        service = FakeProfileService([profile_view()])
        repo = FakeProfileRepository()
        frozen = execution.with_provider_binding(
            await resolve_provider_binding(execution, service, repo)
        )
        repo.view = profile_view(credential_version=4)
        service.views = [repo.view]
        with self.assertRaises(ExecutionProviderBindingChanged):
            await assert_provider_binding_unchanged(frozen, service, repo)

    async def test_fingerprint_drift_detected(self) -> None:
        execution = context()
        service = FakeProfileService([profile_view()])
        repo = FakeProfileRepository()
        frozen = execution.with_provider_binding(
            await resolve_provider_binding(execution, service, repo)
        )
        repo.settings = {"supports_json_mode": False}
        with self.assertRaises(ExecutionProviderBindingChanged):
            await assert_provider_binding_unchanged(frozen, service, repo)


class RunAndArtifactBindingTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_creation_receives_exact_context_owner_scope_and_credential(self) -> None:
        execution = context()
        binding = await resolve_provider_binding(
            execution, FakeProfileService([profile_view()]), FakeProfileRepository()
        )
        frozen = execution.with_provider_binding(binding)
        calls: list[tuple[str, dict[str, object]]] = []

        class RunService:
            async def create_run(self, owner_hash: str, **kwargs):
                calls.append((owner_hash, kwargs))
                return "fake-run", True

        await create_bound_research_run(
            frozen,
            RunService(),
            profile_service=FakeProfileService([profile_view()]),
            profile_repository=FakeProfileRepository(),
            idempotency_key="idempotency-test",
            query="fixed task",
            saved_profile_version_id=CREDENTIAL_ID,
            budget_tier="quick",
        )
        self.assertEqual(calls[0][0], OWNER)
        self.assertEqual(calls[0][1]["saved_profile_version_id"], CREDENTIAL_ID)

    async def test_run_profile_mismatch_fails_before_service_call(self) -> None:
        execution = context()
        binding = await resolve_provider_binding(
            execution, FakeProfileService([profile_view()]), FakeProfileRepository()
        )
        frozen = execution.with_provider_binding(binding)

        class RunService:
            async def create_run(self, *_args, **_kwargs):
                self.fail("must not create a mismatched run")

        with self.assertRaises(ExecutionContextMismatch):
            await create_bound_research_run(
                frozen,
                RunService(),
                profile_service=FakeProfileService([profile_view()]),
                profile_repository=FakeProfileRepository(),
                idempotency_key="key",
                query="query",
                saved_profile_version_id=uuid4(),
                budget_tier="quick",
            )

    async def test_run_creation_stops_on_provider_binding_drift(self) -> None:
        execution = context()
        service = FakeProfileService([profile_view()])
        repository = FakeProfileRepository()
        frozen = execution.with_provider_binding(
            await resolve_provider_binding(execution, service, repository)
        )
        repository.view = profile_view(version=5)
        service.views = [repository.view]
        calls: list[bool] = []

        class RunService:
            async def create_run(self, *_args, **_kwargs):
                calls.append(True)
                return "unexpected", True

        with self.assertRaises(ExecutionProviderBindingChanged):
            await create_bound_research_run(
                frozen,
                RunService(),
                profile_service=service,
                profile_repository=repository,
                idempotency_key="key",
                query="query",
                saved_profile_version_id=CREDENTIAL_ID,
                budget_tier="quick",
            )
        self.assertEqual(calls, [])

    async def test_safe_artifact_and_operator_reference_redact_raw_owner(self) -> None:
        execution = context()
        binding = await resolve_provider_binding(
            execution, FakeProfileService([profile_view()]), FakeProfileRepository()
        )
        frozen = execution.with_provider_binding(binding)
        artifact = frozen.safe_artifact()
        reference = operator_probe_reference(frozen)
        serialized = str(artifact) + str(reference)
        self.assertNotIn(OWNER, serialized)
        self.assertIn("owner_scope_fingerprint", artifact)
        self.assertEqual(reference["execution_context_id"], frozen.execution_context_id)
        verify_operator_probe_reference(frozen, reference)
        changed = dict(reference, execution_context_id="wrong-context")
        with self.assertRaises(ExecutionContextMismatch):
            verify_operator_probe_reference(frozen, changed)


if __name__ == "__main__":
    unittest.main()
