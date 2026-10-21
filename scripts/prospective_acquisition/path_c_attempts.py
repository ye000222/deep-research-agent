"""Immutable PATH-C authorization/identity records and deterministic discovery.

No constructor writes an authorization. Only initialize_attempt writes attempt
records, after disk history and frozen runtime bindings have been validated.
Legacy Attempt 001 is a read-only virtual reference to the original location.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Digest = str


class GenerationAttemptError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class SafeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator("*", mode="after")
    @classmethod
    def utc_only(cls, value: Any) -> Any:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise ValueError("timestamp must be timezone-aware")
            return value.astimezone(UTC)
        return value


class HistoryScope(SafeModel):
    population_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    batch_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    execution_context_id: UUID
    generation_contract_version: str


class FrozenAttemptBinding(HistoryScope):
    execution_context_hash: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    generation_contract_sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    task_schema_sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    seed_amendment_version: Literal["path-c-seed-policy.v1"] = "path-c-seed-policy.v1"
    seed_amendment_sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    provider_fingerprint_contract: Literal["provider-safe-config-fingerprint.v1"] = (
        "provider-safe-config-fingerprint.v1"
    )
    provider_safe_config_fingerprint: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    provider_binding_sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")


def attempt_id_for(scope: HistoryScope, number: int, authorization_id: UUID | None) -> str:
    return digest(
        canonical_bytes(
            {
                "scope": scope.model_dump(mode="json"),
                "attempt_number": number,
                "authorization_id": str(authorization_id) if authorization_id is not None else None,
                "identity_rule": "path-c-attempt-identity.v1"
                if number > 1
                else "legacy-path-c-001.v1",
            }
        )
    )


class PathCAttemptAuthorization(FrozenAttemptBinding):
    artifact_version: Literal["path-c-attempt-authorization.v1"] = "path-c-attempt-authorization.v1"
    authorization_id: UUID
    attempt_number: int = Field(ge=2)
    attempt_id: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    previous_attempt_number: int = Field(ge=1)
    previous_attempt_id: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    previous_terminal_state: Literal["CONSUMED_FAILED"] = "CONSUMED_FAILED"
    previous_state_sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    # Added for the final recovery grant. Optional defaults preserve the exact
    # hash representation of already-frozen Attempts 002-005.
    canonical_runtime_attestation_run: Literal[
        "run_20261013T014431260Z_2a030977"
    ] | None = None
    canonical_runtime_attestation_manifest_sha256: Digest | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    maximum_provider_generation_dispatches: Literal[1] = 1
    retry: Literal[0] = 0
    regeneration: Literal[False] = False
    repair: Literal["none"] = "none"
    provider_fallback: Literal["none"] = "none"
    endpoint_fallback: Literal["none"] = "none"
    request_format_fallback: Literal["none"] = "none"
    no_top_up: Literal[True] = True
    no_replacement: Literal[True] = True
    authorization_status: Literal["AUTHORIZED"] = "AUTHORIZED"
    authorization_source: Literal["EXPLICIT_HUMAN_GOVERNANCE_DECISION"] = (
        "EXPLICIT_HUMAN_GOVERNANCE_DECISION"
    )
    purpose: Literal["ONE_SEPARATE_PATH_C_GENERATION_ATTEMPT"] = (
        "ONE_SEPARATE_PATH_C_GENERATION_ATTEMPT"
    )
    created_at_utc: datetime
    authorization_hash: Digest = Field(default="0" * 64, pattern=r"^[0-9a-f]{64}$")

    @field_validator("maximum_provider_generation_dispatches", "retry", mode="before")
    @classmethod
    def exact_integer_limits(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("call limits must be integer literals")
        return value

    @field_validator("regeneration", "no_top_up", "no_replacement", mode="before")
    @classmethod
    def exact_boolean_limits(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("generation restrictions must be boolean literals")
        return value

    def scope(self) -> HistoryScope:
        return HistoryScope(**{key: getattr(self, key) for key in HistoryScope.model_fields})

    def binding(self) -> FrozenAttemptBinding:
        return FrozenAttemptBinding(
            **{key: getattr(self, key) for key in FrozenAttemptBinding.model_fields}
        )

    def computed_hash(self) -> str:
        return digest(canonical_bytes(self.model_dump(
            mode="json", exclude={"authorization_hash"}, exclude_none=True,
        )))

    def with_integrity_hash(self) -> PathCAttemptAuthorization:
        """Pure value construction for a later governance task; no file writes."""
        return self.model_copy(update={"authorization_hash": self.computed_hash()})

    def validate_integrity(self) -> None:
        if self.authorization_hash != self.computed_hash():
            raise GenerationAttemptError("AUTHORIZATION_HASH_MISMATCH")
        if self.attempt_id != attempt_id_for(
            self.scope(), self.attempt_number, self.authorization_id
        ):
            raise GenerationAttemptError("ATTEMPT_IDENTITY_MISMATCH")
        if self.previous_attempt_number != self.attempt_number - 1:
            raise GenerationAttemptError("SKIPPED_ATTEMPT_NUMBER")
        attestation_bound = (
            self.canonical_runtime_attestation_run is not None
            and self.canonical_runtime_attestation_manifest_sha256 is not None
        )
        if self.attempt_number == 6 and not attestation_bound:
            raise GenerationAttemptError("CANONICAL_ATTESTATION_BINDING_MISSING")
        if self.attempt_number != 6 and attestation_bound:
            raise GenerationAttemptError("UNEXPECTED_CANONICAL_ATTESTATION_BINDING")


class AttemptIdentity(SafeModel):
    artifact_version: Literal["path-c-attempt-identity.v1"] = "path-c-attempt-identity.v1"
    binding: FrozenAttemptBinding
    attempt_number: int = Field(ge=2)
    attempt_id: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    authorization_id: UUID
    authorization_hash: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    authorization_file_sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    previous_attempt_id: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    previous_state_sha256: Digest = Field(pattern=r"^[0-9a-f]{64}$")
    created_at_utc: datetime


@dataclass(frozen=True)
class AttemptRecord:
    number: int
    attempt_id: str
    location: Path
    state: str
    state_sha256: str
    consumed: bool
    scope: HistoryScope
    created_at_utc: datetime
    authorization_id: UUID | None = None
    identity: AttemptIdentity | None = None
    state_data: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class AuthorizedAttempt:
    batch_root: Path
    authorization: PathCAttemptAuthorization
    authorization_file_sha256: str

    @property
    def location(self) -> Path:
        return self.batch_root / f"attempt_{self.authorization.attempt_number:03d}"

    def identity(self, created_at_utc: datetime | None = None) -> AttemptIdentity:
        auth = self.authorization
        return AttemptIdentity(
            binding=auth.binding(),
            attempt_number=auth.attempt_number,
            attempt_id=auth.attempt_id,
            authorization_id=auth.authorization_id,
            authorization_hash=auth.authorization_hash,
            authorization_file_sha256=self.authorization_file_sha256,
            previous_attempt_id=auth.previous_attempt_id,
            previous_state_sha256=auth.previous_state_sha256,
            created_at_utc=created_at_utc or auth.created_at_utc,
        )


@dataclass(frozen=True)
class AttemptSelection:
    latest: AttemptRecord
    authorized: AuthorizedAttempt | None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GenerationAttemptError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _read(path: Path) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink():
        raise GenerationAttemptError("ATTEMPT_SYMLINK_FORBIDDEN")
    try:
        raw = path.read_bytes()
        if len(raw) > 2 * 1024 * 1024:
            raise GenerationAttemptError("ATTEMPT_RECORD_TOO_LARGE")
        value: object = json.loads(raw, object_pairs_hook=_unique_object)
    except (OSError, ValueError, RecursionError) as exc:
        raise GenerationAttemptError("ATTEMPT_ARTIFACT_UNREADABLE") from exc
    if not isinstance(value, dict):
        raise GenerationAttemptError("ATTEMPT_ARTIFACT_NOT_OBJECT")
    return cast(dict[str, Any], value), raw


def _check_root(batch_root: Path, scope: HistoryScope) -> None:
    if batch_root.name != f"{scope.population_id}__{scope.batch_id}":
        raise GenerationAttemptError("LEGACY_LOCATION_MISMATCH")
    if batch_root.is_symlink() or not batch_root.is_dir():
        raise GenerationAttemptError("LEGACY_ATTEMPT_001_MISSING")


def _scope_matches(data: dict[str, Any], scope: HistoryScope) -> bool:
    return (
        data.get("population_id") == scope.population_id
        and data.get("batch_id") == scope.batch_id
        and data.get("execution_context_id") == str(scope.execution_context_id)
        and data.get("contract_version") == scope.generation_contract_version
    )


def read_history(batch_root: Path, scope: HistoryScope) -> tuple[AttemptRecord, ...]:
    """Only the exact frozen legacy location is mapped to number 1; no migration."""
    _check_root(batch_root, scope)
    legacy, raw = _read(batch_root / "generation_state.json")
    if not (
        _scope_matches(legacy, scope)
        and legacy.get("state_version") == "path-c-generation-state.v1"
        and legacy.get("state") == "CONSUMED_FAILED"
        and legacy.get("dispatch_count") == 1
        and legacy.get("physical_outbound_count") == 1
        and legacy.get("blocked_additional_outbound_count") == 0
        and legacy.get("failure_code") == "MODELGATEWAYERROR"
        and "attempt_number" not in legacy
        and "attempt_id" not in legacy
    ):
        raise GenerationAttemptError("LEGACY_ATTEMPT_001_HISTORY_INVALID")
    try:
        legacy_created = datetime.fromisoformat(str(legacy["reserved_at_utc"]))
        if legacy_created.tzinfo is None:
            raise ValueError("legacy timestamp must have timezone")
        legacy_created = legacy_created.astimezone(UTC)
    except (KeyError, ValueError) as exc:
        raise GenerationAttemptError("LEGACY_ATTEMPT_001_TIMESTAMP_INVALID") from exc
    records = [
        AttemptRecord(
            number=1,
            attempt_id=attempt_id_for(scope, 1, None),
            location=batch_root,
            state="CONSUMED_FAILED",
            state_sha256=digest(raw),
            consumed=True,
            scope=scope,
            created_at_utc=legacy_created,
            state_data=legacy,
        )
    ]
    directories: dict[int, Path] = {}
    for path in batch_root.iterdir():
        if not path.name.startswith("attempt_"):
            continue
        match = re.fullmatch(r"attempt_([0-9]{3,})", path.name)
        if match is None or not path.is_dir() or path.is_symlink():
            raise GenerationAttemptError("AMBIGUOUS_ATTEMPT_HISTORY")
        number = int(match[1])
        if number < 2 or path.name != f"attempt_{number:03d}" or number in directories:
            raise GenerationAttemptError("DUPLICATE_ATTEMPT_NUMBER")
        directories[number] = path
    ids = {records[0].attempt_id}
    authorization_ids: set[UUID] = set()
    for number in sorted(directories):
        if number != records[-1].number + 1 or records[-1].state != "CONSUMED_FAILED":
            raise GenerationAttemptError("SKIPPED_OR_NONTERMINAL_ATTEMPT_HISTORY")
        location = directories[number]
        identity_data, identity_raw = _read(location / "attempt_identity.json")
        if not set(AttemptIdentity.model_fields).issubset(identity_data):
            raise GenerationAttemptError("ATTEMPT_IDENTITY_INCOMPLETE")
        try:
            identity = AttemptIdentity.model_validate_json(canonical_bytes(identity_data))
        except ValidationError as exc:
            raise GenerationAttemptError("ATTEMPT_IDENTITY_INVALID") from exc
        identity_scope = HistoryScope(
            **{key: getattr(identity.binding, key) for key in HistoryScope.model_fields}
        )
        if identity.attempt_number != number or identity_scope != scope:
            raise GenerationAttemptError("ATTEMPT_IDENTITY_MISMATCH")
        if identity.attempt_id != attempt_id_for(scope, number, identity.authorization_id):
            raise GenerationAttemptError("ATTEMPT_IDENTITY_MISMATCH")
        if identity.attempt_id in ids or identity.authorization_id in authorization_ids:
            raise GenerationAttemptError("DUPLICATE_ATTEMPT_OR_AUTHORIZATION_ID")
        ids.add(identity.attempt_id)
        authorization_ids.add(identity.authorization_id)
        previous = records[-1]
        if (
            identity.previous_attempt_id != previous.attempt_id
            or identity.previous_state_sha256 != previous.state_sha256
        ):
            raise GenerationAttemptError("PARENT_ATTEMPT_MISMATCH")
        state, state_raw = _read(location / "generation_state.json")
        if not _scope_matches(state, scope) or any(
            state.get(key) != expected
            for key, expected in {
                "attempt_id": identity.attempt_id,
                "attempt_number": number,
                "authorization_id": str(identity.authorization_id),
                "authorization_hash": identity.authorization_hash,
                "authorization_file_sha256": identity.authorization_file_sha256,
                "attempt_identity_sha256": digest(identity_raw),
            }.items()
        ):
            raise GenerationAttemptError("ATTEMPT_STATE_IDENTITY_MISMATCH")
        status = state.get("state")
        if status not in {
            "AUTHORIZED_NOT_DISPATCHED",
            "DISPATCH_RESERVED",
            "CONSUMED_FAILED",
            "SUCCEEDED_AND_FROZEN",
        }:
            raise GenerationAttemptError("ATTEMPT_STATE_INVALID")
        receipt_path = location / "dispatch_reservation.json"
        reserved = receipt_path.exists()
        if reserved:
            receipt, _ = _read(receipt_path)
            if receipt.get("state") != "DISPATCH_RESERVED" or receipt.get("dispatch_count") != 1:
                raise GenerationAttemptError("DISPATCH_RECEIPT_INVALID")
            if any(
                receipt.get(key) != state.get(key)
                for key in (
                    "attempt_id",
                    "attempt_number",
                    "authorization_id",
                    "authorization_hash",
                    "authorization_file_sha256",
                    "attempt_identity_sha256",
                )
            ):
                raise GenerationAttemptError("DISPATCH_RECEIPT_IDENTITY_MISMATCH")
        elif status != "AUTHORIZED_NOT_DISPATCHED":
            raise GenerationAttemptError("CONSUMED_ATTEMPT_WITHOUT_RECEIPT")
        records.append(
            AttemptRecord(
                number=number,
                attempt_id=identity.attempt_id,
                location=location,
                state=str(status),
                state_sha256=digest(state_raw),
                consumed=reserved or status != "AUTHORIZED_NOT_DISPATCHED",
                scope=scope,
                created_at_utc=identity.created_at_utc,
                authorization_id=identity.authorization_id,
                identity=identity,
                state_data=state,
            )
        )
    return tuple(records)


def _read_authorizations(
    batch_root: Path,
    scope: HistoryScope,
    history: tuple[AttemptRecord, ...],
) -> dict[int, tuple[PathCAttemptAuthorization, str]]:
    directory = batch_root / "authorizations"
    if not directory.exists():
        return {}
    if directory.is_symlink() or not directory.is_dir():
        raise GenerationAttemptError("AUTHORIZATION_LOCATION_INVALID")
    result: dict[int, tuple[PathCAttemptAuthorization, str]] = {}
    ids: set[UUID] = set()
    for path in sorted(directory.iterdir()):
        match = re.fullmatch(r"attempt_([0-9]{3,})\.authorization\.json", path.name)
        if match is None or not path.is_file() or path.is_symlink():
            raise GenerationAttemptError("AMBIGUOUS_ATTEMPT_AUTHORIZATION")
        number = int(match[1])
        if number < 2 or path.name != f"attempt_{number:03d}.authorization.json":
            raise GenerationAttemptError("DUPLICATE_ATTEMPT_NUMBER")
        data, raw = _read(path)
        optional_compatibility_fields = {
            "canonical_runtime_attestation_run",
            "canonical_runtime_attestation_manifest_sha256",
        }
        required_fields = (
            set(PathCAttemptAuthorization.model_fields) - optional_compatibility_fields
        )
        if not required_fields.issubset(data):
            raise GenerationAttemptError("AUTHORIZATION_CONTRACT_INCOMPLETE")
        try:
            auth = PathCAttemptAuthorization.model_validate_json(canonical_bytes(data))
        except ValidationError as exc:
            raise GenerationAttemptError("AUTHORIZATION_CONTRACT_INVALID") from exc
        auth.validate_integrity()
        if auth.scope() != scope:
            raise GenerationAttemptError("AUTHORIZATION_SCOPE_MISMATCH")
        if auth.attempt_number != number or number > history[-1].number + 1:
            raise GenerationAttemptError("SKIPPED_ATTEMPT_NUMBER")
        if auth.authorization_id in ids or number in result:
            raise GenerationAttemptError("DUPLICATE_ATTEMPT_AUTHORIZATION")
        ids.add(auth.authorization_id)
        previous = history[number - 2]
        if (
            auth.previous_attempt_id != previous.attempt_id
            or auth.previous_terminal_state != previous.state
            or auth.previous_state_sha256 != previous.state_sha256
        ):
            raise GenerationAttemptError("PARENT_ATTEMPT_MISMATCH")
        if number <= history[-1].number:
            identity = history[number - 1].identity
            if identity is None or (
                identity.authorization_id != auth.authorization_id
                or identity.authorization_hash != auth.authorization_hash
                or identity.authorization_file_sha256 != digest(raw)
                or identity
                != AuthorizedAttempt(batch_root, auth, digest(raw)).identity(
                    created_at_utc=identity.created_at_utc
                )
            ):
                raise GenerationAttemptError("AUTHORIZATION_CHANGED_AFTER_ATTEMPT_CREATION")
        result[number] = auth, digest(raw)
    for record in history[1:]:
        if record.number not in result:
            raise GenerationAttemptError("ATTEMPT_AUTHORIZATION_MISSING")
    return result


def resolve_operator_attempt(batch_root: Path, scope: HistoryScope) -> AttemptSelection:
    """Read-only selection. A missing next grant never creates directories."""
    history = read_history(batch_root, scope)
    authorizations = _read_authorizations(batch_root, scope, history)
    latest = history[-1]
    if latest.state == "SUCCEEDED_AND_FROZEN":
        if latest.number + 1 in authorizations:
            raise GenerationAttemptError("NO_REPLACEMENT_GENERATION_ALLOWED")
        return AttemptSelection(latest, None)
    if latest.state == "DISPATCH_RESERVED" or (
        latest.state == "AUTHORIZED_NOT_DISPATCHED" and latest.consumed
    ):
        raise GenerationAttemptError("GENERATION_AUTHORIZATION_ALREADY_CONSUMED")
    number = latest.number if latest.state == "AUTHORIZED_NOT_DISPATCHED" else latest.number + 1
    if number not in authorizations:
        raise GenerationAttemptError("NO_SEPARATELY_AUTHORIZED_NEXT_ATTEMPT")
    auth, file_hash = authorizations[number]
    return AttemptSelection(latest, AuthorizedAttempt(batch_root, auth, file_hash))


def validate_authorized_attempt(
    attempt: AuthorizedAttempt,
    binding: FrozenAttemptBinding,
) -> None:
    scope = HistoryScope(**{key: getattr(binding, key) for key in HistoryScope.model_fields})
    selection = resolve_operator_attempt(attempt.batch_root, scope)
    if selection.authorized != attempt:
        raise GenerationAttemptError("EXACT_ATTEMPT_AUTHORIZATION_NOT_DISPATCHABLE")
    if attempt.authorization.binding() != binding:
        raise GenerationAttemptError("AUTHORIZATION_FROZEN_BINDING_MISMATCH")


def initialize_attempt(attempt: AuthorizedAttempt, binding: FrozenAttemptBinding) -> dict[str, Any]:
    """The sole new-directory writer; refuses partial history and never overwrites."""
    from scripts.prospective_acquisition.path_c_generation import _exclusive_write, _fsync_directory

    validate_authorized_attempt(attempt, binding)
    if attempt.location.exists():
        # A valid unconsumed initialized attempt may resume, but no partial or
        # consumed directory is repaired or reset by this path.
        state, _ = _read(attempt.location / "generation_state.json")
        return state
    try:
        attempt.location.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise GenerationAttemptError("ATTEMPT_CREATION_CONFLICT") from exc
    _fsync_directory(attempt.batch_root)
    created = datetime.now(UTC)
    identity_bytes = canonical_bytes(attempt.identity(created).model_dump(mode="json")) + b"\n"
    _exclusive_write(attempt.location / "attempt_identity.json", identity_bytes)
    auth = attempt.authorization
    state = {
        "state_version": "path-c-generation-state.v1",
        "state": "AUTHORIZED_NOT_DISPATCHED",
        "population_id": auth.population_id,
        "batch_id": auth.batch_id,
        "execution_context_id": str(auth.execution_context_id),
        "contract_version": auth.generation_contract_version,
        "attempt_number": auth.attempt_number,
        "attempt_id": auth.attempt_id,
        "authorization_id": str(auth.authorization_id),
        "authorization_hash": auth.authorization_hash,
        "authorization_file_sha256": attempt.authorization_file_sha256,
        "attempt_identity_sha256": digest(identity_bytes),
        "authorized_at_utc": created.isoformat(),
    }
    _exclusive_write(attempt.location / "generation_state.json", canonical_bytes(state) + b"\n")
    return state
