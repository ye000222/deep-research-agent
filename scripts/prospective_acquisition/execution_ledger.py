"""Append-only, hash-chained execution ledger for bounded acquisition tasks."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

TASK_STATES = frozenset(
    {
        "NOT_STARTED",
        "PREPARED",
        "EXTERNAL_EFFECT_STARTED",
        "SUCCEEDED",
        "FAILED_PRE_EFFECT",
        "FAILED_AFTER_EFFECT",
    }
)
_TRANSITIONS = {
    "NOT_STARTED": {"PREPARED"},
    "PREPARED": {"EXTERNAL_EFFECT_STARTED", "FAILED_PRE_EFFECT"},
    "FAILED_PRE_EFFECT": {"PREPARED"},
    "EXTERNAL_EFFECT_STARTED": {"SUCCEEDED", "FAILED_AFTER_EFFECT"},
    "SUCCEEDED": set(),
    "FAILED_AFTER_EFFECT": set(),
}


class ExecutionLedgerError(RuntimeError):
    """Raised when ledger state is inconsistent or a transition is unsafe."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl = cast(Any, __import__("fcntl"))

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class TaskExecutionLedger:
    """Serialize task transitions and immutable per-task terminal receipts."""

    def __init__(self, execution_dir: Path) -> None:
        self._root = execution_dir
        self._root.mkdir(parents=True, exist_ok=True)
        self._events = self._root / "execution_ledger.jsonl"
        self._lock = self._root / ".ledger.lock"

    def state(self, task_execution_id: str) -> str:
        with _exclusive_lock(self._lock):
            events = self._read_verified()
            selected = [
                event for event in events if event["task_execution_id"] == task_execution_id
            ]
            state = str(selected[-1]["to_state"]) if selected else "NOT_STARTED"
            receipt = self._receipt_path(task_execution_id)
            if state in {"SUCCEEDED", "FAILED_AFTER_EFFECT"}:
                if not receipt.is_file():
                    raise ExecutionLedgerError("terminal_receipt_missing")
                self._verify_receipt(receipt, task_execution_id, state)
            elif receipt.exists():
                raise ExecutionLedgerError("receipt_without_terminal_transition")
            return state

    def prepare(self, task_execution_id: str, *, task_id: str, task_ordinal: int) -> str:
        with _exclusive_lock(self._lock):
            events = self._read_verified()
            prior = [event for event in events if event["task_execution_id"] == task_execution_id]
            current = str(prior[-1]["to_state"]) if prior else "NOT_STARTED"
            if current in {"SUCCEEDED", "FAILED_AFTER_EFFECT"}:
                if not self._receipt_path(task_execution_id).is_file():
                    raise ExecutionLedgerError("terminal_receipt_missing")
                self._verify_receipt(
                    self._receipt_path(task_execution_id), task_execution_id, current
                )
                if current == "FAILED_AFTER_EFFECT":
                    raise ExecutionLedgerError("task_failed_after_effect")
                return current
            if current not in _TRANSITIONS or "PREPARED" not in _TRANSITIONS[current]:
                raise ExecutionLedgerError("task_not_resumable")
            self._append(events, task_execution_id, current, "PREPARED", {
                "task_id": task_id,
                "task_ordinal": task_ordinal,
            })
            return "PREPARED"

    def mark_external_effect_started(self, task_execution_id: str) -> None:
        self._transition(task_execution_id, "EXTERNAL_EFFECT_STARTED", {})

    def fail_pre_effect(
        self,
        task_execution_id: str,
        *,
        safe_failure_code: str,
        counters: dict[str, int],
    ) -> None:
        if any(value != 0 for value in counters.values()):
            raise ExecutionLedgerError("pre_effect_failure_has_nonzero_effect_counter")
        self._transition(task_execution_id, "FAILED_PRE_EFFECT", {
            "safe_failure_code": safe_failure_code,
            "counters": counters,
        })

    def finish(
        self,
        task_execution_id: str,
        *,
        terminal_state: str,
        receipt: dict[str, object],
    ) -> Path:
        if terminal_state not in {"SUCCEEDED", "FAILED_AFTER_EFFECT"}:
            raise ExecutionLedgerError("invalid_terminal_state")
        with _exclusive_lock(self._lock):
            events = self._read_verified()
            prior = [event for event in events if event["task_execution_id"] == task_execution_id]
            current = str(prior[-1]["to_state"]) if prior else "NOT_STARTED"
            if terminal_state not in _TRANSITIONS.get(current, set()):
                raise ExecutionLedgerError("invalid_task_state_transition")
            receipt_path = self._receipt_path(task_execution_id)
            document = {
                "receipt_version": "prospective-acquisition-task-receipt.v1",
                "task_execution_id": task_execution_id,
                "terminal_state": terminal_state,
                "created_at": datetime.now(UTC).isoformat(),
                **receipt,
            }
            document["receipt_sha256"] = hashlib.sha256(_canonical(document)).hexdigest()
            try:
                with receipt_path.open("xb") as handle:
                    handle.write(_canonical(document) + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except FileExistsError as exc:
                raise ExecutionLedgerError("terminal_receipt_already_exists") from exc
            self._append(events, task_execution_id, current, terminal_state, {
                "receipt_sha256": document["receipt_sha256"],
            })
            return receipt_path

    def _transition(self, task_execution_id: str, target: str, details: dict[str, object]) -> None:
        with _exclusive_lock(self._lock):
            events = self._read_verified()
            prior = [event for event in events if event["task_execution_id"] == task_execution_id]
            current = str(prior[-1]["to_state"]) if prior else "NOT_STARTED"
            if target not in _TRANSITIONS.get(current, set()):
                raise ExecutionLedgerError("invalid_task_state_transition")
            self._append(events, task_execution_id, current, target, details)

    def _read_verified(self) -> list[dict[str, object]]:
        if not self._events.exists():
            return []
        events: list[dict[str, object]] = []
        previous_hash = "0" * 64
        task_states: dict[str, str] = {}
        try:
            lines = self._events.read_bytes().splitlines()
            for index, raw in enumerate(lines, start=1):
                event = json.loads(raw)
                if not isinstance(event, dict):
                    raise ValueError
                digest = event.pop("event_sha256")
                if event.get("seq") != index or event.get("previous_sha256") != previous_hash:
                    raise ValueError
                if hashlib.sha256(_canonical(event)).hexdigest() != digest:
                    raise ValueError
                event["event_sha256"] = digest
                if (
                    event.get("from_state") not in TASK_STATES
                    or event.get("to_state") not in TASK_STATES
                ):
                    raise ValueError
                if event["to_state"] not in _TRANSITIONS.get(str(event["from_state"]), set()):
                    raise ValueError
                task_id = event.get("task_execution_id")
                if not isinstance(task_id, str) or not task_id:
                    raise ValueError
                if event["from_state"] != task_states.get(task_id, "NOT_STARTED"):
                    raise ValueError
                task_states[task_id] = str(event["to_state"])
                previous_hash = str(digest)
                events.append(event)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise ExecutionLedgerError("ledger_integrity_failure") from exc
        return events

    def _append(
        self,
        events: list[dict[str, object]],
        task_execution_id: str,
        from_state: str,
        to_state: str,
        details: dict[str, object],
    ) -> None:
        previous_hash = str(events[-1]["event_sha256"]) if events else "0" * 64
        event: dict[str, object] = {
            "seq": len(events) + 1,
            "task_execution_id": task_execution_id,
            "from_state": from_state,
            "to_state": to_state,
            "timestamp": datetime.now(UTC).isoformat(),
            "details": details,
            "previous_sha256": previous_hash,
        }
        event["event_sha256"] = hashlib.sha256(_canonical(event)).hexdigest()
        with self._events.open("ab") as handle:
            handle.write(_canonical(event) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        events.append(event)

    def _receipt_path(self, task_execution_id: str) -> Path:
        if not task_execution_id or any(
            char not in "0123456789abcdef" for char in task_execution_id
        ):
            raise ExecutionLedgerError("invalid_task_execution_id")
        return self._root / f"{task_execution_id}.receipt.json"

    @staticmethod
    def _verify_receipt(path: Path, task_execution_id: str, terminal_state: str) -> None:
        try:
            document = json.loads(path.read_bytes())
            claimed = document.pop("receipt_sha256")
            actual = hashlib.sha256(_canonical(document)).hexdigest()
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ExecutionLedgerError("terminal_receipt_integrity_failure") from exc
        if (
            actual != claimed
            or document.get("task_execution_id") != task_execution_id
            or document.get("terminal_state") != terminal_state
        ):
            raise ExecutionLedgerError("terminal_receipt_integrity_failure")
