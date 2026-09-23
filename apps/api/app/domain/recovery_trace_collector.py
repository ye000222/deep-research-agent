"""Collect and validate a Recovery-to-Closure event trace."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from app.domain.recovery_trace import RecoveryResearchTrace


class RecoveryTraceError(ValueError):
    """Raised when a Recovery event chain is incomplete or crosses identities."""


class RecoveryTraceCollector:
    """Build a trace from persisted event payloads or domain event objects.

    Events must carry ``recovery_attempt_id`` once they belong to a Recovery
    action. This strict requirement keeps ordinary Research events isolated.
    """

    @staticmethod
    def collect(
        events: Iterable[Mapping[str, object] | object],
        *,
        recovery_attempt_id: UUID | None = None,
    ) -> RecoveryResearchTrace:
        normalized = [_normalize_event(event) for event in events]
        if not normalized:
            raise RecoveryTraceError("Recovery trace requires at least one event")

        attempt_id = recovery_attempt_id or _find_attempt_id(normalized)
        if attempt_id is None:
            raise RecoveryTraceError("Recovery trace has no recovery_attempt_id")

        relevant = [
            event
            for event in normalized
            if _event_attempt_id(event) == attempt_id
        ]
        if len(relevant) != len(normalized):
            raise RecoveryTraceError(
                "Recovery trace contains an event from another attempt or "
                "an event without recovery_attempt_id"
            )

        started = _first(relevant, "recovery.started")
        terminal = _last_terminal(relevant)
        if started is None:
            raise RecoveryTraceError("Recovery trace is missing recovery.started")
        if terminal is None:
            raise RecoveryTraceError(
                "Recovery trace is missing recovery.completed or recovery.failed"
            )

        _check_identity(relevant, started)
        names = tuple(str(event["event"]) for event in relevant)
        _check_sequence(names, terminal["event"])

        run_id = _required_uuid(started.get("run_id"), "run_id")
        question_id = _required_string(started, "question_id")
        gap_id = _required_uuid(started.get("gap_id"), "gap_id")
        completed = terminal["event"] == "recovery.completed"

        query_ids = _ids_for(relevant, "query.execution.", "execution_id")
        reader_ids = _ids_for(relevant, "reader.execution.", "reader_execution_id")
        extraction_ids = _ids_for(relevant, "evidence.extraction.", "extraction_id")
        alignment_ids = _ids_for(relevant, "evidence.alignment.", "alignment_id")
        closure_ids = _ids_for(relevant, "gap.closure.", "evaluation_id")

        if completed:
            _require_success_chain(
                names,
                query_ids=query_ids,
                reader_ids=reader_ids,
                extraction_ids=extraction_ids,
                alignment_ids=alignment_ids,
                closure_ids=closure_ids,
            )

        before_gap = _gap_state(started.get("gap_before"))
        after_gap = _gap_state(terminal.get("gap_after"), default=before_gap)
        before_coverage = _number(started.get("coverage_before"), "coverage_before")
        after_coverage = _number(
            terminal.get("coverage_after"),
            "coverage_after",
            default=before_coverage,
        )
        outcome = str(terminal.get("outcome_type") or terminal["event"])
        closure_transition_reason = _closure_transition_reason(relevant)

        return RecoveryResearchTrace(
            trace_id=uuid5(NAMESPACE_URL, f"recovery-trace:{attempt_id}"),
            run_id=run_id,
            question_id=question_id,
            recovery_attempt_id=attempt_id,
            gap_id=gap_id,
            query_execution_ids=query_ids,
            reader_execution_ids=reader_ids,
            extraction_ids=extraction_ids,
            alignment_ids=alignment_ids,
            closure_evaluation_ids=closure_ids,
            before_gap_state=before_gap,
            after_gap_state=after_gap,
            before_coverage=before_coverage,
            after_coverage=after_coverage,
            outcome_type=outcome,
            closure_transition_reason=closure_transition_reason,
            event_sequence=names,
        )


def _normalize_event(event: Mapping[str, object] | object) -> dict[str, object]:
    if isinstance(event, Mapping):
        value = dict(event)
    elif is_dataclass(event):
        value = asdict(cast(Any, event))
    else:
        value = {
            name: getattr(event, name)
            for name in dir(event)
            if not name.startswith("_") and not callable(getattr(event, name))
        }
    if "event" not in value and "event_type" in value:
        value["event"] = _enum_value(value["event_type"])
    if "refs" in value and isinstance(value["refs"], Mapping):
        refs = dict(value["refs"])
        refs.update(value)
        value = refs
    return value


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _event_attempt_id(event: Mapping[str, object]) -> UUID | None:
    value = event.get("recovery_attempt_id", event.get("attempt_id"))
    return _uuid(value, "recovery_attempt_id", required=False)


def _find_attempt_id(events: list[dict[str, object]]) -> UUID | None:
    for event in events:
        attempt_id = _event_attempt_id(event)
        if attempt_id is not None:
            return attempt_id
    return None


def _first(events: list[dict[str, object]], name: str) -> dict[str, object] | None:
    return next((event for event in events if event.get("event") == name), None)


def _last_terminal(events: list[dict[str, object]]) -> dict[str, object] | None:
    return next(
        (
            event
            for event in reversed(events)
            if event.get("event") in {"recovery.completed", "recovery.failed"}
        ),
        None,
    )


def _check_identity(
    events: list[dict[str, object]],
    started: Mapping[str, object],
) -> None:
    for event in events:
        for key in ("run_id", "question_id", "gap_id"):
            if key in event and _identity_value(event[key]) != _identity_value(
                started.get(key)
            ):
                raise RecoveryTraceError(f"Recovery event {key} does not match")


def _check_sequence(names: tuple[str, ...], terminal_name: object) -> None:
    if names.index("recovery.started") > names.index(str(terminal_name)):
        raise RecoveryTraceError("Recovery started occurs after its terminal event")


def _require_success_chain(
    names: tuple[str, ...],
    *,
    query_ids: tuple[UUID, ...],
    reader_ids: tuple[UUID, ...],
    extraction_ids: tuple[UUID, ...],
    alignment_ids: tuple[UUID, ...],
    closure_ids: tuple[UUID, ...],
) -> None:
    required = (
        ("query.execution.completed", query_ids),
        ("reader.execution.completed", reader_ids),
        ("evidence.extraction.completed", extraction_ids),
        ("evidence.alignment.completed", alignment_ids),
        ("gap.closure.transition", closure_ids),
    )
    for event_name, ids in required:
        if event_name not in names or not ids:
            raise RecoveryTraceError(f"Successful Recovery trace missing {event_name}")
    positions = [
        names.index(event_name)
        for event_name, _ in required
    ]
    if positions != sorted(positions):
        raise RecoveryTraceError("Recovery research events are out of order")


def _ids_for(
    events: list[dict[str, object]],
    prefix: str,
    key: str,
) -> tuple[UUID, ...]:
    ids: list[UUID] = []
    for event in events:
        name = str(event.get("event", ""))
        if not name.startswith(prefix) or key not in event:
            continue
        value = _required_uuid(event[key], key)
        if value not in ids:
            ids.append(value)
    return tuple(ids)


def _closure_transition_reason(events: list[dict[str, object]]) -> str | None:
    for event in reversed(events):
        if event.get("event") == "gap.closure.transition":
            value = event.get("transition_reason", event.get("reason"))
            return str(value) if value is not None else None
    return None


def _gap_state(value: object, *, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set)):
        return tuple(str(item) for item in value)
    raise RecoveryTraceError("gap state must be a string or sequence")


def _required_string(event: Mapping[str, object], key: str) -> str:
    value = event.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RecoveryTraceError(f"Recovery event is missing {key}")
    return value


def _number(value: object, key: str, *, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    raise RecoveryTraceError(f"Recovery event is missing {key}")


def _uuid(value: object, key: str, *, required: bool = True) -> UUID | None:
    if isinstance(value, UUID):
        return value
    if isinstance(value, str):
        try:
            return UUID(value)
        except ValueError as exc:
            raise RecoveryTraceError(f"invalid {key}") from exc
    if required:
        raise RecoveryTraceError(f"Recovery event is missing {key}")
    return None


def _required_uuid(value: object, key: str) -> UUID:
    parsed = _uuid(value, key)
    assert parsed is not None
    return parsed


def _identity_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, UUID):
        return str(value)
    return str(value)
