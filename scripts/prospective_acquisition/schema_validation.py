"""Standards-based validation for frozen prospective-acquisition JSON Schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SchemaDependencyUnavailable(RuntimeError):
    """Raised when the declared JSON Schema validator is absent."""


def load_schema(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON Schema document must be an object")
    if not isinstance(value.get("$schema"), str):
        raise ValueError("JSON Schema dialect is missing; refusing to guess")
    return value


def validator_for_schema(schema: dict[str, Any]) -> Any:
    """Meta-validate and return the validator selected by the declared dialect."""
    try:
        from jsonschema import FormatChecker, validators  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised in dependency-minimal envs
        raise SchemaDependencyUnavailable(
            "required runtime dependency jsonschema[format] is unavailable"
        ) from exc

    validator_type = validators.validator_for(schema, default=None)
    if validator_type is None:
        raise ValueError("unsupported or missing JSON Schema dialect")
    validator_type.check_schema(schema)
    return validator_type(schema, format_checker=FormatChecker())


def validate_instance(schema: dict[str, Any], instance: object) -> tuple[str, ...]:
    validator = validator_for_schema(schema)
    return tuple(error.message for error in validator.iter_errors(instance))
