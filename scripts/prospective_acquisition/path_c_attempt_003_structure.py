"""Content-blind, read-only structural facts. Never emit generated task text."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_SHA256 = "b17e802bad7f4c5fef9aff48b0e5fd20e5cc6b84b3fbc9f2f4a84c534fe787ef"
EXPECTED_SIZE = 25081
POPULATION = "v2_1r_d4r_prospective_population_001"
BATCH = "v2_1r_d4r_prospective_population_001_batch_001"


def structural_facts(raw: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {"sha256": hashlib.sha256(raw).hexdigest(), "byte_size": len(raw)}
    try:
        payload = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        return {**result, "valid_json": False, "exception_type": type(exc).__name__}
    result.update(valid_json=True, top_level_type=type(payload).__name__)
    if not isinstance(payload, dict):
        result["openai_compatible_envelope_valid"] = False
        return result
    # Never print arbitrary model-generated keys; expose only envelope key names.
    known = {"id", "object", "created", "model", "choices", "usage", "error", "system_fingerprint"}
    result["top_level_key_names"] = sorted(k for k in payload if k in known)
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, dict) else None
    present = isinstance(message, dict) and "content" in message
    content = message.get("content") if isinstance(message, dict) else None
    result["completion_content_field_present"] = present
    result["openai_compatible_envelope_valid"] = isinstance(content, str)
    result["completion_content_non_empty"] = isinstance(content, str) and bool(content.strip())
    if not isinstance(content, str):
        return result
    try:
        parsed = json.loads(content)
    except (ValueError, RecursionError) as exc:
        return {**result, "completion_content_parses_as_json": False,
                "exception_type": type(exc).__name__}
    result["completion_content_parses_as_json"] = True
    result["parsed_content_top_level_type"] = type(parsed).__name__
    tasks = parsed.get("tasks") if isinstance(parsed, dict) else None
    result["task_array_count"] = len(tasks) if isinstance(tasks, list) else None
    result["required_schema_keys_present"] = (
        ["tasks"] if isinstance(parsed, dict) and "tasks" in parsed else []
    )
    result["required_schema_keys_missing"] = (
        [] if result["required_schema_keys_present"] else ["tasks"]
    )
    return result


def main() -> int:
    root = Path("/forensic/v2_1r_d4r_path_c_generation") / f"{POPULATION}__{BATCH}"
    path = root / "attempt_003/provider_response.raw.json"
    try:
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("untrusted path")
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != EXPECTED_SHA256 or len(raw) != EXPECTED_SIZE:
            print(json.dumps({"sha256": actual, "byte_size": len(raw), "identity_match": False}))
            return 1
        print(json.dumps({"identity_match": True, **structural_facts(raw)}))
        return 0
    except Exception as exc:
        print(json.dumps({"exception_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
