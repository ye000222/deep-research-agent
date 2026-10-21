"""Never load real Attempt003 response; synthetic redaction assertions only."""
import json

import pytest

from scripts.prospective_acquisition.path_c_attempt_003_structure import structural_facts


def test_no_content_or_secret_fields_emitted() -> None:
    marker = "secret-or-prospective-task-content-never-print"
    raw = json.dumps({
        "id": marker, "model": marker, marker: marker,
        "choices": [{"message": {"content": json.dumps({"tasks": [marker] * 40})}}],
    }).encode()
    result = structural_facts(raw)
    assert marker not in json.dumps(result)
    assert result["valid_json"] and result["openai_compatible_envelope_valid"]
    assert result["task_array_count"] == 40
    assert result["required_schema_keys_present"] == ["tasks"]


@pytest.mark.parametrize("payload", [b"not-json", b"[]", b"{}", b'{"choices":[{}]}'])
def test_invalid_shapes_are_facts_not_repaired(payload: bytes) -> None:
    before = payload
    result = structural_facts(payload)
    assert payload == before
    assert result.get("openai_compatible_envelope_valid", False) is False
