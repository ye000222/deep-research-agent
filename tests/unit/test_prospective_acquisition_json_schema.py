from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

from scripts.prospective_acquisition.schema_validation import (  # noqa: E402
    load_schema,
    validate_instance,
    validator_for_schema,
)

ROOT = Path(__file__).parents[2]
TASK_SCHEMA = ROOT / "artifacts/v2_1r_d4r_a5/task_manifest_schema.json"
POPULATION_SCHEMA = ROOT / "artifacts/v2_1r_d4r_a4/population_manifest_schema.json"


def _task_manifest() -> dict[str, object]:
    return {
        "schema_version": "prospective-acquisition-task-manifest.v1",
        "protocol_version": "prospective-development-population.v1",
        "population_id": "population",
        "acquisition_batch_id": "batch",
        "frozen_at": "2026-10-12T00:00:00Z",
        "task_source_id": "source",
        "selection_rule_hash": "a" * 64,
        "tasks": [
            {
                "task_id": "task-1",
                "task_text": "A generic task",
                "source_corpus_ref": "corpus:1",
                "selection_rank": 0,
                "task_hash": "b" * 64,
            }
        ],
        "task_count": 1,
        "manifest_hash": "c" * 64,
    }


def _population_manifest() -> dict[str, object]:
    members = [
        {
            "claim_id": f"00000000-0000-4000-8000-{index:012d}",
            "population_id": "population",
            "acquisition_batch_id": "batch",
            "run_id": "00000000-0000-4000-8000-000000000001",
            "created_at": "2026-10-12T00:00:00Z",
            "development_only": True,
            "source_provenance_refs": [f"source:{index}"],
            "source_diversity_key": f"{index:064x}",
            "source_owner_diversity_key": f"{index:064x}",
            "content_hash": "a" * 64,
            "protocol_version": "prospective-development-population.v1",
            "exposure_state": "UNINSPECTED",
            "freshness_attestation": "CREATED_AFTER_PROTOCOL_FREEZE",
            "holdout_isolation_attestation": "PROSPECTIVE_NAMESPACE_NO_IMPORT_OR_COPY",
            "record_hash": "b" * 64,
        }
        for index in range(40)
    ]
    return {
        "schema_version": "prospective-development-population-manifest.v1",
        "protocol_version": "prospective-development-population.v1",
        "population_id": "population",
        "acquisition_batch_id": "batch",
        "frozen_at": "2026-10-12T00:00:00Z",
        "development_only": True,
        "members": members,
        "member_count": 40,
        "unique_source_diversity_key_count": 8,
        "unique_known_owner_key_count": 4,
        "population_hash": "c" * 64,
    }


@pytest.mark.parametrize(
    ("schema_path", "instance_factory"),
    [
        (TASK_SCHEMA, _task_manifest),
        (POPULATION_SCHEMA, _population_manifest),
    ],
)
def test_frozen_schemas_meta_validate_and_accept_representative_instance(
    schema_path: Path, instance_factory: Callable[[], dict[str, object]]
) -> None:
    schema = load_schema(schema_path)
    validator = validator_for_schema(schema)
    assert validator.is_valid(instance_factory())


def test_task_schema_rejects_required_type_nested_const_and_extra_field() -> None:
    schema = load_schema(TASK_SCHEMA)
    valid = _task_manifest()
    assert validate_instance(schema, valid) == ()

    missing = copy.deepcopy(valid)
    del missing["task_source_id"]
    assert validate_instance(schema, missing)

    wrong_type = copy.deepcopy(valid)
    wrong_type["task_count"] = "1"
    assert validate_instance(schema, wrong_type)

    bad_nested_const = copy.deepcopy(valid)
    bad_nested_const["tasks"][0]["selection_rank"] = "zero"  # type: ignore[index]
    assert validate_instance(schema, bad_nested_const)

    extra = copy.deepcopy(valid)
    extra["unexpected"] = True
    assert validate_instance(schema, extra)


def test_malformed_schema_is_rejected_by_declared_draft_meta_schema() -> None:
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "not-a-type"}
    with pytest.raises(jsonschema.exceptions.SchemaError):
        validator_for_schema(schema)


def test_declared_dialect_selects_2020_12_and_checks_enum_and_const() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["mode", "enabled"],
        "properties": {
            "mode": {"type": "string", "enum": ["development", "validation"]},
            "enabled": {"const": True},
        },
        "additionalProperties": False,
    }

    validator = validator_for_schema(schema)
    assert validator.META_SCHEMA["$id"].endswith("/2020-12/schema")
    assert validator.is_valid({"mode": "development", "enabled": True})
    assert not validator.is_valid({"mode": "production", "enabled": True})
    assert not validator.is_valid({"mode": "validation", "enabled": False})


def test_population_schema_rejects_missing_wrong_primitive_enum_const_and_extra_fields() -> None:
    schema = load_schema(POPULATION_SCHEMA)
    valid = _population_manifest()
    assert validate_instance(schema, valid) == ()

    missing = copy.deepcopy(valid)
    del missing["development_only"]
    assert validate_instance(schema, missing)

    wrong_primitive = copy.deepcopy(valid)
    wrong_primitive["member_count"] = "40"
    assert validate_instance(schema, wrong_primitive)

    wrong_const = copy.deepcopy(valid)
    wrong_const["development_only"] = False
    assert validate_instance(schema, wrong_const)

    nested_wrong_const = copy.deepcopy(valid)
    nested_wrong_const["members"][0]["exposure_state"] = "REVIEWED"  # type: ignore[index]
    assert validate_instance(schema, nested_wrong_const)

    extra = copy.deepcopy(valid)
    extra["unexpected"] = True
    assert validate_instance(schema, extra)
