"""Runtime packaging and schema fail-closed proof; no Docker or Provider."""

from __future__ import annotations

import asyncio
import builtins
import copy
import importlib.metadata
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from scripts.prospective_acquisition.path_c_generation import PathCGenerationBlocked
from scripts.prospective_acquisition.schema_runtime_probe import (
    _has_runtime_jsonschema_requirement,
    _installed_distribution_requirements,
    probe_schema_runtime,
    synthetic_manifest,
)
from scripts.prospective_acquisition.schema_validation import (
    SchemaDependencyUnavailable,
    load_schema,
    validate_instance,
    validator_for_schema,
)
from tests.unit.test_path_c_generation import FakeGateway, _copy_frozen_root, _execute, _tasks

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "artifacts/v2_1r_d4r_a51/task_manifest_schema.json"


def test_declared_runtime_dependency_not_dev_only() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert "jsonschema[format]>=4.23,<5" in project["dependencies"]
    assert not any(
        r.startswith("jsonschema") for r in project["optional-dependencies"]["dev"]
    )
    dockerfile = (ROOT / "infra/docker/api.Dockerfile").read_text(encoding="utf-8")
    assert "pip install ." in dockerfile
    assert "COPY pyproject.toml" in dockerfile


def test_frozen_forty_task_document_and_invalid_document_no_http() -> None:
    schema = load_schema(SCHEMA)
    with patch("httpx.AsyncClient.send", side_effect=AssertionError("outbound forbidden")) as send:
        validator = validator_for_schema(schema)
        assert type(validator).__name__ == "Draft202012Validator"
        assert validate_instance(schema, synthetic_manifest()) == ()
        invalid = synthetic_manifest()
        invalid["tasks"] = invalid["tasks"][:-1]
        assert validate_instance(schema, invalid)
    send.assert_not_called()


@pytest.mark.parametrize("change", ["type", "missing", "nested", "extra", "const"])
def test_invalid_manifests_still_rejected(change: str) -> None:
    instance = copy.deepcopy(synthetic_manifest())
    if change == "type":
        instance["task_count"] = "40"
    elif change == "missing":
        del instance["manifest_version"]
    elif change == "nested":
        instance["tasks"][0]["task_text"] = ""
    elif change == "extra":
        instance["unexpected"] = True
    else:
        instance["manifest_version"] = "not-the-frozen-version"
    assert validate_instance(load_schema(SCHEMA), instance)


def absent_jsonschema(name: str, *args: Any, **kwargs: Any) -> Any:
    if name == "jsonschema" or name.startswith("jsonschema."):
        raise ModuleNotFoundError("synthetic dependency absence", name="jsonschema")
    return ORIGINAL_IMPORT(name, *args, **kwargs)


ORIGINAL_IMPORT = builtins.__import__


def test_dependency_absence_no_validation_bypass() -> None:
    schema = load_schema(SCHEMA)
    with patch("builtins.__import__", side_effect=absent_jsonschema):
        for operation in (
            lambda: validator_for_schema(schema),
            lambda: validate_instance(schema, synthetic_manifest()),
        ):
            with pytest.raises(SchemaDependencyUnavailable) as error:
                operation()
            assert isinstance(error.value.__cause__, ModuleNotFoundError)
            assert error.value.__cause__.name == "jsonschema"


def test_output_validation_missing_dependency_remains_consumed_failed(tmp_path: Path) -> None:
    checkout = _copy_frozen_root(tmp_path)
    output = tmp_path / "synthetic-attempt"
    gateway = FakeGateway(output / "provider_response.raw.json", parsed={"tasks": _tasks()})
    with (
        patch("builtins.__import__", side_effect=absent_jsonschema),
        pytest.raises(PathCGenerationBlocked) as blocked,
    ):
        asyncio.run(_execute(checkout, output, gateway))
    assert blocked.value.code == "SCHEMADEPENDENCYUNAVAILABLE"
    assert gateway.calls == 1  # fake only, no HTTP/real Provider
    assert (output / "provider_response.raw.json").exists()
    assert not (output / "path_c_task_manifest.json").exists()
    state = json.loads((output / "generation_state.json").read_bytes())
    assert state["state"] == "CONSUMED_FAILED"
    assert state["failure_diagnostics"]["failure_phase"] == "OUTPUT_VALIDATION"
    assert state["failure_diagnostics"]["failure_class"] == "LOCAL_RUNTIME_ERROR"


@pytest.mark.parametrize("runtime", [True, False])
def test_same_probe_requires_runtime_install_metadata(runtime: bool) -> None:
    requirement = "jsonschema[format]<5,>=4.23" + ("" if runtime else '; extra == "dev"')
    with (
        patch("sysconfig.get_paths", return_value={"purelib": "/site-packages"}),
        patch(
            "importlib.metadata.distributions",
            return_value=[
                SimpleNamespace(
                    metadata={"Name": "deep-research-agent"},
                    requires=[requirement],
                )
            ],
        ),
        patch("httpx.AsyncClient.send", side_effect=AssertionError("outbound forbidden")) as send,
    ):
        report = probe_schema_runtime(SCHEMA)
    assert report["validator_initialized"] is True
    assert report["installed_version_supported"] is True
    assert report["valid_document_passed"] is True
    assert report["invalid_document_rejected"] is True
    assert report["status"] == ("PASS" if runtime else "FAIL")
    assert report["provider_generation_calls"] == report["provider_http_count"] == 0
    send.assert_not_called()


def test_probe_ignores_stale_source_tree_egg_info_metadata() -> None:
    installed = SimpleNamespace(
        metadata={"Name": "deep-research-agent"},
        requires=["jsonschema[format]<5,>=4.23"],
    )
    source_shadow = SimpleNamespace(
        metadata={"Name": "deep-research-agent"},
        requires=['jsonschema[format]<5,>=4.23; extra == "dev"'],
    )
    with (
        patch("sysconfig.get_paths", return_value={"purelib": "/site-packages"}),
        patch("importlib.metadata.distributions", return_value=[installed]) as distributions,
        patch("importlib.metadata.requires", return_value=source_shadow.requires),
    ):
        report = probe_schema_runtime(SCHEMA)
    distributions.assert_called_once_with(path=["/site-packages"])
    assert report["runtime_distribution_match_count"] == 1
    assert report["runtime_dependency_declared"] is True
    assert report["jsonschema_requirement_metadata"] == [
        {
            "normalized_name": "jsonschema",
            "extras": ["format"],
            "specifier": "<5,>=4.23",
            "marker_present": False,
            "marker_category": "NONE",
        }
    ]


def test_probe_fails_closed_when_installed_distribution_metadata_is_ambiguous() -> None:
    distribution = SimpleNamespace(
        metadata={"Name": "deep-research-agent"},
        requires=["jsonschema[format]<5,>=4.23"],
    )
    with (
        patch("sysconfig.get_paths", return_value={"purelib": "/site-packages"}),
        patch("importlib.metadata.distributions", return_value=[distribution, distribution]),
    ):
        requirements, count = _installed_distribution_requirements()
    assert requirements == []
    assert count == 2


def test_requirement_parser_accepts_normalized_order_and_rejects_dev_extra() -> None:
    assert _has_runtime_jsonschema_requirement(["jsonschema[format]<5,>=4.23"])
    assert not _has_runtime_jsonschema_requirement(
        ['jsonschema[format]<5,>=4.23; extra == "dev"']
    )
    assert not _has_runtime_jsonschema_requirement(["jsonschema<5,>=4.23"])


def test_same_probe_missing_dependency_fail_closed_safe() -> None:
    with (
        patch("importlib.metadata.requires", return_value=["jsonschema[format]>=4.23,<5"]),
        patch("builtins.__import__", side_effect=absent_jsonschema),
    ):
        report = probe_schema_runtime(SCHEMA)
    assert report["status"] == "FAIL"
    assert report["exception_type"] == "SchemaDependencyUnavailable"
    assert report["cause_exception_type"] == "ModuleNotFoundError"
    assert not report["validator_initialized"]
    assert not report["valid_document_passed"]
    assert "synthetic dependency absence" not in json.dumps(report)


def test_probe_rejects_unsupported_installed_jsonschema_version() -> None:
    original_version = importlib.metadata.version

    def version_for(name: str) -> str:
        if name == "jsonschema":
            return "5.0.0"
        return original_version(name)

    with (
        patch("sysconfig.get_paths", return_value={"purelib": "/site-packages"}),
        patch(
            "importlib.metadata.distributions",
            return_value=[
                SimpleNamespace(
                    metadata={"Name": "deep-research-agent"},
                    requires=["jsonschema[format]<5,>=4.23"],
                )
            ],
        ),
        patch("importlib.metadata.version", side_effect=version_for),
    ):
        report = probe_schema_runtime(SCHEMA)
    assert report["runtime_dependency_declared"] is True
    assert report["installed_version_supported"] is False
    assert report["status"] == "FAIL"


def test_attestation_reuses_same_schema_probe_before_provider_resolution() -> None:
    script = (ROOT / "scripts/operator_verify_canonical_runtime.ps1").read_text(encoding="utf-8")
    probe = (ROOT / "scripts/prospective_acquisition/canonical_runtime_probe.py").read_text(
        encoding="utf-8"
    )
    assert (
        probe.index('_stage = "SCHEMA_RUNTIME_PROBE"')
        < probe.index("async def _resolve_provider")
    )
    assert (
        'probe_schema_runtime(pathlib.Path("/opt/artifacts/v2_1r_d4r_a51/task_manifest_schema.json"))'
        in probe
    )
    assert "schema_runtime = (Get-SafeProperty $probeResult 'schema_runtime').value" in script
    assert "$schemaRuntime.state -ne 'PRESENT'" in script
    assert "pip install" not in script


def test_generation_output_validation_roots_match_canonical_compose_mounts() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    entry = (ROOT / "scripts/prospective_acquisition/path_c_generation_entry.py").read_text(
        encoding="utf-8"
    )
    generation = (ROOT / "scripts/prospective_acquisition/path_c_generation.py").read_text(
        encoding="utf-8"
    )
    assert "./apps/api:/app/apps/api:ro" in compose
    assert 'application_root=Path("/app")' in entry
    assert 'root=Path("/opt")' in entry
    assert 'source_root = app_root if relative.startswith("apps/api/") else root' in generation
    assert "GENERATION_IMPLEMENTATION_FILES" in generation
