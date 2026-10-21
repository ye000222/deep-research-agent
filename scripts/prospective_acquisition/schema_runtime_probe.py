"""Offline runtime probe for the exact frozen PATH-C schema; no generation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import sysconfig
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

from scripts.prospective_acquisition.schema_validation import (
    load_schema,
    validate_instance,
    validator_for_schema,
)

SCHEMA_SHA256 = "f908b6397240f5216746edccd26825582223a3d0747b73c9e340c73a1b6d53d3"
REQUIRED_JSONSCHEMA = Requirement("jsonschema[format]>=4.23,<5")


def _has_runtime_jsonschema_requirement(requirements: list[str]) -> bool:
    """Match the packaged PEP 508 requirement, excluding inactive extras."""
    for raw_requirement in requirements:
        try:
            requirement = Requirement(raw_requirement)
        except InvalidRequirement:
            continue
        if requirement.name.casefold() != REQUIRED_JSONSCHEMA.name:
            continue
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        if (
            set(requirement.extras) == set(REQUIRED_JSONSCHEMA.extras)
            and requirement.specifier == REQUIRED_JSONSCHEMA.specifier
        ):
            return True
    return False


def _installed_distribution_requirements() -> tuple[list[str], int]:
    """Read installed package metadata, not a source-tree egg-info shadow."""
    purelib = sysconfig.get_paths().get("purelib")
    if not purelib:
        return [], 0
    target_name = canonicalize_name("deep-research-agent")
    matches = [
        distribution
        for distribution in importlib.metadata.distributions(path=[purelib])
        if canonicalize_name(distribution.metadata.get("Name", "")) == target_name
    ]
    if len(matches) != 1:
        return [], len(matches)
    return list(matches[0].requires or []), 1


def _safe_jsonschema_requirement_metadata(requirements: list[str]) -> list[dict[str, Any]]:
    """Expose only normalized PEP 508 structure for matching requirements."""
    parsed: list[dict[str, Any]] = []
    for raw_requirement in requirements:
        try:
            requirement = Requirement(raw_requirement)
        except InvalidRequirement:
            continue
        if canonicalize_name(requirement.name) != canonicalize_name("jsonschema"):
            continue
        marker_text = str(requirement.marker) if requirement.marker is not None else ""
        marker_category = (
            "NONE"
            if not marker_text
            else "EXTRA_CONDITIONAL"
            if "extra" in marker_text
            else "ENVIRONMENT_CONDITIONAL"
        )
        parsed.append(
            {
                "normalized_name": canonicalize_name(requirement.name),
                "extras": sorted(requirement.extras),
                "specifier": str(requirement.specifier),
                "marker_present": requirement.marker is not None,
                "marker_category": marker_category,
            }
        )
    return parsed


def synthetic_manifest() -> dict[str, Any]:
    return {
        "manifest_version": "prospective-generic-task-manifest.v1", "task_count": 40,
        "tasks": [
            {"task_id": f"task-{i:03d}", "task_text": f"Synthetic task {i}"}
            for i in range(1, 41)
        ],
    }


def probe_schema_runtime(schema_path: Path) -> dict[str, Any]:
    """Report fixed safe facts, fail closed; no writes, clients or Provider."""
    result: dict[str, Any] = {
        "status": "FAIL", "provider_generation_calls": 0, "provider_http_count": 0,
        "db_writes": 0, "runtime_dependency_declared": False, "validator_initialized": False,
        "valid_document_passed": False, "invalid_document_rejected": False,
    }
    try:
        requirements, distribution_count = _installed_distribution_requirements()
        result["runtime_distribution_match_count"] = distribution_count
        result["jsonschema_requirement_metadata"] = (
            _safe_jsonschema_requirement_metadata(requirements)
        )
        result["runtime_dependency_declared"] = _has_runtime_jsonschema_requirement(
            requirements
        )
        schema_bytes = schema_path.read_bytes()
        actual = hashlib.sha256(schema_bytes).hexdigest()
        result["schema_sha256"] = actual
        if actual != SCHEMA_SHA256:
            result["safe_error_code"] = "FROZEN_SCHEMA_HASH_MISMATCH"
            return result
        schema = load_schema(schema_path)
        validator = validator_for_schema(schema)
        result["validator_initialized"] = True
        result["jsonschema_version"] = importlib.metadata.version("jsonschema")
        result["installed_version_supported"] = SpecifierSet(">=4.23,<5").contains(
            result["jsonschema_version"]
        )
        result["validator_class"] = type(validator).__name__
        result["valid_document_passed"] = validate_instance(schema, synthetic_manifest()) == ()
        invalid = synthetic_manifest()
        invalid["tasks"] = invalid["tasks"][:-1]
        result["invalid_document_rejected"] = bool(validate_instance(schema, invalid))
        if all(result[k] is True for k in (
            "runtime_dependency_declared", "installed_version_supported",
            "validator_initialized",
            "valid_document_passed", "invalid_document_rejected",
        )):
            result["status"] = "PASS"
        else:
            result["safe_error_code"] = "SCHEMA_RUNTIME_NOT_READY"
    except Exception as exc:
        result["exception_type"] = type(exc).__name__
        result["cause_exception_type"] = type(exc.__cause__).__name__ if exc.__cause__ else None
        result["safe_error_code"] = "SCHEMA_RUNTIME_DEPENDENCY_OR_VALIDATION_FAILED"
    return result
