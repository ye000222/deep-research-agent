"""Typed, analysis-only report contracts for the D3 replay harness."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict, cast
from uuid import UUID


class CardinalityInstance(TypedDict):
    claim_id: str
    node_ref: str
    obligation_ref: str
    expected_key: str | None
    matched_binding_refs: list[str]


class NodeReplayReport(TypedDict):
    artifact_version: str
    rows: list[dict[str, Any]]


class ObligationReplayReport(TypedDict):
    artifact_version: str
    rows: list[dict[str, Any]]


class BindingMatchReport(TypedDict):
    artifact_version: str
    rows: list[dict[str, Any]]


class OmissionComparisonReport(TypedDict):
    artifact_version: str
    rows: list[dict[str, Any]]
    classification_counts: dict[str, int]


class RequirednessStabilityReport(TypedDict):
    artifact_version: str
    nodes_tested: int
    all_stable: bool
    obligation_set_fingerprint: str
    requiredness_stability_violation_count: int
    rows: list[dict[str, Any]]


class NodeUsabilitySummary(TypedDict):
    artifact_version: str
    node_count: int
    distribution: dict[str, int]
    missing_required_binding_count: int


class VerificationAssertionReadinessSummary(TypedDict):
    artifact_version: str
    ready_nodes: int
    blocked_nodes_by_reason: dict[str, int]
    verification_assertion_view_constructed: bool


class ContractCoverageReport(TypedDict):
    artifact_version: str
    unsupported_node_shape_count: int
    partially_supported_node_count: int
    source_node_shape_counts: dict[str, int]


class ProvenanceAuditReport(TypedDict):
    artifact_version: str
    nodes_total: int
    nodes_with_complete_trace: int
    obligations_total: int
    obligations_with_complete_trace: int
    provenance_defect_count: int
    provenance_defects: list[dict[str, Any]]


@dataclass(frozen=True)
class CardinalityPolicySummary:
    policy: str
    group_count: int
    obligation_count: int
    expected_item_count: int
    matched_item_count: int
    satisfied_count: int
    partial_count: int
    failed_count: int
    unresolved_count: int
    compatibility_status: str
    groups: tuple[dict[str, Any], ...]
    expected_items: tuple[CardinalityInstance, ...]
    matched_items: tuple[dict[str, str], ...]
    satisfied_obligation_refs: tuple[dict[str, str], ...]
    partial_obligation_refs: tuple[dict[str, str], ...]
    failed_obligation_refs: tuple[dict[str, str], ...]
    unresolved_obligation_refs: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], jsonable(dataclasses.asdict(self)))


@dataclass(frozen=True)
class CardinalityReport:
    artifact_version: str
    policies: dict[str, CardinalityPolicySummary]
    compatibility_status: str
    unresolved_rows: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            {
                "artifact_version": self.artifact_version,
                "compatibility_status": self.compatibility_status,
                "unresolved_rows": jsonable(self.unresolved_rows),
                "policies": {key: value.to_dict() for key, value in self.policies.items()},
            },
        )


@dataclass(frozen=True)
class RowLedgerReport:
    artifact_version: str
    rows: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], jsonable(dataclasses.asdict(self)))


@dataclass(frozen=True)
class ReportContractCheck:
    status: str
    errors: tuple[str, ...]


CARDINALITY_POLICIES = (
    "EXACTLY_ONE",
    "AT_LEAST_ONE",
    "ALL_EXPECTED_MEMBERS",
    "PER_TARGET",
    "PER_VALUE",
)
UNRESOLVED_STATES = {
    "MISSING_REQUIRED_BINDING",
    "INSUFFICIENT_EVIDENCE",
    "DERIVATION_UNRESOLVED",
}


def jsonable(value: Any) -> Any:
    """Convert supported analysis values to deterministic JSON primitives.

    Decimal is encoded as a string to preserve exact value; sets are sorted by
    canonical JSON; tuples become arrays; UUID and date/time values use their
    canonical string forms. Unsupported objects fail explicitly.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return jsonable(value.value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return jsonable(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(child) for child in value]
    if isinstance(value, (set, frozenset)):
        normalized = [jsonable(child) for child in value]
        return sorted(normalized, key=lambda child: canonical_json(child))
    raise TypeError(f"unsupported_analysis_json_type:{type(value).__qualname__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def cardinality_report(obligations: Sequence[Mapping[str, Any]]) -> CardinalityReport:
    by_policy: dict[str, list[Mapping[str, Any]]] = {p: [] for p in CARDINALITY_POLICIES}
    unresolved_rows: list[dict[str, str]] = []
    for index, row in enumerate(obligations):
        policy = row.get("cardinality_policy")
        if not isinstance(policy, str) or policy not in by_policy:
            unresolved_rows.append(
                {
                    "row_index": str(index),
                    "field": "cardinality_policy",
                    "status": "TYPE_OR_VALUE_UNRESOLVED",
                }
            )
            continue
        by_policy[policy].append(row)

    summaries: dict[str, CardinalityPolicySummary] = {}
    for policy, rows in by_policy.items():
        groups: dict[tuple[str, str], list[str]] = {}
        expected_items: list[CardinalityInstance] = []
        matched_items: list[dict[str, str]] = []
        satisfied_refs: list[dict[str, str]] = []
        partial_refs: list[dict[str, str]] = []
        failed_refs: list[dict[str, str]] = []
        unresolved_refs: list[dict[str, str]] = []
        row_errors = 0
        matched_item_count = 0
        for index, row in enumerate(rows):
            claim_id = row.get("claim_id")
            node_ref = row.get("node_ref")
            obligation_ref = row.get("obligation_ref")
            state = row.get("final_obligation_state")
            matched = row.get("matched_bindings")
            cardinality_satisfied = row.get("cardinality_satisfied")
            expected_key_field = {
                "ALL_EXPECTED_MEMBERS": "expected_member_key",
                "PER_TARGET": "expected_target_key",
                "PER_VALUE": "expected_value_key",
            }.get(policy)
            expected_key = row.get(expected_key_field) if expected_key_field else None
            identity_valid = all(
                isinstance(item, str) and item for item in (claim_id, node_ref, obligation_ref)
            )
            refs_valid = isinstance(matched, list) and all(
                isinstance(item, str) and item for item in matched
            )
            state_valid = isinstance(state, str)
            satisfied_valid = isinstance(cardinality_satisfied, bool)
            expected_valid = expected_key_field is None or (
                isinstance(expected_key, str) and bool(expected_key)
            )
            if not (
                identity_valid and refs_valid and state_valid and satisfied_valid and expected_valid
            ):
                row_errors += 1
                unresolved_refs.append(
                    {
                        "claim_id": claim_id if isinstance(claim_id, str) else "",
                        "node_ref": node_ref if isinstance(node_ref, str) else "",
                        "obligation_ref": obligation_ref
                        if isinstance(obligation_ref, str)
                        else f"row:{index}",
                    }
                )
                continue

            claim_id_value = cast(str, claim_id)
            node_ref_value = cast(str, node_ref)
            obligation_ref_value = cast(str, obligation_ref)
            state_value = cast(str, state)
            matched_refs = cast(list[str], matched)
            cardinality_value = cast(bool, cardinality_satisfied)
            expected_key_value = cast(str | None, expected_key)
            group_key = (claim_id_value, node_ref_value)
            groups.setdefault(group_key, []).append(obligation_ref_value)
            expected_items.append(
                {
                    "claim_id": claim_id_value,
                    "node_ref": node_ref_value,
                    "obligation_ref": obligation_ref_value,
                    "expected_key": expected_key_value,
                    "matched_binding_refs": list(matched_refs),
                }
            )
            matched_item_count += len(matched_refs)
            matched_items.extend(
                {
                    "claim_id": claim_id_value,
                    "obligation_ref": obligation_ref_value,
                    "binding_ref": ref,
                }
                for ref in matched_refs
            )
            identity = {
                "claim_id": claim_id_value,
                "node_ref": node_ref_value,
                "obligation_ref": obligation_ref_value,
            }
            if state_value == "SATISFIED" and cardinality_value:
                satisfied_refs.append(identity)
            elif state_value != "SATISFIED" and matched_refs:
                partial_refs.append(identity)
            if not cardinality_value and state_value not in {
                "MISSING_REQUIRED_BINDING",
                "INSUFFICIENT_EVIDENCE",
            }:
                failed_refs.append(identity)
            if state_value in UNRESOLVED_STATES:
                unresolved_refs.append(identity)

        group_rows = tuple(
            {"claim_id": claim_id, "node_ref": node_ref, "obligation_refs": sorted(refs)}
            for (claim_id, node_ref), refs in sorted(groups.items())
        )
        summaries[policy] = CardinalityPolicySummary(
            policy=policy,
            group_count=len(groups),
            obligation_count=len(rows),
            expected_item_count=len(expected_items),
            matched_item_count=matched_item_count,
            satisfied_count=len(satisfied_refs),
            partial_count=len(partial_refs),
            failed_count=len(failed_refs),
            unresolved_count=len(unresolved_refs),
            compatibility_status="LEGACY_TYPE_UNRESOLVED" if row_errors else "RESOLVED",
            groups=group_rows,
            expected_items=tuple(expected_items),
            matched_items=tuple(matched_items),
            satisfied_obligation_refs=tuple(satisfied_refs),
            partial_obligation_refs=tuple(partial_refs),
            failed_obligation_refs=tuple(failed_refs),
            unresolved_obligation_refs=tuple(unresolved_refs),
        )

    status = (
        "LEGACY_TYPE_UNRESOLVED"
        if unresolved_rows
        or any(summary.compatibility_status != "RESOLVED" for summary in summaries.values())
        else "RESOLVED"
    )
    return CardinalityReport(
        artifact_version="v2_1r_grounding_requirement_d3_cardinality.v2",
        policies=summaries,
        compatibility_status=status,
        unresolved_rows=tuple(unresolved_rows),
    )


def validate_report_artifacts(artifacts: Mapping[str, Any]) -> ReportContractCheck:
    """Validate report builder outputs with explicit shape/type contracts."""
    errors: list[str] = []
    row_ledgers = {"node_ledger", "obligation_ledger", "binding_ledger"}
    expected = {
        "node_ledger",
        "obligation_ledger",
        "binding_ledger",
        "omission_comparison",
        "cardinality",
        "requiredness",
        "usability",
        "readiness",
        "coverage",
        "provenance",
        "report_json",
        "report_markdown",
    }
    missing = expected - set(artifacts)
    extra = set(artifacts) - expected
    errors.extend(f"missing_artifact:{name}" for name in sorted(missing))
    errors.extend(f"unexpected_artifact:{name}" for name in sorted(extra))
    for name in row_ledgers:
        value = artifacts.get(name)
        if not isinstance(value, Mapping) or not isinstance(value.get("rows"), list):
            errors.append(f"invalid_row_ledger:{name}")
    omission = artifacts.get("omission_comparison")
    if (
        not isinstance(omission, Mapping)
        or not isinstance(omission.get("rows"), list)
        or not isinstance(omission.get("classification_counts"), Mapping)
    ):
        errors.append("invalid_report:omission_comparison")
    cardinality = artifacts.get("cardinality")
    if not isinstance(cardinality, Mapping) or not isinstance(cardinality.get("policies"), Mapping):
        errors.append("invalid_report:cardinality")
    else:
        for policy, summary in cardinality["policies"].items():
            if not isinstance(summary, Mapping):
                errors.append(f"invalid_cardinality_summary:{policy}")
                continue
            for field in (
                "group_count",
                "obligation_count",
                "expected_item_count",
                "matched_item_count",
                "satisfied_count",
                "partial_count",
                "failed_count",
                "unresolved_count",
            ):
                if type(summary.get(field)) is not int or summary[field] < 0:
                    errors.append(f"invalid_cardinality_count:{policy}:{field}")
            for field in (
                "groups",
                "expected_items",
                "matched_items",
                "satisfied_obligation_refs",
                "partial_obligation_refs",
                "failed_obligation_refs",
                "unresolved_obligation_refs",
            ):
                if not isinstance(summary.get(field), list):
                    errors.append(f"invalid_cardinality_instances:{policy}:{field}")
    for name in ("requiredness", "usability", "readiness", "coverage", "provenance"):
        if not isinstance(artifacts.get(name), Mapping):
            errors.append(f"invalid_report:{name}")
    integer_fields = {
        "requiredness": ("nodes_tested", "requiredness_stability_violation_count"),
        "usability": (
            "node_count",
            "missing_required_binding_count",
            "affected_node_count",
            "unsafe_vacuous_promotion_count",
            "verification_assertion_ready_count",
        ),
        "readiness": ("ready_nodes",),
        "coverage": ("unsupported_node_shape_count", "partially_supported_node_count"),
        "provenance": (
            "nodes_total",
            "nodes_with_complete_trace",
            "obligations_total",
            "obligations_with_complete_trace",
            "provenance_defect_count",
        ),
    }
    for report_name, fields in integer_fields.items():
        report = artifacts.get(report_name)
        if not isinstance(report, Mapping):
            continue
        for field in fields:
            if type(report.get(field)) is not int or report[field] < 0:
                errors.append(f"invalid_report_count:{report_name}:{field}")
    requiredness = artifacts.get("requiredness")
    if isinstance(requiredness, Mapping) and type(requiredness.get("all_stable")) is not bool:
        errors.append("invalid_report_type:requiredness:all_stable")
    usability = artifacts.get("usability")
    if isinstance(usability, Mapping) and not isinstance(usability.get("distribution"), Mapping):
        errors.append("invalid_report_type:usability:distribution")
    readiness = artifacts.get("readiness")
    if isinstance(readiness, Mapping) and not isinstance(
        readiness.get("blocked_nodes_by_reason"), Mapping
    ):
        errors.append("invalid_report_type:readiness:blocked_nodes_by_reason")
    coverage = artifacts.get("coverage")
    if isinstance(coverage, Mapping):
        for field in (
            "source_node_shape_counts",
            "engine_node_shape_coverage",
            "context_type_counts",
        ):
            if not isinstance(coverage.get(field), Mapping):
                errors.append(f"invalid_report_type:coverage:{field}")
    provenance = artifacts.get("provenance")
    if isinstance(provenance, Mapping) and not isinstance(
        provenance.get("provenance_defects"), list
    ):
        errors.append("invalid_report_type:provenance:provenance_defects")
    if isinstance(requiredness, Mapping) and not isinstance(requiredness.get("rows"), list):
        errors.append("invalid_report_type:requiredness:rows")
    omission = artifacts.get("omission_comparison")
    if isinstance(omission, Mapping) and not isinstance(
        omission.get("classification_counts"), Mapping
    ):
        errors.append("invalid_report_type:omission_comparison:classification_counts")
    if not isinstance(artifacts.get("report_json"), Mapping):
        errors.append("invalid_report:report_json")
    if not isinstance(artifacts.get("report_markdown"), str):
        errors.append("invalid_report:report_markdown")
    return ReportContractCheck(status="PASS" if not errors else "UNRESOLVED", errors=tuple(errors))
