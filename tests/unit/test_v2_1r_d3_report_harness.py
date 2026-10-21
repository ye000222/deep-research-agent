from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID

from scripts import replay_v2_1r_grounding_requirements_d3 as replay
from scripts.v2_1r_d3_report_contracts import (
    CARDINALITY_POLICIES,
    canonical_json,
    canonical_sha256,
    cardinality_report,
    jsonable,
    validate_report_artifacts,
)


def _obligation(
    policy: str,
    *,
    ref: str,
    state: str = "SATISFIED",
    matched: list[str] | int | None = None,
    satisfied: bool = True,
    **updates: object,
) -> dict[str, object]:
    row: dict[str, object] = {
        "claim_id": "Q1",
        "node_ref": "N1",
        "obligation_ref": ref,
        "cardinality_policy": policy,
        "final_obligation_state": state,
        "matched_bindings": ["B1"] if matched is None else matched,
        "cardinality_satisfied": satisfied,
        "expected_member_key": "M1" if policy == "ALL_EXPECTED_MEMBERS" else None,
        "expected_target_key": "T1" if policy == "PER_TARGET" else None,
        "expected_value_key": "V1" if policy == "PER_VALUE" else None,
    }
    row.update(updates)
    return row


def test_cardinality_report_has_distinct_integer_counts_and_instance_lists() -> None:
    rows = [
        _obligation(
            "EXACTLY_ONE", ref="O1", state="AMBIGUOUS", matched=["B1", "B2"], satisfied=False
        ),
        _obligation("AT_LEAST_ONE", ref="O2"),
        _obligation(
            "ALL_EXPECTED_MEMBERS",
            ref="O3",
            state="MISSING_REQUIRED_BINDING",
            matched=[],
            satisfied=False,
        ),
        _obligation(
            "PER_TARGET", ref="O4", state="INSUFFICIENT_EVIDENCE", matched=["B3"], satisfied=False
        ),
        _obligation("PER_VALUE", ref="O5", state="TEXTUALLY_INVALID", matched=[], satisfied=False),
    ]
    report = cardinality_report(rows).to_dict()
    assert report["compatibility_status"] == "RESOLVED"
    assert set(report["policies"]) == set(CARDINALITY_POLICIES)
    exact = report["policies"]["EXACTLY_ONE"]
    assert exact["group_count"] == exact["obligation_count"] == 1
    assert exact["expected_item_count"] == 1
    assert exact["matched_item_count"] == 2
    assert exact["partial_count"] == 1
    assert exact["failed_count"] == 1
    assert exact["matched_items"] == [
        {"claim_id": "Q1", "obligation_ref": "O1", "binding_ref": "B1"},
        {"claim_id": "Q1", "obligation_ref": "O1", "binding_ref": "B2"},
    ]
    target = report["policies"]["PER_TARGET"]
    assert target["unresolved_count"] == 1
    assert target["expected_items"][0]["expected_key"] == "T1"


def test_cardinality_empty_zero_cases_are_typed() -> None:
    report = cardinality_report([]).to_dict()
    assert report["compatibility_status"] == "RESOLVED"
    for policy in CARDINALITY_POLICIES:
        row = report["policies"][policy]
        for count_field in (
            "group_count",
            "obligation_count",
            "expected_item_count",
            "matched_item_count",
            "satisfied_count",
            "partial_count",
            "failed_count",
            "unresolved_count",
        ):
            assert type(row[count_field]) is int
            assert row[count_field] == 0
        assert row["groups"] == []
        assert row["expected_items"] == []
        assert row["matched_items"] == []


def test_cardinality_partial_and_multiple_candidate_cases_keep_contract_types() -> None:
    rows = [
        _obligation(
            "PER_TARGET", ref="O1", state="AMBIGUOUS", matched=["B1", "B2"], satisfied=False
        ),
        _obligation(
            "PER_TARGET",
            ref="O2",
            state="MISSING_REQUIRED_BINDING",
            matched=[],
            satisfied=False,
            node_ref="N2",
            expected_target_key="T2",
        ),
        _obligation(
            "ALL_EXPECTED_MEMBERS",
            ref="O3",
            state="AMBIGUOUS",
            matched=["B3"],
            satisfied=False,
            node_ref="N3",
        ),
    ]
    target = cardinality_report(rows).to_dict()["policies"]["PER_TARGET"]
    assert target["group_count"] == 2
    assert target["obligation_count"] == 2
    assert target["matched_item_count"] == 2
    assert target["partial_count"] == 1
    assert target["unresolved_count"] == 1
    assert all(
        type(target[field]) is int
        for field in (
            "group_count",
            "obligation_count",
            "expected_item_count",
            "matched_item_count",
            "satisfied_count",
            "partial_count",
            "failed_count",
            "unresolved_count",
        )
    )
    all_members = cardinality_report(rows).to_dict()["policies"]["ALL_EXPECTED_MEMBERS"]
    assert all_members["expected_items"][0]["expected_key"] == "M1"


def test_cardinality_zero_candidates_and_per_value_partial_cases() -> None:
    rows = [
        _obligation(
            "AT_LEAST_ONE",
            ref="O-zero",
            state="MISSING_REQUIRED_BINDING",
            matched=[],
            satisfied=False,
        ),
        _obligation(
            "PER_VALUE",
            ref="O-value",
            state="AMBIGUOUS",
            matched=["B-value"],
            satisfied=False,
        ),
        _obligation(
            "ALL_EXPECTED_MEMBERS",
            ref="O-member-a",
            matched=["B-member-a"],
            expected_member_key="member-a",
        ),
        _obligation(
            "ALL_EXPECTED_MEMBERS",
            ref="O-member-b",
            state="MISSING_REQUIRED_BINDING",
            matched=[],
            satisfied=False,
            expected_member_key="member-b",
        ),
    ]
    report = cardinality_report(rows).to_dict()["policies"]
    at_least_one = report["AT_LEAST_ONE"]
    assert at_least_one["expected_item_count"] == 1
    assert at_least_one["matched_item_count"] == 0
    assert at_least_one["unresolved_count"] == 1
    per_value = report["PER_VALUE"]
    assert per_value["expected_item_count"] == 1
    assert per_value["matched_item_count"] == 1
    assert per_value["partial_count"] == 1
    assert per_value["expected_items"][0]["expected_key"] == "V1"
    all_members = report["ALL_EXPECTED_MEMBERS"]
    assert all_members["group_count"] == 1
    assert all_members["expected_item_count"] == 2
    assert all_members["satisfied_count"] == 1
    assert all_members["unresolved_count"] == 1


def test_cardinality_legacy_wrong_type_is_explicitly_unresolved() -> None:
    report = cardinality_report([_obligation("EXACTLY_ONE", ref="O1", matched=2)]).to_dict()
    summary = report["policies"]["EXACTLY_ONE"]
    assert report["compatibility_status"] == "LEGACY_TYPE_UNRESOLVED"
    assert summary["compatibility_status"] == "LEGACY_TYPE_UNRESOLVED"
    assert summary["unresolved_count"] == 1
    assert summary["matched_item_count"] == 0


class _DemoEnum(StrEnum):
    VALUE = "demo"


def test_json_serializer_handles_required_analysis_values_deterministically() -> None:
    value = {
        "enum": _DemoEnum.VALUE,
        "tuple": (1, "x"),
        "set": {"z", "a"},
        "decimal": Decimal("1.2300"),
        "uuid": UUID("12345678-1234-5678-1234-567812345678"),
        "datetime": datetime(2026, 9, 30, tzinfo=UTC),
        "nested": {"diagnostics": ["ok", None], "optional": None},
    }
    converted = jsonable(value)
    assert converted["enum"] == "demo"
    assert converted["tuple"] == [1, "x"]
    assert converted["set"] == ["a", "z"]
    assert converted["decimal"] == "1.2300"
    assert converted["uuid"] == "12345678-1234-5678-1234-567812345678"
    assert converted["datetime"] == "2026-09-30T00:00:00+00:00"
    assert canonical_json(value) == canonical_json(value)
    assert canonical_sha256(value) == canonical_sha256(value)


def test_empty_report_builders_and_contracts_do_not_raise() -> None:
    assert replay._requiredness_stability([])["all_stable"] is True
    assert replay._usability_report([], []) == {
        "artifact_version": "v2_1r_grounding_requirement_d3_node_usability.v1",
        "node_count": 0,
        "distribution": {},
        "missing_required_binding_count": 0,
        "affected_node_count": 0,
        "affected_node_refs": [],
        "unsafe_vacuous_promotion_count": 0,
        "verification_assertion_ready_count": 0,
    }
    assert replay._readiness_report([], [])["ready_nodes"] == 0
    assert replay._contract_coverage([], [])["unsupported_node_shape_count"] == 0
    assert replay._provenance_report([], [], [])["provenance_defect_count"] == 0
    assert replay._safety_review([], [])["counts"] == {}
    assert replay._fail_closed_report([], [])["unsafe_promotion_count"] == 0
    omission = replay._compare_known_omissions(
        root_audit={}, phase_c={}, node_rows=[], obligation_rows=[], binding_records=[]
    )
    assert omission["rows"] == []


def test_report_artifact_completeness_validator_rejects_legacy_count_type() -> None:
    empty_cardinality = cardinality_report([]).to_dict()
    artifacts: dict[str, object] = {
        "node_ledger": {"rows": []},
        "obligation_ledger": {"rows": []},
        "binding_ledger": {"rows": []},
        "omission_comparison": {"rows": [], "classification_counts": {}},
        "cardinality": empty_cardinality,
        "requiredness": {
            "nodes_tested": 0,
            "all_stable": True,
            "requiredness_stability_violation_count": 0,
            "rows": [],
        },
        "usability": {
            "node_count": 0,
            "distribution": {},
            "missing_required_binding_count": 0,
            "affected_node_count": 0,
            "unsafe_vacuous_promotion_count": 0,
            "verification_assertion_ready_count": 0,
        },
        "readiness": {"ready_nodes": 0, "blocked_nodes_by_reason": {}},
        "coverage": {
            "unsupported_node_shape_count": 0,
            "partially_supported_node_count": 0,
            "source_node_shape_counts": {},
            "engine_node_shape_coverage": {},
            "context_type_counts": {},
        },
        "provenance": {
            "nodes_total": 0,
            "nodes_with_complete_trace": 0,
            "obligations_total": 0,
            "obligations_with_complete_trace": 0,
            "provenance_defect_count": 0,
            "provenance_defects": [],
        },
        "report_json": {},
        "report_markdown": "harness only",
    }
    assert validate_report_artifacts(artifacts).status == "PASS"
    empty_cardinality["policies"]["EXACTLY_ONE"]["partial_count"] = []
    assert validate_report_artifacts(artifacts).status == "UNRESOLVED"


def test_full_pipeline_harness_runs_in_isolation_without_formal_decision() -> None:
    result = replay._run_isolated_harness_pipeline()
    replay_result = replay._run_isolated_harness_pipeline()
    assert result["status"] == "HARNESS_READY"
    assert replay_result["status"] == "HARNESS_READY"
    assert result["artifact_hashes"] == replay_result["artifact_hashes"]
    assert result["hash_manifest_sha256"] == replay_result["hash_manifest_sha256"]
    assert result["semantic_decision_computed"] is False
    assert result["semantic_output_quarantined"] is True
    assert result["formal_attempt_started"] is False
    assert len(result["required_artifacts"]) == 13
    assert result["provider_calls"] == 0
    assert result["external_http"] == 0
    assert result["semantic_verifier_calls"] == 0
    assert result["extractor_calls"] == 0
    assert result["holdout_accessed"] is False
