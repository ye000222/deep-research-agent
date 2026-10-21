from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import replay_v2_1r_truth_condition_context as replay  # noqa: E402
from truth_condition_context_reference_reconciliation import reconcile_references  # noqa: E402


def test_local_ref_to_runtime_uuid_mapping_and_uuid_alias_lookup() -> None:
    result = reconcile_references(
        [{"proposition_id": "uuid-0"}, {"proposition_id": "uuid-1"}],
        {"p0": "uuid-0", "p1": "uuid-1"},
        {"p0": "INDEPENDENTLY_ASSERTED", "p1": "NOT_INDEPENDENTLY_ASSERTED"},
    )
    props = result["propositions"]
    assert [item["runtime_proposition_id"] for item in props] == ["uuid-0", "uuid-1"]
    assert [item["model_local_ref"] for item in props] == ["p0", "p1"]
    assert result["eligibility_by_ref"] == {
        "uuid-0": "INDEPENDENTLY_ASSERTED",
        "uuid-1": "NOT_INDEPENDENTLY_ASSERTED",
    }
    assert result["issues"] == []


def test_direct_same_space_eligibility_is_preserved() -> None:
    result = reconcile_references(
        [{"local_ref": "p0"}], None, [{"ref": "p0", "eligibility": "UNKNOWN"}]
    )
    assert result["eligibility_by_ref"] == {"p0": "UNKNOWN"}
    assert result["propositions"][0]["eligibility_source_key"] == "p0"
    assert result["issues"] == []


def test_missing_runtime_mapping_fails_closed_with_explicit_issue() -> None:
    result = reconcile_references(
        [{"proposition_id": "uuid-unmapped"}], {"p0": "uuid-0"}, {"p0": "UNKNOWN"}
    )
    assert "LOCAL_REF_MAPPING_MISSING" in result["issue_counts"]
    assert result["propositions"][0]["runtime_proposition_id"] == "uuid-unmapped"


def test_duplicate_local_ref_and_duplicate_runtime_id_are_reported() -> None:
    result = reconcile_references(
        [
            {"local_ref": "p0", "proposition_id": "uuid-0"},
            {"local_ref": "p0", "proposition_id": "uuid-0"},
        ],
        {"p0": "uuid-0"},
        {"p0": "INDEPENDENTLY_ASSERTED"},
    )
    assert result["issue_counts"]["DUPLICATE_LOCAL_REF"] >= 1
    assert result["issue_counts"]["DUPLICATE_RUNTIME_ID"] >= 1


def test_composition_member_and_unknown_ancestry_references_fail_closed() -> None:
    props = [{"proposition_id": "uuid-a"}]
    mapping = {"p0": "uuid-a"}
    normalized = reconcile_references(props, mapping, {"p0": "NOT_INDEPENDENTLY_ASSERTED"})
    groups = [
        {
            "local_ref": "g0",
            "operator": "OR",
            "members": [
                {"member_type": "proposition", "ref": "p0"},
                {"member_type": "proposition", "ref": "p_missing"},
                {"member_type": "group", "ref": "g_missing"},
            ],
        }
    ]
    issues = replay.composition_reference_issues(
        {"groups": groups}, mapping, normalized["propositions"]
    )
    assert {item["code"] for item in issues} == {
        "COMPOSITION_MEMBER_REF_UNRESOLVED",
        "COMPOSITION_GROUP_REF_UNRESOLVED",
    }
    assert normalized["eligibility_by_ref"]["uuid-a"] == "NOT_INDEPENDENTLY_ASSERTED"


def test_orphan_eligibility_key_is_not_silently_ignored() -> None:
    result = reconcile_references(
        [{"local_ref": "p0"}], None, {"p0": "INDEPENDENTLY_ASSERTED", "p9": "UNKNOWN"}
    )
    assert {item["code"] for item in result["issues"]} == {"ORPHAN_ELIGIBILITY_ENTRY"}


def test_proposition_without_eligibility_is_unknown_with_source_absence_reason() -> None:
    result = reconcile_references([{"local_ref": "p0"}], None, {})
    prop = result["propositions"][0]
    assert prop["eligibility"] == "UNKNOWN"
    assert prop["eligibility_reason"] == "SOURCE_ARTIFACT_HAS_NO_RESOLVABLE_RESULT"
    assert "PROPOSITION_WITHOUT_ELIGIBILITY_MAPPING" in result["issue_counts"]


def test_and_and_or_descendant_eligibility_are_mapped_without_inference() -> None:
    result = reconcile_references(
        [{"proposition_id": "uuid-a"}, {"proposition_id": "uuid-b"}],
        {"p0": "uuid-a", "p1": "uuid-b"},
        {"p0": "NOT_INDEPENDENTLY_ASSERTED", "p1": "NOT_INDEPENDENTLY_ASSERTED"},
    )
    groups = replay.normalize_groups(
        {
            "groups": [
                {
                    "group_id": "g_and",
                    "operator": "AND",
                    "members": [{"member_type": "proposition", "proposition_id": "uuid-a"}],
                },
                {
                    "group_id": "g_or",
                    "operator": "OR",
                    "members": [{"member_type": "proposition", "ref": "p1"}],
                },
            ]
        },
        result["local_to_runtime"],
    )
    assert [item["eligibility"] for item in result["propositions"]] == [
        "NOT_INDEPENDENTLY_ASSERTED",
        "NOT_INDEPENDENTLY_ASSERTED",
    ]
    assert groups[0]["member_refs"] == ["uuid-a"]
    assert groups[1]["member_refs"] == ["uuid-b"]
    assert [item["operator"] for item in groups] == ["AND", "OR"]


def test_unknown_composition_ancestry_remains_unknown_and_separate() -> None:
    result = reconcile_references(
        [{"proposition_id": "uuid-a"}],
        {"p0": "uuid-a"},
        {"p0": "UNKNOWN"},
    )
    assert result["eligibility_by_ref"]["uuid-a"] == "UNKNOWN"
    assert result["propositions"][0]["eligibility_source_key"] == "p0"
    assert result["issues"] == []


def test_historical_final16_all_45_eligibility_entries_reconcile() -> None:
    cohorts = replay.load_cohorts()
    rows = cohorts["final_unseen"]
    assert sum(len(row["propositions"]) for row in rows) == 45
    assert (
        sum(
            item["eligibility_source_key"] is not None
            for row in rows
            for item in row["reference_reconciliation"]["propositions"]
        )
        == 45
    )
    assert all(item["eligibility"] != "UNKNOWN" for row in rows for item in row["propositions"])
