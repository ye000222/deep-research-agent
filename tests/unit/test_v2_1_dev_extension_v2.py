from __future__ import annotations

from copy import deepcopy
from typing import Any

from scripts.build_v2_1_dev_extension_v2 import (
    TARGETS,
    _historical_non_holdout_text_pairs,
    _text_pair_key,
    membership_hash,
    pair_id,
    select_v2_candidates,
)


def _claim(
    claim_id: str,
    text: str,
    *,
    group: str,
    source: str = "source-1",
    chunk: str = "chunk-1",
) -> dict[str, Any]:
    return {
        "id": claim_id,
        "run_id": "run-" + group,
        "question_id": "question-" + group,
        "dimension_key": "dimension-" + group,
        "atomic_claim": text,
        "claim_hash": claim_id,
        "source_ids": [source] if source else [],
        "chunk_ids": [chunk] if chunk else [],
    }


def _four_stratum_claims() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pairs = [
        (
            "identity",
            "Acme reported 12 million units annual revenue in 2024.",
            "Acme reported annual revenue of 12 million units during 2024.",
            "source-i",
            "chunk-i",
        ),
        (
            "contradiction",
            "Beta reported 12 million units annual revenue in 2024.",
            "Beta reported annual revenue of 18 million units during 2024.",
            "source-c",
            "chunk-c",
        ),
        (
            "hard",
            "Gamma produced 12 million units annual revenue in 2023.",
            "Gamma produced 12 million units annual revenue in 2024.",
            "source-h",
            "chunk-h",
        ),
        (
            "ambiguity",
            "They reported it in Acme's annual market review.",
            "It was reported by Beta in the annual market review.",
            "source-a",
            "chunk-a",
        ),
    ]
    for group, left, right, source, chunk in pairs:
        rows.extend(
            (
                _claim(group + "-a", left, group=group, source=source, chunk=chunk),
                _claim(group + "-b", right, group=group, source=source, chunk=chunk),
            )
        )
    return rows


def test_v2_selector_is_deterministic_and_blind() -> None:
    claims = _four_stratum_claims()
    targets = {key: 1 for key in TARGETS}

    first, _ = select_v2_candidates(claims, excluded_pair_ids=set(), target_sizes=targets)
    second, _ = select_v2_candidates(claims, excluded_pair_ids=set(), target_sizes=targets)

    assert [row["pair_id"] for row in first] == [row["pair_id"] for row in second]
    assert {row["sampling_stratum"] for row in first} == set(TARGETS)
    assert all(row["human_label"] is None for row in first)
    assert all(row["label_provenance"] is None for row in first)
    assert all(row["review_rationale"] is None for row in first)
    assert all("prediction" not in str(row).casefold() for row in first)


def test_v2_selector_excludes_pair_id_reverse_duplicate_and_text_duplicate() -> None:
    identity_a, identity_b = _four_stratum_claims()[:2]
    excluded_id = pair_id(identity_a["id"], identity_b["id"])
    reversed_records = [deepcopy(identity_b), deepcopy(identity_a)]
    text_duplicate = [
        _claim(
            "text-copy-a",
            identity_a["atomic_claim"],
            group="text-copy",
            source="source-copy",
            chunk="chunk-copy",
        ),
        _claim(
            "text-copy-b",
            identity_b["atomic_claim"],
            group="text-copy",
            source="source-copy",
            chunk="chunk-copy",
        ),
    ]
    eligible = [
        _claim(
            "eligible-a",
            "Acme reported 12 million units annual revenue for 2024.",
            group="eligible",
            source="source-eligible",
            chunk="chunk-eligible",
        ),
        _claim(
            "eligible-b",
            "Acme's 2024 annual revenue reached 12 million units.",
            group="eligible",
            source="source-eligible",
            chunk="chunk-eligible",
        ),
    ]
    excluded_text = _text_pair_key(identity_a["atomic_claim"], identity_b["atomic_claim"])
    all_claims = [*reversed_records, *reversed_records, *text_duplicate, *eligible]
    selected, _ = select_v2_candidates(
        all_claims,
        excluded_pair_ids={excluded_id},
        excluded_text_pairs={excluded_text},
        target_sizes={
            **{key: 0 for key in TARGETS},
            "proposition_identity_candidates": 1,
        },
    )

    assert len(selected) == 1
    assert selected[0]["pair_id"] == pair_id("eligible-a", "eligible-b")
    assert _text_pair_key(selected[0]["claim_a_text"], selected[0]["claim_b_text"]) != excluded_text


def test_membership_hash_is_stable_for_frozen_pair_set() -> None:
    assert membership_hash(["pair-b", "pair-a"]) == membership_hash(["pair-a", "pair-b"])


def test_contradiction_candidate_requires_subject_time_and_scope_alignment() -> None:
    same_subject_different_time = [
        _claim("a", "Acme reported 12 million units annual revenue in 2023.", group="time"),
        _claim("b", "Acme reported 18 million units annual revenue in 2024.", group="time"),
    ]
    different_subject_same_value_slot = [
        _claim("c", "Acme reported 12 million units annual revenue in 2024.", group="subject"),
        _claim("d", "Beta reported 18 million units annual revenue in 2024.", group="subject"),
    ]
    different_scope = [
        _claim(
            "e", "Acme reported 12 million units annual revenue globally in 2024.", group="scope"
        ),
        _claim(
            "f",
            "Acme reported 18 million units annual revenue domestically in 2024.",
            group="scope",
        ),
    ]
    claims = same_subject_different_time + different_subject_same_value_slot + different_scope
    zero_targets = {key: 0 for key in TARGETS}
    selected, pools = select_v2_candidates(
        claims, excluded_pair_ids=set(), target_sizes=zero_targets
    )

    assert selected == []
    assert pools["contradiction_scope_candidates"] == 0


def test_unqualified_time_conflict_needs_shared_passage_provenance() -> None:
    claims = [
        _claim(
            "same-source-a",
            "Acme reported 12 million units annual revenue.",
            group="unqualified-time",
        ),
        _claim(
            "same-source-b",
            "Acme reported 18 million units annual revenue.",
            group="unqualified-time",
        ),
        _claim(
            "other-source-a",
            "Beta reported 12 million units annual revenue.",
            group="other-source",
            source="source-x",
            chunk="chunk-x",
        ),
        _claim(
            "other-source-b",
            "Beta reported 18 million units annual revenue.",
            group="other-source",
            source="source-y",
            chunk="chunk-y",
        ),
    ]
    selected, pools = select_v2_candidates(
        claims,
        excluded_pair_ids=set(),
        target_sizes={key: 0 for key in TARGETS},
    )

    assert selected == []
    assert pools["contradiction_scope_candidates"] == 1


def test_holdout_pair_text_is_never_read_for_historical_text_deduplication() -> None:
    rows: list[dict[str, Any]] = [
        {
            "left_claim_id": "00000000-0000-0000-0000-000000000001",
            "right_claim_id": "00000000-0000-0000-0000-000000000002",
            "left_text": "permitted historical text A",
            "right_text": "permitted historical text B",
        },
        {
            "left_claim_id": "00000000-0000-0000-0000-000000000003",
            "right_claim_id": "00000000-0000-0000-0000-000000000004",
            "left_text": object(),
            "right_text": object(),
        },
    ]
    holdout_pair_id = pair_id(str(rows[1]["left_claim_id"]), str(rows[1]["right_claim_id"]))
    permitted_pair_id = pair_id(str(rows[0]["left_claim_id"]), str(rows[0]["right_claim_id"]))

    result = _historical_non_holdout_text_pairs(
        rows, {holdout_pair_id}, {permitted_pair_id, holdout_pair_id}
    )

    assert len(result) == 1
