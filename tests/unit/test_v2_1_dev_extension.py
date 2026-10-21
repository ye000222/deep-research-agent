from __future__ import annotations

from typing import Any

from scripts.build_v2_1_dev_extension import (
    PAIR_INPUT_FIELDS,
    _prediction_independent_pair_rows,
    pair_id,
    sampling_strata,
    select_extension_pairs,
)


def _pair(
    left_id: str,
    right_id: str,
    left: str,
    right: str,
    overlap: float,
) -> dict[str, Any]:
    return {
        "left_claim_id": left_id,
        "right_claim_id": right_id,
        "run_id": "run-1",
        "question_id": "q1",
        "dimension_key": "q1:d1",
        "left_text": left,
        "right_text": right,
        "lexical_overlap_jaccard": overlap,
        "relation": "same",
        "confidence": 1.0,
    }


def _stratified_pairs() -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for index in range(12):
        pairs.append(
            _pair(
                f"same-a-{index}",
                f"same-b-{index}",
                f"Alpha firm delivered {index + 100} units in 2024.",
                f"Alpha firm delivered {index + 100} units during 2024.",
                0.8,
            )
        )
        pairs.append(
            _pair(
                f"contra-a-{index}",
                f"contra-b-{index}",
                f"Beta sales value {index + 200} units in 2024.",
                f"Beta sales value {index + 201} units in 2024.",
                0.7,
            )
        )
        pairs.append(
            _pair(
                f"hard-a-{index}",
                f"hard-b-{index}",
                f"Gamma output reached {index + 300} units in 2023.",
                f"Gamma output reached {index + 300} units in 2024.",
                0.7,
            )
        )
        pairs.append(
            _pair(
                f"ambig-a-{index}",
                f"ambig-b-{index}",
                f"They changed it substantially in case {index}.",
                f"Those outcomes differ in context {index + 50}.",
                0.05,
            )
        )
    return pairs


def test_strata_are_generic_retrieval_signals_not_labels() -> None:
    same = _pair(
        "a", "b", "Firm supplied 10 units in 2024.", "Firm supplied 10 units during 2024.", 0.8
    )
    contradiction = _pair(
        "c", "d", "Firm supplied 10 units in 2024.", "Firm supplied 20 units in 2024.", 0.7
    )
    hard_negative = _pair(
        "e", "f", "Firm supplied 10 units in 2023.", "Firm supplied 10 units in 2024.", 0.7
    )
    ambiguous = _pair("g", "h", "They changed it.", "Those outcomes differ.", 0.05)

    assert "likely_same_candidate" in sampling_strata(same)
    assert "possible_contradiction_candidate" in sampling_strata(contradiction)
    assert "hard_negative_candidate" in sampling_strata(hard_negative)
    assert "ambiguous_candidate" in sampling_strata(ambiguous)
    assert all(
        "human_label" not in pair for pair in (same, contradiction, hard_negative, ambiguous)
    )


def test_selection_excludes_existing_splits_exposure_and_duplicates() -> None:
    pairs = _stratified_pairs()
    reversed_duplicate = dict(pairs[0])
    reversed_duplicate["left_claim_id"], reversed_duplicate["right_claim_id"] = (
        reversed_duplicate["right_claim_id"],
        reversed_duplicate["left_claim_id"],
    )
    exact_text_duplicate = _pair(
        "copy-a", "copy-b", pairs[0]["left_text"], pairs[0]["right_text"], 0.8
    )
    pairs.extend((reversed_duplicate, exact_text_duplicate))
    excluded = {pair_id(pairs[0]), pair_id(pairs[13]), pair_id(pairs[26]), pair_id(pairs[39])}

    selected = select_extension_pairs(pairs, excluded_pair_ids=excluded)

    assert 30 <= len(selected) <= 40
    selected_ids = [item["pair_id"] for item in selected]
    assert len(selected_ids) == len(set(selected_ids))
    assert not set(selected_ids) & excluded
    assert all(
        not (
            set(item)
            - set(PAIR_INPUT_FIELDS)
            - {"pair_id", "sampling_strata", "primary_sampling_stratum"}
        )
        for item in selected
    )
    assert all("relation" not in item and "confidence" not in item for item in selected)


def test_pair_projection_drops_matcher_predictions() -> None:
    row = _pair("left", "right", "Claim A.", "Claim B.", 0.1)
    projected = _prediction_independent_pair_rows({"pair_results": [row]})

    assert projected == [{key: row[key] for key in PAIR_INPUT_FIELDS}]
    assert "relation" not in projected[0]
    assert "confidence" not in projected[0]


def test_selection_excludes_exact_text_pairs_seen_in_frozen_or_exposed_sets() -> None:
    pairs = _stratified_pairs()
    excluded_pair = pairs[0]
    exact_text_duplicate = _pair(
        "new-left", "new-right", excluded_pair["left_text"], excluded_pair["right_text"], 0.8
    )
    selected = select_extension_pairs(
        [*pairs[1:], exact_text_duplicate],
        excluded_pair_ids={pair_id(excluded_pair)},
        excluded_exact_text_pairs={
            tuple(
                sorted(
                    (excluded_pair["left_text"].casefold(), excluded_pair["right_text"].casefold())
                )
            )
        },
    )

    assert all(item["left_text"] != excluded_pair["left_text"] for item in selected)
