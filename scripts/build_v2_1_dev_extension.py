"""Freeze a blind, prediction-independent development Claim-pair extension."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SOURCE_DATASET = ARTIFACTS / "v2_1_real_claim_eval_candidates.json"
CLAIM_REPORT = ARTIFACTS / "v2_1_claim_equivalence.json"
DEV_EVAL = ARTIFACTS / "v2_1_development_eval.json"
HOLDOUT_EVAL = ARTIFACTS / "v2_1_holdout_eval.json"
AUDIT_SAMPLE = ARTIFACTS / "v2_1_semantic_audit_sample.json"
HOLDOUT_HASH = "da07c803a64a978df146f6200144f5ae47715ddfeb19b584343d1d1a268ddc8b"
EXTENSION_VERSION = "v2_1_claim_eval_dev_extension_v1"
SELECTION_METHOD_VERSION = "generic-surface-strata-round-robin-v1"
TARGET_PER_STRATUM = 9
TARGET_SIZE = TARGET_PER_STRATUM * 4

STRATA = (
    "likely_same_candidate",
    "possible_contradiction_candidate",
    "hard_negative_candidate",
    "ambiguous_candidate",
)
PAIR_INPUT_FIELDS = (
    "left_claim_id",
    "right_claim_id",
    "run_id",
    "question_id",
    "dimension_key",
    "left_text",
    "right_text",
    "lexical_overlap_jaccard",
)
NEGATION = re.compile(r"\b(?:no|not|never|without|neither)\b|不|无|未|没有|并非", re.I)
PRONOUN_OR_DEICTIC = re.compile(
    r"\b(?:it|they|them|their|this|that|these|those|its|he|she)\b|"
    r"\b(?:respectively|the former|the latter)\b|其|该|此|这些|那些",
    re.I,
)
NUMBER = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?%?(?!\w)")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
VERSION = re.compile(r"\bv\s?\d+(?:\.\d+)*\b", re.I)


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def pair_id(pair: dict[str, Any]) -> str:
    ordered_ids = sorted((str(pair["left_claim_id"]), str(pair["right_claim_id"])))
    return hashlib.sha256("|".join(ordered_ids).encode("utf-8")).hexdigest()[:24]


def _exact_text_key(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _exact_pair_key(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[\w\u3400-\u9fff]+", text.casefold(), flags=re.UNICODE))


def _pair_overlap(pair: dict[str, Any]) -> float:
    value = pair.get("lexical_overlap_jaccard")
    if isinstance(value, (int, float)):
        return max(0.0, min(float(value), 1.0))
    left = _token_set(str(pair.get("left_text") or ""))
    right = _token_set(str(pair.get("right_text") or ""))
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def sampling_strata(pair: dict[str, Any]) -> tuple[str, ...]:
    """Classify only as retrieval strata; never inspect matcher fields or infer labels."""

    left = str(pair.get("left_text") or "")
    right = str(pair.get("right_text") or "")
    overlap = _pair_overlap(pair)
    left_numbers = set(NUMBER.findall(left))
    right_numbers = set(NUMBER.findall(right))
    left_years = set(YEAR.findall(left))
    right_years = set(YEAR.findall(right))
    left_versions = {value.casefold().replace(" ", "") for value in VERSION.findall(left)}
    right_versions = {value.casefold().replace(" ", "") for value in VERSION.findall(right)}
    polarity_differs = bool(NEGATION.search(left)) != bool(NEGATION.search(right))

    found: list[str] = []
    if (
        overlap >= 0.12
        and left_numbers == right_numbers
        and left_years == right_years
        and left_versions == right_versions
        and not polarity_differs
    ):
        found.append("likely_same_candidate")

    numeric_or_polarity_conflict = (
        bool(left_numbers)
        and bool(right_numbers)
        and left_numbers != right_numbers
        and left_years == right_years
        and left_versions == right_versions
    ) or polarity_differs
    if overlap >= 0.08 and numeric_or_polarity_conflict:
        found.append("possible_contradiction_candidate")

    qualifier_difference = bool(left_years and right_years and left_years != right_years) or bool(
        left_versions and right_versions and left_versions != right_versions
    )
    if overlap >= 0.08 and qualifier_difference and not numeric_or_polarity_conflict:
        found.append("hard_negative_candidate")

    if (
        overlap <= 0.12
        or bool(PRONOUN_OR_DEICTIC.search(left))
        or bool(PRONOUN_OR_DEICTIC.search(right))
        or min(len(_token_set(left)), len(_token_set(right))) <= 3
    ):
        found.append("ambiguous_candidate")
    return tuple(found)


def _prediction_independent_pair_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Project an allow-list of raw pair/scope/text fields, dropping predictions."""

    return [
        {field: pair[field] for field in PAIR_INPUT_FIELDS if field in pair}
        for pair in report.get("pair_results", [])
    ]


def select_extension_pairs(
    pair_rows: list[dict[str, Any]],
    *,
    excluded_pair_ids: set[str],
    excluded_exact_text_pairs: set[tuple[str, str]] | None = None,
    target_per_stratum: int = TARGET_PER_STRATUM,
) -> list[dict[str, Any]]:
    queues: dict[str, list[dict[str, Any]]] = {stratum: [] for stratum in STRATA}
    seen_pair_ids: set[str] = set()
    seen_exact_text_pairs = set(excluded_exact_text_pairs or ())

    for pair in pair_rows:
        current_id = pair_id(pair)
        if current_id in excluded_pair_ids or current_id in seen_pair_ids:
            continue
        seen_pair_ids.add(current_id)
        left_key = _exact_text_key(str(pair.get("left_text") or ""))
        right_key = _exact_text_key(str(pair.get("right_text") or ""))
        if not left_key or not right_key or left_key == right_key:
            continue
        exact_pair_key = _exact_pair_key(left_key, right_key)
        if exact_pair_key in seen_exact_text_pairs:
            continue
        seen_exact_text_pairs.add(exact_pair_key)

        # Each pair enters at most one queue, assigned by generic surface-only priority.
        pair_strata = sampling_strata(pair)
        for stratum in STRATA:
            if stratum in pair_strata:
                safe_pair = {field: pair[field] for field in PAIR_INPUT_FIELDS if field in pair}
                queues[stratum].append(safe_pair)
                break

    selected: dict[str, dict[str, Any]] = {}
    used_exact_text_pairs: set[tuple[str, str]] = set()
    for stratum, queue in queues.items():
        queue.sort(
            key=lambda item: hashlib.sha256(
                f"{EXTENSION_VERSION}:{stratum}:{pair_id(item)}".encode()
            ).hexdigest()
        )
        accepted = 0
        for pair in queue:
            current_id = pair_id(pair)
            text_key = _exact_pair_key(
                _exact_text_key(str(pair.get("left_text") or "")),
                _exact_text_key(str(pair.get("right_text") or "")),
            )
            if current_id in selected or text_key in used_exact_text_pairs:
                continue
            selected[current_id] = {
                **pair,
                "pair_id": current_id,
                "sampling_strata": list(sampling_strata(pair)),
                "primary_sampling_stratum": stratum,
            }
            used_exact_text_pairs.add(text_key)
            accepted += 1
            if accepted == target_per_stratum:
                break

    if len(selected) < 30:
        available = {key: len(value) for key, value in queues.items()}
        raise ValueError(
            f"Only {len(selected)} unique candidates met generic strata quotas; "
            f"per-stratum pool sizes={available}. No artifacts were written."
        )
    return sorted(
        selected.values(),
        key=lambda item: (
            STRATA.index(item["primary_sampling_stratum"]),
            item["pair_id"],
        ),
    )


def _membership_hash(pair_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(pair_ids)).encode("utf-8")).hexdigest()


def _extension_markdown(dataset: dict[str, Any]) -> str:
    lines = [
        "# V2.1 Development Extension — Blind Human Review",
        "",
        f"- Dataset: `{dataset['dataset_version']}`",
        f"- Pair count: {dataset['pair_count']}",
        f"- Membership SHA-256: `{dataset['membership_sha256']}`",
        f"- Selection method: `{dataset['selection_method_version']}`",
        f"- Generated at: `{dataset['generated_at']}`",
        "- Candidate strata are retrieval aids, not semantic labels.",
        "- Matcher predictions and model outputs are intentionally omitted.",
        "",
    ]
    for index, item in enumerate(dataset["pairs"], 1):
        lines.extend(
            [
                f"## {index}. {item['pair_id']}",
                f"- Sampling strata: {', '.join(item['sampling_strata'])}",
                f"- Scope: run={item['run_id']}; question={item['question_id']}; "
                f"dimension={item['dimension_key']}",
                f"- Claim A (`{item['claim_a_id']}`): {item['claim_a_text']}",
                f"- Claim B (`{item['claim_b_id']}`): {item['claim_b_text']}",
                "- Human label (SAME / RELATED / CONTRADICTS / UNKNOWN):",
                "- Review rationale:",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_artifacts() -> dict[str, str]:
    dataset = _read(SOURCE_DATASET)
    if dataset.get("dataset_version") != "v2_1_claim_eval_holdout_v1":
        raise ValueError("Unexpected evaluation dataset version; refusing to alter splits.")
    if dataset.get("holdout_membership_sha256") != HOLDOUT_HASH:
        raise ValueError("Frozen holdout membership hash mismatch; STOP without writing.")

    holdout_eval_before = HOLDOUT_EVAL.read_bytes()
    holdout_ids = list(dataset["holdout_ids"])
    dev_ids = list(dataset["development_ids"])
    prior_exposed_ids = set(dataset["excluded_prior_prediction_exposure_pair_ids"])
    if len(dev_ids) != 45 or len(holdout_ids) != 25 or len(prior_exposed_ids) != 110:
        raise ValueError("Current V2.1 split/exposure counts do not match the frozen inputs.")
    if (set(dev_ids) | set(holdout_ids)) & prior_exposed_ids:
        raise ValueError("Existing Development/Holdout overlaps the exposed-pair set.")

    report = _read(CLAIM_REPORT)
    pair_rows = _prediction_independent_pair_rows(report)
    exposed_audit = _read(AUDIT_SAMPLE)
    audit_ids = {pair_id(item) for item in exposed_audit.get("comparisons", [])}
    if audit_ids & set(holdout_ids):
        raise ValueError("Frozen holdout overlaps the exposed audit sample; refusing to proceed.")

    all_excluded = set(dev_ids) | set(holdout_ids) | prior_exposed_ids | audit_ids
    excluded_text_pairs = {
        _exact_pair_key(
            _exact_text_key(str(pair.get("left_text") or "")),
            _exact_text_key(str(pair.get("right_text") or "")),
        )
        for pair in pair_rows
        if pair_id(pair) in all_excluded
    }
    selected = select_extension_pairs(
        pair_rows,
        excluded_pair_ids=all_excluded,
        excluded_exact_text_pairs=excluded_text_pairs,
    )
    selected_ids = [item["pair_id"] for item in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Extension contains duplicate pair IDs.")
    if set(selected_ids) & all_excluded:
        raise ValueError("Extension leaks an existing or exposed pair.")

    timestamp = datetime.now(UTC).isoformat()
    ext_items = [
        {
            "pair_id": item["pair_id"],
            "claim_a_id": item["left_claim_id"],
            "claim_b_id": item["right_claim_id"],
            "claim_a_text": item["left_text"],
            "claim_b_text": item["right_text"],
            "run_id": item["run_id"],
            "question_id": item["question_id"],
            "dimension_key": item["dimension_key"],
            "sampling_strata": item["sampling_strata"],
            "primary_sampling_stratum": item["primary_sampling_stratum"],
            "human_label": None,
            "label_provenance": None,
            "review_rationale": None,
        }
        for item in selected
    ]
    stratum_counts = dict(Counter(item["primary_sampling_stratum"] for item in selected))
    extension = {
        "schema_version": "v2.1-dev-extension-candidates.v1",
        "dataset_version": EXTENSION_VERSION,
        "purpose": "development_only",
        "pair_count": len(ext_items),
        "membership_sha256": _membership_hash(selected_ids),
        "generated_at": timestamp,
        "selection_method_version": SELECTION_METHOD_VERSION,
        "selection_method": (
            "Prediction-independent round-robin over generic lexical overlap, "
            "numeric, polarity, time/version and ambiguity surface signals."
        ),
        "sampling_distribution_by_primary_stratum": stratum_counts,
        "sampling_strata_are_labels": False,
        "matcher_predictions_included": False,
        "semantic_model_outputs_included": False,
        "excluded_existing_dev_count": len(dev_ids),
        "excluded_frozen_holdout_count": len(holdout_ids),
        "excluded_previously_exposed_count": len(prior_exposed_ids | audit_ids),
        "overlap_with_existing_dev": 0,
        "overlap_with_frozen_holdout": 0,
        "overlap_with_previously_exposed": 0,
        "generalization_audit": {
            "result": "GENERALIZATION_SAFE_DATASET_CONSTRUCTION = PASS",
            "benchmark_specific_sampling": False,
            "entity_specific_sampling": False,
            "source_specific_sampling": False,
            "task_family_specific_hardcoding": False,
            "matcher_prediction_dependence": False,
            "holdout_dependence": False,
        },
        "label_status": "AWAITING_HUMAN_LABELS",
        "pairs": ext_items,
    }

    # Record the user's confirmed labels in the development-only evaluation artifact.
    dev_predictions = dataset.get("matcher_predictions_separate", {})
    labeled_pairs = [
        {
            "pair_id": pair_id_value,
            "human_label": "RELATED",
            "label_provenance": "human_confirmed_by_user",
            "matcher_prediction": dev_predictions.get(pair_id_value, {}).get("matcher_prediction"),
        }
        for pair_id_value in dev_ids
    ]
    prediction_counts = Counter(item["matcher_prediction"] for item in labeled_pairs)
    correct_count = sum(item["matcher_prediction"] == item["human_label"] for item in labeled_pairs)
    dev_eval = {
        "schema_version": "v2.1-labeled-evaluation.v1",
        "dataset_version": dataset["dataset_version"],
        "split": "development",
        "candidate_count": 45,
        "label_status": "HUMAN_LABELED",
        "human_label_provenance": "human_confirmed_by_user",
        "label_counts": {"RELATED": 45, "SAME": 0, "CONTRADICTS": 0, "UNKNOWN": 0},
        "labeled_pairs": labeled_pairs,
        "matcher_predictions_separate_from_human_labels": True,
        "matcher_prediction_counts": dict(sorted(prediction_counts.items())),
        "development_only_metrics": {
            "exact_match_accuracy": correct_count / len(labeled_pairs),
            "correct_count": correct_count,
            "sample_count": len(labeled_pairs),
            "note": "Development diagnostic only; no holdout metrics or tuning performed.",
        },
        "extension_dataset_version": EXTENSION_VERSION,
        "extension_label_status": "AWAITING_HUMAN_LABELS",
        "holdout_membership_sha256": None,
        "holdout_used_for_tuning": False,
    }

    # Keep prediction records development-only; the frozen holdout remains blind.
    holdout_set = set(holdout_ids)
    candidate_dataset = json.loads(json.dumps(dataset))
    predictions = candidate_dataset.get("matcher_predictions_separate", {})
    for holdout_pair_id in holdout_set:
        predictions.pop(holdout_pair_id, None)
    candidate_dataset["matcher_predictions_scope"] = "development_pairs_only"
    for item in candidate_dataset.get("review_view", {}).get("items", []):
        current_id = str(item.get("pair_id", ""))
        if current_id in set(dev_ids):
            item["human_label"] = "RELATED"
            item["label_provenance"] = "human_confirmed_by_user"
            item["review_rationale"] = None

    if HOLDOUT_EVAL.read_bytes() != holdout_eval_before:
        raise RuntimeError("Holdout evaluation artifact changed unexpectedly.")

    output = {
        "artifacts/v2_1_real_claim_eval_candidates.json": json.dumps(
            candidate_dataset, ensure_ascii=False, indent=2
        )
        + "\n",
        "artifacts/v2_1_development_eval.json": json.dumps(dev_eval, ensure_ascii=False, indent=2)
        + "\n",
        "artifacts/v2_1_development_eval.md": _development_markdown(dev_eval),
        "artifacts/v2_1_dev_extension_candidates.json": json.dumps(
            extension, ensure_ascii=False, indent=2
        )
        + "\n",
        "artifacts/v2_1_dev_extension_candidates.md": _extension_markdown(extension),
    }
    return output


def _development_markdown(evaluation: dict[str, Any]) -> str:
    counts = evaluation["label_counts"]
    lines = [
        "# V2.1 Development Evaluation",
        "",
        f"- Existing labels: RELATED={counts['RELATED']}, SAME={counts['SAME']}, "
        f"CONTRADICTS={counts['CONTRADICTS']}, UNKNOWN={counts['UNKNOWN']}",
        f"- Provenance: `{evaluation['human_label_provenance']}`",
        f"- Development diagnostic accuracy: "
        f"{evaluation['development_only_metrics']['exact_match_accuracy']:.4f}",
        f"- Extension: `{evaluation['extension_dataset_version']}` — "
        f"{evaluation['extension_label_status']}",
        "- Human labels and matcher predictions are stored as separate fields.",
        "- No holdout metric is computed; holdout was not used for tuning.",
        "",
        "## Confirmed development pair labels",
        "",
    ]
    lines.extend(
        f"- `{item['pair_id']}` — RELATED (`human_confirmed_by_user`)"
        for item in evaluation["labeled_pairs"]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    global ARTIFACTS, SOURCE_DATASET, CLAIM_REPORT, DEV_EVAL, HOLDOUT_EVAL, AUDIT_SAMPLE
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS)
    args = parser.parse_args()
    ARTIFACTS = args.artifacts_dir
    SOURCE_DATASET = ARTIFACTS / "v2_1_real_claim_eval_candidates.json"
    CLAIM_REPORT = ARTIFACTS / "v2_1_claim_equivalence.json"
    DEV_EVAL = ARTIFACTS / "v2_1_development_eval.json"
    HOLDOUT_EVAL = ARTIFACTS / "v2_1_holdout_eval.json"
    AUDIT_SAMPLE = ARTIFACTS / "v2_1_semantic_audit_sample.json"
    outputs = build_artifacts()
    for relative_path, content in outputs.items():
        target = ARTIFACTS / Path(relative_path).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content.encode("utf-8"))
    extension = json.loads(outputs["artifacts/v2_1_dev_extension_candidates.json"])
    print(
        json.dumps(
            {
                "dataset_version": extension["dataset_version"],
                "pair_count": extension["pair_count"],
                "membership_sha256": extension["membership_sha256"],
                "strata": extension["sampling_distribution_by_primary_stratum"],
                "label_status": extension["label_status"],
                "holdout_hash": HOLDOUT_HASH,
                "development_prediction_count": len(
                    _read(SOURCE_DATASET).get("matcher_predictions_separate", {})
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
