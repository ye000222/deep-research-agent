"""Deterministic MMR ranking for context candidates."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise

from app.context.policy import ContextPolicy
from app.domain.context import ContextCandidate, ContextItemType

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.%+-]+|[\u3400-\u9fff]")


def lexical_features(content: str) -> frozenset[str]:
    raw = [token.lower() for token in _TOKEN_PATTERN.findall(content)]
    cjk = [token for token in raw if len(token) == 1 and "\u3400" <= token <= "\u9fff"]
    bigrams = [f"{left}{right}" for left, right in pairwise(cjk)]
    return frozenset([*raw, *bigrams])


def similarity(left: str, right: str) -> float:
    left_features = lexical_features(left)
    right_features = lexical_features(right)
    if not left_features or not right_features:
        return 0.0
    return len(left_features & right_features) / len(left_features | right_features)


def mmr_order(
    candidates: Sequence[ContextCandidate],
    policy: ContextPolicy,
) -> tuple[list[int], set[int]]:
    """Return candidate indices in MMR order and policy-pruned indices."""
    protected = [index for index, candidate in enumerate(candidates) if candidate.protected]
    remaining = [index for index, candidate in enumerate(candidates) if not candidate.protected]
    pruned: set[int] = set()
    counts: Counter[ContextItemType] = Counter()
    eligible: list[int] = []
    for index in remaining:
        candidate = candidates[index]
        limit = policy.max_items_by_type.get(candidate.item_type)
        if limit is not None and counts[candidate.item_type] >= limit:
            pruned.add(index)
            continue
        counts[candidate.item_type] += 1
        eligible.append(index)

    selected: list[int] = []
    while eligible:

        def score(index: int) -> tuple[float, float, int]:
            candidate = candidates[index]
            novelty_penalty = max(
                (similarity(candidate.content, candidates[chosen].content) for chosen in selected),
                default=0.0,
            )
            mmr = (
                policy.mmr_lambda * candidate.rank_score
                - (1.0 - policy.mmr_lambda) * novelty_penalty
            )
            return (mmr, candidate.rank_score, -index)

        best = max(eligible, key=score)
        if selected:
            duplicate = max(
                similarity(candidates[best].content, candidates[chosen].content)
                for chosen in selected
            )
            if duplicate >= policy.redundancy_threshold:
                pruned.add(best)
                eligible.remove(best)
                continue
        selected.append(best)
        eligible.remove(best)
    return [*protected, *selected], pruned
