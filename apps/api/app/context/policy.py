"""Node-level context contracts and protected-content policy."""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.context import ContextItemType


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    node_name: str
    mmr_lambda: float
    redundancy_threshold: float
    max_items_by_type: dict[ContextItemType, int]
    compressible_types: frozenset[ContextItemType]


_DEFAULT_POLICY = ContextPolicy(
    node_name="default",
    mmr_lambda=0.72,
    redundancy_threshold=0.92,
    max_items_by_type={},
    compressible_types=frozenset(
        {
            ContextItemType.STATE_SUMMARY,
            ContextItemType.SOURCE_CHUNK,
            ContextItemType.RECENT_ACTION,
            ContextItemType.MEMORY,
        }
    ),
)

_POLICIES = {
    "planner": ContextPolicy(
        node_name="planner",
        mmr_lambda=0.75,
        redundancy_threshold=0.90,
        max_items_by_type={ContextItemType.MEMORY: 8, ContextItemType.RECENT_ACTION: 5},
        compressible_types=_DEFAULT_POLICY.compressible_types,
    ),
    "evidence_extractor": ContextPolicy(
        node_name="evidence_extractor",
        mmr_lambda=0.78,
        redundancy_threshold=0.88,
        max_items_by_type={ContextItemType.SOURCE_CHUNK: 8},
        compressible_types=frozenset({ContextItemType.SOURCE_CHUNK}),
    ),
    "report_writer": ContextPolicy(
        node_name="report_writer",
        mmr_lambda=0.68,
        redundancy_threshold=0.90,
        max_items_by_type={
            ContextItemType.EVIDENCE_CARD: 20,
            ContextItemType.CONFLICT: 8,
            ContextItemType.MEMORY: 6,
        },
        compressible_types=frozenset(
            {ContextItemType.STATE_SUMMARY, ContextItemType.MEMORY, ContextItemType.RECENT_ACTION}
        ),
    ),
}


class ContextPolicyRegistry:
    def resolve(self, node_name: str) -> ContextPolicy:
        return _POLICIES.get(node_name, _DEFAULT_POLICY)
