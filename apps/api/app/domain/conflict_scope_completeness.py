"""Deterministic completeness proof for a bounded C1 conflict universe.

This contract proves execution coverage only. It never infers proposition truth
or conflict absence; the frozen C1/C2 contracts remain authoritative for that.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from itertools import combinations
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.domain.contradiction_detection import (
    PropositionConflictAssessment,
    PropositionPairRef,
)
from app.domain.proposition_support_group import (
    PropositionAssertionBinding,
    PropositionSupportGroup,
)
from app.domain.proposition_verification import (
    ConflictScopeCompleteness,
    VerificationConflictScope,
)
from app.domain.verification_assertion import PropositionPairAssessment

CONFLICT_SCOPE_COMPLETENESS_VERSION = "conflict-scope-completeness-v1"


class ScopeCompletenessReason(StrEnum):
    COMPLETE_COVERAGE = "COMPLETE_COVERAGE"
    EMPTY_UNIVERSE = "EMPTY_UNIVERSE"
    GROUP_UNIVERSE_MISMATCH = "GROUP_UNIVERSE_MISMATCH"
    COVERAGE_SET_MISMATCH = "COVERAGE_SET_MISMATCH"
    PROPOSITION_PAIR_SET_MISMATCH = "PROPOSITION_PAIR_SET_MISMATCH"
    PAIR_EXECUTION_INCOMPLETE = "PAIR_EXECUTION_INCOMPLETE"
    A3_ASSESSMENT_MISSING = "A3_ASSESSMENT_MISSING"
    MISSING_PAIR = "MISSING_PAIR"
    INCONSISTENT_PAIR = "INCONSISTENT_PAIR"
    UNKNOWN_EXECUTION_STATE = "UNKNOWN_EXECUTION_STATE"
    INPUT_IDENTITY_MISMATCH = "INPUT_IDENTITY_MISMATCH"
    UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"


class RequiredGroupPair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    left_group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    right_group_ref: str = Field(pattern=r"^[0-9a-f]{64}$")


class PlannedPropositionPair(BaseModel):
    """Safe identity projection from a real PairCandidatePlan binding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    pair_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    left_proposition_ref: UUID
    right_proposition_ref: UUID


class ConflictScopeCompletenessResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = CONFLICT_SCOPE_COMPLETENESS_VERSION
    completeness: ConflictScopeCompleteness
    candidate_group_refs: tuple[str, ...]
    required_group_pair_refs: tuple[RequiredGroupPair, ...]
    required_proposition_pair_refs: tuple[PropositionPairRef, ...]
    scope_ref: str = Field(pattern=r"^[0-9a-f]{64}$")
    universe_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    coverage_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason_codes: tuple[ScopeCompletenessReason, ...]

    def for_target(self, group_ref: str) -> VerificationConflictScope:
        if group_ref not in self.candidate_group_refs:
            raise ValueError("conflict_scope_target_outside_candidate_universe")
        return VerificationConflictScope(
            target_group_ref=group_ref,
            completeness=self.completeness,
            scope_ref=self.scope_ref,
            candidate_group_refs=self.candidate_group_refs,
        )


class ConflictScopeCompletenessEvaluator:
    """Prove exact bounded pair execution coverage from formal pipeline records."""

    def evaluate(
        self,
        *,
        run_ref: UUID,
        groups: Sequence[PropositionSupportGroup],
        assertion_bindings: Sequence[PropositionAssertionBinding],
        coverage: PropositionConflictAssessment,
        planned_pairs: Sequence[PlannedPropositionPair],
        pair_execution_states: Mapping[str, str],
        pair_assessments: Sequence[PropositionPairAssessment],
        expected_coverage_hash: str | None = None,
    ) -> ConflictScopeCompletenessResult:
        ordered_groups = tuple(sorted(groups, key=lambda item: item.group_ref))
        groups_by_ref = {item.group_ref: item for item in ordered_groups}
        group_refs = tuple(item.group_ref for item in ordered_groups)
        required_group_pairs = tuple(
            RequiredGroupPair(left_group_ref=left, right_group_ref=right)
            for left, right in combinations(group_refs, 2)
        )
        required_group_pair_keys = {
            (item.left_group_ref, item.right_group_ref) for item in required_group_pairs
        }
        coverage_keys = {
            (item.left_group_ref, item.right_group_ref) for item in coverage.group_pair_coverages
        }

        prop_to_assertion: dict[UUID, str] = {}
        assertion_to_prop: dict[str, UUID] = {}
        identity_invalid = False
        for binding in assertion_bindings:
            if (
                binding.proposition_ref in prop_to_assertion
                or binding.assertion_ref in assertion_to_prop
            ):
                identity_invalid = True
                continue
            prop_to_assertion[binding.proposition_ref] = binding.assertion_ref
            assertion_to_prop[binding.assertion_ref] = binding.proposition_ref

        expected_prop_pairs: dict[tuple[UUID, UUID], PropositionPairRef] = {}
        for group_pair in required_group_pairs:
            left = groups_by_ref[group_pair.left_group_ref]
            right = groups_by_ref[group_pair.right_group_ref]
            for left_prop in left.proposition_refs:
                for right_prop in right.proposition_refs:
                    if str(left_prop) < str(right_prop):
                        key = (left_prop, right_prop)
                        pair_ref = PropositionPairRef(
                            left_proposition_ref=left_prop,
                            right_proposition_ref=right_prop,
                            left_assertion_ref=prop_to_assertion.get(left_prop),
                            right_assertion_ref=prop_to_assertion.get(right_prop),
                        )
                    else:
                        key = (right_prop, left_prop)
                        pair_ref = PropositionPairRef(
                            left_proposition_ref=right_prop,
                            right_proposition_ref=left_prop,
                            left_assertion_ref=prop_to_assertion.get(right_prop),
                            right_assertion_ref=prop_to_assertion.get(left_prop),
                        )
                    if key in expected_prop_pairs:
                        identity_invalid = True
                    expected_prop_pairs[key] = pair_ref

        coverage_pairs = {
            (pair.left_proposition_ref, pair.right_proposition_ref): pair
            for item in coverage.group_pair_coverages
            for bucket in (
                item.direct_contradiction_pair_refs,
                item.same_pair_refs,
                item.related_pair_refs,
                item.not_comparable_pair_refs,
                item.unknown_pair_refs,
                item.missing_pair_refs,
                item.inconsistent_pair_refs,
            )
            for pair in bucket
        }
        planned_by_pair: dict[tuple[UUID, UUID], PlannedPropositionPair] = {}
        for pair in planned_pairs:
            key = _canonical_uuid_pair(pair.left_proposition_ref, pair.right_proposition_ref)
            if key in planned_by_pair:
                identity_invalid = True
            planned_by_pair[key] = pair

        a3_pairs: dict[tuple[UUID, UUID], PropositionPairAssessment] = {}
        for assessment in pair_assessments:
            a3_left_prop = assertion_to_prop.get(assessment.left_assertion_ref)
            a3_right_prop = assertion_to_prop.get(assessment.right_assertion_ref)
            if a3_left_prop is None or a3_right_prop is None or a3_left_prop == a3_right_prop:
                identity_invalid = True
                continue
            key = _canonical_uuid_pair(a3_left_prop, a3_right_prop)
            if key in a3_pairs:
                identity_invalid = True
            a3_pairs[key] = assessment

        reasons: set[ScopeCompletenessReason] = set()
        if not ordered_groups:
            reasons.add(ScopeCompletenessReason.EMPTY_UNIVERSE)
            identity_invalid = True
        if len(group_refs) != len(set(group_refs)):
            identity_invalid = True
        summary_refs = {item.group_ref for item in coverage.group_summaries}
        if summary_refs != set(group_refs):
            reasons.add(ScopeCompletenessReason.GROUP_UNIVERSE_MISMATCH)
            identity_invalid = True
        if coverage_keys != required_group_pair_keys:
            reasons.add(ScopeCompletenessReason.COVERAGE_SET_MISMATCH)
            identity_invalid = True

        actual_coverage_pairs = set(coverage_pairs)
        required_pairs = set(expected_prop_pairs)
        if actual_coverage_pairs != required_pairs:
            reasons.add(ScopeCompletenessReason.PROPOSITION_PAIR_SET_MISMATCH)
            identity_invalid = True

        planned_pairs_set = set(planned_by_pair)
        if not required_pairs.issubset(planned_pairs_set):
            reasons.add(ScopeCompletenessReason.PAIR_EXECUTION_INCOMPLETE)
        if not required_pairs.issubset(set(a3_pairs)):
            reasons.add(ScopeCompletenessReason.A3_ASSESSMENT_MISSING)
        if not required_pairs.issubset(actual_coverage_pairs):
            reasons.add(ScopeCompletenessReason.MISSING_PAIR)

        planned_ref_by_pair = {key: item.pair_ref for key, item in planned_by_pair.items()}
        for pair_key in required_pairs & planned_pairs_set:
            execution_pair_ref = planned_ref_by_pair[pair_key]
            state = pair_execution_states.get(execution_pair_ref)
            if state is None or state in {
                "NOT_PROCESSED_BUDGET",
                "NOT_PROCESSED_REQUEST_SIZE",
                "FAILED_ISOLATED",
                "RESERVED",
                "NOT_PROCESSED",
                "INVALID",
                "FAILED",
            }:
                reasons.add(ScopeCompletenessReason.PAIR_EXECUTION_INCOMPLETE)
            elif state != "COMPLETED":
                reasons.add(ScopeCompletenessReason.UNKNOWN_EXECUTION_STATE)

        for group_coverage in coverage.group_pair_coverages:
            left_group = groups_by_ref.get(group_coverage.left_group_ref)
            right_group = groups_by_ref.get(group_coverage.right_group_ref)
            if left_group is None or right_group is None:
                reasons.add(ScopeCompletenessReason.GROUP_UNIVERSE_MISMATCH)
                identity_invalid = True
                continue
            expected_for_group_pair = {
                _canonical_uuid_pair(left_prop, right_prop)
                for left_prop in left_group.proposition_refs
                for right_prop in right_group.proposition_refs
            }
            flattened = {
                (pair.left_proposition_ref, pair.right_proposition_ref)
                for bucket in (
                    group_coverage.direct_contradiction_pair_refs,
                    group_coverage.same_pair_refs,
                    group_coverage.related_pair_refs,
                    group_coverage.not_comparable_pair_refs,
                    group_coverage.unknown_pair_refs,
                    group_coverage.missing_pair_refs,
                    group_coverage.inconsistent_pair_refs,
                )
                for pair in bucket
            }
            expected_count = len(expected_for_group_pair)
            if (
                group_coverage.expected_cross_pair_count != expected_count
                or flattened != expected_for_group_pair
                or group_coverage.assessed_pair_count
                != expected_count - len(group_coverage.missing_pair_refs)
            ):
                reasons.add(ScopeCompletenessReason.COVERAGE_SET_MISMATCH)
                identity_invalid = True
            if group_coverage.missing_pair_refs:
                reasons.add(ScopeCompletenessReason.MISSING_PAIR)
            if group_coverage.inconsistent_pair_refs:
                reasons.add(ScopeCompletenessReason.INCONSISTENT_PAIR)

        if identity_invalid:
            status = ConflictScopeCompleteness.UNKNOWN
            reasons.add(ScopeCompletenessReason.INPUT_IDENTITY_MISMATCH)
        elif reasons:
            status = ConflictScopeCompleteness.INCOMPLETE
        else:
            status = ConflictScopeCompleteness.COMPLETE
            reasons.add(ScopeCompletenessReason.COMPLETE_COVERAGE)

        universe_payload = {
            "version": CONFLICT_SCOPE_COMPLETENESS_VERSION,
            "run_ref": str(run_ref),
            "groups": [item.model_dump(mode="json") for item in ordered_groups],
            "assertion_bindings": [
                item.model_dump(mode="json")
                for item in sorted(assertion_bindings, key=lambda item: str(item.proposition_ref))
            ],
            "candidate_group_refs": list(group_refs),
            "required_group_pairs": [item.model_dump(mode="json") for item in required_group_pairs],
            "required_proposition_pairs": [
                item.model_dump(mode="json")
                for _, item in sorted(
                    expected_prop_pairs.items(),
                    key=lambda item: (str(item[0][0]), str(item[0][1])),
                )
            ],
        }
        universe_hash = _hash(universe_payload)
        scope_ref = _hash(
            {
                "scope_version": CONFLICT_SCOPE_COMPLETENESS_VERSION,
                "universe_hash": universe_hash,
            }
        )
        coverage_payload = coverage.model_dump(mode="json")
        coverage_hash = _hash(coverage_payload)
        if expected_coverage_hash is not None and expected_coverage_hash != coverage_hash:
            status = ConflictScopeCompleteness.UNKNOWN
            reasons.add(ScopeCompletenessReason.INPUT_IDENTITY_MISMATCH)
        input_hash = _hash(
            {
                "universe_hash": universe_hash,
                "coverage_hash": coverage_hash,
                "planned_pairs": [
                    item.model_dump(mode="json")
                    for item in sorted(planned_pairs, key=lambda item: item.pair_ref)
                ],
                "pair_execution_states": dict(sorted(pair_execution_states.items())),
                "pair_assessments": [
                    item.model_dump(mode="json")
                    for item in sorted(
                        pair_assessments,
                        key=lambda item: (
                            item.left_assertion_ref,
                            item.right_assertion_ref,
                        ),
                    )
                ],
            }
        )
        return ConflictScopeCompletenessResult(
            completeness=status,
            candidate_group_refs=group_refs,
            required_group_pair_refs=required_group_pairs,
            required_proposition_pair_refs=tuple(
                expected_prop_pairs[key]
                for key in sorted(
                    expected_prop_pairs,
                    key=lambda item: (str(item[0]), str(item[1])),
                )
            ),
            scope_ref=scope_ref,
            universe_hash=universe_hash,
            input_hash=input_hash,
            coverage_hash=coverage_hash,
            reason_codes=tuple(sorted(reasons, key=lambda item: item.value)),
        )


def _canonical_uuid_pair(left: UUID, right: UUID) -> tuple[UUID, UUID]:
    return (left, right) if str(left) < str(right) else (right, left)


def _hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
