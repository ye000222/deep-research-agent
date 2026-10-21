"""Pure deterministic derivation of V2.4 proposition research gaps."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, cast
from uuid import UUID

from app.domain.atomic_proposition import ClaimPropositionLink, PropositionEvidenceLink
from app.domain.contradiction_detection import (
    CONFLICT_ASSESSMENT_SCHEMA_VERSION,
    ConflictReasonCode,
    PropositionConflictAssessment,
)
from app.domain.corroboration_aggregation import (
    INDEPENDENT_CORROBORATION_SCHEMA_VERSION,
    CorroborationReasonCode,
    IndependentCorroborationAssessment,
    OwnerCorroborationStatus,
)
from app.domain.proposition_evidence_direction import (
    PROPOSITION_EVIDENCE_DIRECTION_SCHEMA_VERSION,
    DirectionReason,
    DirectionStatus,
    PropositionEvidenceDirectionAssessment,
)
from app.domain.proposition_research_gap import (
    GAP_DERIVATION_DIAGNOSTIC_VERSION,
    GAP_SCOPE_POLICY_VERSION,
    PROPOSITION_RESEARCH_GAP_SCHEMA_VERSION,
    DesiredEvidenceRelation,
    GapDiagnostic,
    GapDiagnosticCode,
    GapPriorityClass,
    GapPriorityInputs,
    GapScopePolicy,
    GapSearchability,
    PropositionGapDerivationIssue,
    PropositionGapLifecycle,
    PropositionGapReasonCode,
    PropositionResearchGap,
    PropositionResearchGapType,
    RequiredEvidenceCharacteristics,
    ScopeExpansionPolicy,
)
from app.domain.proposition_support_group import (
    PROPOSITION_SUPPORT_GROUP_SCHEMA_VERSION,
    PropositionSupportGroup,
)
from app.domain.proposition_verification import (
    PROPOSITION_VERIFICATION_SCHEMA_VERSION,
    ConflictScopeCompleteness,
    ConflictState,
    PositiveSupportState,
    PropositionVerificationAssessment,
    PropositionVerificationStatus,
    VerificationReasonCode,
)
from app.domain.semantic_runtime_budget import SemanticRuntimeBudgetState
from app.domain.source_reliability_assessment import (
    SCHEMA_VERSION as SOURCE_RELIABILITY_SCHEMA_VERSION,
)
from app.domain.source_reliability_assessment import (
    SourceReliabilityAssessment,
)

_CAPABILITY_CODES = frozenset(
    {
        GapDiagnosticCode.UNSUPPORTED_VALUE_PROFILE.value,
        GapDiagnosticCode.UNSUPPORTED_SEMANTIC_RELATION.value,
        GapDiagnosticCode.FORMATION_CAPABILITY_LIMITATION.value,
        GapDiagnosticCode.GROUNDING_CAPABILITY_LIMITATION.value,
        GapDiagnosticCode.IDENTITY_CAPABILITY_LIMITATION.value,
    }
)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_hash(value: object) -> str:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _canonical_hash(dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_hash(asdict(cast(Any, value)))
    raise TypeError("unsupported_gap_hash_input")


@dataclass(frozen=True, slots=True)
class PropositionGapDerivationTarget:
    """All already-derived formal inputs for one B1 support group."""

    run_ref: UUID
    objective_scope_ref: str
    target_group: PropositionSupportGroup
    verification_assessment: PropositionVerificationAssessment
    corroboration_assessments: tuple[IndependentCorroborationAssessment, ...] = ()
    conflict_assessment: PropositionConflictAssessment | None = None
    support_groups: tuple[PropositionSupportGroup, ...] = ()
    claim_proposition_links: tuple[ClaimPropositionLink, ...] = ()
    proposition_evidence_links: tuple[PropositionEvidenceLink, ...] = ()
    direction_assessments: tuple[PropositionEvidenceDirectionAssessment, ...] = ()
    diagnostics: tuple[GapDiagnostic, ...] = ()
    budget_state: SemanticRuntimeBudgetState | None = None
    scope_policy: GapScopePolicy = field(default_factory=GapScopePolicy)
    source_reliability_assessments: tuple[SourceReliabilityAssessment, ...] = ()
    parent_research_priority: int | None = None
    attempt_count: int = 0
    research_search_budget_available: bool | None = None

    def __post_init__(self) -> None:
        if not self.support_groups:
            object.__setattr__(self, "support_groups", (self.target_group,))


@dataclass(frozen=True, slots=True)
class PropositionGapDerivationResult:
    """Sorted unique gaps plus isolated per-target derivation issues."""

    gaps: tuple[PropositionResearchGap, ...]
    issues: tuple[PropositionGapDerivationIssue, ...]


class PropositionGapDerivationService:
    """Derive gaps only from frozen semantic assessments and typed diagnostics."""

    def derive(
        self,
        targets: tuple[PropositionGapDerivationTarget, ...],
    ) -> PropositionGapDerivationResult:
        gaps_by_ref: dict[str, PropositionResearchGap] = {}
        issues: list[PropositionGapDerivationIssue] = []
        for target in sorted(
            targets,
            key=lambda item: (
                str(item.run_ref),
                item.target_group.group_ref,
                item.objective_scope_ref,
            ),
        ):
            try:
                derived, target_issues = self._derive_target(target)
            except (ValueError, TypeError, KeyError):
                derived = ()
                target_issues = (
                    self._issue(
                        target,
                        PropositionGapReasonCode.INPUT_REFERENCE_MISMATCH,
                        "TARGET_INPUT_INCONSISTENT",
                    ),
                )
            issues.extend(target_issues)
            for gap in derived:
                previous = gaps_by_ref.get(gap.gap_ref)
                if previous is None:
                    gaps_by_ref[gap.gap_ref] = gap
                elif previous != gap:
                    # Gap identity deliberately excludes assessment hashes and
                    # attempts. For duplicate observations, retain the earliest
                    # lifecycle attempt and use the assessment hash only as a
                    # deterministic tie-breaker for its attached provenance.
                    gaps_by_ref[gap.gap_ref] = min(
                        (previous, gap),
                        key=lambda item: (
                            item.attempt_count,
                            item.verification_assessment_hash,
                        ),
                    )
        ordered_gaps = tuple(
            sorted(gaps_by_ref.values(), key=lambda gap: gap.priority_ordering_key)
        )
        ordered_issues = tuple(
            sorted(
                set(issues),
                key=lambda issue: (
                    str(issue.run_ref),
                    issue.target_group_ref,
                    issue.reason_code.value,
                    issue.assessment_ref or "",
                ),
            )
        )
        return PropositionGapDerivationResult(ordered_gaps, ordered_issues)

    def _derive_target(
        self,
        target: PropositionGapDerivationTarget,
    ) -> tuple[tuple[PropositionResearchGap, ...], tuple[PropositionGapDerivationIssue, ...]]:
        assessment = target.verification_assessment
        group = target.target_group
        assessment_hash = _model_hash(assessment)
        issues: list[PropositionGapDerivationIssue] = []
        if not target.objective_scope_ref.strip():
            return (), (
                self._issue(
                    target,
                    PropositionGapReasonCode.INPUT_REFERENCE_MISMATCH,
                    "OBJECTIVE_SCOPE_REF_MISSING",
                    assessment_hash,
                ),
            )
        versions_valid, version_problem = self._versions_valid(target)
        if not versions_valid:
            return (), (
                self._issue(
                    target,
                    PropositionGapReasonCode.UNKNOWN_CONTRACT_VERSION,
                    version_problem,
                    assessment_hash,
                ),
            )
        if assessment.group_ref != group.group_ref:
            return (), (
                self._issue(
                    target,
                    PropositionGapReasonCode.INPUT_REFERENCE_MISMATCH,
                    "ASSESSMENT_GROUP_REF_MISMATCH",
                    assessment_hash,
                ),
            )
        if not self._references_valid(target):
            return (), (
                self._issue(
                    target,
                    PropositionGapReasonCode.INPUT_REFERENCE_MISMATCH,
                    "FORMAL_INPUT_REFERENCE_MISMATCH",
                    assessment_hash,
                ),
            )
        if group.group_ref not in {item.group_ref for item in target.corroboration_assessments}:
            return (), (
                self._issue(
                    target,
                    PropositionGapReasonCode.CORROBORATION_ASSESSMENT_MISSING,
                    "TARGET_CORROBORATION_ASSESSMENT_MISSING",
                    assessment_hash,
                ),
            )

        diagnostics = self._diagnostic_codes(target, assessment_hash)
        for issue in diagnostics[1]:
            issues.append(issue)
        known_codes = diagnostics[0]
        if any(
            issue.reason_code
            in {
                PropositionGapReasonCode.UNKNOWN_REASON_CODE,
                PropositionGapReasonCode.UNKNOWN_CONTRACT_VERSION,
            }
            for issue in issues
        ):
            return (), tuple(issues)
        # A formally VERIFIED assessment closes this objective. Diagnostics
        # describe limitations but cannot reopen it as another same-objective
        # search or compute-work gap.
        if assessment.verification_status is PropositionVerificationStatus.VERIFIED:
            return (), tuple(issues)
        capability_codes = known_codes & _CAPABILITY_CODES
        if capability_codes:
            gap = self._blocked_gap(
                target,
                PropositionResearchGapType.SEMANTIC_CAPABILITY_BLOCKED,
                PropositionGapReasonCode.SEMANTIC_CAPABILITY_LIMITATION,
                *sorted(capability_codes),
            )
            return (gap,), tuple(issues)

        has_unprocessed_budget = GapDiagnosticCode.NOT_PROCESSED_BUDGET.value in known_codes or any(
            item.direction_status is DirectionStatus.NOT_PROCESSED_BUDGET
            for item in target.direction_assessments
        )
        if has_unprocessed_budget:
            issues.append(
                self._issue(
                    target,
                    PropositionGapReasonCode.SEMANTIC_WORK_BUDGET_NOT_PROCESSED,
                    "NOT_PROCESSED_BUDGET",
                    assessment_hash,
                )
            )

        derived: list[PropositionResearchGap] = []
        if has_unprocessed_budget:
            derived.append(
                self._blocked_gap(
                    target,
                    PropositionResearchGapType.SEMANTIC_WORK_NOT_PROCESSED,
                    PropositionGapReasonCode.SEMANTIC_WORK_BUDGET_NOT_PROCESSED,
                    "NOT_PROCESSED_BUDGET",
                )
            )
            # Do not reinterpret a semantic computation that never ran as a
            # search gap from any otherwise incomplete downstream assessment.
            return tuple(derived), tuple(issues)

        reason_values = {item.value for item in assessment.reason_codes}
        known_upstream_reason_codes = {item.value for item in CorroborationReasonCode} | {
            item.value for item in ConflictReasonCode
        }
        unknown_upstream = set(assessment.upstream_reason_codes) - known_upstream_reason_codes
        if unknown_upstream:
            issues.append(
                self._issue(
                    target,
                    PropositionGapReasonCode.UNKNOWN_REASON_CODE,
                    "UNKNOWN_UPSTREAM_REASON_CODE:" + ",".join(sorted(unknown_upstream)),
                    assessment_hash,
                )
            )
            return (), tuple(issues)
        scope_mismatch = PropositionGapReasonCode.CONFLICT_SCOPE_MISMATCH.value in reason_values
        if scope_mismatch:
            issues.append(
                self._issue(
                    target,
                    PropositionGapReasonCode.CONFLICT_SCOPE_MISMATCH,
                    "CONFLICT_SCOPE_MISMATCH",
                    assessment_hash,
                )
            )

        if assessment.verification_status is PropositionVerificationStatus.UNSUPPORTED:
            if (
                assessment.positive_support is PositiveSupportState.NONE
                and VerificationReasonCode.NO_ELIGIBLE_SUPPORT in assessment.reason_codes
            ):
                derived.append(
                    self._gap(
                        target,
                        PropositionResearchGapType.MISSING_SUPPORT,
                        GapSearchability.SEARCHABLE,
                        PropositionGapReasonCode.NO_ELIGIBLE_SUPPORT,
                        assessment_hash,
                    )
                )
            else:
                issues.append(
                    self._issue(
                        target,
                        PropositionGapReasonCode.VERIFICATION_ASSESSMENT_UNKNOWN,
                        "UNSUPPORTED_WITHOUT_FORMAL_NO_ELIGIBLE_SUPPORT",
                        assessment_hash,
                    )
                )

        elif assessment.verification_status is PropositionVerificationStatus.SUPPORTED:
            if scope_mismatch:
                issues.append(
                    self._issue(
                        target,
                        PropositionGapReasonCode.CONFLICT_SCOPE_MISMATCH,
                        "CONFLICT_SCOPE_MISMATCH",
                        assessment_hash,
                    )
                )
            elif (
                assessment.owner_corroboration_status is OwnerCorroborationStatus.SINGLE_KNOWN_OWNER
                and assessment.conflict_state is ConflictState.CLEAR
            ):
                derived.append(
                    self._gap(
                        target,
                        PropositionResearchGapType.MISSING_INDEPENDENT_CORROBORATION,
                        GapSearchability.SEARCHABLE,
                        PropositionGapReasonCode.OWNER_CORROBORATION_INSUFFICIENT,
                        assessment_hash,
                    )
                )
            elif (
                assessment.owner_corroboration_status is OwnerCorroborationStatus.MULTI_KNOWN_OWNERS
                and assessment.known_support_owner_count >= 2
                and assessment.verification_scope.completeness
                is not ConflictScopeCompleteness.COMPLETE
                and VerificationReasonCode.CONFLICT_SCOPE_NOT_COMPLETE in assessment.reason_codes
            ):
                derived.append(self._scope_gap(target, assessment_hash))
            elif assessment.conflict_state is ConflictState.ACTIVE:
                derived.extend(self._conflict_gaps(target, assessment_hash))
            else:
                issues.append(
                    self._issue(
                        target,
                        PropositionGapReasonCode.VERIFICATION_ASSESSMENT_UNKNOWN,
                        "SUPPORTED_REASON_NOT_ACTIONABLE_OR_UNRESOLVED",
                        assessment_hash,
                    )
                )

        elif assessment.verification_status is PropositionVerificationStatus.CONFLICTED:
            derived.extend(self._conflict_gaps(target, assessment_hash))

        elif assessment.verification_status is PropositionVerificationStatus.UNKNOWN:
            if scope_mismatch:
                pass
            elif (
                VerificationReasonCode.CONFLICT_SCOPE_NOT_COMPLETE in assessment.reason_codes
                and assessment.verification_scope.completeness
                is not ConflictScopeCompleteness.COMPLETE
            ):
                derived.append(self._scope_gap(target, assessment_hash))
            else:
                for diagnostic_code in sorted(known_codes - _CAPABILITY_CODES):
                    if diagnostic_code == GapDiagnosticCode.NOT_PROCESSED_BUDGET.value:
                        continue
                    if diagnostic_code == GapDiagnosticCode.CONTEXT_NOT_COMPARABLE.value:
                        derived.append(self._context_gap(target, assessment_hash))
                issues.append(
                    self._issue(
                        target,
                        PropositionGapReasonCode.VERIFICATION_ASSESSMENT_UNKNOWN,
                        "UNKNOWN_VERIFICATION_STATE_FAIL_CLOSED",
                        assessment_hash,
                    )
                )
        else:
            issues.append(
                self._issue(
                    target,
                    PropositionGapReasonCode.UNKNOWN_REASON_CODE,
                    "UNKNOWN_VERIFICATION_STATUS",
                    assessment_hash,
                )
            )

        return tuple(derived), tuple(issues)

    def _versions_valid(self, target: PropositionGapDerivationTarget) -> tuple[bool, str]:
        if target.target_group.schema_version != PROPOSITION_SUPPORT_GROUP_SCHEMA_VERSION:
            return False, "SUPPORT_GROUP_SCHEMA_VERSION_UNKNOWN"
        assessment = target.verification_assessment
        if assessment.schema_version != PROPOSITION_VERIFICATION_SCHEMA_VERSION:
            return False, "VERIFICATION_SCHEMA_VERSION_UNKNOWN"
        if assessment.support_group_schema_version != PROPOSITION_SUPPORT_GROUP_SCHEMA_VERSION:
            return False, "VERIFICATION_SUPPORT_GROUP_VERSION_UNKNOWN"
        if assessment.corroboration_schema_version != INDEPENDENT_CORROBORATION_SCHEMA_VERSION:
            return False, "VERIFICATION_CORROBORATION_VERSION_UNKNOWN"
        if assessment.conflict_schema_version != CONFLICT_ASSESSMENT_SCHEMA_VERSION:
            return False, "VERIFICATION_CONFLICT_VERSION_MISMATCH"
        for item in target.corroboration_assessments:
            if item.schema_version != INDEPENDENT_CORROBORATION_SCHEMA_VERSION:
                return False, "CORROBORATION_SCHEMA_VERSION_UNKNOWN"
        if target.conflict_assessment is not None and (
            target.conflict_assessment.schema_version != CONFLICT_ASSESSMENT_SCHEMA_VERSION
        ):
            return False, "CONFLICT_SCHEMA_VERSION_UNKNOWN"
        if target.scope_policy.schema_version != GAP_SCOPE_POLICY_VERSION:
            return False, "SCOPE_POLICY_SCHEMA_VERSION_UNKNOWN"
        if any(
            item.schema_version != GAP_DERIVATION_DIAGNOSTIC_VERSION for item in target.diagnostics
        ):
            return False, "GAP_DIAGNOSTIC_SCHEMA_VERSION_UNKNOWN"
        if any(
            item.schema_version != PROPOSITION_EVIDENCE_DIRECTION_SCHEMA_VERSION
            for item in target.direction_assessments
        ):
            return False, "DIRECTION_ASSESSMENT_SCHEMA_VERSION_UNKNOWN"
        known_direction_reasons = {item.value for item in DirectionReason}
        if any(
            set(reason.value for reason in item.reason_codes) - known_direction_reasons
            for item in target.direction_assessments
        ):
            return False, "DIRECTION_REASON_CODE_UNKNOWN"
        if any(
            item.schema_version != SOURCE_RELIABILITY_SCHEMA_VERSION
            for item in target.source_reliability_assessments
        ):
            return False, "SOURCE_RELIABILITY_SCHEMA_VERSION_UNKNOWN"
        return True, ""

    @staticmethod
    def _references_valid(target: PropositionGapDerivationTarget) -> bool:
        group = target.target_group
        props = set(group.proposition_refs)
        by_group = {item.group_ref: item for item in target.corroboration_assessments}
        if group.evidence_memberships and group.group_ref not in by_group:
            return False
        if len(by_group) != len(target.corroboration_assessments):
            return False
        if target.claim_proposition_links and any(
            item.proposition_id not in props for item in target.claim_proposition_links
        ):
            return False
        if target.proposition_evidence_links and any(
            item.proposition_id not in props or item.run_id != target.run_ref
            for item in target.proposition_evidence_links
        ):
            return False
        return target.verification_assessment.group_ref == group.group_ref

    def _diagnostic_codes(
        self,
        target: PropositionGapDerivationTarget,
        assessment_hash: str,
    ) -> tuple[set[str], tuple[PropositionGapDerivationIssue, ...]]:
        codes: set[str] = set()
        issues: list[PropositionGapDerivationIssue] = []
        for diagnostic in target.diagnostics:
            applies = (
                (not diagnostic.target_group_refs and not diagnostic.target_proposition_refs)
                or target.target_group.group_ref in diagnostic.target_group_refs
                or bool(
                    set(diagnostic.target_proposition_refs)
                    & set(target.target_group.proposition_refs)
                )
            )
            if not applies:
                continue
            if diagnostic.schema_version != GAP_DERIVATION_DIAGNOSTIC_VERSION:
                issues.append(
                    self._issue(
                        target,
                        PropositionGapReasonCode.UNKNOWN_CONTRACT_VERSION,
                        "GAP_DIAGNOSTIC_SCHEMA_VERSION_UNKNOWN",
                        assessment_hash,
                    )
                )
                continue
            if diagnostic.code not in {item.value for item in GapDiagnosticCode}:
                issues.append(
                    self._issue(
                        target,
                        PropositionGapReasonCode.UNKNOWN_REASON_CODE,
                        f"UNKNOWN_DIAGNOSTIC:{diagnostic.code}",
                        assessment_hash,
                    )
                )
                continue
            codes.add(diagnostic.code)
        return codes, tuple(issues)

    def _scope_gap(
        self, target: PropositionGapDerivationTarget, assessment_hash: str
    ) -> PropositionResearchGap:
        policy = target.scope_policy
        if policy.expansion is ScopeExpansionPolicy.ALLOWED:
            searchability = GapSearchability.SEARCHABLE
            reason = PropositionGapReasonCode.CONFLICT_SCOPE_NOT_COMPLETE
        elif policy.expansion is ScopeExpansionPolicy.DISALLOWED:
            searchability = GapSearchability.NON_SEARCHABLE
            reason = PropositionGapReasonCode.SCOPE_EXPANSION_DISALLOWED
        else:
            searchability = GapSearchability.UNKNOWN_SEARCHABILITY
            reason = PropositionGapReasonCode.SCOPE_EXPANSION_POLICY_UNKNOWN
        return self._gap(
            target,
            PropositionResearchGapType.SCOPE_COMPLETION,
            searchability,
            reason,
            assessment_hash,
            desired=(DesiredEvidenceRelation.SUPPORTS, DesiredEvidenceRelation.CONTRADICTS),
        )

    def _conflict_gaps(
        self, target: PropositionGapDerivationTarget, assessment_hash: str
    ) -> tuple[PropositionResearchGap, ...]:
        if target.conflict_assessment is None:
            return (
                self._blocked_gap(
                    target,
                    PropositionResearchGapType.CONFLICT_RESOLUTION,
                    PropositionGapReasonCode.CONFLICT_ASSESSMENT_MISSING,
                    "CONFLICT_ASSESSMENT_MISSING",
                ),
            )
        if not target.verification_assessment.active_conflict_refs:
            return (
                self._blocked_gap(
                    target,
                    PropositionResearchGapType.CONFLICT_RESOLUTION,
                    PropositionGapReasonCode.VERIFICATION_ASSESSMENT_UNKNOWN,
                    "CONFLICTED_WITHOUT_ACTIVE_CONFLICT_REFS",
                ),
            )
        target_ref = target.target_group.group_ref
        relevant_coverage = tuple(
            item
            for item in target.conflict_assessment.group_pair_coverages
            if target_ref in {item.left_group_ref, item.right_group_ref}
        )
        not_comparable_refs = tuple(
            f"A2_PAIR:{pair.left_proposition_ref}:{pair.right_proposition_ref}"
            for item in relevant_coverage
            for pair in item.not_comparable_pair_refs
        )
        direct_refs = tuple(
            f"A2_PAIR:{pair.left_proposition_ref}:{pair.right_proposition_ref}"
            for item in relevant_coverage
            for pair in item.direct_contradiction_pair_refs
        )
        if not_comparable_refs and not direct_refs:
            return (
                self._context_gap(target, assessment_hash, not_comparable_refs=not_comparable_refs),
            )
        return (
            self._gap(
                target,
                PropositionResearchGapType.CONFLICT_RESOLUTION,
                GapSearchability.SEARCHABLE,
                PropositionGapReasonCode.ACTIVE_CONFLICT_PRESENT,
                assessment_hash,
                desired=(
                    DesiredEvidenceRelation.SUPPORTS,
                    DesiredEvidenceRelation.CONTRADICTS,
                    DesiredEvidenceRelation.DISAMBIGUATES,
                ),
                conflict_pair_refs=direct_refs,
            ),
        )

    def _context_gap(
        self,
        target: PropositionGapDerivationTarget,
        assessment_hash: str,
        *,
        not_comparable_refs: tuple[str, ...] = (),
    ) -> PropositionResearchGap:
        refs = not_comparable_refs or tuple(
            diagnostic.basis_refs[0]
            for diagnostic in target.diagnostics
            if diagnostic.code == GapDiagnosticCode.CONTEXT_NOT_COMPARABLE.value
            and diagnostic.basis_refs
        )
        if not refs:
            return self._blocked_gap(
                target,
                PropositionResearchGapType.CONTEXT_DISAMBIGUATION,
                PropositionGapReasonCode.NOT_COMPARABLE_CONTEXT,
                "CONTEXT_REFERENCE_MISSING",
            )
        return self._gap(
            target,
            PropositionResearchGapType.CONTEXT_DISAMBIGUATION,
            GapSearchability.SEARCHABLE,
            PropositionGapReasonCode.NOT_COMPARABLE_CONTEXT,
            assessment_hash,
            desired=(DesiredEvidenceRelation.DISAMBIGUATES,),
            context_refs=refs,
            conflict_pair_refs=refs,
        )

    def _gap(
        self,
        target: PropositionGapDerivationTarget,
        gap_type: PropositionResearchGapType,
        searchability: GapSearchability,
        reason: PropositionGapReasonCode,
        assessment_hash: str,
        *,
        desired: tuple[DesiredEvidenceRelation, ...] = (),
        conflict_pair_refs: tuple[str, ...] = (),
        context_refs: tuple[str, ...] = (),
    ) -> PropositionResearchGap:
        group = target.target_group
        assessment = target.verification_assessment
        all_groups = {item.group_ref: item for item in target.support_groups}
        corroboration_by_group = {item.group_ref: item for item in target.corroboration_assessments}
        claim_refs = tuple(
            sorted(
                {
                    item.claim_id
                    for item in target.claim_proposition_links
                    if item.proposition_id in group.proposition_refs
                },
                key=str,
            )
        )
        if not claim_refs:
            claim_refs = tuple(
                sorted(
                    {
                        link.claim_id
                        for membership in group.evidence_memberships
                        for link in membership.provenance_links
                    },
                    key=str,
                )
            )
        support_refs = set(assessment.eligible_support_evidence_refs)
        contradict_refs = set(assessment.active_counterevidence_refs)
        conflict_group_refs = set(assessment.active_conflicting_group_refs)
        for conflict_group_ref in conflict_group_refs:
            opposing = corroboration_by_group.get(conflict_group_ref)
            if opposing is not None:
                support_refs.update(str(ref) for ref in opposing.eligible_support_evidence_refs)
        existing_owner_refs = set(assessment.known_support_owner_keys)
        for membership in group.evidence_memberships:
            for link in membership.provenance_links:
                if link.source_owner_key and link.source_owner_key != "unknown":
                    existing_owner_refs.add(link.source_owner_key)
        for conflict_group_ref in conflict_group_refs:
            opposing_group = all_groups.get(conflict_group_ref)
            if opposing_group is not None:
                existing_owner_refs.update(
                    link.source_owner_key
                    for membership in opposing_group.evidence_memberships
                    for link in membership.provenance_links
                    if link.source_owner_key and link.source_owner_key != "unknown"
                )
        excluded_owners = (
            tuple(sorted(existing_owner_refs))
            if gap_type is PropositionResearchGapType.MISSING_INDEPENDENT_CORROBORATION
            else ()
        )
        required = RequiredEvidenceCharacteristics(
            target_group_ref=group.group_ref,
            target_proposition_refs=group.proposition_refs,
            desired_relations=tuple(
                sorted(
                    set(desired)
                    or (
                        (DesiredEvidenceRelation.SUPPORTS,)
                        if gap_type
                        in {
                            PropositionResearchGapType.MISSING_SUPPORT,
                            PropositionResearchGapType.MISSING_INDEPENDENT_CORROBORATION,
                        }
                        else ()
                    ),
                    key=lambda item: item.value,
                )
            ),
            required_owner_independence=bool(excluded_owners),
            excluded_owner_keys=excluded_owners,
            required_context_refs=tuple(sorted(set(context_refs))),
            support_side_evidence_refs=tuple(sorted(support_refs)),
            contradict_side_evidence_refs=tuple(sorted(contradict_refs)),
            conflict_group_refs=tuple(sorted(conflict_group_refs)),
        )
        priority_class = {
            PropositionResearchGapType.CONFLICT_RESOLUTION: GapPriorityClass.ACTIVE_CONFLICT,
            PropositionResearchGapType.MISSING_SUPPORT: GapPriorityClass.MISSING_SUPPORT,
            PropositionResearchGapType.MISSING_INDEPENDENT_CORROBORATION:
                GapPriorityClass.INDEPENDENT_CORROBORATION,
            PropositionResearchGapType.SCOPE_COMPLETION: GapPriorityClass.SCOPE_OR_CONTEXT,
            PropositionResearchGapType.CONTEXT_DISAMBIGUATION: GapPriorityClass.SCOPE_OR_CONTEXT,
            PropositionResearchGapType.SEMANTIC_CAPABILITY_BLOCKED: GapPriorityClass.BLOCKED_WORK,
            PropositionResearchGapType.SEMANTIC_WORK_NOT_PROCESSED: GapPriorityClass.BLOCKED_WORK,
        }[gap_type]
        gap_ref = self._gap_ref(target, gap_type)
        characteristics = required.model_copy(
            update={"a2_conflict_pair_refs": tuple(sorted(set(conflict_pair_refs)))}
        )
        lifecycle = (
            PropositionGapLifecycle.OPEN
            if searchability is GapSearchability.SEARCHABLE
            else PropositionGapLifecycle.BLOCKED
        )
        effective_reason = reason
        if searchability is GapSearchability.SEARCHABLE:
            if target.research_search_budget_available is False:
                searchability = GapSearchability.NON_SEARCHABLE
                lifecycle = PropositionGapLifecycle.BLOCKED
                effective_reason = PropositionGapReasonCode.RESEARCH_BUDGET_EXHAUSTED
            elif target.research_search_budget_available is None:
                searchability = GapSearchability.UNKNOWN_SEARCHABILITY
                lifecycle = PropositionGapLifecycle.BLOCKED
                effective_reason = PropositionGapReasonCode.RESEARCH_BUDGET_STATE_UNKNOWN
        budget_payload = (
            target.budget_state.model_dump(mode="json") if target.budget_state is not None else None
        )
        budget_ref = _canonical_hash(budget_payload) if budget_payload is not None else None
        source_refs = tuple(
            sorted({_model_hash(item) for item in target.source_reliability_assessments})
        )
        return PropositionResearchGap(
            gap_ref=gap_ref,
            run_ref=target.run_ref,
            target_group_ref=group.group_ref,
            target_proposition_refs=group.proposition_refs,
            parent_claim_refs=claim_refs,
            objective_scope_ref=target.objective_scope_ref,
            verification_assessment_ref=assessment_hash,
            verification_assessment_hash=assessment_hash,
            gap_type=gap_type,
            searchability=searchability,
            reason_codes=(effective_reason,),
            required_evidence=characteristics,
            existing_evidence_refs=tuple(
                sorted(
                    set(assessment.eligible_support_evidence_refs)
                    | set(assessment.active_counterevidence_refs)
                    | set(assessment.unresolved_conflict_refs)
                    | {str(membership.evidence_ref) for membership in group.evidence_memberships}
                )
            ),
            existing_owner_refs=tuple(sorted(existing_owner_refs)),
            existing_conflict_refs=tuple(
                sorted(
                    set(assessment.active_conflict_refs)
                    | set(assessment.unresolved_conflict_refs)
                    | conflict_group_refs
                    | set(conflict_pair_refs)
                )
            ),
            source_reliability_refs=source_refs,
            source_reliability_reason_codes=tuple(
                sorted(
                    {
                        reason
                        for item in target.source_reliability_assessments
                        for reason in item.reason_codes
                    }
                )
            ),
            budget_state_ref=budget_ref,
            budget_state=budget_payload,
            lifecycle_state=lifecycle,
            attempt_count=target.attempt_count,
            priority_inputs=GapPriorityInputs(
                priority_class=priority_class,
                parent_research_priority=target.parent_research_priority,
                attempt_count=target.attempt_count,
            ),
        )

    def _blocked_gap(
        self,
        target: PropositionGapDerivationTarget,
        gap_type: PropositionResearchGapType,
        reason: PropositionGapReasonCode,
        diagnostic_code: str,
    ) -> PropositionResearchGap:
        gap = self._gap(
            target,
            gap_type,
            GapSearchability.NON_SEARCHABLE,
            reason,
            _model_hash(target.verification_assessment),
        )
        if diagnostic_code.startswith("UNKNOWN_"):
            return gap.model_copy(
                update={
                    "searchability": GapSearchability.UNKNOWN_SEARCHABILITY,
                    "lifecycle_state": PropositionGapLifecycle.BLOCKED,
                }
            )
        return gap

    def _gap_ref(
        self,
        target: PropositionGapDerivationTarget,
        gap_type: PropositionResearchGapType,
    ) -> str:
        return _canonical_hash(
            {
                "schema_version": PROPOSITION_RESEARCH_GAP_SCHEMA_VERSION,
                "run_ref": str(target.run_ref),
                "target_group_ref": target.target_group.group_ref,
                "target_proposition_refs": [
                    str(item) for item in target.target_group.proposition_refs
                ],
                "gap_type": gap_type.value,
                "objective_scope_ref": target.objective_scope_ref,
            }
        )

    @staticmethod
    def _issue(
        target: PropositionGapDerivationTarget,
        reason: PropositionGapReasonCode,
        detail: str,
        assessment_ref: str | None = None,
    ) -> PropositionGapDerivationIssue:
        return PropositionGapDerivationIssue(
            run_ref=target.run_ref,
            target_group_ref=target.target_group.group_ref,
            target_proposition_refs=target.target_group.proposition_refs,
            assessment_ref=assessment_ref,
            reason_code=reason,
            unresolved_reason=detail,
        )
