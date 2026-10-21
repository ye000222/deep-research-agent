"""Deterministic Gap-to-Intent derivation and advisory planning projection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import ValidationError

from app.domain.gap_research_intent import (
    DesiredEvidencePurpose,
    GapIntentPlanningProjection,
    GapResearchIntent,
    GapResearchIntentIssue,
    GapResearchIntentIssueCode,
    deterministic_intent_ref,
    deterministic_projection_ref,
)
from app.domain.proposition_research_gap import (
    PROPOSITION_RESEARCH_GAP_SCHEMA_VERSION,
    DesiredEvidenceRelation,
    GapSearchability,
    PropositionGapLifecycle,
    PropositionResearchGap,
    PropositionResearchGapType,
)
from app.domain.research_context import ResearchContext


@dataclass(frozen=True, slots=True)
class GapResearchIntentResult:
    intents: tuple[GapResearchIntent, ...]
    issues: tuple[GapResearchIntentIssue, ...]


@dataclass(frozen=True, slots=True)
class GapIntentPlanningResult:
    projection: GapIntentPlanningProjection | None
    research_context: ResearchContext | None
    issues: tuple[GapResearchIntentIssue, ...]


class GapResearchIntentService:
    """Consume Step 1 Gap objects; never assesses truth or executes research."""

    def derive(self, gaps: Iterable[object]) -> GapResearchIntentResult:
        by_ref: dict[str, GapResearchIntent] = {}
        conflicted_refs: set[str] = set()
        issues: list[GapResearchIntentIssue] = []

        for raw_gap in gaps:
            try:
                gap = self._validate_gap(raw_gap)
                if gap is None:
                    continue
                intent = self._intent_for(gap)
            except _GapIntentBuildError as exc:
                issues.append(exc.issue)
                continue
            except (TypeError, ValueError, ValidationError):
                issues.append(
                    GapResearchIntentIssue(
                        gap_ref=self._safe_gap_ref(raw_gap),
                        issue_code=GapResearchIntentIssueCode.INVALID_GAP,
                        detail_code="GAP_VALIDATION_FAILED",
                    )
                )
                continue

            previous = by_ref.get(intent.intent_ref)
            if previous is None and intent.intent_ref not in conflicted_refs:
                by_ref[intent.intent_ref] = intent
            elif previous != intent:
                by_ref.pop(intent.intent_ref, None)
                conflicted_refs.add(intent.intent_ref)
                issues.append(
                    GapResearchIntentIssue(
                        gap_ref=intent.gap_ref,
                        issue_code=GapResearchIntentIssueCode.INTENT_IDENTITY_CONFLICT,
                        detail_code="SAME_GAP_ATTEMPT_HAS_DIFFERENT_SEMANTICS",
                    )
                )

        intents = tuple(
            sorted(by_ref.values(), key=lambda item: item.priority_ordering_key)
        )
        return GapResearchIntentResult(
            intents=intents,
            issues=tuple(sorted(set(issues), key=_issue_key)),
        )

    def project(
        self,
        intents: Iterable[object],
        *,
        legacy_context: ResearchContext | None = None,
    ) -> GapIntentPlanningResult:
        valid_by_ref: dict[str, GapResearchIntent] = {}
        issues: list[GapResearchIntentIssue] = []
        conflicted_refs: set[str] = set()
        for raw_intent in intents:
            try:
                intent = GapResearchIntent.model_validate(
                    raw_intent.model_dump(mode="python")
                    if isinstance(raw_intent, GapResearchIntent)
                    else raw_intent
                )
                previous = valid_by_ref.get(intent.intent_ref)
                if previous is not None and previous != intent:
                    valid_by_ref.pop(intent.intent_ref, None)
                    conflicted_refs.add(intent.intent_ref)
                    issues.append(
                        GapResearchIntentIssue(
                            gap_ref=intent.gap_ref,
                            issue_code=GapResearchIntentIssueCode.INTENT_IDENTITY_CONFLICT,
                            detail_code="DUPLICATE_PROJECTION_INTENT_CONFLICT",
                        )
                    )
                elif intent.intent_ref not in conflicted_refs:
                    valid_by_ref[intent.intent_ref] = intent
            except (TypeError, ValueError, ValidationError):
                issues.append(
                    GapResearchIntentIssue(
                        issue_code=GapResearchIntentIssueCode.PROJECTION_FAILED,
                        detail_code="INTENT_VALIDATION_FAILED",
                    )
                )

        ordered = tuple(
            sorted(valid_by_ref.values(), key=lambda item: item.priority_ordering_key)
        )
        context = _context_with_intents(legacy_context, ordered)
        if not ordered:
            return GapIntentPlanningResult(
                projection=None,
                research_context=context if legacy_context is not None else None,
                issues=tuple(sorted(set(issues), key=_issue_key)),
            )

        legacy_ref = _hash(legacy_context.as_dict()) if legacy_context is not None else None
        refs = tuple(item.intent_ref for item in ordered)
        projection = GapIntentPlanningProjection(
            projection_ref=deterministic_projection_ref(
                intent_refs=refs,
                legacy_context_ref=legacy_ref,
            ),
            intents=ordered,
            legacy_context_ref=legacy_ref,
        )
        return GapIntentPlanningResult(
            projection=projection,
            research_context=context,
            issues=tuple(sorted(set(issues), key=_issue_key)),
        )

    def _validate_gap(self, raw_gap: object) -> PropositionResearchGap | None:
        data: object = (
            raw_gap.model_dump(mode="python")
            if isinstance(raw_gap, PropositionResearchGap)
            else raw_gap
        )
        gap = PropositionResearchGap.model_validate(data)
        if gap.schema_version != PROPOSITION_RESEARCH_GAP_SCHEMA_VERSION:
            raise _GapIntentBuildError(
                GapResearchIntentIssue(
                    gap_ref=self._safe_gap_ref(gap),
                    issue_code=GapResearchIntentIssueCode.UNKNOWN_GAP_VERSION,
                    detail_code="PROPOSITION_RESEARCH_GAP_SCHEMA_VERSION_UNKNOWN",
                )
            )
        if gap.searchability is not GapSearchability.SEARCHABLE:
            return None
        if gap.lifecycle_state is not PropositionGapLifecycle.OPEN:
            return None
        if gap.gap_type in {
            PropositionResearchGapType.SEMANTIC_CAPABILITY_BLOCKED,
            PropositionResearchGapType.SEMANTIC_WORK_NOT_PROCESSED,
        }:
            raise _GapIntentBuildError(
                GapResearchIntentIssue(
                    gap_ref=gap.gap_ref,
                    issue_code=GapResearchIntentIssueCode.UNSUPPORTED_GAP_TYPE,
                    detail_code="NON_SEARCHABLE_GAP_TYPE_MARKED_SEARCHABLE",
                )
            )
        return gap

    def _intent_for(self, gap: PropositionResearchGap) -> GapResearchIntent:
        purpose = _purpose_for_gap(gap.gap_type)
        if purpose is None:
            raise _GapIntentBuildError(
                GapResearchIntentIssue(
                    gap_ref=gap.gap_ref,
                    issue_code=GapResearchIntentIssueCode.UNSUPPORTED_GAP_TYPE,
                    detail_code="UNSUPPORTED_PROPOSITION_GAP_TYPE",
                )
            )

        required = gap.required_evidence
        relations = set(required.desired_relations)
        if gap.gap_type in {
            PropositionResearchGapType.MISSING_SUPPORT,
            PropositionResearchGapType.MISSING_INDEPENDENT_CORROBORATION,
        } and relations != {DesiredEvidenceRelation.SUPPORTS}:
            raise self._characteristic_issue(gap, "SUPPORT_INTENT_RELATION_MISMATCH")
        if gap.gap_type is PropositionResearchGapType.MISSING_INDEPENDENT_CORROBORATION and (
            not required.required_owner_independence or not required.excluded_owner_keys
        ):
            raise self._characteristic_issue(gap, "OWNER_EXCLUSION_CONSTRAINT_MISSING")
        if gap.gap_type is PropositionResearchGapType.CONFLICT_RESOLUTION and (
            not {
                DesiredEvidenceRelation.SUPPORTS,
                DesiredEvidenceRelation.CONTRADICTS,
                DesiredEvidenceRelation.DISAMBIGUATES,
            }.issubset(relations)
            or not (
                gap.existing_conflict_refs
                or required.conflict_group_refs
                or required.a2_conflict_pair_refs
            )
            or not (
                required.support_side_evidence_refs
                and required.contradict_side_evidence_refs
            )
        ):
            raise self._characteristic_issue(gap, "FORMAL_CONFLICT_SIDES_OR_REFS_MISSING")
        if gap.gap_type is PropositionResearchGapType.CONTEXT_DISAMBIGUATION and (
            DesiredEvidenceRelation.DISAMBIGUATES not in relations
            or not (
                required.required_context_refs
                or required.required_time_version_scope_condition_refs
                or required.a2_conflict_pair_refs
            )
        ):
            raise self._characteristic_issue(gap, "TYPED_CONTEXT_REQUIREMENT_MISSING")
        if gap.gap_type is PropositionResearchGapType.SCOPE_COMPLETION and (
            not gap.objective_scope_ref
        ):
            raise self._characteristic_issue(gap, "FORMAL_SCOPE_OBJECTIVE_MISSING")

        return GapResearchIntent(
            intent_ref=deterministic_intent_ref(
                gap_ref=gap.gap_ref,
                attempt_index=gap.attempt_count,
            ),
            run_ref=gap.run_ref,
            gap_ref=gap.gap_ref,
            target_group_ref=gap.target_group_ref,
            target_proposition_refs=gap.target_proposition_refs,
            parent_claim_refs=gap.parent_claim_refs,
            objective_scope_ref=gap.objective_scope_ref,
            gap_type=gap.gap_type,
            desired_evidence_purpose=purpose,
            required_evidence=required,
            existing_evidence_refs=gap.existing_evidence_refs,
            existing_owner_refs=gap.existing_owner_refs,
            existing_conflict_refs=gap.existing_conflict_refs,
            gap_reason_codes=gap.reason_codes,
            source_reliability_refs=gap.source_reliability_refs,
            source_reliability_reason_codes=gap.source_reliability_reason_codes,
            attempt_index=gap.attempt_count,
            priority_inputs=gap.priority_inputs,
            budget_state_ref=gap.budget_state_ref,
            budget_state=(
                tuple(sorted(gap.budget_state.items()))
                if gap.budget_state is not None
                else None
            ),
        )

    @staticmethod
    def _characteristic_issue(
        gap: PropositionResearchGap, detail: str
    ) -> _GapIntentBuildError:
        return _GapIntentBuildError(
            GapResearchIntentIssue(
                gap_ref=gap.gap_ref,
                issue_code=GapResearchIntentIssueCode.INCONSISTENT_REQUIRED_CHARACTERISTICS,
                detail_code=detail,
            )
        )

    @staticmethod
    def _safe_gap_ref(value: object) -> str | None:
        candidate = getattr(value, "gap_ref", None)
        return candidate if isinstance(candidate, str) and len(candidate) == 64 else None


def _context_with_intents(
    legacy_context: ResearchContext | None,
    intents: tuple[GapResearchIntent, ...],
) -> ResearchContext:
    base = legacy_context or ResearchContext()
    return ResearchContext(
        evidence_failure_reason=base.evidence_failure_reason,
        missing_evidence_type=base.missing_evidence_type,
        refined_need_type=base.refined_need_type,
        query_hints=base.query_hints,
        existing_source_owners=base.existing_source_owners,
        existing_domains=base.existing_domains,
        required_source_count=base.required_source_count,
        missing_source_count=base.missing_source_count,
        target_claim_id=base.target_claim_id,
        proposition_gap_intents=intents,
    )


class _GapIntentBuildError(Exception):
    def __init__(self, issue: GapResearchIntentIssue) -> None:
        self.issue = issue
        super().__init__(issue.detail_code)


def _purpose_for_gap(
    gap_type: PropositionResearchGapType,
) -> DesiredEvidencePurpose | None:
    return {
        PropositionResearchGapType.MISSING_SUPPORT: DesiredEvidencePurpose.SUPPORT,
        PropositionResearchGapType.MISSING_INDEPENDENT_CORROBORATION:
            DesiredEvidencePurpose.INDEPENDENT_CORROBORATION,
        PropositionResearchGapType.CONFLICT_RESOLUTION:
            DesiredEvidencePurpose.CONFLICT_DISAMBIGUATION,
        PropositionResearchGapType.CONTEXT_DISAMBIGUATION:
            DesiredEvidencePurpose.CONTEXT_RESOLUTION,
        PropositionResearchGapType.SCOPE_COMPLETION: DesiredEvidencePurpose.SCOPE_EXTENSION,
        PropositionResearchGapType.SEMANTIC_CAPABILITY_BLOCKED: None,
        PropositionResearchGapType.SEMANTIC_WORK_NOT_PROCESSED: None,
    }[gap_type]


def _hash(value: object) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _issue_key(issue: GapResearchIntentIssue) -> tuple[str, str, str]:
    return issue.gap_ref or "", issue.issue_code.value, issue.detail_code
