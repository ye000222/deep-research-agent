"""Phase 14.0-C Claim verification analysis.

Distinguishes the three states the closure loop currently conflates:

1. evidence directly supports the claim (verified, multi-source);
2. evidence exists but only *describes* the claim (relevant, unverified);
3. evidence supports the claim but comes from a single owner and still needs
   independent validation.

Pure domain logic: no database access, no search, no scheduler.  Only generic
requirement state and per-item evidence facts are read, so the verdict never
depends on a question id, gap id, dimension key or benchmark identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from app.domain.evidence_alignment import EvidenceAlignmentStatus
from app.domain.evidence_failure_classification import (
    EvidenceItemFacts,
    MissingEvidenceType,
)
from app.domain.gap_closure import GapRequirement, VerificationStatus


class ClaimSupportLevel(StrEnum):
    DIRECT_SUPPORT = "direct_support"
    RELEVANT_ONLY = "relevant_only"
    SINGLE_SOURCE = "single_source"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ClaimVerificationResult:
    """How strongly the collected evidence backs one claim."""

    claim_id: UUID | None
    verification_status: VerificationStatus
    support_level: ClaimSupportLevel
    missing_support_type: MissingEvidenceType

    def as_dict(self) -> dict[str, object]:
        return {
            "claim_id": str(self.claim_id) if self.claim_id is not None else None,
            "verification_status": self.verification_status.value,
            "support_level": self.support_level.value,
            "missing_support_type": self.missing_support_type.value,
        }


class ClaimVerificationAnalyzer:
    """Classify claim support from generic verification/alignment state."""

    def analyze(
        self,
        requirement: GapRequirement,
        facts: Sequence[EvidenceItemFacts] = (),
        *,
        claim_id: UUID | None = None,
    ) -> ClaimVerificationResult:
        """Return one deterministic support verdict for a requirement's claim."""

        items = tuple(facts)
        verified = requirement.verification_status is VerificationStatus.VERIFIED
        aligned = sum(
            1
            for item in items
            if item.alignment_status is EvidenceAlignmentStatus.ALIGNED
        )
        partial = sum(
            1
            for item in items
            if item.alignment_status is EvidenceAlignmentStatus.PARTIAL
        )
        has_independent_headroom = (
            requirement.current_independent_sources
            >= max(2, requirement.required_independent_sources)
        )

        if not items:
            # Aggregate-only view (per-item facts were never observed): fall
            # back to requirement-level counters so no verdict is fabricated.
            if requirement.current_evidence_count == 0:
                return self._result(
                    requirement,
                    claim_id,
                    ClaimSupportLevel.UNSUPPORTED,
                    MissingEvidenceType.SUPPORTING_EVIDENCE,
                )
            if verified:
                if not has_independent_headroom:
                    return self._result(
                        requirement,
                        claim_id,
                        ClaimSupportLevel.SINGLE_SOURCE,
                        MissingEvidenceType.INDEPENDENT_VALIDATION,
                    )
                return self._result(
                    requirement,
                    claim_id,
                    ClaimSupportLevel.DIRECT_SUPPORT,
                    MissingEvidenceType.UNKNOWN,
                )
            return self._result(
                requirement,
                claim_id,
                ClaimSupportLevel.RELEVANT_ONLY,
                MissingEvidenceType.VERIFICATION_SUPPORT,
            )

        if aligned > 0 and verified:
            # Case 1 vs case 3: direct support still needs independence.
            if not has_independent_headroom:
                return self._result(
                    requirement,
                    claim_id,
                    ClaimSupportLevel.SINGLE_SOURCE,
                    MissingEvidenceType.INDEPENDENT_VALIDATION,
                )
            return self._result(
                requirement,
                claim_id,
                ClaimSupportLevel.DIRECT_SUPPORT,
                MissingEvidenceType.UNKNOWN,
            )
        if aligned > 0 or partial > 0:
            # Case 2: evidence is merely related/descriptive, not proving.
            return self._result(
                requirement,
                claim_id,
                ClaimSupportLevel.RELEVANT_ONLY,
                MissingEvidenceType.VERIFICATION_SUPPORT,
            )
        if requirement.current_evidence_count > 0:
            # Evidence exists for the gap but none aligns with this claim.
            return self._result(
                requirement,
                claim_id,
                ClaimSupportLevel.UNKNOWN,
                MissingEvidenceType.DIMENSION_ALIGNED_EVIDENCE,
            )
        return self._result(
            requirement,
            claim_id,
            ClaimSupportLevel.UNSUPPORTED,
            MissingEvidenceType.SUPPORTING_EVIDENCE,
        )

    def _result(
        self,
        requirement: GapRequirement,
        claim_id: UUID | None,
        level: ClaimSupportLevel,
        missing: MissingEvidenceType,
    ) -> ClaimVerificationResult:
        return ClaimVerificationResult(
            claim_id=claim_id,
            verification_status=requirement.verification_status,
            support_level=level,
            missing_support_type=missing,
        )
