"""Small, shadow-only value objects for qualifier preservation evidence.

These objects deliberately describe source-to-output preservation, not proposition
identity, matching, or verification confidence.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class PreservationEvidenceStrength(StrEnum):
    NONE = "NONE"
    WEAK = "WEAK"
    MEDIUM = "MEDIUM"
    STRONG = "STRONG"


class SourceApplicability(StrEnum):
    SUPPORTED_SIGNAL = "SUPPORTED_SIGNAL"
    NO_SUPPORTED_SIGNAL = "NO_SUPPORTED_SIGNAL"
    DETECTOR_COVERAGE_LIMIT = "DETECTOR_COVERAGE_LIMIT"


class QualifierObligation(BaseModel):
    """A material source-side signal the decomposition must preserve."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: str
    normalized_value: str
    source_text: str
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)
    attachment_context: str = ""
    materiality: str = "material"
    detection_evidence: str
    applicability: SourceApplicability = SourceApplicability.SUPPORTED_SIGNAL


class SemanticCarrier(BaseModel):
    """One deterministic carrier of semantics in a proposition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: str
    semantic_role: str
    field_path: str
    value: str
    normalized_value: str
    evidence_strength: PreservationEvidenceStrength
    carrier_kind: str = "structured"
    proposition_ref: str | None = None
    related_roles: tuple[str, ...] = ()
    source_span: str | None = None


class PropositionSemanticView(BaseModel):
    """Canonical preservation-only projection of a proposition's valid fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proposition_index: int = Field(ge=0)
    proposition_ref: str | None = None
    carriers: tuple[SemanticCarrier, ...] = ()
    source_spans: tuple[str, ...] = ()


class PreservationEvidence(BaseModel):
    """Auditable result linking one source obligation to one proposition view."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    obligation: QualifierObligation
    proposition_index: int | None = None
    evidence_strength: PreservationEvidenceStrength = PreservationEvidenceStrength.NONE
    semantic_compatible: bool = False
    attachment_compatible: bool = False
    carrier_paths: tuple[str, ...] = ()
    reason_code: str
