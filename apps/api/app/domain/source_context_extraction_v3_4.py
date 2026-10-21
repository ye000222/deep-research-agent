"""Versioned v3.4 source response carrying typed value-component bindings.

The v3.3 domain and wire models remain unchanged. Existing v3.3 payloads are
not upgraded by this contract.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from app.domain.source_atomic_assertion import (
    SOURCE_SEMANTIC_RUNTIME_RESPONSE_SCHEMA_VERSION,
    SourceSemanticRuntimeResponseV1,
)
from app.domain.source_context_extraction_v3_3 import (
    SOURCE_CONTEXT_EXTRACTION_SCHEMA_V33,
    SourceExtractionResponseV33,
)
from app.domain.source_typed_value_grounding import TypedValueEvidenceBinding

SOURCE_CONTEXT_EXTRACTION_SCHEMA_V34: Literal["source-context-extraction-schema-v3.4"] = (
    "source-context-extraction-schema-v3.4"
)
SOURCE_CONTEXT_EXTRACTION_DOMAIN_V34: Literal["source-context-extraction-domain-v3.4"] = (
    "source-context-extraction-domain-v3.4"
)
SOURCE_CONTEXT_EXTRACTION_PROMPT_V34: Literal["source-context-extraction.generic.v3.4"] = (
    "source-context-extraction.generic.v3.4"
)
SOURCE_SEMANTIC_RUNTIME_RESPONSE_V2: Literal["source-semantic-runtime-response-v2"] = (
    "source-semantic-runtime-response-v2"
)
SOURCE_SEMANTIC_RUNTIME_PROMPT_V2: Literal["source-semantic-runtime.generic.v2"] = (
    "source-semantic-runtime.generic.v2"
)
SOURCE_CONTEXT_EXTRACTION_INSTRUCTIONS_V34 = """Extract only source-grounded assertions under
source-context-extraction-schema-v3.4.
Preserve the v3.3 assertion, context, relation, polarity, qualifier, and provenance
semantics. For QUANTITY scalar values, bind NUMERIC_MAGNITUDE and every present
NUMERIC_OPERATOR, UNIT, or CURRENCY component. For RANGE values, bind both
LOWER_BOUND and UPPER_BOUND; the exact span must also encode the range operator.
Each typed_value_binding must point to one supplied evidence_ref and include its
occurrence_index and absolute source char_start/char_end, assertion_ref, VALUE role,
source_ref, snapshot_ref, and contract_version. Offsets are in canonical source
coordinates, not context-window coordinates. Bind each component to the exact span
that contains it; repeated text requires the correct explicit occurrence_index.
These are evidence proposals, not grounding verdicts. Never emit a grounding status.
Do not search the whole source for numbers, infer implicit units/currency, convert
units or currencies, infer values from context, or fill missing components. Leave
unsupported, ambiguous, or unbound values unresolved. In particular, three-letter
uppercase suffixes are ambiguous between units and currencies and remain unresolved.
Do not emit chain-of-thought."""


class SourceExtractionResponseV34(SourceExtractionResponseV33):
    """v3.3 extraction fields plus explicit assertion-value component bindings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Any = Field(
        default=SOURCE_CONTEXT_EXTRACTION_SCHEMA_V34,
        json_schema_extra={"const": SOURCE_CONTEXT_EXTRACTION_SCHEMA_V34},
    )
    typed_value_bindings: tuple[TypedValueEvidenceBinding, ...] = Field(
        default=(), max_length=1_024
    )

    @field_validator("schema_version")
    @classmethod
    def exact_v34_schema_version(cls, value: Any) -> str:
        if value != SOURCE_CONTEXT_EXTRACTION_SCHEMA_V34:
            raise ValueError("source_extraction_schema_version_mismatch")
        return str(value)

    @model_validator(mode="after")
    def binding_refs_are_unique(self) -> SourceExtractionResponseV34:
        refs = tuple(item.binding_ref for item in self.typed_value_bindings)
        if len(set(refs)) != len(refs):
            raise ValueError("typed_value_binding_refs_must_be_unique")
        return self


class SourceSemanticRuntimeResponseV2(SourceSemanticRuntimeResponseV1):
    """One-call runtime envelope bound to the versioned v3.4 source contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Any = Field(
        default=SOURCE_SEMANTIC_RUNTIME_RESPONSE_V2,
        json_schema_extra={"const": SOURCE_SEMANTIC_RUNTIME_RESPONSE_V2},
    )
    source_extraction: SourceExtractionResponseV34

    @field_validator("schema_version")
    @classmethod
    def exact_v2_response_version(cls, value: Any) -> str:
        if value != SOURCE_SEMANTIC_RUNTIME_RESPONSE_V2:
            raise ValueError("runtime_response_schema_version_mismatch")
        return str(value)


def source_extraction_v34_to_v33_payload(
    response: SourceExtractionResponseV34,
) -> dict[str, object]:
    """Create an explicit legacy-shaped payload only for unchanged v3.3 assembly."""

    payload = response.model_dump(
        mode="python", exclude={"typed_value_bindings"}, exclude_unset=True
    )
    payload["schema_version"] = SOURCE_CONTEXT_EXTRACTION_SCHEMA_V33
    return payload


def v2_response_to_v1_payload(response: SourceSemanticRuntimeResponseV2) -> dict[str, object]:
    """Prepare the frozen v3.3 assembler input; v3.4 bindings stay sidecar data."""

    payload = response.model_dump(mode="python", exclude={"schema_version"}, exclude_unset=True)
    payload["schema_version"] = SOURCE_SEMANTIC_RUNTIME_RESPONSE_SCHEMA_VERSION
    payload["source_extraction"] = source_extraction_v34_to_v33_payload(response.source_extraction)
    return payload
