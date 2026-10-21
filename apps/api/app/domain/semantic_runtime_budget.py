"""Explicit immutable work limits for the non-persistent semantic bridge."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_COUNTERS = frozenset(
    {
        "provider_ap_decomposition_calls",
        "provider_source_semantic_extraction_calls",
        "provider_identity_resolution_calls",
        "provider_formation_validation_calls",
        "evidence_inputs",
        "propositions",
        "source_assertions",
        "candidate_pairs",
        "direction_assessments",
    }
)
_LIMIT_FIELDS = {
    "provider_ap_decomposition_calls": "max_ap_decomposition_calls",
    "provider_source_semantic_extraction_calls": "max_source_semantic_extraction_calls",
    "provider_identity_resolution_calls": "max_identity_resolution_calls",
    "provider_formation_validation_calls": "max_formation_validation_calls",
    "evidence_inputs": "max_evidence_inputs",
    "candidate_pairs": "max_candidate_pairs",
    "direction_assessments": "max_direction_assessments",
}


class SemanticRuntimeBudget(BaseModel):
    """Caller-supplied limits; zero disables that category and has no semantic effect."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_ap_decomposition_calls: int = Field(ge=0)
    max_source_semantic_extraction_calls: int = Field(ge=0)
    max_identity_resolution_calls: int = Field(ge=0)
    max_formation_validation_calls: int = Field(default=0, ge=0)
    max_evidence_inputs: int = Field(ge=0)
    max_propositions_per_evidence: int = Field(ge=0)
    max_source_assertions_per_evidence: int = Field(ge=0)
    max_candidate_pairs: int = Field(ge=0)
    max_direction_assessments: int = Field(ge=0)
    # Explicit caller opt-in: zero disables the new AP/AP stage.
    proposition_pair_batch_size: int = Field(default=0, ge=0, le=64)
    proposition_pair_request_bytes: int = Field(default=0, ge=0)


class SemanticRuntimeBudgetState(BaseModel):
    """Deterministic consumed-work counters, independent of semantic outcomes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_ap_decomposition_calls: int = Field(default=0, ge=0)
    provider_source_semantic_extraction_calls: int = Field(default=0, ge=0)
    provider_identity_resolution_calls: int = Field(default=0, ge=0)
    provider_formation_validation_calls: int = Field(default=0, ge=0)
    evidence_inputs: int = Field(default=0, ge=0)
    propositions: int = Field(default=0, ge=0)
    source_assertions: int = Field(default=0, ge=0)
    candidate_pairs: int = Field(default=0, ge=0)
    direction_assessments: int = Field(default=0, ge=0)

    def consume(
        self,
        budget: SemanticRuntimeBudget,
        counter: str,
        amount: int = 1,
    ) -> tuple[SemanticRuntimeBudgetState, bool]:
        """Consume bounded work or return the unchanged state and ``False``."""

        if amount < 0 or counter not in _COUNTERS:
            raise ValueError("invalid_semantic_runtime_budget_counter")
        limit_field = _LIMIT_FIELDS.get(counter)
        if limit_field is None:
            raise ValueError("counter_has_no_single_budget_limit")
        if getattr(self, counter) + amount > getattr(budget, limit_field):
            return self, False
        return self.model_copy(update={counter: getattr(self, counter) + amount}), True

    def consume_direction_pair(
        self, budget: SemanticRuntimeBudget
    ) -> tuple[SemanticRuntimeBudgetState, bool]:
        """Atomically reserve one candidate comparison and its direction result."""

        if not self.can_consume_direction(budget):
            return self, False
        return (
            self.model_copy(
                update={
                    "candidate_pairs": self.candidate_pairs + 1,
                    "direction_assessments": self.direction_assessments + 1,
                }
            ),
            True,
        )

    def consume_evidence_scope(
        self,
        budget: SemanticRuntimeBudget,
        *,
        propositions: int,
        source_assertions: int,
    ) -> tuple[SemanticRuntimeBudgetState, bool]:
        """Account one explicit scope, enforcing per-evidence structural limits."""

        if propositions < 0 or source_assertions < 0:
            raise ValueError("invalid_semantic_runtime_budget_counter")
        if (
            self.evidence_inputs + 1 > budget.max_evidence_inputs
            or propositions > budget.max_propositions_per_evidence
            or source_assertions > budget.max_source_assertions_per_evidence
        ):
            return self, False
        return (
            self.model_copy(
                update={
                    "evidence_inputs": self.evidence_inputs + 1,
                    "propositions": self.propositions + propositions,
                    "source_assertions": self.source_assertions + source_assertions,
                }
            ),
            True,
        )

    def can_consume_direction(self, budget: SemanticRuntimeBudget) -> bool:
        return (
            self.candidate_pairs < budget.max_candidate_pairs
            and self.direction_assessments < budget.max_direction_assessments
        )
