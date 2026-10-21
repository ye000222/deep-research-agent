"""Version-neutral boundary and V2.3 deterministic-rebuild adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, Protocol
from uuid import UUID

from app.domain.run_state import (
    EvidenceUniverse,
    SemanticExecutionStatus,
    SemanticProductBinding,
    SemanticStateSnapshot,
    canonical_hash,
)

SEMANTIC_ENGINE_V23 = "v2.3-deterministic-rebuild-v1"
_BUNDLE_FIELDS = {
    "claims": "claim_refs",
    "propositions": "proposition_refs",
    "formation_assessments": "formation_assessments",
    "proposition_groups": "proposition_groups",
    "pair_assessments": "pair_assessments",
    "proposition_evidence_links": "proposition_evidence_links",
    "corroboration_assessments": "corroboration_assessments",
    "conflicts": "conflicts",
    "verification_assessments": "verification_assessments",
    "conflict_scope_completeness": "conflict_scope_completeness",
}


def _string_refs(value: object, *, code: str) -> set[str]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise ValueError(code)
    if len(set(value)) != len(value):
        raise ValueError(code)
    return set(value)


class V23DeterministicSemanticEngine:
    """Wrap the already-computed V2.3 product bundle; never recompute semantics."""

    REQUIRED_PRODUCTS: tuple[tuple[str, str], ...] = (
        ("claims", "semantic-shadow-verification-snapshot-v2"),
        ("propositions", "atomic-proposition-schema-v3"),
        ("formation_assessments", "target-proposition-formation-response-v1"),
        ("proposition_groups", "proposition-support-group-v1"),
        ("pair_assessments", "proposition-pair-identity-runtime-v1"),
        ("proposition_evidence_links", "proposition-evidence-direction-v1"),
        ("corroboration_assessments", "independent-corroboration-v1"),
        ("conflicts", "proposition-conflict-assessment-v1"),
        ("verification_assessments", "proposition-verification-assessment-v1"),
        ("conflict_scope_completeness", "conflict-scope-completeness-v1"),
    )
    _VERSION_FIELDS: ClassVar[dict[str, str]] = {
        "claims": "semantic_snapshot",
        "propositions": "atomic_proposition",
        "formation_assessments": "target_formation_schema",
        "proposition_groups": "proposition_groups",
        "pair_assessments": "proposition_pair_identity",
        "proposition_evidence_links": "direction",
        "corroboration_assessments": "corroboration",
        "conflicts": "conflicts",
        "verification_assessments": "verification",
        "conflict_scope_completeness": "conflict_scope_completeness",
    }

    async def evaluate(
        self,
        *,
        run_ref: UUID,
        evidence_universe: EvidenceUniverse,
        execution_context: Mapping[str, object],
        previous_semantic_state_ref: str | None = None,
    ) -> SemanticStateSnapshot:
        del previous_semantic_state_ref
        if evidence_universe.run_ref != run_ref:
            raise ValueError("semantic_engine_universe_run_mismatch")
        bundle = execution_context.get("semantic_products")
        bundle_ref = execution_context.get("product_bundle_ref")
        if not isinstance(bundle, Mapping) or not isinstance(bundle_ref, str):
            raise ValueError("semantic_engine_projection_missing")
        if bundle.get("run_ref") != str(run_ref):
            raise ValueError("semantic_engine_projection_run_mismatch")
        if bundle.get("schema_version") != "semantic-shadow-verification-snapshot-v2":
            raise ValueError("semantic_engine_product_bundle_schema_mismatch")

        schema_versions = bundle.get("schema_versions")
        if not isinstance(schema_versions, Mapping):
            raise ValueError("semantic_schema_versions_missing")
        required_versions = tuple(sorted(self.REQUIRED_PRODUCTS))
        schema_missing = [
            kind
            for kind, version in required_versions
            if schema_versions.get(self._VERSION_FIELDS[kind]) != version
        ]
        expected_claims = {
            str(item.claim_ref)
            for item in evidence_universe.entries
            if item.accepted and item.claim_ref is not None
        }
        expected_evidence = {str(item) for item in evidence_universe.semantic_evidence_refs}
        actual_claims = _string_refs(bundle.get("claim_refs"), code="semantic_claim_refs_invalid")
        actual_evidence = _string_refs(
            bundle.get("evidence_refs"), code="semantic_evidence_refs_invalid"
        )
        missing: list[str] = []
        missing.extend(f"schema:{kind}" for kind in schema_missing)
        deferred: set[str] = set()
        bindings: list[SemanticProductBinding] = []
        required_refs: list[tuple[str, str, str]] = []
        output_refs: list[tuple[str, str, str]] = []
        for kind, version in required_versions:
            value = bundle.get(_BUNDLE_FIELDS[kind])
            if value is None:
                missing.append(kind)
                required_refs.append((kind, f"missing:{kind}", canonical_hash({"missing": kind})))
                continue
            digest = canonical_hash(value)
            product_ref = f"{bundle_ref}#{kind}"
            required_refs.append((kind, product_ref, digest))
            if kind in schema_missing or not _valid_product_shape(kind, value):
                if not _valid_product_shape(kind, value):
                    missing.append(f"product:{kind}:invalid_shape")
                continue
            output_refs.append((kind, product_ref, digest))
            bindings.append(
                SemanticProductBinding(
                    product_kind=kind,
                    product_ref=product_ref,
                    content_hash=digest,
                    product_version=version,
                    run_ref=run_ref,
                    evidence_universe_hash=evidence_universe.evidence_universe_hash,
                    semantic_execution_ref="pending",
                    semantic_engine_version=SEMANTIC_ENGINE_V23,
                )
            )

        if actual_claims != expected_claims:
            missing.append("claim_universe_coverage")
        if actual_evidence != expected_evidence:
            missing.append("evidence_universe_coverage")
        # C1/C2 and conflict-scope products must be present as formal outputs;
        # UNKNOWN outcomes inside those products remain valid semantic results.
        if not isinstance(bundle.get("conflicts"), Mapping):
            missing.append("conflicts")
        if not isinstance(bundle.get("verification_assessments"), (tuple, list)):
            missing.append("verification_assessments")
        if not isinstance(bundle.get("conflict_scope_completeness"), Mapping):
            missing.append("conflict_scope_completeness")
        for kind in (
            "formation_assessments",
            "proposition_groups",
            "pair_assessments",
            "proposition_evidence_links",
            "corroboration_assessments",
        ):
            if not isinstance(bundle.get(_BUNDLE_FIELDS[kind]), (tuple, list)):
                missing.append(kind)

        if _contains_value(bundle, "NOT_PROCESSED_BUDGET"):
            deferred.add("NOT_PROCESSED_BUDGET")
            missing.append("required_work_not_processed_budget")

        diagnostics = bundle.get("diagnostics", ())
        if not isinstance(diagnostics, (tuple, list)):
            missing.append("diagnostics_invalid")
            diagnostics = ()
        diagnostic_codes = tuple(
            sorted(
                {
                    str(item.get("reason_code"))
                    for item in diagnostics
                    if isinstance(item, Mapping) and isinstance(item.get("reason_code"), str)
                }
            )
        )
        diagnostics_hash = canonical_hash(
            sorted(diagnostics, key=canonical_hash)
        )
        budget_state_hash = canonical_hash(bundle.get("budget_state", {}))

        input_hash = canonical_hash(
            {
                "run_ref": str(run_ref),
                "evidence_universe_hash": evidence_universe.evidence_universe_hash,
                "semantic_evidence_hash": evidence_universe.semantic_evidence_hash,
                "semantic_evidence_refs": [
                    str(item) for item in evidence_universe.semantic_evidence_refs
                ],
                "evidence_entries": [
                    item.model_dump(mode="json")
                    for item in evidence_universe.entries
                    if item.semantic_eligible
                ],
                "required_product_versions": required_versions,
            }
        )
        status = (
            SemanticExecutionStatus.INCOMPLETE
            if missing
            else SemanticExecutionStatus.COMPLETE
        )
        limitations = tuple(sorted({f"MISSING_OR_UNBOUND:{item}" for item in missing}))
        execution_ref = SemanticStateSnapshot.deterministic_execution_ref(
            run_ref=run_ref,
            semantic_engine_version=SEMANTIC_ENGINE_V23,
            evidence_universe_hash=evidence_universe.evidence_universe_hash,
            semantic_evidence_hash=evidence_universe.semantic_evidence_hash,
            required_product_versions=required_versions,
        )
        bound = tuple(
            sorted(
                (
                    item.model_copy(update={"semantic_execution_ref": execution_ref})
                    for item in bindings
                ),
                key=lambda item: (item.product_kind, item.product_ref, item.content_hash),
            )
        )
        output_hash = canonical_hash([item.model_dump(mode="json") for item in bound])
        statuses = tuple(
            sorted(
                (
                    kind,
                    "AVAILABLE"
                    if (
                        kind not in missing
                        and f"schema:{kind}" not in missing
                        and f"product:{kind}:invalid_shape" not in missing
                    )
                    else "MISSING",
                )
                for kind, _ in required_versions
            )
        )
        return SemanticStateSnapshot.build(
            run_ref=run_ref,
            semantic_execution_ref=execution_ref,
            semantic_engine_version=SEMANTIC_ENGINE_V23,
            evidence_universe_hash=evidence_universe.evidence_universe_hash,
            semantic_evidence_hash=evidence_universe.semantic_evidence_hash,
            semantic_input_hash=input_hash,
            semantic_output_hash=output_hash,
            budget_state_hash=budget_state_hash,
            diagnostics_hash=diagnostics_hash,
            diagnostic_codes=diagnostic_codes,
            semantic_evidence_refs=evidence_universe.semantic_evidence_refs,
            required_product_versions=required_versions,
            required_product_refs=tuple(sorted(required_refs)),
            output_product_refs=tuple(sorted(output_refs)),
            output_product_bindings=bound,
            semantic_statuses=statuses,
            known_limitations=limitations,
            capability_states=("V2_3_DETERMINISTIC_REBUILD",),
            deferred_states=tuple(deferred),
            provenance_refs=(bundle_ref,),
            execution_status=status,
            failure_codes=(),
        )

    def failed_snapshot(
        self,
        *,
        run_ref: UUID,
        evidence_universe: EvidenceUniverse,
        failure_code: str = "SEMANTIC_REBUILD_FAILED",
    ) -> SemanticStateSnapshot:
        """Represent a failed rebuild without publishing any semantic products."""
        if evidence_universe.run_ref != run_ref:
            raise ValueError("semantic_engine_universe_run_mismatch")
        required_versions = tuple(sorted(self.REQUIRED_PRODUCTS))
        execution_ref = SemanticStateSnapshot.deterministic_execution_ref(
            run_ref=run_ref,
            semantic_engine_version=SEMANTIC_ENGINE_V23,
            evidence_universe_hash=evidence_universe.evidence_universe_hash,
            semantic_evidence_hash=evidence_universe.semantic_evidence_hash,
            required_product_versions=required_versions,
        )
        failed_refs = tuple(
            sorted(
                (
                    kind,
                    f"failed:{kind}",
                    canonical_hash({"failed": kind, "failure_code": failure_code}),
                )
                for kind, _ in required_versions
            )
        )
        return SemanticStateSnapshot.build(
            run_ref=run_ref,
            semantic_execution_ref=execution_ref,
            semantic_engine_version=SEMANTIC_ENGINE_V23,
            evidence_universe_hash=evidence_universe.evidence_universe_hash,
            semantic_evidence_hash=evidence_universe.semantic_evidence_hash,
            semantic_input_hash=canonical_hash(
                {
                    "run_ref": str(run_ref),
                    "evidence_universe_hash": evidence_universe.evidence_universe_hash,
                    "semantic_evidence_hash": evidence_universe.semantic_evidence_hash,
                    "semantic_evidence_refs": [
                        str(item) for item in evidence_universe.semantic_evidence_refs
                    ],
                    "evidence_entries": [
                        item.model_dump(mode="json")
                        for item in evidence_universe.entries
                        if item.semantic_eligible
                    ],
                    "required_product_versions": required_versions,
                }
            ),
            semantic_output_hash=canonical_hash([]),
            semantic_evidence_refs=evidence_universe.semantic_evidence_refs,
            required_product_versions=required_versions,
            required_product_refs=failed_refs,
            output_product_refs=(),
            output_product_bindings=(),
            semantic_statuses=tuple(
                sorted((kind, "FAILED") for kind, _ in required_versions)
            ),
            known_limitations=("SEMANTIC_REBUILD_FAILED",),
            capability_states=("V2_3_DETERMINISTIC_REBUILD",),
            provenance_refs=(),
            execution_status=SemanticExecutionStatus.FAILED,
            failure_codes=(failure_code,),
        )


class SemanticEngine(Protocol):
    """Produce a fully bound semantic envelope; callers do not assemble V2 products."""

    async def evaluate(
        self,
        *,
        run_ref: UUID,
        evidence_universe: EvidenceUniverse,
        execution_context: Mapping[str, object],
        previous_semantic_state_ref: str | None = None,
    ) -> SemanticStateSnapshot:
        """Rebuild from durable inputs and report explicit completion status."""


def _contains_value(value: object, target: str) -> bool:
    if value == target:
        return True
    if isinstance(value, Mapping):
        return any(_contains_value(item, target) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_value(item, target) for item in value)
    return False


def _valid_product_shape(kind: str, value: object) -> bool:
    if kind in {"conflicts", "conflict_scope_completeness"}:
        return isinstance(value, Mapping)
    if kind in {"claims", "propositions"}:
        return isinstance(value, (tuple, list)) and all(isinstance(item, str) for item in value)
    return isinstance(value, (tuple, list))
