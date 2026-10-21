# ADR 0005: Atomic Proposition as the Verification Unit

- **Status:** Accepted for architecture/shadow design; production integration not authorized
- **Date:** 2026-09-25
- **Decision scope:** V2.1R contract only

## Context

The persisted Claim is a first-class, source-centric and report-friendly research assertion. Historical diagnosis found that Claims are largely formed from source text and do not reliably converge when separate sources express the same assertion. Whole-Claim semantic equivalence is insufficient when one Claim bundles independently testable facts or when two Claims share only a common component. Making Claim itself atomic would erase useful source/context boundaries and conflate database identity with semantic identity.

## Decision

Keep Claim and its identity unchanged. Define Atomic Proposition as a finer, independently verifiable assertion with subject, predicate/relation, typed object/value, polarity, material qualifiers, modality/attribution, and provenance links. Permit one Claim to express one or many propositions and many Claims to link to the same proposition. Keep Evidence-to-proposition relations separate from proposition identity. Use the conservative relation surface SAME, RELATED, CONTRADICTS, UNKNOWN; unknown or ambiguous structures fail closed. Contradiction analysis first requires compatible scope and measurement context. Candidate clustering must require direct pairwise compatibility, not transitive inference.

## Why shadow-first

The historical diagnosis motivates a finer unit but does not demonstrate decomposition quality or reduced false merges. A bounded, blinded, non-persistent shadow sample can test atomicity, qualifier retention, shared proposition observability, and error rates without changing research decisions. Any dual-write or consumer integration requires separate review after the shadow evidence and pre-registered thresholds.

## Why no historical migration now

No database schema, Claim rewrite/merge, ID change, Evidence relink, report recomputation, or historical backfill is justified before the contract and decomposition errors are measured. Claim IDs remain durable database identities. Optional backfill is a future separately approved phase with mapping, provenance, and rollback controls.

## Holdout governance

The existing Holdout is legacy/quarantined for final generalization claims because pristine unseen status cannot be guaranteed. It is not inspected, modified, relabeled, or used for this architecture. No replacement Holdout is created until after implementation, development tuning, and prompt/schema/threshold freeze. This keeps final generalization evaluation independent from design iteration.

## Anti-overfitting requirement

Every rule must have a domain-independent rationale. No benchmark-, question-, entity-, source-, vendor-, model-, or task-family-specific branches, answer-derived templates, or evaluation-derived synonym dictionaries are permitted. Synthetic fixtures test invariants only. Static audit is not an empirical generalization claim.

## Consequences

- Claim remains the durable report/provenance unit; proposition is a proposed verification/corroboration unit.
- Material time, scope, version, unit, condition, polarity, modality, attribution, and numeric semantics cannot be discarded to manufacture identity.
- Ambiguous decomposition is reviewable UNKNOWN/unresolved rather than forced.
- Verification, reporting, and Gap Closure remain unchanged until separately authorized phases.
- This ADR authorizes architecture and shadow design only; it does not declare V2.1, V2.2, or V2 ready.
