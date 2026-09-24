# ADR 0004: V2 Claim as a First-Class, Evidence-Grounded Contract

- Status: Accepted for V2.0 design; passive contracts only
- Date: 2026-09-24
- Scope: V2.0 Claim architecture and baseline

## Context

V1.1 persists extracted claim text, a Claim row, an optional Evidence→Claim FK, source owner identity, quotes/chunks, graph edges and conflicts. It does not persist structured qualifiers or a canonical per-Claim verification result. Dimension-level `GapRequirement` verification and graph Claim status are different projections. Historical qualification artifacts do not contain the raw rows needed to reconstruct the missing per-Claim funnel.

## Decisions

1. **Claim is first-class** because verification, corroboration, contradiction and provenance must be attributable to one proposition rather than inferred from an aggregate dimension.
2. **Claim != Dimension.** A dimension is a coverage/closure scope; one dimension may contain multiple Claims and closing it does not verify each Claim.
3. **Evidence count != corroboration.** Corroboration requires explicit accepted links and distinct canonical `source_owner_key` values for the same Claim.
4. **Source Role != Source Identity.** Roles classify source type; owner identity is derived from the existing source identity mechanism. No competing owner system is introduced.
5. **Coverage-v1 remains unchanged.** Claim metrics use the independent definition `claim-v2-baseline-v1` to preserve longitudinal V1 comparability.
6. **Semantic equivalence is deferred to V2.1.** V2.0 provides qualifier-preserving records and only deterministic structured candidates; it does not ship embeddings, LLM matching or semantic reranking.
7. **Full verification is deferred to V2.2.** V2.0 defines the minimal five-state contract but produces no verification decisions. `UNKNOWN` is not false.
8. **No V2.0 migration or runtime integration.** Existing tables are enough to inventory the old funnel; future many-to-many ClaimEvidence links or verification history require a separately reviewed migration after semantics are stable.
9. **No Research behavior change.** Contracts, fixtures, diagnostics and offline tooling do not enter Planner/Search/Acceptance/Closure/Report paths.

## Consequences

- Historical `verified_claims` zeroes cannot be interpreted as actual per-Claim verdict counts.
- V1.1 legacy `supported` means accepted evidence with at least two owner keys in the graph helper; it is not the V2 `VERIFIED` state.
- Missing historical Claim-level data is reported as `NOT_RECONSTRUCTABLE`, never imputed as 0.
- V2.1 receives two CanonicalClaim records including original text, structured fields and qualifiers; it returns `SAME | RELATED | CONTRADICTS | UNKNOWN` with rationale/confidence and no execution side effects.
- Later persistence may use a many-to-many link and versioned verification results, but no backfill is required to run V1.1.

## Rejected alternatives

- Reusing dimension closure as Claim verification (false inference).
- Treating two evidence rows or URLs as independent corroboration (identity error).
- Renaming legacy `supported` to `verified` (semantic overclaim).
- Adding four new tables immediately (unproven schema duplication and migration surface).
- Triggering a live run to fill missing historic raw data (not necessary for a deterministic architecture baseline; explicitly out of scope).
