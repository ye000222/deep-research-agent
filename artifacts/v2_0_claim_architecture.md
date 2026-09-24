# V2.0 Claim Architecture

## Scope and boundary

V2.0 introduces passive contracts, diagnostics, deterministic semantics fixtures, and historical baseline tooling only. It is not wired into the Research Loop. **Research behavior diff: NONE.** There is no schema migration, no new feature flag, no live Research run, and no Coverage-v1 change.

## CanonicalClaim contract

`CanonicalClaim` in `app.domain.claim_architecture` preserves:

| Field | Meaning |
|---|---|
| `claim_id` | Persistent entity identity; not semantic identity |
| `run_id`, `question_id`, `dimension_key` | Scope/provenance boundary |
| `original_text` | Exact extracted proposition text as received |
| `normalized_text` | Conservative whitespace-normalized proposition; not a semantic rewrite |
| `claim_type` | FACTUAL, QUANTITATIVE, COMPARATIVE, CAUSAL, DESCRIPTIVE, UNKNOWN |
| `subject`, `predicate`, `object_or_value` | Optional structured proposition fields |
| `qualifiers` | Preserved keys including time, date_range, location, unit, version, population_scope, methodology, condition |
| `polarity` | Optional explicit polarity; absence remains unknown |
| `provenance` | Evidence/source/owner/quote/method references |
| `created_at` | Record creation time |

`exact_proposition_key` is a provisional lexical key only. Same `claim_id` does not imply same meaning; matching keys do not establish equivalence beyond the stated lexical normalization.

## Evidence and Source contracts

`ClaimEvidenceLink` distinguishes Evidence→Claim relations (`SUPPORTS`, `CONTRADICTS`, `MENTIONS`, `INSUFFICIENT`) from Claim→Claim relations (`SAME`, `RELATED`, `CONTRADICTS`, `UNKNOWN`). It carries evidence ID, source ID, owner key, confidence, quote reference and method. Scope/source-owner validation fails closed on mismatches.

Source Role remains separate from Source Identity. Role classifies a publisher/page; `source_owner_key` identifies a canonical owner. Independence counts distinct owner keys, not URLs, pages, or roles.

## Claim status vs dimension closure

V2 verification contract: `UNSUPPORTED`, `SUPPORTED`, `VERIFIED`, `CONFLICTED`, `UNKNOWN`.

- UNSUPPORTED: no accepted supporting Evidence link.
- SUPPORTED: accepted support exists, but the explicit verification contract is not met.
- VERIFIED: an explicit versioned Claim-level verification contract is satisfied.
- CONFLICTED: accepted Evidence supports an unresolved direct semantic contradiction.
- UNKNOWN: evidence/state is insufficient to decide reliably.

V2.0 defines but does not calculate or persist these results. `GapRequirement` dimension closure is a separate state and can never promote a Claim to VERIFIED.

## Persistence / compatibility decision

No migration in V2.0. V1.1 already persists `research_claims`, nullable `research_evidence.claim_id`, source owner identity, quote/snapshot/chunk provenance, graph edges and evidence-pair conflicts. Historical Evidence without a Claim link remains valid and unknown. The current one-Evidence-row-to-one-Claim shape is adequate for audit but not an ideal future many-to-many contract. After V2.1/V2.2 semantics stabilize, consider `claim_evidence_links` and versioned `claim_verification_results`; no dangerous bulk backfill is authorized.

## Claim funnel metrics

New namespace: **`claim-v2-baseline-v1`**. Keep existing `coverage-v1` byte-for-byte and comparable. Report funnel components independently (no composite score): supported claim rate, multi-evidence claim rate, independently corroborated claim rate, verified claim rate, conflicted claim rate, unknown verification rate, evidence/supported claim, owners/supported claim. Efficiency fields include tokens, runtime, LLM calls, searches, readers, extraction calls, accepted evidence, and ratios when available.

Historical V1.1 run IDs are inventoried across four task families. Claim-level historical values are NOT_RECONSTRUCTABLE from checked-in artifacts because raw evidence/claim/owner rows are absent; the local persisted database is unreachable from this process. Missing is not zero. See `v2_0_claim_baseline.md/json`.

## Golden fixture result

Seven deterministic structured fixtures cover exact same proposition, structured paraphrase candidate, time qualifier difference, numeric contradiction candidate, unit-equivalent value, polarity contradiction, and version difference. The semantics primitive only uses explicit structured fields; no embedding, LLM equivalence, semantic reranker, or broad text classifier is introduced. The paraphrase fixture’s structured fields explicitly share a predicate; full unstructured paraphrase matching remains V2.1.

## Architecture integration map

```text
Research Question → Planner → Search → Reader → Evidence Candidate → Acceptance
                                                           │
                         V1.1 live ─────────────────────────┘
                                                           ↓
Canonical Claim → Claim-Evidence Link → Source Identity → Claim Verification
       V2.0 contract/fixtures/baseline only ─────────────────┘
                                                           ↓
Claim-level Gap → ResearchNeed → Targeted Research
```

V2.0 stops at contract + offline baseline; all arrows after accepted Evidence are future interfaces, not active behavior changes.

| Phase | Ownership |
|---|---|
| V2.0 | Claim Architecture + Baseline + diagnostics |
| V2.1 | Semantic Claim Equivalence; input two claims with original/structured fields/qualifiers; output SAME/RELATED/CONTRADICTS/UNKNOWN + rationale/confidence; no side effects |
| V2.2 | Independent corroboration, per-Claim verification, contradiction detection |
| V2.3 | Evidence Formation Reliability |
| V2.4 | Claim-driven Gap Closure |
| V2.5 | Planner/Retrieval variance reduction only if evidence justifies |
| V2 RC | Report upgrade, cross-family qualification, release hardening |

## Diagnostics

Finite initial codes: `CLAIM_NORMALIZATION_FAILED`, `CLAIM_MISSING_SUBJECT`, `CLAIM_MISSING_VALUE`, `QUALIFIER_AMBIGUOUS`, `SOURCE_IDENTITY_UNKNOWN`, `EVIDENCE_LINK_MISSING`, `VERIFICATION_NOT_COMPUTABLE`, `RELATION_UNKNOWN`. Three-state observations preserve UNKNOWN/NOT_OBSERVED instead of coercing to false.

## Validation and stop gate

The historical V1.1 qualification rows were queried from the local PostgreSQL instance in a read-only transaction; the analyzer did not create or mutate research runs. The configured Docker hostname `postgres` was not resolvable from this shell, but `127.0.0.1:5432` was reachable.

### Bounded L2 blocker diagnosis and test-only fix

The timed-out run `01a0d22b-d35e-756f-a62f-173194a91cef` ultimately completed at 137.05 seconds after creation, with `completed_with_limitations`, 10 accepted Evidence, a verified report, and 22 searches. Its terminal time was about 13.6 seconds after the original 120-second test wait. It was not an infinite research loop. The decisive queue evidence is that an older outbox item (`01a0d223-8655-7dbb-8a9e-7cd6058740f4`, created about nine minutes earlier) and the target outbox item were both published at 06:48:02, in creation order; the older run executed first and finished at 06:49:07, immediately before the target started at 06:49:09. The test helper used the global `claim_batch()` and enqueued every claimed historical outbox item onto the current test's queue before returning for its target. Reused L2 database state therefore caused stale task work to delay the target. The worker teardown error followed the 120-second assertion unwinding while that target task was still active; with the outbox scoped to the requested run, teardown completed normally.

Fixes are test-only in `tests/integration/test_deterministic_research_smoke.py`: claim/publish only the target run's outbox row, allow 1–3 success repetitions for staged diagnostics (default remains 3), use the configured temporary `ARTIFACT_ROOT` instead of writing the repository artifact, and set the per-run wait ceiling to 180 seconds. The 180-second ceiling provides margin over the prior observed maximum of 117.602 seconds; it is not a production deadline and is below the measured, finite completion time. Production Research code and deadline policy are unchanged.

### Final validation

- L1 unit/API: **PASS**; focused V2.0 unit set: **20 passed**.
- Static: `ruff check` **PASS**; `mypy apps/api/app` (174 files) and analyzer mypy **PASS**; `git diff --check` **PASS**.
- PostgreSQL checkpoint/redelivery L2: **PASS**, 1 test.
- Deterministic success L2: **3/3 PASS**. Run IDs: `01a0d277-1f11-7527-ae06-1f1aa83fcc89`, `01a0d27a-56f1-764c-a8dd-9737d9f30735`, `01a0d27b-63bb-76aa-aa71-141ef67b6558`. All were `completed_with_limitations`, each had 10 candidate and 10 accepted Evidence, 22 searches, and a verified/fetchable report. Creation-to-terminal times were 72.879s, 68.409s and 113.577s.
- Deterministic failure L2: **PASS**. Run `01a0d278-f393-7d7c-aed8-92eb05bc642a` had 0 candidate/accepted Evidence and terminal reason `REPORT_NO_ACCEPTED_EVIDENCE`; no report was fabricated.
- Four required deterministic cases' pytest wall times total **339.77s (5m 39.77s)**, below the 15-minute suite budget. Worker context exited cleanly in each passing test; the existing harness does not record teardown duration separately.
- Isolation: target-only outbox dispatch prevented historical pending work from being placed on the current queue. The dedicated L2 database has no nonterminal outbox rows after the suite. Redis queue names are unique per test. Test artifact output is directed to pytest's temporary directory.
- Network/model boundaries: deterministic fixture Search, fixture HTTP transport with an allowlist that fails on unapproved DNS, and monkeypatched deterministic LLM gateway were active; **public network = NO, live Search = NO, live LLM = NO**.

No live qualification was run because Research behavior diff is NONE. Stop-gate result: **V2.0 READY → V2.1**, subject to the explicit phase boundary: this authorizes proceeding only after user confirmation; no V2.1 work is started here.
