# V2.0 Existing Claim Semantics Audit

**Audit base:** `815d76c87011eebbd1da685c4f1c054d0f073f81`
**Branch:** `feature/v2-claim-architecture`
**L2 blocker resolution completed:** `2026-09-24 08:22:41 UTC`
**Research behavior diff:** `NONE`

## Current live data path

`EvidenceExtractor` produces scored candidate evidence. `ResearchTools.record_page` stores a `research_claims` row keyed by `(run_id, question_id, claim_hash)` and a `research_evidence` row whose nullable `claim_id` points to the Claim. The evidence row carries its `source_id`, evidence relation, acceptance result, exact quote, and optional source snapshot/chunk. `research_sources` provides `source_owner_key`; source identity is joined through `source_id`, not inferred from source role.

The Claim ID is a stable database entity ID for a persisted record. The hash is only a lexical deduplication key: whitespace is collapsed and case is folded. It does not equate semantic paraphrases, nor does it implement qualifier-aware equivalence. Original `atomic_claim` text is retained, but there is no structured time/location/unit/version/scope qualifier object.

## Findings by requested question

| # | Finding |
|---:|---|
| 1 | Claims are materialized while `ResearchTools.record_page` persists extraction output. Rejected evidence can also have a Claim record; accepted graph evidence is a separate condition. |
| 2 | `claim_id` is a persisted UUID; stable hash lookup reuses a row for the same run/question and lexical fingerprint. |
| 3 | Claim text is not semantically canonical. The current hash normalization is whitespace/case only. |
| 4 | `research_evidence.claim_id` is a nullable FK: a practical one-evidence-row-to-one-claim link. Relation/quote/acceptance are on that evidence row. |
| 5 | `research_evidence.source_id` joins the source row; `source_owner_key` is the distinct-owner identity used for corroboration. |
| 6 | `ClaimVerificationAnalyzer` consumes `GapRequirement.verification_status` and dimension-level counts/alignment. Its optional `claim_id` is an output label, not a persisted evidence-grounded claim verdict. |
| 7 | Historical `verified_claims` is often zero/informational because V1.1 persists no canonical per-Claim verification result. A zero snapshot cannot be reinterpreted as an observed count of unverified Claims. |
| 8 | Dimension verification is on `GapRequirement` (question/dimension/requirement); graph Claim status is on `(run, question, claim_hash)`. These are different keys and evidence aggregation levels. |
| 9 | Live: `research_claims`, `research_evidence.claim_id`, source owner identity, deterministic claim edges/conflicts, graph projection, and dimension closure state. |
| 10 | Aggregate/compatibility: `KnownClaimRef`, budget claim-risk IDs, quality snapshot `claim_count`/`citation_support`, and the optional-ID analyzer output. No persisted semantic identity, explicit qualifier model, or claim verification history was found. |

## Legacy graph statuses are not V2 verification states

The graph helper currently yields `rejected` without accepted evidence, `partial` with accepted evidence but fewer than two owners, `supported` with at least two distinct owners, and `disputed` for refuting evidence. `candidate` is the row default. In particular, legacy `supported` is owner corroboration, not a claim-level `VERIFIED` decision, and `partial` does not necessarily mean a semantic partial proposition.

The deterministic relation layer stores Claim-to-Claim `supports`, `supplements`, or `contradicts` edges and evidence-pair conflicts. The current `supports` string at that layer must not be conflated with Evidence-to-Claim `SUPPORTS`.

## Persistence and historical access

The existing V1.1 schema already carries the baseline provenance needed for offline audit. V2.0 adds no migration or runtime writes. A separate link table may become appropriate if V2.1/V2.2 require many-to-many evidence links or relation-level provenance; a versioned per-Claim verification result is also absent today.

The Phase 17.0 suite identifies ten qualification runs across four task families. Although the configured Docker hostname `postgres` does not resolve in this process, the existing local PostgreSQL port is reachable at `127.0.0.1:5432`; the baseline analyzer used that connection in read-only transaction mode. Claim/evidence/owner funnel and cost values were queried for those ten runs. Per-Claim VERIFIED/UNKNOWN remains **NOT_RECONSTRUCTABLE** because V1.1 has no persisted per-Claim verification result. See [v2_0_claim_baseline.md](v2_0_claim_baseline.md).
