# DeepResearch Agent V1.1 Release Notes — Candidate

Previous release: `v1.0.0`
New release: `v1.1.0`
Current candidate identity: `v1.1.0-rc.1`
Validated research/runtime base commit: `466eefd301a6dfa24e39b58d0cf50a1b8c59a037`
Frozen reference: `v1-pre-rc-reference-1`

This is an incremental release in the V1 product line, not a new V2 product. The existing `v1.0.0` tag and GitHub Release are immutable and unchanged.

## Main changes since v1.0.0

- Expanded the research pipeline with explicit candidate/dispatch, reader, evidence-selection, and evidence lifecycle observability.
- Added question-level research state and recovery execution/context tracing, including recovery outcome and gap-closure feedback paths.
- Added evidence-to-requirement alignment and closure evaluation integration, plus feedback-driven query execution wiring.
- Improved provider failure classification, retry/fallback routing, and research continuation under provider degradation.
- Added benchmark, evidence-quality, gap-closure, and release qualification analysis artifacts and regression coverage.
- Improved release reproducibility: committed the audited runtime/migration file set, aligned build/source provenance, and verified clean-worktree Docker/schema/report plumbing.

These are pipeline, reliability, evaluation, and packaging changes. They do not imply uniformly high research coverage; results remain task-family dependent.

## Qualification reference

Phase 17.0 cross-family qualification:

| Family | Coverage |
|---|---:|
| Model comparison | `0.7143 ± 0.1010` |
| AI-agent competition | `0.8088` (one run; limited sample) |
| Multimodal manufacturing | `0.6532 ± 0.1466` |
| Low-altitude economy | `0.2500 ± 0.1782` |
| Cross-family macro-average | Mean ≈ `0.6066`; median ≈ `0.6837` |

Low-altitude economy remains a severe task-family limitation. The deployment smoke is plumbing evidence, not a new qualification benchmark.

## Frozen features

- `independent_source_targeting_enabled=true` — ACTIVE_V1.
- `evidence_aware_context_enabled=false` — EXPERIMENTAL_DISABLED.
- `evidence_input_quality_enabled=false` — EXPERIMENTAL_DISABLED.

## Deployment and limitations

The maintained deployment path is local Docker Compose. Copy `.env.example` to `.env`, configure provider credentials through the supported profile UI, then use `docker compose up -d --build`; see README for checkpoint initialization, API endpoints, and health checks. Production deployment hardening is not claimed.

See [`v1_known_issues.md`](./v1_known_issues.md) for low-altitude coverage weakness, run variance, the per-claim verified-claim limitation, q1 projection inconsistency, historical provenance gaps, and disabled experimental features. Semantic claim equivalence, cross-source corroboration, per-claim independent verification, source-role redesign, advanced evidence-input quality, and bounded variance-reduction work remain deferred.

## Version-correction scope

The v1.1.0 change is a release/package metadata correction based on the validated research/runtime commit above; it does not change Research Intelligence behavior. This release-metadata commit establishes the final candidate identity. The final candidate will receive clean-worktree reproducibility checks and a single plumbing smoke; no benchmark qualification suite is rerun. These notes describe a candidate, not an already-published release. No v1.1.0 tag, push, GitHub Release, or production deployment has been performed.
