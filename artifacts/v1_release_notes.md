# DeepResearch Agent V1 Release Notes (Candidate)

Release target: `v1.0.0`

Current candidate identity: `v1.0.0-rc.1`
Frozen reference: `v1-pre-rc-reference-1`

## Product scope

V1 is a generic autonomous research agent with evidence grounding, gap-aware research, independent-source targeting, event traceability, and evidence-backed report generation. It is not positioned as a mature, uniformly high-stability deep-research product; results vary by task family and provider availability.

## Main capabilities

- Docker Compose deployment with API, web, worker, dispatcher, beat, PostgreSQL, Redis and SearXNG.
- Multi-step research plans, query/search, safe web reading, evidence extraction and acceptance, coverage/gap evaluation, and report verification.
- Persisted run events and state, report retrieval, and explicit no-evidence failure rather than fabricated reports.
- Independent-source targeting is active in the frozen V1 reference configuration.

## Frozen features

- `independent_source_targeting_enabled=true` — ACTIVE_V1.
- `evidence_aware_context_enabled=false` — EXPERIMENTAL_DISABLED.
- `evidence_input_quality_enabled=false` — EXPERIMENTAL_DISABLED.

## Qualification summary

Phase 17.0 cross-family qualification references:

| Family | Result |
|---|---:|
| Model comparison | `0.7143 ± 0.1010` |
| AI-agent competition | `0.8088` |
| Multimodal manufacturing | `0.6532 ± 0.1466` |
| Low-altitude economy | `0.2500 ± 0.1782` |
| Cross-family | Mean ≈ `0.6066`; median ≈ `0.6837` |

Low-altitude economy is a known severe family limitation, not representative evidence of uniform performance. Phase 17.1 standard deployment smoke produced accepted evidence, persisted a report, fetched it through the application service and verified it. The separate no-evidence path correctly refused to fabricate a report.

## Deployment

The maintained deployment path in this candidate is local Docker Compose. Copy `.env.example` to `.env`, configure provider credentials through the supported profile UI, then use `docker compose up -d --build`; see README for checkpoint initialization, API endpoints and health checks. Production hardening/deployment is not claimed by this candidate.

## Known limitations

See [`v1_known_issues.md`](./v1_known_issues.md): low-altitude task-family weakness, run variance, no per-claim verified-claim output, q1 projection inconsistency without proven user-visible impact, historical provenance gaps, and disabled experimental features.

## Deferred beyond V1

Semantic claim equivalence; cross-source claim corroboration; per-claim independent verification; source-role semantic redesign; advanced evidence-input quality; and bounded variance-reduction work.

This is a packaging candidate note, not a publication announcement. The release gate is currently blocked by uncommitted source/schema provenance; do not treat it as a released `v1.0.0` build.
