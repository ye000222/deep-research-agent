# V1.1.0 Release Manifest

## Release identity

- Previous release: `v1.0.0` → `25b19db78448a04456d8111c318fcbe0254ca91d` (immutable).
- Release target: `v1.1.0`.
- Runtime-validated base: `36c920f58196a258dbcc215f5f76aec9a0582243`.
- Research/runtime qualification base: `466eefd301a6dfa24e39b58d0cf50a1b8c59a037`.
- Release decision: **V1.1.0 RELEASE READY WITH KNOWN RESEARCH VARIANCE**.
- The final release commit adds release documentation only on top of the runtime-validated base; it does not change runtime or Research Intelligence behavior.

## Frozen reference and runtime

- Reference: `v1-pre-rc-reference-1`; tier: `standard`; metric definition: `coverage-v1`.
- Frozen flags: `evidence_aware_context_enabled=false`, `independent_source_targeting_enabled=true`, `evidence_input_quality_enabled=false`.
- The validated runtime reported source revision `36c920f58196a258dbcc215f5f76aec9a0582243` across API, Worker, Dispatcher, Beat, Web, and launcher.
- Clean-candidate Docker builds, migration verification, service startup/readiness, static quality gates, and provenance checks passed.

## Live research variance record

Final-candidate smoke `01a0cdb6-4548-7d7c-a2c8-e9e86e665896` reached terminal state with `REPORT_NO_ACCEPTED_EVIDENCE`: 19 search-query starts, 13 completed searches, 31 readable sources, 15 extraction results, and zero Candidate/Accepted Evidence. No unsupported Report was persisted, as required by the no-fabrication invariant.

The read-only variance audit found no confirmed configuration regression, correctness bug, or P0. The strongest explanation was live retrieval/source-content variance: the final run's plan and extracted source inputs differed from the historical standard run. That historical run demonstrated report persistence and verification, but is not a controlled quality comparison against the final candidate.

## Qualification reference

Phase 17.0 cross-family qualification: model comparison `0.7143 ± 0.1010`; AI-agent competition `0.8088` (one run); multimodal manufacturing `0.6532 ± 0.1466`; low-altitude economy `0.2500 ± 0.1782`; cross-family macro-average mean ≈ `0.6066`, median ≈ `0.6837`.

## Limitations and feature disposition

- Known research run-to-run variance; live research success is not guaranteed for every task.
- Severe low-altitude economy coverage weakness; limited per-claim verification; q1 projection inconsistency; source-role limitations.
- `independent_source_targeting_enabled=true` is ACTIVE_V1.
- `evidence_aware_context_enabled=false` and `evidence_input_quality_enabled=false` remain EXPERIMENTAL_DISABLED.
- Production deployment hardening is not claimed; local Docker Compose is the maintained deployment path.
- V2-deferred work includes semantic claim equivalence, cross-source corroboration, per-claim independent verification, source-role redesign, advanced evidence-input quality, and bounded variance reduction.

Structured qualification evidence is retained in [`v1_release_manifest.json`](./v1_release_manifest.json). Publication identity is established by the Git tag and hosting release metadata, not by a mutable pre-publication status sentence in this manifest.
