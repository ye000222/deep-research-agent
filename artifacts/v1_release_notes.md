# Deep Research Agent V1.1.0 Release Notes

Previous release: `v1.0.0`
Release: `v1.1.0`
Runtime-validated base: `36c920f58196a258dbcc215f5f76aec9a0582243`
Research/runtime qualification base: `466eefd301a6dfa24e39b58d0cf50a1b8c59a037`

V1.1.0 is an incremental release in the V1 product line, not a new V2 product. The existing `v1.0.0` release remains unchanged and immutable. The final release documentation commit is documentation-only and does not change runtime or Research Intelligence behavior.

## What's New

- Expanded observability across candidate dispatch, reader, evidence selection, and evidence lifecycle stages.
- Added question-level research state and recovery execution/context tracing, including recovery outcomes and gap-closure feedback.
- Added evidence-to-requirement alignment and closure evaluation integration, with feedback-driven query execution wiring.
- Improved provider failure classification, retry/fallback routing, and research continuation under provider degradation.
- Added benchmark, evidence-quality, gap-closure, and release-qualification analysis artifacts and regression coverage.
- Improved release reproducibility through audited runtime/migration provenance and clean-worktree Docker, schema, and report-plumbing validation.

## Validation

Runtime and reproducibility validation was completed against the runtime-validated base before this documentation-only finalization. Validation covered clean Docker builds, migration verification, API/Worker/runtime provenance, static quality gates, and live research execution plumbing.

The final-candidate standard smoke reached Search, Reader, and Evidence Extraction but produced no Candidate or Accepted Evidence; the application correctly declined to persist an unsupported report. The final smoke variance audit classified this as live research variance, with no confirmed configuration regression, correctness bug, or P0. A prior standard run demonstrated persisted and verified report plumbing, but the two runs differ in plan and retrieved-source outcomes and are not a controlled quality comparison. A deployment smoke is plumbing evidence, not a qualification benchmark.

Phase 17.0 cross-family qualification reference:

| Family | Coverage |
|---|---:|
| Model comparison | `0.7143 ± 0.1010` |
| AI-agent competition | `0.8088` (one run; limited sample) |
| Multimodal manufacturing | `0.6532 ± 0.1466` |
| Low-altitude economy | `0.2500 ± 0.1782` |
| Cross-family macro-average | Mean ≈ `0.6066`; median ≈ `0.6837` |

## Known Limitations

- Research outcomes have meaningful run-to-run variance; research completion and coverage are not guaranteed for every live task.
- Low-altitude economy remains a severe task-family coverage weakness.
- Per-claim verified-claim output remains limited; cross-source corroboration and semantic claim equivalence are incomplete.
- The q1 projection inconsistency and source-role semantics remain known limitations.
- Production deployment hardening is not claimed. The maintained deployment path is local Docker Compose; see the README for setup and health checks.

See [`v1_known_issues.md`](./v1_known_issues.md) for the detailed limitations and their disposition.

## Experimental Features

- `independent_source_targeting_enabled=true` — ACTIVE_V1.
- `evidence_aware_context_enabled=false` — EXPERIMENTAL_DISABLED.
- `evidence_input_quality_enabled=false` — EXPERIMENTAL_DISABLED.

## V2 Deferred Work

Semantic claim equivalence, cross-source corroboration, per-claim independent verification, source-role redesign, advanced evidence-input quality, and bounded variance-reduction work remain deferred. These limitations are not represented as completed capabilities in this release.
