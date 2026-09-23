# V1.1.0 Final Release Gate

## Decision

**V1.1.0 RELEASE READY WITH KNOWN RESEARCH VARIANCE.** The final-candidate smoke produced zero Accepted Evidence and correctly did not create an unsupported Report. The subsequent read-only variance audit found no confirmed configuration regression, correctness bug, or P0; the smoke outcome is classified as live research variance, not a deployment-plumbing failure.

## Candidate and immutability

- Previous immutable release: `v1.0.0` → `25b19db78448a04456d8111c318fcbe0254ca91d`.
- Runtime-validated base: `36c920f58196a258dbcc215f5f76aec9a0582243`.
- Research/runtime qualification base: `466eefd301a6dfa24e39b58d0cf50a1b8c59a037`.
- The release finalization commit is documentation-only, has the validated runtime base as its direct parent, and does not change runtime or Research Intelligence behavior.

## Final gates

| Gate | Result | Evidence |
|---|---|---|
| Migrations and clean database chain | PASS | Single Alembic head `20260918_0025`; clean chain from `20260912_0023` verified. |
| Version, provenance, frozen flags | PASS | Runtime identity matched `36c920f…`; flags resolved `false/true/false`. |
| Pytest, Ruff, MyPy, diff and secret checks | PASS | Candidate clean-worktree checks passed; one historical non-allow-listed analysis script retains known pre-existing MyPy errors. |
| Clean Docker build, startup and API readiness | PASS | API/Worker/Web images built; Compose services started; health and readiness passed. |
| Final-candidate live research smoke | RESEARCH OUTCOME FAILED; NON-BLOCKING VARIANCE | Run `01a0cdb6-4548-7d7c-a2c8-e9e86e665896`: 31 readable sources, 15 extraction results, zero Candidate/Accepted Evidence; terminal reason `REPORT_NO_ACCEPTED_EVIDENCE`. |
| No-evidence report invariant | PASS | The application correctly persisted no unsupported Report. A separate audited no-evidence case also ended with `REPORT_NO_ACCEPTED_EVIDENCE`. |
| Report plumbing | PREVIOUSLY DEMONSTRATED | Historical standard run `01a0c9c9-f951-76f0-a801-a10b65806236` persisted and verified a Report. Its differing plan/retrieval means this is plumbing evidence, not a controlled quality comparison. |
| Confirmed P0 | NONE | No confirmed configuration regression or correctness bug was found in the variance audit. |

## Known non-blocking limitations

- Live research completion and coverage have meaningful run-to-run variance.
- Low-altitude economy coverage remains weak; per-claim verification remains limited; q1 projection and source-role limitations remain documented.
- `evidence_aware_context_enabled` and `evidence_input_quality_enabled` remain disabled experimental features.
- Production deployment hardening is not claimed; local Docker Compose remains the maintained deployment path.

Detailed structured evidence is retained in [`v1_final_release_gate.json`](./v1_final_release_gate.json), and the cross-run analysis is described in the release notes and manifest. Publication state is authoritative in the Git remote and hosting release metadata.
