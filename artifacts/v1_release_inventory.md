# V1.1.0 Release Inventory

## Current disposition

**V1.1.0 RELEASE READY WITH KNOWN RESEARCH VARIANCE.** Runtime-validated base: `36c920f58196a258dbcc215f5f76aec9a0582243`; research/runtime qualification base: `466eefd301a6dfa24e39b58d0cf50a1b8c59a037`. The final release commit is limited to release documentation. Previous release `v1.0.0` remains immutable at `25b19db78448a04456d8111c318fcbe0254ca91d`.

| Area | Evidence | Disposition |
|---|---|---|
| Runtime and source provenance | Validated API, Worker, Dispatcher, Beat, Web and launcher source revision `36c920f…`; frozen flags `false/true/false` | PASS |
| Database migrations | Migrations 0024/0025 committed; clean chain verified to single head `20260918_0025` | PASS |
| Docker and services | Clean no-cache builds, Compose startup, API health/readiness | PASS |
| Static quality and release checks | Pytest, Ruff, maintained MyPy gates, diff check, secret scan passed; historical non-allow-listed MyPy exception is documented | PASS WITH DOCUMENTED EXCLUSION |
| Final-candidate smoke | Run `01a0cdb6-4548-7d7c-a2c8-e9e86e665896`: 19 query starts, 13 searches completed, 31 readable sources, 15 extraction results, zero Candidate/Accepted Evidence | LIVE RESEARCH VARIANCE; not a confirmed deployment or correctness P0 |
| Report behavior | No report was persisted at zero accepted evidence; the invariant is correct. A historical standard run persisted and verified a report, but is not a controlled quality comparison. | INVARIANT PASS; plumbing previously demonstrated |
| Research quality limitations | Low-altitude weakness, run-to-run variance, limited per-claim verification, q1 projection inconsistency, source-role semantics | KNOWN LIMITATIONS |
| Experimental features | Evidence-aware context and evidence-input quality remain disabled | EXPERIMENTAL_DISABLED |

## Historical inventory provenance

The detailed pre-candidate workspace inventory remains preserved in [`v1_release_inventory.json`](./v1_release_inventory.json) as historical audit evidence. Its old dirty-tree, uncommitted-source, and migration conclusions describe the earlier audit point and are superseded by the clean-candidate verification recorded in the final gate. Historical Phase artifacts, local captures, and unknown files are not rewritten or removed by this release-documentation update.

Publication state is authoritative in the Git remote and hosting release metadata; this inventory records qualification facts and does not serve as a live tag/Release status tracker.
