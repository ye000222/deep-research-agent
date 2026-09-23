# V1 Release Required Files — Closeout Allow-list

Generated 2026-09-23 for the V1 reproducibility closeout. The JSON companion contains the explicit proposed allow-list; it is not permission to stage unrelated workspace content.

## Include in the candidate

- Modified application/runtime/configuration files, including API and Worker wiring, web package metadata, `.env.example`, Compose, and Python project configuration.
- Alembic revisions `20260918_0024` and `20260918_0025`, required to reproduce the current schema from revision `20260912_0023`.
- Untracked API/domain/DB/tool modules that the real API/Worker import graph loads, plus the associated regression tests and maintained unit tests.
- RC setup/provenance scripts, benchmark fixtures, maintained analysis/reproducibility tools, README and ADR/architecture docs.
- Stable V1 release notes and known-issues inventory.

The release inventory, release manifest, final gate, pre-RC reference and Phase 17 qualification reports are post-candidate audit outputs. They remain in the primary workspace and are refreshed only after the candidate SHA/build/smoke are known; they are not inputs to the candidate source tree.

## Preserve, do not stage

- `pytest_*.txt` and `phase记录.md`: temporary test output / local notes.
- `scripts/cancel_phase17_invalid_runs.py`, `scripts/set_phase14_3_experiment_mode.py`, `scripts/start_phase17_screening.py`: one-off mutating or run-launch scripts, not release runtime.
- `scripts/correct_phase15_2_comparability.py`: preserved historical one-off; it has mypy errors and is not required to reproduce the current runtime or release gates.
- `artifacts/*.local.json`, screenshots, and `artifacts/.secrets/`: local-only. No secret values were read into or copied to release artifacts.
- Other unclassified historical files remain untouched; UNKNOWN is not treated as permission to delete or stage them.

## Runtime evidence and provenance

An actual import audit of `app.main`, `app.worker.tasks`, and `app.worker.dispatcher` found current untracked modules in the import graph, including the GapRequirement persistence adapter, Evidence Alignment/Closure modules, provider adapters, and Evidence Input Quality. Those are release-required because a clean runtime import must reproduce the candidate currently under test.

The current source revision implementation was not truthful release provenance: it hashed a selected set of source files and emitted a 12-character content digest. The allow-listed `scripts/start.ps1` change reports the 40-character Git commit SHA for a clean tree and appends `-dirty` when the worktree has changes. This is packaging/provenance only; it does not change research behavior.

## Migration proof

Revisions form a single linear chain:

```text
20260912_0023 → 20260918_0024 → 20260918_0025 (head)
```

A disposable database was migrated from an empty schema through `20260912_0023`, then to head. The resulting revision was `20260918_0025`; `gap_requirements` and its `migration_source` column existed. The disposable database was dropped after verification. The active research database was not altered.
