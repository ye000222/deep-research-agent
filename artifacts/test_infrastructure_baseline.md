# Test Infrastructure Baseline — before refactor

**Repository baseline:** branch `chore/test-infrastructure-v1.1`, starting HEAD
`87464a5113407e5860cbb7b55c8e605abd376676`. The immutable `v1.1.0` tag peels to
`38e3ce1a64450cd4a3175bffab6f8ad140cf7fbf`.

This is an audit snapshot of the existing test path before changing CI/test
infrastructure. It contains no Research Run or benchmark output.

## Existing test inventory

| Existing job/path | Purpose | Trigger/blocking | Dependencies and network | Deterministic / skip | Cost observation |
|---|---|---|---|---|---|
| `CI / api` | Ruff, mypy, all pytest-discovered tests | every PR and main push; blocking | Python dependencies; no live provider required by the tests audited | Postgres integration tests skip unless `RUN_POSTGRES_INTEGRATION=1` | Local `python -m pytest -q`: 34.36 s; seven integration cases were skipped by environment gating |
| `CI / PostgreSQL integration` | migrations, repository/persistence integration, checkpoint/redelivery and recovery behavior | every PR and main push; blocking | ephemeral PostgreSQL + Redis; no public search or live model required | explicit `RUN_POSTGRES_INTEGRATION=1`; clean CI databases | actual job timings were not retained in the repository baseline |
| `CI / web` | install and production web build | every PR and main push; blocking | pnpm package registry | deterministic build for pinned runtime/dependency lock state | install/build timing not retained |
| `CI / compose` | Compose file parse/config validation | every PR and main push; blocking | Docker CLI/config interpolation; no service startup or image build | deterministic configuration validation | seconds-scale expected; exact prior job time not retained |
| `CI / Static V1 release checks` | release gate, including lint/type/pytest, optional PostgreSQL integration, web build, golden evaluation, secret/release checks | runs after all other jobs; blocking when dependencies pass | installs Python and pnpm; starts a second PostgreSQL service | repeats gates already run by API/PostgreSQL/Web jobs; dependency failure can conditionally skip this aggregate job | source shows duplicate full-suite execution; no historical per-step timings retained |
| `scripts/benchmark_research.py`, `verify_v1_real_runs.py`, Phase 15/17 evaluators | real Research quality/qualification | operator/release qualification invocation, not ordinary unit CI | real provider, model, network, Reader, DB, and long Research Loop | live and intentionally variable | project baseline reports standard/live Research around 60+ minutes |

Tests are organized in `tests/unit`, `tests/api`, and `tests/integration`. The
integration directory has five existing test modules; PostgreSQL-backed cases
use the `RUN_POSTGRES_INTEGRATION` guard. There was no formal pytest marker
taxonomy, deterministic API-to-worker success smoke, test-only local search
fixture, or explicit L3 trigger policy.

## Existing time-cost funnel

```text
GitHub job/service provisioning (not separately retained)
  -> repeated dependency installation in independent jobs
  -> API Ruff + mypy + full pytest discovery
  -> PostgreSQL database creation + migrations + checkpoint setup + integrations
  -> Web dependency install + build
  -> Compose config only (no container build)
  -> aggregate release job waits for all above
       -> installs Python + pnpm again
       -> reruns static/full pytest
       -> reruns PostgreSQL integration when enabled
       -> reruns web build
       -> golden evaluation + secret/release checks
  -> separate, manually invoked 60+ minute live qualification/research
```

The 60+ minute cost comes from a full standard/live research run: variable
Planner/provider output, public search and DNS, third-party page availability,
fetch/parse attempts, live extraction/verification model calls, bounded research
iterations, deadline and report work. It is not the ordinary pytest suite: the
observed local baseline pytest run completed in 34.36 seconds. CI dependency
install, service provisioning and individual historical gate durations were
not recorded, so they remain **not measured**, not assumed to be fast/slow.

The previous Compose job only parsed configuration; CI did not build a fresh
application image. The integration and API jobs used the checkout's Python code.

## Known conditional skips

- PostgreSQL integration tests under default `pytest`: `EXPECTED_CONDITIONAL_SKIP`
  unless `RUN_POSTGRES_INTEGRATION=1` and a migrated database are supplied.
- Aggregate release job: dependency-based skip after a required upstream job
  failure was an expected workflow condition, but the skipped aggregate did not
  indicate that its unique release checks ran.
- No L3 live qualification job was part of ordinary PR CI; it was a manual
  qualification action with no standardized policy in `docs/testing.md`.

## Source files audited

`.github/workflows/ci.yml`, `pyproject.toml`, `tests/conftest.py`,
`tests/unit/**`, `tests/api/**`, `tests/integration/**`, `scripts/release_gate.py`,
`scripts/benchmark_research.py`, `scripts/verify_v1_real_runs.py`,
`apps/api/app/worker/dispatcher.py`, `apps/api/app/worker/tasks.py`,
`apps/api/app/infrastructure/runtime.py`, `apps/api/app/services/evidence_extractor.py`,
`apps/api/app/services/report_writer.py`, and `docker-compose.yml`.
