# Test & Evaluation Levels

This repository separates fast engineering feedback from deterministic runtime
integration and expensive live research quality evaluation. The test helpers in
`tests/support` are test-only adapters; production modules do not import them and
there is no production setting that enables them.

## L1 — Fast Engineering Gate

Blocking on every pull request and push to `main`.

| Gate | What it verifies | External dependencies | Target |
|---|---|---|---:|
| API | Ruff, mypy, unit and API tests | None; tests use in-process or mocked boundaries | <= 5 min |
| PostgreSQL | Clean service database, migrations, schema/repository/persistence integration | Ephemeral PostgreSQL and Redis | <= 10 min |
| Web | Type/build checks through the web production build | Package registry during dependency install; no app backend | <= 5 min |
| Static/release consistency | Compose configuration, migration-head/release-file sanity, golden fixture and secret-safe checks | Docker CLI for config parsing only; no image build | <= 5 min |

The API job intentionally runs `tests/unit` and `tests/api`; database-backed
integration tests belong to the PostgreSQL job instead of silently skipping
inside the default pytest invocation. The PostgreSQL job starts from fresh CI
service databases, creates isolated business, integration, and L2 databases,
applies Alembic migrations, and initializes checkpoint schemas. No developer
database or historical Research Run is used.

Initial L1 target is <= 15 minutes across parallel jobs. CI job setup and
dependency installation are included in the GitHub job duration; each job's
steps are parallel with the other L1 jobs. The PostgreSQL job records migration,
runner-side readiness, and test-suite timings in its step summary.

## L2 — Deterministic Integration Smoke

Blocking in the PostgreSQL integration job on normal PRs. Target: success path
under 10 minutes, empty-evidence failure path under 5 minutes, and the complete
L2 job under 15 minutes.

```text
FastAPI API (TestClient)
  -> persisted run + transactional outbox
  -> production outbox repository claim/publish state transitions
  -> Redis/Celery queue
  -> registered production worker task
  -> production ResearchLoop and provider adapter contract
  -> deterministic search boundary
  -> production PublicWebReader over allow-listed in-memory HTTP fixtures
  -> production EvidenceExtractorService with deterministic model-boundary output
  -> existing evidence scoring/acceptance and PostgreSQL persistence
  -> production report writer and report persistence
  -> API report fetch
  -> persisted verification endpoint
```

The success test runs three distinct API-created Runs through the real queue and
worker. It asserts terminal success or completed-with-limitations, candidate and
accepted Evidence, persisted/fetchable Report, and `verified=true` on every Run.
The separate empty-evidence path asserts zero Evidence, no Report, and the
existing `REPORT_NO_ACCEPTED_EVIDENCE` terminal reason.

Search results and model output are deterministic test adapters. Reader content
comes from the two small HTML files in `tests/fixtures/research_smoke`; the real
Reader parses them through `PublicWebReader`, while its HTTP transport is an
allow-listed `httpx.MockTransport`. DNS and HTTP outside the exact fixture hosts
and URLs fail the test. There is no public search, public webpage, DNS, or live
LLM dependency. Evidence still goes through the production extraction service,
existing acceptance rules, repository writes, report service, fetch path, and
verification path; the test never inserts accepted Evidence directly.

No fixture provider, fake LLM, or fixture flag exists in production settings or
provider registration. Test code patches only the worker's external adapter
constructors within the test process. The production provider registry test
asserts that the fixture provider is not selectable by production defaults.

The job checks out and runs the current commit's Python source; it does not use
a cached application image. API and Worker are exercised in-process/embedded
against real ephemeral PostgreSQL and Redis. Docker image build/boot is not part
of L2; Compose syntax is checked in L1. Therefore a future Dockerfile/image-only
change still needs a focused container build/boot validation before release.

## L3 — Live Research Qualification

L3 is an expensive quality evaluation, not normal CI smoke. It retains the real
Planner, providers, public network, Reader, model, Research Loop, Evidence,
Gap/Coverage, and evaluation paths. Do not replace it with deterministic
fixtures when a Research behavior change is under review.

Trigger L3 manually in the controlled qualification environment for:

- Planner, Search/query generation, provider routing semantics, Reader,
  extraction, acceptance, verification, Gap/Coverage, budget/recovery/deadline
  semantics, or major Research architecture changes;
- formal release qualification when the release gate explicitly requires it.

Ordinary docs, release-note, version metadata, CI tooling, web-only, and
test-fixture changes do not trigger L3. L3 is not scheduled on every PR and has
no automatic nightly trigger in this repository. Use the existing reviewed
qualification entry points (`scripts/benchmark_research.py` and the applicable
qualification/evaluation script) only with an explicitly selected benchmark,
credentials, budget, and operator approval. This infrastructure task did not
run L3.

## Change-impact matrix

| Change area | Required validation |
|---|---|
| Docs, release notes, version metadata | L1; focused artifact/consistency checks as applicable |
| CI tooling, test fixtures, package/build configuration | L1; L2 when runtime integration or fixture execution is affected |
| Docker/Compose/runtime wiring | L1 + L2; focused image build/boot validation because L2 does not build images |
| Migration, database model, repository persistence | L1 PostgreSQL + L2 |
| API/Worker/outbox/queue plumbing | L1 + L2 |
| Planner, Research Need/Context, query generation/ranking, Search/provider semantics | L1 + L2 + L3 |
| Reader, extraction, Acceptance, claim verification, Gap/Coverage | L1 + L2 + L3 |
| Budget, Recovery, deadline, or major Research architecture | L1 + L2 + L3; benchmark qualification when the change alters research outcomes |

## Commands

```bash
# Fast non-database L1 checks
ruff check apps/api tests
mypy apps/api/app
pytest tests/unit tests/api -q

# PostgreSQL L1 + deterministic L2 (requires fresh migrated DBs and Redis)
RUN_POSTGRES_INTEGRATION=1 RUN_DETERMINISTIC_SMOKE=1 pytest tests/integration -q

# The deterministic L2 tests alone, after their isolated DB/checkpoint schemas exist
RUN_POSTGRES_INTEGRATION=1 RUN_DETERMINISTIC_SMOKE=1 pytest tests/integration/test_deterministic_research_smoke.py -q

# Compose and release consistency
docker compose config --quiet
python scripts/release_gate.py --skip-compose --skip-static --skip-integration --skip-web
```

The L2 tests require `L2_DATABASE_URL`, `L2_CHECKPOINT_DATABASE_URI`, and
`REDIS_URL`; the workflow provisions these from clean services. A missing L2
environment is an explicit skip locally, not a pass. The workflow sets the flag
and required URLs, so its L2 tests cannot be silently skipped.
