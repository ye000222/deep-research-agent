# Post-V1.1 Test Infrastructure Optimization — Validation Result

Branch: `chore/test-infrastructure-v1.1`
Baseline HEAD: `87464a5113407e5860cbb7b55c8e605abd376676`
Release tag preserved: `v1.1.0` → `38e3ce1a64450cd4a3175bffab6f8ad140cf7fbf`

## Local validation

| Gate | Result | Observation |
|---|---|---|
| Ruff (`apps/api`, `tests`) | PASS | No lint findings |
| Mypy (`apps/api/app`) | PASS | 173 source files, no issues |
| Full default pytest discovery | PASS | Database-backed cases remain explicitly environment-gated |
| L1 unit + API tests | PASS | 19.85 seconds locally |
| PostgreSQL integration suite | PASS | 7 passed, 2 deterministic L2 cases skipped by their separate flag |
| Deterministic L2 | PASS | Success path 3/3; empty-evidence invariant passed; see `test_infrastructure_result.json` |
| Web TypeScript checks | PASS | App and Node tsconfig checks |
| Web production build | PASS | Built using the already-installed local Vite binary |
| Compose config | PASS | `docker compose config --quiet` exited 0; Docker config access warnings only |
| Static/release gate | PASS | Required files, migration head, golden evaluation, command and secret checks passed |
| `git diff --check` | PASS | No whitespace errors |

The ordinary local `pnpm --filter @deep-research/web build` command safely stopped
because pnpm wanted to purge the existing `node_modules` directory and no TTY
was available for confirmation. No purge/forced cleanup was attempted. The same
installed TypeScript and Vite binaries were invoked directly and both checks
passed; CI performs a clean dependency install before its normal build command.

## Deterministic L2 result

The success path uses the real API, PostgreSQL persistence and transactional
outbox records, production outbox repository claim/publish transitions, real
Redis/Celery queue and registered worker task, production ResearchLoop and
Reader, real extraction/scoring/acceptance and persistence, report writing,
report fetch, and verification. Search responses and model outputs are test-only
deterministic adapters; Reader HTTP is restricted to the two in-memory fixtures.
No live search or LLM calls were made. No accepted Evidence was inserted by the
test itself.

| Success run | Run ID | Terminal | Candidate / accepted Evidence | Report fetch | Verified | Duration |
|---:|---|---|---:|---:|---:|---:|
| 1 | `01a0cf21-9893-744a-b80a-143a1a888f6d` | completed_with_limitations | 10 / 10 | 200 | yes | 51.983 s |
| 2 | `01a0cf22-641a-7767-b4df-1caa1fb3037f` | completed_with_limitations | 10 / 10 | 200 | yes | 84.416 s |
| 3 | `01a0cf23-ae67-733a-baab-646ad3c84736` | completed_with_limitations | 10 / 10 | 200 | yes | 117.602 s |

Three-run success path: **259.291 s**.
Empty-evidence run: `01a0cf25-8c4f-7038-936a-a08304a5e51f`, **48.665 s**, zero
accepted Evidence, report endpoint returns 409, terminal reason
`REPORT_NO_ACCEPTED_EVIDENCE`.
Combined deterministic L2: **307.956 s** (~5m 08s), within the 15-minute target.

## Scope and limitations

- No files under `apps/api` or Research production behavior were changed.
- No Research smoke, benchmark, live qualification, or V2 work was run.
- L2 does not build or boot Docker images; Compose configuration parsing is
  covered separately. Image-only changes still need focused container validation.
- Local isolated PostgreSQL databases and Redis DBs were created solely for these
  tests; pre-existing project and historical artifacts were left untouched.
- GitHub Actions has not been pushed or observed from this local run.
