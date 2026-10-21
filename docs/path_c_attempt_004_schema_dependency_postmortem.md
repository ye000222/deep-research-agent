# PATH-C Attempt 004 schema runtime dependency closure

## A-B. Missing dependency and exact raising boundary

The validator directly imports:
`from jsonschema import FormatChecker, validators` in schema_validation.py.
ImportError is wrapped in SchemaDependencyUnavailable; the unchanged generation
core turns type(exc).__name__.upper() into SCHEMADEPENDENCYUNAVAILABLE after
output validation fails. Deterministic absence tests prove ModuleNotFoundError
with name=jsonschema -> SchemaDependencyUnavailable -> CONSUMED_FAILED with
LOCAL_RUNTIME_ERROR/OUTPUT_VALIDATION and no manifest. Real historical inner
ImportError details were not provided, so its exact subtype/subdependency is
not independently recovered; no Docker or historical forensic access occurred.

Actual source chain: operator_generate_path_c_manifest.ps1 -> canonical
attestation -> Compose API ephemeral runtime -> path_c_generation_entry ->
Gateway result -> ProviderTaskResponse + structural validation -> frozen task
manifest schema validation -> validator_for_schema -> jsonschema imports.
The reported Provider HTTP200/raw capture succeeded before this local failure.

## C-E. Environment and packaging discrepancy

Local dev/test Python:3.12.14, jsonschema:4.26.0, import confirmed. Before this
fix local installed project metadata also declares jsonschema only under
extra==dev. Existing JSON Schema tests importorskip the library, so a green
test run alone does not attest runtime availability. New tests do not skip it.

pyproject previously declared jsonschema[format]>=4.23,<5 only in dev extras.
API Dockerfile copies pyproject and runs pip install ., not .[dev]. Normal API
and the operator share deep-research-agent-api:local. Operator mounts current
source/scripts/contracts over that image, without installing dependencies or
building an image. Thus updated schema_validation.py can execute over an image
that does not contain its required import. This declaration/build-path defect
is proven; exact old-image package inventory and build time were not inspected.
There is no Python lock file in the root; pnpm-lock.yaml is unrelated.

## F. Narrow canonical fix

Move the SAME jsonschema[format]>=4.23,<5 requirement to project.dependencies;
remove the duplicate dev-only declaration. No version-range adjustment, schema
change, validation loosening, alternate engine or exception-acceptance path.
Only the dependency error explanation is updated from dev to runtime wording.
Dockerfile still uses the normal canonical pip install . build step, unchanged.
No dynamic pip installation, vendoring or site-packages mounts are introduced.

## G-I. Host rebuild and exact-runtime attestation

The API image must be rebuilt. Changing a bind-mounted pyproject/source file
does not alter packages already installed in an existing image. Rebuilding
does not restart the resident API/Worker or execute a Research task.

Existing operator_verify_canonical_runtime.ps1 is extended (no new governance
framework). It hashes/imports the two schema modules from current checkout,
mounts the frozen A51 schema read-only, and executes the same schema_runtime_probe
before DB/Provider resolution. The probe reads actual installed project
Requires-Dist metadata, requires a non-dev jsonschema[format] declaration,
checks frozen schema SHA, initializes its declared-dialect validator, validates
a synthetic40-task manifest and rejects a synthetic39-task manifest. PASS is
mandatory in Test-ProbeSuccess and is included in canonical runtime artifacts.
Absent dependency/metadata, invalid schema or validation regression fails closed.
The probe creates no client, request, file or DB row.

Human-only sequence (not executed by Codex):

```powershell
Set-Location D:\agentProject
docker compose build api
if ($LASTEXITCODE -ne 0) { throw 'API_IMAGE_REBUILD_FAILED' }
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\operator_verify_canonical_runtime.ps1
if ($LASTEXITCODE -ne 0) { throw 'CANONICAL_SCHEMA_RUNTIME_ATTESTATION_FAILED' }
```

Do not run the generation script after these commands. No Attempt005 is
authorized by this task. Successful host evidence must include schema_runtime
PASS, installed jsonschema version, validator class, valid/invalid document
results and zero Provider counters. Local tests with synthetic runtime metadata
do not replace this real-image evidence. Local installed project metadata is
still the old dev-only metadata; Codex did not reinstall packages.

## H. Offline proof

Tests cover runtime dependency declaration/build input, real installed engine,
valid40 tasks, invalid39 and required/type/nested/extra/const violations,
missing dependency fail-closed without bypass, same probe accepting correct
runtime metadata and rejecting dev-only metadata, absent dependency safe report,
and same output-validation path retaining consumed failure/no manifest.
PowerShell self-tests include missing/failed schema gate rejection and retain
the other original integrity tests; SelfTest exits before Docker operations.
Final targeted regression/type/lint/diff results are in the delivery report.

## J. Historical and safety accounting

Attempt001/002/003/004 canonical evidence was not accessed, reset or changed.
Attempt004 raw response remains forensic only and was not adopted as success
or transformed into a persisted manifest. No raw-task text was inspected.
Attempt005 was not created/authorized; canonical global absence was not checked.
Provider calls=0, External HTTP=0, Docker=0, DB writes=0, Holdout=0. No stage/
commit. Frozen schema/prompt/seed/attempt contracts and prior generic fixes are
unchanged. Closure is packaging-complete but host rebuild/attestation pending.
