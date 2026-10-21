# PATH-C Attempt 002 timeout postmortem

## Evidence and limits

Human-authoritative outcome: CONSUMED_FAILED, physical outbound 1, blocked
additional outbound 0, MODEL_TIMEOUT / READ_TIMEOUT, HTTP 200 headers received,
no preserved raw response, seed omitted. Reported interval is approximately
7.65 seconds. This interval alone is not a timeout configuration measurement.
Historical attempt bytes live in the canonical volume; they were not accessed,
modified, or re-diagnosed in this offline task. No Docker was run.

The inspected operator source sets `httpx.Timeout(10.0)` after guarded-client
construction. Thus its configured connect/read/write/pool limits are all 10s.
The initial guarded HTTPX client default is 5s, but the explicit assignment
replaces it before the actual request. There is no request-level timeout
override, Gateway deadline, asyncio.wait_for or outer PATH-C total deadline.
Historical diagnostic records supplied here do not include timeout extensions
or a source hash proving this exact runtime configuration; the 10s finding is
code-path evidence, not an independently recovered historical runtime snapshot.

## Actual call chain

operator_generate_path_c_manifest.ps1
-> path_c_generation_entry._run
-> execute_path_c_generation / _execute_guarded_core
-> LLMGateway.generate_structured(allow_regeneration=False)
-> openai-compatible chat JSON-mode adapter
-> client.post
-> BudgetedAsyncTransport
-> RawResponseCaptureTransport
-> AsyncHTTPTransport(retries=0)
-> response.aread
-> raw freeze
-> Gateway response JSON and structured-output parsing.

Canonical request stream=False; the adapter does not enable streaming (no
stream:true in its HTTP request). An async response byte stream used by HTTPX
internally is not Provider SSE streaming. HTTP 200 only establishes that headers
arrived. A read timeout while consuming the body prevents both raw freeze and
structured parsing; it is not a provider HTTP error or a schema error.

Capture forwards the original request, including timeout extensions, unchanged.
It creates no HTTP client, supplies no alternate timeout, and adds no early-read
deadline. response.aread uses the underlying transport's read inactivity limit.
No capture/adapter semantics modification is necessary.

## Production comparison and generic correction

Ordinary Research Worker model generation previously uses
httpx.Timeout(90.0, connect=10.0): connect 10s, read/write/pool 90s.
API infrastructure's shared client uses 10s; Provider test endpoint uses
30s/connect10s. Neither is the ordinary Research Worker model-generation policy.
The operator matched the shared API client instead of the Worker generation
policy; no frozen contract requires this shorter timeout.

The Worker policy is extracted, numerically unchanged, into
app.llm.model_http_policy.model_generation_timeout. Both Worker and PATH-C now
use it. PATH-C still uses retry=0 and its unchanged one-outbound guard. The host
wrapper read-only mounts the new generic module into the ephemeral image import
path, avoiding a stale-container import. No image rebuild or Docker execution
was performed in this task.

New policy: connect10, read90, write90, pool90 seconds. These are phase inactivity
limits, NOT a 90s total deadline, and do not guarantee completion within 90s.
No Provider/model/Claim-specific branch or newly tuned timeout number was added.
Worker retry policy, Search/Reader timeouts, API shared client, governance,
Provider binding, seed, task count, prompts, historical attempts are unchanged.

## Deterministic proof

Virtual delayed body tests explicitly enforce request timeout extensions;
MockTransport does not enforce deadlines itself. No sockets or real sleeps.
- HTTP200 + virtual15s body: old10s fails, authoritative90s succeeds.
- Virtual91s body: MODEL_TIMEOUT / READ_TIMEOUT, CONSUMED_FAILED, one outbound.
- ConnectTimeout remains CONNECT_TIMEOUT, not READ_TIMEOUT.
- Capture preserves exact timeout extensions and HTTP200 evidence.
- Successful raw body freezes before structured parsing.
- JSON mode, no seed, non-streaming, 40-task prompt, one dispatch retained.
- Existing single-call, seed, adapter, safe failure-observability regressions run.

## Safety

Attempt001/002 remain consumed historical failures; this task neither accesses
nor writes their canonical files. Attempt003 was not authorized or created;
canonical global absence was not inspected (Docker prohibited).
Provider calls=0, External HTTP=0, Docker=0, DB writes=0. No stage/commit.
The root cause of Provider/body delay itself is not established; this closes
the identified timeout-policy wiring defect, not a promise of live success.
