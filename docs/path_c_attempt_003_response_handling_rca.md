# PATH-C Attempt 003 HTTP-200 response-handling RCA

## A. Installed response semantics

Offline test environment: HTTPX 0.28.1; optional brotli/brotlicffi absent.
No Docker inspection was performed, so the historical container HTTPX version
is not independently established. Installed source inspected directly:
AsyncHTTPTransport returns Response(stream=AsyncResponseStream(...), original
headers/extensions). Response.aread collects aiter_bytes, which uses the
Content-Encoding decoder; the bytes are decoded entity bytes, not compressed
wire bytes. Successful aiter_raw closes the original response automatically.
Response(content=bytes) eagerly invokes synchronous read/iter_bytes in its
constructor, applying Content-Encoding again if that header remains.

## B. Capture semantics and actual boundary

Capture awaits the original response.aread, hashes/freezes those decoded bytes,
then reconstructed Response(content=decoded_bytes, headers=original_headers,
extensions=original_extensions, request=original_request). Original response is
also closed in finally. The constructor, before returning to AsyncClient, is
the exact faulty boundary for gzip: GZipDecoder receives decoded JSON bytes.
The concrete exception is httpx.DecodingError, caused by zlib.error. No task
content mutation or timeout behavior is involved.

## C-D. Reproduction before any capture modification

Same capture path through guarded AsyncClient: plain synthetic HTTP200 passes;
gzip HTTP200 fails after exact decoded-body freeze. Same gzip envelope through
ObservedPathCGateway and the generation core fails with MODELGATEWAYERROR.
Initial run: 2 failed / 8 passed / 2 conditional brotli skips. The old constructor
statement is retained as a deterministic exception-boundary test.
Gzip double-decompression: REPRODUCED. Brotli: NOT_TESTED_OPTIONAL_SUPPORT_ABSENT;
no dependency installed. Three brotli checks in the final suite are conditional
skips (capture consumption, old boundary, full adapter path).

## E-G. Exception taxonomy

DecodingError subclasses RequestError. The old generic _post_json handler
caught it as RequestError and emitted MODEL_NETWORK_ERROR. Its detail fell back
to REQUEST_ERROR, which PATH-C safely redacted. Native DecodingError itself was
also categorized as NETWORK_ERROR before the fix. NETWORK was wrong for the
reproduced local reconstruction error.

Now DecodingError, StreamConsumed and StreamClosed map to non-retryable
MODEL_RESPONSE_INVALID with enumerated safe detail codes. Envelope JSON decode
errors likewise map to MODEL_RESPONSE_INVALID / JSON_DECODE_ERROR. Genuine
ReadError remains MODEL_NETWORK_ERROR / READ_ERROR. Existing structured model
output/schema failures retain their existing categories. No arbitrary message,
header, credential or semantic content is added to diagnostics.

## Response lifecycle audit

Synthetic tests cover StreamConsumed, StreamClosed, ReadError during read,
ReadError during close, and a SyncByteStream passed to an async transport.
The last is a RuntimeError (contract misuse), not a network error. For normal
response consumption, original response closes and replacement remains readable
via aread/content/json and aclose is safe/idempotent. Sync close on an
AsyncClient-bound response is a RuntimeError, not the normal adapter path.
Preserved network_stream extension referencing the closed original did not
break normal Gateway parsing. Deliberately malformed http_version raises
AttributeError only when that property is accessed, not normal json parsing.
No speculative extension/lifecycle patch was made. Request association and
HTTP status are preserved.

## H. Real Attempt 003 structural evidence

Canonical raw response was not accessible in the offline environment and was
not inspected, printed, copied, summarized or selected. Supplied evidence says
size25081/SHA b17e802bad7f4c5fef9aff48b0e5fd20e5cc6b84b3fbc9f2f4a84c534fe787ef.
Envelope validity, completion JSON validity, actual task count and historical
Content-Encoding are UNKNOWN. The generic defect and matching failure class
are proven; this does not prove the exact historical response used gzip.

To close real-response structural uncertainty, a minimal content-blind host
entry was prepared but not run. Its volume and helper mounts are read-only,
network disabled, no build/pull/retry. It checks the supplied SHA/size BEFORE
parsing the fixed Attempt003 raw path and emits only hashes/counts/types/booleans
and allowlisted envelope key names. Arbitrary generated keys/strings, model/id
values, errors, headers and task text are never emitted. It cannot authorize,
generate, write DB or alter consumed evidence. It cannot recover missing
historical compression headers from decoded raw bytes.

## I. Justified generic fix

After decoding, strip Content-Encoding, old Content-Length and Transfer-Encoding
from replacement headers. HTTPX generates the new decoded-body Content-Length.
All decoded content bytes remain exact; status/request and remaining headers/
extensions remain intact. Compression handling is generic, with no Provider,
model or PATH-C-text branch. Capture still freezes bytes before Gateway parsing.
No JSON repair, truncation, regeneration, fallback or second request is added.

## J. Deterministic checks

Plain/gzip same-path consumption and adapter parsing now pass with unchanged
decoded-body hashes. Both request-shape checks retain JSON mode, no seed, 40
requested tasks and one outbound. Genuine network and local response failures
are distinct and safe-redacted. Full targeted suite and type/lint/diff results
are reported in the delivery response; optional brotli skips are explicit.

## K. Safety and closure boundary

Attempt001/002/003 canonical files were not accessed or modified, and no
historical diagnostics were rewritten. Attempt004 was not authorized/created;
canonical global absence was not inspected. Real Provider=0, External HTTP=0,
Docker=0, DB writes=0, Holdout=0. No stage/commit. Known defect closure does not
guarantee future live Provider success or certify the unseen raw task output.
