# PATH-C generation attempt identity and authorization v1

This document defines an implementation contract, not an authorization.
Attempt 001 remains consumed. No Attempt 002 grant or state is created by it.

## Legacy compatibility

Only `v2_1r_d4r_path_c_generation/<frozen-population>__<frozen-batch>/generation_state.json`
is recognized as legacy Attempt 001. Its population, batch, ExecutionContext,
generation-contract version and the documented consumed failure accounting must
match. Resolution does not move, copy, rewrite or enrich that file.

The resolver computes a **virtual reference** from the frozen scope, number 1,
null authorization ID and `legacy-path-c-001.v1`. This is a reference rule,
not a claim that an attempt ID or authorization ID was persisted in 001.
The original state bytes supply the parent-state SHA-256 for a later grant.

## Future storage

Within the same batch root:

```text
generation_state.json                      original legacy 001; unchanged
authorizations/
  attempt_NNN.authorization.json           separately approved, immutable grant
attempt_NNN/
  attempt_identity.json                    create-once immutable identity
  generation_state.json                    this attempt's lifecycle only
  dispatch_reservation.json                create-once consumption receipt
  provider_response.raw.json               safe success response, if present
  path_c_task_manifest.json                 valid frozen result, if present
  generation_metadata.json                 this attempt only
  hash_manifest.json                       this attempt only
```

NNN is canonical zero-padded decimal, starting at 002. No mutable global index
or database is required. Numbered directories must be contiguous, identities
and authorization IDs unique, and each successor must bind its predecessor's
terminal failed state and exact state-file hash. Missing/partial/ambiguous
history, aliases, symlinks and duplicate JSON keys fail closed.

## Separate authorization

`PathCAttemptAuthorization` is frozen and rejects extra fields. A future human
governance task must create the exact next grant using no-overwrite publication.
It binds the authorization UUID, attempt number and deterministic ID, previous
attempt reference and terminal state/hash, frozen ExecutionContext ID/hash,
effective generation contract/version/hash, task schema hash, seed amendment,
Provider-safe fingerprint and full safe-binding hash, UTC creation time,
explicit human-governance origin, authorized status and generation purpose.

The call limits are constants: one dispatch, zero retry, no regeneration,
repair, fallback, top-up or replacement. No model, endpoint, owner hash or
credential override is a valid field. Actual Provider resolution still uses
the existing authenticated frozen context and live owner-scoped drift check.

The authorization hash is SHA-256 of the normalized model JSON excluding only
`authorization_hash`. This is an integrity binding, not a cryptographic signature
or permission for code to fabricate approval. Creation remains a separate,
explicit governance action. The raw grant-file hash is frozen in the immutable
attempt identity, so reformatting or rehashing a consumed grant cannot reuse it.

Attempt IDs are SHA-256 of canonical JSON containing the history scope,
attempt number, authorization UUID and `path-c-attempt-identity.v1` rule ID.
This allows a later governance task to prepare a grant without changing code.

## Execution ordering and crash safety

Read history and grant -> validate integrity and exact next identity -> validate
frozen runtime binding -> verify frozen generation files -> live Provider drift
check -> create identity and AUTHORIZED_NOT_DISPATCHED state -> exclusive durable
dispatch receipt -> DISPATCH_RESERVED -> at most one HTTP admission -> terminal
failure or verified frozen success.

Initialization does not consume an allowance; the exclusive dispatch receipt
does. A valid initialized, unconsumed state may resume. A receipt with an
AUTHORIZED_NOT_DISPATCHED state means reservation interrupted state replacement:
it remains consumed and cannot resume. Partial identity initialization is never
automatically repaired. Existing single-outbound guards reject extra transport
attempts and failure diagnostics are committed only within the current attempt.

The state and metadata carry attempt number/ID, authorization ID/hash, grant-file
hash and immutable identity-file hash. Thus nested safe failure diagnostics are
unambiguously attempt-bound without changing historical 001 diagnostics.

Operator reruns without a new grant return
`NO_SEPARATELY_AUTHORIZED_NEXT_ATTEMPT` before credentials, DB or HTTP work.
A verified success can be fetched again without dispatch. Success does not
permit replacement generation; neither success nor failure creates a next grant.

The public generation entry requires an authorized attempt. The private shared
core remains directly exercisable by isolated fake tests of legacy generation
and failure diagnostics; the operator never calls its unnumbered test mode.
