# PATH-C Attempt 003 host authorization

This is the separately approved next attempt, not a retry/reset of Attempt 002.
Attempt 002 remains permanently consumed. Its READ_TIMEOUT failure exposed a
generic timeout-policy wiring defect, now corrected and regression-tested.
Attempt 001's root cause remains unrecoverable from existing evidence. No
Provider defect or guaranteed Provider success is claimed.

The existing Attempt-002-named PowerShell/Python entry is minimally generalized
with an explicit approved-attempt parameter restricted to 2 or 3. It never
automatically increments, skips history, approves Attempt 004, or executes the
generation entry. No new authorization model/framework is introduced.

In the same host execution, canonical artifacts storage is read directly:

- The original legacy Attempt 001 state is verified against its frozen SHA.
- Existing multi-attempt history/authorization readers validate Attempt 002's
  identity, raw authorization file hash, parent linkage, state and reservation.
- Both attempts must be CONSUMED_FAILED; Attempt 002 must have one physical
  outbound and TIMEOUT / MODEL_TIMEOUT / READ_TIMEOUT with HTTP200 evidence.
- Historical authorization binding must match the frozen context/Provider.
- Attempt 002's exact terminal file bytes determine the parent SHA for the new
  authorization; no supplied summary or manually copied hash substitutes for it.
- Any later-attempt evidence, existing next authorization, malformed/ambiguous
  history or numbering mismatch blocks publication without overwriting files.
- Current-checkout imported timeout policy must resolve to connect10/read90/
  write90/pool90. AST checks verify PATH-C's actual builder usage, absence of a
  later timeout assignment in _run, and Worker's use of the same policy.

The existing immutable authorization contract is reused. A complete validated
temporary file is fsynced and atomically hard-linked, no overwrite, to
`authorizations/attempt_003.authorization.json`. History integrity is checked
again immediately before publication and original historical file hashes are
checked afterward. No attempt_003 identity/state/reservation directory is
initialized. The operator freezes authorization only, then stops.

The host wrapper uses canonical Compose volume discovery, a local image only,
no network, and read-only current-checkout code. The only writable mount is the
canonical artifacts volume. It propagates failure without retry or generation.

Deterministic tests use exclusively synthetic temporary histories; no real
authorization or canonical history was accessed/created by Codex. Test coverage
includes lineage/hash/grant/receipt/binding corruption, numbering, existing
evidence, timeout drift, write failure, zero dispatch and historical immutability.

The existing filenames retain `_002` for compatibility; passing approved 3 is
required for this new governance decision. Actual identifiers and parent hash
are reported only after a successful human host execution.
