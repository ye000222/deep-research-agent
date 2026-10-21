# PATH-C Attempt 002 host authorization entry

This is authorization publication, not generation. The human governance
decision approves a separately numbered Attempt 002 after postmortem,
observability closure, and multi-attempt authorization isolation. Attempt 001
is permanently consumed and remains unchanged. Its root cause remains
unrecoverable from existing evidence; this entry does not claim it was fixed.

The PowerShell entry discovers the resolved Compose API artifacts volume and
an already-local API image. It does not pull, build, retry, or start services.
Its ephemeral container has no network, a read-only root filesystem/current
checkout, and only the canonical artifacts volume writable.

The Python entry verifies frozen contracts and the authoritative safe context
artifact, reuses the existing authorization model and history reader, checks
the exact supplied parent SHA-256, and rejects any later-attempt evidence.
The context hash is a frozen reference (not a fabricated/reconstructed owner
scope); real generation separately verifies authenticated scope and live
Provider drift through its unchanged existing path.

Before publication it repeats parent/absence checks. A fully serialized,
validated, fsynced temporary file is atomically hard-linked to the final name:
`authorizations/attempt_002.authorization.json`. An existing target cannot be
replaced. Temporary files are removed; historical state is never written.
The directory is fsynced in the Linux host container. No attempt state,
dispatch receipt, raw response, metadata or task manifest is created.

The entry prints identifiers, hashes, governance rationale and dispatch count
zero, then exits. It has no generation-entry invocation. Host execution and
real authorization publication were not performed by Codex.
