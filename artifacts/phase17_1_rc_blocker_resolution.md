# Phase 17.1 RC Blocker Resolution

## A. Lease root cause

Classification: `EXPECTED_LEASE_HANDOFF`, not a real lease-expiry P0.

For run `01a0c9a8-9cd5-7844-921d-069730a4c766`, the worker acquired a 900-second lease at `15:07:46.505Z`. The worker heartbeat/renewal interval is 30 seconds. The run was explicitly terminalized by user cancellation at `15:12:41.411Z`; cancellation cleared `lease_owner`, `lease_until`, and `worker_task_id`. The still-running worker logged `ResearchLeaseLostError` at `15:12:47.915Z`, after about 328 seconds of task runtime—well below the 900-second TTL.

There was one `run.started`, no `run.recovered`, no duplicate task receipt for this run, and no evidence of duplicate execution or lost writes. The final run state was `cancelled/user_cancelled`; the lease-loss log was a stale-worker write after terminalization and did not cause the final state.

## B. No-accepted-evidence root cause

Run `01a0c9ad-2018-7730-a8ef-1066181f4cc7` was a `quick` tier failure-path/deployment smoke. Run-stamped data shows 2 searches, 2 readable events, 0 candidate evidence, 0 Accepted Evidence, 2 `question_budget` selection skips, 2 source fetch failures, and no report row. It terminated with `REPORT_NO_ACCEPTED_EVIDENCE`.

This is Case B: the system correctly refused to fabricate an unsupported Report. It is not a Report pipeline P0. The second quick attempt (`01a0c9c7-5081-7936-aeb2-700e74bfb1b4`) reproduced the same expected no-accepted-evidence behavior. Quick tier is therefore unsuitable as a success-path smoke under this research workload.

## C. Original P0 classifications

Both original P0 classifications were too strong:

- `ResearchLeaseLostError`: not a production lease-reliability P0; it is an expected stale-worker handoff after explicit cancellation.
- `REPORT_NO_ACCEPTED_EVIDENCE`: not a Report subsystem P0 for the quick run; it is the intended no-evidence invariant.

The actual release gap was the absence of a separately defined standard success-path deployment smoke.

## D. Minimal fix

No production fix was required and no production code was modified. The resolution was limited to root-cause classification, smoke-path separation, and release-gate evidence.

## E. Success-path deployment smoke

Run `01a0c9c9-f951-76f0-a801-a10b65806236` used the frozen RC configuration:

- `reference_config_id=v1-pre-rc-reference-1`
- `evidence_aware_context_enabled=false`
- `independent_source_targeting_enabled=true`
- `evidence_input_quality_enabled=false`
- `tier=standard`

Result: `completed_with_limitations`. It produced 5 persisted Evidence rows, 3 Accepted Evidence rows, persisted one Report, and the application `ResearchRunService` fetched that Report with `status=verified` and `verification_result.verified=true`. The run completed through `sources_exhausted → report.writing_started → report.verified → run.completed`.

Success-path smoke: **PASS**.

## F. Failure-path invariant

The quick run retained as the failure-path smoke has 0 Accepted Evidence, no fabricated Report, and terminal reason `REPORT_NO_ACCEPTED_EVIDENCE`.

Failure-path invariant: **PASS**.

## G. Final release gates

| Gate | Result |
|---|---|
| ResearchLeaseLostError | NON-BLOCKING / EXPECTED HANDOFF |
| REPORT_NO_ACCEPTED_EVIDENCE | EXPECTED FAILURE |
| Standard success-path deployment smoke | PASS |
| No-evidence failure-path invariant | PASS |
| Production lease/report fix | NOT REQUIRED |

Existing non-P0 provenance, working-tree hygiene, and research-quality limitations remain unchanged and are not reworked in this blocker-resolution task.

## H. RC decision

**V1 RC READY WITH KNOWN ISSUES**.

No V1 publication, production tag, Phase 16, V2, or Research Intelligence work was started.

STOP.
