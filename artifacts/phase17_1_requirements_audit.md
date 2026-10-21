# Phase 17.1 Result — V1 Release Candidate Hardening

## Decision

**V1 RC NOT READY**

This is the corrected result after auditing all original requirements 1–52 and executing the missing deployment smoke verification. No new Research Intelligence work was performed.

## Smoke evidence

- Standard smoke `01a0c9a8-9cd5-7844-921d-069730a4c766`: Worker received and executed the Run, but the task emitted `ResearchLeaseLostError`; it was cancelled and is not a pass.
- Quick smoke `01a0c9ad-2018-7730-a8ef-1066181f4cc7`: Worker received the Run and produced Search/Reader/Evidence/Closure events, but the Run ended as `failed / REPORT_NO_ACCEPTED_EVIDENCE`; no Report row was created, so Report fetch/verification failed.
- Docker Compose, API health, Worker, Dispatcher, Beat, Postgres, Redis and migration head were independently verified.

## A. Frozen RC Configuration

| Field | Value | Result |
|---|---|---|
| `reference_config_id` | `v1-pre-rc-reference-1` | Partial: legacy qualification Runs lack the stamp |
| `evidence_aware_context_enabled` | `false` | Pass |
| `independent_source_targeting_enabled` | `true` | Pass |
| `evidence_input_quality_enabled` | `false` | Pass |
| tier | `standard` for qualification | Pass |
| metric definition | `coverage-v1` | Pass |
| RC version | `v1.0.0-rc.1` | Pass |

## B. Configuration Drift Audit

Configured API/Worker defaults and current launcher resolve consistently. Historical Run metadata is not fully reproducible because explicit `reference_config_id` is absent on legacy qualifying Runs. Therefore configuration values pass, but provenance completeness does not.

## C. Regression Results

The existing Phase 17.0 qualification set remains the regression reference: 10 qualifying terminal Runs across the four required families. No new Phase 17.0 qualification was run. Historical regression data shows no catastrophic quality regression, but this does not override the failed deployment smoke.

## D. Reliability Audit

Historical qualification terminalization and verified-report rate were acceptable (10 terminal, 9/10 verified). The fresh smoke exposed a Worker lease-loss path and a no-accepted-evidence report failure. Reliability is therefore not release-clean.

## E. State / Data Integrity

Migrations are at `20260918_0025`, the single Alembic head. q1 remains a documented projection inconsistency; its complete user-visible impact trace was not established. No confirmed data-corruption blocker was found, but the state audit is not fully closed.

## F. Report Integrity

Historical reports: 9/10 verified. Fresh quick smoke: `REPORT_NO_ACCEPTED_EVIDENCE`, no report persisted, no fetch/verification possible. This fails the required deployment smoke report criterion.

## G. Variance Risk

Observed family variance remains documented: model comparison SD ≈ 0.101, multimodal manufacturing SD ≈ 0.147, low-altitude SD ≈ 0.178. It is currently classified primarily as external/provider and execution variance; complete per-signal attribution is still partial.

## H. Known Limitations

- Low-altitude economy: coverage `0.2500 ± 0.1782`; severe family limitation, P2, non-blocking only after deployment/reliability blockers are closed.
- Run-to-run variance: expected external/execution variance, P2.
- Per-claim verified claims = 0: V2 architectural limitation.
- q1 requirement projection inconsistency: P2 triage issue, no release-impact evidence established yet.
- Historical Phase 15.2 lint exclusion: P1 hygiene item, documented.
- Legacy Runs missing explicit `reference_config_id`: P1 provenance item, new launchers stamp it.

## I. Release Hygiene

`pytest -q`, `ruff check .`, `mypy apps/api/app`, Phase 17 script mypy and `git diff --check` pass. The historical Phase 15.2 analyzer is explicitly excluded and documented. The working tree still contains many untracked phase/debug artifacts, so release artifact separation is not clean.

## J. Deployment Smoke

Service-level smoke passed. Full required smoke did not:

```text
services start             PASS
API healthy                PASS
Worker receives task       PASS
DB accessible              PASS
Run reaches terminal       PASS, but failed
Report can be fetched      FAIL
Report verified            FAIL
```

## K. Documentation / Reproducibility

Configuration, Compose, env example and known-issue documentation exist. A clean-clone, end-to-end reproduction guide and complete release artifact exclusion list were not fully verified.

## L. Blocker Classification

| Issue | Severity | Release blocker | Resolution |
|---|---|---:|---|
| Deployment smoke cannot produce/fetch a verified Report | P0 | Yes | Fix or diagnose report/no-accepted-evidence path, then rerun smoke |
| Worker `ResearchLeaseLostError` observed in smoke | P0 | Yes pending diagnosis | Verify lease/deadline/terminalization path |
| Legacy qualifying Runs missing `reference_config_id` | P1 | No, if accepted | Preserve as historical provenance limitation or rerun only if required |
| Working tree contains temporary/untracked release material | P1 | Pending release packaging | Produce explicit release inventory/exclusion |
| q1 projection inconsistency | P2 | No evidence of release impact | Keep triage record |
| Low-altitude family weakness | P2 | No | Known limitation; post-V1 |
| Run-to-run variance | P2 | No | Known limitation; post-V1 |
| Per-claim verification unavailable | V2 | No | V2 architecture backlog |

## M. Final Gates

| Gate | Result |
|---|---|
| pytest -q | PASS |
| ruff check . | PASS with documented historical exclusion |
| mypy apps/api/app | PASS |
| Phase 17 script mypy | PASS |
| git diff --check | PASS |
| DB migration | PASS |
| Configuration values | PASS |
| Provenance completeness | PARTIAL |
| Deployment smoke | FAIL |
| Report generation/fetch/verification on fresh smoke | FAIL |
| No P0 blocker | FAIL |

## N. Ten required final answers

1. **Reference Configuration 是否完全冻结？** 不完全。当前默认值和 Worker/API 解析一致，但历史 qualifying Runs 缺少显式 `reference_config_id`。
2. **API / Worker / Launcher / Run metadata 是否一致？** 当前 API/Worker/Launcher 一致；历史 Run metadata 不完整，因此整体为 PARTIAL。
3. **是否存在 configuration drift？** 当前 flags 未发现 drift；存在 provenance drift/缺失，而非已确认的 flag 值漂移。
4. **Run 是否可以可靠终止？** 历史 Runs 能终止，但 smoke 出现 `ResearchLeaseLostError`，deadline 路径未充分证明，不能判定可靠。
5. **Report 是否可以可靠生成并 verified？** 历史 9/10 verified；fresh smoke 失败且无 Report，不能判定 release-ready。
6. **是否存在数据一致性/状态一致性的 release blocker？** 未发现已确认的数据损坏；但 q1 影响未完全追踪，当前确定的 release blockers 来自 smoke/reliability。
7. **Docker deployment 是否可以从干净配置启动？** 服务可以启动并健康；完整 Run→终态→Report smoke 未通过。
8. **全仓质量 gates 是否全部通过？** 静态/测试 gates 通过，历史脚本有明确 exclusion；release packaging hygiene 仍未收口。
9. **当前有哪些 Known Limitations，但不阻止 V1？** 低空经济弱、run variance、per-claim verification=0、q1 projection inconsistency、历史 provenance/lint 均已记录；但只有在 P0 smoke blockers 关闭后才可作为非阻塞项接受。
10. **当前版本是否已经达到 V1 RC READY？** 否。当前为 **V1 RC NOT READY**。

## STOP

Phase 17.1 stops here. No public release, production tag, deployment, new qualification, Phase 16 or V2 work is started.
