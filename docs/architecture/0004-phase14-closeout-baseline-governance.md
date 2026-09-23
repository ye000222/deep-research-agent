# 0004：Phase 14 收口与 Benchmark 基线治理（Phase 14.5）

状态：生效。本文件固化 Phase 14.3 / 14.4 / Historical Baseline Regression Audit
的实验结论为后续阶段的治理规则。规则的程序化表达在
`apps/api/app/domain/benchmark_comparability.py`（纯 domain，无 DB/Provider 依赖）。

## 1. V1 默认行为（Task A–D）

- `EVIDENCE_AWARE_CONTEXT_ENABLED` 默认 `false`（config / docker-compose / .env.example
  一致）。默认路径 Planner → SearchTarget(base query) → Provider，flag=false 时
  provider query 与 SearchTarget 逐字节一致，且不产生
  `research.query.context.enriched` 事件（门控在 resolver 访问之前，无副作用）。
- 显式 `=true` 完整保留 14.2 行为，供未来实验使用。
- ResearchContext 架构、14.1 Candidate Adaptation、诊断事件
  （`research.context.recorded` / `research.need.refined` / `query_candidate.generated`）
  全部保留。
- Context 生命周期：`research_context_by_question`（usage_snapshot，
  ResearchContextResolver 唯一所有者）在 flag=false 后无主路径消费者，仅作为
  runtime 诊断状态继续存在；不实现 TTL/invalidation/versioning（Phase 14.4 实测
  stale_context_usage_count = 0，stickiness 无证据）。它不进入 Planner / Ranking /
  Budget / Provider / Coverage / Closure。

## 2. Comparability Contract（Task E/F）

任意两个 Run 比较前，先构造双方 `BenchmarkComparisonKey` 并调用 `compare_keys`：

- identity 字段（硬）：`benchmark_id`、`normalized_goal`、`metric_definition_version`、
  `plan_shape_policy`、`evidence_aware_context_enabled`（实验关键配置）。
- identity 字段（软）：`benchmark_version`、`tier`、`plan_template_run_id`。
- 明确排除：run_id、timestamp、瞬时 provider health、random request id。
- 判定：硬字段已知且不同 → `NOT_COMPARABLE`；任一身份字段缺失 → `UNKNOWN`
  （不猜测）；仅软字段不同 → `PARTIALLY_COMPARABLE`；全部一致 → `COMPARABLE`。
- 语言策略：非 `COMPARABLE` 的报告禁止使用 regression / improvement /
  quality increased / quality decreased 等强因果纵向语言，必须改用
  cross-run difference / exploratory comparison / not regression-safe。
- 比较分级不阻止用户查看任何两个 Run。

## 3. Metric Definition Versioning 与 Replay-first（Task G/H）

- 当前指标语义版本：`coverage-v1`（覆盖 coverage 语义、required_sources 双源规则、
  evidence acceptance 语义三者的联合修订标识）。语义变化时 bump。
- 版本化之前持久化的 run 一律视为 `unversioned-legacy`（未知语义），
  **不得**默认等同当前版本——audit 实测旧时代同任务 stored mean 0.6191 按当前口径
  replay 只有 0.369（虚高 +0.2501），头条 0.8+ run 则完全复现（漂移 0）。
- 历史 vs 当前比较：`replay_policy()` 判定。stored 版本 == 当前版本才允许直接
  比较存值；否则必须用当前定义对历史持久化数据只读重算，报告需并列展示
  stored 与 current-definition replay 两个值，禁止静默混用。

## 4. 正式 V1 Benchmark Baseline 规则（Task I）

Phase 14.3 的三个临时 baseline run（01a0bf8b-fdc4 / 01a0bf8b-fdde /
01a0bfa7-3c70）**不**转为长期 Golden Baseline：它们的 benchmark_id 为空
（未注册基准）。未来 Phase 15 开始前，正式 baseline 必须满足
`evaluate_baseline_requirements()` 全部条件：

1. registered benchmark（benchmark_id 非空）
2. fixed benchmark version
3. fixed goal（normalized_goal）
4. fixed tier
5. fixed plan policy（plan shape policy）
6. fixed metric definition（metric_definition_version 已知且为当前版本）
7. known feature flags（含 `evidence_aware_context_enabled`）

本阶段只定义规则，不重新跑 baseline。

## 5. 收口结论

Phase 14 = CLOSED。14.0 证据失败诊断、14.1 Evidence-aware candidate adaptation、
14.2 主路径集成实验、14.3 A/B 验证、14.4 因果归因、14.4 Supplemental 历史审计、
14.5 收口。后续效率问题（Evidence → Accepted → Claim Verification → Gap Closure →
Coverage）归 Phase 15。
