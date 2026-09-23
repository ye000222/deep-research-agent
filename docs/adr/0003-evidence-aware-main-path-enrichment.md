# ADR 0003：Evidence-Aware 主路径 Query Enrichment 默认关闭（Phase 14 收口）

状态：已接受（Phase 14.5 Closeout）

## 决策

Main-path Evidence-Aware Query Enrichment（Phase 14.2：ResearchContext 自动向每个主
SearchTarget 追加 query hints）从 V1 默认行为中退出：

`EVIDENCE_AWARE_CONTEXT_ENABLED` 默认值为 `false`。默认运行路径回到
Planner → SearchTarget(base query) → Provider，逐字节与 Phase 12.4 基线一致
（不访问 resolver、不产生 `research.query.context.enriched` 事件）。

这是默认策略调整，不是删除能力；feature flag 显式置 `true` 即可完整恢复 14.2 行为。

## 保留（通用研究诊断/适配基础设施）

- ResearchContext / ResearchContextResolver / ResearchContextEnricher 抽象
- Evidence Failure Classification 与 Research Need refinement
- ClosureFeedback.metadata、ResearchQueryIntent / ResearchQueryPlan / QueryCandidate
- Phase 14.1 Candidate Adaptation（Evidence Failure → Research Need → QueryCandidate
  适配路径继续生效）
- 诊断事件 `research.context.recorded` / `research.need.refined` /
  `query_candidate.generated` 不因本决策而停止

## 默认关闭的部分

仅 ResearchContext → 主研究 SearchTarget 的自动 hint appending（14.2 主路径集成实验）。

## 证据

Phase 14.3（3+3 A/B，同任务）：

- baseline coverage ≈ 0.4524，candidate（14.2 开启）≈ 0.3929；
- accepted evidence 14.33 → 10.67；
- 未产生正向质量收益。

Phase 14.4（因果归因）：

- mean_overlap_ratio = 1.0，unique_hint_contribution = []；
- 主路径 enrichment 与 14.1 candidate hints 高度重复，未提供新的研究信息。

Historical Baseline Regression Audit（Phase 14.4 Supplemental）：

- 不存在 "0.85 → 0.45" 的行为回归（跨任务比较）；同任务当前口径下
  当前 baseline（0.4524）优于历史 cohort（replay 0.369）；
- 14.2 的开启/关闭不是覆盖率差异的来源；
- 后续问题在 Phase 15 的 Evidence → Accepted → Verification → Closure → Coverage
  转换效率，而非继续扩展 query enrichment。

## 可逆性

feature flag 保持可用。未来仅在同时满足以下两个条件时才考虑重新默认开启：

1. 出现 distinct context consumer（不同于 14.1 candidate hints 的消费者）；
2. 新的同任务 benchmark 证据显示正向质量收益。

## 关联治理

- 比较契约与指标版本：`apps/api/app/domain/benchmark_comparability.py`
  （COMPARABLE / PARTIALLY_COMPARABLE / NOT_COMPARABLE / UNKNOWN、
  `metric_definition_version = coverage-v1`、replay-first、V1 baseline 资格规则）。
- 历史比较必须 replay-first：stored 版本 != 当前版本时禁止直接比较存值。
