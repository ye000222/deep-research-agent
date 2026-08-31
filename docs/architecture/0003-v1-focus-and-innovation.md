# V1 方向重定：Evidence-Aware & Memory-Aware Research Agent

修订日期：2026-08-26

## 1. 产品定位

项目不再以“能够搜索并生成报告”作为主要卖点。搜索、网页阅读、总结和引用是基础能力，不能单独构成创新点。

V1 的定位冻结为：

> 面向长周期研究的证据驱动、自适应上下文管理 Research Agent

英文定位：

> Evidence-Aware & Memory-Aware Deep Research Agent

V1 只重点打磨三个互相闭环的能力：

1. **Evidence Graph**：结论、证据、来源、支持/反驳关系和冲突可追溯。
2. **Coverage-driven Research Loop**：按研究维度覆盖度、信息缺口和边际信息增益决定继续、补搜、重规划或停止。
3. **Context Budget Manager**：在有限上下文预算内动态选择、压缩、重新检索和保护关键证据。

长期 Memory 是 Context Manager 的持久化基础，但跨任务经验固化、衰减和遗忘不作为 V1 的首要展示亮点，先完成当前 Run 的 Working Memory、Task Memory 和 Evidence Retrieval。

## 2. V1 的核心闭环

```text
研究目标
  ↓
研究维度与验收标准
  ↓
Evidence Graph / Coverage Map
  ↓
缺口检测（missing / weak / stale / conflict / diversity）
  ↓
信息增益驱动的下一动作
  ├─ 查已有 Evidence / Memory
  ├─ 搜索 Web
  ├─ 阅读来源
  ├─ 提取并保存关系化证据
  └─ 运行受控数据分析
  ↓
Context Budget Manager 装配最小充分上下文
  ↓
Evaluator 重算覆盖度、来源质量、冲突和边际信息增益
  ↓
继续研究 / 重规划 / 带限制写作 / 停止
```

页面三栏与算法模块对应：

- `RESEARCH PLAN`：研究维度、Coverage Map 和动态 Gap。
- `AGENT ACTIVITY`：Context Manager、Tool Decision、信息增益和可审计决策理由。
- `EVIDENCE`：Evidence Graph、Claim 关系、来源独立性和 Conflict。

## 3. Evidence Graph 设计冻结

### 3.1 图关系

最小关系集合：

- `supports`：证据直接支持 Claim
- `contradicts`：证据反驳 Claim 或与同范围事实冲突
- `supplements`：证据补充范围、定义、时间或上下文
- `derived_from`：分析结果或复合 Claim 的来源关系
- `unresolved`：冲突暂时无法消解，必须进入研究限制

### 3.2 数据原则

- Search Snippet 永远不是 Evidence。
- Evidence 必须绑定 Source Snapshot、Source Chunk、Exact Quote 和定位信息。
- 核心 Claim 必须满足“双独立来源”或“单一高可信一手来源”规则。
- 转载同一原始材料的多个 URL 只计一个独立来源。
- 不同市场定义、时间、地区、单位的数字不能直接合并；先创建 Conflict。
- Conflict 不允许静默选择；必须解析为范围差异、权威性差异或 unresolved。

### 3.3 V1 交付顺序

1. 创建 `claims`、`source_snapshots`、`source_chunks`、`conflicts` 和关系表。
2. 将当前 `research_evidence` 迁移为 Graph 的边和兼容视图。
3. 增加 Claim 状态：candidate、supported、partial、disputed、stale、rejected。
4. 将 Conflict 作为 Gap 的一种来源，触发专门验证任务。
5. 报告引用从 Claim → Evidence → Chunk → Snapshot → URL 完整回溯。

## 4. Coverage 与 Information Gain

### 4.1 Coverage Map

每个研究问题拆为带权 Acceptance Dimension。维度状态固定为：

- `0`：无可验证 Evidence
- `0.5`：弱证据、单一来源、范围部分匹配或过期
- `1`：满足验收标准

`QuestionCoverage = Σ(DimensionWeight × DimensionStatus) / Σ DimensionWeight`

前端必须能解释总覆盖度由哪些维度构成，并展示每个维度的已知、未知和证据数量；禁止只展示一个不可解释的 LLM 百分比。

### 4.2 信息增益

每次 Search、Read 或 Extract 完成后计算相对上一轮的边际收益。V1 最小可解释指标：

```text
InformationGain =
  0.35 × 新增有效 Evidence 比例
+ 0.25 × 新增 Claim 比例
+ 0.20 × 新增独立 Source 比例
+ 0.20 × Coverage 提升
- 重复 Evidence 惩罚
```

该分数必须保存到 Action/Evaluation 事件中，并用于下一动作排序。它不是替代质量门的唯一指标。

### 4.3 停止规则

质量停止仍需满足全部硬门槛：关键维度覆盖、来源质量、独立来源、引用完整度和高严重度冲突处理。

在硬门槛可接受的前提下，若连续两轮 `InformationGain < 0.10`、没有新增关键 Evidence 且没有可行动 Gap，则允许 `stagnation` 或 `sources_exhausted` 停止，并在报告中披露限制。

## 5. Context Budget Manager

### 5.1 每次调用的预算分配

Context 不再按“把所有相关内容塞进 Prompt”实现。每次调用先根据模型 Context Window 计算硬预算，再分配：

```text
System / Safety / Output Schema  10%
Research State / Current Gap     15%
Recent Actions                    10%
Evidence Cards / Source Chunks   40%
Working / Task Memory             15%
Buffer / Output Reserve           10%
```

实际比例可由模型能力快照调整，但必须记录分配版本和最终 Token 使用量。

### 5.2 永不丢失

- 用户目标、时间、地区、语言和禁止项
- 当前 Gap 和 Acceptance Criteria
- 关键数字、单位、日期、实体和 Exact Quote
- Claim、Evidence、Snapshot、Chunk 和 Citation ID
- 高严重度 Conflict、预算和停止条件

### 5.3 压缩顺序

1. 删除重复 Search Snippet、重复事件和低分候选。
2. 用来源 Owner 和 Content Hash 去重，保留观点多样性。
3. 旧 Action 压缩为阶段摘要。
4. Source Chunk 压缩为带 provenance 的 Evidence Card。
5. Evidence Card 压缩为 Question Summary。
6. 仍超限时返回 `CONTEXT_BUDGET_INSUFFICIENT`，禁止静默截断关键证据。

每次调用写入 Context Manifest：选入项、拒绝项、压缩项、Token、Prompt Hash、配置版本和 provenance；不保存 API Key 或未经脱敏的完整 Prompt。

## 6. 当前代码与新方向的对应关系

已具备：

- 阶段级 LangGraph StateGraph 和 PostgreSQL Checkpoint。
- 基础 Research Loop、Coverage、Gap、Evidence Score 和报告引用。
- ResearchState Snapshot/Patch 和可审计活动轨迹。

待补齐：

- 细粒度 Graph 节点和信息增益字段。
- Evidence Graph 的 Claim、Snapshot、Chunk、Conflict 关系。
- 可解释 Coverage Map，而非单一总百分比。
- Context Manifest、预算分配、压缩和 provenance。
- 当前 Run 的 Working/Task Memory 检索与固化。

## 7. 后续版本边界

V1 完成上述三个亮点后，再进入 V1.1/V2：

- 跨任务 Semantic / Episodic / Procedural Memory。
- Memory 衰减、过期、Superseded、Forgotten 和重建索引。
- Embedding/pgvector、PDF/文件、多 Agent 并行和完整 Human-in-the-loop。

这些能力不能提前稀释 V1 的三个核心亮点。
