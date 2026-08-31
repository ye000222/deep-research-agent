# ADR 0001：PostgreSQL 作为业务事实库与 Checkpoint 存储

状态：已接受

## 决策

V1 使用 PostgreSQL 18。业务数据位于 `deep_research` 数据库；LangGraph Checkpoint 位于独立的
`deep_research_checkpoint` 数据库，并使用官方 `langgraph-checkpoint-postgres` 实现。

## 原因

研究任务、来源、证据、Claim 与 Citation 具有强关系和审计需求；PostgreSQL 的事务、JSONB、全文能力与
后续 pgvector 扩展能减少基础设施分裂。Checkpoint 与业务表分库可避免内部序列化结构侵入业务模型，也能
独立迁移、备份和限权。

## 约束

- Checkpoint 只保存可序列化的轻量 Agent State。
- 整页正文、API Key、数据库连接和客户端对象不得进入 State。
- Checkpoint schema 通过显式管理命令初始化，不在 Web 进程启动时自动变更。
- 业务表使用 Alembic；Checkpoint 表由官方 saver 的 `setup()` 管理。
