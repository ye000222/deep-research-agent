## 变更内容

-

## 验证

- [ ] Python tests
- [ ] Python lint / typecheck
- [ ] Web build / typecheck
- [ ] Docker Compose config

## Agent 边界检查

- [ ] State 仍只保存轻量、可序列化数据
- [ ] API Key 未进入日志、Prompt、事件、队列或 Checkpoint
- [ ] 新 Tool 具备策略、预算、幂等和错误归一化
- [ ] 新结论路径具备 Evidence / Citation 可追溯性
