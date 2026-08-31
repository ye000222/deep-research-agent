# ADR 0002：BYOK Provider Gateway 与凭据隔离

状态：已接受

## 决策

用户保存 API 协议、Endpoint 与精确模型；API Key 由服务端使用 AES-256-GCM 加密持久化。运行绑定只保存
Provider 能力快照与不可变凭据版本引用，不保存明文 API Key。所有模型调用先转换为 Provider-neutral
request，再由 Adapter 在调用边界解密并执行。

## 原因

不同 AI API 在 Tool Calling、结构化输出、流式事件、用量统计和取消语义上并不等价。显式 Capability
Matrix 可以在运行前拒绝不满足最低能力的模型，或记录降级策略，避免“OpenAI-compatible”被误认为完全兼容。

## 安全边界

- API Key 不进入 LangGraph State、Prompt、日志、Redis、Celery Payload 或事件流。
- 数据库只保存 ciphertext、12 字节 nonce、AAD/主密钥版本、HMAC 指纹和末四位提示。
- 开发环境在被 Git 忽略的 artifacts/.secrets 中生成持久主密钥；生产环境必须注入
  SECRET_MASTER_KEY_BASE64，不允许依赖容器文件系统临时密钥。
- 浏览器只保存 HttpOnly 签名会话 Cookie；读取配置时只返回模型元数据和密钥末四位。
- 更新密钥创建新 Credential Version 并撤销旧版本；普通模型字段更新不触碰密钥。
- 前端使用密码输入框，不写入 localStorage、sessionStorage 或 URL。
