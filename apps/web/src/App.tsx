import {FormEvent, useEffect, useMemo, useRef, useState} from "react";

import {ProviderProfileForm} from "./ProviderProfileForm";

type ApiHealth = "checking" | "ready" | "unavailable" | "version_mismatch";
type PlanStatus = "done" | "active" | "partial" | "degraded" | "pending" | "blocked";
type BudgetTier = "quick" | "standard" | "deep";
type ProgressStageState = "done" | "active" | "pending" | "failed";

type ProgressStage = {
  key: string;
  label: string;
  weight: number;
};

const PROGRESS_STAGES: ProgressStage[] = [
  {key: "init", label: "目标分析", weight: 0.05},
  {key: "plan", label: "研究计划", weight: 0.10},
  {key: "research", label: "搜索与阅读", weight: 0.25},
  {key: "evaluate", label: "证据评估", weight: 0.25},
  {key: "replan", label: "缺口补充", weight: 0.15},
  {key: "write", label: "报告生成", weight: 0.15},
  {key: "verify", label: "最终核验", weight: 0.05},
];

const PHASE_TO_STAGE: Record<string, string> = {
  initializing: "init",
  init: "init",
  analyze_query: "init",
  planning: "plan",
  plan: "plan",
  researching: "research",
  research: "research",
  evaluating: "evaluate",
  evaluate: "evaluate",
  writing: "write",
  write: "write",
  verifying: "verify",
  verify: "verify",
  terminal: "verify",
  finalize: "verify",
};

const BUDGET_STOP_REASONS = new Set([
  "research_budget_exhausted",
  "page_budget_exhausted",
  "search_budget_exhausted",
  "token_budget_exhausted",
  "iteration_budget_exhausted",
  "deadline_exhausted",
  "logical_query_budget_exhausted",
  "provider_request_budget_exhausted",
  "fetched_page_budget_exhausted",
  "extracted_page_budget_exhausted",
  "action_budget_exhausted",
]);

function isBudgetStopReason(reason: string | null | undefined): boolean {
  return Boolean(
    reason && (BUDGET_STOP_REASONS.has(reason) || reason.endsWith("_budget_exhausted")),
  );
}

function budgetStopLabel(reason: string | null | undefined): string {
  const labels: Record<string, string> = {
    research_budget_exhausted: "研究预算已停止扩展",
    page_budget_exhausted: "页面预算已耗尽",
    search_budget_exhausted: "搜索预算已耗尽",
    token_budget_exhausted: "Token 预算已耗尽或触发调用前保护",
    iteration_budget_exhausted: "研究轮次预算已耗尽",
    deadline_exhausted: "研究截止时间已到",
    logical_query_budget_exhausted: "逻辑查询预算已耗尽",
    provider_request_budget_exhausted: "上游请求预算已耗尽",
    fetched_page_budget_exhausted: "页面抓取预算已耗尽",
    extracted_page_budget_exhausted: "页面抽取预算已耗尽",
    action_budget_exhausted: "调度动作预算已耗尽",
  };
  return reason ? labels[reason] ?? reason : "";
}

type ResearchRun = {
  run_id: string;
  status: string;
  phase: string;
  state_version: number;
  termination_reason: string | null;
  budget_snapshot: Record<string, unknown>;
  usage_snapshot: Record<string, unknown>;
  quality_snapshot: Record<string, unknown>;
  event_url: string;
  created_at: string;
  started_at: string | null;
};

type AgentEvent = {
  seq: number;
  timestamp: string;
  phase: string;
  event_type: string;
  public_summary: string;
  refs: Record<string, unknown>;
  metrics: Record<string, unknown> | null;
};
type EvidenceItem = {
  evidence_id: string;
  question_id: string;
  claim: string;
  exact_quote: string;
  relation: string;
  source_title: string;
  source_url: string;
  source_domain: string;
  source_reliability: number;
  relevance: number;
  confidence: number;
  evidence_score: number;
  accepted: boolean;
  rejection_reason: string | null;
};

type ReportCitation = {
  citation_number: number;
  evidence_id: string;
  question_id: string;
  claim: string;
  exact_quote: string;
  source_title: string;
  source_url: string;
  source_domain: string;
  source_content_hash: string;
  accessed_at: string;
};

type ResearchReport = {
  report_id: string;
  run_id: string;
  version: number;
  title: string;
  final_markdown: string;
  limitations: string[];
  verification_result: Record<string, unknown>;
  status: string;
  citations: ReportCitation[];
};

type ContextItemMetric = {
  item_type: string;
  token_count: number;
  selected: boolean;
  protected: boolean;
  selected_reason_code: string;
  compression_level: string;
  source_ref_type: string | null;
  compression_artifact_id: string | null;
};

type CompressionArtifactMetric = {
  artifact_id: string;
  compression_level: string;
  token_before: number;
  token_after: number;
  validation_status: string;
  provenance_refs: string[];
};

type ContextManifest = {
  manifest_id: string;
  node_name: string;
  model: string;
  input_budget: number;
  output_reserve: number;
  selected_count: number;
  rejected_count: number;
  compressed_count: number;
  token_before: number;
  token_after: number;
  compression_ratio: number;
  truncated: boolean;
  created_at: string;
  items: ContextItemMetric[];
  compression_artifacts: CompressionArtifactMetric[];
};

type PlanItem = {
  number: string;
  questionId?: string;
  title: string;
  status: PlanStatus;
  reason?: string;
  coverage?: number;
  acceptedEvidence?: number;
  independentSources?: number;
};

type CoverageDimension = {
  dimension_key: string;
  question: string;
  priority: number;
  coverage: number;
  accepted_evidence: number;
  independent_sources: number;
  missing_reasons: string[];
  requirement_statuses: Array<{
    dimension_key: string;
    criterion: string;
    coverage: number;
    accepted_evidence: number;
    independent_sources: number;
    required_sources: number;
  }>;
};

type KnowledgeLedger = {
  known: Array<Record<string, unknown>>;
  coverage_map: CoverageDimension[];
  quality: Record<string, unknown>;
};
type GapLedger = {
  gaps: Array<Record<string, unknown>>;
  open_count: number;
};
type ActionLedger = {
  next_action: Record<string, unknown> | null;
  events: AgentEvent[];
};
type EvaluationSnapshot = Record<string, unknown> & {
  verdict?: string;
  source_quality?: number;
  citation_support?: number;
};
type LLMCall = {
  call_id: string;
  node: string;
  adapter: string;
  model: string;
  strategy: string;
  provider_request_id: string | null;
  finish_reason: string | null;
  status: string;
  usage: Record<string, unknown>;
  latency_ms: number;
  retry_mode: string;
  error_code: string | null;
  detail_code: string | null;
  diagnostics: Record<string, unknown>;
  created_at: string;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const WEB_SOURCE_REVISION = import.meta.env.VITE_SOURCE_REVISION ?? "development";
const STOPPED_STATUSES = new Set([
  "completed",
  "completed_with_limitations",
  "failed",
  "cancelled",
  "interrupted",
  "credentials_required",
]);

function parseEventStream(raw: string): AgentEvent[] {
  const events: AgentEvent[] = [];
  for (const block of raw
    .replaceAll("\r\n", "\n")
    .split("\n\n")) {
    const data = block.split("\n").find((line) => line.startsWith("data: "))?.slice(6);
    if (!data) continue;
    try {
      events.push(JSON.parse(data) as AgentEvent);
    } catch {
      // A malformed historical event must not freeze all subsequent polling.
      // The persisted run/evidence snapshots remain authoritative.
    }
  }
  return events;
}

function latestFailureDetail(events: AgentEvent[]): string | null {
  const failure = [...events].reverse().find((event) => event.event_type === "run.failed");
  return typeof failure?.refs.detail_code === "string" ? failure.refs.detail_code : null;
}

function latestModelDiagnostics(events: AgentEvent[]): string | null {
  const failure = [...events].reverse().find((event) => event.event_type === "run.failed");
  const value = failure?.refs.model_diagnostics;
  if (typeof value !== "object" || value === null) return null;
  const diagnostics = value as Record<string, unknown>;
  const parts: string[] = [];
  const attemptStage = typeof diagnostics.attempt_stage === "string" ? diagnostics.attempt_stage : null;
  const compactTrigger = typeof diagnostics.compact_trigger === "string" ? diagnostics.compact_trigger : null;
  const strategy = typeof diagnostics.structured_output_strategy === "string" ? diagnostics.structured_output_strategy : null;
  const finishReason = typeof diagnostics.finish_reason === "string" ? diagnostics.finish_reason : null;
  const retryMode = typeof diagnostics.retry_mode === "string" ? diagnostics.retry_mode : null;
  const failureNode = typeof diagnostics.failure_node === "string" ? diagnostics.failure_node : null;
  const outputTokens = typeof diagnostics.output_tokens === "number" ? diagnostics.output_tokens : null;
  const maxOutputTokens = typeof diagnostics.max_output_tokens === "number" ? diagnostics.max_output_tokens : null;
  if (attemptStage === "planner_compact") parts.push("调用阶段 Compact Planner");
  if (failureNode) parts.push(`失败边界 ${failureNode}`);
  if (compactTrigger === "compact_length") parts.push("进入原因：普通计划长度截断");
  if (compactTrigger === "compact_invalid_or_schema") parts.push("进入原因：普通计划格式校验失败");
  if (strategy) parts.push(`策略 ${strategy}`);
  if (finishReason) parts.push(`finish_reason=${finishReason}`);
  if (outputTokens !== null && maxOutputTokens !== null && maxOutputTokens > 0) {
    parts.push(`输出 ${outputTokens}/${maxOutputTokens} Token`);
  }
  if (typeof diagnostics.response_length === "number") {
    parts.push(`正文 ${diagnostics.response_length} 字符`);
  }
  if (typeof diagnostics.reasoning_tokens === "number") {
    parts.push(`推理用量 ${diagnostics.reasoning_tokens} Token`);
  } else if (diagnostics.reasoning_content_present === 1) {
    parts.push("Provider 返回了思考内容（不展示或保存正文）");
  }
  if (retryMode === "compact_transport_retry") parts.push("重试：Compact 网络重试");
  else if (retryMode) parts.push(`重试 ${retryMode}`);
  return parts.length > 0 ? parts.join("；") : null;
}

function runMessage(run: ResearchRun, events: AgentEvent[] = []): string {
  if (run.status === "queued" && run.termination_reason === "model_transport_retry_pending") {
    const delay = metric(run.usage_snapshot, "model_retry_after_seconds");
    const attempt = metric(run.usage_snapshot, "model_transport_requeues");
    return `模型网络暂时不可达；研究状态和检查点已保留，系统将在约 ${delay} 秒后自动恢复（持久化重试 ${attempt}/4）。`;
  }
  if (run.status === "queued" && run.termination_reason === "search_transport_retry_pending") {
    const delay = metric(run.usage_snapshot, "search_retry_after_seconds");
    const attempt = metric(run.usage_snapshot, "search_transport_requeues");
    return `搜索服务暂时不可达；研究状态和检查点已保留，系统将在约 ${delay} 秒后自动恢复（持久化重试 ${attempt}/4）。`;
  }
  if (run.status === "queued") {
    const createdAt = Date.parse(run.created_at);
    const waitedSeconds = Number.isFinite(createdAt)
      ? Math.max(0, Math.floor((Date.now() - createdAt) / 1000))
      : 0;
    const waited = waitedSeconds >= 60
      ? `已等待 ${Math.floor(waitedSeconds / 60)} 分 ${waitedSeconds % 60} 秒`
      : `已等待 ${waitedSeconds} 秒`;
    return `任务 ${run.run_id.slice(0, 8)}… 已入队，等待 Dispatcher/Worker（${waited}）。`;
  }
  if (run.status === "running") return `任务正在执行 ${run.phase} 阶段，状态版本 ${run.state_version}。`;
  if (run.status === "interrupted" && run.termination_reason === "planner_not_implemented") {
    return "Worker 执行闭环已验证；当前任务来自旧版本，可重新点击开始研究生成真实计划。";
  }
  if (run.status === "interrupted" && run.termination_reason === "tool_layer_not_implemented") {
    return "真实研究计划已生成并持久化；当前开发版本按设计停在 Web Tool 接入点。";
  }
  if (run.status === "interrupted" && run.termination_reason === "evaluator_not_implemented") {
    return "首个 Research Loop 已完成：搜索、网页读取、证据抽取与质量快照均已持久化；当前任务来自旧版本，可继续恢复。";
  }
  if (run.status === "interrupted" && isBudgetStopReason(run.termination_reason)) {
    return budgetStopLabel(run.termination_reason) + "；现有计划和证据已保留，可直接生成带限制报告。";
  }
  if (run.status === "interrupted" && run.termination_reason === "writer_not_implemented") {
    return "该任务来自旧版本且证据研究已完成，可直接生成研究报告。";
  }
  if (run.status === "completed") return "研究任务已完成，报告和引用均已通过校验。";
  if (run.status === "completed_with_limitations") {
    const quality = run.quality_snapshot;
    const unmet = [
      metric(quality, "coverage") < 0.85 ? "总体覆盖度" : null,
      metric(quality, "priority_one_coverage") < 0.80 ? "P1 覆盖度" : null,
      metric(quality, "source_quality") < 0.75 ? "来源质量" : null,
      metric(quality, "cross_validation") < 0.70 ? "交叉验证" : null,
      metric(quality, "critical_gaps") > 0 ? "关键缺口" : null,
    ].filter((item): item is string => item !== null);
    const reason = isBudgetStopReason(run.termination_reason)
      ? `停止原因：${budgetStopLabel(run.termination_reason)}。`
      : "";
    return `研究任务已完成并生成报告；未通过质量门：${unmet.join("、") || "历史终止原因"}。${reason}`;
  }
  if (run.status === "failed") {
    const failures: Record<string, string> = {
      MODEL_REQUEST_INVALID: "模型 API 拒绝了请求，请检查 API 协议是否匹配该服务。",
      MODEL_AUTHENTICATION_FAILED: "模型鉴权失败，请更新 API Key。",
      MODEL_NETWORK_ERROR: "模型服务网络连接失败；Planner 会自动重试三次，耗尽后可恢复任务。",
      MODEL_PROVIDER_UNAVAILABLE: "模型服务暂时不可用；Planner 会自动重试三次。",
      MODEL_RATE_LIMITED: "模型 API 已限流，请稍后重试。",
      MODEL_TIMEOUT: "模型调用超时，请稍后重试。",
      RESEARCH_BUDGET_EXHAUSTED: "研究预算已停止扩展；请查看具体页面、搜索、Token 或轮次停止原因。",
      page_budget_exhausted: "页面读取预算已耗尽；现有计划和证据已保留。",
      search_budget_exhausted: "搜索次数预算已耗尽；现有计划和证据已保留。",
      token_budget_exhausted: "模型 Token 预算已耗尽或触发调用前保护；现有计划和证据已保留。",
      iteration_budget_exhausted: "研究轮次预算已耗尽；现有计划和证据已保留。",
      STATE_VALIDATION_FAILED: "研究状态未通过确定性不变量校验；系统已安全停止并保留诊断位置，未继续执行不可信状态。",
      MODEL_OUTPUT_INVALID: "模型未返回可解析的 JSON 研究计划；系统已尝试 JSON Mode 和 Prompt JSON，请确认该模型支持结构化输出。",
      MODEL_OUTPUT_TRUNCATED: "模型输出因达到 Token 上限而截断，JSON 尚未闭合；系统已执行紧凑重答但仍未完成。",
      PLAN_OUTPUT_BUDGET_EXCEEDED: "Compact Planner 达到输出上限，未得到完整可验证的计划。请查看进入原因及模型输出诊断；思考模式也可能消耗输出预算。",
      MODEL_OUTPUT_SCHEMA_INVALID: "模型经过一次结构纠正后，研究计划仍未通过 Schema 校验。",
      MODEL_CAPABILITY_INSUFFICIENT: "兼容模型连续无法生成可校验的结构化证据；系统已提前熔断，避免耗尽全部研究预算。",
      EVIDENCE_OUTPUT_SCHEMA_INVALID: "单页证据抽取未通过 Schema 校验；新版会隔离该来源并保留其他有效证据。",
      REPORT_NO_ACCEPTED_EVIDENCE: "研究期间未获得可验证证据，系统拒绝生成无证据报告；请查看搜索、网页读取和证据抽取的具体失败码。",
      CONTEXT_MANIFEST_PERSISTENCE_FAILED: "研究上下文保存失败；系统已保留任务状态，请稍后恢复任务重试。",
      SEARCH_PROVIDER_DEGRADED: "搜索服务在全部回退策略后仍不可用；系统已保留检查点并执行有界持久化恢复。",
      SEARCH_TIMEOUT: "搜索服务调用超时；系统已保留检查点并执行有界持久化恢复。",
      SEARCH_NETWORK_ERROR: "搜索服务网络连接失败；系统已保留检查点并执行有界持久化恢复。",
      SEARCH_PROVIDER_UNAVAILABLE: "搜索服务暂时不可用；系统已保留检查点并执行有界持久化恢复。",
      CREDENTIAL_UNAVAILABLE: "任务绑定的 API 凭据不可用或无法通过完整性校验，请重新保存连接配置后新建研究任务。",
    };
    const summary = failures[run.termination_reason ?? ""] ?? `研究任务失败：${run.termination_reason ?? "未知原因"}`;
    const detail = latestFailureDetail(events);
    const diagnostics = latestModelDiagnostics(events);
    const suffix = diagnostics ? ` ${diagnostics}。` : "";
    if (!detail) return `${summary}${suffix}`;
    if (detail.includes("FINISH_LENGTH") || detail.includes("FINISH_MAX_TOKENS")) {
      return `${summary} 诊断：Provider 返回长度截断。${suffix}`;
    }
    if (detail.includes("PROMPT_JSON")) {
      return `${summary} 诊断：Provider 已降级到 Prompt JSON，未使用可靠的原生 JSON Mode。${suffix}`;
    }
    if (detail.includes("SEARCH_PROVIDER_EXHAUSTED")) {
      return `${summary} 诊断：搜索 Provider 在本轮均不可用，未获得可读取候选。${suffix}`;
    }
    if (detail.includes("EVIDENCE_EXTRACTION_UNAVAILABLE")) {
      return `${summary} 诊断：网页已读取，但证据抽取模型全部超时或网络失败。${suffix}`;
    }
    if (detail.includes("EVIDENCE_VALIDATION_EMPTY")) {
      return `${summary} 诊断：网页已读取，但候选证据没有通过原文定位、来源范围或安全校验。${suffix}`;
    }
    if (detail.includes("PAGE_BUDGET_OVERRUN")) {
      return `${summary} 诊断：页面预算在外部读取前未能获得足够槽位；已保留当前证据并可从检查点恢复。${suffix}`;
    }
    if (detail.includes("SCHEMA_INVALID")) {
      return `${summary} 诊断：JSON 语法有效，但字段结构不满足 ResearchPlan Schema。${suffix}`;
    }
    return `${summary} 诊断码：${detail}。${suffix}`;
  }
  if (run.status === "cancelled") return "研究任务已取消。";
  return `任务状态 ${run.status}，阶段 ${run.phase}。`;
}

function eventLabel(eventType: string): string {
  const labels: Record<string, string> = {
    "run.created": "Research API",
    "model.retry_deferred": "Durable Retry",
    "search.retry_deferred": "Search Recovery",
    "run.started": "Celery Worker",
    "state.initialized": "State Runtime",
    "state.patch_applied": "State Runtime",
    "run.interrupted": "Execution Guard",
    "run.failed": "Failure Boundary",
    "run.cancelled": "Research API",
    "run.status_changed": "Run Lifecycle",
    "plan.generated": "Research Planner",
    "plan.revised": "Research Replanner",
    "model.retry_scheduled": "Retry Policy",
    "gap.opened": "Gap Detector",
    "action.selected": "Tool Policy",
    "tool.called": "Web Search",
    "tool.failed": "Tool Boundary",
    "search.completed": "Search Provider",
    "source.read": "Web Reader",
    "source.rejected": "Reader Guard",
    "context.assembled": "Context Manager",
    "evidence.extraction_started": "Evidence Extractor",
    "evidence.extracted": "Evidence Extractor",
    "evidence.failed": "Evidence Boundary",
    "question.researched": "Research State",
    "question.retry_scheduled": "Evaluator",
    "question.technical_retry_scheduled": "Technical Retry",
    "question.technical_degraded": "Provider Boundary",
    "evaluation.completed": "Evaluator",
    "research.information_gain_calculated": "Information Gain",
    "research.continued": "Research Loop",
    "evaluation.pending": "Execution Guard",
    "report.writing_started": "Report Writer",
    "report.section_completed": "Section Verifier",
    "report.verified": "Citation Verifier",
    "run.completed": "Run Lifecycle",
  };
  return labels[eventType] ?? eventType;
}

function metric(snapshot: Record<string, unknown> | undefined, key: string): number {
  const value = snapshot?.[key];
  return typeof value === "number" ? value : 0;
}

function modelTokenUsage(snapshot: Record<string, unknown> | undefined): number {
  const direct = metric(snapshot, "model_tokens");
  if (direct > 0) return direct;
  const nestedTotal = (key: string): number => {
    const value = snapshot?.[key];
    return typeof value === "object" && value !== null
      ? metric(value as Record<string, unknown>, "total_tokens")
      : 0;
  };
  return nestedTotal("planner") + nestedTotal("writer") + metric(snapshot, "evidence_total_tokens");
}

function progressStageIndex(stageKey: string): number {
  return Math.max(0, PROGRESS_STAGES.findIndex((stage) => stage.key === stageKey));
}

function progressStageState(
  stage: ProgressStage,
  currentIndex: number,
  completedPlanItems: number,
  planCount: number,
  coverage: number,
  events: AgentEvent[],
  failedStageKey: string | null,
): ProgressStageState {
  const index = progressStageIndex(stage.key);
  if (failedStageKey) {
    const failedIndex = progressStageIndex(failedStageKey);
    if (index < failedIndex) return "done";
    if (index === failedIndex) return "failed";
    return "pending";
  }
  if (index < currentIndex) return "done";
  if (index > currentIndex) return "pending";
  if (stage.key === "init") return events.some((item) => item.event_type === "run.started") ? "done" : "active";
  if (stage.key === "plan") return planCount > 0 && completedPlanItems >= planCount ? "done" : "active";
  if (stage.key === "research") return coverage > 0 ? "active" : "active";
  if (stage.key === "evaluate") return events.some((item) => item.event_type === "evaluation.completed") ? "active" : "active";
  if (stage.key === "replan") return events.some((item) => item.event_type === "plan.revised") ? "active" : "pending";
  if (stage.key === "write") return events.some((item) => item.event_type === "report.section_completed") ? "active" : "pending";
  if (stage.key === "verify") return events.some((item) => item.event_type === "report.verified") ? "done" : "pending";
  return "pending";
}

function failureStageKey(run: ResearchRun, events: AgentEvent[]): string | null {
  if (run.status !== "failed") return null;
  const reason = run.termination_reason ?? "";
  if (["MODEL_OUTPUT_INVALID", "MODEL_OUTPUT_TRUNCATED", "MODEL_OUTPUT_SCHEMA_INVALID", "PLAN_OUTPUT_BUDGET_EXCEEDED"].includes(reason)) return "plan";
  if (reason === "REPORT_NO_ACCEPTED_EVIDENCE" || isBudgetStopReason(reason)) return "evaluate";
  if (["MODEL_CAPABILITY_INSUFFICIENT", "EVIDENCE_OUTPUT_SCHEMA_INVALID"].includes(reason)) return "research";
  if (reason.startsWith("REPORT_") || reason.startsWith("WRITER_")) return "write";
  if (events.some((event) => event.event_type === "report.writing_started")) return "write";
  if (events.some((event) => event.event_type === "evaluation.completed")) return "evaluate";
  if (events.some((event) => event.event_type === "search.completed")) return "research";
  if (events.some((event) => event.event_type === "plan.generated")) return "research";
  return "plan";
}

function phaseProgress(
  run: ResearchRun | null,
  completedPlanItems: number,
  planCount: number,
  coverage: number,
  events: AgentEvent[],
): {overall: number; stageKey: string; stageProgress: number; stageStates: Record<string, ProgressStageState>} {
  if (!run) {
    return {
      overall: 0,
      stageKey: "init",
      stageProgress: 0,
      stageStates: Object.fromEntries(PROGRESS_STAGES.map((stage) => [stage.key, "pending"])),
    };
  }
  const terminalComplete = ["completed", "completed_with_limitations"].includes(run.status);
  // A queued run has no execution progress yet.  Do not render the init-stage
  // placeholder (5% * 50% = 3%) as if the research itself had advanced.
  if (run.status === "queued") {
    return {
      overall: 0,
      stageKey: "init",
      stageProgress: 0,
      stageStates: Object.fromEntries(PROGRESS_STAGES.map((stage) => [stage.key, "pending"])) as Record<string, ProgressStageState>,
    };
  }
  const failedStageKey = failureStageKey(run, events);
  const latestReplan = [...events].reverse().find((item) => item.event_type === "plan.revised");
  const latestResearchProgress = [...events].reverse().find((item) =>
    ["question.researched", "action.selected", "tool.called", "search.completed", "source.read", "evaluation.completed"].includes(item.event_type),
  );
  const activeStageKey = latestReplan && (!latestResearchProgress || latestReplan.seq > latestResearchProgress.seq)
    ? "replan"
    : PHASE_TO_STAGE[run.phase] ?? "init";
  const stageKey = failedStageKey ?? activeStageKey;
  const currentIndex = progressStageIndex(stageKey);
  const searchBudget = metric(run.budget_snapshot, "max_searches");
  const pageBudget = metric(run.budget_snapshot, "max_pages");
  const researchActivity = Math.max(
    coverage,
    searchBudget > 0 ? metric(run.usage_snapshot, "searches") / searchBudget : 0,
    pageBudget > 0 ? metric(run.usage_snapshot, "pages") / pageBudget : 0,
  );
  const normalStageProgress = stageKey === "plan"
    ? planCount > 0 ? completedPlanItems / planCount : 0.35
    : stageKey === "research" || stageKey === "evaluate" || stageKey === "replan"
      ? coverage
      : stageKey === "write"
        ? (events.filter((item) => item.event_type === "report.section_completed").length > 0 ? 0.65 : 0.15)
        : stageKey === "verify"
          ? events.some((item) => item.event_type === "report.verified") ? 1 : 0.2
          : 0.5;
  const stageProgress = failedStageKey
    ? stageKey === "plan"
      ? planCount > 0 ? completedPlanItems / planCount : 0.25
      : stageKey === "research"
        ? Math.min(0.95, researchActivity)
        : stageKey === "evaluate"
          ? coverage
          : stageKey === "write"
            ? events.filter((item) => item.event_type === "report.section_completed").length > 0 ? 0.65 : 0.1
            : 0
    : normalStageProgress;
  const completedWeight = PROGRESS_STAGES
    .slice(0, currentIndex)
    .reduce((sum, stage) => sum + stage.weight, 0);
  const overall = terminalComplete
    ? 1
    : Math.min(0.99, Math.max(0, completedWeight + PROGRESS_STAGES[currentIndex].weight * Math.min(1, stageProgress)));
  const stageStates = Object.fromEntries(
    PROGRESS_STAGES.map((stage) => [stage.key, progressStageState(stage, currentIndex, completedPlanItems, planCount, coverage, events, failedStageKey)]),
  ) as Record<string, ProgressStageState>;
  return {overall, stageKey, stageProgress: Math.min(1, Math.max(0, stageProgress)), stageStates};
}

function coverageDimensions(
  snapshot: Record<string, unknown> | undefined,
): CoverageDimension[] {
  const value = snapshot?.coverage_map;
  if (!Array.isArray(value)) {
    return [];
  }
  return value.flatMap((item) => {
    if (
      typeof item !== "object" ||
      item === null ||
      typeof item.dimension_key !== "string" ||
      typeof item.question !== "string" ||
      typeof item.priority !== "number" ||
      typeof item.coverage !== "number" ||
      typeof item.accepted_evidence !== "number" ||
      typeof item.independent_sources !== "number"
    ) {
      return [];
    }
    return [{
      dimension_key: item.dimension_key,
      question: item.question,
      priority: item.priority,
      coverage: item.coverage,
      accepted_evidence: item.accepted_evidence,
      independent_sources: item.independent_sources,
      missing_reasons: Array.isArray(item.missing_reasons)
        ? item.missing_reasons.filter((reason: unknown): reason is string => typeof reason === "string")
        : [],
      requirement_statuses: Array.isArray(item.requirement_statuses)
        ? item.requirement_statuses.filter(
            (status: unknown): status is CoverageDimension["requirement_statuses"][number] =>
              typeof status === "object" &&
              status !== null &&
              "dimension_key" in status &&
              "criterion" in status &&
              "coverage" in status &&
              "accepted_evidence" in status &&
              "independent_sources" in status &&
              "required_sources" in status &&
              typeof status.dimension_key === "string" &&
              typeof status.criterion === "string" &&
              typeof status.coverage === "number" &&
              typeof status.accepted_evidence === "number" &&
              typeof status.independent_sources === "number" &&
              typeof status.required_sources === "number",
          )
        : [],
    }];
  });
}

function ReportMarkdown({report}: {report: ResearchReport}) {
  const citations = new Map(
    report.citations.map((citation) => [citation.citation_number, citation]),
  );

  function reportHeading(value: string): string {
    if (report.citations.length === 0) return value;
    const compact = value.toLocaleLowerCase().replace(/[^\p{L}\p{N}]+/gu, "");
    const contradictsCitations = [
      "无法生成研究报告",
      "未提供evidencecards",
      "未提供证据卡",
      "noevidencecardsprovided",
      "cannotgeneratereport",
    ].some((marker) => compact.includes(marker));
    return contradictsCitations ? "研究报告（基于已验证证据生成）" : value;
  }

  function renderText(value: string) {
    return value.split(/(\[\d+\])/g).map((part, index) => {
      const match = /^\[(\d+)\]$/.exec(part);
      const citation = match ? citations.get(Number(match[1])) : undefined;
      return citation ? (
        <a
          className="report-citation"
          href={citation.source_url}
          key={`${part}-${index}`}
          target="_blank"
          rel="noreferrer"
          title={citation.claim}
        >
          {part}
        </a>
      ) : <span key={`${part}-${index}`}>{part}</span>;
    });
  }

  return (
    <div className="report-markdown">
      {report.final_markdown.split("\n").map((line, index) => {
        if (!line.trim()) return <span className="report-space" key={index} />;
        if (line.startsWith("# ")) return <h2 key={index}>{renderText(reportHeading(line.slice(2)))}</h2>;
        if (line.startsWith("## ")) return <h3 key={index}>{renderText(line.slice(3))}</h3>;
        if (line.startsWith("- ")) return <p className="report-list-item" key={index}>{renderText(line.slice(2))}</p>;
        if (/^\d+\. /.test(line)) return null;
        return <p key={index}>{renderText(line)}</p>;
      })}
    </div>
  );
}

function App() {
  const [health, setHealth] = useState<ApiHealth>("checking");
  const [apiSourceRevision, setApiSourceRevision] = useState("unknown");
  const [providerConfigured, setProviderConfigured] = useState(false);
  const [credentialVersionId, setCredentialVersionId] = useState("");
  const [creatingRun, setCreatingRun] = useState(false);
  const [endingRun, setEndingRun] = useState(false);
  const [budgetTier, setBudgetTier] = useState<BudgetTier>("standard");
  const [pollRevision, setPollRevision] = useState(0);
  const [activeRunId, setActiveRunId] = useState("");
  const [activeRun, setActiveRun] = useState<ResearchRun | null>(null);
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [evidence, setEvidence] = useState<EvidenceItem[]>([]);
  const [report, setReport] = useState<ResearchReport | null>(null);
  const [contextMetrics, setContextMetrics] = useState<ContextManifest[]>([]);
  const [memoryAccesses, setMemoryAccesses] = useState<Record<string, unknown>[]>([]);
  const [knowledgeLedger, setKnowledgeLedger] = useState<KnowledgeLedger | null>(null);
  const [gapLedger, setGapLedger] = useState<GapLedger | null>(null);
  const [actionLedger, setActionLedger] = useState<ActionLedger | null>(null);
  const [evaluations, setEvaluations] = useState<EvaluationSnapshot[]>([]);
  const [llmCalls, setLlmCalls] = useState<LLMCall[]>([]);
  const eventCursor = useRef(0);
  const [query, setQuery] = useState(
    "研究工业视觉缺陷检测领域的发展情况，分析技术路线、厂商、代表产品、大模型应用与未来三年趋势。",
  );
  const [message, setMessage] = useState("正在从服务端恢复模型配置…");

  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE_URL}/healthz`),
      fetch(`${API_BASE_URL}/api/v1/meta`),
    ])
      .then(async ([healthResponse, metaResponse]) => {
        if (!healthResponse.ok || !metaResponse.ok) throw new Error("unhealthy");
        const meta = await metaResponse.json() as {source_revision?: string};
        const apiRevision = meta.source_revision ?? "unknown";
        setApiSourceRevision(apiRevision);
        setHealth(apiRevision === WEB_SOURCE_REVISION ? "ready" : "version_mismatch");
      })
      .catch(() => setHealth("unavailable"));
  }, []);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/v1/research-runs?limit=1`, {credentials: "include"})
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const runs = (await response.json()) as ResearchRun[];
        if (runs.length > 0) {
          eventCursor.current = 0;
          setEvents([]);
          setEvidence([]);
          setReport(null);
          setContextMetrics([]);
          setMemoryAccesses([]);
          setKnowledgeLedger(null);
          setGapLedger(null);
          setActionLedger(null);
          setEvaluations([]);
          setLlmCalls([]);
          setActiveRun(runs[0]);
          setActiveRunId(runs[0].run_id);
        }
      })
      .catch((error) => {
        setMessage(`恢复最近任务失败：${error instanceof Error ? error.message : "未知错误"}`);
      });
  }, []);

  useEffect(() => {
    if (!activeRunId) return;
    let cancelled = false;
    let timer: number | undefined;

    async function poll() {
      try {
        const eventHeaders: Record<string, string> = {};
        if (eventCursor.current > 0) {
          eventHeaders["Last-Event-ID"] = String(eventCursor.current);
        }
        const [statusResponse, eventResponse, evidenceResponse, contextResponse, memoryResponse, knowledgeResponse, gapsResponse, actionsResponse, evaluationsResponse, llmCallsResponse] = await Promise.all([
          fetch(API_BASE_URL + "/api/v1/research-runs/" + activeRunId, {
            credentials: "include",
          }),
          fetch(API_BASE_URL + "/api/v1/research-runs/" + activeRunId + "/events?follow=false", {
            credentials: "include",
            headers: eventHeaders,
          }),
          fetch(API_BASE_URL + "/api/v1/research-runs/" + activeRunId + "/evidence", {
            credentials: "include",
          }),
          fetch(API_BASE_URL + "/api/v1/research-runs/" + activeRunId + "/context-metrics", {
            credentials: "include",
          }),
          fetch(API_BASE_URL + "/api/v1/research-runs/" + activeRunId + "/memory-accesses", {
            credentials: "include",
          }),
          fetch(API_BASE_URL + "/api/v1/research-runs/" + activeRunId + "/knowledge", {
            credentials: "include",
          }),
          fetch(API_BASE_URL + "/api/v1/research-runs/" + activeRunId + "/gaps", {
            credentials: "include",
          }),
          fetch(API_BASE_URL + "/api/v1/research-runs/" + activeRunId + "/actions", {
            credentials: "include",
          }),
          fetch(API_BASE_URL + "/api/v1/research-runs/" + activeRunId + "/evaluations", {
            credentials: "include",
          }),
          fetch(API_BASE_URL + "/api/v1/research-runs/" + activeRunId + "/llm-calls", {
            credentials: "include",
          }),
        ]);
        if (!statusResponse.ok) throw new Error(`状态接口 HTTP ${statusResponse.status}`);
        if (!eventResponse.ok) throw new Error(`事件接口 HTTP ${eventResponse.status}`);
        if (!evidenceResponse.ok) throw new Error(`证据接口 HTTP ${evidenceResponse.status}`);
        if (!contextResponse.ok) throw new Error(`上下文指标接口 HTTP ${contextResponse.status}`);
        if (!memoryResponse.ok) throw new Error("Memory 接口 HTTP " + memoryResponse.status);

        const run = (await statusResponse.json()) as ResearchRun;
        const incoming = parseEventStream(await eventResponse.text());
        const currentEvidence = (await evidenceResponse.json()) as EvidenceItem[];
        const currentContextMetrics = (await contextResponse.json()) as ContextManifest[];
        const currentMemoryAccesses = (await memoryResponse.json()) as Record<string, unknown>[];
        const currentKnowledge = knowledgeResponse.ok
          ? (await knowledgeResponse.json()) as KnowledgeLedger
          : null;
        const currentGaps = gapsResponse.ok ? (await gapsResponse.json()) as GapLedger : null;
        const currentActions = actionsResponse.ok
          ? (await actionsResponse.json()) as ActionLedger
          : null;
        const currentEvaluations = evaluationsResponse.ok
          ? (await evaluationsResponse.json()) as EvaluationSnapshot[]
          : [];
        const currentLlmCalls = llmCallsResponse.ok
          ? (await llmCallsResponse.json()) as LLMCall[]
          : [];
        if (!cancelled) {
          if (incoming.length > 0) {
            eventCursor.current = Math.max(
              eventCursor.current,
              ...incoming.map((item) => item.seq),
            );
            setEvents((current) => {
              const merged = new Map(current.map((item) => [item.seq, item]));
              incoming.forEach((item) => merged.set(item.seq, item));
              return [...merged.values()].sort((left, right) => left.seq - right.seq);
            });
          }
          setActiveRun(run);
          setEvidence(currentEvidence);
          setContextMetrics(currentContextMetrics);
          setMemoryAccesses(currentMemoryAccesses);
          setKnowledgeLedger(currentKnowledge);
          setGapLedger(currentGaps);
          setActionLedger(currentActions);
          setEvaluations(currentEvaluations);
          setLlmCalls(currentLlmCalls);
          setMessage(runMessage(run));
          if (!STOPPED_STATUSES.has(run.status)) {
            timer = window.setTimeout(() => void poll(), 1000);
          }
        }
      } catch (error) {
        if (!cancelled) {
          setMessage(`读取执行进度失败：${error instanceof Error ? error.message : "未知错误"}`);
          timer = window.setTimeout(() => void poll(), 2000);
        }
      }
    }

    void poll();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [activeRunId, pollRevision]);

  useEffect(() => {
    if (!activeRunId || !activeRun || !["completed", "completed_with_limitations"].includes(activeRun.status)) {
      setReport(null);
      return;
    }
    let cancelled = false;
    fetch(`${API_BASE_URL}/api/v1/research-runs/${activeRunId}/report`, {
      credentials: "include",
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(`报告接口 HTTP ${response.status}`);
        return (await response.json()) as ResearchReport;
      })
      .then((payload) => {
        if (!cancelled) setReport(payload);
      })
      .catch((error) => {
        if (!cancelled) {
          setMessage(`读取研究报告失败：${error instanceof Error ? error.message : "未知错误"}`);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [activeRun, activeRunId]);

  const planItems = useMemo<PlanItem[]>(() => {
    const currentCoverage = coverageDimensions(activeRun?.quality_snapshot);
    const generated = events.find((item) => item.event_type === "plan.generated");
    const questions = generated?.refs.questions;
    const questionIds = generated?.refs.question_ids;
    if (Array.isArray(questions) && questions.length > 0) {
      return questions.map((question, index) => {
        const questionId = Array.isArray(questionIds) ? String(questionIds[index] ?? "") : "";
        const finished = [...events]
          .reverse()
          .find(
            (item) =>
              item.event_type === "question.researched" &&
              item.refs.question_id === questionId,
          );
        const active = !["completed", "completed_with_limitations", "cancelled"].includes(activeRun?.status ?? "") && events.some(
          (item) =>
            item.refs.question_id === questionId &&
            [
              "gap.opened",
              "action.selected",
              "tool.called",
              "search.completed",
              "source.read",
              "context.assembled",
              "evidence.extraction_started",
              "evidence.extracted",
              "question.retry_scheduled",
            ].includes(item.event_type),
        );
        const finalStatus = finished?.refs.status;
        const technicalStatus = [...events]
          .reverse()
          .find(
            (item) =>
              ["question.technical_retry_scheduled", "question.technical_degraded"].includes(
                item.event_type,
              ) && item.refs.question_id === questionId,
          );
        const dimension = currentCoverage.find(
          (item) => item.dimension_key === questionId,
        );
        const failedActive = activeRun?.status === "failed" && active;
        const status: PlanStatus =
          dimension && dimension.coverage >= 1
            ? "done"
            : dimension && dimension.coverage > 0
              ? "partial"
              : technicalStatus?.event_type === "question.technical_degraded"
                ? "degraded"
                : finalStatus === "blocked" || failedActive
              ? "blocked"
              : active
                ? "active"
                : "pending";
        return {
          number: String(index + 1).padStart(2, "0"),
          questionId,
          title: String(question),
          status,
          coverage: dimension?.coverage ?? 0,
          acceptedEvidence: dimension?.accepted_evidence ?? 0,
          independentSources: dimension?.independent_sources ?? 0,
          reason:
            status === "partial"
              ? dimension?.missing_reasons.join("；")
              : status === "degraded"
                ? "搜索服务连续故障，未计入研究失败"
                : status === "blocked"
                  ? dimension?.missing_reasons.join("；") || "有效检索后仍缺少证据"
                  : undefined,
        };
      });
    }

    const created = events.some((item) => item.event_type === "run.created");
    const started = events.some((item) => item.event_type === "run.started");
    const interrupted = events.some((item) => item.event_type === "run.interrupted");
    const failed = events.some((item) => item.event_type === "run.failed");
    return [
      {number: "01", title: "任务与不可变配置落库", status: created ? "done" : "active", coverage: created ? 1 : 0},
      {
        number: "02",
        title: "Outbox 发布与 Worker Lease",
        status: started ? "done" : created ? "active" : "pending",
        coverage: started ? 1 : 0,
      },
      {
        number: "03",
        title: "Planner 生成研究计划",
        status: interrupted || failed ? "blocked" : started ? "active" : "pending",
        coverage: interrupted || failed ? 0 : started ? 0.5 : 0,
      },
      {number: "04", title: "Research Loop 与证据评估", status: "pending", coverage: 0},
      {number: "05", title: "报告、引用与事实核验", status: "pending", coverage: 0},
    ];
  }, [activeRun, events]);

  async function startResearch(event: FormEvent) {
    event.preventDefault();
    if (!providerConfigured || !credentialVersionId) {
      setMessage("请先保存模型与 API Key 配置。");
      return;
    }
    if (!query.trim()) {
      setMessage("请输入明确的研究目标。");
      return;
    }

    setCreatingRun(true);
    setMessage("正在创建可恢复研究任务并写入 Dispatch Outbox…");
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/research-runs`, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
          "Idempotency-Key": crypto.randomUUID(),
        },
        body: JSON.stringify({
          query,
          saved_profile_version_id: credentialVersionId,
          budget_tier: budgetTier,
        }),
      });
      const payload = (await response.json()) as ResearchRun & {
        detail?: {message?: string; error_code?: string};
      };
      if (!response.ok) {
        throw new Error(payload.detail?.message ?? payload.detail?.error_code ?? `HTTP ${response.status}`);
      }
      eventCursor.current = 0;
      setEvents([]);
      setEvidence([]);
      setReport(null);
      setActiveRun(payload);
      setActiveRunId(payload.run_id);
      setMessage(runMessage(payload));
    } catch (error) {
      setMessage(`创建研究任务失败：${error instanceof Error ? error.message : "未知错误"}`);
    } finally {
      setCreatingRun(false);
    }
  }

  async function resumeResearch() {
    if (!activeRunId) return;
    setCreatingRun(true);
    setMessage("正在恢复现有任务；已持久化计划和证据不会重新生成…");
    try {
      const response = await fetch(
        API_BASE_URL + "/api/v1/research-runs/" + activeRunId + "/resume",
        {
          method: "POST",
          credentials: "include",
        },
      );
      const payload = (await response.json()) as ResearchRun & {
        detail?: {message?: string; error_code?: string};
      };
      if (!response.ok) {
        throw new Error(
          payload.detail?.message ??
            payload.detail?.error_code ??
            "HTTP " + response.status,
        );
      }
      setActiveRun(payload);
      setMessage(runMessage(payload));
      setPollRevision((current) => current + 1);
    } catch (error) {
      setMessage(
        "恢复研究任务失败：" + (error instanceof Error ? error.message : "未知错误"),
      );
    } finally {
      setCreatingRun(false);
    }
  }
  async function endResearch() {
    if (!activeRunId || endingRun) return;
    if (!window.confirm("确定结束当前研究吗？已保存的计划、来源和证据会保留，但任务不会继续执行。")) {
      return;
    }
    setEndingRun(true);
    setMessage("正在结束研究并保存当前快照…");
    try {
      const response = await fetch(
        API_BASE_URL + "/api/v1/research-runs/" + activeRunId + "/cancel",
        {
          method: "POST",
          credentials: "include",
        },
      );
      const payload = (await response.json()) as ResearchRun & {
        detail?: {message?: string; error_code?: string};
      };
      if (!response.ok) {
        throw new Error(
          payload.detail?.message ??
            payload.detail?.error_code ??
            "HTTP " + response.status,
        );
      }
      setActiveRun(payload);
      setMessage(runMessage(payload));
      setPollRevision((current) => current + 1);
    } catch (error) {
      setMessage(
        "结束研究失败：" + (error instanceof Error ? error.message : "未知错误"),
      );
    } finally {
      setEndingRun(false);
    }
  }

  const completedPlanItems = planItems.filter((item) => item.status === "done").length;
  const quality = activeRun?.quality_snapshot;
  const acceptedEvidence = metric(quality, "accepted_evidence");
  const coverage = metric(quality, "coverage");
  const sources = metric(quality, "source_count");
  const conflicts = metric(quality, "conflict_count");
  const citations = metric(quality, "citation_count");
  const informationGain = metric(quality, "information_gain");
  const lowGainStreak = metric(quality, "low_information_gain_streak");
  const qualityGates = [
    {key: "coverage", label: "总体覆盖度", value: coverage, target: 0.85},
    {key: "priority_one_coverage", label: "P1 覆盖度", value: metric(quality, "priority_one_coverage"), target: 0.80},
    {key: "source_quality", label: "来源质量", value: metric(quality, "source_quality"), target: 0.75},
    {key: "cross_validation", label: "交叉验证", value: metric(quality, "cross_validation"), target: 0.70},
  ];
  const criticalGaps = metric(quality, "critical_gaps");
  const passedQualityGates = qualityGates.filter((gate) => gate.value >= gate.target).length
    + (criticalGaps === 0 ? 1 : 0);
  const coverageMap = coverageDimensions(quality);
  const usage = activeRun?.usage_snapshot;
  const budget = activeRun?.budget_snapshot;
  const evidenceFunnel = [
    {
      label: "候选 URL",
      value: typeof usage?.candidate_urls === "number"
        ? usage.candidate_urls
        : events.reduce(
          (sum, event) => sum + metric(event.metrics ?? undefined, "result_count"),
          0,
        ),
    },
    {
      label: "读取页面",
      value: typeof usage?.pages === "number"
        ? usage.pages
        : events.filter((event) => event.event_type === "source.read").length,
    },
    {
      label: "抽取候选",
      value: typeof quality?.candidate_evidence === "number"
        ? quality.candidate_evidence
        : events.reduce(
          (sum, event) => sum + metric(event.metrics ?? undefined, "candidate_count"),
          0,
        ),
    },
    {
      label: "接受证据",
      value: typeof quality?.accepted_evidence === "number"
        ? quality.accepted_evidence
        : events.reduce(
          (sum, event) => sum + metric(event.metrics ?? undefined, "accepted_count"),
          0,
        ),
    },
  ];
  const totalContextBefore = contextMetrics.reduce((sum, item) => sum + item.token_before, 0);
  const totalContextAfter = contextMetrics.reduce((sum, item) => sum + item.token_after, 0);
  const contextSavings = totalContextBefore > 0
    ? Math.max(0, 1 - totalContextAfter / totalContextBefore)
    : 0;
  const compressedContextItems = contextMetrics.reduce((sum, item) => sum + item.compressed_count, 0);
  const prunedContextItems = contextMetrics.reduce((sum, item) => sum + item.rejected_count, 0);
  const recentContextMetrics = contextMetrics.slice(-5).reverse();
  const latestResearchDecision = [...events]
    .reverse()
    .find((event) =>
      ["research.continued", "evaluation.completed", "report.writing_started"].includes(
        event.event_type,
      ),
    );

  const memoryHits = memoryAccesses.filter((item) => item.result === "hit").length;
  const memoryMisses = memoryAccesses.filter((item) => item.result === "miss").length;
  const unknownDimensions = coverageMap.filter((item) => item.coverage < 1);
  const openGapCount = gapLedger?.open_count ?? unknownDimensions.length;
  const nextAction = [...events].reverse().find((event) => event.event_type === "action.selected");
  const serverNextAction = actionLedger?.next_action;
  const latestEvaluation = evaluations.length > 0 ? evaluations[evaluations.length - 1] : null;
  const nextActionText = typeof serverNextAction?.type === "string"
    ? serverNextAction.type
    : nextAction
      ? eventLabel(nextAction.event_type)
      : "等待评估";
  const progress = phaseProgress(
    activeRun,
    completedPlanItems,
    planItems.length,
    coverage,
    events,
  );
  const budgetItems = [
    {key: "scheduler_actions", label: "调度动作", used: metric(usage, "scheduler_actions"), max: metric(budget, "max_scheduler_actions")},
    {key: "logical_queries", label: "逻辑查询", used: metric(usage, "logical_queries") || metric(usage, "searches"), max: metric(budget, "max_logical_queries") || metric(budget, "max_searches")},
    {key: "provider_requests", label: "上游请求", used: metric(usage, "search_provider_requests"), max: metric(budget, "max_provider_requests")},
    {key: "pages_fetched", label: "抓取页面", used: metric(usage, "pages_fetched"), max: metric(budget, "max_pages_fetched")},
    {key: "pages_extracted", label: "抽取页面", used: metric(usage, "pages_extracted") || metric(usage, "pages"), max: metric(budget, "max_pages_extracted") || metric(budget, "max_pages")},
    {key: "model_tokens", label: "模型 Token", used: modelTokenUsage(usage), max: metric(budget, "max_tokens")},
  ];
  const budgetUpdatedEvent = [...events].reverse().find(
    (event) => event.event_type === "budget.updated",
  );
  const writingStartedEvent = [...events].reverse().find(
    (event) => event.event_type === "report.writing_started",
  );
  const persistedStopReason = typeof usage?.research_stop_reason === "string"
    ? usage.research_stop_reason
    : activeRun?.termination_reason ?? null;
  const writingReason = typeof writingStartedEvent?.refs.reason === "string"
    ? writingStartedEvent.refs.reason
    : null;
  // The persisted run/usage projection is authoritative for budget diagnostics. A writer
  // event may carry a generic quality-gate reason after research already hit a
  // precise page/search/token/iteration limit; preferring it made a full
  // budget appear as "not stopped" in the dashboard.
  const researchStopReason = isBudgetStopReason(persistedStopReason)
    ? persistedStopReason
    : writingReason ?? persistedStopReason;
  const budgetDiagnostics = {
    maxTokens: metric(budget, "max_tokens"),
    remainingTokens: typeof usage?.model_tokens_remaining === "number"
      ? usage.model_tokens_remaining
      : metric(budget, "max_tokens") > 0
        ? Math.max(0, metric(budget, "max_tokens") - modelTokenUsage(usage))
        : null,
    writerReserve: typeof usage?.writer_token_reserve === "number"
      ? usage.writer_token_reserve
      : budgetUpdatedEvent
        ? metric(budgetUpdatedEvent.metrics ?? undefined, "writer_token_reserve")
        : null,
    pendingQuestions: typeof usage?.pending_questions === "number"
      ? usage.pending_questions
      : budgetUpdatedEvent
        ? metric(budgetUpdatedEvent.metrics ?? undefined, "pending_questions")
        : null,
    yielded: typeof usage?.question_yields === "number"
      ? usage.question_yields
      : events.filter((event) => event.event_type === "budget.question_yielded").length,
    settled: typeof usage?.settled_reservations === "number"
      ? usage.settled_reservations
      : events.filter((event) => event.event_type === "budget.settled").length,
    budgetStopped: isBudgetStopReason(researchStopReason),
    stopReason: researchStopReason,
  };
  const recentGainEvents = events
    .filter((event) => event.event_type === "research.information_gain_calculated")
    .slice(-8);
  const recentGainAverage = recentGainEvents.length > 0
    ? recentGainEvents.reduce(
      (sum, event) => sum + metric(event.metrics ?? undefined, "information_gain"),
      0,
    ) / recentGainEvents.length
    : 0;
  const sourceReadEvents = events.filter((event) => event.event_type === "source.read");
  const acceptedBySource = new Map<string, number>();
  for (const event of events) {
    if (event.event_type !== "evidence.extracted") continue;
    const sourceId = typeof event.refs.source_id === "string" ? event.refs.source_id : null;
    if (!sourceId) continue;
    acceptedBySource.set(
      sourceId,
      Math.max(acceptedBySource.get(sourceId) ?? 0, metric(event.metrics ?? undefined, "accepted_count")),
    );
  }
  const hasPersistedPageDiagnostics = typeof usage?.zero_yield_pages === "number";
  const zeroYieldPages = hasPersistedPageDiagnostics
    ? Number(usage.zero_yield_pages)
    : sourceReadEvents.filter((event) => {
      const sourceId = typeof event.refs.source_id === "string" ? event.refs.source_id : null;
      return !sourceId || (acceptedBySource.get(sourceId) ?? 0) === 0;
    }).length;
  const measuredPages = typeof usage?.pages === "number" ? usage.pages : sourceReadEvents.length;
  const zeroYieldPageRatio = measuredPages > 0
    ? zeroYieldPages / measuredPages
    : 0;
  const pagesPerAcceptedEvidence = acceptedEvidence > 0
    ? measuredPages / acceptedEvidence
    : 0;
  const currentActionEvent = [...events].reverse().find((event) =>
    ["action.selected", "tool.called", "source.read", "evidence.extraction_started", "evaluation.completed", "research.continued"].includes(event.event_type),
  );
  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">DR</span>
          <div>
            <strong>DeepResearch</strong>
            <span>Evidence-driven Research Agent</span>
          </div>
        </div>
        <div className={`health health--${health}`}>
          <span />
          API {health === "ready"
            ? `已连接 · REV ${apiSourceRevision}`
            : health === "checking"
              ? "检查中"
              : health === "version_mismatch"
                ? `版本不一致 · WEB ${WEB_SOURCE_REVISION} / API ${apiSourceRevision}`
                : "未连接"}
        </div>
      </header>

      <section className="hero">
        <p className="eyebrow">RESEARCH WORKSPACE / V1</p>
        <h1>从研究目标，到可追溯的证据结论。</h1>
        <p>计划、检索、阅读、提取、补缺、交叉验证与写作都进入同一个可恢复状态循环。</p>
      </section>

      <form className="research-form" onSubmit={startResearch}>
        <div className="form-title">
          <div>
            <span>NEW RESEARCH</span>
            <h2>定义本次研究</h2>
          </div>
          <span className="privacy">API Key 服务端加密保存，不进入 Agent State / 日志 / 队列</span>
        </div>

        <label className="query-field">
          <span>研究问题</span>
          <textarea value={query} onChange={(event) => setQuery(event.target.value)} />
        </label>

        <ProviderProfileForm
          onStatusChange={(configured, statusMessage, savedCredentialVersionId) => {
            setProviderConfigured(configured);
            setCredentialVersionId(savedCredentialVersionId ?? "");
            if (!activeRunId) setMessage(statusMessage);
          }}
        />
        <div className="budget-row">
          <label className="budget-field">
            <span>研究预算</span>
            <select
              value={budgetTier}
              disabled={creatingRun}
              onChange={(event) => setBudgetTier(event.target.value as BudgetTier)}
            >
              <option value="quick">Quick · 5 轮 / 5 次搜索 / 10 页</option>
              <option value="standard">Standard · 20 轮 / 20 次搜索 / 30 页</option>
              <option value="deep">Deep · 30 轮 / 30 次搜索 / 60 页</option>
            </select>
          </label>
          <p>预算在任务创建时固化；Evaluator 会在质量达标或预算耗尽时自动停止。</p>
        </div>
        <div className="research-actions">
          {activeRun && ["queued", "running"].includes(activeRun.status) ? (
            <button
              className="danger-button"
              type="button"
              disabled={creatingRun || endingRun}
              onClick={() => void endResearch()}
            >
              {endingRun ? "结束中…" : "结束研究"}
            </button>
          ) : null}
          {activeRun &&
          ["failed", "interrupted", "cancelled"].includes(activeRun.status) ? (
            <button
              className="secondary-button"
              type="button"
              disabled={creatingRun}
              onClick={() => void resumeResearch()}
            >
              {isBudgetStopReason(activeRun.termination_reason) ||
              activeRun.termination_reason === "writer_not_implemented"
                ? "生成研究报告" : "继续当前任务"}
            </button>
          ) : null}
          <button className="start-button" type="submit" disabled={creatingRun || endingRun}>
            {creatingRun ? "处理中…" : endingRun ? "结束中…" : "开始研究"} <span>→</span>
          </button>
        </div>
        <p className="form-message">{message}</p>
      </form>

      {activeRun && (
        <section className="research-progress" aria-label="Research progress">
          <div className="research-progress__header">
            <div>
              <span className="eyebrow">RESEARCH PROGRESS</span>
              <h2>{Math.round(progress.overall * 100)}% <small>{activeRun.status === "running" ? "执行中" : activeRun.status === "failed" ? "失败" : activeRun.status}</small></h2>
              <p>{STOPPED_STATUSES.has(activeRun.status) ? runMessage(activeRun, events) : currentActionEvent?.public_summary ?? runMessage(activeRun, events)}</p>
            </div>
            <div className="progress-current-action">
              <span>{activeRun.status === "failed" ? "失败阶段" : "当前阶段"}</span>
              <strong>{PROGRESS_STAGES.find((stage) => stage.key === progress.stageKey)?.label ?? "初始化"}</strong>
              <small>{activeRun.status === "failed" ? "该阶段未完成" : `${Math.round(progress.stageProgress * 100)}% 阶段进度`}</small>
            </div>
          </div>
          <div className={`overall-progress-bar${activeRun.status === "failed" ? " overall-progress-bar--failed" : ""}`} role="progressbar" aria-valuenow={Math.round(progress.overall * 100)} aria-valuemin={0} aria-valuemax={100}>
            <i style={{width: `${Math.round(progress.overall * 100)}%`}} />
          </div>
          <div className="progress-stage-list">
            {PROGRESS_STAGES.map((stage) => (
              <div className={`progress-stage progress-stage--${progress.stageStates[stage.key]}`} key={stage.key}>
                <i>{progress.stageStates[stage.key] === "done" ? "✓" : progress.stageStates[stage.key] === "active" ? "●" : progress.stageStates[stage.key] === "failed" ? "×" : "○"}</i>
                <span>{stage.label}</span>
              </div>
            ))}
          </div>
          <div className="progress-details">
            <div className="progress-quality">
              <div className="progress-section-heading"><span>QUALITY GATE</span><b>{passedQualityGates} / 5 passed</b></div>
              <div className="progress-quality-grid">
                {qualityGates.map((gate) => (
                  <div className={gate.value >= gate.target ? "quality-gate--passed" : "quality-gate--failed"} key={gate.key}>
                    <strong>{Math.round(gate.value * 100)}%</strong>
                    <span>{gate.label} · ≥{Math.round(gate.target * 100)}%</span>
                  </div>
                ))}
                <div className={criticalGaps === 0 ? "quality-gate--passed" : "quality-gate--failed"}>
                  <strong>{criticalGaps}</strong>
                  <span>关键缺口 · 必须为 0</span>
                </div>
              </div>
            </div>
            <div className="progress-budget">
              <div className="progress-section-heading"><span>RESOURCE BUDGET</span><b>实际消耗 / 上限</b></div>
              {budgetItems.map((item) => {
                const ratio = item.max > 0 ? Math.min(1, item.used / item.max) : 0;
                return (
                  <div className="budget-meter" key={item.key}>
                    <div><span>{item.label}</span><b>{item.used.toLocaleString()} / {item.max.toLocaleString()}</b></div>
                    <div className="budget-meter__bar"><i className={ratio >= .85 ? "budget-meter--warning" : ""} style={{width: `${Math.round(ratio * 100)}%`}} /></div>
                  </div>
                );
              })}
              <div className="budget-diagnostics">
                <div className="progress-section-heading"><span>BUDGET DIAGNOSTICS</span><b>预留 / 让出 / 待研究</b></div>
                <div className="budget-diag-grid">
                  <div><strong>{budgetDiagnostics.remainingTokens !== null ? budgetDiagnostics.remainingTokens.toLocaleString() : "—"}</strong><span>剩余研究额度 (Token)</span></div>
                  <div><strong>{budgetDiagnostics.writerReserve !== null ? budgetDiagnostics.writerReserve.toLocaleString() : "—"}</strong><span>报告预留</span></div>
                  <div><strong>{budgetDiagnostics.pendingQuestions !== null ? budgetDiagnostics.pendingQuestions : "—"}</strong><span>尚未研究问题数</span></div>
                  <div><strong>{budgetDiagnostics.yielded}</strong><span>本题让出次数</span></div>
                  <div><strong>{budgetDiagnostics.settled}</strong><span>已结算预留</span></div>
                  <div><strong>{budgetDiagnostics.budgetStopped ? "是" : "否"}</strong><span>预算触发停止</span></div>
                </div>
                {budgetDiagnostics.stopReason ? (
                  <div className="budget-diag-stop">
                    <span>停止原因</span><b>{budgetDiagnostics.stopReason}</b>
                  </div>
                ) : null}
              </div>
            </div>
            <div className="progress-gain">
              <div className="progress-section-heading"><span>INFORMATION GAIN</span><b>{Math.round(informationGain * 100)}% latest</b></div>
              <div className="gain-bars" aria-label="Information gain trend">
                {recentGainEvents.length === 0 ? <small>等待第一轮 Evaluation</small> : recentGainEvents.map((event, index) => {
                  const gain = typeof event.metrics?.information_gain === "number" ? event.metrics.information_gain : informationGain;
                  return <i key={event.seq} title={`第 ${index + 1} 轮 ${Math.round(gain * 100)}%`} style={{height: `${Math.max(8, Math.round(gain * 100))}%`}} />;
                })}
              </div>
              <div className="gain-summary" aria-label="Information gain diagnostics">
                <span>最近 8 轮均值 <b>{Math.round(recentGainAverage * 100)}%</b></span>
                <span>零产出页面 <b>{Math.round(zeroYieldPageRatio * 100)}%</b></span>
                <span>有效证据页面成本 <b>{pagesPerAcceptedEvidence.toFixed(1)} 页/条</b></span>
              </div>
              <small>低增益连续轮次：{lowGainStreak} / 2；Evaluator 将据此判断继续、重规划或停止。</small>
            </div>
            <div className="progress-funnel" aria-label="Evidence funnel">
              <div className="progress-section-heading"><span>EVIDENCE FUNNEL</span><b>候选 → 接受</b></div>
              <div className="funnel-list">
                {evidenceFunnel.map((item) => (
                  <div key={item.label} className="funnel-item">
                    <span>{item.label}</span><strong>{item.value.toLocaleString()}</strong>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      )}

      <section className="workspace-grid">
        <article className="panel plan-panel">
          <div className="panel-heading">
            <span>RESEARCH PLAN</span>
            <b>{completedPlanItems} / {planItems.length}</b>
          </div>
          <div className="plan-list">
            {planItems.map((item) => (
              <div className={`plan-item plan-item--${item.status}`} key={item.number}>
                <span>{item.number}</span>
                <div>
                  <p>{item.title}</p>
                  {item.coverage !== undefined && planItems.length > 0 && (
                    <div className="plan-progress" title={`覆盖率 ${Math.round(item.coverage * 100)}%`}>
                      <i style={{width: `${Math.round(item.coverage * 100)}%`}} />
                    </div>
                  )}
                  {item.reason && <small>{item.reason}</small>}
                  {item.questionId && <small className="plan-evidence-count">{item.acceptedEvidence ?? 0} 条证据 · {item.independentSources ?? 0} 个独立来源</small>}
                </div>
                <i>
                  {item.status === "done"
                    ? "✓"
                    : item.status === "active"
                      ? "●"
                      : item.status === "partial"
                        ? "≈"
                        : item.status === "degraded"
                          ? "⚠"
                          : item.status === "blocked"
                            ? "!"
                            : "○"}
                </i>
              </div>
            ))}
          </div>
          {coverageMap.length > 0 && (
            <section className="coverage-map" aria-label="Research Coverage Map">
              <div className="coverage-map__heading">
                <span>RESEARCH COVERAGE MAP</span>
                <b>{Math.round(coverage * 100)}%</b>
              </div>
              {coverageMap.map((dimension) => (
                <article className="coverage-dimension" key={dimension.dimension_key}>
                  <div>
                    <span>P{dimension.priority}</span>
                    <b>{Math.round(dimension.coverage * 100)}%</b>
                  </div>
                  <p>{dimension.question}</p>
                  <div className="coverage-bar">
                    <i style={{width: String(Math.round(dimension.coverage * 100) + "%")}} />
                  </div>
                  <small>
                    {dimension.accepted_evidence} 条证据 / {dimension.independent_sources} 个独立来源
                    {dimension.missing_reasons.length > 0
                      ? [" · 缺口：", dimension.missing_reasons.join("、")].join("")
                      : " · 已满足验收条件"}
                  </small>
                </article>
              ))}
            </section>
          )}
        </article>

        <article className="panel activity-panel">
          <div className="panel-heading">
            <span>AGENT ACTIVITY</span>
            <b>{events.length} 条可重放事件</b>
          </div>
          {contextMetrics.length > 0 && (
            <section className="context-metrics" aria-label="Context Budget Metrics">
              <div className="context-metrics__heading">
                <span>CONTEXT BUDGET MANAGER</span>
                <b>{Math.round(contextSavings * 100)}% SAVED</b>
              </div>
              <div className="context-metrics__summary">
                <div><strong>{totalContextBefore}</strong><span>压缩前 Token</span></div>
                <div><strong>{totalContextAfter}</strong><span>实际进入 Context</span></div>
                <div><strong>{compressedContextItems}</strong><span>层级压缩</span></div>
                <div><strong>{prunedContextItems}</strong><span>MMR / Budget 裁剪</span></div>
              </div>
              <div className="context-metrics__runs">
                {recentContextMetrics.map((item) => (
                  <article key={item.manifest_id}>
                    <div>
                      <b>{item.node_name}</b>
                      <span>{item.token_after} / {item.input_budget} tokens</span>
                    </div>
                    <div className="context-budget-bar">
                      <i style={{width: `${Math.min(100, Math.round(item.token_after / Math.max(1, item.input_budget) * 100))}%`}} />
                    </div>
                    <small>
                      {item.selected_count} selected · {item.rejected_count} pruned · {item.compressed_count} compressed
                      {item.truncated ? " · budget guarded" : ""}
                    </small>
                  </article>
                ))}
              </div>
            </section>
          )}
          <section className="explainability-grid" aria-label="Agent Explainability">
            <article className="explainability-card">
              <div className="context-metrics__heading"><span>KNOWN / UNKNOWN / NEXT</span><b>STATE LEDGER</b></div>
              <div className="ledger-row"><span>已知 Known</span><strong>{knowledgeLedger?.known.length ?? acceptedEvidence} 条 Claim</strong></div>
              <div className="ledger-row"><span>未知 Unknown</span><strong>{openGapCount} 个 Gap</strong></div>
              <div className="ledger-row"><span>下一步 Next</span><strong>{nextActionText}</strong></div>
              <small>{typeof serverNextAction?.public_decision_summary === "string"
                ? serverNextAction.public_decision_summary
                : nextAction?.public_summary ?? "Evaluator 将根据覆盖度、来源质量和信息增益决定下一动作。"}</small>
            </article>
            <article className="explainability-card">
              <div className="context-metrics__heading"><span>RESEARCH MEMORY</span><b>{memoryHits} HIT / {memoryMisses} MISS</b></div>
              <div className="ledger-row"><span>Working / Episodic</span><strong>{events.length} 条事件</strong></div>
              <div className="ledger-row"><span>Semantic Retrieval</span><strong>{memoryAccesses.length} 次</strong></div>
              <small>历史记忆只作为线索，重新验证后才能进入当前 Evidence。</small>
            </article>
            <article className="explainability-card">
              <div className="context-metrics__heading"><span>EVALUATION</span><b>{Math.round(coverage * 100)}%</b></div>
              <div className="ledger-row"><span>Source Quality</span><strong>{Math.round(Number(latestEvaluation?.source_quality ?? metric(quality, "source_quality")) * 100)}%</strong></div>
              <div className="ledger-row"><span>Cross Validation</span><strong>{Math.round(metric(quality, "cross_validation") * 100)}%</strong></div>
              <div className="ledger-row"><span>Information Gain</span><strong>{Math.round(informationGain * 100)}%</strong></div>
              <small>{latestEvaluation?.verdict
                ? `服务端 Evaluation verdict：${latestEvaluation.verdict}`
                : latestResearchDecision?.public_summary ?? "尚未产生 Evaluation 决策。"}</small>
            </article>
            <article className="explainability-card">
              <div className="context-metrics__heading"><span>MODEL CALLS</span><b>{llmCalls.length}</b></div>
              <div className="ledger-row"><span>成功 / 失败</span><strong>{llmCalls.filter((call) => call.status === "success").length} / {llmCalls.filter((call) => call.status !== "success").length}</strong></div>
              <div className="ledger-row"><span>重试调用</span><strong>{llmCalls.filter((call) => call.retry_mode !== "none").length}</strong></div>
              <small>{llmCalls.length > 0
                ? `最近：${llmCalls[llmCalls.length - 1].node} · ${llmCalls[llmCalls.length - 1].strategy} · ${llmCalls[llmCalls.length - 1].latency_ms}ms`
                : "尚未记录模型调用诊断。"}</small>
            </article>
          </section>
          <div className="timeline">
            {events.length === 0 ? (
              <div>
                <time>—</time>
                <p><b>Waiting</b>创建任务后显示真实操作轨迹</p>
              </div>
            ) : events.map((item) => (
              <div key={item.seq}>
                <time>#{String(item.seq).padStart(3, "0")}</time>
                <p>
                  <b>{eventLabel(item.event_type)}</b>
                  {item.public_summary}
                </p>
              </div>
            ))}
          </div>
        </article>

        <article className="panel evidence-panel">
          <div className="panel-heading">
            <span>EVIDENCE</span>
            <b>{acceptedEvidence} accepted · 新任务接受线 ≥78%</b>
          </div>
          <div className="metric-grid">
            <div><strong>{Math.round(coverage * 100)}%</strong><span>覆盖度</span></div>
            <div><strong>{sources}</strong><span>来源</span></div>
            <div><strong>{conflicts}</strong><span>冲突</span></div>
            <div><strong>{citations}</strong><span>引用</span></div>
            <div><strong>{Math.round(informationGain * 100)}%</strong><span>本轮信息增益</span></div>
            <div><strong>{lowGainStreak}</strong><span>低增益连续轮次</span></div>
          </div>
          {latestResearchDecision && (
            <section className="research-decision">
              <span>WHY CONTINUE / STOP</span>
              <p>{latestResearchDecision.public_summary}</p>
            </section>
          )}
          <p className="empty-state">
            {activeRun
              ? `运行状态：${activeRun.status} / ${activeRun.phase}；质量指标只读取服务端快照。`
              : "研究开始后，这里显示证据质量、来源独立性和冲突状态。"}
          </p>
          <div className="evidence-list">
            {evidence.length === 0 ? (
              <p>尚无经过网页原文校验的候选证据。</p>
            ) : evidence.map((item) => (
              <article
                className={`evidence-card ${item.accepted ? "evidence-card--accepted" : "evidence-card--rejected"}`}
                key={item.evidence_id}
              >
                <div>
                  <span>{item.question_id}</span>
                  <b>{item.accepted ? "ACCEPTED" : "REJECTED"}</b>
                  <strong>{Math.round(item.evidence_score * 100)}%</strong>
                </div>
                <h3>{item.claim}</h3>
                <blockquote>{item.exact_quote}</blockquote>
                <small>
                  来源 {Math.round(item.source_reliability * 100)}%
                  · 相关 {Math.round(item.relevance * 100)}%
                  · 置信 {Math.round(item.confidence * 100)}%
                </small>
                <a href={item.source_url} target="_blank" rel="noreferrer">
                  {item.source_title} · {item.source_domain}
                </a>
                {!item.accepted && item.rejection_reason ? (
                  <small>{item.rejection_reason}</small>
                ) : null}
              </article>
            ))}
          </div>
        </article>
      </section>

      {report ? (
        <section className="report-viewer">
          <div className="panel-heading">
            <span>RESEARCH REPORT / V{report.version}</span>
            <b>
              {Math.round(metric(report.verification_result, "citation_completeness") * 100)}% 引用完整
              · {report.citations.length} 条引用
            </b>
          </div>
          <ReportMarkdown report={report} />
          <div className="report-sources">
            <h3>引用证据</h3>
            {report.citations.map((citation) => (
              <article key={citation.citation_number}>
                <strong>[{citation.citation_number}] {citation.claim}</strong>
                <blockquote>{citation.exact_quote}</blockquote>
                <a href={citation.source_url} target="_blank" rel="noreferrer">
                  {citation.source_title} · {citation.source_domain}
                </a>
                <small>快照 {citation.source_content_hash.slice(0, 12)}</small>
              </article>
            ))}
          </div>
        </section>
      ) : null}
    </main>
  );
}

export default App;
