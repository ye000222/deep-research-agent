import {FormEvent, useEffect, useMemo, useRef, useState} from "react";

import {ProviderProfileForm} from "./ProviderProfileForm";

type ApiHealth = "checking" | "ready" | "unavailable";
type PlanStatus = "done" | "active" | "pending" | "blocked";
type BudgetTier = "quick" | "standard" | "deep";

type ResearchRun = {
  run_id: string;
  status: string;
  phase: string;
  state_version: number;
  termination_reason: string | null;
  quality_snapshot: Record<string, unknown>;
  event_url: string;
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
};

type CoverageDimension = {
  dimension_key: string;
  question: string;
  priority: number;
  coverage: number;
  accepted_evidence: number;
  independent_sources: number;
  missing_reasons: string[];
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

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const STOPPED_STATUSES = new Set([
  "completed",
  "completed_with_limitations",
  "failed",
  "cancelled",
  "interrupted",
  "credentials_required",
]);

function parseEventStream(raw: string): AgentEvent[] {
  return raw
    .replaceAll("\r\n", "\n")
    .split("\n\n")
    .map((block) => block.split("\n").find((line) => line.startsWith("data: "))?.slice(6))
    .filter((data): data is string => Boolean(data))
    .map((data) => JSON.parse(data) as AgentEvent);
}

function runMessage(run: ResearchRun): string {
  if (run.status === "queued") return `任务 ${run.run_id.slice(0, 8)}… 已入队，等待 Dispatcher 发布。`;
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
  if (run.status === "interrupted" && run.termination_reason === "research_budget_exhausted") {
    return "研究已达到预算上限；现有计划和证据已保留，可直接生成带限制报告。";
  }
  if (run.status === "interrupted" && run.termination_reason === "writer_not_implemented") {
    return "该任务来自旧版本且证据研究已完成，可直接生成研究报告。";
  }
  if (run.status === "completed") return "研究任务已完成，报告和引用均已通过校验。";
  if (run.status === "completed_with_limitations") {
    return "研究任务已完成并生成报告；未达到的质量门槛已在研究限制中披露。";
  }
  if (run.status === "failed") {
    const failures: Record<string, string> = {
      MODEL_REQUEST_INVALID: "模型 API 拒绝了请求，请检查 API 协议是否匹配该服务。",
      MODEL_AUTHENTICATION_FAILED: "模型鉴权失败，请更新 API Key。",
      MODEL_NETWORK_ERROR: "模型服务网络连接失败；Planner 会自动重试三次，耗尽后可恢复任务。",
      MODEL_PROVIDER_UNAVAILABLE: "模型服务暂时不可用；Planner 会自动重试三次。",
      MODEL_RATE_LIMITED: "模型 API 已限流，请稍后重试。",
      MODEL_TIMEOUT: "模型调用超时，请稍后重试。",
      MODEL_OUTPUT_INVALID: "兼容模型两次均未返回可解析的 JSON 研究计划，请检查模型是否支持 JSON 输出。",
      MODEL_OUTPUT_SCHEMA_INVALID: "模型经过一次结构纠正后，研究计划仍未通过 Schema 校验。",
      EVIDENCE_OUTPUT_SCHEMA_INVALID: "单页证据抽取未通过 Schema 校验；新版会隔离该来源并保留其他有效证据。",
    };
    return failures[run.termination_reason ?? ""] ?? `研究任务失败：${run.termination_reason ?? "未知原因"}`;
  }
  if (run.status === "cancelled") return "研究任务已取消。";
  return `任务状态 ${run.status}，阶段 ${run.phase}。`;
}

function eventLabel(eventType: string): string {
  const labels: Record<string, string> = {
    "run.created": "Research API",
    "run.started": "Celery Worker",
    "state.initialized": "State Runtime",
    "state.patch_applied": "State Runtime",
    "run.interrupted": "Execution Guard",
    "run.failed": "Failure Boundary",
    "run.cancelled": "Research API",
    "run.status_changed": "Run Lifecycle",
    "plan.generated": "Research Planner",
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
    }];
  });
}

function ReportMarkdown({report}: {report: ResearchReport}) {
  const citations = new Map(
    report.citations.map((citation) => [citation.citation_number, citation]),
  );

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
        if (line.startsWith("# ")) return <h2 key={index}>{renderText(line.slice(2))}</h2>;
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
  const eventCursor = useRef(0);
  const [query, setQuery] = useState(
    "研究工业视觉缺陷检测领域的发展情况，分析技术路线、厂商、代表产品、大模型应用与未来三年趋势。",
  );
  const [message, setMessage] = useState("正在从服务端恢复模型配置…");

  useEffect(() => {
    fetch(`${API_BASE_URL}/healthz`)
      .then((response) => {
        if (!response.ok) throw new Error("unhealthy");
        setHealth("ready");
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
        const [statusResponse, eventResponse, evidenceResponse, contextResponse, memoryResponse, knowledgeResponse, gapsResponse, actionsResponse, evaluationsResponse] = await Promise.all([
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
        if (incoming.length > 0) {
          eventCursor.current = Math.max(eventCursor.current, ...incoming.map((item) => item.seq));
          setEvents((current) => {
            const merged = new Map(current.map((item) => [item.seq, item]));
            incoming.forEach((item) => merged.set(item.seq, item));
            return [...merged.values()].sort((left, right) => left.seq - right.seq);
          });
        }
        if (!cancelled) {
          setActiveRun(run);
          setEvidence(currentEvidence);
          setContextMetrics(currentContextMetrics);
          setMemoryAccesses(currentMemoryAccesses);
          setKnowledgeLedger(currentKnowledge);
          setGapLedger(currentGaps);
          setActionLedger(currentActions);
          setEvaluations(currentEvaluations);
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
        const active = events.some(
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
        const failedActive = activeRun?.status === "failed" && active;
        const status: PlanStatus =
          finalStatus === "researched"
            ? "done"
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
        };
      });
    }

    const created = events.some((item) => item.event_type === "run.created");
    const started = events.some((item) => item.event_type === "run.started");
    const interrupted = events.some((item) => item.event_type === "run.interrupted");
    const failed = events.some((item) => item.event_type === "run.failed");
    return [
      {number: "01", title: "任务与不可变配置落库", status: created ? "done" : "active"},
      {
        number: "02",
        title: "Outbox 发布与 Worker Lease",
        status: started ? "done" : created ? "active" : "pending",
      },
      {
        number: "03",
        title: "Planner 生成研究计划",
        status: interrupted || failed ? "blocked" : started ? "active" : "pending",
      },
      {number: "04", title: "Research Loop 与证据评估", status: "pending"},
      {number: "05", title: "报告、引用与事实核验", status: "pending"},
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
  const coverageMap = coverageDimensions(quality);
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
          API {health === "ready" ? "已连接" : health === "checking" ? "检查中" : "未连接"}
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
              <option value="standard">Standard · 15 轮 / 15 次搜索 / 30 页</option>
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
              {["research_budget_exhausted", "writer_not_implemented"].includes(
                activeRun.termination_reason ?? "",
              ) ? "生成研究报告" : "继续当前任务"}
            </button>
          ) : null}
          <button className="start-button" type="submit" disabled={creatingRun || endingRun}>
            {creatingRun ? "处理中…" : endingRun ? "结束中…" : "开始研究"} <span>→</span>
          </button>
        </div>
        <p className="form-message">{message}</p>
      </form>

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
                <p>{item.title}</p>
                <i>{item.status === "done" ? "✓" : item.status === "active" ? "●" : item.status === "blocked" ? "!" : "○"}</i>
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
            <b>{acceptedEvidence} accepted</b>
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
                  <strong>{Math.round(item.evidence_score * 100)}</strong>
                </div>
                <h3>{item.claim}</h3>
                <blockquote>{item.exact_quote}</blockquote>
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
