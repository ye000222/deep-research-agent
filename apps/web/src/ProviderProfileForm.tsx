import {useEffect, useRef, useState} from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

const DEFAULT_BASE_URLS: Record<string, string> = {
  openai_responses: "https://api.openai.com/v1",
  anthropic_messages: "https://api.anthropic.com/v1",
  google_gemini: "https://generativelanguage.googleapis.com/v1beta",
  openai_compatible_chat: "https://api.openai.com/v1",
};

type ProviderProfile = {
  profile_id: string;
  name: string;
  adapter_type: string;
  base_url: string;
  model: string;
  credential_version_id: string;
  credential_last_four: string;
  has_saved_credential: boolean;
};

type Props = {
  onStatusChange: (
    configured: boolean,
    message: string,
    credentialVersionId?: string,
  ) => void;
};

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "未知错误";
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    credentials: "include",
    headers: {"Content-Type": "application/json", ...init?.headers},
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail?.message ?? body.detail?.error_code ?? detail;
    } catch {
      // Keep the status fallback when the response body is not JSON.
    }
    throw new Error(detail);
  }
  return response.status === 204 ? (undefined as T) : response.json();
}

export function ProviderProfileForm({onStatusChange}: Props) {
  const [profile, setProfile] = useState<ProviderProfile | null>(null);
  const [name, setName] = useState("默认研究模型");
  const [adapterType, setAdapterType] = useState("openai_responses");
  const [baseUrl, setBaseUrl] = useState(DEFAULT_BASE_URLS.openai_responses);
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);

  useEffect(() => {
    void loadProfile();
  }, []);

  async function loadProfile() {
    try {
      const profiles = await api<ProviderProfile[]>("/api/v1/llm/profiles");
      const saved = profiles.find((item) => item.has_saved_credential) ?? null;
      setProfile(saved);
      if (saved) {
        setName(saved.name);
        setAdapterType(saved.adapter_type);
        setBaseUrl(saved.base_url);
        setModel(saved.model);
        onStatusChange(
          true,
          `已从服务端恢复模型配置；API Key 已加密保存（尾号 ${saved.credential_last_four}）。`,
          saved.credential_version_id,
        );
      } else {
        onStatusChange(false, "尚未保存模型配置。首次保存后，刷新页面也会自动恢复。");
      }
    } catch (error) {
      onStatusChange(false, `读取模型配置失败：${errorMessage(error)}`);
    }
  }

  async function saveProfile() {
    if (busyRef.current) return;
    if (
      profile &&
      !apiKey.trim() &&
      name.trim() === profile.name &&
      adapterType === profile.adapter_type &&
      baseUrl.trim() === profile.base_url &&
      model.trim() === profile.model
    ) {
      onStatusChange(
        true,
        "配置没有变化，无需重复更新。",
        profile.credential_version_id,
      );
      return;
    }
    if (!name.trim() || !baseUrl.trim() || !model.trim()) {
      onStatusChange(Boolean(profile), "连接名称、Base URL 和模型名称不能为空。");
      return;
    }
    if (!profile && !apiKey.trim()) {
      onStatusChange(false, "首次保存必须输入 API Key。");
      return;
    }
    busyRef.current = true;
    setBusy(true);
    try {
      let saved: ProviderProfile;
      if (!profile) {
        saved = await api<ProviderProfile>("/api/v1/llm/profiles", {
          method: "POST",
          body: JSON.stringify({
            name: name.trim(),
            adapter_type: adapterType,
            base_url: baseUrl.trim(),
            model: model.trim(),
            api_key: apiKey,
            is_default: true,
          }),
        });
      } else {
        saved = await api<ProviderProfile>(`/api/v1/llm/profiles/${profile.profile_id}`, {
          method: "PATCH",
          body: JSON.stringify({
            adapter_type: adapterType,
            name: name.trim(),
            base_url: baseUrl.trim(),
            model: model.trim(),
            is_default: true,
          }),
        });
        if (apiKey.trim()) {
          saved = await api<ProviderProfile>(
            `/api/v1/llm/profiles/${profile.profile_id}/credentials/rotate`,
            {method: "POST", body: JSON.stringify({api_key: apiKey})},
          );
        }
      }
      setProfile(saved);
      setApiKey("");
      onStatusChange(
        true,
        `配置已持久化；密钥尾号 ${saved.credential_last_four}，明文不会返回浏览器。`,
        saved.credential_version_id,
      );
    } catch (error) {
      onStatusChange(Boolean(profile), `保存失败：${errorMessage(error)}`);
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  async function deleteProfile() {
    if (!profile || busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    try {
      await api<void>(`/api/v1/llm/profiles/${profile.profile_id}`, {method: "DELETE"});
      setProfile(null);
      setApiKey("");
      onStatusChange(false, "已删除服务端保存的模型配置和密钥版本。");
    } catch (error) {
      onStatusChange(true, `删除失败：${errorMessage(error)}`);
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  return (
    <div className="connection-grid">
      <label>
        <span>连接名称</span>
        <input value={name} onChange={(event) => setName(event.target.value)} />
      </label>
      <label>
        <span>API 协议</span>
        <select
          value={adapterType}
          onChange={(event) => {
            setAdapterType(event.target.value);
            setBaseUrl(DEFAULT_BASE_URLS[event.target.value]);
          }}
        >
          <option value="openai_responses">OpenAI Responses</option>
          <option value="anthropic_messages">Anthropic Messages</option>
          <option value="google_gemini">Google Gemini</option>
          <option value="openai_compatible_chat">OpenAI Compatible（DeepSeek / Qwen 等）</option>
        </select>
      </label>
      <label className="wide-field">
        <span>Base URL</span>
        <input value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} />
      </label>
      <label>
        <span>模型名称</span>
        <input value={model} onChange={(event) => setModel(event.target.value)} placeholder="模型 ID" />
      </label>
      <label>
        <span>API Key</span>
        <input
          type="password"
          value={apiKey}
          onChange={(event) => setApiKey(event.target.value)}
          autoComplete="new-password"
          placeholder={profile ? `已保存 ····${profile.credential_last_four}（留空不修改）` : "首次保存时必填"}
        />
      </label>
      <div className="connection-actions">
        {profile && (
          <button className="danger-button" type="button" disabled={busy} onClick={() => void deleteProfile()}>
            删除配置
          </button>
        )}
        <button className="secondary-button" type="button" disabled={busy} onClick={() => void saveProfile()}>
          {busy ? "保存中…" : profile ? "更新配置" : "保存配置"}
        </button>
      </div>
    </div>
  );
}
