"""Explicit adapters for the four supported model API protocols."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, cast
from urllib.parse import quote

import httpx
from pydantic import SecretStr

from app.domain.providers import (
    AdapterType,
    CanonicalModelRequest,
    CanonicalModelResult,
    TokenUsage,
    UsageAccuracy,
)


class ModelGatewayError(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        retryable: bool,
        detail_code: str | None = None,
        usage: TokenUsage | None = None,
        diagnostics: Mapping[str, str | int] | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.detail_code = detail_code
        self.usage = usage
        self.diagnostics = dict(diagnostics or {})


class LLMGateway:
    """Translate canonical requests without retaining credentials or conversation state."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def generate_structured(
        self,
        *,
        adapter_type: AdapterType,
        base_url: str,
        api_key: SecretStr,
        request: CanonicalModelRequest,
        allow_regeneration: bool = True,
    ) -> CanonicalModelResult:
        if request.response_contract is None:
            raise ValueError("structured generation requires a response contract")
        prompt = _content_text(request)
        if adapter_type == AdapterType.OPENAI_RESPONSES:
            payload = await self._openai_responses(base_url, api_key, request, prompt)
            text = _openai_response_text(payload)
            usage = _usage(
                payload.get("usage"),
                input_key="input_tokens",
                output_key="output_tokens",
            )
            request_id = _string_or_none(payload.get("id"))
            finish_reason = _string_or_none(payload.get("status"))
            strategy = "native_json_schema"
        elif adapter_type == AdapterType.ANTHROPIC_MESSAGES:
            payload = await self._anthropic_messages(base_url, api_key, request, prompt)
            text = _anthropic_text(payload)
            usage = _usage(
                payload.get("usage"),
                input_key="input_tokens",
                output_key="output_tokens",
            )
            request_id = _string_or_none(payload.get("id"))
            finish_reason = _string_or_none(payload.get("stop_reason"))
            strategy = "native_json_schema"
        elif adapter_type == AdapterType.GOOGLE_GEMINI:
            payload = await self._google_gemini(base_url, api_key, request, prompt)
            text = _gemini_text(payload)
            usage = _usage(
                payload.get("usageMetadata"),
                input_key="promptTokenCount",
                output_key="candidatesTokenCount",
                total_key="totalTokenCount",
            )
            request_id = _string_or_none(payload.get("responseId"))
            finish_reason = _gemini_finish_reason(payload)
            strategy = "native_json_schema"
        elif adapter_type == AdapterType.OPENAI_COMPATIBLE_CHAT:
            payload, used_json_mode = await self._openai_compatible(
                base_url,
                api_key,
                request,
                prompt,
            )
            text = _chat_completion_text(payload)
            usage = _usage(
                payload.get("usage"),
                input_key="prompt_tokens",
                output_key="completion_tokens",
                total_key="total_tokens",
            )
            request_id = _string_or_none(payload.get("id"))
            finish_reason = _chat_completion_finish_reason(payload)
            strategy = "json_mode" if used_json_mode else "prompt_json"
        else:  # pragma: no cover - exhaustive enum guard
            raise ModelGatewayError("UNSUPPORTED_ADAPTER", retryable=False)

        try:
            parsed = _parse_json_object(text)
        except ModelGatewayError as exc:
            if (
                not allow_regeneration
                or adapter_type != AdapterType.OPENAI_COMPATIBLE_CHAT
                or exc.code != "MODEL_OUTPUT_INVALID"
            ):
                error_code = _structured_output_error_code(
                    exc.code,
                    finish_reason=finish_reason,
                )
                raise ModelGatewayError(
                    error_code,
                    retryable=exc.retryable,
                    detail_code=_structured_output_failure_detail(
                        strategy=strategy,
                        finish_reason=finish_reason,
                        text=text,
                        parse_detail=exc.detail_code,
                    ),
                    usage=usage,
                    diagnostics=_structured_output_diagnostics(
                        strategy=strategy,
                        finish_reason=finish_reason,
                        usage=usage,
                        max_output_tokens=request.max_output_tokens,
                        text=text,
                        provider_request_id=request_id,
                        retry_mode=request.metadata.get("retry_mode", "none"),
                    ),
                ) from exc
            retry_payload, retry_json_mode = await self._openai_compatible(
                base_url,
                api_key,
                request,
                prompt,
                repair=True,
                force_prompt_json=not _finish_reason_is_length(finish_reason),
                compact_repair=_finish_reason_is_length(finish_reason),
            )
            text = _chat_completion_text(retry_payload)
            retry_usage = _usage(
                retry_payload.get("usage"),
                input_key="prompt_tokens",
                output_key="completion_tokens",
                total_key="total_tokens",
            )
            usage = _combine_usage(usage, retry_usage)
            request_id = _string_or_none(retry_payload.get("id"))
            finish_reason = _chat_completion_finish_reason(retry_payload)
            retry_strategy = "json_mode" if retry_json_mode else "prompt_json"
            strategy = f"{retry_strategy}_regenerated_once"
            try:
                parsed = _parse_json_object(text)
            except ModelGatewayError as retry_exc:
                error_code = _structured_output_error_code(
                    retry_exc.code,
                    finish_reason=finish_reason,
                )
                raise ModelGatewayError(
                    error_code,
                    retryable=retry_exc.retryable,
                    detail_code=_structured_output_failure_detail(
                        strategy=strategy,
                        finish_reason=finish_reason,
                        text=text,
                        parse_detail=retry_exc.detail_code,
                    ),
                    usage=usage,
                    diagnostics=_structured_output_diagnostics(
                        strategy=strategy,
                        finish_reason=finish_reason,
                        usage=usage,
                        max_output_tokens=request.max_output_tokens,
                        text=text,
                        provider_request_id=request_id,
                        retry_mode=request.metadata.get(
                            "retry_mode", "generic_regeneration"
                        ),
                    ),
                ) from retry_exc
        return CanonicalModelResult(
            text=text,
            parsed_object=parsed,
            finish_reason=finish_reason,
            usage=usage,
            provider_request_id=request_id,
            capability_strategy={"structured_output": strategy},
        )

    async def _openai_responses(
        self,
        base_url: str,
        api_key: SecretStr,
        request: CanonicalModelRequest,
        prompt: str,
    ) -> dict[str, Any]:
        body = {
            "model": request.model,
            "instructions": request.instructions,
            "input": prompt,
            "max_output_tokens": request.max_output_tokens,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "research_plan",
                    "schema": request.response_contract,
                    "strict": True,
                }
            },
        }
        return await self._post_json(
            _endpoint(base_url, "responses"),
            headers={"Authorization": f"Bearer {api_key.get_secret_value()}"},
            body=body,
        )

    async def _anthropic_messages(
        self,
        base_url: str,
        api_key: SecretStr,
        request: CanonicalModelRequest,
        prompt: str,
    ) -> dict[str, Any]:
        body = {
            "model": request.model,
            "max_tokens": request.max_output_tokens,
            "system": request.instructions,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": request.response_contract,
                }
            },
        }
        return await self._post_json(
            _endpoint(base_url, "messages"),
            headers={
                "x-api-key": api_key.get_secret_value(),
                "anthropic-version": "2023-06-01",
            },
            body=body,
        )

    async def _google_gemini(
        self,
        base_url: str,
        api_key: SecretStr,
        request: CanonicalModelRequest,
        prompt: str,
    ) -> dict[str, Any]:
        model = quote(request.model.removeprefix("models/"), safe="")
        body = {
            "systemInstruction": {"parts": [{"text": request.instructions}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": request.max_output_tokens,
                "responseMimeType": "application/json",
                "responseSchema": request.response_contract,
            },
        }
        return await self._post_json(
            _endpoint(base_url, f"models/{model}:generateContent"),
            headers={"x-goog-api-key": api_key.get_secret_value()},
            body=body,
        )

    async def _openai_compatible(
        self,
        base_url: str,
        api_key: SecretStr,
        request: CanonicalModelRequest,
        prompt: str,
        *,
        repair: bool = False,
        force_prompt_json: bool = False,
        compact_repair: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        schema = json.dumps(
            request.response_contract,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        system_prompt = (
            f"{request.instructions}\n\n"
            "只输出一个 JSON 对象, 不要输出 Markdown、代码围栏或解释文字。\n"
            f"必须满足以下 JSON Schema:\n{schema}"
        )
        user_prompt = prompt
        if repair:
            compact_instruction = (
                "上一次输出达到长度上限。请显著缩短所有字符串和数组, "
                "只保留满足 Schema 的最少内容, 并务必在 token 上限内闭合 JSON。"
                if compact_repair
                else "前一次响应无法解析。请重新生成完整 JSON。"
            )
            user_prompt = (
                f"{prompt}\n\n"
                f"{compact_instruction} 确保首字符是 {{, 末字符是 }}。"
            )
        body: dict[str, Any] = {
            "model": request.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": request.max_output_tokens,
            "temperature": _generation_temperature(request),
        }
        if not force_prompt_json:
            body["response_format"] = {"type": "json_object"}
        url = _endpoint(base_url, "chat/completions")
        headers = {"Authorization": f"Bearer {api_key.get_secret_value()}"}
        if force_prompt_json:
            return await self._post_json(url, headers=headers, body=body), False
        try:
            return await self._post_json(url, headers=headers, body=body), True
        except ModelGatewayError as exc:
            if exc.code != "MODEL_REQUEST_INVALID":
                raise
        body.pop("response_format")
        return await self._post_json(url, headers=headers, body=body), False

    async def _post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        try:
            # Keep the worker-scoped HTTP connection reusable. A research run makes
            # several sequential calls to the same provider; forcing `Connection:
            # close` made every compact/schema retry pay for a fresh DNS/TCP/TLS
            # handshake and turned otherwise healthy follow-up calls into sporadic
            # connect timeouts.
            response = await self._client.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise ModelGatewayError(
                "MODEL_TIMEOUT",
                retryable=True,
                detail_code=_request_error_detail(exc),
            ) from exc
        except httpx.RequestError as exc:
            raise ModelGatewayError(
                "MODEL_NETWORK_ERROR",
                retryable=True,
                detail_code=_request_error_detail(exc),
            ) from exc
        if 300 <= response.status_code < 400:
            raise ModelGatewayError("PROVIDER_REDIRECT_BLOCKED", retryable=False)
        if response.status_code in {401, 403}:
            raise ModelGatewayError("MODEL_AUTHENTICATION_FAILED", retryable=False)
        if response.status_code == 429:
            raise ModelGatewayError("MODEL_RATE_LIMITED", retryable=True)
        if response.status_code >= 500:
            raise ModelGatewayError("MODEL_PROVIDER_UNAVAILABLE", retryable=True)
        if response.status_code >= 400:
            raise ModelGatewayError("MODEL_REQUEST_INVALID", retryable=False)
        payload: object = response.json()
        if not isinstance(payload, dict):
            raise ModelGatewayError("MODEL_RESPONSE_INVALID", retryable=False)
        return cast(dict[str, Any], payload)


def _request_error_detail(exc: httpx.RequestError) -> str:
    categories: tuple[tuple[type[httpx.RequestError], str], ...] = (
        (httpx.ConnectTimeout, "CONNECT_TIMEOUT"),
        (httpx.ReadTimeout, "READ_TIMEOUT"),
        (httpx.WriteTimeout, "WRITE_TIMEOUT"),
        (httpx.PoolTimeout, "POOL_TIMEOUT"),
        (httpx.ConnectError, "CONNECT_ERROR"),
        (httpx.ReadError, "READ_ERROR"),
        (httpx.WriteError, "WRITE_ERROR"),
        (httpx.RemoteProtocolError, "REMOTE_PROTOCOL_ERROR"),
        (httpx.LocalProtocolError, "LOCAL_PROTOCOL_ERROR"),
    )
    for error_type, detail_code in categories:
        if isinstance(exc, error_type):
            return detail_code
    return "REQUEST_ERROR"


def _endpoint(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _content_text(request: CanonicalModelRequest) -> str:
    parts = [part.value for part in request.content_parts if part.kind == "text"]
    return "\n\n".join(str(value) for value in parts)


def _openai_response_text(payload: Mapping[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct:
        return direct
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str):
                        return text
    raise ModelGatewayError("MODEL_RESPONSE_INVALID", retryable=False)


def _anthropic_text(payload: Mapping[str, Any]) -> str:
    content = payload.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    return text
    raise ModelGatewayError("MODEL_RESPONSE_INVALID", retryable=False)


def _gemini_text(payload: Mapping[str, Any]) -> str:
    candidates = payload.get("candidates")
    if isinstance(candidates, list) and candidates:
        candidate = candidates[0]
        if isinstance(candidate, dict):
            content = candidate.get("content")
            if isinstance(content, dict):
                parts = content.get("parts")
                if isinstance(parts, list):
                    texts = [part.get("text") for part in parts if isinstance(part, dict)]
                    joined = "".join(text for text in texts if isinstance(text, str))
                    if joined:
                        return joined
    raise ModelGatewayError("MODEL_RESPONSE_INVALID", retryable=False)


def _chat_completion_text(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    text = "".join(
                        str(part.get("text", ""))
                        for part in content
                        if isinstance(part, dict) and isinstance(part.get("text"), str)
                    )
                    if text:
                        return text
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tool_call in tool_calls:
                        if not isinstance(tool_call, dict):
                            continue
                        function = tool_call.get("function")
                        if isinstance(function, dict) and isinstance(
                            function.get("arguments"), str
                        ):
                            return str(function["arguments"])
    raise ModelGatewayError("MODEL_RESPONSE_INVALID", retryable=False)


def _parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    # Reasoning-capable compatible models may wrap the final answer in a
    # private think block. It is not part of the JSON contract.
    candidate = re.sub(
        r"<think>.*?</think>", "", candidate, flags=re.IGNORECASE | re.DOTALL
    ).strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            candidate = "\n".join(lines[1:-1])
            if candidate.lstrip().startswith("json"):
                candidate = candidate.lstrip()[4:].lstrip()
    last_error: json.JSONDecodeError | None = None
    try:
        parsed: object = json.loads(candidate)
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
    except json.JSONDecodeError as exc:
        last_error = exc

    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate, index)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return cast(dict[str, Any], parsed)
    raise ModelGatewayError(
        "MODEL_OUTPUT_INVALID",
        retryable=False,
        detail_code=_json_parse_failure_detail(candidate, last_error),
    )


def _chat_completion_finish_reason(payload: Mapping[str, Any]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    return _string_or_none(first.get("finish_reason"))


def _gemini_finish_reason(payload: Mapping[str, Any]) -> str | None:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    first = candidates[0]
    if not isinstance(first, dict):
        return None
    return _string_or_none(first.get("finishReason"))


def _structured_output_failure_detail(
    *,
    strategy: str,
    finish_reason: str | None,
    text: str,
    parse_detail: str | None,
) -> str:
    safe_strategy = re.sub(r"[^A-Z0-9]+", "_", strategy.upper()).strip("_")
    safe_finish = re.sub(
        r"[^A-Z0-9]+", "_", (finish_reason or "UNKNOWN").upper()
    ).strip("_")
    safe_parse = re.sub(
        r"[^A-Z0-9]+", "_", (parse_detail or "PARSE_UNKNOWN").upper()
    ).strip("_")
    return (
        f"OUTPUT_INVALID_{safe_strategy}_FINISH_{safe_finish}_"
        f"{safe_parse}_CHARS_{len(text)}"
    )[:100]


def _structured_output_diagnostics(
    *,
    strategy: str,
    finish_reason: str | None,
    usage: TokenUsage,
    max_output_tokens: int,
    text: str,
    provider_request_id: str | None,
    retry_mode: str,
) -> dict[str, str | int]:
    return {
        "structured_output_strategy": strategy,
        "finish_reason": finish_reason or "unknown",
        "output_tokens": usage.output_tokens,
        "max_output_tokens": max_output_tokens,
        "response_length": len(text),
        "provider_request_id": provider_request_id or "unknown",
        "retry_mode": retry_mode,
    }


def _generation_temperature(request: CanonicalModelRequest) -> float:
    value = request.generation_parameters.get("temperature", 0.2)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return max(0.0, min(float(value), 2.0))
    return 0.2


def _finish_reason_is_length(finish_reason: str | None) -> bool:
    normalized = (finish_reason or "").casefold()
    return normalized in {"length", "max_tokens", "max_output_tokens"}


def _structured_output_error_code(
    parse_error_code: str,
    *,
    finish_reason: str | None,
) -> str:
    if parse_error_code == "MODEL_OUTPUT_INVALID" and _finish_reason_is_length(
        finish_reason
    ):
        return "MODEL_OUTPUT_TRUNCATED"
    return parse_error_code


def _json_parse_failure_detail(
    candidate: str,
    error: json.JSONDecodeError | None,
) -> str:
    if not candidate:
        return "PARSE_EMPTY_RESPONSE"
    if candidate.startswith("{") and not candidate.rstrip().endswith("}"):
        return "PARSE_TRUNCATED_OBJECT"
    if re.search(r",\s*[}\]]\s*$", candidate):
        return "PARSE_TRAILING_COMMA"
    if error is None:
        return "PARSE_NON_OBJECT_JSON"
    safe_message = re.sub(r"[^A-Z0-9]+", "_", error.msg.upper()).strip("_")
    return f"PARSE_{safe_message[:32]}_AT_{error.pos}"


def _combine_usage(first: TokenUsage, second: TokenUsage) -> TokenUsage:
    accuracy = (
        UsageAccuracy.EXACT
        if first.accuracy == second.accuracy == UsageAccuracy.EXACT
        else UsageAccuracy.UNAVAILABLE
    )
    return TokenUsage(
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        total_tokens=first.total_tokens + second.total_tokens,
        accuracy=accuracy,
    )


def _usage(
    raw: object,
    *,
    input_key: str,
    output_key: str,
    total_key: str | None = None,
) -> TokenUsage:
    values = raw if isinstance(raw, dict) else {}
    input_tokens = _non_negative_int(values.get(input_key))
    output_tokens = _non_negative_int(values.get(output_key))
    total_tokens = _non_negative_int(values.get(total_key)) if total_key else 0
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        accuracy=UsageAccuracy.EXACT if raw is not None else UsageAccuracy.UNAVAILABLE,
    )


def _non_negative_int(value: object) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None
