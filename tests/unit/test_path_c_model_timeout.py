"""Virtual-clock body delay obeys actual request extensions; no sockets/sleeps."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
import pytest
from app.domain.providers import AdapterType
from app.llm.model_http_policy import model_generation_timeout
from pydantic import SecretStr

from scripts.prospective_acquisition import path_c_generation as generation
from scripts.prospective_acquisition.capture_transport import RawResponseCaptureTransport
from scripts.prospective_acquisition.control import CallBudget
from scripts.prospective_acquisition.path_c_failure_diagnostics import (
    FailureDiagnosticsRecorder,
    ObservedPathCGateway,
)
from scripts.prospective_acquisition.path_c_generation_entry import build_path_c_model_client
from tests.unit.test_path_c_generation import _context, _copy_frozen_root, _tasks


class DelayedBody(httpx.AsyncByteStream):
    def __init__(self, request: httpx.Request, delay: float, content: bytes) -> None:
        self.request = request
        self.delay = delay
        self.content = content

    async def __aiter__(self) -> AsyncIterator[bytes]:
        # MockTransport itself does not enforce timeouts. Explicitly model one
        # delayed network-read interval against the propagated HTTPX read limit.
        limit = self.request.extensions['timeout']['read']
        if self.delay > limit:
            raise httpx.ReadTimeout('synthetic read inactivity', request=self.request)
        yield self.content


@pytest.mark.parametrize('delay,connect,old_policy,expected', [
    (15.0, False, False, None), (91.0, False, False, 'READ_TIMEOUT'),
    (0.0, True, False, 'CONNECT_TIMEOUT'), (15.0, False, True, 'READ_TIMEOUT'),
])
def test_real_path_virtual_delay_and_single_dispatch(
    tmp_path: Path, delay: float, connect: bool, old_policy: bool, expected: str | None
) -> None:
    root = _copy_frozen_root(tmp_path)
    out = tmp_path / 'isolated-fake-attempt'
    context, binding = _context()
    budget = CallBudget(1)
    observations = FailureDiagnosticsRecorder()
    requests: list[httpx.Request] = []
    tasks = _tasks()
    body = json.dumps({
        'choices': [
            {'message': {'content': json.dumps({'tasks': tasks})}, 'finish_reason': 'stop'},
        ],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 2, 'total_tokens': 3},
    }).encode()

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.extensions['timeout'] == (
            httpx.Timeout(10.0).as_dict() if old_policy else model_generation_timeout().as_dict()
        )
        payload = json.loads(request.content)
        assert 'seed' not in payload
        assert payload.get('stream', False) is False
        assert payload['response_format'] == {'type': 'json_object'}
        assert '40' in payload['messages'][0]['content']
        if connect:
            raise httpx.ConnectTimeout('synthetic connect inactivity', request=request)
        return httpx.Response(200, stream=DelayedBody(request, delay, body), request=request)

    async def run() -> None:
        capture = RawResponseCaptureTransport(
            httpx.MockTransport(handler), out / 'provider_response.raw.json', observations,
        )
        client = build_path_c_model_client(budget, capture)
        if old_policy:
            client.timeout = httpx.Timeout(10.0)
        async def recheck() -> None:
            return None
        kwargs: dict[str, Any] = dict(
            root=root, output_dir=out, context=context, provider_binding=binding,
            binding_recheck=recheck, gateway=ObservedPathCGateway(client, observations),
            api_key=SecretStr('synthetic-key'), base_url='https://example.invalid/v1',
            adapter_type=AdapterType.OPENAI_COMPATIBLE_CHAT,
            context_manifest_id=UUID(generation.EXECUTION_CONTEXT_ID),
            physical_budget=budget, raw_response_path=out / 'provider_response.raw.json',
            failure_diagnostics=observations,
        )
        try:
            if expected:
                with pytest.raises(generation.PathCGenerationBlocked):
                    await generation._execute_guarded_core(**kwargs)
                state = json.loads((out / 'generation_state.json').read_bytes())
                assert state['state'] == 'CONSUMED_FAILED'
                assert observations.gateway_error_code == 'MODEL_TIMEOUT'
                assert observations.gateway_detail_code == expected
                assert observations.provider_http_status == (None if connect else 200)
                assert observations.response_received == (not connect)
                assert not (out / 'provider_response.raw.json').exists()
            else:
                result = await generation._execute_guarded_core(**kwargs)
                assert result.state == 'SUCCEEDED_AND_FROZEN'
                assert observations.provider_http_status == 200
                assert observations.raw_response_preserved is True
            assert budget.dispatched == 1
            assert observations.physical_outbound_count == 1
            assert len(requests) == 1
        finally:
            await client.aclose()

    asyncio.run(run())


def test_policy_not_total_deadline() -> None:
    assert model_generation_timeout().as_dict() == {
        'connect': 10.0, 'read': 90.0, 'write': 90.0, 'pool': 90.0,
    }
    assert httpx.Timeout(10.0).read == 10.0  # old operator policy
