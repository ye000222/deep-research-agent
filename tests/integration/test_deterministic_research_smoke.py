from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from app.core.config import Settings
from app.infrastructure.db.outbox import TaskDispatchOutboxRepository
from app.infrastructure.db.postgres import PostgresRuntime
from app.infrastructure.db.run_models import TaskDispatchOutboxRow
from app.main import create_app
from app.worker.celery_app import celery_app
from celery.contrib.testing.worker import start_worker
from pydantic import SecretStr
from sqlalchemy import select

from tests.support.deterministic_research import (
    SOURCE_A_URL,
    SOURCE_B_URL,
    DeterministicFixtureLLMGateway,
    DeterministicFixtureSearchProvider,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.deterministic_smoke,
    pytest.mark.skipif(
        os.getenv("RUN_DETERMINISTIC_SMOKE") != "1",
        reason="set RUN_DETERMINISTIC_SMOKE=1 with isolated PostgreSQL, checkpoint DB, and Redis",
    ),
]

_TEST_MASTER_KEY = "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY="
_FOUNDING_SENTENCE = "Example Research Labs was founded in 2018."
_HEADQUARTERS_SENTENCE = "Example Research Labs is headquartered in Berlin, Germany."


def test_deterministic_success_smoke_repeats_three_times(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from app.worker import tasks
    from fastapi.testclient import TestClient

    database_url, checkpoint_url, redis_url = _test_urls(monkeypatch)
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    fixture_dir = Path(__file__).parents[1] / "fixtures" / "research_smoke"
    html_by_url = {
        SOURCE_A_URL: (fixture_dir / "source_a.html").read_text(encoding="utf-8"),
        SOURCE_B_URL: (fixture_dir / "source_b.html").read_text(encoding="utf-8"),
    }
    http_log: list[str] = []
    search_calls: list[str] = []
    model_calls: list[str] = []
    _install_worker_boundaries(
        monkeypatch,
        tasks=tasks,
        html_by_url=html_by_url,
        http_log=http_log,
        search_calls=search_calls,
        model_calls=model_calls,
        empty_evidence=False,
    )

    settings = _app_settings(database_url, checkpoint_url, redis_url, tmp_path)
    queue = f"l2-deterministic-{uuid4().hex}"
    _configure_queue(monkeypatch, queue, redis_url)
    app = create_app(settings)
    results: list[dict[str, object]] = []
    started = time.perf_counter()

    with TestClient(app) as client:
        profile = client.post(
            "/api/v1/llm/profiles",
            json={
                "name": "Deterministic L2 fixture",
                "adapter_type": "openai_compatible_chat",
                "base_url": "https://model.fixture.invalid/v1",
                "model": "deterministic-fixture-v1",
                "api_key": "test-only-never-sent-1234",
                "is_default": True,
            },
        )
        assert profile.status_code == 201, profile.text
        profile_version_id = profile.json()["credential_version_id"]

        with start_worker(
            celery_app,
            queues=[queue],
            pool="solo",
            concurrency=1,
            perform_ping_check=False,
            shutdown_timeout=15,
        ):
            for ordinal in range(1, 4):
                run_started = time.perf_counter()
                created = client.post(
                    "/api/v1/research-runs",
                    headers={"Idempotency-Key": f"l2-success-{uuid4().hex}"},
                    json={
                        "query": (
                            "Summarize the founding year and headquarters city of "
                            "Example Research Labs using the supplied sources."
                        ),
                        "saved_profile_version_id": profile_version_id,
                        "budget_tier": "standard",
                    },
                )
                assert created.status_code == 202, created.text
                run_id = created.json()["run_id"]
                dispatch_errors = _dispatch_created_outbox_once(
                    database_url=database_url,
                    run_id=run_id,
                )
                assert dispatch_errors == []
                outbox_state = asyncio.run(_outbox_state(database_url, run_id))
                assert outbox_state == "published"

                terminal = _wait_for_terminal(client, run_id, timeout_seconds=120)
                elapsed = round(time.perf_counter() - run_started, 3)
                assert terminal["status"] in {"completed", "completed_with_limitations"}

                candidates_response = client.get(f"/api/v1/research-runs/{run_id}/evidence")
                assert candidates_response.status_code == 200
                candidates = candidates_response.json()
                accepted = [item for item in candidates if item["accepted"]]
                assert candidates, f"run {ordinal} produced no candidate evidence"
                assert accepted, f"run {ordinal} produced no accepted evidence"

                report_response = client.get(f"/api/v1/research-runs/{run_id}/report")
                assert report_response.status_code == 200, report_response.text
                verification = client.get(f"/api/v1/research-runs/{run_id}/verification")
                assert verification.status_code == 200, verification.text
                assert verification.json()["verified"] is True
                results.append(
                    {
                        "ordinal": ordinal,
                        "run_id": run_id,
                        "status": terminal["status"],
                        "duration_seconds": elapsed,
                        "candidate_evidence": len(candidates),
                        "accepted_evidence": len(accepted),
                        "report_fetch": report_response.status_code,
                        "report_verified": verification.json()["verified"],
                    }
                )

    assert len(results) == 3
    assert len(search_calls) > 0
    assert "research_planning" in model_calls
    assert "evidence_extraction" in model_calls
    assert "report_writing" in model_calls
    assert set(http_log).issubset(set(html_by_url))
    duration = round(time.perf_counter() - started, 3)
    assert duration < 600, f"L2 success smoke exceeded 10-minute target: {duration}s"
    _write_smoke_artifact(
        {
            "kind": "deterministic_success",
            "repeatability": "3/3 PASS",
            "duration_seconds": duration,
            "runs": results,
            "http_requests": len(http_log),
            "live_llm_calls": 0,
            "search_fixture_calls": len(search_calls),
        }
    )


def test_deterministic_failure_path_does_not_fabricate_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from app.worker import tasks
    from fastapi.testclient import TestClient

    database_url, checkpoint_url, redis_url = _test_urls(monkeypatch)
    monkeypatch.setenv("ARTIFACT_ROOT", str(tmp_path))
    fixture_dir = Path(__file__).parents[1] / "fixtures" / "research_smoke"
    html_by_url = {
        SOURCE_A_URL: (fixture_dir / "source_a.html").read_text(encoding="utf-8"),
        SOURCE_B_URL: (fixture_dir / "source_b.html").read_text(encoding="utf-8"),
    }
    http_log: list[str] = []
    search_calls: list[str] = []
    model_calls: list[str] = []
    _install_worker_boundaries(
        monkeypatch,
        tasks=tasks,
        html_by_url=html_by_url,
        http_log=http_log,
        search_calls=search_calls,
        model_calls=model_calls,
        empty_evidence=True,
    )
    settings = _app_settings(database_url, checkpoint_url, redis_url, tmp_path)
    queue = f"l2-empty-{uuid4().hex}"
    _configure_queue(monkeypatch, queue, redis_url)
    app = create_app(settings)
    started = time.perf_counter()

    with TestClient(app) as client:
        profile = client.post(
            "/api/v1/llm/profiles",
            json={
                "name": "Deterministic L2 empty fixture",
                "adapter_type": "openai_compatible_chat",
                "base_url": "https://model.fixture.invalid/v1",
                "model": "deterministic-fixture-v1",
                "api_key": "test-only-never-sent-1234",
                "is_default": True,
            },
        )
        assert profile.status_code == 201, profile.text
        created = client.post(
            "/api/v1/research-runs",
            headers={"Idempotency-Key": f"l2-empty-{uuid4().hex}"},
            json={
                "query": "Summarize the supplied Example Research Labs sources.",
                "saved_profile_version_id": profile.json()["credential_version_id"],
                "budget_tier": "standard",
            },
        )
        assert created.status_code == 202, created.text
        run_id = created.json()["run_id"]

        with start_worker(
            celery_app,
            queues=[queue],
            pool="solo",
            concurrency=1,
            perform_ping_check=False,
            shutdown_timeout=15,
        ):
            dispatch_errors = _dispatch_created_outbox_once(
                database_url=database_url,
                run_id=run_id,
            )
            assert dispatch_errors == []
            assert asyncio.run(_outbox_state(database_url, run_id)) == "published"
            terminal = _wait_for_terminal(client, run_id, timeout_seconds=90)

        evidence_response = client.get(f"/api/v1/research-runs/{run_id}/evidence")
        assert evidence_response.status_code == 200
        assert evidence_response.json() == []
        report_response = client.get(f"/api/v1/research-runs/{run_id}/report")
        assert report_response.status_code == 409
        assert report_response.json()["detail"]["error_code"] == "REPORT_NOT_READY"
        assert terminal["status"] == "failed"
        assert terminal["termination_reason"] == "REPORT_NO_ACCEPTED_EVIDENCE"

    duration = round(time.perf_counter() - started, 3)
    assert duration < 300, f"L2 failure smoke exceeded 5-minute target: {duration}s"
    _write_smoke_artifact(
        {
            "kind": "deterministic_failure_path",
            "result": "PASS",
            "duration_seconds": duration,
            "run_id": run_id,
            "terminal_status": terminal["status"],
            "termination_reason": terminal["termination_reason"],
            "accepted_evidence": 0,
            "report_fetch_status": report_response.status_code,
            "search_fixture_calls": len(search_calls),
            "http_requests": len(http_log),
            "live_llm_calls": 0,
        }
    )


def _test_urls(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str, str]:
    database_url = os.getenv("L2_DATABASE_URL")
    checkpoint_url = os.getenv("L2_CHECKPOINT_DATABASE_URI")
    redis_url = os.getenv("REDIS_URL")
    if not database_url or not checkpoint_url or not redis_url:
        pytest.skip("L2_DATABASE_URL, L2_CHECKPOINT_DATABASE_URI, and REDIS_URL are required")
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("CHECKPOINT_DATABASE_URI", checkpoint_url)
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.setenv("PUBLIC_WEB_HTTP_PROXY", "")
    monkeypatch.setenv("MODEL_HTTP_PROXY", "")
    monkeypatch.setenv("SECRET_MASTER_KEY_BASE64", _TEST_MASTER_KEY)
    monkeypatch.setenv("PERSIST_PROVIDER_CREDENTIALS", "true")
    return database_url, checkpoint_url, redis_url


def _app_settings(
    database_url: str,
    checkpoint_url: str,
    redis_url: str,
    artifact_root: Path,
) -> Settings:
    return Settings(
        app_env="test",
        database_url=database_url,
        checkpoint_database_uri=checkpoint_url,
        redis_url=redis_url,
        persist_provider_credentials=True,
        secret_master_key_base64=SecretStr(_TEST_MASTER_KEY),
        external_probes_enabled=False,
        artifact_root=artifact_root,
    )


def _configure_queue(monkeypatch: pytest.MonkeyPatch, queue: str, redis_url: str) -> None:
    monkeypatch.setattr(celery_app.conf, "broker_url", redis_url)
    monkeypatch.setattr(celery_app.conf, "task_default_queue", queue)
    monkeypatch.setattr(celery_app.conf, "task_default_exchange", queue)
    monkeypatch.setattr(celery_app.conf, "task_default_routing_key", queue)


def _install_worker_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tasks: Any,
    html_by_url: dict[str, str],
    http_log: list[str],
    search_calls: list[str],
    model_calls: list[str],
    empty_evidence: bool,
) -> None:
    from app.tools import web_reader

    from tests.support import deterministic_research

    monkeypatch.setattr(
        tasks,
        "SearXNGSearchProvider",
        lambda *_args, **_kwargs: DeterministicFixtureSearchProvider(
            empty=False, calls=search_calls
        ),
    )
    monkeypatch.setattr(
        tasks,
        "LLMGateway",
        lambda *_args, **_kwargs: DeterministicFixtureLLMGateway(
            empty_evidence=empty_evidence, calls=model_calls
        ),
    )
    deterministic_research.install_fixture_http_transport(
        monkeypatch,
        html_by_url=html_by_url,
        request_log=http_log,
    )

    original_getaddrinfo = socket.getaddrinfo
    fixture_hosts = {"cognex.com", "keyence.com"}
    infrastructure_hosts = {"127.0.0.1", "localhost", "::1"}

    def fixture_dns(host: str, *args: object, **kwargs: object) -> list[tuple[Any, ...]]:
        if host in infrastructure_hosts:
            port = int(args[0]) if args else 0
            address: tuple[object, ...] = (host, port, 0, 0) if ":" in host else (host, port)
            return [
                (
                    socket.AF_INET6 if ":" in host else socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    address,
                )
            ]
        if host in fixture_hosts:
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("93.184.216.34", 0),
                )
            ]
        raise AssertionError(f"L2 attempted DNS for unapproved host: {host}")

    monkeypatch.setattr(web_reader.socket, "getaddrinfo", fixture_dns)
    # Keep the original resolver available to make accidental unused imports explicit.
    assert callable(original_getaddrinfo)


def _dispatch_created_outbox_once(
    *,
    database_url: str,
    run_id: str,
) -> list[str]:
    return asyncio.run(_publish_outbox_task(database_url, run_id))


async def _publish_outbox_task(database_url: str, run_id: str) -> list[str]:
    database = PostgresRuntime(database_url)
    repository = TaskDispatchOutboxRepository(database.session_factory)
    target_run = UUID(run_id)
    deadline = time.monotonic() + 15
    failures: list[str] = []
    try:
        while time.monotonic() < deadline:
            batch = await repository.claim_batch()
            for dispatch in batch:
                try:
                    await asyncio.to_thread(
                        celery_app.send_task,
                        "deep_research.execute_run",
                        args=(str(dispatch.run_id),),
                        task_id=dispatch.dispatch_key,
                    )
                except Exception as exc:
                    await repository.mark_retry(
                        dispatch.outbox_id,
                        error_code=type(exc).__name__,
                    )
                    if dispatch.run_id == target_run:
                        failures.append(type(exc).__name__)
                else:
                    await repository.mark_published(dispatch.outbox_id)
                    if dispatch.run_id == target_run:
                        return failures
            if not batch:
                await asyncio.sleep(0.05)
        failures.append("outbox_target_not_published_within_15_seconds")
        return failures
    finally:
        await database.close()


async def _outbox_state(database_url: str, run_id: str) -> str | None:
    database = PostgresRuntime(database_url)
    try:
        async with database.session_factory() as session:
            row = await session.scalar(
                select(TaskDispatchOutboxRow).where(
                    TaskDispatchOutboxRow.run_id == UUID(run_id)
                )
            )
            return row.status if row is not None else None
    finally:
        await database.close()


def _wait_for_terminal(client: Any, run_id: str, *, timeout_seconds: int) -> dict[str, Any]:
    terminal_statuses = {"failed", "cancelled", "completed", "completed_with_limitations"}
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/research-runs/{run_id}")
        assert response.status_code == 200, response.text
        last = response.json()
        if last["status"] in terminal_statuses:
            return last
        time.sleep(0.1)
    raise AssertionError(f"run did not reach terminal state within {timeout_seconds}s: {last}")


def _write_smoke_artifact(payload: dict[str, object]) -> None:
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)
    path = artifact_dir / "test_infrastructure_result.json"
    previous: dict[str, object] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
        except json.JSONDecodeError:
            previous = {}
    kind = str(payload.get("kind", "unknown"))
    previous[kind] = payload
    path.write_text(json.dumps(previous, indent=2, sort_keys=True) + "\n", encoding="utf-8")
