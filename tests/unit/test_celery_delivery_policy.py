from app.llm.adapters import ModelGatewayError
from app.worker.celery_app import celery_app
from app.worker.tasks import _should_defer_model_error


def test_celery_worker_loss_policy_requires_idempotent_redelivery() -> None:
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True
    assert celery_app.conf.worker_prefetch_multiplier == 1
    assert celery_app.conf.task_ignore_result is True


def test_transient_model_failures_use_durable_redelivery() -> None:
    for code in (
        "MODEL_NETWORK_ERROR",
        "MODEL_PROVIDER_UNAVAILABLE",
        "MODEL_RATE_LIMITED",
        "MODEL_TIMEOUT",
    ):
        assert _should_defer_model_error(
            ModelGatewayError(code, retryable=True, detail_code="CONNECT_ERROR")
        )


def test_non_transient_model_failures_never_loop_through_outbox() -> None:
    assert not _should_defer_model_error(
        ModelGatewayError("MODEL_AUTHENTICATION_FAILED", retryable=False)
    )
    assert not _should_defer_model_error(
        ModelGatewayError("MODEL_NETWORK_ERROR", retryable=False)
    )
