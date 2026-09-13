from app.infrastructure.db.reports import _verification_budget_ledger_entry


def test_deterministic_verification_releases_tokens_without_fake_consumption() -> None:
    released, entry = _verification_budget_ledger_entry(
        allocated_tokens=4_500,
        verification_calls=1,
        report_version=2,
        budget_exhausted=False,
    )

    assert released == 4_500
    assert entry == {
        "node": "verification",
        "phase": "writing",
        "allocated_tokens": 4_500,
        "actual_tokens": 0,
        "status": "deterministic_no_model",
        "verification_calls": 1,
        "report_version": 2,
    }


def test_exhausted_verification_budget_still_does_not_consume_model_tokens() -> None:
    released, entry = _verification_budget_ledger_entry(
        allocated_tokens=3_000,
        verification_calls=2,
        report_version=3,
        budget_exhausted=True,
    )

    assert released == 3_000
    assert entry["actual_tokens"] == 0
    assert entry["status"] == "budget_exhausted"
