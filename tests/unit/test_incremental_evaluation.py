from app.domain.incremental_evaluation import plan_incremental_invalidation


def test_new_evidence_invalidates_question_and_global_but_not_all_report() -> None:
    plan = plan_incremental_invalidation(
        claim_ids={"c1"}, question_ids={"q1"}, evidence_changed=True
    )
    assert plan.scopes == ("evidence", "question", "global")
    assert plan.invalidate_report_draft is True
    assert plan.reason == "evidence_changed"


def test_revocation_cascades_to_sections_and_report() -> None:
    plan = plan_incremental_invalidation(
        claim_ids={"c1"}, question_ids={"q1"}, evidence_revoked=True
    )
    assert plan.scopes == ("evidence", "question", "global", "section", "report")
    assert plan.invalidate_report_draft is True


def test_no_dependency_change_is_noop() -> None:
    plan = plan_incremental_invalidation()
    assert plan.scopes == ()
    assert plan.invalidate_report_draft is False
