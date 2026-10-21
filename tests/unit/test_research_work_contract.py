from uuid import UUID

import pytest
from app.domain.research_work import (
    LegacyResearchObjective,
    WorkAttempt,
    WorkAttemptState,
    WorkResumePoint,
    deterministic_attempt_ref,
    legacy_objective_hash,
    legacy_work_identity,
    resolve_work_resume_point,
)


def _objective(*, question_id: str = "q1") -> LegacyResearchObjective:
    return LegacyResearchObjective(
        question_id=question_id,
        question="Find primary evidence for the research question",
        rationale="The plan identifies a source-backed question.",
        requirements=("primary evidence",),
        priority=1,
        search_hints=("official source",),
        acceptance_context=(("q1:d1", "primary evidence"),),
        plan_version=1,
        bounds={"max_attempts": 3},
    )


def test_legacy_work_identity_is_stable_and_question_scoped() -> None:
    objective = _objective()
    digest = legacy_objective_hash(objective)
    run = UUID("11111111-1111-4111-8111-111111111111")
    first = legacy_work_identity(
        run_ref=run, plan_version=1, question_id="q1", objective_hash=digest
    )
    same = legacy_work_identity(
        run_ref=run, plan_version=1, question_id="q1", objective_hash=digest
    )
    other = legacy_work_identity(
        run_ref=run, plan_version=1, question_id="q2", objective_hash=digest
    )
    assert first == same
    assert first != other
    assert objective.question_id == "q1"


def test_attempt_reference_is_monotonic_identity_and_validated() -> None:
    work = UUID("22222222-2222-4222-8222-222222222222")
    assert deterministic_attempt_ref(work, 1) == deterministic_attempt_ref(work, 1)
    assert deterministic_attempt_ref(work, 1) != deterministic_attempt_ref(work, 2)
    with pytest.raises(ValueError, match="attempt_index_must_be_positive"):
        deterministic_attempt_ref(work, 0)


def test_work_attempt_is_immutable_and_does_not_encode_semantic_truth() -> None:
    attempt = WorkAttempt(
        attempt_ref=deterministic_attempt_ref(
            UUID("22222222-2222-4222-8222-222222222222"), 1
        ),
        work_ref=UUID("22222222-2222-4222-8222-222222222222"),
        attempt_index=1,
        state=WorkAttemptState.RESERVED,
        reservation_identity="work:attempt:1",
        source_run_state_ref="runs/example/run-state/hash.json",
        source_run_state_version=1,
    )
    assert "verified" not in attempt.model_dump(mode="json")
    with pytest.raises(ValueError):
        attempt.attempt_index = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("state", "refs", "metadata", "expected"),
    [
        (
            WorkAttemptState.RESERVED,
            {},
            {},
            WorkResumePoint.TARGET_NOT_STARTED,
        ),
        (
            WorkAttemptState.EXECUTING,
            {"query_refs": (UUID("33333333-3333-4333-8333-333333333333"),)},
            {},
            WorkResumePoint.QUERY_DURABLE,
        ),
        (
            WorkAttemptState.EXECUTING,
            {},
            {"search_result_durable": True},
            WorkResumePoint.SEARCH_RESULT_DURABLE,
        ),
        (
            WorkAttemptState.EXECUTING,
            {},
            {"reader_artifact_ref": "runs/r/sources/s.txt"},
            WorkResumePoint.READER_RESULT_DURABLE,
        ),
        (
            WorkAttemptState.EXECUTING,
            {"produced_evidence_refs": (UUID("44444444-4444-4444-8444-444444444444"),)},
            {},
            WorkResumePoint.EVIDENCE_DURABLE,
        ),
        (
            WorkAttemptState.COMPLETED,
            {},
            {},
            WorkResumePoint.ATTEMPT_ALREADY_COMPLETE,
        ),
    ],
)
def test_resume_point_is_derived_from_durable_attempt_fields(
    state: WorkAttemptState,
    refs: dict[str, tuple[UUID, ...]],
    metadata: dict[str, object],
    expected: WorkResumePoint,
) -> None:
    work_ref = UUID("22222222-2222-4222-8222-222222222222")
    attempt = WorkAttempt(
        attempt_ref=deterministic_attempt_ref(work_ref, 1),
        work_ref=work_ref,
        attempt_index=1,
        state=state,
        reservation_identity="work:attempt:1",
        source_run_state_ref="runs/example/run-state/hash.json",
        source_run_state_version=1,
        resume_metadata=metadata,
        **refs,
    )
    assert resolve_work_resume_point(attempt) is expected
