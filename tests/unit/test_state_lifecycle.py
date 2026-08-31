from uuid import uuid4

from app.agent.reducers.state_reducer import apply_patch
from app.domain.state import ResearchState, RunStatus, StatePatch, StopReason


def test_resume_patch_clears_terminal_stop_reason() -> None:
    failed = ResearchState(
        run_id=uuid4(),
        status=RunStatus.FAILED,
        stop_reason=StopReason.FATAL_ERROR,
    )

    resumed = apply_patch(
        failed,
        StatePatch(
            patch_id=uuid4(),
            base_version=failed.state_version,
            status=RunStatus.QUEUED,
            clear_stop_reason=True,
        ),
    )

    assert resumed.status is RunStatus.QUEUED
    assert resumed.stop_reason is None
    assert resumed.state_version == failed.state_version + 1
