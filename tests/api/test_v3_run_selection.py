from __future__ import annotations

import pytest
from app.api.v1.research_runs import ResearchRunCreate


def test_research_run_defaults_to_legacy_and_allows_explicit_v3() -> None:
    base = {
        "query": "topic",
        "saved_profile_version_id": "00000000-0000-0000-0000-000000000001",
    }
    assert ResearchRunCreate.model_validate(base).orchestrator_version == "V2_LEGACY"
    assert (
        ResearchRunCreate.model_validate({**base, "orchestrator_version": "V3"})
        .orchestrator_version
        == "V3"
    )


def test_research_run_rejects_unknown_orchestrator_version() -> None:
    with pytest.raises(ValueError):
        ResearchRunCreate.model_validate(
            {
                "query": "topic",
                "saved_profile_version_id": "00000000-0000-0000-0000-000000000001",
                "orchestrator_version": "V3_NEXT",
            }
        )
