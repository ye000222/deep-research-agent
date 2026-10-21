from __future__ import annotations

from uuid import UUID

from sqlalchemy.dialects import postgresql

from app.infrastructure.db.research_runs import ResearchRunRepository


def test_run_binding_query_requires_active_credential_owned_by_request_owner() -> None:
    owner_hash = "a" * 64
    credential_version_id = UUID("20000000-0000-4000-8000-000000000002")

    statement = ResearchRunRepository._active_credential_binding_query(
        owner_hash, credential_version_id
    )
    compiled = statement.compile(dialect=postgresql.dialect())
    sql = str(compiled)

    assert "llm_credential_versions.profile_id = llm_provider_profiles.id" in sql
    assert "llm_provider_profiles.owner_hash" in sql
    assert "llm_provider_profiles.status" in sql
    assert "llm_provider_profiles.deleted_at IS NULL" in sql
    assert "llm_credential_versions.id" in sql
    assert "llm_credential_versions.revoked_at IS NULL" in sql
    assert "llm_credential_versions.deleted_at IS NULL" in sql
    assert owner_hash in compiled.params.values()
    assert credential_version_id in compiled.params.values()
