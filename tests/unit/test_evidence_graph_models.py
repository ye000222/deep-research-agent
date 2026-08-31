from app.infrastructure.db import (
    evidence_graph_models,
    research_models,
    run_models,
)
from app.infrastructure.db.evidence_graph_models import (
    ResearchClaimEdgeRow,
    ResearchClaimRow,
    ResearchConflictRow,
    ResearchSourceChunkRow,
    ResearchSourceSnapshotRow,
)
from app.infrastructure.db.report_models import ReportCitationRow
from app.infrastructure.db.research_models import ResearchEvidenceRow, ResearchSourceRow


def test_evidence_graph_tables_are_registered() -> None:
    assert evidence_graph_models
    assert research_models
    assert run_models
    assert ResearchClaimRow.__tablename__ == "research_claims"
    assert ResearchSourceSnapshotRow.__tablename__ == "research_source_snapshots"
    assert ResearchSourceChunkRow.__tablename__ == "research_source_chunks"
    assert ResearchClaimEdgeRow.__tablename__ == "research_claim_edges"
    assert ResearchConflictRow.__tablename__ == "research_conflicts"


def test_legacy_source_and_evidence_have_graph_compatibility_columns() -> None:
    source_columns = ResearchSourceRow.__table__.c
    assert "source_owner_key" in source_columns
    assert "original_source_id" in source_columns

    evidence_columns = ResearchEvidenceRow.__table__.c
    assert "claim_id" in evidence_columns
    assert "snapshot_id" in evidence_columns
    assert "chunk_id" in evidence_columns

    foreign_key_targets = {
        foreign_key.target_fullname
        for column_name in ("claim_id", "snapshot_id", "chunk_id")
        for foreign_key in evidence_columns[column_name].foreign_keys
    }
    assert foreign_key_targets == {
        "research_claims.id",
        "research_source_snapshots.id",
        "research_source_chunks.id",
    }


def test_evidence_graph_enforces_identity_and_provenance_indexes() -> None:
    claim_indexes = {index.name for index in ResearchClaimRow.__table__.indexes}
    snapshot_indexes = {index.name for index in ResearchSourceSnapshotRow.__table__.indexes}
    chunk_indexes = {index.name for index in ResearchSourceChunkRow.__table__.indexes}
    edge_indexes = {index.name for index in ResearchClaimEdgeRow.__table__.indexes}
    conflict_indexes = {index.name for index in ResearchConflictRow.__table__.indexes}

    assert "uq_research_claim_hash" in claim_indexes
    assert "uq_research_source_snapshot_hash" in snapshot_indexes
    assert "uq_research_source_chunk_hash" in chunk_indexes
    assert "uq_research_claim_edge" in edge_indexes
    assert "uq_research_conflict_evidence_pair" in conflict_indexes


def test_report_citation_binds_the_complete_provenance_chain() -> None:
    citation_columns = ReportCitationRow.__table__.c
    assert {"claim_id", "snapshot_id", "chunk_id"}.issubset(citation_columns.keys())
    assert all(
        not citation_columns[column_name].nullable
        for column_name in ("claim_id", "snapshot_id", "chunk_id")
    )

    foreign_key_targets = {
        foreign_key.target_fullname
        for column_name in ("claim_id", "snapshot_id", "chunk_id")
        for foreign_key in citation_columns[column_name].foreign_keys
    }
    assert foreign_key_targets == {
        "research_claims.id",
        "research_source_snapshots.id",
        "research_source_chunks.id",
    }
