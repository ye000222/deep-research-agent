from uuid import uuid4

from app.domain.reports import ReportCitationView
from app.infrastructure.db.analysis_models import AnalysisArtifactClaimRow


def test_analysis_artifact_claim_edge_defaults_to_derived_from():
    row = AnalysisArtifactClaimRow(
        analysis_artifact_id=uuid4(),
        claim_id=uuid4(),
        relation="derived_from",
        confidence=1.0,
        created_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
    )
    assert row.relation == "derived_from"
    assert row.confidence == 1.0


def test_report_citation_view_exposes_optional_analysis_artifact():
    payload = {
        "citation_number": 1,
        "evidence_id": uuid4(),
        "claim_id": uuid4(),
        "snapshot_id": uuid4(),
        "chunk_id": uuid4(),
        "question_id": "q1",
        "claim": "claim",
        "exact_quote": "quote",
        "source_title": "source",
        "source_url": "https://example.com",
        "source_domain": "example.com",
        "source_content_hash": "a" * 64,
        "snapshot_content_hash": "b" * 64,
        "chunk_char_start": 0,
        "chunk_char_end": 5,
        "accessed_at": __import__("datetime").datetime.now(__import__("datetime").UTC),
    }
    assert ReportCitationView.model_validate(payload).analysis_artifact_id is None
