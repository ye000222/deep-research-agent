from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from app.domain.document_evidence_context import _identity_hash, _sha256
from app.domain.evidence_graph import claim_fingerprint
from app.infrastructure.db.evidence_graph_models import (
    ResearchClaimRow,
    ResearchSourceSnapshotRow,
)
from app.infrastructure.db.research_models import ResearchSourceRow
from app.infrastructure.db.research_tools import PagePersistenceResult
from app.services.durable_semantic_input import (
    DurableSemanticInputError,
    DurableSemanticInputLoader,
)


class _ScalarRows:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def all(self) -> list[Any]:
        return self._rows


class _Session:
    def __init__(self, dataset: dict[str, Any]) -> None:
        self.dataset = dataset
        self.query_index = 0

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def scalars(self, _statement: object) -> _ScalarRows:
        self.query_index += 1
        if self.query_index == 1:
            return _ScalarRows(self.dataset["events"])
        return _ScalarRows(self.dataset["evidence"])

    async def get(self, model: type[Any], identity: UUID) -> Any | None:
        key = {
            ResearchClaimRow: "claims",
            ResearchSourceRow: "sources",
            ResearchSourceSnapshotRow: "snapshots",
        }.get(model)
        if key is None:
            return None
        return self.dataset[key].get(identity)


class _SessionFactory:
    def __init__(self, dataset: dict[str, Any]) -> None:
        self.dataset = dataset

    def __call__(self) -> _Session:
        return _Session(self.dataset)


class _Artifacts:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    async def read_text(self, uri: str) -> str:
        if uri not in self.values:
            raise FileNotFoundError(uri)
        return self.values[uri]


def _dataset(
    *, defect: str | None = None, text: str = "The reported value is 42 units."
) -> tuple[dict[str, Any], dict[str, str], UUID, str]:
    run_id = uuid4()
    page_ref = "candidate-page-1"
    candidate_ref = "candidate-1"
    claim_id, evidence_id, source_id, snapshot_id = (uuid4() for _ in range(4))
    parent_source_id = uuid4()
    artifact_uri = "runs/test/source.txt"
    content_hash = hashlib.sha256(text.encode()).hexdigest()
    claim_text = text
    claim_row = SimpleNamespace(
        id=claim_id,
        run_id=run_id,
        question_id="q1",
        dimension_key="q1:d1",
        atomic_claim=claim_text,
        claim_hash=claim_fingerprint(claim_text),
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    source_row = SimpleNamespace(
        id=source_id,
        run_id=run_id,
        source_owner_key="example.org",
        original_source_id=(
            uuid4()
            if defect == "provenance"
            else parent_source_id if defect == "provenance_valid" else None
        ),
        content_hash=content_hash,
    )
    snapshot_row = SimpleNamespace(
        id=snapshot_id,
        run_id=run_id,
        source_id=source_id,
        content_hash=("0" * 64 if defect == "snapshot_hash" else content_hash),
        artifact_uri=artifact_uri,
    )
    quote = text if "12.4%" in text else "42 units"
    start = text.index(quote)
    end = start + len(quote)
    locator = {
        "source_locator_type": "DOCUMENT_BLOCK",
        "source_page_start": None,
        "source_page_end": None,
        "source_block_ordinal": 0,
    }
    snapshot_ref = _sha256(content_hash)
    document_ref = _identity_hash(
        "document", {"source_ref": str(source_id), "snapshot_ref": snapshot_ref}
    )
    exact_text = "wrong quote" if defect == "quote" else text[start:end]
    span_ref = _identity_hash(
        "canonical-evidence-span",
        {
            "document_ref": document_ref,
            "start": start,
            "end": end,
            "content_sha256": hashlib.sha256(exact_text.encode()).hexdigest(),
        },
    )
    if defect == "span_offset":
        start += 1
    localization = SimpleNamespace(
        global_id=1,
        event_type="evidence.formation.diagnostic",
        refs={
            "page_ref": page_ref,
            "candidate_ref": candidate_ref,
            "source_ref": str(source_id),
            "snapshot_ref": str(snapshot_id),
            "stage": "QUOTE_LOCALIZATION",
            "outcome": "SUCCEEDED",
            "diagnostic_version": "evidence-formation-diagnostic.v1",
        },
        metrics={
            "facts": {
                "localized": True,
                "char_start": start,
                "char_end": end,
                "span_ref": span_ref,
                **locator,
            }
        },
    )
    persistence = SimpleNamespace(
        global_id=2,
        event_type="evidence.formation.diagnostic",
        refs={
            "page_ref": page_ref,
            "candidate_ref": candidate_ref,
            "source_ref": str(source_id),
            "snapshot_ref": str(snapshot_id),
            "evidence_ref": str(evidence_id),
            "stage": "PERSISTENCE",
            "outcome": "SUCCEEDED",
            "diagnostic_version": "evidence-formation-diagnostic.v1",
        },
        metrics={"facts": {"row_accepted": True, "accepted_and_located": True}},
    )
    evidence_row = SimpleNamespace(
        id=evidence_id,
        run_id=run_id,
        accepted=True,
        claim_id=claim_id,
        claim=claim_text,
        snapshot_id=snapshot_id,
        source_id=source_id,
        chunk_id=None,
        question_id="q1",
        exact_quote=quote,
    )
    dataset: dict[str, Any] = {
        "events": [localization, persistence] if defect != "event" else [persistence],
        "evidence": [evidence_row] if defect != "evidence" else [],
        "claims": {} if defect == "claim" else {claim_id: claim_row},
        "sources": (
            {}
            if defect == "source"
            else {
                source_id: source_row,
                **(
                    {
                        parent_source_id: SimpleNamespace(
                            id=parent_source_id,
                            run_id=run_id,
                            original_source_id=None,
                            source_owner_key="example.org",
                            content_hash=content_hash,
                        )
                    }
                    if defect == "provenance_valid"
                    else {}
                ),
            }
        ),
        "snapshots": {} if defect == "snapshot" else {snapshot_id: snapshot_row},
    }
    if defect == "evidence_row_claim_binding":
        evidence_row.claim_id = uuid4()
    if defect == "evidence_text_binding":
        evidence_row.claim = "different claim text"
    if defect == "event_binding":
        persistence.refs["source_ref"] = str(uuid4())
    artifacts = {} if defect == "artifact" else {artifact_uri: text}
    if defect == "artifact_content_hash":
        artifacts[artifact_uri] = "different document text"
    return dataset, artifacts, run_id, page_ref


def test_loader_reconstructs_semantic_input_across_instances_deterministically() -> None:
    dataset, artifact_data, run_id, page_ref = _dataset()
    expected_page_result = PagePersistenceResult(
        inserted_count=1,
        accepted_count=1,
        claim_receipts=(),  # Assert semantic fields below, not Python object identity.
    )
    loader_a = DurableSemanticInputLoader(_SessionFactory(dataset), _Artifacts(artifact_data))  # type: ignore[arg-type]
    result_a = _run_immediate(loader_a.load_page(run_id, page_ref))
    del loader_a
    loader_b = DurableSemanticInputLoader(_SessionFactory(dataset), _Artifacts(artifact_data))  # type: ignore[arg-type]
    result_b = _run_immediate(loader_b.load_page(run_id, page_ref))

    assert result_a.inserted_count == expected_page_result.inserted_count
    assert result_a.accepted_count == expected_page_result.accepted_count
    assert result_a.receipt_status == "AVAILABLE"
    assert len(result_a.claim_receipts) == 1
    assert result_a == result_b
    receipt = result_a.claim_receipts[0]
    assert receipt.claim_text == "The reported value is 42 units."
    assert receipt.claim_content_hash == claim_fingerprint(receipt.claim_text)
    assert receipt.source_owner_key == "example.org"
    assert receipt.evidence_receipts[0].question_id == "q1"
    assert receipt.evidence_receipts[0].dimension_key == "q1:d1"
    assert receipt.evidence_receipts[0].accepted_and_persisted is True
    assert receipt.evidence_ids == (result_a.claim_receipts[0].evidence_ids[0],)
    assert receipt.evidence_receipts[0].canonical_span is not None
    assert receipt.evidence_receipts[0].canonical_span.exact_text == "42 units"
    assert receipt.source_provenance_refs[-2] == f"source_terminal:{receipt.source_ids[0]}"


def test_loader_reconstructs_existing_source_provenance_chain() -> None:
    dataset, artifact_data, run_id, page_ref = _dataset(defect="provenance_valid")
    loader = DurableSemanticInputLoader(_SessionFactory(dataset), _Artifacts(artifact_data))  # type: ignore[arg-type]
    result = _run_immediate(loader.load_page(run_id, page_ref))
    receipt = result.claim_receipts[0]
    terminal_id = next(
        identity for identity in dataset["sources"] if identity not in receipt.source_ids
    )
    assert f"source_terminal:{terminal_id}" in receipt.source_provenance_refs
    expected_diversity_key = hashlib.sha256(str(terminal_id).encode("ascii")).hexdigest()
    assert receipt.source_diversity_key == expected_diversity_key


@pytest.mark.parametrize(
    "defect",
    [
        "event",
        "claim",
        "evidence",
        "source",
        "snapshot",
        "provenance",
        "artifact",
        "snapshot_hash",
        "span_offset",
        "quote",
        "artifact_content_hash",
        "evidence_row_claim_binding",
        "evidence_text_binding",
        "event_binding",
    ],
)
def test_loader_fails_closed_on_incomplete_or_mismatched_durable_input(defect: str) -> None:
    dataset, artifact_data, run_id, page_ref = _dataset(defect=defect)
    loader = DurableSemanticInputLoader(_SessionFactory(dataset), _Artifacts(artifact_data))  # type: ignore[arg-type]
    with pytest.raises(DurableSemanticInputError):
        _run_immediate(loader.load_page(run_id, page_ref))


def test_loader_rejects_evidence_refs_from_another_page() -> None:
    dataset, artifact_data, run_id, page_ref = _dataset()
    loader = DurableSemanticInputLoader(_SessionFactory(dataset), _Artifacts(artifact_data))  # type: ignore[arg-type]
    with pytest.raises(DurableSemanticInputError, match="EVIDENCE_PAGE_BINDING_MISMATCH"):
        _run_immediate(loader.load_page(run_id, page_ref, (uuid4(),)))


def _run_immediate(coroutine: Any) -> Any:
    """Run fake-only async code without initializing the broken host event loop."""
    try:
        yielded = coroutine.send(None)
    except StopIteration as completed:
        return completed.value
    coroutine.close()
    raise AssertionError(f"fake async path unexpectedly suspended: {yielded!r}")
