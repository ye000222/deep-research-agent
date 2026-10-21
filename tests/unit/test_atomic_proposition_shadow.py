from __future__ import annotations

from pathlib import Path

from app.services.atomic_proposition_decomposer import AtomicPropositionDecomposer
from app.services.atomic_proposition_shadow import AtomicPropositionShadow


def test_shadow_facade_has_no_persistence_or_production_import_wiring() -> None:
    source = Path("apps/api/app/services/atomic_proposition_shadow.py").read_text(encoding="utf-8")
    assert "repository" not in source.lower()
    assert "insert" not in source.lower()
    assert "update" not in source.lower()
    assert "delete" not in source.lower()

    production_paths = (
        "apps/api/app/services/research_loop.py",
        "apps/api/app/services/evidence_extractor.py",
        "apps/api/app/services/report_writer.py",
        "apps/api/app/domain/gap_requirement.py",
    )
    for path in production_paths:
        candidate = Path(path)
        if candidate.exists():
            text = candidate.read_text(encoding="utf-8").lower()
            assert "atomic_proposition_shadow" not in text
            assert "atomic_proposition_decomposer" not in text

    # The facade is dependency-injected and has no persistence capability in its type surface.
    assert "repository" not in str(AtomicPropositionShadow.__init__.__annotations__).lower()
    assert AtomicPropositionDecomposer.__module__.endswith("atomic_proposition_decomposer")
