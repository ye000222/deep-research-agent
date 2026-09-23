"""Unit tests for the Phase 13.0 Gap Closure Bottleneck classifier.

The analyzer lives in ``scripts/`` (a read-only ops tool that is intentionally
not part of the runtime package), so it is loaded here by file path. These tests
cover only the pure classification decision table and need no database.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_gap_closure_bottleneck.py"
MODULE_NAME = "analyze_gap_closure_bottleneck"


@pytest.fixture(scope="module")
def analyzer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module uses ``from __future__ import annotations``
    # plus ``@dataclass``, whose annotation resolution needs sys.modules[__module__].
    sys.modules[MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.modules[MODULE_NAME]
    return module


def _signals(analyzer: ModuleType, **overrides: object) -> object:
    defaults: dict[str, object] = {
        "question_id": "q1",
        "dimension_key": "q1:d1",
        "priority": 1,
        "requirement_type": "claim_verification",
        "failure_reason": "claim_not_verified",
        "closure_result": "partial",
        "required_evidence_count": 1,
        "current_evidence_count": 1,
        "required_independent_sources": 0,
        "current_independent_sources": 0,
        "current_coverage": 1.0,
        "required_coverage": 1.0,
        "alignment_status": "aligned",
    }
    defaults.update(overrides)
    return analyzer.GapSignals(**defaults)


INDEP = {"requirement_type": "independent_source", "failure_reason": "unknown"}
MISS_SRC_ON_CLAIM = {
    "requirement_type": "claim_verification",
    "failure_reason": "missing_independent_source",
}
NO_VALID_PATH = {"requirement_type": "evidence_quality", "failure_reason": "no_valid_path"}
NOT_ALIGNED_UNKNOWN = {
    "requirement_type": "claim_verification",
    "failure_reason": "unknown",
    "alignment_status": "not_aligned",
}
CLAIM_NOT_VERIFIED = {
    "requirement_type": "claim_verification",
    "failure_reason": "claim_not_verified",
}
CLAIM_UNKNOWN = {"requirement_type": "claim_verification", "failure_reason": "unknown"}
EQ_LOW = {
    "requirement_type": "evidence_quality",
    "failure_reason": "insufficient_evidence",
    "required_evidence_count": 1,
    "current_evidence_count": 1,
    "current_coverage": 0.4,
    "required_coverage": 1.0,
}
EQ_NOT_ENOUGH = {
    "requirement_type": "evidence_quality",
    "failure_reason": "insufficient_evidence",
    "required_evidence_count": 3,
    "current_evidence_count": 1,
}
CLAIM_INSUFFICIENT = {
    "requirement_type": "claim_verification",
    "failure_reason": "insufficient_evidence",
}
RESIDUAL_SHORT = {
    "requirement_type": "unknown",
    "failure_reason": "unknown",
    "required_evidence_count": 2,
    "current_evidence_count": 1,
}


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (INDEP, "MISSING_INDEPENDENT_SOURCE"),
        (MISS_SRC_ON_CLAIM, "MISSING_INDEPENDENT_SOURCE"),
        (NO_VALID_PATH, "DIMENSION_MISMATCH"),
        (NOT_ALIGNED_UNKNOWN, "DIMENSION_MISMATCH"),
        (CLAIM_NOT_VERIFIED, "CLAIM_NOT_VERIFIED"),
        (CLAIM_UNKNOWN, "CLAIM_NOT_VERIFIED"),
        (EQ_LOW, "EVIDENCE_QUALITY_LOW"),
        (EQ_NOT_ENOUGH, "INSUFFICIENT_EVIDENCE"),
        (CLAIM_INSUFFICIENT, "INSUFFICIENT_EVIDENCE"),
        (RESIDUAL_SHORT, "INSUFFICIENT_EVIDENCE"),
    ],
)
def test_classify_gap_decision_table(
    analyzer: ModuleType, overrides: dict[str, object], expected: str
) -> None:
    signals = _signals(analyzer, **overrides)
    assert analyzer.classify_gap(signals) == expected


def test_every_bucket_is_reachable(analyzer: ModuleType) -> None:
    probes = (INDEP, CLAIM_NOT_VERIFIED, CLAIM_INSUFFICIENT, NO_VALID_PATH, EQ_LOW)
    buckets = {analyzer.classify_gap(_signals(analyzer, **probe)) for probe in probes}
    assert buckets == {
        "MISSING_INDEPENDENT_SOURCE",
        "CLAIM_NOT_VERIFIED",
        "INSUFFICIENT_EVIDENCE",
        "DIMENSION_MISMATCH",
        "EVIDENCE_QUALITY_LOW",
    }
