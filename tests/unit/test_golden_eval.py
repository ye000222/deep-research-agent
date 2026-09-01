import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from evals.graders.evaluation_grader import grade_dataset


def test_v1_golden_dataset_termination_decisions():
    cases = json.loads(Path("evals/datasets/v1_golden.json").read_text(encoding="utf-8"))
    result = grade_dataset(cases)
    assert result["accuracy"] == 1.0
    assert result["stop_decision_accuracy"] >= 0.90
    assert result["gap_recall"] >= 0.85
    assert result["evidence_support_precision"] >= 0.95
    assert result["early_stop_error_rate"] <= 0.05
    assert result["release_gate_passed"] is True
