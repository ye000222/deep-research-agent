import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from evals.graders.evaluation_grader import grade_dataset


def test_v1_golden_dataset_termination_decisions():
    cases = json.loads(Path("evals/datasets/v1_golden.json").read_text(encoding="utf-8"))
    result = grade_dataset(cases)
    assert result["accuracy"] == 1.0
