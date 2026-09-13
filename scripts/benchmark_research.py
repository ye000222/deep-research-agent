"""Compare exported, paired run measurements without making paid Provider calls."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.graders.performance_grader import PerformanceSample, compare_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    baseline = [PerformanceSample(**row) for row in json.loads(args.baseline.read_text("utf-8"))]
    candidate = [PerformanceSample(**row) for row in json.loads(args.candidate.read_text("utf-8"))]
    print(json.dumps(compare_samples(baseline, candidate), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
