"""Paired performance summaries. Failure runs are retained in every cost metric."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class PerformanceSample:
    case_id: str
    repetition: int
    config_hash: str
    elapsed_ms: float
    total_tokens: int
    coverage: float
    support_precision: float
    technical_success: bool
    quality_success: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.case_id, str)
            or not isinstance(self.config_hash, str)
            or type(self.repetition) is not int
            or type(self.total_tokens) is not int
            or type(self.technical_success) is not bool
            or type(self.quality_success) is not bool
            or any(type(x) not in (int, float) for x in (
                self.elapsed_ms, self.coverage, self.support_precision
            ))
        ):
            raise ValueError("invalid measurement types")
        if self.quality_success and not self.technical_success:
            raise ValueError("quality success requires technical success")
        if not self.case_id.strip() or not self.config_hash.strip() or self.repetition < 0:
            raise ValueError("missing pairing identity")
        if not math.isfinite(self.elapsed_ms) or self.elapsed_ms < 0 or self.total_tokens < 0:
            raise ValueError("invalid cost")
        if any(
            not math.isfinite(x) or not 0 <= x <= 1
            for x in (
                self.coverage,
                self.support_precision,
            )
        ):
            raise ValueError("invalid quality metric")


def compare_samples(
    baseline: list[PerformanceSample],
    candidate: list[PerformanceSample],
) -> dict[str, object]:
    def index(samples: list[PerformanceSample]) -> dict[tuple[str, int], PerformanceSample]:
        result = {(s.case_id, s.repetition): s for s in samples}
        if len(result) != len(samples) or not result:
            raise ValueError("empty or duplicate samples")
        return result

    left, right = index(baseline), index(candidate)
    if left.keys() != right.keys():
        raise ValueError("unpaired samples; missing failures must not be dropped")
    if any(left[k].config_hash != right[k].config_hash for k in left):
        raise ValueError("provider/model/budget configuration differs")
    regressions = [
        {"case_id": k[0], "repetition": k[1]}
        for k in sorted(left)
        if right[k].support_precision < left[k].support_precision
        or right[k].technical_success < left[k].technical_success
        or right[k].quality_success < left[k].quality_success
    ]

    def costs(samples: list[PerformanceSample]) -> dict[str, float]:
        times = sorted(s.elapsed_ms for s in samples)
        return {
            "median_elapsed_ms": median(times),
            "p95_elapsed_ms": times[math.ceil(len(times) * 0.95) - 1],
            "median_total_tokens": median(s.total_tokens for s in samples),
            "median_coverage": median(s.coverage for s in samples),
        }

    # Repeats share a research question and are not independent observations.
    # Average paired deltas within each question, then resample whole questions.
    metrics = ("support_precision", "technical_success", "quality_success", "coverage")
    case_ids = sorted({k[0] for k in left})
    intervals: dict[str, object] = {}
    for metric in metrics:
        deltas = []
        for case_id in case_ids:
            keys = [k for k in left if k[0] == case_id]
            deltas.append(sum(
                float(getattr(right[k], metric)) - float(getattr(left[k], metric))
                for k in keys
            ) / len(keys))
        rng = random.Random(20260910)
        bootstrap = sorted(
            sum(rng.choices(deltas, k=len(deltas))) / len(deltas)
            for _ in range(2000)
        )
        intervals[metric] = {
            "mean_paired_delta": sum(deltas) / len(deltas),
            "lower_95": bootstrap[49] if len(deltas) >= 2 else None,
            "upper_95": bootstrap[1949] if len(deltas) >= 2 else None,
        }

    return {
        "sample_count": len(left),
        "case_count": len({k[0] for k in left}),
        "baseline": costs(baseline),
        "candidate": costs(candidate),
        "quality_regressions": regressions,
        "observed_quality_nonregression": not regressions,
        "clustered_intervals": intervals,
        "interval_method": (
            "paired case-cluster percentile bootstrap; 2000 resamples; seed 20260910"
        ),
        "minimum_case_count_met": len(case_ids) >= 30,
        "release_approved": False,
        "limitation": (
            "Empirical intervals cannot establish unseen failure risk; requires "
            "representative live pairs, absolute quality gates and human review."
        ),
    }
