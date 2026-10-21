# Phase 17.0 Cross-Benchmark Results

- Benchmark families with data: **4**
- Mean: `0.606567`
- Median: `0.683733`
- Min: `0.25`
- Max: `0.8088`
- Status: **QUALIFIED_DATASET**

Terminal results use the frozen reference configuration; infrastructure-invalid attempts are excluded and retained in provenance.

## Per-benchmark terminal results

| Benchmark | Family | Runs | Completed | Failed | Coverage mean | Coverage SD |
|---|---|---:|---:|---:|---:|---:|
| `v1-dev-model-comparison` | technical_model_comparison | 3 | 3 | 0 | 0.714267 | 0.101022 |
| `v1-gen-ai-agent-competition` | industry_competition | 1 | 1 | 0 | 0.8088 | 0.0 |
| `v1-gen-low-altitude-economy` | industry_trend | 3 | 2 | 1 | 0.25 | 0.178228 |
| `v1-gen-multimodal-manufacturing` | technical_trend | 3 | 3 | 0 | 0.6532 | 0.146597 |

## Failure modes

{"entailment_failures": 77, "failed_runs": 1, "provider_attempt_failures": 157, "source_role_mismatch": 94, "source_space_exhausted": 43}
