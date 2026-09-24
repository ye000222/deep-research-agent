# V2.0 Claim Baseline

- Metric definition: `claim-v2-baseline-v1`
- Source: `read-only PostgreSQL query over the Phase 17 persisted qualification run IDs`
- Historical qualification runs inventoried: **10**
- Coverage definition: `coverage-v1` unchanged; not re-scored here.

## Run-level Claim funnel

| Family | Run | Cand. | Accepted | Claims | Supported | Multi-evidence | Multi-owner | Verified | Conflict* | Data |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| technical_model_comparison | `01a0c3ef-928b-7f0d-8006-40843c571b57` | 55 | 23 | 55 | 23 | 0 | 0 | NOT_RECONSTRUCTABLE | 0 | RECONSTRUCTED_FROM_READ_ONLY_DB |
| technical_model_comparison | `01a0c412-10b0-7c3f-8cf4-d56aeecbbea0` | 62 | 37 | 62 | 37 | 0 | 0 | NOT_RECONSTRUCTABLE | 0 | RECONSTRUCTED_FROM_READ_ONLY_DB |
| technical_model_comparison | `01a0c412-5ae6-7b9c-9353-7be1ad74c293` | 47 | 33 | 47 | 33 | 0 | 0 | NOT_RECONSTRUCTABLE | 0 | RECONSTRUCTED_FROM_READ_ONLY_DB |
| technical_trend | `01a0c826-baca-7d7e-9d4d-ab2f7f16a7b7` | 80 | 43 | 80 | 41 | 0 | 0 | NOT_RECONSTRUCTABLE | 0 | RECONSTRUCTED_FROM_READ_ONLY_DB |
| technical_trend | `01a0c861-bc5a-79cc-9eb2-4112322576f3` | 73 | 38 | 73 | 36 | 0 | 0 | NOT_RECONSTRUCTABLE | 0 | RECONSTRUCTED_FROM_READ_ONLY_DB |
| technical_trend | `01a0c861-f646-71d5-942a-fe97bde7ac09` | 42 | 24 | 42 | 24 | 0 | 0 | NOT_RECONSTRUCTABLE | 0 | RECONSTRUCTED_FROM_READ_ONLY_DB |
| industry_competition | `01a0c826-bae4-7e50-bb8e-be90740a8075` | 145 | 79 | 145 | 79 | 0 | 0 | NOT_RECONSTRUCTABLE | 0 | RECONSTRUCTED_FROM_READ_ONLY_DB |
| industry_trend | `01a0c826-baf4-7a02-895f-5e3cf68e9efe` | 85 | 24 | 85 | 24 | 0 | 0 | NOT_RECONSTRUCTABLE | 0 | RECONSTRUCTED_FROM_READ_ONLY_DB |
| industry_trend | `01a0c861-bc75-7492-966e-7f51b0dcb2d8` | 80 | 31 | 80 | 31 | 0 | 0 | NOT_RECONSTRUCTABLE | 0 | RECONSTRUCTED_FROM_READ_ONLY_DB |
| industry_trend | `01a0c89b-f23d-79f5-8786-86580a650e50` | 0 | 0 | 0 | 0 | 0 | 0 | NOT_RECONSTRUCTABLE | 0 | RECONSTRUCTED_FROM_READ_ONLY_DB |

`Conflicted*` is the legacy graph `disputed` status proxy, not a V2 claim-verification verdict.

## Task-family aggregates

| Family | Runs | Candidates | Accepted | Claims | Supported | Multi-evidence | Multi-owner | Verified | Tokens | Runtime s | Searches | Readers | Extraction |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| industry_competition | 1 | 145 | 79 | 145 | 79 | 0 | 0 | NOT_RECONSTRUCTABLE | 138233 | 2457.761 | 41 | 56 | 41 |
| industry_trend | 3 | 165 | 55 | 165 | 55 | 0 | 0 | NOT_RECONSTRUCTABLE | 144776 | 7196.689 | 144 | 115 | 49 |
| technical_model_comparison | 3 | 164 | 93 | 164 | 93 | 0 | 0 | NOT_RECONSTRUCTABLE | 210318 | 8805.48 | 131 | 185 | 69 |
| technical_trend | 3 | 195 | 105 | 195 | 101 | 0 | 0 | NOT_RECONSTRUCTABLE | 183424 | 7331.502 | 110 | 125 | 56 |

Task-family rates and unit-cost ratios are in JSON; LLM calls and per-Claim verification remain NOT_RECONSTRUCTABLE.

## Interpretation and data gaps

- V1.1 has no persisted per-Claim verification result; verified/unknown counts remain NOT_RECONSTRUCTABLE.
- Legacy `disputed` is only a conflict proxy, not a final V2 contradiction verdict.
- Coverage-v1 is unchanged; this artifact reports Claim and efficiency fields only.

No run was created and no research behavior was changed.
