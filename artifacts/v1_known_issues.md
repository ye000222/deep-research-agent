# V1 Known Issues Registry

| Issue | Type | Severity | Release blocker | User impact | Target |
|---|---|---|---|---|---|
| Low-altitude economy weakness (coverage 0.25 ± 0.1782) | LIMITATION | P2 | No | Some industry-trend runs may end with limited/no evidence | Post-V1 |
| Run-to-run variance | EXPECTED_VARIANCE | P2 | No | Quality and runtime vary with provider/search availability | Post-V1 |
| Per-claim verified claims = 0 | ARCHITECTURAL_V2 | V2 | No | Report verification remains evidence/report-level, not full claim-level | V2 |
| q1 requirement projection inconsistency | KNOWN_CORRECTNESS_ISSUE | P2 | No, no proven user-visible release impact | Internal state may show stale requirement counts in affected historical views | Post-V1 |
| Historical Phase 15.2 analysis lint | RELEASE_HYGIENE | P1 | No, excluded from maintained release lint scope | One-off historical analyzer is not part of RC tooling | Historical tooling |
| Legacy qualification runs lack explicit reference_config_id | RELEASE_PROVENANCE | P1 | No, flags and benchmark identity remain auditable | New launchers stamp the ID; historical runs are retained unchanged | V1 hardening |
| `independent_source_targeting` | ACTIVE_V1 | — | No | Frozen V1 behavior under `v1-pre-rc-reference-1` | V1 |
| `independent_source_targeting` | ACTIVE_V1 | — | No | Frozen V1 behavior under `v1-pre-rc-reference-1` | V1 |
| `evidence_aware_context` | EXPERIMENTAL_DISABLED | — | No | Kept available for controlled future experiments; disabled in V1 reference config | V2 evaluation |
| `evidence_input_quality` | EXPERIMENTAL_DISABLED | — | No | Kept available for controlled future experiments; disabled in V1 reference config | V2 evaluation |

Phase 17.1 does not change Research Intelligence to address these items.
