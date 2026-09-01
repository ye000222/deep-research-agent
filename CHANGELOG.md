# Changelog

## [Unreleased] — V1 strict release candidate

- Added PostgreSQL 18 integration CI with independent LangGraph checkpoint database.
- Added Knowledge, Gap, Action and Evaluation read APIs for server-backed explainability.
- Added deterministic semantic citation support verification and expanded Golden Eval coverage.
- Added Memory projection joins, bounded dynamic planning, Provider Capability visibility and Outbox failure probes.
- Added SSRF/DNS rebinding and cross-host redirect security regression tests.
- Added strict `scripts/release_gate.py` with static checks, full tests, integration smoke, web build, Golden Eval and high-confidence secret scan.
- Added GitHub release job that uploads `artifacts/release_gate.json`.
