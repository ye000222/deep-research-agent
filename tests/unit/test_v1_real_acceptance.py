import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from scripts.verify_v1_real_runs import AcceptanceRun, evaluate_real_runs

ROOT = Path(__file__).resolve().parents[2]


def _passing_run(run_id: str) -> AcceptanceRun:
    return AcceptanceRun(
        run_id=run_id,
        owner_hash="owner",
        normalized_goal="goal",
        status="completed",
        phase="terminal",
        termination_reason="quality_met",
        saved_profile_id="profile",
        credential_version_id="credential",
        model="model",
        budget_tier="standard",
        source_revision="abc123",
        scoring_rule_version="v1",
        prompt_bundle_version="v1",
        graph_schema_revision="v1",
        coverage=0.90,
        priority_one_coverage=0.85,
        cross_validation=0.75,
        critical_gaps=0,
        report_count=1,
        verified_report_count=1,
        writing_started_count=1,
        report_verified_count=1,
        run_completed_count=1,
        run_failed_count=0,
    )


def test_three_consecutive_real_runs_pass_only_when_every_gate_passes() -> None:
    result = evaluate_real_runs([_passing_run("r3"), _passing_run("r2"), _passing_run("r1")])

    assert result["passed"] is True
    assert result["failures"] == []


def test_limited_report_cannot_satisfy_real_closeout() -> None:
    limited = replace(
        _passing_run("r3"),
        status="completed_with_limitations",
        termination_reason="completed_with_limitations",
        priority_one_coverage=0.0,
    )

    result = evaluate_real_runs([limited, _passing_run("r2"), _passing_run("r1")])

    assert result["passed"] is False
    assert any("status_completed" in failure for failure in result["failures"])
    assert any("priority_one_coverage" in failure for failure in result["failures"])


def test_configuration_mismatch_and_unidentified_revision_fail() -> None:
    first = replace(_passing_run("r3"), source_revision="development")
    second = replace(_passing_run("r2"), source_revision="development", model="other")
    third = replace(_passing_run("r1"), source_revision="development")

    result = evaluate_real_runs([first, second, third])

    assert result["passed"] is False
    assert "run_configuration_fingerprint_mismatch" in result["failures"]
    assert "source_revision_not_release_identifiable" in result["failures"]


def test_fewer_than_three_runs_fail() -> None:
    result = evaluate_real_runs([_passing_run("r1")])

    assert result["passed"] is False
    assert "expected_3_consecutive_runs_found_1" in result["failures"]


def test_closeout_cli_rejects_skipped_checks() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "release_gate.py"),
            "--v1-closeout",
            "--skip-static",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "does not allow any --skip-* option" in result.stderr


def test_closeout_cli_requires_identifiable_source_revision() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "release_gate.py"),
            "--v1-closeout",
            "--v1-owner-hash",
            "owner",
            "--v1-source-revision",
            "development",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "must identify the candidate build" in result.stderr
