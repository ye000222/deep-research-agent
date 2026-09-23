"""V1 release gate checks for local and CI execution."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_venv_python = ROOT / ".venv" / (
    Path("Scripts") / "python.exe" if sys.platform == "win32" else Path("bin") / "python"
)
PYTHON = str(_venv_python if _venv_python.exists() else Path(sys.executable))
PNPM = "pnpm.cmd" if sys.platform == "win32" else "pnpm"
REQUIRED = (
    "docker-compose.yml",
    "README.md",
    "apps/api/app/retrieval/projections.py",
    "apps/api/app/evaluation/report_verifier.py",
    "evals/datasets/v1_golden.json",
    "scripts/verify_v1_real_runs.py",
)


def migration_head() -> tuple[str, Path] | tuple[None, None]:
    """Return the newest single Alembic revision tracked in the repository.

    Release validation must follow the repository's actual migration chain rather
    than a stale, hard-coded filename.  The timestamped revision naming convention
    is part of the project's migration policy.
    """

    versions = ROOT / "apps" / "api" / "alembic" / "versions"
    candidates: list[tuple[str, Path]] = []
    for path in versions.glob("*.py"):
        match = re.match(r"^(\d{8}_\d{4})_.*\.py$", path.name)
        if match:
            candidates.append((match.group(1), path))
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[0])

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{20,}\b"),
)


def check_files() -> list[str]:
    missing = [item for item in REQUIRED if not (ROOT / item).exists()]
    _, head_path = migration_head()
    if head_path is None:
        missing.append("apps/api/alembic/versions/<timestamped migration head>.py")
    return missing


def check_mysql_references() -> list[str]:
    hits: list[str] = []
    for path in (ROOT / "apps", ROOT / "infra", ROOT / "scripts"):
        for file in path.rglob("*"):
            if (
                file.name == "release_gate.py"
                or not file.is_file()
                or file.suffix not in {".py", ".ps1", ".yml", ".yaml", ".toml"}
            ):
                continue
            text = file.read_text(encoding="utf-8", errors="ignore").lower()
            if any(token in text for token in ("mysql", "pymysql", "aiomysql")):
                hits.append(str(file.relative_to(ROOT)))
    return hits


def check_compose() -> bool:
    result = subprocess.run(
        ["docker", "compose", "config", "--quiet"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _run(name: str, command: list[str], *, cwd: Path = ROOT) -> dict[str, object]:
    """Run one release command and keep output bounded and machine-readable."""

    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=os.environ.copy(),
    )
    output = (completed.stdout + completed.stderr).strip()
    return {
        "name": name,
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "output_tail": output[-2000:],
    }


def check_secret_scan() -> list[str]:
    """Scan tracked text files for high-confidence accidental credential literals."""

    listing = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if listing.returncode != 0:
        return ["git ls-files failed"]
    hits: list[str] = []
    for raw_path in listing.stdout.split(b"\0"):
        if not raw_path:
            continue
        path = ROOT / raw_path.decode("utf-8", errors="ignore")
        text_suffixes = {".py", ".ts", ".tsx", ".js", ".json", ".yml", ".yaml", ".toml", ".env"}
        if path.suffix.lower() not in text_suffixes:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            hits.append(str(path.relative_to(ROOT)))
    return hits


def check_golden_eval() -> dict[str, object]:
    result = subprocess.run(
        [PYTHON, "-c", (
            "import json; from pathlib import Path; "
            "from evals.graders.evaluation_grader import grade_dataset; "
            "cases=json.loads(Path('evals/datasets/v1_golden.json').read_text(encoding='utf-8')); "
            "r=grade_dataset(cases); print(json.dumps(r, ensure_ascii=False)); "
            "raise SystemExit(0 if r['release_gate_passed'] else 1)"
        )],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = (result.stdout + result.stderr).strip()
    return {"passed": result.returncode == 0, "output": output[-4000:]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-compose", action="store_true")
    parser.add_argument("--skip-integration", action="store_true")
    parser.add_argument("--skip-web", action="store_true")
    parser.add_argument("--skip-static", action="store_true")
    parser.add_argument("--v1-closeout", action="store_true")
    parser.add_argument("--v1-owner-hash")
    parser.add_argument("--v1-source-revision")
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
    if args.v1_closeout and any(
        (args.skip_compose, args.skip_integration, args.skip_web, args.skip_static)
    ):
        parser.error("--v1-closeout does not allow any --skip-* option")
    owner_hash = args.v1_owner_hash or os.getenv("V1_ACCEPTANCE_OWNER_HASH")
    source_revision = args.v1_source_revision or os.getenv("SOURCE_REVISION")
    if args.v1_closeout and not owner_hash:
        parser.error("--v1-owner-hash or V1_ACCEPTANCE_OWNER_HASH is required")
    if args.v1_closeout and (
        not source_revision
        or source_revision.strip().casefold() in {"development", "unknown"}
    ):
        parser.error(
            "--v1-source-revision or SOURCE_REVISION must identify the candidate build"
        )
    missing = check_files()
    mysql = check_mysql_references()
    compose_ok = True if args.skip_compose else check_compose()
    commands: list[dict[str, object]] = []
    if not args.skip_static:
        commands.extend(
            [
                _run("ruff", [PYTHON, "-m", "ruff", "check", "apps/api", "tests"]),
                _run("mypy", [PYTHON, "-m", "mypy", "apps/api/app"]),
                _run("pytest", [PYTHON, "-m", "pytest", "-q"]),
            ]
        )
    if not args.skip_integration:
        if os.getenv("RUN_POSTGRES_INTEGRATION") == "1":
            commands.append(
                _run(
                    "postgres_integration",
                    [PYTHON, "-m", "pytest", "tests/integration", "-q"],
                )
            )
        else:
            commands.append(
                {
                    "name": "postgres_integration",
                    "passed": False,
                    "returncode": 2,
                    "output_tail": "RUN_POSTGRES_INTEGRATION=1 is required for the strict gate",
                }
            )
    if not args.skip_web:
        commands.append(
            _run("web_build", [PNPM, "--filter", "@deep-research/web", "build"])
        )
    real_acceptance: dict[str, object] = {
        "name": "real_v1_acceptance",
        "requested": args.v1_closeout,
        "passed": False,
        "returncode": 2,
        "output_tail": "not requested; static checks cannot establish V1 closeout",
    }
    if args.v1_closeout:
        assert owner_hash is not None
        assert source_revision is not None
        acceptance_report = (
            args.report_path.with_name("v1_real_acceptance.json")
            if args.report_path is not None
            else ROOT / "artifacts" / "v1_real_acceptance.json"
        )
        acceptance_command = [
            PYTHON,
            "scripts/verify_v1_real_runs.py",
            "--owner-hash",
            owner_hash,
            "--count",
            "3",
            "--report-path",
            str(acceptance_report),
            "--expected-source-revision",
            source_revision,
        ]
        real_acceptance = {
            **_run("real_v1_acceptance", acceptance_command),
            "requested": True,
        }
    secret_hits = check_secret_scan()
    golden = check_golden_eval()
    head_revision, _ = migration_head()
    command_ok = all(bool(item["passed"]) for item in commands)
    checks = {
        "required_files": not missing,
        "no_mysql_references": not mysql,
        "compose_config": compose_ok,
        "migration_head": head_revision or "unknown",
        "commands": command_ok,
        "golden_eval": bool(golden["passed"]),
        "secret_scan": not secret_hits,
    }
    if args.v1_closeout:
        checks["real_v1_acceptance"] = bool(real_acceptance["passed"])
    payload = {
        "passed": all(checks.values()),
        "scope": "v1_closeout" if args.v1_closeout else "static_release_checks",
        "checks": checks,
        "missing_files": missing,
        "forbidden_references": mysql,
        "commands": commands,
        "golden_eval": golden,
        "real_v1_acceptance": real_acceptance,
        "secret_hits": secret_hits,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(rendered)
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(rendered + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
