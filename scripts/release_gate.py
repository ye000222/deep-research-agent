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
    "apps/api/alembic/versions/20260831_0016_evaluation_snapshots.py",
    "apps/api/app/retrieval/projections.py",
    "apps/api/app/evaluation/report_verifier.py",
    "evals/datasets/v1_golden.json",
)

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{20,}\b"),
)


def check_files() -> list[str]:
    return [item for item in REQUIRED if not (ROOT / item).exists()]


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
    parser.add_argument("--report-path", type=Path)
    args = parser.parse_args()
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
    secret_hits = check_secret_scan()
    golden = check_golden_eval()
    command_ok = all(bool(item["passed"]) for item in commands)
    checks = {
        "required_files": not missing,
        "no_mysql_references": not mysql,
        "compose_config": compose_ok,
        "migration_head": "20260831_0016" if not missing else "unknown",
        "commands": command_ok,
        "golden_eval": bool(golden["passed"]),
        "secret_scan": not secret_hits,
    }
    payload = {
        "passed": all(checks.values()),
        "checks": checks,
        "missing_files": missing,
        "forbidden_references": mysql,
        "commands": commands,
        "golden_eval": golden,
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
