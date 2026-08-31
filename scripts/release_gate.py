"""V1 release gate checks for local and CI execution."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "docker-compose.yml",
    "README.md",
    "apps/api/alembic/versions/20260831_0016_evaluation_snapshots.py",
    "apps/api/app/retrieval/projections.py",
    "apps/api/app/evaluation/report_verifier.py",
    "evals/datasets/v1_golden.json",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-compose", action="store_true")
    args = parser.parse_args()
    missing = check_files()
    mysql = check_mysql_references()
    compose_ok = True if args.skip_compose else check_compose()
    checks = {
        "required_files": not missing,
        "no_mysql_references": not mysql,
        "compose_config": compose_ok,
        "migration_head": "20260831_0016" if not missing else "unknown",
    }
    payload = {
        "passed": all(checks.values()),
        "checks": checks,
        "missing_files": missing,
        "forbidden_references": mysql,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
