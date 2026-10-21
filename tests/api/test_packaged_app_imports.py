from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_packaged_api_imports_without_repository_scripts(tmp_path: Path) -> None:
    api_package_root = Path(__file__).resolve().parents[2] / "apps" / "api"
    program = """
import importlib.abc
import sys

class BlockRepositoryScripts(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'scripts' or fullname.startswith('scripts.'):
            raise ModuleNotFoundError('repository scripts are not packaged')
        return None

sys.meta_path.insert(0, BlockRepositoryScripts())
import app.main
import app.services.prospective_acquisition_freeze
import app.services.prospective_acquisition_execution_identity
print('PACKAGED_API_IMPORT_OK')
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(api_package_root)
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "PACKAGED_API_IMPORT_OK" in result.stdout


def test_script_compatibility_imports_are_canonical_reexports() -> None:
    from app.services.prospective_acquisition_execution_identity import (
        create_execution_context as canonical_factory,
    )

    from scripts.prospective_acquisition.execution_identity import (
        create_execution_context as compatibility_factory,
    )

    assert compatibility_factory is canonical_factory
