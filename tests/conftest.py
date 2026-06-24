from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

# Hermetic tests: never invoke external SCIP toolchains (scip-python is a full
# pyright analysis costing 30-100s per project and its availability varies by
# host). The built-in Python emitter provides the deterministic coverage the
# assertions rely on. Tests that specifically target external tools can
# monkeypatch.delenv this.
os.environ.setdefault("ULTIMATE_INDEXER_DISABLE_EXTERNAL_SCIP", "1")


@pytest.fixture()
def fixture_project(tmp_path: Path) -> Path:
    source = Path(__file__).parent / "fixtures" / "sample_project"
    target = tmp_path / "sample_project"
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns(".ultimate_indexer", "__pycache__"),
    )
    return target
