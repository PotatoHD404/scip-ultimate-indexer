from __future__ import annotations

import shutil
from pathlib import Path

import pytest


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
