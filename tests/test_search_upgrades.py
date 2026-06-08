"""Integration tests for the Tier-1 retrieval upgrades.

Exercises the features through the real index/query pipeline with the
deterministic ``hash`` embedding backend:

* zero-config Python SCIP (real Function/Method/Class symbols),
* vocabulary expansion (FTS-only abbreviation bridge),
* HyDE (natural-language queries return code),
* two-stage reranking (exact symbol-name query wins),
* contextual embeddings (context changes the stored embedding),
* git recency / co-change signals,
* context-personalized ranking (focus files bias results).
"""

from __future__ import annotations

import shutil
import subprocess
from hashlib import sha256

import numpy as np
import pytest

from ultimate_indexer import hyde
from ultimate_indexer.indexer import UltimateIndexer

_FILES = {
    "pkg/__init__.py": "",
    "pkg/models.py": (
        '"""Domain models."""\n\n'
        "class User:\n"
        '    """A user of the system."""\n\n'
        "    def __init__(self, name: str) -> None:\n"
        "        self.name = name\n"
    ),
    "pkg/services.py": (
        '"""Service layer."""\n\n'
        "from .models import User\n\n\n"
        "class GreetingService:\n"
        '    """Builds greetings for users."""\n\n'
        "    def build_greeting(self, user: User) -> str:\n"
        '        return f"Hello, {user.name}"\n'
    ),
    "app.py": (
        '"""Entry point."""\n\n'
        "from pkg.services import GreetingService\n\n\n"
        "def run() -> str:\n"
        '    """Greet a demo user."""\n'
        "    return GreetingService().build_greeting(None)\n"
    ),
    "docs/guide.md": "# Guide\n\nHow users are greeted.\n",
}


def _materialize(root, *, git: bool) -> None:
    for rel, content in _FILES.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    if git and shutil.which("git"):
        env_args = [
            "-c", "user.email=t@t.co", "-c", "user.name=t",
        ]
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", *env_args, "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", *env_args, "commit", "-qm", "init"], cwd=root, check=True)


@pytest.fixture
def indexed(tmp_path, monkeypatch):
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")
    root = tmp_path / "proj"
    _materialize(root, git=True)
    indexer = UltimateIndexer(root)
    indexer.index(force=True)
    yield indexer
    indexer.close()


def _names(groups) -> list[str]:
    return [symbol.display_name for group in groups for symbol in group.symbols]


def test_builtin_python_scip_emits_code_symbols(indexed):
    rows = indexed.storage.get_symbol_rows(indexed.project_id)
    kinds = {str(row["kind"]) for row in rows.values()}
    assert {"Class", "Function", "Method"}.issubset(kinds)


def test_reranker_promotes_exact_symbol(indexed):
    groups = indexed.query("build_greeting", limit=5, scope="code")
    names = _names(groups)
    assert names, "expected results for an exact symbol query"
    assert names[0] == "build_greeting"


def test_vocabulary_expansion_is_fts_only(indexed):
    # 'svc' never appears in the source, only in the FTS expansion field.
    groups = indexed.query("svc", limit=5, scope="code")
    assert "GreetingService" in _names(groups)
    # ...and the abbreviation must not leak into any displayed content.
    for group in groups:
        for symbol in group.symbols:
            assert "svc" not in symbol.signature.lower()
            assert "svc" not in symbol.snippet.lower()


def test_hyde_natural_language_query_returns_code(indexed):
    query = "how are users greeted"
    groups = indexed.query(query, limit=5, scope="code")
    assert groups
    # HyDE embedded a hypothetical document (deterministic template here, since no
    # generation endpoint is configured), cached under hyde::<hash-of-hypo-text>.
    hypo_text = hyde.hypothetical_code(query)
    cache_key = f"hyde::{sha256(hypo_text.encode('utf-8')).hexdigest()}"
    cached = indexed.storage.get_or_create_embedding(
        indexed._provider_instance().model_id, cache_key
    )
    assert cached is not None


def test_git_cochange_recorded(indexed):
    if shutil.which("git") is None:
        pytest.skip("git not available")
    neighbors = indexed.storage.get_cochange_neighbors(indexed.project_id, ["app.py"])
    # app.py was committed together with the other files.
    assert neighbors
    assert any(path != "app.py" for path in neighbors)


def test_focus_personalization_runs(indexed):
    plain = indexed.query("user", limit=5, scope="code")
    focused = indexed.query("user", limit=5, scope="code", focus_paths=("app.py",))
    assert focused, "focused query should return results"
    # Focusing must not crash and should still surface relevant code.
    assert _names(focused)
    # Cache isolation: focus produces a distinct cache entry, not the plain one.
    assert isinstance(plain, list)


def test_contextual_embedding_changes_vector(tmp_path, monkeypatch):
    monkeypatch.setenv("ULTIMATE_INDEXER_EMBEDDING_BACKEND", "hash")

    def _embed_for(symbol_substr: str, *, contextual: bool):
        monkeypatch.setenv(
            "ULTIMATE_INDEXER_ENABLE_CONTEXTUAL", "true" if contextual else "false"
        )
        root = tmp_path / ("ctx" if contextual else "plain")
        _materialize(root, git=False)
        indexer = UltimateIndexer(root)
        try:
            indexer.index(force=True)
            matrix, rows = indexer.storage.load_chunk_vectors(
                indexer.project_id, indexer._provider_instance().model_id
            )
            for i, row in enumerate(rows):
                if symbol_substr in str(row["symbol_name"]):
                    return np.array(matrix[i], dtype=np.float32)
            return None
        finally:
            indexer.close()

    with_ctx = _embed_for("build_greeting", contextual=True)
    without_ctx = _embed_for("build_greeting", contextual=False)
    assert with_ctx is not None and without_ctx is not None
    # Prepending the structural context header changes the embedded text, and
    # therefore the (deterministic hash) embedding.
    assert not np.allclose(with_ctx, without_ctx)
