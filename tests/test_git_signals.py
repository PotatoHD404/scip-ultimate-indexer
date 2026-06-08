from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest

from ultimate_indexer.git_signals import (
    GitSignals,
    cochanged_paths,
    collect_git_signals,
    recency_churn_boost,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git(args: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    """Run a git command; raise CalledProcessError on failure."""
    subprocess.run(
        ["git"] + args,
        cwd=str(cwd),
        check=True,
        capture_output=True,
        env=env,
    )


def _make_repo(tmp_path: Path) -> Path:
    """Create a fresh git repo in tmp_path/repo."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(["-c", "init.defaultBranch=main", "init"], cwd=repo)
    _git(["-c", "user.email=test@example.com", "-c", "user.name=Test", "commit",
          "--allow-empty", "-m", "init"], cwd=repo)
    return repo


def _commit(
    repo: Path,
    files: list[tuple[str, str]],
    message: str,
    ts: int,
) -> None:
    """Write files and make a commit with a controlled timestamp."""
    import os
    for rel, content in files:
        dest = repo / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        _git(["add", rel], cwd=repo)

    date_str = f"{ts} +0000"
    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": date_str,
        "GIT_COMMITTER_DATE": date_str,
    }
    subprocess.run(
        [
            "git",
            "-c", "user.email=test@example.com",
            "-c", "user.name=Test",
            "commit",
            "-m", message,
        ],
        cwd=str(repo),
        check=True,
        capture_output=True,
        env=env,
    )


# ---------------------------------------------------------------------------
# Skip marker
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None,
    reason="git not installed",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGitSignalsEmpty:
    def test_empty_classmethod(self) -> None:
        s = GitSignals.empty()
        assert s.recency == {}
        assert s.churn == {}
        assert s.cochange == {}

    def test_non_git_dir_returns_empty(self, tmp_path: Path) -> None:
        non_git = tmp_path / "not_a_repo"
        non_git.mkdir()
        signals = collect_git_signals(non_git)
        assert signals.recency == {}
        assert signals.churn == {}
        assert signals.cochange == {}


class TestRecency:
    def test_recently_committed_file_has_higher_recency(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)

        old_ts = int(time.time()) - 180 * 86400  # 180 days ago
        new_ts = int(time.time()) - 1 * 86400    # 1 day ago

        _commit(repo, [("old_file.py", "x=1")], "old", old_ts)
        _commit(repo, [("new_file.py", "y=2")], "new", new_ts)

        signals = collect_git_signals(repo)

        assert "old_file.py" in signals.recency
        assert "new_file.py" in signals.recency
        assert signals.recency["new_file.py"] > signals.recency["old_file.py"]

    def test_recency_is_in_unit_interval(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        ts = int(time.time()) - 7 * 86400
        _commit(repo, [("a.py", "a=1")], "c1", ts)

        signals = collect_git_signals(repo)

        for val in signals.recency.values():
            assert 0.0 <= val <= 1.0

    def test_very_old_file_recency_near_zero(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        ancient_ts = int(time.time()) - 3650 * 86400  # ~10 years ago
        _commit(repo, [("ancient.py", "z=0")], "ancient", ancient_ts)

        signals = collect_git_signals(repo)

        assert signals.recency.get("ancient.py", 0.0) < 0.01


class TestChurn:
    def test_churn_counts_normalized(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        now = int(time.time())

        # hot.py touched 3 times, cold.py once
        _commit(repo, [("hot.py", "v=1"), ("cold.py", "c=0")], "c1", now - 100)
        _commit(repo, [("hot.py", "v=2")], "c2", now - 90)
        _commit(repo, [("hot.py", "v=3")], "c3", now - 80)

        signals = collect_git_signals(repo)

        assert signals.churn["hot.py"] == pytest.approx(1.0)
        assert signals.churn["cold.py"] == pytest.approx(1.0 / 3.0)

    def test_churn_is_in_unit_interval(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        now = int(time.time())
        _commit(repo, [("a.py", "1")], "c1", now - 50)
        _commit(repo, [("b.py", "2")], "c2", now - 40)

        signals = collect_git_signals(repo)

        for val in signals.churn.values():
            assert 0.0 <= val <= 1.0


class TestCoChange:
    def test_files_committed_together_form_cochange_pair(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        now = int(time.time())

        # a.py + b.py always committed together (2 times)
        _commit(repo, [("a.py", "a=1"), ("b.py", "b=1")], "c1", now - 200)
        _commit(repo, [("a.py", "a=2"), ("b.py", "b=2")], "c2", now - 100)
        # c.py committed alone
        _commit(repo, [("c.py", "c=1")], "c3", now - 50)

        signals = collect_git_signals(repo)

        pair = ("a.py", "b.py")
        assert pair in signals.cochange
        assert signals.cochange[pair] == pytest.approx(1.0)

    def test_cochange_normalized_to_unit_interval(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        now = int(time.time())
        _commit(repo, [("x.py", "1"), ("y.py", "2")], "c1", now - 100)

        signals = collect_git_signals(repo)

        for val in signals.cochange.values():
            assert 0.0 <= val <= 1.0

    def test_cochange_pairs_ordered(self, tmp_path: Path) -> None:
        """All pairs (a, b) must satisfy a < b."""
        repo = _make_repo(tmp_path)
        now = int(time.time())
        _commit(
            repo,
            [("z_file.py", "z"), ("a_file.py", "a"), ("m_file.py", "m")],
            "c1",
            now - 100,
        )

        signals = collect_git_signals(repo)

        for a, b in signals.cochange:
            assert a < b

    def test_large_commit_skipped(self, tmp_path: Path) -> None:
        """Commits with more files than max_files_per_commit must not contribute pairs."""
        repo = _make_repo(tmp_path)
        now = int(time.time())

        # Create a mechanical commit with 5 files, limit=3 -> skip
        files = [(f"gen_{i}.py", f"x={i}") for i in range(5)]
        _commit(repo, files, "mega", now - 100)

        signals = collect_git_signals(repo, max_files_per_commit=3)

        # No cochange should be recorded since the single commit was skipped
        assert signals.cochange == {}

    def test_solo_file_commits_produce_no_cochange(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        now = int(time.time())
        _commit(repo, [("a.py", "1")], "c1", now - 100)
        _commit(repo, [("b.py", "2")], "c2", now - 50)

        signals = collect_git_signals(repo)

        assert signals.cochange == {}


class TestRecencyChurnBoost:
    def test_known_file_returns_weighted_sum(self) -> None:
        signals = GitSignals(
            recency={"a.py": 0.8},
            churn={"a.py": 0.5},
            cochange={},
        )
        boost = recency_churn_boost(signals, "a.py")
        assert boost == pytest.approx(0.6 * 0.8 + 0.4 * 0.5)

    def test_unknown_file_returns_zero(self) -> None:
        signals = GitSignals.empty()
        assert recency_churn_boost(signals, "missing.py") == pytest.approx(0.0)

    def test_custom_weights(self) -> None:
        signals = GitSignals(
            recency={"f.py": 1.0},
            churn={"f.py": 1.0},
            cochange={},
        )
        assert recency_churn_boost(signals, "f.py", recency_weight=1.0, churn_weight=0.0) == pytest.approx(1.0)


class TestCochangedPaths:
    def test_returns_related_files(self) -> None:
        signals = GitSignals(
            recency={},
            churn={},
            cochange={
                ("a.py", "b.py"): 1.0,
                ("a.py", "c.py"): 0.5,
                ("b.py", "d.py"): 0.8,
            },
        )
        result = cochanged_paths(signals, ["a.py"])
        assert "b.py" in result
        assert "c.py" in result
        assert result["b.py"] == pytest.approx(1.0)
        assert result["c.py"] == pytest.approx(0.5)
        # a.py itself should not appear
        assert "a.py" not in result

    def test_looks_up_both_orderings(self) -> None:
        """cochange stores (a,b) with a<b; focus on 'b.py' should still find 'a.py'."""
        signals = GitSignals(
            recency={},
            churn={},
            cochange={("a.py", "b.py"): 0.9},
        )
        result = cochanged_paths(signals, ["b.py"])
        assert "a.py" in result
        assert result["a.py"] == pytest.approx(0.9)

    def test_top_n_truncation(self) -> None:
        cochange: dict[tuple[str, str], float] = {}
        for i in range(20):
            pair = ("focus.py", f"file_{i:02d}.py") if "focus.py" < f"file_{i:02d}.py" else (f"file_{i:02d}.py", "focus.py")
            # always keep ordering (a<b)
            a, b = (min("focus.py", f"file_{i:02d}.py"), max("focus.py", f"file_{i:02d}.py"))
            cochange[(a, b)] = float(i + 1) / 20.0

        signals = GitSignals(recency={}, churn={}, cochange=cochange)
        result = cochanged_paths(signals, ["focus.py"], top_n=5)
        assert len(result) == 5
        # Highest weights returned
        assert all(w >= min(result.values()) for w in result.values())

    def test_empty_focus_returns_empty(self) -> None:
        signals = GitSignals(
            recency={},
            churn={},
            cochange={("a.py", "b.py"): 1.0},
        )
        assert cochanged_paths(signals, []) == {}

    def test_focus_not_in_cochange_returns_empty(self) -> None:
        signals = GitSignals(
            recency={},
            churn={},
            cochange={("x.py", "y.py"): 1.0},
        )
        assert cochanged_paths(signals, ["z.py"]) == {}

    def test_max_weight_across_multiple_focus_files(self) -> None:
        signals = GitSignals(
            recency={},
            churn={},
            cochange={
                ("a.py", "shared.py"): 0.3,
                ("b.py", "shared.py"): 0.7,
            },
        )
        result = cochanged_paths(signals, ["a.py", "b.py"])
        # shared.py couples to both; max should be 0.7
        assert result.get("shared.py") == pytest.approx(0.7)


class TestIntegration:
    def test_full_workflow(self, tmp_path: Path) -> None:
        """Build a small repo and verify all three signal types end up populated."""
        repo = _make_repo(tmp_path)
        now = int(time.time())

        _commit(
            repo,
            [("src/main.py", "m=1"), ("src/utils.py", "u=1")],
            "feature",
            now - 10 * 86400,
        )
        _commit(repo, [("src/main.py", "m=2")], "hotfix", now - 1 * 86400)
        _commit(repo, [("tests/test_main.py", "t=1")], "tests", now - 5 * 86400)

        signals = collect_git_signals(repo, now_ts=float(now))

        # Recency: main.py most recently updated
        assert signals.recency["src/main.py"] > signals.recency["tests/test_main.py"]

        # Churn: main.py has 2 commits, others 1
        assert signals.churn["src/main.py"] == pytest.approx(1.0)
        assert signals.churn["src/utils.py"] == pytest.approx(0.5)

        # Co-change: main + utils co-changed once
        assert ("src/main.py", "src/utils.py") in signals.cochange
