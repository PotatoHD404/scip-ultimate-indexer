from __future__ import annotations

import math
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Sentinel used in git-log --pretty=format to delimit commit records.
# Must not appear in git hashes (hex chars only) or POSIX file paths.
_SENTINEL = "__GS_COMMIT__"


@dataclass(slots=True)
class GitSignals:
    """Aggregated git-history signals for a project."""

    recency: dict[str, float]
    """repo-relative POSIX path -> [0,1], 1.0 = most recent commit"""

    churn: dict[str, float]
    """repo-relative POSIX path -> [0,1] normalized commit-count"""

    cochange: dict[tuple[str, str], float]
    """ordered pair (a,b) with a<b -> [0,1] coupling strength"""

    @classmethod
    def empty(cls) -> GitSignals:
        """Return a GitSignals with all empty dicts."""
        return cls(recency={}, churn={}, cochange={})


def _run(args: list[str], cwd: Path, timeout: float) -> str:
    """Run a subprocess and return stdout as a string; raise on failure."""
    result = subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


def collect_git_signals(
    project_root: Path,
    *,
    history_limit: int = 2000,
    half_life_days: float = 90.0,
    now_ts: float | None = None,
    max_files_per_commit: int = 30,
    timeout_seconds: float = 20.0,
) -> GitSignals:
    """Mine git log for recency, churn, and co-change signals.

    Degrades gracefully: returns GitSignals.empty() on any failure.
    """
    try:
        return _collect(
            project_root=project_root,
            history_limit=history_limit,
            half_life_days=half_life_days,
            now_ts=now_ts,
            max_files_per_commit=max_files_per_commit,
            timeout_seconds=timeout_seconds,
        )
    except Exception:  # noqa: BLE001
        return GitSignals.empty()


def _collect(
    project_root: Path,
    history_limit: int,
    half_life_days: float,
    now_ts: float | None,
    max_files_per_commit: int,
    timeout_seconds: float,
) -> GitSignals:
    if shutil.which("git") is None:
        return GitSignals.empty()

    root = project_root.resolve()

    # Verify this is a git repo and get the toplevel
    try:
        toplevel_raw = _run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            cwd=root,
            timeout=timeout_seconds,
        ).strip()
    except Exception:  # noqa: BLE001
        return GitSignals.empty()

    toplevel = Path(toplevel_raw).resolve()

    # Build a prefix to strip from paths when project_root is a subdir of toplevel
    try:
        rel_prefix = root.relative_to(toplevel)
    except ValueError:
        rel_prefix = None

    # Fetch git log.  We use a visible sentinel that cannot appear in a git
    # hash or POSIX path.  Format per commit:
    #   __GS_COMMIT__<hash>|<unix_ts>\n<file1>\n<file2>\n\n
    log_output = _run(
        [
            "git",
            "-C",
            str(root),
            "log",
            "--no-merges",
            f"-n{history_limit}",
            "--name-only",
            f"--pretty=format:{_SENTINEL}%H|%ct",
        ],
        cwd=root,
        timeout=timeout_seconds,
    )

    commits = _parse_log(log_output)

    now = now_ts if now_ts is not None else time.time()
    half_life_seconds = half_life_days * 86400.0

    recency_raw: dict[str, float] = {}
    churn_counts: dict[str, int] = defaultdict(int)
    cochange_raw: dict[tuple[str, str], float] = defaultdict(float)

    for ts, raw_paths in commits:
        paths = _normalize_paths(raw_paths, toplevel, rel_prefix)
        if not paths:
            continue

        k = len(paths)

        # Recency: keep the most-recent ts per file
        for p in paths:
            prev = recency_raw.get(p)
            if prev is None or ts > prev:
                recency_raw[p] = ts

        # Churn: count commits per file
        for p in paths:
            churn_counts[p] += 1

        # Co-change: skip commits that are too large
        if 2 <= k <= max_files_per_commit:
            weight = 1.0 / math.log2(k + 1)
            sorted_paths = sorted(paths)
            for i in range(len(sorted_paths)):
                for j in range(i + 1, len(sorted_paths)):
                    pair = (sorted_paths[i], sorted_paths[j])
                    cochange_raw[pair] += weight

    # Build recency dict: apply exponential decay
    recency: dict[str, float] = {}
    for path, last_ts in recency_raw.items():
        elapsed = now - last_ts
        val = 0.5 ** (elapsed / half_life_seconds)
        recency[path] = max(0.0, min(1.0, val))

    # Build churn dict: normalize by max
    churn: dict[str, float] = {}
    if churn_counts:
        max_count = max(churn_counts.values())
        if max_count > 0:
            churn = {p: c / max_count for p, c in churn_counts.items()}

    # Build cochange dict: normalize by max, keep >= 0.05
    cochange: dict[tuple[str, str], float] = {}
    if cochange_raw:
        max_weight = max(cochange_raw.values())
        if max_weight > 0:
            for pair, w in cochange_raw.items():
                normalized = w / max_weight
                if normalized >= 0.05:
                    cochange[pair] = normalized

    return GitSignals(recency=recency, churn=churn, cochange=cochange)


def _parse_log(log_output: str) -> list[tuple[float, list[str]]]:
    """Parse git log output into a list of (unix_ts, [paths]).

    Expected line format (one per commit header):
      __GS_COMMIT__<40-char-hash>|<unix_ts>
    Followed by zero or more file paths, then a blank line before the next commit.
    """
    commits: list[tuple[float, list[str]]] = []
    current_ts: float | None = None
    current_paths: list[str] = []

    for line in log_output.splitlines():
        if line.startswith(_SENTINEL):
            # Flush previous commit
            if current_ts is not None:
                commits.append((current_ts, current_paths))
            # Parse new commit header: __GS_COMMIT__<hash>|<ts>
            rest = line[len(_SENTINEL):]
            parts = rest.split("|", 1)
            if len(parts) == 2:
                try:
                    current_ts = float(parts[1].strip())
                except ValueError:
                    current_ts = None
            else:
                current_ts = None
            current_paths = []
        else:
            stripped = line.strip()
            if stripped and current_ts is not None:
                current_paths.append(stripped)

    # Flush the last commit
    if current_ts is not None:
        commits.append((current_ts, current_paths))

    return commits


def _normalize_paths(
    raw_paths: list[str],
    toplevel: Path,
    rel_prefix: Path | None,
) -> list[str]:
    """Convert raw git paths to paths relative to project_root."""
    result: list[str] = []
    for p in raw_paths:
        if not p:
            continue
        if rel_prefix is not None and rel_prefix != Path("."):
            prefix_str = rel_prefix.as_posix() + "/"
            if p.startswith(prefix_str):
                result.append(p[len(prefix_str):])
            else:
                # File is outside project_root; keep as-is relative to toplevel
                result.append(p)
        else:
            result.append(p)
    return result


def recency_churn_boost(
    signals: GitSignals,
    relpath: str,
    *,
    recency_weight: float = 0.6,
    churn_weight: float = 0.4,
) -> float:
    """Return a combined [0,1] boost score for a file.

    Unknown files score 0.
    """
    return (
        recency_weight * signals.recency.get(relpath, 0.0)
        + churn_weight * signals.churn.get(relpath, 0.0)
    )


def cochanged_paths(
    signals: GitSignals,
    focus_paths: Iterable[str],
    *,
    top_n: int = 50,
) -> dict[str, float]:
    """Return paths that co-change with any of the focus_paths.

    Maps other_path -> max coupling weight to any focus path.
    Result is sorted by weight descending and truncated to top_n.
    """
    focus_set = set(focus_paths)
    best: dict[str, float] = {}

    for (a, b), weight in signals.cochange.items():
        if a in focus_set and b not in focus_set:
            if weight > best.get(b, 0.0):
                best[b] = weight
        elif b in focus_set and a not in focus_set:
            if weight > best.get(a, 0.0):
                best[a] = weight

    sorted_items = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    return dict(sorted_items[:top_n])
