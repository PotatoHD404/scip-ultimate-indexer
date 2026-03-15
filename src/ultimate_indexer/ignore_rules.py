from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pathspec

from .constants import DEFAULT_IGNORED_DIR_NAMES, DEFAULT_IGNORE_PATTERNS


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def _load_nested_gitignore_patterns(project_root: Path) -> list[str]:
    patterns: list[str] = []
    skip_dirs = {
        "node_modules",
        ".git",
        ".svn",
        ".hg",
        "dist",
        "build",
        "__pycache__",
        ".venv",
        "venv",
        "target",
        ".gradle",
        ".next",
        ".ultimate_indexer",
    }
    for gitignore in project_root.rglob(".gitignore"):
        if gitignore.parent == project_root:
            continue
        if any(part in skip_dirs for part in gitignore.parent.parts):
            continue
        rel_dir = gitignore.parent.relative_to(project_root).as_posix()
        for line in _read_lines(gitignore):
            trimmed = line.strip()
            if not trimmed or trimmed.startswith("#"):
                continue
            if trimmed.startswith("!"):
                patterns.append(f"!{rel_dir}/{trimmed[1:]}")
            else:
                patterns.append(f"{rel_dir}/{trimmed}")
    return patterns


def _find_upward_ignore_file(start: Path, filename: str) -> Path | None:
    current = start.resolve()
    while True:
        candidate = current / filename
        if candidate.exists():
            return candidate
        if current.parent == current:
            return None
        current = current.parent


@dataclass(slots=True)
class IgnoreMatcher:
    project_root: Path
    spec: pathspec.PathSpec
    ignored_dir_names: set[str]

    def ignores(self, relative_path: str) -> bool:
        normalized = relative_path.replace(os.sep, "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        if not normalized:
            return False
        parts = {part.lower() for part in Path(normalized).parts}
        if parts.intersection(self.ignored_dir_names):
            return True
        return self.spec.match_file(normalized)


def create_ignore_matcher(project_root: Path) -> IgnoreMatcher:
    patterns: list[str] = list(DEFAULT_IGNORE_PATTERNS)

    respect_gitignore = os.getenv("RESPECT_GITIGNORE", "true").lower() != "false"
    if respect_gitignore:
        root_gitignore = project_root / ".gitignore"
        if root_gitignore.exists():
            patterns.extend(_read_lines(root_gitignore))
        patterns.extend(_load_nested_gitignore_patterns(project_root))

    socraticodeignore = project_root / ".socraticodeignore"
    if socraticodeignore.exists():
        patterns.extend(_read_lines(socraticodeignore))

    cgcignore = _find_upward_ignore_file(project_root, ".cgcignore")
    if cgcignore is not None:
        patterns.extend(_read_lines(cgcignore))

    ignore_dirs_env = os.getenv("IGNORE_DIRS", "")
    ignored_dir_names = {
        item.strip().lower()
        for item in ignore_dirs_env.split(",")
        if item.strip()
    }
    ignored_dir_names.update({item.lower() for item in DEFAULT_IGNORED_DIR_NAMES})
    ignored_dir_names.update({".svn", ".hg"})

    spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    return IgnoreMatcher(project_root=project_root, spec=spec, ignored_dir_names=ignored_dir_names)
