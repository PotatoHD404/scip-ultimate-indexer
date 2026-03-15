from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .constants import SCIP_EXTENSION_MAP, SCIP_RUN_ORDER


def _normalize_scip_language(language: str) -> str:
    normalized = language.strip().lower()
    if normalized in {"js", "jsx", "javascript"}:
        return "typescript"
    if normalized in {"ts", "tsx", "typescript"}:
        return "typescript"
    if normalized in {"c", "cpp", "c++"}:
        return "cpp"
    if language == "javascript":
        return "typescript"
    if language == "c":
        return "cpp"
    return normalized


def _language_command(language: str, binary: str, output_file: Path) -> list[str] | None:
    out = str(output_file)
    if language == "python":
        return [binary, "index", ".", "--output", out]
    if language == "typescript":
        return [binary, "index", "--output", out]
    if language == "go":
        return [binary, "--output", out]
    if language == "rust":
        return [binary, "index", "--output", out]
    if language == "java":
        return [binary, "index", "--output", out]
    if language == "cpp":
        return [binary, f"--index-output-path={out}"]
    return None


def detect_scip_languages(files: list[Path]) -> list[str]:
    detected: set[str] = set()
    configured = {
        _normalize_scip_language(item)
        for item in os.getenv("SCIP_LANGUAGES", "").split(",")
        if item.strip()
    }
    for path in files:
        mapping = SCIP_EXTENSION_MAP.get(path.suffix.lower())
        if mapping is None:
            continue
        language = _normalize_scip_language(mapping[0])
        if configured and language not in configured:
            continue
        detected.add(language)
    return [language for language in SCIP_RUN_ORDER if language in detected]


@dataclass(slots=True)
class ScipRunResult:
    language: str
    index_path: Path


def run_scip_indexers(project_root: Path, files: list[Path], cache_dir: Path) -> list[ScipRunResult]:
    results: list[ScipRunResult] = []
    for language in detect_scip_languages(files):
        binary = None
        install_hint = "unknown"
        for _, (mapped_language, mapped_binary, hint) in SCIP_EXTENSION_MAP.items():
            if _normalize_scip_language(mapped_language) == language:
                binary = shutil.which(mapped_binary)
                install_hint = hint
                if binary:
                    break
        if binary is None:
            continue

        output_file = cache_dir / f"{language}.scip"
        command = _language_command(language, binary, output_file)
        if command is None:
            continue
        try:
            completed = subprocess.run(
                command,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=300,
            )
        except Exception:
            continue
        if completed.returncode != 0 or not output_file.exists():
            _ = install_hint
            continue
        results.append(ScipRunResult(language=language, index_path=output_file))
    return results
