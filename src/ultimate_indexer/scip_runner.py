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
        return [binary, "scip", ".", "--output", out]
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


@dataclass(slots=True)
class ScipRequirement:
    language: str
    binary_name: str
    install_hint: str
    files: tuple[str, ...]


@dataclass(slots=True)
class ScipRunFailure:
    language: str
    binary_name: str
    install_hint: str
    command: tuple[str, ...]
    detail: str


@dataclass(slots=True)
class ScipRunReport:
    results: list[ScipRunResult]
    missing: list[ScipRequirement]
    failed: list[ScipRunFailure]


class StructuredIndexingRequiredError(RuntimeError):
    def __init__(self, missing: list[ScipRequirement], failed: list[ScipRunFailure]) -> None:
        self.missing = missing
        self.failed = failed
        super().__init__(self.render_message())

    def render_message(self) -> str:
        lines = [
            "Structured parsing is available for detected languages, so fallback indexing was skipped.",
        ]
        for requirement in self.missing:
            lines.append("")
            lines.append(
                f"{requirement.language}: `{requirement.binary_name}` is not installed."
            )
            lines.append(f"Install it with: {requirement.install_hint}")
            if requirement.files:
                lines.append(f"Detected files: {', '.join(requirement.files[:5])}")
        for failure in self.failed:
            lines.append("")
            lines.append(
                f"{failure.language}: `{failure.binary_name}` failed while building the SCIP index."
            )
            lines.append(f"Command: {' '.join(failure.command)}")
            lines.append(f"Details: {failure.detail}")
        lines.append("")
        lines.append("Then rerun indexing, or pass a ready-made index with `--scip-path`.")
        return "\n".join(lines)


def _language_tooling(language: str, files: list[Path]) -> ScipRequirement | None:
    binary_name = None
    install_hint = "unknown"
    for extension, (mapped_language, mapped_binary, hint) in SCIP_EXTENSION_MAP.items():
        if _normalize_scip_language(mapped_language) != language:
            continue
        binary_name = mapped_binary
        install_hint = hint
        break
    if binary_name is None:
        return None
    matching_files = tuple(
        path.as_posix()
        for path in files
        if _normalize_scip_language(SCIP_EXTENSION_MAP.get(path.suffix.lower(), ("", "", ""))[0]) == language
    )
    return ScipRequirement(
        language=language,
        binary_name=binary_name,
        install_hint=install_hint,
        files=matching_files,
    )


def run_scip_indexers(project_root: Path, files: list[Path], cache_dir: Path) -> ScipRunReport:
    results: list[ScipRunResult] = []
    missing: list[ScipRequirement] = []
    failed: list[ScipRunFailure] = []
    detected_languages = detect_scip_languages(files)
    for language in detected_languages:
        requirement = _language_tooling(language, files)
        if requirement is None:
            continue
        binary = shutil.which(requirement.binary_name)
        if binary is None:
            missing.append(requirement)
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
        except Exception as exc:
            failed.append(
                ScipRunFailure(
                    language=language,
                    binary_name=requirement.binary_name,
                    install_hint=requirement.install_hint,
                    command=tuple(command),
                    detail=str(exc),
                )
            )
            continue
        if completed.returncode != 0 or not output_file.exists():
            detail = completed.stderr.strip() or completed.stdout.strip() or "SCIP command did not produce an index."
            failed.append(
                ScipRunFailure(
                    language=language,
                    binary_name=requirement.binary_name,
                    install_hint=requirement.install_hint,
                    command=tuple(command),
                    detail=detail,
                )
            )
            continue
        results.append(ScipRunResult(language=language, index_path=output_file))
    return ScipRunReport(results=results, missing=missing, failed=failed)
