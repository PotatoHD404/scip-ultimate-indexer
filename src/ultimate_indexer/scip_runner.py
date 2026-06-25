from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .constants import SCIP_EXTENSION_MAP, SCIP_RUN_ORDER
from .python_scip import emit_python_scip


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


def _language_command(
    language: str, binary: str, output_file: Path, invocation_root: Path | None = None
) -> list[str] | None:
    out = str(output_file)
    if language == "python":
        # scip-python crashes ("Cannot read properties of undefined" in
        # makePackage) when it cannot infer a package name/version — e.g. on a
        # project without git metadata. Always pass both explicitly.
        project_name = re.sub(
            r"[^A-Za-z0-9._-]+", "-", invocation_root.name if invocation_root else "project"
        ).strip("-") or "project"
        return [
            binary, "index", ".",
            "--project-name", project_name,
            "--project-version", "0.0.0",
            "--output", out,
        ]
    if language == "typescript":
        return [binary, "index", "--output", out]
    if language == "go":
        return [binary, "--output", out]
    if language == "rust":
        return [binary, "scip", ".", "--output", out]
    if language == "java":
        cmd = [binary, "index", "--output", out]
        # scip-java aborts ("Multiple build tools detected") when a repo carries
        # both Maven and Gradle metadata; pick one explicitly (prefer Maven).
        if invocation_root is not None and (invocation_root / "pom.xml").exists():
            gradle = ("build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts")
            if any((invocation_root / g).exists() for g in gradle):
                cmd.append("--build-tool=maven")
        return cmd
    if language == "cpp":
        return [binary, f"--index-output-path={out}"]
    return None


def _ensure_compile_db(
    invocation_root: Path, cache_dir: Path, timeout_seconds: int | None
) -> Path | None:
    """scip-clang requires a compile_commands.json. Prefer the project's own; for
    a CMake project without one, generate it into the cache (best-effort) with
    `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` so the user's tree is never written to.
    Returns the database path to use, or None if none could be obtained."""
    existing = invocation_root / "compile_commands.json"
    if existing.exists():
        return existing
    if not (invocation_root / "CMakeLists.txt").exists():
        return None
    cmake = shutil.which("cmake")
    if cmake is None:
        return None
    build_dir = cache_dir / "cmake-compdb"
    try:
        subprocess.run(
            [cmake, "-S", str(invocation_root), "-B", str(build_dir),
             "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except Exception:
        return None
    generated = build_dir / "compile_commands.json"
    return generated if generated.exists() else None


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
    source_root: Path


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
    working_directory: str
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
            lines.append(f"Working directory: {failure.working_directory}")
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


SCIP_PROJECT_MARKERS: dict[str, tuple[str, ...]] = {
    "typescript": ("tsconfig.json", "jsconfig.json"),
    "go": ("go.mod",),
    "rust": ("Cargo.toml",),
    "java": ("pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts"),
}


def _files_for_language(files: list[Path], language: str) -> list[Path]:
    matched: list[Path] = []
    for path in files:
        mapping = SCIP_EXTENSION_MAP.get(path.suffix.lower())
        if mapping is None:
            continue
        if _normalize_scip_language(mapping[0]) == language:
            matched.append(path)
    return matched


def _nearest_project_root(path: Path, project_root: Path, markers: tuple[str, ...]) -> Path | None:
    current = path.parent
    resolved_root = project_root.resolve()
    while True:
        if any((current / marker).exists() for marker in markers):
            return current
        if current == resolved_root:
            return None
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _group_invocation_roots(project_root: Path, language: str, files: list[Path]) -> list[Path]:
    markers = SCIP_PROJECT_MARKERS.get(language)
    if not markers:
        return [project_root]

    roots: set[Path] = set()
    for path in _files_for_language(files, language):
        root = _nearest_project_root(path.resolve(), project_root.resolve(), markers)
        if root is not None:
            roots.add(root)
    return sorted(roots)


def _output_file_for_root(cache_dir: Path, language: str, invocation_root: Path, project_root: Path) -> Path:
    relative_root = invocation_root.resolve().relative_to(project_root.resolve()).as_posix()
    suffix = "root" if relative_root == "." else sha256(relative_root.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{language}-{suffix}.scip"


def _emit_builtin_python_scip(
    project_root: Path, files: list[Path], cache_dir: Path
) -> ScipRunResult | None:
    """Produce a SCIP index for Python using the in-tree emitter (no external tool).

    Returns ``None`` when there is nothing to emit or emission fails, so callers
    can degrade to generic non-SCIP coverage.
    """
    python_files = _files_for_language(files, "python")
    if not python_files:
        return None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_file = cache_dir / "python-builtin.scip"
        emit_python_scip(project_root.resolve(), python_files, output_file)
    except Exception:
        return None
    if not output_file.exists():
        return None
    return ScipRunResult(
        language="python", index_path=output_file, source_root=project_root.resolve()
    )


def run_scip_indexers(
    project_root: Path,
    files: list[Path],
    cache_dir: Path,
    timeout_seconds: int | None = None,
) -> ScipRunReport:
    results: list[ScipRunResult] = []
    missing: list[ScipRequirement] = []
    failed: list[ScipRunFailure] = []
    detected_languages = detect_scip_languages(files)
    # Operator/test switch: skip external SCIP tools entirely (offline CI,
    # locked-down hosts, deterministic test runs). The built-in Python emitter
    # still provides zero-config coverage.
    disable_external = os.getenv(
        "ULTIMATE_INDEXER_DISABLE_EXTERNAL_SCIP", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    available_binaries: dict[str, tuple[ScipRequirement, str]] = {}
    for language in detected_languages:
        requirement = _language_tooling(language, files)
        if requirement is None:
            continue
        if disable_external:
            # Explicitly disabled: no language is treated as a hard requirement;
            # Python uses the in-tree emitter, everything else uses generic
            # fallback coverage.
            continue
        binary = shutil.which(requirement.binary_name)
        if binary is None:
            # Python has an in-tree emitter, so a missing external tool is not
            # fatal — it is handled in the run loop below.
            if language == "python":
                continue
            missing.append(requirement)
            continue
        available_binaries[language] = (requirement, binary)
    if missing:
        return ScipRunReport(results=results, missing=missing, failed=failed)

    for language in detected_languages:
        available = available_binaries.get(language)
        if available is None:
            if language == "python":
                builtin = _emit_builtin_python_scip(project_root, files, cache_dir)
                if builtin is not None:
                    results.append(builtin)
            continue
        requirement, binary = available
        invocation_roots = _group_invocation_roots(project_root, language, files)
        if not invocation_roots:
            continue
        language_succeeded = False
        for invocation_root in invocation_roots:
            output_file = _output_file_for_root(cache_dir, language, invocation_root, project_root)
            command = _language_command(language, binary, output_file, invocation_root)
            if command is None:
                continue
            if language == "cpp":
                compdb = _ensure_compile_db(invocation_root, cache_dir, timeout_seconds)
                if compdb is None:
                    failed.append(
                        ScipRunFailure(
                            language=language,
                            binary_name=requirement.binary_name,
                            install_hint=(
                                "scip-clang needs a compile_commands.json. A CMake project "
                                "is configured automatically; otherwise generate one (e.g. "
                                "`bear -- <build>` or a CMake build with "
                                "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)."
                            ),
                            working_directory=str(invocation_root),
                            command=tuple(command),
                            detail="No compile_commands.json found and none could be generated.",
                        )
                    )
                    continue
                if compdb != invocation_root / "compile_commands.json":
                    command = [*command, f"--compdb-path={compdb}"]
            # Remove any index left by a previous run so a tool that exits 0
            # without writing output cannot silently pass off stale data.
            output_file.unlink(missing_ok=True)
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(invocation_root),
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
            except Exception as exc:
                failed.append(
                    ScipRunFailure(
                        language=language,
                        binary_name=requirement.binary_name,
                        install_hint=requirement.install_hint,
                        working_directory=str(invocation_root),
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
                        working_directory=str(invocation_root),
                        command=tuple(command),
                        detail=detail,
                    )
                )
                continue
            results.append(ScipRunResult(language=language, index_path=output_file, source_root=invocation_root))
            language_succeeded = True
        # External scip-python was present but produced no usable index (e.g. a
        # broken runtime). Fall back to the in-tree emitter so Python projects
        # still get function/class symbols rather than coarse fallback coverage.
        if language == "python" and not language_succeeded:
            builtin = _emit_builtin_python_scip(project_root, files, cache_dir)
            if builtin is not None:
                results.append(builtin)
    return ScipRunReport(results=results, missing=missing, failed=failed)
