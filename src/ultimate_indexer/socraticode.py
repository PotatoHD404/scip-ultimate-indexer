from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .embeddings import hash_text
from .models import ArtifactSpec, ChunkRecord, EdgeRecord, FileRecord, SymbolRecord


CONFIG_FILENAME = ".socraticodecontextartifacts.json"
TEXT_EXTENSIONS = {
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".sql",
    ".graphql",
    ".proto",
    ".toml",
    ".env",
    ".ini",
}


@dataclass(slots=True)
class ArtifactBundle:
    files: list[FileRecord]
    symbols: list[SymbolRecord]
    edges: list[EdgeRecord]
    chunks: list[ChunkRecord]


def load_artifact_specs(project_root: Path) -> list[ArtifactSpec]:
    config_path = project_root / CONFIG_FILENAME
    if not config_path.exists():
        return []
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    artifacts = raw.get("artifacts", [])
    return [
        ArtifactSpec(
            name=item["name"],
            path=item["path"],
            description=item["description"],
        )
        for item in artifacts
    ]


def iter_artifact_files(project_root: Path, specs: Iterable[ArtifactSpec]) -> list[tuple[ArtifactSpec, Path]]:
    results: list[tuple[ArtifactSpec, Path]] = []
    for spec in specs:
        resolved = (project_root / spec.path).resolve()
        if resolved.is_file():
            results.append((spec, resolved))
            continue
        if not resolved.is_dir():
            continue
        for path in sorted(resolved.rglob("*")):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            results.append((spec, path))
    return results


def _line_chunks(
    text: str,
    width: int = 80,
    overlap: int = 12,
) -> list[tuple[int, int, str]]:
    lines = text.splitlines()
    if not lines:
        return []
    chunks: list[tuple[int, int, str]] = []
    step = max(1, width - overlap)
    for start in range(0, len(lines), step):
        end = min(len(lines), start + width)
        chunks.append((start + 1, end, "\n".join(lines[start:end]).strip()))
        if end >= len(lines):
            break
    return chunks


def ingest_socraticode_artifacts(project_id: str, project_root: Path) -> ArtifactBundle:
    specs = load_artifact_specs(project_root)
    files: list[FileRecord] = []
    symbols: list[SymbolRecord] = []
    edges: list[EdgeRecord] = []
    chunks: list[ChunkRecord] = []

    for spec, artifact_path in iter_artifact_files(project_root, specs):
        content = artifact_path.read_text(encoding="utf-8")
        relative_path = str(artifact_path.relative_to(project_root))
        file_symbol_id = f"file::{relative_path}"
        artifact_symbol_id = f"artifact::{spec.name}::{relative_path}"
        digest = hash_text(content)
        files.append(
            FileRecord(
                project_id=project_id,
                relative_path=relative_path,
                abs_path=str(artifact_path),
                language=artifact_path.suffix.lower().lstrip(".") or "text",
                content_hash=digest,
                content=content,
                source_kind="artifact",
                artifact_name=spec.name,
            )
        )
        symbols.append(
            SymbolRecord(
                project_id=project_id,
                symbol_id=file_symbol_id,
                scip_symbol=file_symbol_id,
                display_name=artifact_path.name,
                kind="File",
                relative_path=relative_path,
                start_line=1,
                end_line=max(1, len(content.splitlines())),
                signature=relative_path,
                docstring=spec.description,
                snippet="\n".join(content.splitlines()[: min(20, len(content.splitlines()))]),
                source_kind="artifact",
            )
        )
        symbols.append(
            SymbolRecord(
                project_id=project_id,
                symbol_id=artifact_symbol_id,
                scip_symbol=artifact_symbol_id,
                display_name=spec.name,
                kind="Artifact",
                relative_path=relative_path,
                start_line=1,
                end_line=max(1, len(content.splitlines())),
                signature=relative_path,
                docstring=spec.description,
                snippet="\n".join(content.splitlines()[: min(40, len(content.splitlines()))]),
                enclosing_symbol_id=file_symbol_id,
                source_kind="artifact",
            )
        )
        edges.append(
            EdgeRecord(
                project_id=project_id,
                source_symbol_id=file_symbol_id,
                target_symbol_id=artifact_symbol_id,
                edge_type="contains",
                weight=0.55,
            )
        )
        overview = "\n".join(
            part
            for part in [
                relative_path,
                spec.name,
                spec.description,
                "\n".join(content.splitlines()[:30]),
            ]
            if part
        )
        chunks.append(
            ChunkRecord(
                project_id=project_id,
                chunk_id=sha256(f"{relative_path}:overview".encode("utf-8")).hexdigest()[:32],
                relative_path=relative_path,
                symbol_id=artifact_symbol_id,
                symbol_name=spec.name,
                artifact_name=spec.name,
                chunk_kind="artifact-overview",
                start_line=1,
                end_line=min(30, len(content.splitlines()) or 1),
                content=overview,
                content_hash=hash_text(overview),
            )
        )
        for index, (start, end, chunk_text) in enumerate(_line_chunks(content), start=1):
            if not chunk_text:
                continue
            chunks.append(
                ChunkRecord(
                    project_id=project_id,
                    chunk_id=sha256(f"{relative_path}:{index}:{start}:{end}".encode("utf-8")).hexdigest()[:32],
                    relative_path=relative_path,
                    symbol_id=artifact_symbol_id,
                    symbol_name=spec.name,
                    artifact_name=spec.name,
                    chunk_kind="artifact",
                    start_line=start,
                    end_line=end,
                    content=chunk_text,
                    content_hash=hash_text(chunk_text),
                )
            )

    return ArtifactBundle(files=files, symbols=symbols, edges=edges, chunks=chunks)
