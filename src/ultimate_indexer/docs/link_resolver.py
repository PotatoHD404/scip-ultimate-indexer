"""Cross-file and intra-file link resolution for documentation.

This module resolves raw parsed links to concrete chunk-to-chunk edges,
enabling graph-based navigation between related documentation sections.
It handles:

- Cross-file links: [text](other-file.md)
- Intra-file anchor links: [text](#heading)
- Cross-file anchor links: [text](other-file.md#heading)
- OpenAPI $ref links
- Parent-child hierarchy relationships
- Sequential chunk relationships

The module follows patterns similar to the edge resolution in scip_parser.py
and produces EdgeRecord-compatible output.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .chunker import DocumentChunk
from .markdown_parser import ParsedLink, SectionHeader

logger = logging.getLogger(__name__)


class LinkResolver:
    """Resolves raw parsed links to concrete chunk-to-chunk edges."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir).resolve()
        # file_path -> list of chunks
        self.file_chunks: dict[str, list[DocumentChunk]] = {}
        # file_path -> {anchor: line_number}
        self.file_anchors: dict[str, dict[str, int]] = {}
        # Normalized path mapping for resolution
        self._path_index: dict[str, str] = {}

    def register_file(
        self,
        file_path: str,
        chunks: list[DocumentChunk],
        anchors: dict[str, int],
    ):
        """Register a parsed file's chunks and anchors."""
        norm_path = self._normalize_path(file_path)
        self.file_chunks[norm_path] = chunks
        self.file_anchors[norm_path] = anchors

        # Build path index with various resolution forms
        p = Path(norm_path)
        self._path_index[norm_path] = norm_path
        # Also index without extension for flexibility
        self._path_index[str(p.with_suffix(''))] = norm_path
        # Index by filename only
        self._path_index[p.name] = norm_path
        self._path_index[p.stem] = norm_path

    def resolve_links(
        self,
        links: list[ParsedLink],
        source_file: str,
        link_weights: dict[str, float],
    ) -> list[dict[str, Any]]:
        """Resolve parsed links into graph edges between chunks.
        
        Returns a list of edge dictionaries compatible with EdgeRecord.
        """
        edges: list[dict[str, Any]] = []
        source_norm = self._normalize_path(source_file)

        for link in links:
            try:
                edge = self._resolve_single_link(link, source_norm, link_weights)
                if edge:
                    edges.append(edge)
            except Exception as e:
                logger.debug(f"Failed to resolve link {link.target_raw}: {e}")

        return edges

    def _resolve_single_link(
        self,
        link: ParsedLink,
        source_norm: str,
        weights: dict[str, float],
    ) -> dict[str, Any] | None:
        """Resolve a single link to a graph edge."""
        # Find source chunk
        source_chunk = self._find_chunk_by_anchor(
            source_norm, link.source_anchor
        )
        if not source_chunk:
            # Fall back to first chunk of the file
            source_chunks = self.file_chunks.get(source_norm, [])
            if source_chunks:
                source_chunk = source_chunks[0]
            else:
                return None

        # Resolve target file
        target_file_norm = self._resolve_file_path(
            link.target_file, source_norm
        )
        if not target_file_norm:
            logger.debug(
                f"Cannot resolve target file '{link.target_file}' "
                f"from '{source_norm}'"
            )
            return None

        # Find target chunk
        if link.target_anchor:
            target_chunk = self._find_chunk_by_anchor(
                target_file_norm, link.target_anchor
            )
        else:
            # Link to file without anchor -> first chunk
            target_chunks = self.file_chunks.get(target_file_norm, [])
            target_chunk = target_chunks[0] if target_chunks else None

        if not target_chunk:
            logger.debug(
                f"Cannot find target chunk for anchor '{link.target_anchor}' "
                f"in '{target_file_norm}'"
            )
            return None

        if source_chunk.chunk_id == target_chunk.chunk_id:
            return None  # No self-loops

        link_type = link.link_type or 'cross_file'
        weight = weights.get(link_type, 1.0)

        return {
            'source_chunk_id': source_chunk.chunk_id,
            'target_chunk_id': target_chunk.chunk_id,
            'edge_type': link_type,
            'weight': weight,
            'context': link.context_text,
        }

    def _find_chunk_by_anchor(
        self,
        file_path: str,
        anchor: str | None,
    ) -> DocumentChunk | None:
        """Find the chunk containing a specific anchor."""
        chunks = self.file_chunks.get(file_path, [])
        if not chunks:
            return None

        if not anchor:
            return chunks[0]

        # Direct anchor match
        for chunk in chunks:
            if anchor == chunk.anchor or anchor in chunk.anchors:
                return chunk

        # Fuzzy anchor match (handle slight slug differences)
        anchor_lower = anchor.lower().replace('-', '').replace('_', '')
        for chunk in chunks:
            chunk_anchors = [chunk.anchor] + chunk.anchors if chunk.anchor else chunk.anchors
            for ca in chunk_anchors:
                if ca and ca.lower().replace('-', '').replace('_', '') == anchor_lower:
                    return chunk

        # Fall back: use line number from file's anchor registry
        file_anchors = self.file_anchors.get(file_path, {})
        if anchor in file_anchors:
            target_line = file_anchors[anchor]
            # Find chunk containing this line
            for chunk in chunks:
                if chunk.start_line <= target_line < chunk.end_line:
                    return chunk
            # If line-based match fails, find nearest chunk
            best_chunk: DocumentChunk | None = None
            best_dist = float('inf')
            for chunk in chunks:
                dist = abs(chunk.start_line - target_line)
                if dist < best_dist:
                    best_dist = dist
                    best_chunk = chunk
            return best_chunk

        return None

    def _resolve_file_path(
        self,
        target_file: str | None,
        source_file: str,
    ) -> str | None:
        """Resolve a target file reference to a normalized path."""
        if not target_file:
            return source_file

        # Remove any anchor from the file path
        clean_target = target_file.split('#')[0] if '#' in target_file else target_file
        if not clean_target:
            return source_file

        # Try direct match in index
        if clean_target in self._path_index:
            return self._path_index[clean_target]

        # Try relative to source file
        source_dir = str(Path(source_file).parent)
        relative = os.path.normpath(os.path.join(source_dir, clean_target))
        if relative in self._path_index:
            return self._path_index[relative]

        # Try relative to base dir
        from_base = os.path.normpath(os.path.join(str(self.base_dir), clean_target))
        if from_base in self._path_index:
            return self._path_index[from_base]

        # Try normalized form
        norm = self._normalize_path(clean_target)
        if norm in self._path_index:
            return self._path_index[norm]

        # Try with common extensions
        for ext in ('.md', '.markdown', '.yaml', '.yml', '.json'):
            for candidate in [clean_target + ext, relative + ext, norm + ext]:
                if candidate in self._path_index:
                    return self._path_index[candidate]

        return None

    def _normalize_path(self, path: str) -> str:
        """Normalize a file path for consistent indexing."""
        p = Path(path)
        if p.is_absolute():
            try:
                return str(p.resolve().relative_to(self.base_dir))
            except ValueError:
                return str(p.resolve())
        return os.path.normpath(path)

    def build_hierarchy_edges(
        self,
        file_path: str,
        chunks: list[DocumentChunk],
        hierarchy_weight: float,
    ) -> list[dict[str, Any]]:
        """Build parent-child hierarchy edges between chunks."""
        edges: list[dict[str, Any]] = []
        if len(chunks) < 2:
            return edges

        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                parent = chunks[i]
                child = chunks[j]

                if not parent.headers or not child.headers:
                    continue

                parent_level = parent.headers[-1].level if parent.headers else 0
                child_level = child.headers[-1].level if child.headers else 0

                # Direct parent-child: child is one level deeper
                if child_level == parent_level + 1:
                    edges.append({
                        'source_chunk_id': parent.chunk_id,
                        'target_chunk_id': child.chunk_id,
                        'edge_type': 'hierarchy',
                        'weight': hierarchy_weight,
                    })
                elif child_level <= parent_level:
                    # Stop looking for children when we hit a sibling/uncle
                    break

        # Sequential edges
        for i in range(len(chunks) - 1):
            edges.append({
                'source_chunk_id': chunks[i].chunk_id,
                'target_chunk_id': chunks[i + 1].chunk_id,
                'edge_type': 'sequence',
                'weight': hierarchy_weight * 0.5,
            })

        return edges
