"""Smart document chunking with context preservation.

This module provides intelligent chunking of parsed documentation sections,
preserving context through breadcrumbs, header hierarchies, and overlapping
content for better retrieval quality.

The chunker follows patterns similar to the fallback.py module but adds
documentation-specific features like:
- Breadcrumb generation from header hierarchy
- Header context preservation in each chunk
- Smart splitting that respects paragraph boundaries
- Configurable chunk sizes based on token estimates
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .markdown_parser import SectionHeader

logger = logging.getLogger(__name__)

# Default configuration constants
DEFAULT_MAX_CHUNK_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 64
DEFAULT_MIN_CHUNK_TOKENS = 50
CHARS_PER_TOKEN = 4.0  # Approximate token-to-char ratio


@dataclass(slots=True)
class DocumentChunk:
    """A retrieval-ready chunk of documentation content.
    
    This is designed to be compatible with the existing ChunkRecord model
    in models.py, with additional fields for document-specific context.
    """
    chunk_id: str
    file_path: str
    chunk_type: str  # markdown_section, markdown_text, openapi_endpoint, openapi_schema
    content: str
    breadcrumb: str = ""  # e.g., "API Guide > Authentication > OAuth2"
    headers: list[SectionHeader] = ()  # type: ignore
    metadata: dict[str, Any] = ()  # type: ignore
    start_line: int = 0
    end_line: int = 0
    anchor: str | None = None
    anchors: list[str] = ()  # type: ignore
    
    def __post_init__(self):
        if self.headers is None:
            object.__setattr__(self, 'headers', [])
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})
        if self.anchors is None:
            object.__setattr__(self, 'anchors', [])
    
    @property
    def token_estimate(self) -> int:
        return len(self.content) // CHARS_PER_TOKEN


class DocumentChunker:
    """Chunks parsed document sections into retrieval-ready pieces."""

    def __init__(
        self,
        max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        min_chunk_tokens: int = DEFAULT_MIN_CHUNK_TOKENS,
        include_breadcrumb: bool = True,
    ):
        self.max_chars = int(max_chunk_tokens * CHARS_PER_TOKEN)
        self.overlap_chars = int(overlap_tokens * CHARS_PER_TOKEN)
        self.min_chars = int(min_chunk_tokens * CHARS_PER_TOKEN)
        self.include_breadcrumb = include_breadcrumb

    def chunk_sections(
        self,
        file_path: str,
        sections: list[dict[str, Any]],
    ) -> list[DocumentChunk]:
        """Chunk a list of parsed sections into DocumentChunks."""
        chunks: list[DocumentChunk] = []

        # Build header hierarchy for breadcrumbs
        header_stack: list[SectionHeader] = []

        for section in sections:
            header: SectionHeader | None = section.get('header')
            level = section.get('level', 0)
            content = section.get('content', '')
            anchor = section.get('anchor')
            anchors = section.get('anchors_in_range', [])
            chunk_type = section.get('chunk_type', 'markdown_section')
            metadata = section.get('metadata', {})

            if not content.strip():
                continue

            # Update header stack for breadcrumb
            if header:
                while header_stack and header_stack[-1].level >= level:
                    header_stack.pop()
                header_stack.append(header)

            breadcrumb = self._build_breadcrumb(header_stack) if self.include_breadcrumb else ""

            # Determine headers list for this chunk
            headers_list = list(header_stack)

            if len(content) <= self.max_chars:
                # Content fits in one chunk
                chunk = DocumentChunk(
                    chunk_id="",  # Will be auto-generated later
                    file_path=file_path,
                    chunk_type=chunk_type,
                    content=content,
                    breadcrumb=breadcrumb,
                    headers=headers_list,
                    metadata=metadata,
                    anchor=anchor,
                    anchors=anchors,
                    start_line=section.get('start_line', 0),
                    end_line=section.get('end_line', 0),
                )
                chunks.append(chunk)
            else:
                # Need to split into sub-chunks
                sub_chunks = self._split_content(
                    content=content,
                    file_path=file_path,
                    chunk_type=chunk_type,
                    breadcrumb=breadcrumb,
                    headers=headers_list,
                    metadata=metadata,
                    anchor=anchor,
                    anchors=anchors,
                    start_line=section.get('start_line', 0),
                    end_line=section.get('end_line', 0),
                )
                chunks.extend(sub_chunks)

        return chunks

    def _split_content(
        self,
        content: str,
        file_path: str,
        chunk_type: str,
        breadcrumb: str,
        headers: list[SectionHeader],
        metadata: dict,
        anchor: str | None,
        anchors: list[str],
        start_line: int,
        end_line: int,
    ) -> list[DocumentChunk]:
        """Split oversized content into overlapping chunks."""
        chunks: list[DocumentChunk] = []
        paragraphs = content.split('\n\n')

        current_parts: list[str] = []
        current_len = 0
        chunk_index = 0

        for para in paragraphs:
            para_len = len(para)

            if current_len + para_len + 2 > self.max_chars and current_parts:
                # Emit current chunk
                chunk_content = '\n\n'.join(current_parts)
                chunk = DocumentChunk(
                    chunk_id="",
                    file_path=file_path,
                    chunk_type=chunk_type,
                    content=chunk_content,
                    breadcrumb=breadcrumb,
                    headers=headers,
                    metadata={**metadata, 'sub_chunk': chunk_index},
                    anchor=anchor if chunk_index == 0 else None,
                    anchors=anchors if chunk_index == 0 else [],
                    start_line=start_line,
                    end_line=end_line,
                )
                chunks.append(chunk)
                chunk_index += 1

                # Keep overlap: take last portion
                overlap_parts: list[str] = []
                overlap_len = 0
                for p in reversed(current_parts):
                    if overlap_len + len(p) <= self.overlap_chars:
                        overlap_parts.insert(0, p)
                        overlap_len += len(p) + 2
                    else:
                        break
                current_parts = overlap_parts
                current_len = overlap_len

            current_parts.append(para)
            current_len += para_len + 2

        # Emit remaining
        if current_parts:
            remaining = '\n\n'.join(current_parts)
            if len(remaining) >= self.min_chars or not chunks:
                chunks.append(DocumentChunk(
                    chunk_id="",
                    file_path=file_path,
                    chunk_type=chunk_type,
                    content=remaining,
                    breadcrumb=breadcrumb,
                    headers=headers,
                    metadata={**metadata, 'sub_chunk': chunk_index},
                    anchor=anchor if chunk_index == 0 and not chunks else None,
                    anchors=[],
                    start_line=start_line,
                    end_line=end_line,
                ))
            elif chunks:
                # Append to last chunk if too small
                last = chunks[-1]
                object.__setattr__(last, 'content', last.content + '\n\n' + remaining)

        return chunks

    def _build_breadcrumb(self, header_stack: list[SectionHeader]) -> str:
        """Build a breadcrumb string from the header hierarchy."""
        if not header_stack:
            return ""
        return " > ".join(h.text for h in header_stack)
