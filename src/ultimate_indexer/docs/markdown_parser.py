"""Parser for Markdown documentation files.

This module parses Markdown files into sections, links, and anchors, following
patterns similar to scip_parser.py. It extracts:

- Section headers with GitHub-style anchor slugs
- Cross-file links [text](other-file.md)
- Intra-file anchor links [text](#heading)
- Cross-file anchor links [text](other-file.md#heading)
- HTML anchor tags <a name="..."> and <a id="...">

The parser skips links inside code blocks and handles reference-style link definitions.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Regex patterns for Markdown parsing
MD_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+?)(?:\s+#*)?\s*$', re.MULTILINE)
MD_LINK_RE = re.compile(r'\[([^\]]*)\]\(([^)]+)\)')
MD_REF_LINK_DEF_RE = re.compile(r'^\[([^\]]+)\]:\s+(.+)$', re.MULTILINE)
MD_CODE_BLOCK_RE = re.compile(r'^```.*?^```', re.MULTILINE | re.DOTALL)
HTML_ANCHOR_RE = re.compile(r'<a\s+(?:[^>]*?\s+)?(?:name|id)\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)


def slugify_heading(text: str) -> str:
    """Convert heading text to a GitHub-style anchor slug.
    
    This matches GitHub's behavior for generating anchor IDs from headings.
    """
    # Remove inline code, bold, italic markers
    slug = re.sub(r'[`*_~]', '', text)
    # Remove HTML tags
    slug = re.sub(r'<[^>]+>', '', slug)
    slug = slug.lower().strip()
    # Remove special characters except hyphens
    slug = re.sub(r'[^\w\s-]', '', slug)
    # Replace spaces with hyphens
    slug = re.sub(r'\s+', '-', slug)
    # Collapse multiple hyphens
    slug = re.sub(r'-+', '-', slug)
    return slug.strip('-')


@dataclass(slots=True)
class SectionHeader:
    """Represents a Markdown heading."""
    level: int  # 1-6
    text: str
    anchor: str  # Generated anchor slug
    line_number: int
    file_path: str


@dataclass(slots=True)
class ParsedLink:
    """A link found during parsing, before resolution."""
    source_file: str
    source_anchor: str | None
    target_raw: str  # The raw href/ref string
    target_file: str | None  # Resolved file path (None until resolved)
    target_anchor: str | None
    link_type: str  # cross_file, intra_anchor, cross_anchor, hierarchy, openapi_ref
    context_text: str = ""  # Surrounding text for context
    
    @property
    def is_resolved(self) -> bool:
        return self.target_file is not None


@dataclass(slots=True)
class MarkdownSection:
    """A parsed section of a Markdown file."""
    header: SectionHeader | None
    level: int
    content: str
    start_line: int
    end_line: int
    anchor: str | None
    anchors_in_range: list[str] = field(default_factory=list)


class MarkdownParser:
    """Parses Markdown files into sections, links, and anchors."""

    def __init__(self, file_path: str, content: str):
        self.file_path = file_path
        self.raw_content = content
        self.lines = content.split('\n')
        self.headers: list[SectionHeader] = []
        self.links: list[ParsedLink] = []
        self.anchors: dict[str, int] = {}  # anchor -> line number
        self.ref_link_defs: dict[str, str] = {}
        self._code_block_ranges: list[tuple[int, int]] = []

    def parse(self) -> tuple[list[SectionHeader], list[ParsedLink], dict[str, int]]:
        """Full parse: headers, links, anchors."""
        self._find_code_blocks()
        self._parse_ref_link_definitions()
        self._parse_headers()
        self._parse_links()
        self._parse_html_anchors()
        return self.headers, self.links, self.anchors

    def _find_code_blocks(self):
        """Identify code block line ranges to skip during parsing."""
        in_block = False
        start = 0
        for i, line in enumerate(self.lines):
            stripped = line.strip()
            if stripped.startswith('```'):
                if not in_block:
                    in_block = True
                    start = i
                else:
                    in_block = False
                    self._code_block_ranges.append((start, i))
        # Handle unclosed code block
        if in_block:
            self._code_block_ranges.append((start, len(self.lines)))

    def _in_code_block(self, line_num: int) -> bool:
        return any(start <= line_num <= end for start, end in self._code_block_ranges)

    def _parse_ref_link_definitions(self):
        """Parse reference-style link definitions."""
        for match in MD_REF_LINK_DEF_RE.finditer(self.raw_content):
            label = match.group(1).lower()
            url = match.group(2).strip()
            self.ref_link_defs[label] = url

    def _parse_headers(self):
        """Parse Markdown headers and generate anchors."""
        # Track duplicate headers for anchor disambiguation
        anchor_counts: dict[str, int] = {}

        for i, line in enumerate(self.lines):
            if self._in_code_block(i):
                continue
            match = MD_HEADING_RE.match(line)
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                base_anchor = slugify_heading(text)

                # Handle duplicate anchors (GitHub-style: append -1, -2, etc.)
                if base_anchor in anchor_counts:
                    anchor_counts[base_anchor] += 1
                    anchor = f"{base_anchor}-{anchor_counts[base_anchor]}"
                else:
                    anchor_counts[base_anchor] = 0
                    anchor = base_anchor

                header = SectionHeader(
                    level=level,
                    text=text,
                    anchor=anchor,
                    line_number=i,
                    file_path=self.file_path,
                )
                self.headers.append(header)
                self.anchors[anchor] = i

    def _parse_links(self):
        """Parse Markdown links and classify by type."""
        for i, line in enumerate(self.lines):
            if self._in_code_block(i):
                continue
            for match in MD_LINK_RE.finditer(line):
                link_text = match.group(1)
                href = match.group(2).strip()

                # Skip external URLs, mailto, etc.
                if re.match(r'^(https?://|mailto:|ftp://)', href):
                    continue

                if href.startswith('#'):
                    # Pure anchor link within same file
                    anchor = href[1:]
                    self.links.append(ParsedLink(
                        source_file=self.file_path,
                        source_anchor=self._nearest_anchor(i),
                        target_raw=href,
                        target_file=self.file_path,
                        target_anchor=anchor,
                        link_type='intra_anchor',
                        context_text=link_text,
                    ))
                    continue

                # Parse file#anchor pattern
                target_file: str | None = None
                target_anchor: str | None = None
                link_type = 'cross_file'

                if '#' in href:
                    file_part, anchor_part = href.rsplit('#', 1)
                    target_anchor = anchor_part
                    if file_part:
                        target_file = file_part
                        link_type = 'cross_anchor'
                    else:
                        target_file = self.file_path
                        link_type = 'intra_anchor'
                else:
                    target_file = href
                    link_type = 'cross_file'

                self.links.append(ParsedLink(
                    source_file=self.file_path,
                    source_anchor=self._nearest_anchor(i),
                    target_raw=href,
                    target_file=target_file,
                    target_anchor=target_anchor,
                    link_type=link_type,
                    context_text=link_text,
                ))

    def _parse_html_anchors(self):
        """Find <a name="..."> or <a id="..."> anchors."""
        for i, line in enumerate(self.lines):
            if self._in_code_block(i):
                continue
            for match in HTML_ANCHOR_RE.finditer(line):
                anchor_name = match.group(1)
                self.anchors[anchor_name] = i

    def _nearest_anchor(self, line_num: int) -> str | None:
        """Find the nearest preceding header anchor for a given line."""
        best: str | None = None
        for header in self.headers:
            if header.line_number <= line_num:
                best = header.anchor
            else:
                break
        return best

    def get_sections(self) -> list[MarkdownSection]:
        """Split document into sections based on headers."""
        if not self.headers:
            return [MarkdownSection(
                header=None,
                level=0,
                content=self.raw_content,
                start_line=0,
                end_line=len(self.lines),
                anchor=None,
                anchors_in_range=list(self.anchors.keys()),
            )]

        sections: list[MarkdownSection] = []
        for idx, header in enumerate(self.headers):
            start = header.line_number
            end = self.headers[idx + 1].line_number if idx + 1 < len(self.headers) else len(self.lines)
            section_content = '\n'.join(self.lines[start:end]).strip()

            # Find all anchors within this section's line range
            section_anchors = [
                a for a, ln in self.anchors.items()
                if start <= ln < end
            ]

            sections.append(MarkdownSection(
                header=header,
                level=header.level,
                content=section_content,
                start_line=start,
                end_line=end,
                anchor=header.anchor,
                anchors_in_range=section_anchors,
            ))

        # Handle content before first header
        if self.headers and self.headers[0].line_number > 0:
            preamble = '\n'.join(self.lines[:self.headers[0].line_number]).strip()
            if preamble:
                preamble_anchors = [
                    a for a, ln in self.anchors.items()
                    if ln < self.headers[0].line_number
                ]
                sections.insert(0, MarkdownSection(
                    header=None,
                    level=0,
                    content=preamble,
                    start_line=0,
                    end_line=self.headers[0].line_number,
                    anchor=None,
                    anchors_in_range=preamble_anchors,
                ))

        return sections
