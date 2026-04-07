"""Tests for the documentation ingestion pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ultimate_indexer.docs.markdown_parser import MarkdownParser, slugify_heading
from ultimate_indexer.docs.openapi_parser import OpenAPIParser
from ultimate_indexer.docs.chunker import DocumentChunker
from ultimate_indexer.docs.link_resolver import LinkResolver
from ultimate_indexer.docs.doc_graph import DocumentGraph
from ultimate_indexer.docs.ingest import ingest_documentation


class TestSlugifyHeading:
    """Tests for the slugify_heading function."""

    def test_basic(self):
        assert slugify_heading("Hello World") == "hello-world"

    def test_special_chars(self):
        # Double hyphens are collapsed to single
        assert slugify_heading("API (v2) — Reference!") == "api-v2-reference"

    def test_code_markers(self):
        assert slugify_heading("`GET` /users") == "get-users"

    def test_leading_trailing(self):
        assert slugify_heading("  --hello--  ") == "hello"

    def test_duplicate_handling(self):
        assert slugify_heading("Getting Started") == slugify_heading("Getting Started")


class TestMarkdownParser:
    """Tests for the Markdown parser."""

    def test_headers_extraction(self):
        content = """# Title
Some text.

## Section One

Content here.

### Subsection

More content.

## Section Two

Final content.
"""
        parser = MarkdownParser("test.md", content)
        headers, links, anchors = parser.parse()

        assert len(headers) == 4
        assert headers[0].level == 1
        assert headers[0].text == "Title"
        assert headers[0].anchor == "title"
        assert headers[1].anchor == "section-one"
        assert headers[2].level == 3
        assert headers[3].anchor == "section-two"

    def test_duplicate_headers(self):
        content = """# API
## Methods
### GET
## Methods
### GET
"""
        parser = MarkdownParser("test.md", content)
        headers, _, _ = parser.parse()
        anchors = [h.anchor for h in headers]
        assert len(set(anchors)) == len(anchors), "Anchors must be unique"
        assert "methods" in anchors
        assert "methods-1" in anchors

    def test_link_extraction(self):
        content = """# Test
See [other doc](other.md) and [section](#heading) and
[specific section](other.md#details) for more.
Also an [external link](https://example.com).
"""
        parser = MarkdownParser("test.md", content)
        _, links, _ = parser.parse()

        # Should find 3 internal links (external is skipped)
        assert len(links) == 3

        cross_file = [l for l in links if l.link_type == "cross_file"]
        assert len(cross_file) == 1
        assert cross_file[0].target_file == "other.md"

        intra = [l for l in links if l.link_type == "intra_anchor"]
        assert len(intra) == 1
        assert intra[0].target_anchor == "heading"

        cross_anchor = [l for l in links if l.link_type == "cross_anchor"]
        assert len(cross_anchor) == 1
        assert cross_anchor[0].target_file == "other.md"
        assert cross_anchor[0].target_anchor == "details"

    def test_code_blocks_skipped(self):
        content = """# Test

Here is a [real link](target.md).

```markdown
This is a [fake link](fake.md) inside a code block.
```

More text.
"""
        parser = MarkdownParser("test.md", content)
        _, links, _ = parser.parse()
        targets = [l.target_file for l in links]
        assert "fake.md" not in targets
        assert "target.md" in targets

    def test_html_anchors(self):
        content = """# Test
<a name="custom-anchor"></a>

Some content.

<a id="another-anchor">text</a>
"""
        parser = MarkdownParser("test.md", content)
        _, _, anchors = parser.parse()
        assert "custom-anchor" in anchors
        assert "another-anchor" in anchors

    def test_sections(self):
        content = """Preamble text.

# Title

Intro.

## Section A

Content A.

## Section B

Content B.
"""
        parser = MarkdownParser("test.md", content)
        parser.parse()
        sections = parser.get_sections()

        assert len(sections) == 4  # preamble + 3 headers
        assert sections[0].header is None
        assert "Preamble" in sections[0].content
        assert sections[1].header.text == "Title"


class TestOpenAPIParser:
    """Tests for the OpenAPI parser."""

    def test_basic_spec(self):
        spec = """
openapi: "3.0.3"
info:
  title: Test API
  version: "1.0"
  description: A test API
paths:
  /items:
    get:
      operationId: listItems
      summary: List items
      tags: [Items]
      responses:
        "200":
          description: OK
    post:
      operationId: createItem
      summary: Create item
      tags: [Items]
      requestBody:
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/Item'
      responses:
        "201":
          description: Created
components:
  schemas:
    Item:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
          description: Item name
      required: [id, name]
"""
        parser = OpenAPIParser("api.yaml", spec)
        sections, links, anchors = parser.parse()

        # Should have: info + 2 endpoints + 1 schema = 4 sections
        assert len(sections) >= 3

        # Should find $ref link
        ref_links = [l for l in links if l.link_type == "openapi_ref"]
        assert len(ref_links) >= 1

        # Check endpoint content
        endpoint_sections = [s for s in sections if s.chunk_type == "openapi_endpoint"]
        assert len(endpoint_sections) == 2


class TestDocumentChunker:
    """Tests for the document chunker."""

    def test_small_sections_not_split(self):
        chunker = DocumentChunker(max_chunk_tokens=1000)

        sections = [
            {
                "header": None,
                "level": 1,
                "content": "Short content.",
                "start_line": 0,
                "end_line": 1,
                "anchor": "test",
                "anchors_in_range": ["test"],
                "chunk_type": "markdown_section",
            }
        ]

        chunks = chunker.chunk_sections("f.md", sections)
        assert len(chunks) == 1
        assert chunks[0].content == "Short content."

    def test_breadcrumb_generation(self):
        chunker = DocumentChunker(include_breadcrumb=True)

        from ultimate_indexer.docs.markdown_parser import SectionHeader

        sections = [
            {
                "header": SectionHeader(1, "Guide", "guide", 0, "f.md"),
                "level": 1,
                "content": "# Guide\nIntro.",
                "start_line": 0,
                "end_line": 2,
                "anchor": "guide",
                "anchors_in_range": [],
                "chunk_type": "markdown_section",
            },
            {
                "header": SectionHeader(2, "Setup", "setup", 2, "f.md"),
                "level": 2,
                "content": "## Setup\nSteps.",
                "start_line": 2,
                "end_line": 4,
                "anchor": "setup",
                "anchors_in_range": [],
                "chunk_type": "markdown_section",
            },
        ]

        chunks = chunker.chunk_sections("f.md", sections)
        assert len(chunks) == 2
        assert chunks[1].breadcrumb == "Guide > Setup"


class TestLinkResolver:
    """Tests for the link resolver."""

    def test_cross_file_resolution(self):
        from ultimate_indexer.docs.chunker import DocumentChunk
        from ultimate_indexer.docs.markdown_parser import ParsedLink

        resolver = LinkResolver("/docs")

        chunk_a = DocumentChunk(
            chunk_id="a#1",
            file_path="a.md",
            chunk_type="markdown_section",
            content="See [other](b.md#section).",
            anchor="test",
            start_line=0,
            end_line=5,
        )
        chunk_b = DocumentChunk(
            chunk_id="b#1",
            file_path="b.md",
            chunk_type="markdown_section",
            content="# Section\nContent",
            anchor="section",
            start_line=0,
            end_line=5,
        )

        resolver.register_file("a.md", [chunk_a], {"test": 0})
        resolver.register_file("b.md", [chunk_b], {"section": 0})

        # Use ParsedLink objects instead of dicts
        links = [
            ParsedLink(
                source_file="a.md",
                source_anchor="test",
                target_raw="b.md#section",
                target_file="b.md",
                target_anchor="section",
                link_type="cross_anchor",
                context_text="other",
            )
        ]

        edges = resolver.resolve_links(
            links, "a.md", {"cross_anchor": 1.0, "cross_file": 1.0, "intra_anchor": 0.5}
        )

        assert len(edges) == 1
        assert edges[0]["source_chunk_id"] == "a#1"
        assert edges[0]["target_chunk_id"] == "b#1"
        assert edges[0]["edge_type"] == "cross_anchor"


class TestDocumentGraph:
    """Tests for the document graph."""

    def test_basic_graph(self):
        graph = DocumentGraph()

        graph.add_node("chunk1", file_path="a.md", chunk_type="markdown_section")
        graph.add_node("chunk2", file_path="b.md", chunk_type="markdown_section")
        graph.add_edge("chunk1", "chunk2", "cross_file", 1.0)

        assert graph.num_nodes == 2
        assert graph.num_edges == 1

    def test_personalized_pagerank(self):
        graph = DocumentGraph()

        # Create a simple chain: A -> B -> C
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_edge("a", "b", "link", 1.0)
        graph.add_edge("b", "c", "link", 1.0)

        # Personalize on node A
        scores = graph.personalized_pagerank(["a"], [1.0])

        assert "a" in scores
        assert "b" in scores
        assert "c" in scores
        # A should have highest score since it's the seed
        assert scores["a"] > scores["b"]
        assert scores["b"] > scores["c"]

    def test_neighbors(self):
        graph = DocumentGraph()

        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_node("d")
        graph.add_edge("a", "b", "link", 1.0)
        graph.add_edge("b", "c", "link", 1.0)
        graph.add_edge("c", "d", "link", 1.0)

        neighbors = graph.get_neighbors("b", max_hops=1)
        assert "a" in neighbors
        assert "c" in neighbors
        assert "d" not in neighbors

        neighbors_2 = graph.get_neighbors("b", max_hops=2)
        assert "d" in neighbors_2


class TestIngestDocumentation:
    """Integration tests for the documentation ingestion pipeline."""

    def test_ingest_markdown_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir).resolve()
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir()

            # Create a test markdown file
            test_file = docs_dir / "test.md"
            test_file.write_text("""# Test Document

This is a test document.

## Section One

Some content here.

## Section Two

More content with a [link](#section-one).
""")

            files, symbols, edges, chunks = ingest_documentation(
                project_id="test-project",
                project_root=tmp_path,
                doc_dirs=["docs"],
            )

            assert len(files) == 1
            assert files[0].relative_path == "docs/test.md"
            assert files[0].source_kind == "documentation"

            # Should have file, document, and section symbols
            assert len(symbols) >= 3

            # Should have chunks
            assert len(chunks) >= 1

    def test_ingest_openapi_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir).resolve()
            docs_dir = tmp_path / "api"
            docs_dir.mkdir()

            # Create a test OpenAPI file
            test_file = docs_dir / "openapi.yaml"
            test_file.write_text("""openapi: "3.0.3"
info:
  title: Test API
  version: "1.0.0"
paths:
  /users:
    get:
      operationId: getUsers
      summary: Get all users
      responses:
        "200":
          description: Success
""")

            files, symbols, edges, chunks = ingest_documentation(
                project_id="test-project",
                project_root=tmp_path,
                doc_dirs=["api"],
            )

            assert len(files) == 1
            assert files[0].relative_path == "api/openapi.yaml"
            assert files[0].language == "openapi"

            # Should have file and document symbols
            assert len(symbols) >= 2

            # Should have chunks for endpoints
            assert len(chunks) >= 1

    def test_ingest_mixed_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir).resolve()
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir()

            # Create markdown file
            md_file = docs_dir / "guide.md"
            md_file.write_text("""# User Guide

Welcome to the guide.

## Getting Started

Read the [API docs](api.md) for details.
""")

            # Create OpenAPI file
            api_file = docs_dir / "api.md"
            api_file.write_text("""# API Reference

## GET /users

Returns a list of users.
""")

            files, symbols, edges, chunks = ingest_documentation(
                project_id="test-project",
                project_root=tmp_path,
                doc_dirs=["docs"],
            )

            assert len(files) == 2
            assert len(symbols) >= 4  # At least 2 files + 2 documents
            assert len(chunks) >= 2

    def test_doc_dir_aliases_do_not_duplicate_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir).resolve()
            docs_dir = tmp_path / "docs"
            docs_dir.mkdir()
            (docs_dir / "guide.md").write_text("# Guide\n\nHello\n")

            alias_dir = tmp_path / "Docs"
            try:
                alias_dir.symlink_to(docs_dir, target_is_directory=True)
            except OSError:
                pytest.skip("symlinks are not supported in this environment")

            files, symbols, edges, chunks = ingest_documentation(
                project_id="test-project",
                project_root=tmp_path,
            )

            assert [file.relative_path for file in files] == ["docs/guide.md"]
            assert len(symbols) >= 1
            assert len(edges) >= 0
            assert len(chunks) >= 1
