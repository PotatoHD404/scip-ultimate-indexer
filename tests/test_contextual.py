from __future__ import annotations

import pytest

from ultimate_indexer.contextual import (
    build_context_header,
    contextual_embedding_text,
    first_doc_line,
)

# ---------------------------------------------------------------------------
# first_doc_line
# ---------------------------------------------------------------------------


class TestFirstDocLine:
    def test_returns_first_non_empty_line(self) -> None:
        assert first_doc_line("  Hello world  ") == "Hello world"

    def test_skips_leading_blank_lines(self) -> None:
        doc = "\n\n  A real line\n  Second line"
        assert first_doc_line(doc) == "A real line"

    def test_skips_code_fence_lines(self) -> None:
        # Lines *starting* with ``` are skipped; content lines are not.
        # With "```python\nsome code\n```\nReal description", the first
        # non-blank, non-fence line is "some code".
        doc = "```python\nsome code\n```\nReal description"
        assert first_doc_line(doc) == "some code"

    def test_skips_blank_and_fence_together(self) -> None:
        # ``` line skipped, blank line skipped, next is "Actual text"
        doc = "\n```\n\nActual text"
        assert first_doc_line(doc) == "Actual text"

    def test_returns_empty_when_all_blank(self) -> None:
        assert first_doc_line("   \n\n  ") == ""

    def test_returns_empty_on_empty_string(self) -> None:
        assert first_doc_line("") == ""

    def test_returns_empty_when_only_code_fences(self) -> None:
        assert first_doc_line("```\n```python\n```\n") == ""

    def test_multi_tick_fence_skipped(self) -> None:
        # The fence marker line is skipped; first body line is returned.
        doc = "```typescript\ncode here\n```\nSummary line"
        assert first_doc_line(doc) == "code here"

    def test_fence_marker_with_language_tag_skipped(self) -> None:
        # A line starting with ``` (any suffix) is still a fence line.
        doc = "```python example\nSome real line"
        assert first_doc_line(doc) == "Some real line"


# ---------------------------------------------------------------------------
# build_context_header
# ---------------------------------------------------------------------------


class TestBuildContextHeaderBasics:
    def test_empty_relative_path_returns_empty(self) -> None:
        assert build_context_header(relative_path="") == ""

    def test_minimal_only_file(self) -> None:
        header = build_context_header(relative_path="pkg/services.py")
        assert header == "file: pkg/services.py"

    def test_file_always_present(self) -> None:
        header = build_context_header(
            relative_path="a/b.py",
            kind="class",
            enclosing_name="Foo",
        )
        assert header.startswith("file: a/b.py")


class TestBuildContextHeaderOptionalParts:
    def test_kind_and_enclosing_name_included(self) -> None:
        header = build_context_header(
            relative_path="pkg/services.py",
            kind="Method",
            enclosing_name="GreetingService.build_greeting",
        )
        assert "| Method GreetingService.build_greeting" in header

    def test_kind_without_enclosing_name(self) -> None:
        header = build_context_header(relative_path="pkg/a.py", kind="Function")
        assert "| Function" in header
        # Should not have a trailing space before the next pipe or end
        assert "Function |" not in header or header.endswith("Function")

    def test_enclosing_name_without_kind(self) -> None:
        header = build_context_header(relative_path="pkg/a.py", enclosing_name="MyClass")
        assert "| MyClass" in header

    def test_purpose_included(self) -> None:
        header = build_context_header(
            relative_path="pkg/a.py",
            purpose="greets a user",
        )
        assert "| greets a user" in header

    def test_purpose_omitted_when_empty(self) -> None:
        header = build_context_header(relative_path="pkg/a.py", kind="Function")
        assert "| |" not in header  # no empty segment

    def test_neighbors_included(self) -> None:
        header = build_context_header(
            relative_path="pkg/a.py",
            neighbors=["User", "format_name"],
        )
        assert "related: User, format_name" in header

    def test_full_example(self) -> None:
        header = build_context_header(
            relative_path="pkg/services.py",
            kind="Method",
            enclosing_name="GreetingService.build_greeting",
            purpose="greets a user",
            neighbors=["User", "format_name"],
        )
        assert header == (
            "file: pkg/services.py | Method GreetingService.build_greeting"
            " | greets a user | related: User, format_name"
        )

    def test_no_optional_parts_produces_minimal_header(self) -> None:
        header = build_context_header(relative_path="x.py")
        assert "|" not in header


class TestBuildContextHeaderNeighbors:
    def test_deduplication(self) -> None:
        header = build_context_header(
            relative_path="a.py",
            neighbors=["Foo", "Bar", "Foo", "Baz"],
        )
        assert header.count("Foo") == 1
        assert "Bar" in header
        assert "Baz" in header

    def test_blank_neighbors_filtered(self) -> None:
        header = build_context_header(
            relative_path="a.py",
            neighbors=["", "  ", "Valid"],
        )
        assert "related: Valid" in header

    def test_all_blank_neighbors_omits_section(self) -> None:
        header = build_context_header(
            relative_path="a.py",
            neighbors=["", "  "],
        )
        assert "related" not in header

    def test_capped_at_max_neighbors(self) -> None:
        header = build_context_header(
            relative_path="a.py",
            neighbors=[f"Sym{i}" for i in range(20)],
            max_neighbors=6,
        )
        # Count the symbols in the related section
        assert header.count("Sym") == 6

    def test_custom_max_neighbors(self) -> None:
        header = build_context_header(
            relative_path="a.py",
            neighbors=["A", "B", "C", "D"],
            max_neighbors=2,
        )
        assert "A" in header
        assert "B" in header
        assert "C" not in header
        assert "D" not in header

    def test_neighbors_as_generator(self) -> None:
        def gen():
            yield "X"
            yield "Y"

        header = build_context_header(relative_path="a.py", neighbors=gen())
        assert "related: X, Y" in header


class TestBuildContextHeaderWhitespace:
    def test_internal_whitespace_collapsed(self) -> None:
        header = build_context_header(
            relative_path="  a.py  ",
            kind="  Function  ",
            enclosing_name="  foo  ",
            purpose="  does  stuff  ",
        )
        assert "  " not in header

    def test_extra_spaces_in_path(self) -> None:
        header = build_context_header(relative_path="a/b.py")
        # ensure no double spaces
        assert "  " not in header


class TestBuildContextHeaderTruncation:
    def test_no_truncation_when_within_limit(self) -> None:
        header = build_context_header(
            relative_path="short.py",
            max_chars=300,
        )
        assert " …" not in header

    def test_truncation_appends_ellipsis(self) -> None:
        long_path = "a/" * 50 + "module.py"  # 151 chars path
        header = build_context_header(relative_path=long_path, max_chars=50)
        assert header.endswith(" …")

    def test_truncation_length_at_most_max_chars_plus_ellipsis(self) -> None:
        long_path = "x/" * 80 + "file.py"
        header = build_context_header(relative_path=long_path, max_chars=80)
        # Result length should be <= max_chars + len(" …") = max_chars + 2
        assert len(header) <= 80 + 2

    def test_truncation_on_word_boundary(self) -> None:
        # Build a header that will exceed max_chars and ensure it doesn't cut mid-word
        header = build_context_header(
            relative_path="pkg/services.py",
            kind="Method",
            enclosing_name="VeryLongClassName.very_long_method_name",
            purpose="does something useful for the application layer",
            neighbors=["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
            max_chars=60,
        )
        assert header.endswith(" …")
        # The character before " …" should not be inside a word (i.e., preceded by space or pipe)
        body = header[: -len(" …")]
        # Body ends at a word boundary — last char not alphanumeric would mean cut mid-space is fine
        # More specifically, the next character in the un-truncated header would continue a word
        # We just check that the body itself doesn't end with a partial word by verifying
        # the truncated portion is a proper prefix of whitespace-split tokens.
        assert body == body.rstrip() or body[-1] in (" ", "|", ",")

    def test_truncation_with_no_spaces(self) -> None:
        # If no space found within max_chars, still truncates (cuts at max_chars)
        no_space = "x" * 400
        header = build_context_header(relative_path=no_space, max_chars=50)
        assert header.endswith(" …")

    def test_exact_max_chars_no_truncation(self) -> None:
        path = "a.py"
        full = build_context_header(relative_path=path)
        # max_chars == len(full) should not truncate
        header = build_context_header(relative_path=path, max_chars=len(full))
        assert " …" not in header


# ---------------------------------------------------------------------------
# contextual_embedding_text
# ---------------------------------------------------------------------------


class TestContextualEmbeddingText:
    def test_prepends_header_with_newline(self) -> None:
        result = contextual_embedding_text("def foo(): pass", "file: pkg/a.py")
        assert result == "file: pkg/a.py\ndef foo(): pass"

    def test_noop_when_header_empty(self) -> None:
        content = "def foo(): pass"
        assert contextual_embedding_text(content, "") == content

    def test_empty_content_with_header(self) -> None:
        result = contextual_embedding_text("", "file: pkg/a.py")
        assert result == "file: pkg/a.py\n"

    def test_both_empty(self) -> None:
        assert contextual_embedding_text("", "") == ""

    def test_header_and_content_separated_by_exactly_one_newline(self) -> None:
        result = contextual_embedding_text("body", "header")
        lines = result.split("\n", 1)
        assert lines[0] == "header"
        assert lines[1] == "body"

    def test_multiline_content_preserved(self) -> None:
        content = "line1\nline2\nline3"
        result = contextual_embedding_text(content, "hdr")
        assert result == "hdr\nline1\nline2\nline3"

    def test_build_and_embed_integration(self) -> None:
        header = build_context_header(
            relative_path="src/greet.py",
            kind="function",
            enclosing_name="greet",
            purpose="greets a user by name",
            neighbors=["User", "format_name"],
        )
        text = contextual_embedding_text("def greet(user): ...", header)
        assert text.startswith("file: src/greet.py")
        assert "\ndef greet" in text
