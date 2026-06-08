from __future__ import annotations

import re

import pytest

from ultimate_indexer.hyde import (
    STOPWORDS,
    hypothetical_code,
    hyde_query_text,
    keywords,
    looks_like_natural_language,
)


# ---------------------------------------------------------------------------
# looks_like_natural_language
# ---------------------------------------------------------------------------

class TestLooksLikeNaturalLanguage:
    # --- bare symbols / identifiers → False ---

    def test_single_snake_case_identifier(self):
        assert looks_like_natural_language("build_greeting") is False

    def test_single_pascal_case(self):
        assert looks_like_natural_language("GreetingService") is False

    def test_single_word(self):
        assert looks_like_natural_language("User") is False

    def test_dotted_path(self):
        assert looks_like_natural_language("pkg.services.Foo") is False

    def test_package_qualified_symbol(self):
        assert looks_like_natural_language("http.Handler") is False

    # --- natural language questions → True ---

    def test_ends_with_question_mark(self):
        assert looks_like_natural_language("how does auth work?") is True

    def test_how_question(self):
        assert looks_like_natural_language("how is authentication handled") is True

    def test_what_question(self):
        assert looks_like_natural_language("what does the login flow do") is True

    def test_where_question(self):
        assert looks_like_natural_language("where is the token validated") is True

    def test_find_intent(self):
        assert looks_like_natural_language("find all database connections") is True

    def test_three_plain_words(self):
        # 3+ whitespace tokens without question word → still NL
        assert looks_like_natural_language("database connection pool") is True

    def test_intent_word_triggers_true(self):
        assert looks_like_natural_language("handle user errors") is True

    # --- 2-token edge cases ---

    def test_two_tokens_with_stopword(self):
        assert looks_like_natural_language("how authentication") is True

    def test_two_tokens_no_stopword(self):
        # Two content words, no stopword/intent word → False
        assert looks_like_natural_language("database migration") is False

    def test_two_tokens_question_mark(self):
        assert looks_like_natural_language("auth?") is True  # ends with ?

    # --- edge inputs ---

    def test_empty_string(self):
        assert looks_like_natural_language("") is False

    def test_whitespace_only(self):
        assert looks_like_natural_language("   ") is False

    def test_single_char(self):
        assert looks_like_natural_language("x") is False


# ---------------------------------------------------------------------------
# STOPWORDS
# ---------------------------------------------------------------------------

class TestStopwords:
    def test_is_frozenset(self):
        assert isinstance(STOPWORDS, frozenset)

    def test_common_words_present(self):
        for word in ("the", "a", "is", "are", "how", "of", "to", "in"):
            assert word in STOPWORDS, f"'{word}' missing from STOPWORDS"

    def test_domain_keywords_absent(self):
        for word in ("authentication", "token", "database", "service", "handler"):
            assert word not in STOPWORDS, f"'{word}' should not be in STOPWORDS"


# ---------------------------------------------------------------------------
# keywords
# ---------------------------------------------------------------------------

class TestKeywords:
    def test_stopwords_removed(self):
        result = keywords("how is authentication handled")
        assert "how" not in result
        assert "is" not in result
        # "handled" is not a stopword — it should be kept
        assert "handled" in result
        assert "authentication" in result

    def test_order_preserved(self):
        result = keywords("token validation service")
        assert result == ["token", "validation", "service"]

    def test_deduplication(self):
        result = keywords("auth auth authentication")
        counts = {tok: result.count(tok) for tok in result}
        for tok, cnt in counts.items():
            assert cnt == 1, f"'{tok}' appears {cnt} times"

    def test_short_tokens_dropped(self):
        result = keywords("a go db service")
        # "a" dropped (stopword + length), "go" kept (≥2 chars, not stopword)
        # "db" kept, "service" kept
        assert "a" not in result
        assert "db" in result
        assert "service" in result

    def test_lowercased(self):
        result = keywords("Authentication TOKEN Service")
        assert all(tok == tok.lower() for tok in result)

    def test_empty_query(self):
        assert keywords("") == []

    def test_only_stopwords(self):
        assert keywords("how is the") == []

    def test_punctuation_split(self):
        # Non-word chars act as token separators
        result = keywords("validate.token.request")
        assert "validate" in result
        assert "token" in result
        assert "request" in result


# ---------------------------------------------------------------------------
# hypothetical_code
# ---------------------------------------------------------------------------

class TestHypotheticalCode:
    def test_contains_snake_case_from_keywords(self):
        kws = keywords("how is authentication handled")
        code = hypothetical_code("how is authentication handled")
        snake = "_".join(kws[:4])
        assert snake in code

    def test_contains_camel_case_variant(self):
        code = hypothetical_code("authentication token validation")
        kws = keywords("authentication token validation")
        # camelCase of first pair
        camel = kws[0] + kws[1].capitalize()
        assert camel in code

    def test_contains_docstring_with_query(self):
        query = "how is authentication handled"
        code = hypothetical_code(query)
        assert query in code

    def test_is_deterministic(self):
        query = "how does the login service validate tokens"
        first = hypothetical_code(query)
        second = hypothetical_code(query)
        assert first == second

    def test_different_queries_differ(self):
        a = hypothetical_code("authentication flow")
        b = hypothetical_code("database connection pool")
        assert a != b

    def test_empty_keywords_returns_query_unchanged(self):
        # Query consisting entirely of stopwords → no keywords → return as-is
        query = "how is the"
        assert hypothetical_code(query) == query

    def test_function_def_present(self):
        code = hypothetical_code("token validation service")
        assert code.startswith("def ")

    def test_service_handle_pattern(self):
        code = hypothetical_code("token validation service")
        assert "service.handle(" in code or "_service" in code

    def test_single_keyword_query(self):
        # One meaningful keyword — should still produce code
        code = hypothetical_code("authentication")
        assert "authentication" in code
        assert "def " in code

    def test_pair_variants_in_comment(self):
        code = hypothetical_code("token validation service")
        # At least one pair variant should appear in the identifiers comment
        assert "identifiers:" in code

    def test_language_hint_parameter_accepted(self):
        # language_hint accepted without error (currently always Python output)
        code = hypothetical_code("authentication service", language_hint="go")
        assert "def " in code


# ---------------------------------------------------------------------------
# hyde_query_text
# ---------------------------------------------------------------------------

class TestHydeQueryText:
    def test_nl_query_augmented_when_enabled(self):
        query = "how is authentication handled"
        result = hyde_query_text(query)
        assert result.startswith(query + "\n")
        # Code part present
        assert "def " in result

    def test_passthrough_when_disabled(self):
        query = "how is authentication handled"
        result = hyde_query_text(query, enabled=False)
        assert result == query

    def test_bare_symbol_not_augmented(self):
        query = "build_greeting"
        result = hyde_query_text(query)
        assert result == query

    def test_dotted_path_not_augmented(self):
        query = "pkg.services.Foo"
        result = hyde_query_text(query)
        assert result == query

    def test_deterministic(self):
        query = "how does the authentication service work"
        assert hyde_query_text(query) == hyde_query_text(query)

    def test_result_contains_original_query(self):
        query = "where is the token validated"
        result = hyde_query_text(query)
        assert query in result

    def test_language_hint_propagated(self):
        query = "how is authentication handled"
        result = hyde_query_text(query, language_hint="go")
        # Currently always python-shaped, but should not error
        assert "def " in result

    def test_empty_query_passthrough(self):
        # Empty → not NL → return unchanged
        result = hyde_query_text("")
        assert result == ""
