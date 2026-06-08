from __future__ import annotations

import pytest

from ultimate_indexer.expansions import (
    ABBREVIATIONS,
    _REVERSE,
    _STOPWORDS,
    expand_tokens,
    expansion_text,
    split_identifier,
)


# ---------------------------------------------------------------------------
# split_identifier
# ---------------------------------------------------------------------------


class TestSplitIdentifier:
    def test_camel_case(self) -> None:
        assert split_identifier("authnHandler") == ["authn", "handler"]

    def test_snake_case_with_mixed_caps(self) -> None:
        assert split_identifier("get_user_byID") == ["get", "user", "by", "id"]

    def test_all_caps_acronym_followed_by_word(self) -> None:
        assert split_identifier("HTTPServer2") == ["http", "server"]

    def test_dotted_path(self) -> None:
        assert split_identifier("pkg.services.GreetingService") == [
            "pkg",
            "services",
            "greeting",
            "service",
        ]

    def test_pascal_case(self) -> None:
        assert split_identifier("GreetingService") == ["greeting", "service"]

    def test_snake_case(self) -> None:
        assert split_identifier("get_user_id") == ["get", "user", "id"]

    def test_kebab_case(self) -> None:
        assert split_identifier("get-user-id") == ["get", "user", "id"]

    def test_all_uppercase(self) -> None:
        assert split_identifier("HTTP") == ["http"]

    def test_empty_string(self) -> None:
        assert split_identifier("") == []

    def test_pure_digit_dropped(self) -> None:
        # "Server2" → ["server"] because "2" is pure-digit
        result = split_identifier("Server2")
        assert "2" not in result
        assert "server" in result

    def test_digit_to_letter_boundary(self) -> None:
        result = split_identifier("foo2Bar")
        assert "foo" in result
        assert "bar" in result

    def test_single_word_lowercase(self) -> None:
        assert split_identifier("handler") == ["handler"]

    def test_underscores_only(self) -> None:
        assert split_identifier("___") == []

    def test_leading_trailing_underscores(self) -> None:
        assert split_identifier("__init__") == ["init"]

    def test_mixed_delimiter(self) -> None:
        result = split_identifier("my_handler-fn")
        assert result == ["my", "handler", "fn"]


# ---------------------------------------------------------------------------
# ABBREVIATIONS
# ---------------------------------------------------------------------------


class TestAbbreviations:
    def test_minimum_size(self) -> None:
        assert len(ABBREVIATIONS) >= 50

    def test_known_entries(self) -> None:
        assert "authentication" in ABBREVIATIONS["auth"]
        assert "authorization" in ABBREVIATIONS["auth"]
        assert "authentication" in ABBREVIATIONS["authn"]
        assert "authorization" in ABBREVIATIONS["authz"]
        assert "configuration" in ABBREVIATIONS["cfg"]
        assert "database" in ABBREVIATIONS["db"]
        assert "message" in ABBREVIATIONS["msg"]
        assert "request" in ABBREVIATIONS["req"]
        assert "response" in ABBREVIATIONS["resp"]
        assert "response" in ABBREVIATIONS["res"]
        assert "context" in ABBREVIATIONS["ctx"]
        assert "index" in ABBREVIATIONS["idx"]
        assert "repository" in ABBREVIATIONS["repo"]
        assert "implementation" in ABBREVIATIONS["impl"]
        assert "initialize" in ABBREVIATIONS["init"]
        assert "button" in ABBREVIATIONS["btn"]
        assert "image" in ABBREVIATIONS["img"]
        assert "function" in ABBREVIATIONS["fn"]
        assert "function" in ABBREVIATIONS["func"]
        assert "value" in ABBREVIATIONS["val"]
        assert "temporary" in ABBREVIATIONS["tmp"]
        assert "temporary" in ABBREVIATIONS["temp"]
        assert "error" in ABBREVIATIONS["err"]
        assert "address" in ABBREVIATIONS["addr"]
        assert "number" in ABBREVIATIONS["num"]
        assert "object" in ABBREVIATIONS["obj"]
        assert "service" in ABBREVIATIONS["svc"]
        assert "manager" in ABBREVIATIONS["mgr"]
        assert "utility" in ABBREVIATIONS["util"]
        assert "synchronize" in ABBREVIATIONS["sync"]
        assert "asynchronous" in ABBREVIATIONS["async"]
        assert "administrator" in ABBREVIATIONS["admin"]
        assert "administration" in ABBREVIATIONS["admin"]
        assert "permission" in ABBREVIATIONS["perm"]
        assert "organization" in ABBREVIATIONS["org"]
        assert "environment" in ABBREVIATIONS["env"]
        assert "variable" in ABBREVIATIONS["var"]
        assert "parameter" in ABBREVIATIONS["param"]
        assert "argument" in ABBREVIATIONS["arg"]
        assert "document" in ABBREVIATIONS["doc"]
        assert "documentation" in ABBREVIATIONS["doc"]
        assert "specification" in ABBREVIATIONS["spec"]
        assert "calculate" in ABBREVIATIONS["calc"]
        assert "user" in ABBREVIATIONS["usr"]
        assert "password" in ABBREVIATIONS["pwd"]
        assert "password" in ABBREVIATIONS["passwd"]
        assert "account" in ABBREVIATIONS["acct"]
        assert "quantity" in ABBREVIATIONS["qty"]
        assert "amount" in ABBREVIATIONS["amt"]
        assert "description" in ABBREVIATIONS["desc"]
        assert "command" in ABBREVIATIONS["cmd"]
        assert "connection" in ABBREVIATIONS["conn"]
        assert "execute" in ABBREVIATIONS["exec"]
        assert "generate" in ABBREVIATIONS["gen"]
        assert "delete" in ABBREVIATIONS["del"]
        assert "insert" in ABBREVIATIONS["ins"]
        assert "select" in ABBREVIATIONS["sel"]

    def test_keys_are_lowercase(self) -> None:
        for key in ABBREVIATIONS:
            assert key == key.lower(), f"Key not lowercase: {key!r}"


# ---------------------------------------------------------------------------
# expand_tokens
# ---------------------------------------------------------------------------


class TestExpandTokens:
    def test_forward_expansion(self) -> None:
        result = expand_tokens(["authn"])
        assert "authentication" in result
        assert "authn" in result

    def test_reverse_expansion(self) -> None:
        # "authentication" should pull in "authn" and "auth"
        result = expand_tokens(["authentication"])
        assert "authn" in result or "auth" in result
        assert "authentication" in result

    def test_bidirectionality_auth(self) -> None:
        forward = expand_tokens(["auth"])
        reverse = expand_tokens(["authentication"])
        # Both expansions should mention "authentication" and "auth"
        assert "authentication" in forward
        assert "auth" in reverse

    def test_originals_always_included(self) -> None:
        tokens = ["foo", "bar", "baz"]
        result = expand_tokens(tokens)
        for tok in tokens:
            assert tok in result

    def test_empty_iterable(self) -> None:
        assert expand_tokens([]) == set()

    def test_unknown_token_passes_through(self) -> None:
        result = expand_tokens(["xyzzy"])
        assert "xyzzy" in result

    def test_multiple_tokens(self) -> None:
        result = expand_tokens(["err", "ctx"])
        assert "error" in result
        assert "context" in result
        assert "err" in result
        assert "ctx" in result

    def test_cfg_expands(self) -> None:
        result = expand_tokens(["cfg"])
        assert "configuration" in result
        assert "config" in result

    def test_reverse_index_populated(self) -> None:
        # "authentication" should have reverse entries for "auth" and "authn"
        assert "auth" in _REVERSE["authentication"]
        assert "authn" in _REVERSE["authentication"]


# ---------------------------------------------------------------------------
# expansion_text
# ---------------------------------------------------------------------------


class TestExpansionText:
    def test_basic_identifier(self) -> None:
        text = expansion_text("authnHandler")
        terms = text.split()
        assert "authn" in terms
        assert "handler" in terms
        # Expansion should bring in "authentication"
        assert "authentication" in terms

    def test_empty_name_returns_empty(self) -> None:
        assert expansion_text("") == ""

    def test_stopwords_removed(self) -> None:
        text = expansion_text("getUser")
        terms = text.split()
        for stopword in ("def", "self", "return", "the", "is", "str", "int"):
            assert stopword not in terms, f"Stopword {stopword!r} leaked into output"

    def test_signature_identifiers_included(self) -> None:
        text = expansion_text("process", signature="def process(req: Request) -> Response:")
        terms = text.split()
        # "req" should appear and expand
        assert "req" in terms or "request" in terms

    def test_path_components_included(self) -> None:
        text = expansion_text("handle", path="services/auth/handler.go")
        terms = text.split()
        assert "services" in terms or "auth" in terms or "handler" in terms

    def test_extra_terms_included(self) -> None:
        text = expansion_text("connect", extra=["db", "pool"])
        terms = text.split()
        assert "db" in terms or "database" in terms

    def test_max_terms_cap(self) -> None:
        # Provide lots of input to trigger the cap
        long_name = "_".join(["word"] * 100)
        text = expansion_text(long_name, max_terms=10)
        assert len(text.split()) <= 10

    def test_dedup_no_duplicates(self) -> None:
        text = expansion_text("authHandler", signature="authHandler authn")
        terms = text.split()
        assert len(terms) == len(set(terms)), "Duplicate terms found"

    def test_all_lowercase(self) -> None:
        text = expansion_text("HTTPServer", path="pkg/HTTP/server.go")
        assert text == text.lower()

    def test_dotted_path_name(self) -> None:
        text = expansion_text("pkg.services.GreetingService")
        terms = text.split()
        assert "greeting" in terms
        assert "service" in terms

    def test_max_terms_default_is_48(self) -> None:
        # With lots of distinct words, default cap of 48 applies
        many_words = " ".join(f"word{i}" for i in range(200))
        text = expansion_text(many_words)
        assert len(text.split()) <= 48

    def test_returns_string(self) -> None:
        result = expansion_text("foo")
        assert isinstance(result, str)

    def test_short_tokens_dropped(self) -> None:
        # Single-char tokens should be filtered (len > 1)
        text = expansion_text("a")
        assert text == ""

    def test_path_stem_split(self) -> None:
        text = expansion_text("fn", path="internal/userRepo/storage.go")
        terms = text.split()
        # "userRepo" → ["user", "repo"] → "user", "repo", "repository"
        assert "user" in terms or "repo" in terms or "repository" in terms

    def test_expansion_includes_reverse(self) -> None:
        # "authentication" in extra should pull in "authn"/"auth" abbreviations
        text = expansion_text("login", extra=["authentication"])
        terms = text.split()
        assert "authn" in terms or "auth" in terms

    def test_stopword_del_not_in_output_when_stopword(self) -> None:
        # "del" is both an abbreviation key and a stopword; stopword filtering wins
        text = expansion_text("del")
        terms = text.split()
        assert "del" not in terms
