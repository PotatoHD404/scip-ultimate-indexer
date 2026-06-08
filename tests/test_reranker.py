from __future__ import annotations

import pytest

from ultimate_indexer.reranker import RerankItem, combine, feature_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item(
    id: str,
    name: str = "",
    signature: str = "",
    docstring: str = "",
    snippet: str = "",
    path: str = "",
    kind: str = "",
    stage1: float = 0.0,
) -> RerankItem:
    return RerankItem(
        id=id,
        name=name,
        signature=signature,
        docstring=docstring,
        snippet=snippet,
        path=path,
        kind=kind,
        stage1=stage1,
    )


# ---------------------------------------------------------------------------
# feature_scores: exact name match outranks weak overlap
# ---------------------------------------------------------------------------

class TestFeatureScores:
    def test_exact_name_match_beats_weak_overlap(self) -> None:
        """Symbol whose name exactly matches a query term should score higher."""
        items = [
            _item("exact", name="getUserProfile"),
            _item("weak", name="doSomething", docstring="user profile info"),
        ]
        scores = feature_scores("getUserProfile", items)
        assert scores["exact"] > scores["weak"], (
            f"exact={scores['exact']:.4f} should > weak={scores['weak']:.4f}"
        )

    def test_exact_name_token_match(self) -> None:
        """exact_name=1.0 path: one query term equals a name token."""
        items = [_item("a", name="fetchUser")]
        scores = feature_scores("fetch user data", items)
        # 'fetch' and 'user' are both name tokens → should yield high score
        assert scores["a"] > 0.5

    def test_text_coverage_docstring(self) -> None:
        """text_coverage picks up terms from docstring."""
        items = [
            _item("doc_rich", name="foo", docstring="handles authentication token refresh"),
            _item("no_doc", name="bar"),
        ]
        scores = feature_scores("authentication token", items)
        assert scores["doc_rich"] > scores["no_doc"]

    def test_text_coverage_signature(self) -> None:
        """text_coverage picks up terms from signature."""
        items = [
            _item("sig", name="x", signature="def process(payment_id: str) -> None"),
            _item("empty", name="y"),
        ]
        scores = feature_scores("payment process", items)
        assert scores["sig"] > scores["empty"]

    def test_name_coverage_partial(self) -> None:
        """name_coverage rewards items whose name covers many query terms."""
        items = [
            _item("full", name="getUserAccount"),   # covers 'get', 'user', 'account'
            _item("partial", name="getRecord"),     # covers only 'get'
        ]
        scores = feature_scores("get user account", items)
        assert scores["full"] > scores["partial"]

    def test_path_match(self) -> None:
        """path_match contributes score when query terms appear in the path."""
        items = [
            _item("on_path", name="handler", path="src/payments/handler.py"),
            _item("off_path", name="handler", path="src/utils/helper.py"),
        ]
        scores = feature_scores("payments", items)
        assert scores["on_path"] > scores["off_path"]

    def test_phrase_bonus_triggers(self) -> None:
        """phrase_bonus fires when the collapsed query appears in docstring/snippet."""
        items = [
            _item("phrase", name="x", docstring="this handles token refresh cycle"),
            _item("scattered", name="y", docstring="refresh: token is updated by cycle"),
        ]
        # 'token refresh' appears verbatim in phrase item
        scores = feature_scores("token refresh", items)
        assert scores["phrase"] > scores["scattered"]

    def test_scores_in_unit_interval(self) -> None:
        """All scores must be in [0, 1]."""
        items = [
            _item("a", name="processPayment", signature="def processPayment(amount: float)"),
            _item("b", name="x", path="a/b/c.py"),
            _item("c"),
        ]
        scores = feature_scores("process payment amount", items)
        for id_, score in scores.items():
            assert 0.0 <= score <= 1.0, f"id={id_}: score={score} out of range"

    def test_empty_query_returns_all_zeros(self) -> None:
        """When query has no usable terms, every score should be 0.0."""
        items = [_item("a", name="foo"), _item("b", name="bar")]
        scores = feature_scores("", items)
        assert all(v == 0.0 for v in scores.values())

    def test_stopword_only_query_returns_all_zeros(self) -> None:
        """A query made entirely of stopwords has no usable terms."""
        items = [_item("a", name="get"), _item("b", name="set")]
        scores = feature_scores("get and set", items)
        # 'and' is stopword; 'get' and 'set' are stopwords too
        assert all(v == 0.0 for v in scores.values())

    def test_short_term_query_returns_zeros(self) -> None:
        """Single-char query terms are dropped; result is all-zero."""
        items = [_item("x", name="a")]
        scores = feature_scores("a b", items)
        assert all(v == 0.0 for v in scores.values())

    def test_empty_items_returns_empty_dict(self) -> None:
        scores = feature_scores("anything", [])
        assert scores == {}

    def test_determinism(self) -> None:
        """Calling feature_scores twice with the same input gives identical output."""
        items = [
            _item("a", name="fetchData", docstring="fetch raw data from db"),
            _item("b", name="saveRecord", path="db/save.py"),
        ]
        s1 = feature_scores("fetch data", items)
        s2 = feature_scores("fetch data", items)
        assert s1 == s2

    def test_no_cross_item_normalisation(self) -> None:
        """Scores are absolute per-item; adding an unrelated item doesn't change others."""
        items_base = [_item("a", name="parseToken")]
        items_extended = [
            _item("a", name="parseToken"),
            _item("b", name="zzz_unrelated_zzz"),
        ]
        s_base = feature_scores("parse token", items_base)
        s_ext = feature_scores("parse token", items_extended)
        assert s_base["a"] == s_ext["a"]

    def test_snake_case_splitting(self) -> None:
        """snake_case identifiers are split so individual tokens are matchable."""
        items = [_item("s", name="parse_token_stream")]
        scores = feature_scores("token stream", items)
        assert scores["s"] > 0.3

    def test_camel_case_splitting(self) -> None:
        """camelCase identifiers are split so individual tokens are matchable."""
        items = [_item("c", name="parseTokenStream")]
        scores = feature_scores("parse token", items)
        assert scores["c"] > 0.3


# ---------------------------------------------------------------------------
# combine: blend logic
# ---------------------------------------------------------------------------

class TestCombine:
    def test_blend_zero_is_pure_stage1(self) -> None:
        """blend=0 → ranking determined entirely by stage1 score."""
        items = [
            _item("a", stage1=0.9),
            _item("b", stage1=0.5),
            _item("c", stage1=0.1),
        ]
        rerank = {"a": 0.1, "b": 0.5, "c": 0.9}  # inverted order
        result = combine(items, rerank, blend=0.0)
        ids = [r[0] for r in result]
        assert ids == ["a", "b", "c"]

    def test_blend_one_is_pure_rerank(self) -> None:
        """blend=1 → ranking determined entirely by rerank score."""
        items = [
            _item("a", stage1=0.9),
            _item("b", stage1=0.5),
            _item("c", stage1=0.1),
        ]
        rerank = {"a": 0.1, "b": 0.5, "c": 0.9}
        result = combine(items, rerank, blend=1.0)
        ids = [r[0] for r in result]
        assert ids == ["c", "b", "a"]

    def test_blend_half_interpolates(self) -> None:
        """blend=0.5 → both stage1 and rerank contribute equally."""
        items = [
            _item("a", stage1=1.0),
            _item("b", stage1=0.0),
        ]
        rerank = {"a": 0.0, "b": 1.0}
        result = combine(items, rerank, blend=0.5)
        # Both end up with final=0.5; stable tie-break → 'a' first (lexicographic)
        assert result[0][1] == pytest.approx(0.5)
        assert result[1][1] == pytest.approx(0.5)
        assert [r[0] for r in result] == ["a", "b"]

    def test_stable_tie_breaking_by_id(self) -> None:
        """When final scores are equal, items are ordered by id ascending."""
        items = [
            _item("charlie", stage1=0.5),
            _item("alpha", stage1=0.5),
            _item("bravo", stage1=0.5),
        ]
        rerank = {}  # all default to 0.0
        result = combine(items, rerank, blend=0.0)
        ids = [r[0] for r in result]
        assert ids == ["alpha", "bravo", "charlie"]

    def test_empty_query_combine_falls_back_to_stage1(self) -> None:
        """Empty query → all-zero rerank → combine with blend=0.5 uses stage1 order."""
        items = [
            _item("a", name="foo", stage1=0.8),
            _item("b", name="bar", stage1=0.3),
        ]
        rerank = feature_scores("", items)  # all zeros
        result = combine(items, rerank, blend=0.5)
        ids = [r[0] for r in result]
        assert ids == ["a", "b"]

    def test_missing_rerank_key_defaults_to_zero(self) -> None:
        """If an item id is absent from the rerank dict its rerank score is 0.0."""
        items = [_item("a", stage1=0.5), _item("b", stage1=0.5)]
        rerank = {"a": 0.8}  # 'b' missing
        result = combine(items, rerank, blend=1.0)
        ids = [r[0] for r in result]
        assert ids[0] == "a"

    def test_all_zero_stage1_no_division_error(self) -> None:
        """When all stage1 scores are 0, combine should not raise."""
        items = [_item("a", stage1=0.0), _item("b", stage1=0.0)]
        rerank = {"a": 0.7, "b": 0.3}
        result = combine(items, rerank, blend=0.5)
        assert result[0][0] == "a"

    def test_final_scores_reflect_blend_weight(self) -> None:
        """Verify arithmetic: final = (1-blend)*s1_norm + blend*rerank_score."""
        items = [_item("x", stage1=0.6)]
        rerank = {"x": 0.4}
        result = combine(items, rerank, blend=0.5)
        # stage1_norm = 0.6/0.6 = 1.0; final = 0.5*1.0 + 0.5*0.4 = 0.7
        assert result[0][1] == pytest.approx(0.7)

    def test_blend_out_of_range_raises(self) -> None:
        items = [_item("a")]
        with pytest.raises(ValueError, match="blend must be in"):
            combine(items, {}, blend=1.5)

    def test_combine_determinism(self) -> None:
        """Calling combine twice gives identical output."""
        items = [_item("a", stage1=0.6), _item("b", stage1=0.9), _item("c", stage1=0.3)]
        rerank = {"a": 0.8, "b": 0.2, "c": 0.5}
        r1 = combine(items, rerank, blend=0.4)
        r2 = combine(items, rerank, blend=0.4)
        assert r1 == r2

    def test_single_item(self) -> None:
        """combine with a single item should not crash and return that item."""
        items = [_item("only", stage1=0.5)]
        rerank = {"only": 0.9}
        result = combine(items, rerank, blend=0.5)
        assert len(result) == 1
        assert result[0][0] == "only"

    def test_empty_items_returns_empty(self) -> None:
        result = combine([], {}, blend=0.5)
        assert result == []
