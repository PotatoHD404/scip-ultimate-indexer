from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RerankItem:
    """A candidate symbol coming out of first-stage retrieval."""

    id: str
    name: str = ""
    signature: str = ""
    docstring: str = ""
    snippet: str = ""
    path: str = ""
    kind: str = ""
    stage1: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "in", "on", "at", "to", "of", "for",
        "and", "or", "not", "is", "it", "be", "as", "by", "with",
        "this", "that", "from", "are", "was", "were", "has", "have",
        "do", "did", "get", "set",
    }
)

_CAMEL_SPLIT_RE = re.compile(
    r"""
    (?<=[a-z0-9])(?=[A-Z])   # lower/digit → upper
    | (?<=[A-Z])(?=[A-Z][a-z]) # e.g. HTTPServer → HTTP + Server
    """,
    re.VERBOSE,
)


def _tokenize_query(query: str) -> list[str]:
    """Return lowercase query tokens, dropping stopwords and <2-char terms.

    Query terms are split on identifier boundaries (snake_case / camelCase) the
    same way symbol names are, so a query like ``build_greeting`` yields
    ``["build", "greeting"]`` and matches a symbol named ``build_greeting``.
    """
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in re.split(r"\W+", query.lower()):
        for sub in _split_identifier(raw):
            if len(sub) >= 2 and sub not in _STOPWORDS and sub not in seen:
                seen.add(sub)
                tokens.append(sub)
    return tokens


def _split_identifier(name: str) -> list[str]:
    """Split a camelCase / snake_case / PascalCase identifier into lowercase tokens."""
    # First break on snake_case / dot / path separators
    parts = re.split(r"[_.\-/\\]+", name)
    tokens: list[str] = []
    for part in parts:
        # Then split camelCase
        for segment in _CAMEL_SPLIT_RE.split(part):
            t = segment.lower()
            if t:
                tokens.append(t)
    return tokens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def feature_scores(query: str, items: list[RerankItem]) -> dict[str, float]:
    """Return a deterministic relevance score in [0, 1] per item.id.

    Scores are absolute per-item values; they are *not* normalised across the
    candidate set so that they remain stable regardless of the other candidates
    present.

    Parameters
    ----------
    query:
        The user query string.
    items:
        Candidate symbols to score.

    Returns
    -------
    dict mapping item.id → float score in [0, 1].
    """
    q_terms = _tokenize_query(query)

    if not q_terms:
        return {item.id: 0.0 for item in items}

    q_set = set(q_terms)
    q_phrase = " ".join(q_terms)  # normalised phrase for phrase_bonus

    result: dict[str, float] = {}

    for item in items:
        name_tokens = _split_identifier(item.name)
        name_token_set = set(name_tokens)

        # ---- exact_name ------------------------------------------------
        # Full credit if any query term equals a name token exactly.
        # Partial credit (0.5 * best substring ratio) otherwise.
        if q_set & name_token_set:
            exact_name = 1.0
        else:
            # partial: best fraction of a query term found as substring of name (lc)
            name_lc = item.name.lower()
            best = 0.0
            for t in q_terms:
                if t in name_lc:
                    # award partial proportional to term length / name length
                    best = max(best, len(t) / max(len(name_lc), 1))
            exact_name = 0.5 * min(best, 1.0)

        # ---- name_coverage ---------------------------------------------
        # Fraction of query terms that appear in the name token set.
        name_coverage = len(q_set & name_token_set) / len(q_set)

        # ---- text_coverage ---------------------------------------------
        # Fraction of DISTINCT query terms found anywhere in the combined text.
        combined_text = " ".join(
            [item.name, item.signature, item.docstring, item.snippet, item.path]
        ).lower()
        text_hits = sum(1 for t in q_set if t in combined_text)
        text_coverage = text_hits / len(q_set)

        # ---- path_match ------------------------------------------------
        path_tokens = set(re.split(r"[/\\._ -]+", item.path.lower())) - {""}
        path_match = len(q_set & path_tokens) / len(q_set)

        # ---- phrase_bonus ----------------------------------------------
        long_text = (item.docstring + " " + item.snippet).lower()
        phrase_bonus = 1.0 if q_phrase in long_text else 0.0

        # ---- combine with fixed weights --------------------------------
        score = (
            0.35 * exact_name
            + 0.30 * text_coverage
            + 0.20 * name_coverage
            + 0.10 * path_match
            + 0.05 * phrase_bonus
        )
        result[item.id] = max(0.0, min(1.0, score))

    return result


def combine(
    items: list[RerankItem],
    rerank: dict[str, float],
    *,
    blend: float = 0.5,
) -> list[tuple[str, float]]:
    """Blend first-stage and rerank scores and return sorted (id, final_score) pairs.

    Parameters
    ----------
    items:
        Candidate symbols with their ``stage1`` scores.
    rerank:
        Per-id rerank scores as returned by :func:`feature_scores`.
    blend:
        Weight given to the rerank score (``0`` = pure stage-1 order,
        ``1`` = pure rerank order).  Must be in ``[0, 1]``.

    Returns
    -------
    List of ``(id, final_score)`` sorted by ``final_score`` descending,
    with ``id`` ascending as a stable tie-breaker.
    """
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"blend must be in [0, 1], got {blend!r}")

    # Normalise stage1 scores to [0, 1]
    max_stage1 = max((item.stage1 for item in items), default=0.0)
    if max_stage1 <= 0.0:
        stage1_norms: dict[str, float] = {item.id: 0.0 for item in items}
    else:
        stage1_norms = {item.id: item.stage1 / max_stage1 for item in items}

    scored: list[tuple[str, float]] = []
    for item in items:
        s1 = stage1_norms[item.id]
        s2 = rerank.get(item.id, 0.0)
        final = (1.0 - blend) * s1 + blend * s2
        scored.append((item.id, final))

    # Primary sort: final descending; secondary: id ascending (stable tie-break)
    scored.sort(key=lambda t: (-t[1], t[0]))
    return scored
