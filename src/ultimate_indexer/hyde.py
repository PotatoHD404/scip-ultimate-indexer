from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

STOPWORDS: frozenset[str] = frozenset(
    {
        # Articles / determiners
        "the", "a", "an", "this", "that", "these", "those", "its", "their",
        # Prepositions
        "of", "to", "in", "on", "for", "with", "at", "by", "from", "about",
        "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further",
        # Conjunctions
        "and", "or", "but", "nor", "so", "yet",
        # Pronouns
        "it", "we", "our", "my", "i", "you", "he", "she", "they", "them",
        "your", "his", "her", "us",
        # Auxiliaries / copulas
        "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "have", "has", "had",
        "can", "could", "will", "would", "should", "shall", "may", "might",
        # Common question / intent words (kept as stopwords for keyword extraction,
        # but also tested in the NL-detection heuristic below)
        "how", "what", "where", "why", "when", "which", "who",
        "does", "is", "are",
        # Filler verbs that don't carry domain meaning
        "get", "set", "use", "show", "find", "make", "let", "take", "give",
        "implement", "handle", "work", "done", "used",
        # Misc high-frequency words
        "as", "if", "then", "else", "not", "no", "all", "any", "each",
        "also", "more", "very", "just", "like", "up", "new",
    }
)

# ---------------------------------------------------------------------------
# Intent / question words used for NL detection
# ---------------------------------------------------------------------------

_INTENT_WORDS: frozenset[str] = frozenset(
    {
        "how", "what", "where", "why", "when", "which", "who",
        "does", "is", "are", "can", "find", "show", "handle",
        "implement", "use", "work",
    }
)

# Matches a single "identifier-or-dotted-path" token (no spaces).
# e.g. build_greeting, GreetingService, pkg.services.Foo, User
_IDENT_RE = re.compile(r"^[\w][\w.]*$")


def looks_like_natural_language(query: str) -> bool:
    """Return True when *query* reads like a natural-language question/phrase.

    Heuristic rules (applied in order):

    1. If the query is a single token with no whitespace that looks like an
       identifier or dotted path → **False**.
    2. If it ends with ``?`` → **True**.
    3. Count whitespace-separated tokens:
       - 0 tokens after strip → **False**.
       - 1 token → **False** (single symbol or word).
       - 2 tokens → **True** only when at least one token is a stopword/
         intent word, otherwise **False**.
       - 3+ tokens → **True**.
    4. Presence of any intent/question word anywhere → **True**.
    """
    stripped = query.strip()
    if not stripped:
        return False

    # Rule 1 – single token, no internal spaces
    if " " not in stripped and "\t" not in stripped:
        if _IDENT_RE.match(stripped):
            return False

    # Rule 2 – ends with "?"
    if stripped.endswith("?"):
        return True

    tokens = stripped.split()
    n = len(tokens)

    if n <= 1:
        return False

    lower_tokens = {t.lower() for t in tokens}

    # Rule 4 – intent word present anywhere (applies to 2+ token queries)
    if lower_tokens & _INTENT_WORDS:
        return True

    # Rule 3 – 2-token case: need a stopword
    if n == 2:
        return bool(lower_tokens & STOPWORDS)

    # 3+ tokens without any intent/stopword match → still NL
    return True


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[^\w]+")


def keywords(query: str) -> list[str]:
    """Return lowercase content tokens from *query* with STOPWORDS removed.

    Tokens shorter than 2 characters are dropped.  Order is preserved and
    duplicates are removed (first occurrence wins).
    """
    raw_tokens = _TOKEN_RE.split(query.strip().lower())
    seen: set[str] = set()
    result: list[str] = []
    for tok in raw_tokens:
        if len(tok) < 2:
            continue
        if tok in STOPWORDS:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        result.append(tok)
    return result


# ---------------------------------------------------------------------------
# Identifier helpers
# ---------------------------------------------------------------------------

def _to_snake_case(words: list[str]) -> str:
    """Join *words* with underscores."""
    return "_".join(words)


def _to_camel_case(words: list[str]) -> str:
    """Return lowerCamelCase of *words*."""
    if not words:
        return ""
    return words[0] + "".join(w.capitalize() for w in words[1:])


# ---------------------------------------------------------------------------
# Hypothetical code generation
# ---------------------------------------------------------------------------

def hypothetical_code(query: str, *, language_hint: str = "python") -> str:  # noqa: ARG001
    """Generate a deterministic pseudo-code snippet from *query*.

    The produced text contains snake_case and camelCase identifier variants
    derived from the query keywords so that its embedding lands close to real
    code implementing those concepts.

    The *language_hint* parameter is accepted for future extension but
    currently always emits Python-shaped text.

    Returns the original query unchanged when no keywords can be extracted.
    """
    kws = keywords(query)
    if not kws:
        return query

    # Function name: snake_case of up to 4 keywords
    fn_kws = kws[:4]
    fn_name = _to_snake_case(fn_kws)

    # Build identifier variants for consecutive pairs of keywords
    pair_variants: list[str] = []
    for i in range(len(kws) - 1):
        pair = kws[i : i + 2]
        pair_variants.append(_to_snake_case(pair))
        pair_variants.append(_to_camel_case(pair))

    # Primary service / manager name derived from all keywords
    all_snake = _to_snake_case(kws)
    all_camel = _to_camel_case(kws)

    # Build the snippet line by line for full determinism
    lines: list[str] = []
    lines.append(f"def {fn_name}(request):")
    lines.append(f'    """')
    lines.append(f"    {query}")
    lines.append(f'    """')

    # Comment listing raw keywords + identifier variants
    kw_comment = ", ".join(kws)
    variant_comment = ", ".join(pair_variants[:6]) if pair_variants else all_snake
    lines.append(f"    # keywords: {kw_comment}")
    lines.append(f"    # identifiers: {variant_comment}")

    # Plausible statements
    lines.append(f"    {all_snake}_service = {all_camel}Service(request)")
    lines.append(f"    result = {all_snake}_service.handle(request)")
    lines.append("    return result")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HyDE entry point
# ---------------------------------------------------------------------------

def hyde_query_text(
    query: str,
    *,
    enabled: bool = True,
    language_hint: str = "python",
) -> str:
    """Return the text to embed for *query*, optionally augmented with HyDE.

    When *enabled* is True and *query* looks like natural language, returns::

        <original query>\\n<hypothetical_code(query)>

    Otherwise returns *query* unchanged.
    """
    if enabled and looks_like_natural_language(query):
        return query + "\n" + hypothetical_code(query, language_hint=language_hint)
    return query
