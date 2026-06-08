from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable


# ---------------------------------------------------------------------------
# Identifier splitting
# ---------------------------------------------------------------------------

# Sequence of transitions that mark a word boundary inside an identifier:
#   1. End of an all-caps acronym before a capital+lower run  (e.g. HTTP|Server)
#   2. Lower-to-upper transition                              (e.g. authn|Handler)
#   3. Upper-to-lower (only when preceded by >1 uppercase)   already handled by (1)
#   4. Letter-to-digit or digit-to-letter boundary            (e.g. Server|2)
_SPLIT_RE = re.compile(
    r"""
    (?<=[A-Z])(?=[A-Z][a-z])   # e.g. HTTP|Server
    | (?<=[a-z0-9])(?=[A-Z])   # e.g. authn|Handler
    | (?<=[A-Za-z])(?=[0-9])   # e.g. Server|2
    | (?<=[0-9])(?=[A-Za-z])   # e.g. 2|Server
    """,
    re.VERBOSE,
)

# Characters that are natural delimiters (non-alphanumeric)
_DELIM_RE = re.compile(r"[^A-Za-z0-9]+")


def split_identifier(name: str) -> list[str]:
    """Split an identifier into lowercase word tokens.

    Handles camelCase, PascalCase, snake_case, kebab-case, dotted paths, and
    letter/digit boundaries.  Pure-digit tokens and empty strings are dropped.

    Examples::

        >>> split_identifier("authnHandler")
        ['authn', 'handler']
        >>> split_identifier("get_user_byID")
        ['get', 'user', 'by', 'id']
        >>> split_identifier("HTTPServer2")
        ['http', 'server']
        >>> split_identifier("pkg.services.GreetingService")
        ['pkg', 'services', 'greeting', 'service']
    """
    # First split on natural delimiters (underscores, dots, hyphens, spaces…)
    parts = _DELIM_RE.split(name)
    tokens: list[str] = []
    for part in parts:
        if not part:
            continue
        # Further split on camelCase / PascalCase / acronym transitions
        for sub in _SPLIT_RE.split(part):
            lower = sub.lower()
            # Drop pure-digit tokens and empties
            if lower and not lower.isdigit():
                tokens.append(lower)
    return tokens


# ---------------------------------------------------------------------------
# Abbreviation dictionary
# ---------------------------------------------------------------------------

ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "auth": ("authentication", "authorization"),
    "authn": ("authentication",),
    "authz": ("authorization",),
    "cfg": ("configuration", "config"),
    "conf": ("configuration", "config"),
    "config": ("configuration",),
    "db": ("database",),
    "msg": ("message",),
    "req": ("request",),
    "resp": ("response",),
    "res": ("response",),
    "ctx": ("context",),
    "idx": ("index",),
    "repo": ("repository",),
    "impl": ("implementation",),
    "init": ("initialize", "initialization"),
    "btn": ("button",),
    "img": ("image",),
    "fn": ("function",),
    "func": ("function",),
    "val": ("value",),
    "tmp": ("temporary",),
    "temp": ("temporary",),
    "err": ("error",),
    "addr": ("address",),
    "num": ("number",),
    "obj": ("object",),
    "svc": ("service",),
    "mgr": ("manager",),
    "util": ("utility",),
    "sync": ("synchronize",),
    "async": ("asynchronous",),
    "admin": ("administrator", "administration"),
    "perm": ("permission",),
    "org": ("organization",),
    "env": ("environment",),
    "var": ("variable",),
    "param": ("parameter",),
    "arg": ("argument",),
    "doc": ("document", "documentation"),
    "spec": ("specification",),
    "calc": ("calculate", "calculation"),
    "usr": ("user",),
    "pwd": ("password",),
    "passwd": ("password",),
    "acct": ("account",),
    "qty": ("quantity",),
    "amt": ("amount",),
    "desc": ("description",),
    "cmd": ("command",),
    "conn": ("connection",),
    "exec": ("execute", "execution"),
    "gen": ("generate", "generation"),
    "del": ("delete",),
    "ins": ("insert",),
    "sel": ("select",),
    "pkg": ("package",),
    "svc": ("service",),
    "srv": ("server", "service"),
    "svr": ("server",),
    "lst": ("list",),
    "arr": ("array",),
    "dict": ("dictionary",),
    "iter": ("iterator",),
    "buf": ("buffer",),
    "ptr": ("pointer",),
    "ref": ("reference",),
    "hdl": ("handler",),
    "hdr": ("header",),
    "fwd": ("forward",),
    "bck": ("backend",),
    "fe": ("frontend",),
    "be": ("backend",),
    "ui": ("interface",),
    "api": ("interface",),
}

# ---------------------------------------------------------------------------
# Reverse index: expansion word → set of abbreviations
# ---------------------------------------------------------------------------

_REVERSE: dict[str, set[str]] = defaultdict(set)
for _abbr, _expansions in ABBREVIATIONS.items():
    for _exp in _expansions:
        _REVERSE[_exp].add(_abbr)


def expand_tokens(tokens: Iterable[str]) -> set[str]:
    """Expand a collection of lowercase tokens using :data:`ABBREVIATIONS`.

    For each token, returns the token itself, any known abbreviation
    expansions, **and** any abbreviations that point back to this token
    (bi-directional lookup).  The original tokens are always included.

    Args:
        tokens: Iterable of lowercase string tokens.

    Returns:
        A set containing originals plus all bidirectional expansions.
    """
    result: set[str] = set()
    for tok in tokens:
        result.add(tok)
        # Forward: abbreviation → full words
        if tok in ABBREVIATIONS:
            result.update(ABBREVIATIONS[tok])
        # Reverse: full word → abbreviations that expand to it
        if tok in _REVERSE:
            result.update(_REVERSE[tok])
    return result


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    {
        # Python keywords / builtins that appear in signatures
        "def",
        "self",
        "cls",
        "return",
        "none",
        "true",
        "false",
        "pass",
        "raise",
        "class",
        "import",
        "from",
        "as",
        "with",
        "yield",
        "lambda",
        "global",
        "nonlocal",
        "assert",
        "del",
        # English function words
        "the",
        "a",
        "an",
        "is",
        "of",
        "to",
        "and",
        "or",
        "in",
        "on",
        "at",
        "by",
        "for",
        "not",
        "be",
        "if",
        "it",
        # Common primitive type tokens (too generic)
        "str",
        "int",
        "bool",
        "float",
        "list",
        "dict",
        "set",
        "type",
        "any",
        "void",
        "null",
    }
)

# ---------------------------------------------------------------------------
# Regex for extracting identifiers from signature text
# ---------------------------------------------------------------------------

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def expansion_text(
    name: str,
    *,
    signature: str = "",
    path: str = "",
    extra: Iterable[str] = (),
    max_terms: int = 48,
) -> str:
    """Build a deduplicated, lowercase expansion string suitable for FTS.

    Combines tokens from:

    * :func:`split_identifier` applied to *name*
    * Identifiers extracted from *signature*, each also split
    * Path stem and path component words from *path*, each split
    * *extra* tokens, each split

    All collected tokens are passed through :func:`expand_tokens` for
    abbreviation expansion, then filtered against :data:`_STOPWORDS`.
    The result is deduplicated preserving first-occurrence order and
    capped at *max_terms*.

    Args:
        name: Symbol display name (e.g. ``"authnHandler"``).
        signature: Optional type/function signature text.
        path: Optional file path (e.g. ``"services/auth/handler.go"``).
        extra: Additional raw terms (e.g. docstring keywords).
        max_terms: Maximum number of output tokens.

    Returns:
        Space-joined string of expansion terms, or ``""`` if empty.
    """
    raw_tokens: list[str] = []

    # 1. Name
    raw_tokens.extend(split_identifier(name))

    # 2. Signature identifiers
    for ident in _IDENT_RE.findall(signature):
        raw_tokens.extend(split_identifier(ident))

    # 3. Path components
    if path:
        # Split path on slashes and dots, then split each component
        for part in re.split(r"[/\\]", path):
            # Remove file extension(s)
            stem = part.split(".")[0]
            raw_tokens.extend(split_identifier(stem))

    # 4. Extra terms
    for term in extra:
        raw_tokens.extend(split_identifier(term))

    # Expand and filter stopwords
    expanded = expand_tokens(raw_tokens)
    filtered = [t for t in expanded if t not in _STOPWORDS and len(t) > 1]

    # Deduplicate preserving rough order (first occurrence from raw_tokens + expansions)
    seen: set[str] = set()
    ordered: list[str] = []
    # Walk raw_tokens first to preserve order, then remaining expansions
    for tok in raw_tokens:
        if tok not in seen and tok not in _STOPWORDS and len(tok) > 1:
            seen.add(tok)
            ordered.append(tok)
    for tok in sorted(filtered):  # deterministic ordering for expansion-only terms
        if tok not in seen:
            seen.add(tok)
            ordered.append(tok)

    capped = ordered[:max_terms]
    return " ".join(capped) if capped else ""
