from __future__ import annotations

import re
from typing import Iterable

# ---------------------------------------------------------------------------
# Contextual chunk embeddings
# ---------------------------------------------------------------------------
# These pure functions build a compact context header to be prepended to the
# *dense-embedding* text only.  They must NOT touch the lexical/BM25 text.
# No I/O, no side effects.
# ---------------------------------------------------------------------------


def first_doc_line(docstring: str) -> str:
    """Return the first non-empty, non-code-fence line of a docstring.

    Strips leading/trailing whitespace from each candidate line.  Lines that
    start with ``` (after stripping) are skipped.  Returns ``""`` when no
    suitable line is found.
    """
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("```"):
            return stripped
    return ""


def build_context_header(
    *,
    relative_path: str,
    kind: str = "",
    enclosing_name: str = "",
    purpose: str = "",
    neighbors: Iterable[str] = (),
    max_neighbors: int = 6,
    max_chars: int = 300,
) -> str:
    """Build a compact, single-line-ish context preface for a code/doc chunk.

    The header is intended to be prepended to the dense-embedding text so the
    embedding model understands where the chunk lives.  It is **not** suitable
    for lexical/BM25 indexing.

    Composition rules:

    - Always includes ``file: <relative_path>`` (returns ``""`` when
      *relative_path* is empty).
    - If *kind* or *enclosing_name* is non-empty, appends
      ``| <kind> <enclosing_name>`` (omitting whichever is empty).
    - If *purpose* is non-empty, appends ``| <purpose>``.
    - If *neighbors* contains non-empty strings, appends
      ``| related: n1, n2, …`` deduplicated, blank-filtered, capped at
      *max_neighbors*.
    - Internal runs of whitespace are collapsed to a single space.
    - The result is truncated to *max_chars* on a word boundary; ``" …"`` is
      appended when truncation occurs.
    """
    if not relative_path:
        return ""

    parts: list[str] = [f"file: {relative_path}"]

    # Kind / enclosing name segment
    kind_stripped = kind.strip()
    name_stripped = enclosing_name.strip()
    if kind_stripped or name_stripped:
        segment = " ".join(filter(None, [kind_stripped, name_stripped]))
        parts.append(segment)

    # Purpose segment
    purpose_stripped = purpose.strip()
    if purpose_stripped:
        parts.append(purpose_stripped)

    # Neighbors segment
    seen: set[str] = set()
    unique_neighbors: list[str] = []
    for n in neighbors:
        n_stripped = n.strip()
        if n_stripped and n_stripped not in seen:
            seen.add(n_stripped)
            unique_neighbors.append(n_stripped)
        if len(unique_neighbors) == max_neighbors:
            break
    if unique_neighbors:
        parts.append("related: " + ", ".join(unique_neighbors))

    header = " | ".join(parts)

    # Collapse internal whitespace
    header = re.sub(r"\s+", " ", header).strip()

    # Truncate to max_chars on a word boundary
    if len(header) > max_chars:
        truncated = header[:max_chars]
        # Step back to the last word boundary
        boundary = truncated.rfind(" ")
        if boundary > 0:
            truncated = truncated[:boundary]
        header = truncated + " …"

    return header


def contextual_embedding_text(content: str, context_header: str) -> str:
    """Prepend *context_header* to *content* for dense embedding.

    When *context_header* is non-empty, returns ``f"{context_header}\\n{content}"``.
    When *context_header* is empty, returns *content* unchanged.
    """
    if context_header:
        return f"{context_header}\n{content}"
    return content
