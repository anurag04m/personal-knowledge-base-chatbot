"""
query_router.py
───────────────
Intercepts user questions about modules/topics and answers them directly
from the structured sidecar JSON — bypassing FAISS and BM25 entirely.

Exposed surface
───────────────
    route_query(question, module_topics)
        → (answer: str | None, routed: bool)

If `routed` is True  → use `answer` directly, skip RAG.
If `routed` is False → fall through to the normal RAG pipeline.
"""

from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Intent detection
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that signal the user wants module/topic information
MODULE_QUERY_RE = re.compile(
    r"""
    (?:
        (?:topics?|subjects?|contents?|syllabus|units?)   # topic noun
        [\w\s]*?                                           # optional filler
        (?:in|of|for|under|covered\s+in)                  # preposition
        [\w\s]*?
        (?:module|unit|chapter)                            # module noun
    )
    |
    (?:
        (?:module|unit|chapter)                            # module first
        [\w\s]*?
        (?:topics?|subjects?|contents?|cover|include|has|have|contain)
    )
    |
    (?:
        (?:list|what\s+(?:are|is)\s+(?:the)?)             # list/what are
        [\w\s]*?
        (?:module|unit|chapter)                            # module noun
    )
    |
    (?:
        (?:module|unit|chapter)
        [\s\-–]*
        (?:[IVXivx]{1,6}|\d{1,2})                        # number right after
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Extracts the module number from a question (Roman or Arabic)
MODULE_NUMBER_RE = re.compile(
    r"(?:module|unit|chapter)[\s\-–]*([IVXivx]{1,6}|\d{1,2})",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Key matching helpers
# ─────────────────────────────────────────────────────────────────────────────

def roman_to_int(s: str) -> int:
    roman = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    s = s.upper()
    total, prev = 0, 0
    for ch in reversed(s):
        val = roman.get(ch, 0)
        total += val if val >= prev else -val
        prev = val
    return total


def normalise_number(s: str) -> str:
    """Convert any module number to a canonical comparable form (Arabic int str)."""
    s = s.strip().upper()
    if re.fullmatch(r"[IVXLCDM]+", s):
        return str(roman_to_int(s))
    try:
        return str(int(s))
    except ValueError:
        return s


def find_module_key(
    query_number: str,
    module_topics: dict[str, list[str]],
) -> Optional[str]:
    """
    Find the key in `module_topics` whose embedded number matches
    `query_number`.  Handles Roman ↔ Arabic mismatch.
    Returns None if not found.
    """
    target = normalise_number(query_number)

    for key in module_topics:
        # Extract the number embedded in the key (e.g. "Module-II: …" → "II")
        m = re.search(r"Module-([IVXivx]+|\d+)", key, re.IGNORECASE)
        if m and normalise_number(m.group(1)) == target:
            return key

    # Fuzzy fallback using difflib (handles minor typos in the key itself)
    close = get_close_matches(f"Module-{target}", list(module_topics.keys()), n=1, cutoff=0.6)
    return close[0] if close else None


# ─────────────────────────────────────────────────────────────────────────────
# Answer formatter
# ─────────────────────────────────────────────────────────────────────────────

def _format_topics_answer(module_key: str, topics: list[str]) -> str:
    if not topics:
        return (
            f"The knowledge base has an entry for **{module_key}**, but "
            f"no explicit topic list was found in the document. "
            f"Try asking a specific question about the module's content."
        )

    header = f"**{module_key}** covers the following topics:\n"
    bullets = "\n".join(f"  • {t}" for t in topics)
    return header + bullets


def _format_all_modules_answer(module_topics: dict[str, list[str]]) -> str:
    if not module_topics:
        return "No module structure was detected in the document."

    lines = ["Here is an overview of all modules:\n"]
    for key, topics in module_topics.items():
        count = len(topics)
        topic_preview = ", ".join(topics[:3])
        if count > 3:
            topic_preview += f" … (+{count - 3} more)"
        lines.append(f"**{key}** — {topic_preview}" if topics else f"**{key}** — (no topics listed)")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public router
# ─────────────────────────────────────────────────────────────────────────────

def route_query(
    question: str,
    module_topics: dict[str, list[str]],
) -> tuple[Optional[str], bool]:
    """
    Attempt to answer the question from structured module data.

    Returns
    ───────
    (answer, True)  if the question was handled here.
    (None,   False) if the question should go to the RAG pipeline.
    """
    if not module_topics:
        return None, False

    q = question.strip()

    # ── Guard: is this even a module-type question? ──────────────────────────
    if not MODULE_QUERY_RE.search(q):
        return None, False

    # ── Check for "all modules" / overview intent ────────────────────────────
    overview_re = re.compile(
        r"\b(all\s+modules?|overview|list\s+(?:all|the)\s+modules?|"
        r"how\s+many\s+modules?|modules?\s+(?:are\s+there|covered))\b",
        re.IGNORECASE,
    )
    if overview_re.search(q):
        return _format_all_modules_answer(module_topics), True

    # ── Extract the specific module number ───────────────────────────────────
    m = MODULE_NUMBER_RE.search(q)
    if not m:
        # Question matched the pattern but no number found — let RAG handle
        return None, False

    query_number = m.group(1)
    matched_key = find_module_key(query_number, module_topics)

    if matched_key is None:
        # Module not in the knowledge structure → tell user, no RAG fallback
        # We still return routed=True so the caller can show this clearly
        available = ", ".join(module_topics.keys()) or "none"
        answer = (
            f"I could not find **Module-{query_number}** in the structured "
            f"knowledge base.\n\n"
            f"Available modules: {available}\n\n"
            f"I'll now search the document text for related information."
        )
        # Return routed=False so the caller still runs RAG as fallback
        return answer, False

    topics = module_topics[matched_key]
    return _format_topics_answer(matched_key, topics), True