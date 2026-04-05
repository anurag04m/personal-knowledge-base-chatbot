"""
query_router.py  (v2)
──────────────────────
Intercepts structured questions about modules before RAG runs.

Changes from v1
───────────────
• Added "how many modules" / count intent handler  →  fixes Issue 2
• Extended MODULE_QUERY_RE to catch more phrasings including count queries
• Added is_followup() helper for the caller to detect vague follow-ups
• All logic is pure Python (zero LLM calls, zero latency)
"""

from __future__ import annotations

import re
from difflib import get_close_matches
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Intent patterns
# ─────────────────────────────────────────────────────────────────────────────

MODULE_QUERY_RE = re.compile(
    r"""
    # A: "topics in module X", "subjects covered in unit 3"
    (?:
        (?:topics?|subjects?|contents?|syllabus|units?|chapters?)
        [\w\s,]*?
        (?:in|of|for|under|covered\s+in|inside)
        [\w\s]*?
        (?:module|unit|chapter)
    )
    |
    # B: "module X topics/covers/has"
    (?:
        (?:module|unit|chapter)
        [\w\s]*?
        (?:topics?|subjects?|contents?|cover|include|has|have|contain|teach)
    )
    |
    # C: "list / what are the topics of module X"
    (?:
        (?:list|show|give|tell\s+me|what\s+(?:are|is)\s+(?:the)?)
        [\w\s]*?
        (?:module|unit|chapter)
    )
    |
    # D: bare "module X" with a number right after
    (?:
        (?:module|unit|chapter)
        [\s\-–]*
        (?:[IVXivx]{1,6}|\d{1,2})
        \b
    )
    |
    # E: "how many modules", "number of modules", "total modules"
    (?:
        (?:how\s+many|number\s+of|total(?:\s+number\s+of)?|count\s+(?:of\s+)?)
        [\w\s]*?
        (?:modules?|units?|chapters?)
    )
    |
    # F: "modules in this file/document", "are there X modules"
    (?:
        (?:modules?|units?|chapters?)
        [\w\s]*?
        (?:in\s+(?:this|the)\s+(?:file|document|pdf|book|notes?)|are\s+there)
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Counts specifically
COUNT_QUERY_RE = re.compile(
    r"""
    (?:how\s+many|number\s+of|total(?:\s+number\s+of)?|count\s+(?:of\s+)?)
    [\w\s]*?(?:modules?|units?|chapters?)
    |
    (?:modules?|units?|chapters?)[\w\s]*?
    (?:are\s+there|exist|present|available|in\s+(?:this|the)\s+(?:file|document|pdf|book))
    """,
    re.VERBOSE | re.IGNORECASE,
)

MODULE_NUMBER_RE = re.compile(
    r"(?:module|unit|chapter)[\s\-–]*([IVXivx]{1,6}|\d{1,2})\b",
    re.IGNORECASE,
)

OVERVIEW_RE = re.compile(
    r"\b(all\s+modules?|overview|list\s+(?:all|the)\s+modules?|"
    r"modules?\s+(?:are\s+there|covered|available|present)|"
    r"entire\s+syllabus|full\s+syllabus)\b",
    re.IGNORECASE,
)

# Vague follow-up detector (no LLM needed)
FOLLOWUP_PRONOUN_RE = re.compile(
    r"^(?:(?:and|also|but|so)\s+)?"
    r"(?:(?:is|are|was|were|does|do|did|has|have|had|can|could|would|should)\s+)?"
    r"(?:it|this|that|they|them|these|those|he|she|its)\b",
    re.IGNORECASE,
)

FOLLOWUP_SHORT_STARTERS = re.compile(
    r"^(?:how|why|when|where|what about|tell me more|elaborate|explain more|"
    r"can you explain|and how|and why|and what|explain it)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
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
    s = s.strip().upper()
    if re.fullmatch(r"[IVXLCDM]+", s):
        return str(roman_to_int(s))
    try:
        return str(int(s))
    except ValueError:
        return s


def find_module_key(query_number: str, module_topics: dict) -> Optional[str]:
    target = normalise_number(query_number)
    for key in module_topics:
        m = re.search(r"Module-([IVXivx]+|\d+)", key, re.IGNORECASE)
        if m and normalise_number(m.group(1)) == target:
            return key
    close = get_close_matches(f"Module-{target}", list(module_topics.keys()), n=1, cutoff=0.6)
    return close[0] if close else None


def is_followup(question: str) -> bool:
    """
    Heuristic: True if the question is a vague follow-up that likely
    refers to the previous topic rather than introducing a new one.
    The caller uses this to decide whether to reuse cached context.
    """
    q = question.strip()

    # Starts with a pronoun → strong follow-up signal
    if FOLLOWUP_PRONOUN_RE.match(q):
        return True

    # Starts with a short follow-up phrase
    if FOLLOWUP_SHORT_STARTERS.match(q):
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────────────

def _format_topics_answer(module_key: str, topics: list[str]) -> str:
    if not topics:
        return (
            f"**{module_key}** was detected in the document, but no explicit "
            f"topic list was found under it. Try asking a specific question "
            f"about the module's content."
        )
    bullets = "\n".join(f"  • {t}" for t in topics)
    return f"**{module_key}** covers the following topics:\n{bullets}"


def _format_count_answer(module_topics: dict) -> str:
    count = len(module_topics)
    if count == 0:
        return "No module structure was detected in the document."
    keys_list = "\n".join(f"  {i+1}. {k}" for i, k in enumerate(module_topics.keys()))
    plural = "s" if count != 1 else ""
    return (
        f"There {'is' if count == 1 else 'are'} **{count} module{plural}** "
        f"in this document:\n{keys_list}"
    )


def _format_all_modules_answer(module_topics: dict) -> str:
    if not module_topics:
        return "No module structure was detected in the document."
    lines = [f"This document has **{len(module_topics)} module(s)**:\n"]
    for key, topics in module_topics.items():
        preview = ", ".join(topics[:3])
        if len(topics) > 3:
            preview += f" … (+{len(topics) - 3} more)"
        lines.append(f"**{key}** — " + (preview if topics else "(no topics listed)"))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Public router
# ─────────────────────────────────────────────────────────────────────────────

def route_query(
    question: str,
    module_topics: dict[str, list[str]],
) -> tuple[Optional[str], bool]:
    """
    Returns (answer, routed).
    routed=True  → answer is complete, skip RAG.
    routed=False → answer is None or a hint; RAG should still run.
    """
    if not module_topics:
        return None, False

    q = question.strip()

    if not MODULE_QUERY_RE.search(q):
        return None, False

    # Count intent — no number in question
    if COUNT_QUERY_RE.search(q) and not MODULE_NUMBER_RE.search(q):
        return _format_count_answer(module_topics), True

    # Overview intent
    if OVERVIEW_RE.search(q) and not MODULE_NUMBER_RE.search(q):
        return _format_all_modules_answer(module_topics), True

    # Specific module
    m = MODULE_NUMBER_RE.search(q)
    if not m:
        return None, False

    matched_key = find_module_key(m.group(1), module_topics)
    if matched_key is None:
        available = ", ".join(module_topics.keys()) or "none"
        hint = (
            f"**Module-{m.group(1)}** was not found in the structured index.\n"
            f"Available: {available}\n\n"
            f"Searching document text for related information…"
        )
        return hint, False

    return _format_topics_answer(matched_key, module_topics[matched_key]), True