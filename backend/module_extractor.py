"""
module_extractor.py
───────────────────
Structured knowledge layer for the Personal Knowledge Base Chatbot.
Extracts module-wise topics from PDF text during ingestion and stores
them as a JSON sidecar file that the chatbot queries directly.

Design goals
─────────────
• Zero hallucination  — we only emit topics that are textually present.
• No hardcoding       — module patterns and topic heuristics are driven
                        by regex, not static lists.
• Scalable            — works on large PDFs; O(n) over lines.
• LLM-optional        — pure regex/heuristic pass runs first; an optional
                        LLM refinement step can be enabled with one flag.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Constants / tunables
# ─────────────────────────────────────────────────────────────────────────────

# Matches headings like:
#   MODULE-I   Module 1   MODULE – III   Module IV   UNIT-2   UNIT II
MODULE_HEADING_RE = re.compile(
    r"""
    ^\s*                            # optional leading whitespace
    (?:MODULE|UNIT|CHAPTER)         # keyword
    [\s\-–—]*                       # separator
    (
        [IVXivx]{1,6}               # Roman numerals   I  II  III  IV …
        |
        \d{1,2}                     # Arabic digits    1  2  3 …
    )
    [\s:\-–—]*                      # optional trailing separator
    (.*)                            # optional inline title after the number
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Lines that are almost certainly "topics" / sub-headings:
#   - Short (≤ 12 words)
#   - Title-cased or ALL-CAPS
#   - Not a sentence (no terminal period followed by space and more text)
#   - Not a page-number artefact (pure digits / roman numerals alone)

TOPIC_MIN_WORDS = 2
TOPIC_MAX_WORDS = 12

# Bullet / list markers that prefix topic lines
BULLET_RE = re.compile(r"^[\s]*[•\-–—*►▶◆○●\u25cf\u2022\uf0b7]+[\s]+")

# Lines to discard: page numbers, running headers, URL-only lines, very short
NOISE_RE = re.compile(
    r"""
    ^\s*\d+\s*$                  # bare page number
    | ^\s*[ivxlcdmIVXLCDM]+\s*$ # bare roman numeral
    | https?://\S+               # URL
    | ^.{0,3}$                   # very short (≤ 3 chars)
    """,
    re.VERBOSE,
)

# Sentence-like lines (likely paragraph body, not a heading/topic)
SENTENCE_RE = re.compile(r"\.\s+[A-Z]")  # period + space + capital inside text


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalise_text(text: str) -> str:
    """Unicode-normalise, collapse whitespace, strip."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def roman_to_int(s: str) -> int:
    """Convert a Roman numeral string to int (e.g. 'IV' → 4)."""
    roman = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    s = s.upper()
    total, prev = 0, 0
    for ch in reversed(s):
        val = roman.get(ch, 0)
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total


def canonical_module_key(raw_number: str, inline_title: str) -> str:
    """
    Produce a stable, human-readable key such as 'Module-I' or 'Module-3'.
    We prefer Roman numerals if that's what the source uses; otherwise keep
    the digit form.
    """
    raw_number = raw_number.strip()
    if re.fullmatch(r"[IVXivx]+", raw_number):
        num_str = raw_number.upper()          # keep Roman
    else:
        num_str = str(int(raw_number))        # normalise leading zeros

    title_part = normalise_text(inline_title)
    if title_part:
        return f"Module-{num_str}: {title_part}"
    return f"Module-{num_str}"


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction
# ─────────────────────────────────────────────────────────────────────────────

def is_likely_topic(line: str) -> bool:
    """
    Heuristic: is this line a topic/sub-heading rather than body text?

    Rules (ALL must hold):
    1. Not noise (page numbers, URLs, stubs)
    2. Not a full sentence
    3. Word count in [TOPIC_MIN_WORDS, TOPIC_MAX_WORDS]
    4. Line is title-cased, ALL-CAPS, or starts with a bullet marker
    """
    clean = BULLET_RE.sub("", line).strip()   # strip bullet prefix
    clean = normalise_text(clean)

    if not clean:
        return False
    if NOISE_RE.search(clean):
        return False
    if SENTENCE_RE.search(clean):
        return False
    if re.search(r"\bpage\s+\d+\b", clean, re.IGNORECASE):
        return False
    if re.search(r"[=→]", clean):  # equations / arrows
        return False
    if re.search(r"\b[Pp]\d+\b", clean):  # P0, P1 etc (tables)
        return False
    if re.search(r"\b\d+\b", clean) and len(clean.split()) <= 4:
        return False  # short numeric-heavy lines
    # ❌ Filter long descriptive lines (likely sentences)
    if len(clean.split()) > 8:
        return False
    # ❌ Filter lines containing common verbs (not headings)
    if re.search(r"\b(is|are|was|were|be|being|been|have|has|had|can|could|should|will|would)\b", clean, re.IGNORECASE):
        return False
    # ❌ Lines with colon followed by long explanation
    if ":" in clean and len(clean.split()) > 6:
        return False
    # ❌ Table-like rows (A B C etc.)
    if re.fullmatch(r"[A-Z\s]{3,}", clean):
        return False

    words = clean.split()
    if not (TOPIC_MIN_WORDS <= len(words) <= TOPIC_MAX_WORDS):
        return False

    # Casing check: title-case, ALL-CAPS, or starts with a number (numbered list)
    is_title = clean.istitle()
    is_upper = clean.isupper() and 2 <= len(clean.split()) <= 5
    starts_numbered = re.match(r"^\d+[\.\)]\s+\S", clean)
    has_bullet = bool(BULLET_RE.match(line))

    return bool(is_title or is_upper or starts_numbered or has_bullet)


def extract_module_topics_from_text(full_text: str) -> dict[str, list[str]]:
    """
    Parse `full_text` (entire concatenated PDF text) and return a dict:
        { "Module-I": ["Topic A", "Topic B"], ... }

    Algorithm
    ─────────
    1. Split into lines.
    2. Scan for MODULE_HEADING_RE lines → module boundary markers.
    3. Between consecutive boundaries, collect candidate topic lines via
       `is_likely_topic`.
    4. De-duplicate and return.
    """
    lines = full_text.splitlines()

    # ── Pass 1: locate module boundaries ────────────────────────────────────
    boundaries: list[tuple[int, str]] = []   # (line_index, module_key)

    for idx, raw_line in enumerate(lines):
        line = normalise_text(raw_line)
        m = MODULE_HEADING_RE.match(line)
        if m:
            num_str, inline_title = m.group(1), m.group(2) or ""
            key = canonical_module_key(num_str, inline_title)
            boundaries.append((idx, key))

    if not boundaries:
        return {}

    # ── Pass 2: for each module span, collect topics ─────────────────────────
    module_topics: dict[str, list[str]] = {}

    for span_idx, (start_line, module_key) in enumerate(boundaries):
        # The span ends just before the next module heading (or EOF)
        end_line = (
            boundaries[span_idx + 1][0]
            if span_idx + 1 < len(boundaries)
            else len(lines)
        )

        seen: set[str] = set()
        topics: list[str] = []

        for line in lines[start_line + 1: end_line]:
            candidate = normalise_text(BULLET_RE.sub("", line))
            # Strip leading numbering like "1." "2)" etc.
            candidate = re.sub(r"^\d+[\.\)]\s+", "", candidate)

            if not candidate or candidate in seen:
                continue

            if is_likely_topic(line):
                seen.add(candidate)
                topics.append(candidate)

        module_topics[module_key] = topics

    return module_topics


# ─────────────────────────────────────────────────────────────────────────────
# LLM refinement (optional)
# ─────────────────────────────────────────────────────────────────────────────

def refine_with_llm(
    module_topics: dict[str, list[str]],
    raw_text_by_module: dict[str, str],
    model: str = "llama3",
) -> dict[str, list[str]]:
    """
    For modules where regex found 0 topics, ask the LLM to extract them
    from the raw text chunk.  We still constrain it tightly to avoid
    hallucination.

    Only imported/called when use_llm=True in extract_and_save().
    """
    try:
        import ollama
    except ImportError:
        print("[module_extractor] ollama not available; skipping LLM refinement.")
        return module_topics

    refined = dict(module_topics)

    for key, topics in module_topics.items():
        if topics:          # regex already found something → trust it
            continue

        raw_chunk = raw_text_by_module.get(key, "")
        if not raw_chunk.strip():
            continue

        prompt = f"""
You are a precise information extractor.
Below is a passage from an educational document under the heading "{key}".

Your task: list ONLY the topic names or sub-headings that are EXPLICITLY
present in the text. Do NOT invent, infer, or paraphrase topics.

Rules:
- Output one topic per line.
- If no clear topics are present, output exactly: NONE
- Do not add explanations.

Text:
\"\"\"
{raw_chunk[:2000]}
\"\"\"

Topics:
"""
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            content = resp["message"]["content"].strip()
            if content.upper() == "NONE" or not content:
                continue
            extracted = [
                normalise_text(l.lstrip("-•*► ").strip())
                for l in content.splitlines()
                if l.strip() and l.strip().upper() != "NONE"
            ]
            refined[key] = extracted
        except Exception as exc:
            print(f"[module_extractor] LLM call failed for {key}: {exc}")

    return refined


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_save(
    documents: list,                  # list of LangChain Document objects
    output_path: str = "module_topics.json",
    use_llm: bool = False,
    llm_model: str = "llama3",
) -> dict[str, list[str]]:
    """
    Entry point called from ingest.py.

    Parameters
    ──────────
    documents   : LangChain Document list (from PyPDFLoader).
    output_path : where to save the JSON sidecar.
    use_llm     : if True, use LLM to fill in modules with 0 regex topics.
    llm_model   : Ollama model name for LLM refinement.

    Returns the module→topics dict.
    """
    # Concatenate all page text in order
    full_text = "\n".join(doc.page_content for doc in documents)

    module_topics = extract_module_topics_from_text(full_text)

    if use_llm and module_topics:
        # Build per-module raw text for LLM context
        lines = full_text.splitlines()
        boundaries: list[tuple[int, str]] = []
        for idx, raw_line in enumerate(lines):
            m = MODULE_HEADING_RE.match(normalise_text(raw_line))
            if m:
                key = canonical_module_key(m.group(1), m.group(2) or "")
                boundaries.append((idx, key))

        raw_text_by_module: dict[str, str] = {}
        for i, (start, key) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(lines)
            raw_text_by_module[key] = "\n".join(lines[start:end])

        module_topics = refine_with_llm(module_topics, raw_text_by_module, llm_model)

    # Persist
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(module_topics, fh, indent=2, ensure_ascii=False)

    total_topics = sum(len(v) for v in module_topics.values())
    print(
        f"[module_extractor] Found {len(module_topics)} module(s), "
        f"{total_topics} topic(s) → saved to '{output_path}'"
    )
    return module_topics


def load_module_topics(path: str = "module_topics.json") -> dict[str, list[str]]:
    """Load the JSON sidecar produced during ingestion."""
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)