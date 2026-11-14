"""Measures of a filing's text: plain text from EDGAR's HTML, its sections, dictionary counts, similarity.

Steps for one annual report:

1. **Text** - :func:`plain_text` drops what a reader never sees (inline XBRL's hidden header, scripts, styles), turns
   block tags into line breaks and the rest of the markup into spaces, and decodes entities. Old filings in plain
   text pass through the same steps unharmed.
2. **Sections** - :func:`section` finds Item 7 (management's discussion and analysis) and Item 1A (risk factors) by
   their headings at the start of a line, taking the longest span between a heading and the next item's: a table of
   contents lists the same headings with nothing between them.
3. **Words** - :func:`words` counts the alphabetic tokens of two letters or more, upper-cased as the
   Loughran-McDonald dictionary lists them.
4. **Measures** - :func:`dictionary_counts` counts the words of each Loughran-McDonald category (a word belongs to a
   category the year it was added there, and leaves it when the dictionary marks it removed - a negative year);
   :func:`similarity` compares two reports' word counts (cosine of the term-frequency vectors, and the Jaccard overlap
   of their vocabularies) - Cohen, Malloy and Nguyen's "Lazy Prices" measure of how much a company changed its report.
"""

from __future__ import annotations

import html
import io
import math
import re
from collections import Counter
from collections.abc import Mapping

import pandas as pd

#: Loughran-McDonald categories counted, as their dictionary names the columns.
CATEGORIES = ("Negative", "Positive", "Uncertainty", "Litigious", "Strong_Modal", "Weak_Modal", "Constraining")
#: A section shorter than this is a heading or a cross-reference ("see the annual report, Exhibit 13"), not the text.
MIN_SECTION_WORDS = 250
#: Method names stored beside each measure: a new dictionary or measure is a new method, kept beside the old.
LM_METHOD = "lm-1993-2025"
SIMILARITY_METHOD = "bow-1"

_HIDDEN = re.compile(r"<ix:header>.*?</ix:header>|<(script|style|head)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_BLOCK = re.compile(r"<\s*/?\s*(?:p|div|br|tr|li|h[1-6]|table|page)\b[^>]*>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]*>")
_SPACES = re.compile(r"[ \t\r\f\v\u00a0\u2002-\u200b]+")
_WORD = re.compile(r"[A-Za-z]{2,}")

_SEP = r"[\s.:\-\u2013\u2014]*"
MDNA_START = re.compile(rf"(?im)^\s*item{_SEP}7{_SEP}management\W{{0,3}}s?\s+discussion")
MDNA_END = re.compile(rf"(?im)^\s*item{_SEP}(?:7a\b|8{_SEP}(?:consolidated\s+)?financial\s+statements)")
RISK_START = re.compile(rf"(?im)^\s*item{_SEP}1a{_SEP}risk\s+factors")
RISK_END = re.compile(rf"(?im)^\s*item{_SEP}(?:1b\b|2{_SEP}properties|1c\b)")
SECTIONS = {"mdna": (MDNA_START, MDNA_END), "risk": (RISK_START, RISK_END)}
NEWLINE = chr(10)


def plain_text(raw: bytes) -> str:
    """The text a reader of the filing sees, one block per line."""
    text = raw.decode("utf-8", errors="replace")
    text = _HIDDEN.sub(" ", text)
    text = _BLOCK.sub("\n", text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    lines = (_SPACES.sub(" ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def section(text: str, start: re.Pattern[str], end: re.Pattern[str]) -> str:
    """The longest span from a ``start`` heading's line to the next ``end`` heading (empty when there is none)."""
    best = ""
    starts = list(start.finditer(text))
    for n, m in enumerate(starts):
        line_end = text.find(NEWLINE, m.end())  # the section starts after its heading's line
        begin = m.end() if line_end < 0 else line_end
        stop = end.search(text, begin)
        if stop is None:
            continue
        if n + 1 < len(starts) and starts[n + 1].start() < stop.start():
            continue  # the same heading comes again before the end: this one is a contents entry
        if stop.start() - begin > len(best):
            best = text[begin : stop.start()]
    return best


def words(text: str) -> Counter[str]:
    """Upper-cased alphabetic tokens of two letters or more, counted."""
    return Counter(w.upper() for w in _WORD.findall(text))


def lm_dictionary(csv: bytes) -> dict[str, frozenset[str]]:
    """``{category: words}`` from the Loughran-McDonald master dictionary (its CSV)."""
    frame = pd.read_csv(io.BytesIO(csv), usecols=["Word", *CATEGORIES], keep_default_na=False)
    frame["Word"] = frame["Word"].astype(str).str.upper()
    return {c: frozenset(frame.loc[pd.to_numeric(frame[c], errors="coerce") > 0, "Word"]) for c in CATEGORIES}


def dictionary_counts(counts: Mapping[str, int], dictionary: Mapping[str, frozenset[str]]) -> dict[str, float]:
    """``words`` (all tokens) and the tokens of each category, keyed by the category in lower case."""
    out = {"words": float(sum(counts.values()))}
    for category, vocab in dictionary.items():
        out[category.lower()] = float(sum(n for w, n in counts.items() if w in vocab))
    return out


def similarity(a: Mapping[str, int], b: Mapping[str, int]) -> dict[str, float]:
    """Cosine similarity of two term-frequency vectors and the Jaccard overlap of their vocabularies."""
    dot = sum(n * b.get(w, 0) for w, n in a.items())
    norm = math.sqrt(sum(n * n for n in a.values())) * math.sqrt(sum(n * n for n in b.values()))
    union = len(a.keys() | b.keys())
    return {
        "sim_cosine": dot / norm if norm else float("nan"),
        "sim_jaccard": len(a.keys() & b.keys()) / union if union else float("nan"),
    }


def report_counts(raw: bytes) -> dict[str, Counter[str]]:
    """Word counts of a report and of its sections: ``{"all": ..., "mdna": ..., "risk": ...}`` (short ones left out)."""
    text = plain_text(raw)
    out = {"all": words(text)}
    for name, (start, end) in SECTIONS.items():
        counts = words(section(text, start, end))
        if sum(counts.values()) >= MIN_SECTION_WORDS:
            out[name] = counts
    return out


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\n+")


def sample_sentences(text: str, n: int, *, min_words: int = 8, max_words: int = 60) -> list[str]:
    """Up to ``n`` prose sentences spread evenly through ``text`` (the same ones every time).

    A sentence counts when it has ``min_words`` to ``max_words`` words, most of them alphabetic - table rows and
    headings do not.
    """
    picked = []
    for s in _SENTENCE_END.split(text):
        tokens = s.split()
        if min_words <= len(tokens) <= max_words and sum(t.isalpha() for t in tokens) >= 0.7 * len(tokens):
            picked.append(" ".join(tokens))
    if len(picked) <= n:
        return picked
    step = len(picked) / n
    return [picked[int(i * step)] for i in range(n)]


def measures(
    counts: Mapping[str, Counter[str]],
    previous: Mapping[str, Counter[str]] | None,
    dictionary: Mapping[str, frozenset[str]],
) -> list[tuple[str, str, str, float]]:
    """``(section, measure, method, value)`` rows of one report, compared with the company's ``previous`` report."""
    rows = []
    for name, c in counts.items():
        rows += [(name, k, LM_METHOD, v) for k, v in dictionary_counts(c, dictionary).items()]
        if previous is not None and name in previous:
            rows += [(name, k, SIMILARITY_METHOD, v) for k, v in similarity(c, previous[name]).items()]
    return [r for r in rows if not math.isnan(r[3])]
