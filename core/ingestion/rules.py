"""
core/ingestion/rules.py

Individual cleaning rule functions.

Contract
--------
Every rule is a pure function:
    rule(text: str, **kwargs) -> tuple[str, CleaningRecord]

- Deterministic: same input always produces same output.
- No side effects.
- Returns the record even if the text is unchanged (op=NOOP or chars_before==chars_after).
- Never raises on valid string input.

Rules are composited in Cleaner (cleaner.py); they are not called directly
by downstream code.

Rule catalogue
--------------
1. fix_encoding        — ftfy-based mojibake / encoding repair
2. strip_control_chars — remove non-printable control characters (keep \n \t)
3. normalize_unicode   — NFC normalisation; full-width → half-width; ligature expand
4. strip_html          — BeautifulSoup HTML tag removal + entity decode
5. strip_boilerplate   — regex-based removal of repeating boilerplate lines
6. normalize_whitespace — collapse runs of spaces/tabs; normalise line endings;
                          remove leading/trailing blank lines
7. fix_repeated_punct  — collapse runs of punctuation (e.g. !!!!! → !、……… → …)
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable

from core.ingestion.cleaning_record import CleaningRecord, TransformOp

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------
# ftfy and BeautifulSoup are soft dependencies.  If they are not installed
# the corresponding rules degrade gracefully (log as NOOP with a detail note).

try:
    import ftfy as _ftfy
    _FTFY_AVAILABLE = True
except ImportError:
    _FTFY_AVAILABLE = False

try:
    from bs4 import BeautifulSoup as _BS
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
RuleResult = tuple[str, CleaningRecord]


# ---------------------------------------------------------------------------
# Rule 1 — Encoding / mojibake repair
# ---------------------------------------------------------------------------

def fix_encoding(text: str) -> RuleResult:
    """
    Use ftfy to fix mojibake, mis-decoded UTF-8, and common encoding errors
    that appear when scraping or converting enterprise docs.

    If ftfy is not installed, returns the text unchanged (NOOP with note).
    This is intentional: ftfy is a soft dep; the rule degrades gracefully.
    """
    before = len(text)
    if not _FTFY_AVAILABLE:
        return text, CleaningRecord(
            op=TransformOp.ENCODING_FIX,
            chars_before=before,
            chars_after=before,
            detail="ftfy not installed; skipped",
        )

    fixed = _ftfy.fix_text(text)
    detail = f"ftfy fixed {before - len(fixed)} chars" if len(fixed) != before else ""
    return fixed, CleaningRecord(
        op=TransformOp.ENCODING_FIX,
        chars_before=before,
        chars_after=len(fixed),
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Rule 2 — Control character removal
# ---------------------------------------------------------------------------

# Keep: printable chars, \n (0x0A), \t (0x09), \r (0x0D, normalised later)
# Strip: NUL (0x00), SOH–US (0x01–0x1F minus the above), DEL (0x7F), private-use
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffe\uffff]"
)


def strip_control_chars(text: str) -> RuleResult:
    """Remove non-printable control characters while preserving \\n and \\t."""
    before = len(text)
    cleaned, n = _CONTROL_CHAR_RE.subn("", text)
    return cleaned, CleaningRecord(
        op=TransformOp.CONTROL_CHAR_STRIP,
        chars_before=before,
        chars_after=len(cleaned),
        detail=f"removed {n} control chars" if n else "",
    )


# ---------------------------------------------------------------------------
# Rule 3 — Unicode normalisation
# ---------------------------------------------------------------------------

# Full-width ASCII → half-width (common in Chinese enterprise docs copy-pasted
# from PDFs: Ａ→A, ０→0, ：→:, etc.)
_FULLWIDTH_RE = re.compile(r"[\uff01-\uff5e]")


def _fw_to_hw(char: str) -> str:
    """Convert one full-width ASCII character to its half-width equivalent."""
    return chr(ord(char) - 0xFEE0)


def normalize_unicode(text: str) -> RuleResult:
    """
    1. NFC normalisation (canonical composition).
    2. Full-width ASCII punctuation/digits → half-width.
    3. Ideographic space (U+3000) → regular space.

    Does NOT change Chinese characters — only normalises the non-ideographic
    overlay characters that come from copy-paste artefacts.
    """
    before = len(text)

    # NFC first (decomposes then recomposes; handles accented chars)
    text = unicodedata.normalize("NFC", text)
    # Full-width → half-width
    text = _FULLWIDTH_RE.sub(lambda m: _fw_to_hw(m.group()), text)
    # Ideographic space → regular space
    text = text.replace("\u3000", " ")

    return text, CleaningRecord(
        op=TransformOp.UNICODE_NORM,
        chars_before=before,
        chars_after=len(text),
        detail="" if len(text) == before else f"normalised {before - len(text)} chars",
    )


# ---------------------------------------------------------------------------
# Rule 4 — HTML stripping
# ---------------------------------------------------------------------------

# Minimal HTML detection heuristic: presence of tags like <p>, <div>, <br/> etc.
_HTML_TAG_RE = re.compile(r"<[a-zA-Z/][^>]{0,200}>")


def strip_html(text: str) -> RuleResult:
    """
    Remove HTML tags and decode HTML entities.
    Uses BeautifulSoup if available; falls back to a simple regex strip.

    Policy: preserve newlines implied by block elements (<p>, <br>, <li>).
    """
    before = len(text)

    # Fast path: skip if text does not look like HTML
    if not _HTML_TAG_RE.search(text):
        return text, CleaningRecord(
            op=TransformOp.HTML_STRIP,
            chars_before=before,
            chars_after=before,
            detail="no HTML detected",
        )

    if _BS4_AVAILABLE:
        soup = _BS(text, "html.parser")
        # Insert newlines at block boundaries before stripping
        for tag in soup.find_all(["p", "div", "br", "li", "tr", "h1", "h2",
                                   "h3", "h4", "h5", "h6", "blockquote"]):
            tag.insert_before("\n")
        cleaned = soup.get_text(separator="")
        detail = f"bs4 stripped HTML; {before - len(cleaned)} chars removed"
    else:
        # Regex fallback: insert newlines at block tags then strip all tags
        text = re.sub(r"<(?:p|div|br|li|tr|h[1-6]|blockquote)[^>]*>",
                      "\n", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"<[^>]+>", "", text)
        # Decode common HTML entities
        cleaned = (cleaned
                   .replace("&amp;", "&")
                   .replace("&lt;", "<")
                   .replace("&gt;", ">")
                   .replace("&nbsp;", " ")
                   .replace("&quot;", '"')
                   .replace("&#39;", "'"))
        detail = f"regex stripped HTML; {before - len(cleaned)} chars removed"

    return cleaned, CleaningRecord(
        op=TransformOp.HTML_STRIP,
        chars_before=before,
        chars_after=len(cleaned),
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Rule 5 — Boilerplate stripping
# ---------------------------------------------------------------------------

# Default boilerplate patterns.  Each is a compiled regex that matches a whole
# line (or a multi-line block).  Add per-business patterns via the Cleaner's
# extra_boilerplate_patterns config.
_DEFAULT_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    # Page headers/footers: "第 N 页 / Page N of M"
    re.compile(r"^第\s*\d+\s*页.*$", re.MULTILINE),
    re.compile(r"^[Pp]age\s+\d+\s*(of\s+\d+)?.*$", re.MULTILINE),
    # Copyright / confidentiality lines
    re.compile(r"^.*(?:版权所有|copyright\s+©?|confidential|内部资料|仅供内部).*$",
               re.MULTILINE | re.IGNORECASE),
    # Repeated separator lines (---, ===, ···)
    re.compile(r"^[\-=·—\*]{4,}\s*$", re.MULTILINE),
    # "Print date" / "Export date" artefacts
    re.compile(r"^(?:打印日期|导出时间|export(?:ed)?\s+(?:on|date)|print(?:ed)?\s+(?:on|date)).*$",
               re.MULTILINE | re.IGNORECASE),
    # Blank "本页无正文" / "（以下空白）" filler lines common in Chinese policy docs
    re.compile(r"^[\s（(]*(?:以下空白|本页无正文|此页空白)[\s）)]*$", re.MULTILINE),
]


def strip_boilerplate(
    text: str,
    extra_patterns: list[re.Pattern] | None = None,
    *,
    use_defaults: bool = True,
) -> RuleResult:
    """
    Remove known boilerplate lines using regex patterns.

    Parameters
    ----------
    extra_patterns:
        Additional compiled patterns prepended to the default list.
        These always run regardless of `use_defaults`.
    use_defaults:
        If False, the built-in _DEFAULT_BOILERPLATE_PATTERNS are skipped.
        Use when a CleaningProfile sets disable_default_boilerplate=True.
    """
    before = len(text)
    patterns: list[re.Pattern] = list(extra_patterns or [])
    if use_defaults:
        patterns = patterns + _DEFAULT_BOILERPLATE_PATTERNS

    if not patterns:
        return text, CleaningRecord(
            op=TransformOp.BOILERPLATE_STRIP,
            chars_before=before,
            chars_after=before,
            detail="no patterns active",
        )

    matches_total = 0
    for pattern in patterns:
        text, n = pattern.subn("", text)
        matches_total += n

    return text, CleaningRecord(
        op=TransformOp.BOILERPLATE_STRIP,
        chars_before=before,
        chars_after=len(text),
        detail=f"removed {matches_total} boilerplate match(es)" if matches_total else "",
    )


# ---------------------------------------------------------------------------
# Rule 6 — Whitespace normalisation
# ---------------------------------------------------------------------------

_MULTI_SPACE_RE  = re.compile(r"[ \t]+")          # runs of spaces/tabs → single space
_CRLF_RE         = re.compile(r"\r\n|\r")         # Windows/Mac line endings → \n
_MULTI_BLANK_RE  = re.compile(r"\n{3,}")          # 3+ consecutive blank lines → 2


def normalize_whitespace(text: str) -> RuleResult:
    """
    1. \r\n and \r → \n.
    2. Runs of spaces/tabs on a line → single space.
    3. 3+ consecutive blank lines → 2 blank lines (preserve paragraph breaks).
    4. Strip leading/trailing whitespace from the whole text.

    Preserves intentional single blank lines (paragraph separators).
    """
    before = len(text)

    text = _CRLF_RE.sub("\n", text)
    # Normalise spaces/tabs within each line but do not collapse newlines yet
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    text = text.strip()

    return text, CleaningRecord(
        op=TransformOp.WHITESPACE_NORM,
        chars_before=before,
        chars_after=len(text),
        detail="" if len(text) == before else f"{before - len(text)} chars normalised",
    )


# ---------------------------------------------------------------------------
# Rule 7 — Repeated punctuation collapse
# ---------------------------------------------------------------------------

# Collapse runs of the same punctuation character (≥3) to a single instance.
# Covers: !!! → !  ??? → ?  ...... → …  ~~~~~ → ~
# Chinese ellipsis (……) is kept as-is (2 chars, not collapsed).
# Dashes: ---- → — (em dash)
_REPEATED_PUNCT_RE = re.compile(
    r"([!?~`@#$%^&*_+=|<>])\1{2,}"   # 3+ ASCII punctuation repeats
    r"|([.。]){3,}"                    # 3+ dots/Chinese full-stops → ellipsis
    r"|([-—]){3,}"                     # 3+ dashes → em dash
)


def _collapse_punct(m: re.Match) -> str:
    if m.group(1):
        return m.group(1)
    if m.group(2):
        return "…"
    if m.group(3):
        return "—"
    return m.group()


def fix_repeated_punct(text: str) -> RuleResult:
    """Collapse runs of repeated punctuation that are visual noise."""
    before = len(text)
    cleaned, n = _REPEATED_PUNCT_RE.subn(_collapse_punct, text)
    return cleaned, CleaningRecord(
        op=TransformOp.REPEATED_PUNCT_FIX,
        chars_before=before,
        chars_after=len(cleaned),
        detail=f"collapsed {n} punct run(s)" if n else "",
    )


# ---------------------------------------------------------------------------
# Rule registry — ordered list of (name, callable) for the default pipeline
# ---------------------------------------------------------------------------

DEFAULT_RULE_PIPELINE: list[tuple[str, Callable[[str], RuleResult]]] = [
    ("encoding_fix",        fix_encoding),
    ("control_char_strip",  strip_control_chars),
    ("unicode_norm",        normalize_unicode),
    ("html_strip",          strip_html),
    ("boilerplate_strip",   strip_boilerplate),
    ("whitespace_norm",     normalize_whitespace),
    ("repeated_punct_fix",  fix_repeated_punct),
]
