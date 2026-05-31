"""
core/query/normalizer.py

QueryNormalizer: deterministic, rule-based normalisation of raw query strings.

Always runs regardless of the rewrite switch.  This is the minimum
cleaning every query receives before being sent to ES or embedding.

Rules applied (in order)
------------------------
1. Strip leading/trailing whitespace.
2. Collapse internal whitespace runs (spaces and tabs) to single space.
3. Normalise Unicode: NFC + full-width → half-width ASCII.
4. Remove control characters (keep \\n \\t).
5. Collapse 3+ consecutive newlines to a single space
   (multi-line queries treated as a single search intent).
6. Final strip.

What is NOT done here
---------------------
- No semantic rewriting (that is the LLM rewriter).
- No stopword removal (BM25 handles term weighting).
- No tokenisation (ES/embedder handle that).
- No query expansion (Phase 2 concern).
"""

from __future__ import annotations

import re
import unicodedata


# Full-width ASCII → half-width
_FULLWIDTH_RE = re.compile(r"[\uff01-\uff5e]")
_CONTROL_RE   = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_WS_RE  = re.compile(r"[ \t]+")
_MULTI_NL_RE  = re.compile(r"\n{2,}")


def _fw_to_hw(ch: str) -> str:
    return chr(ord(ch) - 0xFEE0)


class QueryNormalizer:
    """
    Applies deterministic rule-based normalisation to a raw query string.
    Stateless — one instance can be shared across all requests.
    """

    def normalize(self, raw: str) -> str:
        """
        Return the normalised query string.
        Input is never stored. Returns empty string for blank input.
        """
        if not raw or not raw.strip():
            return ""

        text = raw

        # 1. Control chars
        text = _CONTROL_RE.sub("", text)

        # 2. Unicode NFC + full-width → half-width
        text = unicodedata.normalize("NFC", text)
        text = _FULLWIDTH_RE.sub(lambda m: _fw_to_hw(m.group()), text)
        # Ideographic space → regular space
        text = text.replace("\u3000", " ")

        # 3. Collapse whitespace within lines
        text = _MULTI_WS_RE.sub(" ", text)

        # 4. Collapse multi-line to single space
        #    (a query spanning multiple lines is one intent)
        text = _MULTI_NL_RE.sub(" ", text)
        text = text.replace("\n", " ")

        # 5. Final strip and whitespace collapse
        text = _MULTI_WS_RE.sub(" ", text).strip()

        return text
