"""
core/ingestion/token_counter.py

Token counting for the chunker length guardrail.

Strategy
--------
Primary:  tiktoken with cl100k_base encoding (used by bge-series tokenisers;
          good approximation for Chinese + English mixed text).
          tiktoken requires downloading the encoding file on first use, which
          fails in air-gapped environments.  The loader catches that error and
          falls back automatically.

Fallback: character-ratio estimate that is CONSISTENT with truncation:
          CJK chars  → 1 token each  (bge tokeniser is ~1 CJK char = 1 token)
          ASCII/other→ 1 token per 3 chars (conservative; over-estimates slightly)
          truncate_to_tokens uses the same ratio, so count(truncate(t, N)) ≤ N
          is always guaranteed.

Public API
----------
    counter = TokenCounter()
    n = counter.count("some text")
    fits = counter.fits(text, max_tokens=400, reserve=80)
    short = counter.truncate_to_tokens(text, max_tokens=100)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import tiktoken as _tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


def _is_cjk_char(ch: str) -> bool:
    """True for CJK ideograph characters counted as one token in fallback mode."""
    codepoint = ord(ch)
    return (
        0x4E00 <= codepoint <= 0x9FFF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
    )


def _cjk_char_count(text: str) -> int:
    """Count CJK ideograph characters in text."""
    return sum(1 for ch in text if _is_cjk_char(ch))


def _fallback_count(text: str) -> int:
    """
    Conservative token estimate.
    CJK chars  : 1 token each
    Other chars: 1 token per 3 chars (ASCII prose is ~4 chars/token but
                 we use 3 to stay safely under the guardrail)
    """
    cjk = _cjk_char_count(text)
    other = len(text) - cjk
    return cjk + max(0, (other + 2) // 3)   # ceiling division for other


def _fallback_truncate(text: str, max_tokens: int) -> str:
    """
    Truncate text so that _fallback_count(result) <= max_tokens.

    We walk the string accumulating our token count and stop when we reach
    the budget.  This is O(n) but exact under the fallback model, ensuring
    count(truncate(t, N)) <= N invariant holds.
    """
    cjk_tokens = 0
    non_cjk_chars = 0
    result_chars: list[str] = []
    for ch in text:
        if _is_cjk_char(ch):
            next_cjk_tokens = cjk_tokens + 1
            next_non_cjk_chars = non_cjk_chars
        else:
            next_cjk_tokens = cjk_tokens
            next_non_cjk_chars = non_cjk_chars + 1

        next_cost = next_cjk_tokens + (next_non_cjk_chars + 2) // 3
        if next_cost > max_tokens:
            break

        cjk_tokens = next_cjk_tokens
        non_cjk_chars = next_non_cjk_chars
        result_chars.append(ch)
    return "".join(result_chars)


class TokenCounter:
    """
    Counts tokens in a text string.

    Parameters
    ----------
    encoding_name:
        tiktoken encoding name.  cl100k_base is a good approximation for
        bge-series models on Chinese + English mixed text.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._enc = None
        if _TIKTOKEN_AVAILABLE:
            try:
                # tiktoken caches encoding files in TIKTOKEN_CACHE_DIR
                # (default: ~/.tiktoken).  In an air-gapped environment,
                # pre-download the .tiktoken file and set that env var.
                self._enc = _tiktoken.get_encoding(encoding_name)
                logger.debug("TokenCounter using tiktoken %r", encoding_name)
            except Exception:
                # Network unavailable (air-gap) or corrupt cache — use fallback.
                # This is expected in intranet deployments without tiktoken cache.
                logger.debug(
                    "tiktoken encoding %r unavailable; using character-ratio fallback. "
                    "Pre-download the encoding file and set TIKTOKEN_CACHE_DIR for "
                    "accurate token counts.",
                    encoding_name,
                )

        if self._enc is None:
            logger.debug("TokenCounter using character-ratio fallback")

    @property
    def using_tiktoken(self) -> bool:
        return self._enc is not None

    def count(self, text: str) -> int:
        """Return the token count of `text`."""
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text))
        return _fallback_count(text)

    def fits(self, text: str, max_tokens: int, *, reserve: int = 0) -> bool:
        """
        True if text fits within (max_tokens - reserve) tokens.
        """
        effective = max_tokens - reserve
        if effective <= 0:
            return False
        return self.count(text) <= effective

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Return the longest prefix of `text` whose token count ≤ max_tokens.
        Invariant: count(truncate_to_tokens(t, N)) <= N.
        """
        if not text or max_tokens <= 0:
            return ""
        if self._enc is not None:
            tokens = self._enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return self._enc.decode(tokens[:max_tokens])
        # Fallback: exact under the fallback count model
        return _fallback_truncate(text, max_tokens)

    def token_suffix(self, text: str, max_tokens: int) -> str:
        """
        Return the last `max_tokens` tokens of `text` (suffix version of
        truncate_to_tokens).  Used for overlap prepending.
        """
        if not text or max_tokens <= 0:
            return ""
        if self._enc is not None:
            tokens = self._enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return self._enc.decode(tokens[-max_tokens:])
        # Fallback: reverse, truncate, reverse
        reversed_text = text[::-1]
        truncated_rev = _fallback_truncate(reversed_text, max_tokens)
        return truncated_rev[::-1]
