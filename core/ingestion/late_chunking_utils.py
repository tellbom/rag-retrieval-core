"""
core/ingestion/late_chunking_utils.py

Shared utilities for Late Chunking (Phase 2).

Late Chunking overview
----------------------
Instead of embedding each chunk's context_text independently, Late Chunking:
  1. Assembles a "group text" from all sibling chunks under the same lc_group_id,
     concatenated in position order with GROUP_SEP between them.
  2. Calls TEI /embed_all on the group text to get a token-level vector matrix.
  3. Uses TEI /tokenize to map each chunk's (char_start, char_end) to a token
     index range within the group text.
  4. Mean-pools the token vectors in that range to produce one vector per chunk.
  5. L2-normalises the pooled vector (to match /embed behaviour when normalize=True).

This file is imported by both StructuralChunker (to build group_text consistently
when writing char offsets) and Embedder (to rebuild group_text for pooling).
Both must use GROUP_SEP and build_group_text() — never inline the join logic.

Public API
----------
GROUP_SEP                               : str  — separator used between sibling texts
build_group_text(texts: list[str]) -> str
find_token_range(
    char_start: int,
    char_end: int,
    token_offsets: list[tuple[int, int]],
) -> tuple[int, int] | None
mean_pool(
    token_vectors: list[list[float]],
    token_start: int,
    token_end: int,
) -> list[float] | None
l2_normalize(vector: list[float]) -> list[float]
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Group text construction
# ---------------------------------------------------------------------------

# Separator inserted between sibling chunk texts when building the group text.
# Must be a single newline so that char offsets computed in the chunker remain
# valid when the Embedder reconstructs the same group_text.
GROUP_SEP: str = "\n"


def build_group_text(texts: list[str]) -> str:
    """
    Join sibling chunk texts (already sorted by position) into one group text.

    Both StructuralChunker (when computing char_start/char_end) and Embedder
    (when reconstructing group_text for /embed_all) must call this function —
    never inline the join — so the separator contract is enforced in one place.

    Parameters
    ----------
    texts:  Ordered list of chunk.text values (sorted by chunk.position).

    Returns
    -------
    Single string: texts[0] + GROUP_SEP + texts[1] + ... + texts[-1].
    Empty string if texts is empty.
    """
    return GROUP_SEP.join(texts)


# ---------------------------------------------------------------------------
# Token range mapping
# ---------------------------------------------------------------------------

def find_token_range(
    char_start: int,
    char_end: int,
    token_offsets: list[tuple[int, int]],
) -> tuple[int, int] | None:
    """
    Map a character range [char_start, char_end) to a token index range
    [token_start, token_end) within the token_offsets list returned by
    TEI /tokenize.

    token_offsets is a list of (char_start_of_token, char_end_of_token) pairs,
    one per token (including special tokens such as [CLS] and [SEP]).

    Algorithm
    ---------
    - token_start: first token index whose char span overlaps or starts at
      char_start.
    - token_end:   first token index whose char span starts at or after char_end
      (exclusive upper bound).
    - Special tokens (those with char_start == char_end, i.e. zero-width spans)
      are skipped when searching boundaries.

    Returns
    -------
    (token_start, token_end) with token_end > token_start, or None if the
    character range cannot be reliably mapped (e.g. range falls outside all
    token spans, or results in an empty token range).
    """
    if not token_offsets or char_start >= char_end:
        return None

    t_start: int | None = None
    t_end: int | None = None

    for idx, (tok_cs, tok_ce) in enumerate(token_offsets):
        is_zero_width = (tok_cs == tok_ce)

        # t_start: first non-zero-width token that overlaps [char_start, char_end)
        if t_start is None and not is_zero_width and tok_ce > char_start:
            t_start = idx

        # t_end: first token (including zero-width) whose start is at or beyond char_end
        # This correctly excludes trailing special tokens (e.g. [SEP] at position char_end)
        if t_start is not None and tok_cs >= char_end:
            t_end = idx
            break

    # If we never found t_end, all remaining tokens up to end are within range
    if t_start is not None and t_end is None:
        t_end = len(token_offsets)

    if t_start is None or t_end is None or t_end <= t_start:
        logger.debug(
            "find_token_range: could not map char[%d:%d] in %d tokens",
            char_start, char_end, len(token_offsets),
        )
        return None

    return t_start, t_end


# ---------------------------------------------------------------------------
# Pooling and normalisation
# ---------------------------------------------------------------------------

def mean_pool(
    token_vectors: list[list[float]],
    token_start: int,
    token_end: int,
) -> list[float] | None:
    """
    Mean-pool token vectors in the index range [token_start, token_end).

    Parameters
    ----------
    token_vectors : Full token-level vector matrix from /embed_all.
                    Shape: [num_tokens × dimension].
    token_start   : Inclusive start index (from find_token_range).
    token_end     : Exclusive end index (from find_token_range).

    Returns
    -------
    list[float] of length == dimension, or None if the slice is empty or
    the vector matrix is malformed.
    """
    if not token_vectors:
        return None
    if token_start < 0 or token_end > len(token_vectors) or token_start >= token_end:
        logger.debug(
            "mean_pool: invalid range [%d:%d] for %d token vectors",
            token_start, token_end, len(token_vectors),
        )
        return None

    slice_vecs = token_vectors[token_start:token_end]
    dim = len(slice_vecs[0])

    # Verify all vectors in slice have the same dimension
    if any(len(v) != dim for v in slice_vecs):
        logger.debug("mean_pool: inconsistent vector dimensions in slice")
        return None

    pooled = [0.0] * dim
    n = len(slice_vecs)
    for vec in slice_vecs:
        for i, val in enumerate(vec):
            pooled[i] += val
    return [v / n for v in pooled]


def l2_normalize(vector: list[float]) -> list[float]:
    """
    L2-normalise a vector in-place style (returns a new list).

    TEI /embed returns normalised vectors when normalize=True in the model
    config.  /embed_all returns raw token vectors that are NOT normalised.
    After mean pooling we must normalise so that the resulting chunk vector
    lives in the same unit-sphere space as the /embed vectors used at query time.

    If the vector norm is zero (zero vector), returns the original vector
    unchanged to avoid division by zero.
    """
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return list(vector)
    return [v / norm for v in vector]
