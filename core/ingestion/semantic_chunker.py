"""
core/ingestion/semantic_chunker.py

SemanticChunker: secondary, bounded chunker that operates ONLY on chunks
flagged with `needs_semantic_split=True` by the StructuralChunker.

Design constraints (from plan.md + design review)
--------------------------------------------------
1. **Never fires on small units.**  Any chunk without `needs_semantic_split`
   passes through unchanged.  This is not optional — the structural chunker
   already set correct boundaries for those chunks.

2. **Deterministic.**  Given the same input chunk and config, always produces
   the same sub-chunks.  No randomness, no embedding-model dependency in this
   phase (that belongs to P2 late-chunking).

3. **Respects the length guardrail.**  Every output sub-chunk satisfies
   `token_count ≤ max_tokens`.

4. **Strategy: `late_chunking` (Phase 1 default).**
   Split at natural sentence/paragraph boundaries rather than using an
   embedding model.  This gives deterministic, reproducible splits that work
   in an air-gapped environment with no model calls.  The boundary detection
   uses sentence-end punctuation for Chinese + English text.

   `similarity_drop` strategy is the Phase 2 upgrade path — it requires a
   model call to compute sentence embeddings and find semantic boundary drops.
   In Phase 1 it falls back to `late_chunking` with a logged warning.

5. **Parent-child integrity.**  Sub-chunks generated from a parent chunk
   carry the parent chunk's chunk_id as their `parent_id`, at
   `hierarchy_level = parent.hierarchy_level + 1`.

6. **context_text preserved.**  The parent's context prefix is prepended to
   each sub-chunk so the context_preservation_tokens intent is maintained.

7. **chunk_id stability.**  Sub-chunk IDs are derived from the parent's
   chunk_id + sub-position, so IDs are stable across re-runs.

Public API
----------
    chunker = SemanticChunker(cfg)
    result = chunker.split(chunking_result)   # ChunkingResult → ChunkingResult
"""

from __future__ import annotations

import hashlib
import logging
import re

from core.config.models import ChunkingConfig
from core.ingestion.chunk import Chunk, ChunkingResult
from core.ingestion.late_chunking_utils import GROUP_SEP
from core.ingestion.token_counter import TokenCounter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence boundary patterns
# ---------------------------------------------------------------------------
# Matches the end of a sentence in Chinese or English text.
# Chinese sentence-ending punctuation: 。！？…… ; punctuation + closing bracket
# English: . ! ? followed by space or end-of-string
_SENTENCE_END_RE = re.compile(
    r"(?<=[。！？…\?!])"            # Chinese/English sentence-end punctuation
    r"|(?<=\.)\s+(?=[A-Z\u4e00-\u9fff])"  # English: period + space before capital/CJK
    r"|(?<=\n\n)"                   # blank line (paragraph break)
)

# Minimum sub-chunk size: never split below this many tokens
# (prevents producing useless single-sentence fragments)
_MIN_SUBCHUNK_TOKENS = 20


def _sub_chunk_id(parent_chunk_id: str, sub_position: int) -> str:
    """Stable sub-chunk id derived from parent chunk_id + sub-position."""
    raw = f"{parent_chunk_id}:sub:{sub_position}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _split_into_sentences(text: str) -> list[tuple[str, int, int]]:
    """
    Split text into sentence-level fragments using punctuation boundaries.

    Returns a list of (fragment_text, char_start, char_end) tuples where
    char_start and char_end are character offsets into the original `text`.
    Empty or whitespace-only fragments are dropped.

    Using re.finditer on the boundary pattern to locate split points, then
    slicing the original text directly — this preserves original characters
    (including inter-sentence punctuation and whitespace) so that
    text[char_start:char_end] == fragment_text for every returned tuple.
    """
    if not text:
        return []

    # Find all boundary positions (the index where a new sentence starts)
    boundary_positions: list[int] = [0]
    for m in _SENTENCE_END_RE.finditer(text):
        pos = m.end()
        if pos < len(text):
            boundary_positions.append(pos)
    boundary_positions.append(len(text))

    fragments: list[tuple[str, int, int]] = []
    for i in range(len(boundary_positions) - 1):
        start = boundary_positions[i]
        end = boundary_positions[i + 1]
        fragment = text[start:end]
        stripped = fragment.strip()
        if not stripped:
            continue
        # Adjust start/end to the stripped content within original text
        leading = len(fragment) - len(fragment.lstrip())
        actual_start = start + leading
        actual_end = actual_start + len(stripped)
        fragments.append((stripped, actual_start, actual_end))

    return fragments


def _greedy_pack(
    sentences: list[tuple[str, int, int]],
    token_budget: int,
    counter: TokenCounter,
    original_text: str,
    min_tokens: int = _MIN_SUBCHUNK_TOKENS,
) -> list[tuple[str, int, int]]:
    """
    Greedily pack sentences into groups of at most `token_budget` tokens.

    Parameters
    ----------
    sentences:     list of (text, char_start, char_end) from _split_into_sentences.
    token_budget:  Maximum tokens per output group.
    counter:       TokenCounter instance.
    original_text: The original chunk text; used to slice group text directly
                   so that output text == original_text[group_start:group_end].
    min_tokens:    Minimum tokens for a group; trailing tiny groups are merged
                   into the previous one when possible.

    Returns
    -------
    list of (group_text, group_char_start, group_char_end) where
    group_text == original_text[group_char_start:group_char_end].
    """
    if not sentences:
        return []

    # Each group accumulates sentence indices
    groups: list[list[int]] = []   # list of lists of sentence indices
    current: list[int] = []
    current_tokens = 0

    for idx, (sent_text, sent_start, sent_end) in enumerate(sentences):
        s_tokens = counter.count(sent_text)
        # Hard-cap: a sentence exceeding budget is truncated
        if s_tokens > token_budget:
            s_tokens = token_budget  # will be enforced when slicing below

        if current_tokens + s_tokens <= token_budget:
            current.append(idx)
            current_tokens += s_tokens
        else:
            if current:
                groups.append(current)
            current = [idx]
            current_tokens = s_tokens

    if current:
        groups.append(current)

    # Merge trailing tiny group into previous if possible
    if len(groups) >= 2:
        last_tokens = sum(counter.count(sentences[i][0]) for i in groups[-1])
        if last_tokens < min_tokens:
            prev_tokens = sum(counter.count(sentences[i][0]) for i in groups[-2])
            if prev_tokens + last_tokens <= token_budget:
                groups[-2].extend(groups[-1])
                groups.pop()

    # Build output: slice original_text using the group's char span
    result: list[tuple[str, int, int]] = []
    for group_indices in groups:
        g_start = sentences[group_indices[0]][1]   # char_start of first sentence
        g_end   = sentences[group_indices[-1]][2]  # char_end of last sentence
        # Clamp in case of truncation on the last sentence in a single-sentence group
        g_end = min(g_end, g_start + len(
            counter.truncate_to_tokens(original_text[g_start:g_end], token_budget)
        ))
        group_text = original_text[g_start:g_end]
        if group_text.strip():
            result.append((group_text, g_start, g_end))

    return result


class SemanticChunker:
    """
    Splits oversized structural chunks into sub-chunks at natural boundaries.

    Parameters
    ----------
    cfg:      ChunkingConfig
    counter:  TokenCounter (shared, stateless)
    """

    def __init__(
        self,
        cfg: ChunkingConfig,
        counter: TokenCounter | None = None,
    ) -> None:
        self._cfg = cfg
        self._length = cfg.length
        self._semantic = cfg.semantic
        self._counter = counter or TokenCounter()

        # Token budget for the text portion of a sub-chunk
        # (reserves space for the context prefix inherited from parent)
        self._text_budget = (
            self._length.max_tokens - self._length.context_preservation_tokens
        )
        if self._text_budget <= 0:
            raise ValueError(
                f"max_tokens ({self._length.max_tokens}) must be greater than "
                f"context_preservation_tokens ({self._length.context_preservation_tokens})"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, result: ChunkingResult) -> ChunkingResult:
        """
        Process a ChunkingResult from the StructuralChunker.

        Chunks with `needs_semantic_split=False` pass through unchanged.
        Chunks with `needs_semantic_split=True` are replaced by their
        sub-chunks (with the original chunk removed from the output).

        Returns a new ChunkingResult with all chunks in document order.
        """
        if not self._semantic.enabled:
            # Semantic chunking is globally disabled — pass through unchanged
            return result

        output_chunks: list[Chunk] = []
        split_count = 0

        for chunk in result.chunks:
            if not chunk.needs_semantic_split:
                output_chunks.append(chunk)
                continue

            sub_chunks = self._split_chunk(chunk)

            if len(sub_chunks) <= 1:
                # Split produced only one piece — keep original, clear flag
                cleared = _clear_split_flag(chunk, sub_chunks[0].text if sub_chunks else chunk.text)
                output_chunks.append(cleared)
            else:
                output_chunks.extend(sub_chunks)
                split_count += 1
                logger.debug(
                    "Semantic split: chunk_id=%s → %d sub-chunks",
                    chunk.chunk_id, len(sub_chunks),
                )

        remaining_flags = sum(1 for c in output_chunks if c.needs_semantic_split)
        return ChunkingResult(
            doc_id=result.doc_id,
            chunks=output_chunks,
            needs_split_count=remaining_flags,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _split_chunk(self, chunk: Chunk) -> list[Chunk]:
        """
        Split one oversized chunk.  Returns a list of sub-Chunks.
        Falls back to the original chunk (as a single-element list) if
        splitting would not reduce size meaningfully.
        """
        strategy = self._semantic.strategy

        if strategy == "similarity_drop":
            # Phase 2: requires embedding model calls.
            # In Phase 1, fall back to late_chunking with a warning.
            logger.warning(
                "similarity_drop strategy requires embedding model (Phase 2). "
                "Falling back to late_chunking for chunk_id=%s.",
                chunk.chunk_id,
            )
            return self._split_late_chunking(chunk)

        # Default: late_chunking (deterministic sentence-boundary splitting)
        return self._split_late_chunking(chunk)

    def _split_late_chunking(self, chunk: Chunk) -> list[Chunk]:
        """
        Split at sentence/paragraph boundaries (no model calls).
        Each sub-chunk satisfies token_count ≤ max_tokens.

        Sub-chunks produced here carry lc_group_id / char_start / char_end so
        that the Embedder can apply Late Chunking pooling over the parent chunk's
        text.  The semantic group text is chunk.text itself; char offsets are
        relative to it.
        """
        sentences = _split_into_sentences(chunk.text)

        if not sentences:
            return [chunk]

        # If there's only one "sentence" and it already fits, no split needed
        if len(sentences) == 1:
            if self._counter.count(chunk.text) <= self._length.max_tokens:
                return [chunk]

        packed = _greedy_pack(
            sentences,
            token_budget=self._text_budget,
            counter=self._counter,
            original_text=chunk.text,
        )

        # If packing didn't actually split (returned one group), skip
        if len(packed) <= 1:
            # Still enforce the hard guardrail with truncation
            truncated = self._counter.truncate_to_tokens(
                chunk.text, self._text_budget
            )
            return [_replace_text(chunk, truncated, self._counter, self._length.context_preservation_tokens)]

        # Reconstruct context prefix from the parent's context_text
        context_prefix = _extract_context_prefix(
            chunk, self._length.context_preservation_tokens, self._counter
        )

        # lc_group_id for semantic sub-chunks is scoped to the parent chunk
        lc_group_id = f"sem:{chunk.chunk_id}"

        sub_chunks: list[Chunk] = []
        for sub_pos, (text, g_char_start, g_char_end) in enumerate(packed):
            sub_id = _sub_chunk_id(chunk.chunk_id, sub_pos)
            context_text = (context_prefix + "\n" + text).strip() if context_prefix else text
            token_count = self._counter.count(context_text)

            sub_chunks.append(Chunk(
                chunk_id=sub_id,
                doc_id=chunk.doc_id,
                parent_id=chunk.chunk_id,          # parent is the original chunk
                hierarchy_level=chunk.hierarchy_level + 1,
                position=sub_pos,
                text=text,
                context_text=context_text,
                heading_path=chunk.heading_path,
                business_type=chunk.business_type,
                source_metadata=chunk.source_metadata,
                config_version=chunk.config_version,
                derived_keywords=chunk.derived_keywords,
                derived_entities=chunk.derived_entities,
                derived_questions=chunk.derived_questions,
                needs_semantic_split=False,         # sub-chunks are never re-split
                token_count=token_count,
                lc_group_id=lc_group_id,
                char_start=g_char_start,
                char_end=g_char_end,
            ))

        return sub_chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_split_flag(chunk: Chunk, new_text: str | None = None) -> Chunk:
    """Return a copy of chunk with needs_semantic_split=False."""
    return Chunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        parent_id=chunk.parent_id,
        hierarchy_level=chunk.hierarchy_level,
        position=chunk.position,
        text=new_text if new_text is not None else chunk.text,
        context_text=chunk.context_text,
        heading_path=chunk.heading_path,
        business_type=chunk.business_type,
        source_metadata=chunk.source_metadata,
        config_version=chunk.config_version,
        derived_keywords=chunk.derived_keywords,
        derived_entities=chunk.derived_entities,
        derived_questions=chunk.derived_questions,
        needs_semantic_split=False,
        token_count=chunk.token_count,
        named_vectors=chunk.named_vectors,
        embedding_model_versions=chunk.embedding_model_versions,
    )


def _replace_text(
    chunk: Chunk, new_text: str, counter: TokenCounter, ctx_reserve: int
) -> Chunk:
    """Return a copy of chunk with text replaced and token_count recalculated."""
    context_prefix = _extract_context_prefix(chunk, ctx_reserve, counter)
    new_context = (context_prefix + "\n" + new_text).strip() if context_prefix else new_text
    return Chunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        parent_id=chunk.parent_id,
        hierarchy_level=chunk.hierarchy_level,
        position=chunk.position,
        text=new_text,
        context_text=new_context,
        heading_path=chunk.heading_path,
        business_type=chunk.business_type,
        source_metadata=chunk.source_metadata,
        config_version=chunk.config_version,
        derived_keywords=chunk.derived_keywords,
        derived_entities=chunk.derived_entities,
        derived_questions=chunk.derived_questions,
        needs_semantic_split=False,
        token_count=counter.count(new_context),
    )


def _extract_context_prefix(
    chunk: Chunk, ctx_reserve: int, counter: TokenCounter
) -> str:
    """
    Re-derive the context prefix from the chunk's existing context_text.
    The prefix is context_text minus the trailing chunk.text portion.
    Returns empty string if ctx_reserve == 0 or prefix cannot be determined.
    """
    if ctx_reserve <= 0:
        return ""
    # If context_text starts with the text itself, there's no prefix
    if chunk.context_text.strip() == chunk.text.strip():
        return ""
    # The prefix is everything in context_text before the chunk text appears
    idx = chunk.context_text.find(chunk.text)
    if idx > 0:
        prefix = chunk.context_text[:idx].strip()
        return counter.truncate_to_tokens(prefix, ctx_reserve)
    # Fallback: use heading path
    if chunk.heading_path:
        candidate = " > ".join(chunk.heading_path)
        return counter.truncate_to_tokens(candidate, ctx_reserve)
    return ""
