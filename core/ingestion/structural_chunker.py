"""
core/ingestion/structural_chunker.py

StructuralChunker: converts a parsed document into Chunks with:
  - Parent-child hierarchy (parent_id, hierarchy_level)
  - Contextual padding: context_preservation_tokens prepended to child chunks
  - Length guardrail: hard max_tokens enforced in tokens (not chars)
  - overlap_tokens between adjacent sibling chunks
  - needs_semantic_split flag set for chunks exceeding semantic trigger threshold

Pipeline position
-----------------
    EnhancedDocument
        → StructuralParser  → list[StructuralNode]
        → StructuralChunker → ChunkingResult
        → SemanticChunker   (P1-07, acts on needs_semantic_split=True chunks)
        → Embedder          (P1-08)

Chunking algorithm
------------------
1. Parse nodes from StructuralParser (ordered, flat list).
2. Walk nodes:
   - Heading nodes: create a parent chunk; subsequent non-heading nodes are
     children of the most recent heading at an equal or higher level.
   - Non-heading nodes (paragraph, table, list, step, clause): create leaf chunks
     as children of the current heading context.
3. For each chunk:
   a. Count tokens of the raw text.
   b. If token_count > max_tokens - context_preservation_tokens:
      → Hard-truncate to max_tokens - context_preservation_tokens.
   c. Build context_text = context_padding_prefix + text, where prefix is
      the context_preservation_tokens-length excerpt of the parent heading
      text (or the document-level context_padding derived field if present).
   d. If token_count > semantic.min_trigger_tokens: set needs_semantic_split=True.
4. Overlapping adjacent sibling chunks: append the last overlap_tokens tokens
   of chunk N as a prefix to chunk N+1 (if both are leaf paragraphs).

Context padding source priority
---------------------------------
1. EnhancedDocument.context_padding (LLM-generated document context blurb)
2. Nearest ancestor heading text
3. doc_id (fallback)
"""

from __future__ import annotations

import logging

from core.config.models import ChunkingConfig
from core.ingestion.chunk import Chunk, ChunkingResult
from core.ingestion.enhanced_document import EnhancedDocument
from core.ingestion.late_chunking_utils import GROUP_SEP, build_group_text
from core.ingestion.structural_parser import NodeType, StructuralNode, StructuralParser
from core.ingestion.token_counter import TokenCounter

logger = logging.getLogger(__name__)


class StructuralChunker:
    """
    Converts an EnhancedDocument into a ChunkingResult.

    Parameters
    ----------
    cfg:        ChunkingConfig (length, structural, semantic settings)
    counter:    TokenCounter instance (shared; stateless)
    parser:     StructuralParser instance (shared; stateless)
    """

    def __init__(
        self,
        cfg: ChunkingConfig,
        counter: TokenCounter | None = None,
        parser: StructuralParser | None = None,
    ) -> None:
        self._cfg = cfg
        self._length = cfg.length
        self._structural = cfg.structural
        self._semantic = cfg.semantic
        self._counter = counter or TokenCounter()
        self._parser = parser or StructuralParser(
            enabled_levels=list(self._structural.levels)
        )

        # Effective token budget for a single chunk's own text
        # (reserves room for context_preservation_tokens)
        self._text_token_budget = (
            self._length.max_tokens - self._length.context_preservation_tokens
        )
        if self._text_token_budget <= 0:
            raise ValueError(
                f"max_tokens ({self._length.max_tokens}) must be greater than "
                f"context_preservation_tokens ({self._length.context_preservation_tokens})"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, doc: EnhancedDocument, config_version: str = "") -> ChunkingResult:
        """
        Chunk one EnhancedDocument.

        Parameters
        ----------
        doc:            The document to chunk.
        config_version: Stamped onto each Chunk for provenance tracking.

        Returns
        -------
        ChunkingResult with all chunks in document order.
        """
        if not doc.text.strip():
            return ChunkingResult(doc_id=doc.doc_id, chunks=[], needs_split_count=0)

        nodes = self._parser.parse(doc.text)
        chunks = self._build_chunks(doc, nodes, config_version)
        needs_split = sum(1 for c in chunks if c.needs_semantic_split)

        logger.debug(
            "Chunked doc_id=%s: %d chunks, %d need semantic split",
            doc.doc_id, len(chunks), needs_split,
        )
        return ChunkingResult(
            doc_id=doc.doc_id,
            chunks=chunks,
            needs_split_count=needs_split,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_chunks(
        self,
        doc: EnhancedDocument,
        nodes: list[StructuralNode],
        config_version: str,
    ) -> list[Chunk]:
        """Walk nodes and produce Chunks with parent-child links."""
        chunks: list[Chunk] = []

        # Heading stack: list of (chunk_id, heading_text, level)
        heading_stack: list[tuple[str, str, int]] = []
        # heading_path for breadcrumb context
        heading_path: list[str] = []

        # Document-level context blurb (from LLM enhancement or fallback)
        doc_context_blurb = doc.context_padding or ""

        position_counter = 0  # global position across all chunks

        # sibling-level position per (parent_id, level) key
        sibling_positions: dict[tuple[str | None, int], int] = {}

        # Late Chunking: accumulated sibling text length per lc_group_id.
        # Used to compute char_start/char_end without post-processing.
        # key = lc_group_id, value = current accumulated length (chars)
        # After writing a sibling we add len(text) + len(GROUP_SEP).
        lc_group_offset: dict[str, int] = {}

        for node in nodes:
            is_heading = node.node_type in (NodeType.HEADING, NodeType.CLAUSE)

            # --- Determine parent ---
            parent_id: str | None = None
            level = node.level

            if is_heading:
                # Pop stack until we find a heading of strictly lower level
                while heading_stack and heading_stack[-1][2] >= level:
                    heading_stack.pop()
                parent_id = heading_stack[-1][0] if heading_stack else None
            else:
                # Leaf: parent is the current innermost heading
                parent_id = heading_stack[-1][0] if heading_stack else None

            # --- Sibling position ---
            sib_key = (parent_id, level)
            pos = sibling_positions.get(sib_key, 0)
            sibling_positions[sib_key] = pos + 1
            position_counter += 1

            # --- Build context_text ---
            context_prefix = self._build_context_prefix(
                doc_context_blurb, heading_path, node
            )
            node_text = node.text.strip()

            # --- Apply length guardrail to the node's own text ---
            was_truncated = self._counter.count(node_text) > self._text_token_budget
            node_text = self._apply_length_guardrail(node_text)

            # --- Build context_text (prefix + node text) ---
            context_text = (context_prefix + "\n" + node_text).strip() if context_prefix else node_text

            # --- Count tokens of context_text ---
            token_count = self._counter.count(context_text)

            # --- chunk_id ---
            chunk_id = Chunk.make_chunk_id(doc.doc_id, level, position_counter)

            # --- needs_semantic_split ---
            needs_split = (
                self._semantic.enabled
                and self._counter.count(node_text) > self._semantic.min_trigger_tokens
            )

            # --- Late Chunking: compute lc_group_id / char_start / char_end ---
            # lc_group_id is the structural parent's chunk_id (or doc_id at root).
            # char offsets are relative to build_group_text() over all siblings
            # with the same lc_group_id, in position order.
            # Truncated chunks cannot have reliable offsets → set None (fallback).
            lc_group_id: str | None
            lc_char_start: int | None
            lc_char_end: int | None

            if was_truncated:
                # Truncated text is not a faithful slice of the original node;
                # char offsets would be unreliable.
                lc_group_id = None
                lc_char_start = None
                lc_char_end = None
            else:
                lc_group_id = parent_id if parent_id is not None else doc.doc_id
                current_offset = lc_group_offset.get(lc_group_id, 0)
                lc_char_start = current_offset
                lc_char_end = current_offset + len(node_text)
                # Advance offset: text length + separator (GROUP_SEP) for next sibling
                lc_group_offset[lc_group_id] = lc_char_end + len(GROUP_SEP)

            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                parent_id=parent_id,
                hierarchy_level=level,
                position=pos,
                text=node_text,
                context_text=context_text,
                heading_path=list(heading_path),
                business_type=doc.business_type,
                source_metadata=doc.source_metadata,
                config_version=config_version,
                derived_keywords=doc.keywords,
                derived_entities=doc.entities,
                derived_questions=doc.potential_questions,
                needs_semantic_split=needs_split,
                token_count=token_count,
                lc_group_id=lc_group_id,
                char_start=lc_char_start,
                char_end=lc_char_end,
            )
            chunks.append(chunk)

            # --- Update heading stack / path ---
            if is_heading:
                heading_stack.append((chunk_id, node.heading or node_text, level))
                # Rebuild heading_path from stack
                heading_path = [h for _, h, _ in heading_stack]

        # --- Apply overlap between adjacent leaf siblings ---
        chunks = self._apply_overlap(chunks)

        return chunks

    def _build_context_prefix(
        self,
        doc_blurb: str,
        heading_path: list[str],
        node: StructuralNode,
    ) -> str:
        """
        Build the context prefix that is prepended to a chunk's text.
        Budget: context_preservation_tokens.
        Priority: doc_blurb > heading breadcrumb > none.
        """
        if not self._length.context_preservation_tokens:
            return ""

        if doc_blurb:
            candidate = doc_blurb
        elif heading_path:
            candidate = " > ".join(heading_path)
        else:
            return ""

        # Truncate the prefix to fit within context_preservation_tokens
        return self._counter.truncate_to_tokens(
            candidate, self._length.context_preservation_tokens
        )

    def _apply_length_guardrail(self, text: str) -> str:
        """
        Hard-truncate text to fit within the effective token budget
        (max_tokens minus context_preservation_tokens reserve).
        """
        if self._counter.count(text) <= self._text_token_budget:
            return text
        truncated = self._counter.truncate_to_tokens(text, self._text_token_budget)
        logger.debug(
            "Chunk truncated from %d to %d tokens (budget=%d)",
            self._counter.count(text),
            self._counter.count(truncated),
            self._text_token_budget,
        )
        return truncated

    def _apply_overlap(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Prepend the last overlap_tokens of chunk[i-1] to chunk[i]
        when they are adjacent leaf siblings at the same level with the same parent.
        Heading chunks are never overlapped.
        """
        overlap = self._length.overlap_tokens
        if overlap <= 0 or len(chunks) < 2:
            return chunks

        result = list(chunks)
        for i in range(1, len(result)):
            prev = result[i - 1]
            curr = result[i]

            # Only overlap leaf nodes at the same level with same parent
            if (
                prev.parent_id != curr.parent_id
                or prev.hierarchy_level != curr.hierarchy_level
                or curr.needs_semantic_split  # P1-07 will handle these
            ):
                continue

            # Don't overlap headings
            if prev.hierarchy_level <= 1:
                continue

            # Get the last `overlap` tokens of the previous chunk's text
            overlap_text = self._counter.token_suffix(prev.text, overlap)
            if not overlap_text:
                continue

            new_text = (overlap_text + "\n" + curr.text).strip()
            new_context = (
                (curr.context_text.replace(curr.text, new_text, 1))
                if curr.text in curr.context_text
                else new_text
            )
            new_token_count = self._counter.count(new_context)

            # Only apply if it still fits within max_tokens
            if new_token_count <= self._length.max_tokens:
                result[i] = Chunk(
                    chunk_id=curr.chunk_id,
                    doc_id=curr.doc_id,
                    parent_id=curr.parent_id,
                    hierarchy_level=curr.hierarchy_level,
                    position=curr.position,
                    text=new_text,
                    context_text=new_context,
                    heading_path=curr.heading_path,
                    business_type=curr.business_type,
                    source_metadata=curr.source_metadata,
                    config_version=curr.config_version,
                    derived_keywords=curr.derived_keywords,
                    derived_entities=curr.derived_entities,
                    derived_questions=curr.derived_questions,
                    needs_semantic_split=curr.needs_semantic_split,
                    token_count=new_token_count,
                    # Overlap prepends text from prev chunk → text is no longer
                    # a contiguous slice of any single group_text; disable LC.
                    lc_group_id=None,
                    char_start=None,
                    char_end=None,
                )

        return result

