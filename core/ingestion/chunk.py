"""
core/ingestion/chunk.py

Chunk: the canonical data unit produced by the structural chunker and
consumed by every downstream stage (semantic chunker, embedder, indexer).

Design rules
------------
- `text` is the authoritative chunk text. Never mutated after creation.
- `context_text` is the padded version: context_padding + text (or just text
  if no padding). This is what gets embedded and indexed in ES/Qdrant.
  It is wider than `text` by up to `context_preservation_tokens`.
- `chunk_id` is a stable, deterministic identifier derived from doc_id +
  position within the document. Same input always produces same chunk_id.
- `needs_semantic_split` is set by the structural chunker when a chunk's
  token count exceeds the semantic trigger threshold. P1-07 acts on these.
- `embedding_model_versions` is populated by the Embedder (P1-08).
- `named_vectors` is populated by the Embedder (P1-08).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """
    One chunk of a document, ready for embedding and indexing.

    Identity fields
    ---------------
    chunk_id        : stable hash-based ID (doc_id + position)
    doc_id          : parent document identifier
    parent_id       : chunk_id of the structural parent, or None for root chunks
    hierarchy_level : 0 = document root, 1 = section, 2 = subsection, etc.
    position        : ordinal position among siblings at the same level

    Text fields
    -----------
    text            : authoritative canonical chunk text (never mutated)
    context_text    : text prepended with context_padding blurb; used for
                      embedding and ES full-text indexing
    heading_path    : list of ancestor headings for citation context
                      e.g. ["Chapter 3", "3.2 Safety Procedures"]

    Metadata (carried from EnhancedDocument)
    -----------------------------------------
    business_type, source_metadata, config_version

    LLM-derived doc-level fields (optional)
    ---------------------------------------
    derived_keywords, derived_entities, derived_questions
        Document-level retrieval hints copied from EnhancedDocument. All chunks
        from the same document carry the same values. None means enhancement was
        disabled, degraded, or that field was not generated.

    Pipeline state flags
    --------------------
    needs_semantic_split : True when token_count > semantic.min_trigger_tokens
                           P1-07 will split this chunk further
    token_count          : token count of `context_text` (set by chunker)

    Embedder fields (populated by P1-08)
    -------------------------------------
    named_vectors            : {vector_name: [float, ...]}
    embedding_model_versions : {model_id: version_string}

    Late Chunking fields (Phase 2, optional)
    -----------------------------------------
    lc_group_id  : Opaque key that groups sibling chunks sharing the same
                   Late Chunking "parent text".  None means this chunk does not
                   participate in Late Chunking and the Embedder must fall back
                   to ordinary /embed.
                   — StructuralChunker sets this to the structural parent's
                     chunk_id (or doc_id when parent_id is None).
                   — SemanticChunker sets this to "sem:<parent_chunk_id>" for
                     sub-chunks it produces.
                   — Chunks modified by _apply_overlap or truncated by the
                     length guardrail have lc_group_id=None (fallback).
    char_start   : Start character offset of chunk.text within the group text
                   built by late_chunking_utils.build_group_text() for this
                   lc_group_id.  None when lc_group_id is None.
    char_end     : Exclusive end character offset (char_start + len(text)).
                   None when lc_group_id is None.
    """

    # Identity
    chunk_id: str
    doc_id: str
    parent_id: str | None
    hierarchy_level: int
    position: int

    # Text
    text: str
    context_text: str           # text + context padding; what gets embedded
    heading_path: list[str]     # ancestor heading breadcrumb

    # Metadata
    business_type: str = ""
    source_metadata: dict = field(default_factory=dict)
    config_version: str = ""

    # LLM-derived doc-level fields (None = not generated / enhancement off)
    derived_keywords: list[str] | None = None
    derived_entities: list[str] | None = None
    derived_questions: list[str] | None = None

    # Pipeline state
    needs_semantic_split: bool = False
    token_count: int = 0        # token count of context_text

    # Embedder output (populated later)
    named_vectors: dict[str, list[float]] = field(default_factory=dict)
    embedding_model_versions: dict[str, str] = field(default_factory=dict)

    # Late Chunking metadata (Phase 2; None = not eligible, fallback to /embed)
    lc_group_id: str | None = None
    char_start: int | None = None
    char_end: int | None = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def make_chunk_id(doc_id: str, hierarchy_level: int, position: int) -> str:
        """
        Deterministic chunk_id: SHA-1 of "{doc_id}:{hierarchy_level}:{position}".
        Stable across re-runs given the same doc_id and position.
        """
        raw = f"{doc_id}:{hierarchy_level}:{position}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def summary(self) -> str:
        return (
            f"chunk_id={self.chunk_id} doc_id={self.doc_id} "
            f"level={self.hierarchy_level} pos={self.position} "
            f"tokens={self.token_count} "
            f"needs_split={self.needs_semantic_split} "
            f"text_len={len(self.text)}"
        )


@dataclass
class ChunkingResult:
    """
    Output of the structural chunker for one document.

    chunks            : all chunks in document order (DFS traversal)
    needs_split_count : number of chunks flagged for semantic splitting
    """
    doc_id: str
    chunks: list[Chunk]
    needs_split_count: int = 0

    @property
    def total(self) -> int:
        return len(self.chunks)
