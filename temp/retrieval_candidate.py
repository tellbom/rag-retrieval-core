"""
core/query/retrieval_candidate.py

RetrievalCandidate: the canonical DTO for a single retrieved chunk,
carrying scores from every pipeline stage end-to-end.

Design rules
------------
- One instance per unique chunk_id in the pipeline.
- Scores are accumulated in-place as the chunk passes through stages:
    ES BM25 retriever   → bm25_score
    Qdrant dense        → dense_scores[vector_name]
    Fusion engine       → rrf_score
    Reranker            → rerank_score
- At every stage a score may be None (the chunk was not recalled by
  that retriever, or rerank has not run yet). None is explicit — never
  use 0.0 as a sentinel for "not scored".
- `text` and `payload` carry enough data to build context and citations
  without a second round-trip to ES or Qdrant.
- `source_retriever_ids` records which retriever(s) recalled this chunk
  (useful for debugging and eval).

This DTO is what FusionEngine, Reranker, ContextBuilder all work on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalCandidate:
    """
    One retrieved chunk with full score provenance.

    Identity
    --------
    chunk_id        : canonical chunk identifier (dedup key across retrievers)
    doc_id          : parent document identifier

    Content (from ES highlight or Qdrant payload)
    ------
    text            : chunk text (context_text stored at index time)
    payload         : full payload dict (metadata, filters, provenance fields)
    highlight       : ES highlight fragments (None if not from ES or no highlight)

    Scores (None = not yet computed / not recalled by this retriever)
    ------
    bm25_score      : raw ES BM25 score
    dense_scores    : {vector_name: float} — one per dense model that recalled it
    rrf_score       : service-layer RRF / weighted RRF score (set by FusionEngine)
    rerank_score    : cross-encoder score (set by Reranker)

    Provenance
    ----------
    source_retriever_ids : set of retriever ids that recalled this chunk
    rank_in_retriever    : {retriever_id: rank} — 1-based rank within each retriever
    """

    # Identity
    chunk_id: str
    doc_id: str

    # Content
    text: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    highlight: list[str] | None = None

    # Scores
    bm25_score: float | None = None
    dense_scores: dict[str, float] = field(default_factory=dict)
    rrf_score: float | None = None
    rerank_score: float | None = None

    # Provenance
    source_retriever_ids: set[str] = field(default_factory=set)
    rank_in_retriever: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def primary_dense_score(self) -> float | None:
        """Return the first (primary) dense score, or None if no dense scores."""
        if not self.dense_scores:
            return None
        return next(iter(self.dense_scores.values()))

    def has_rerank_score(self) -> bool:
        return self.rerank_score is not None

    def sort_key_rrf(self) -> float:
        """Sort key for RRF ordering (higher is better). -inf if no rrf_score."""
        return self.rrf_score if self.rrf_score is not None else float("-inf")

    def sort_key_rerank(self) -> float:
        """Sort key for rerank ordering (higher is better)."""
        if self.rerank_score is not None:
            return self.rerank_score
        return self.sort_key_rrf()

    def summary(self) -> str:
        return (
            f"chunk_id={self.chunk_id} "
            f"bm25={self.bm25_score:.4f} " if self.bm25_score is not None else
            f"chunk_id={self.chunk_id} bm25=None "
            f"dense={self.dense_scores} "
            f"rrf={self.rrf_score} rerank={self.rerank_score}"
        )
