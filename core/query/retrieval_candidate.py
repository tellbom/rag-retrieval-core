"""Canonical retrieval candidate DTO shared by the query pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalCandidate:
    """One retrieved chunk with score provenance from every retrieval tier."""

    chunk_id: str
    doc_id: str

    text: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    highlight: list[str] | None = None

    bm25_score: float | None = None
    dense_scores: dict[str, float] = field(default_factory=dict)
    rrf_score: float | None = None
    rerank_score: float | None = None

    source_retriever_ids: set[str] = field(default_factory=set)
    rank_in_retriever: dict[str, int] = field(default_factory=dict)

    def primary_dense_score(self) -> float | None:
        """Return the first dense score, preserving insertion order."""
        if not self.dense_scores:
            return None
        return next(iter(self.dense_scores.values()))

    def has_rerank_score(self) -> bool:
        return self.rerank_score is not None

    def sort_key_rrf(self) -> float:
        return self.rrf_score if self.rrf_score is not None else float("-inf")

    def sort_key_rerank(self) -> float:
        if self.rerank_score is not None:
            return self.rerank_score
        return self.sort_key_rrf()

    def summary(self) -> str:
        bm25 = f"{self.bm25_score:.4f}" if self.bm25_score is not None else "None"
        return (
            f"chunk_id={self.chunk_id} "
            f"bm25={bm25} "
            f"dense={self.dense_scores} "
            f"rrf={self.rrf_score} "
            f"rerank={self.rerank_score}"
        )
