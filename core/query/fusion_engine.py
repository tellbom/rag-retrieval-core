"""Service-layer Reciprocal Rank Fusion for query results."""

from __future__ import annotations

import logging
from dataclasses import replace

from core.config.models import FusionConfig, RetrieverConfig, RetrievalConfig
from core.query.retrieval_candidate import RetrievalCandidate
from core.query.retriever_pool import RetrieverResult

logger = logging.getLogger(__name__)


class FusionEngine:
    """Merge per-retriever ranked lists into one deduped RRF-ranked pool."""

    def __init__(
        self,
        fusion_cfg: FusionConfig,
        retriever_cfgs: list[RetrieverConfig],
    ) -> None:
        self._cfg = fusion_cfg
        self._weights = {retriever.id: retriever.weight for retriever in retriever_cfgs}

    def fuse(
        self,
        retriever_results: list[RetrieverResult],
        *,
        pool_top_k: int | None = None,
    ) -> list[RetrievalCandidate]:
        """Fuse retriever results with RRF or weighted RRF."""
        top_k = pool_top_k if pool_top_k is not None else self._cfg.pool_top_k

        merged: dict[str, RetrievalCandidate] = {}
        for result in retriever_results:
            for candidate in result.candidates:
                existing = merged.get(candidate.chunk_id)
                if existing is None:
                    merged[candidate.chunk_id] = candidate
                    continue
                merged[candidate.chunk_id] = _merge_candidate(existing, candidate)

        if not merged:
            return []

        scored: list[tuple[str, float]] = []
        for chunk_id, candidate in merged.items():
            rrf_score = self._score_candidate(candidate)
            scored.append((chunk_id, rrf_score))

        scored.sort(key=lambda item: item[1], reverse=True)

        fused = [
            replace(merged[chunk_id], rrf_score=rrf_score)
            for chunk_id, rrf_score in scored[:top_k]
        ]

        logger.debug(
            "FusionEngine: %d unique candidates from %d retriever(s), returning %d",
            len(merged),
            len(retriever_results),
            len(fused),
        )
        return fused

    @classmethod
    def from_retrieval_config(cls, retrieval_cfg: RetrievalConfig) -> "FusionEngine":
        return cls(
            fusion_cfg=retrieval_cfg.fusion,
            retriever_cfgs=retrieval_cfg.retrievers,
        )

    def _score_candidate(self, candidate: RetrievalCandidate) -> float:
        score = 0.0
        for retriever_id, rank in candidate.rank_in_retriever.items():
            weight = self._weights.get(retriever_id, 1.0)
            if self._cfg.method == "rrf":
                weight = 1.0
            score += weight / (self._cfg.k + rank)
        return score


def _merge_candidate(
    existing: RetrievalCandidate,
    incoming: RetrievalCandidate,
) -> RetrievalCandidate:
    """Merge two candidate records for the same chunk."""
    return replace(
        existing,
        bm25_score=(
            existing.bm25_score
            if existing.bm25_score is not None
            else incoming.bm25_score
        ),
        dense_scores={**incoming.dense_scores, **existing.dense_scores},
        highlight=existing.highlight if existing.highlight is not None else incoming.highlight,
        source_retriever_ids=(
            existing.source_retriever_ids | incoming.source_retriever_ids
        ),
        rank_in_retriever={
            **incoming.rank_in_retriever,
            **existing.rank_in_retriever,
        },
    )
