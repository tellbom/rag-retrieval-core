"""Ordered query pipeline runner."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.pipeline.protocols import (
    AnswerGeneratorProtocol,
    ContextBuilderProtocol,
    FusionEngineProtocol,
    PreprocessorProtocol,
    RerankerProtocol,
    RetrieverPoolProtocol,
)
from core.query.answer_generator import GeneratedAnswer
from core.query.context_builder import BuiltContext
from core.query.processed_query import QueryFilters
from core.query.retrieval_candidate import RetrievalCandidate

logger = logging.getLogger(__name__)


@dataclass
class TopKLadder:
    recall_top_k: int = 100
    rrf_pool_k: int = 200
    rerank_top_k: int = 50
    context_top_k: int = 8


@dataclass
class QueryPipelineResult:
    query: str
    processed_query_text: str
    answer: GeneratedAnswer
    reranked: bool
    retriever_candidate_counts: dict[str, int] = field(default_factory=dict)
    fused_count: int = 0
    rerank_input_count: int = 0

    def summary(self) -> str:
        return (
            f"query={self.query!r} "
            f"effective={self.processed_query_text!r} "
            f"reranked={self.reranked} "
            f"grounded={self.answer.grounded} "
            f"context_blocks={self.answer.context_used}"
        )


class QueryPipeline:
    """Run preprocess -> retrieve -> fuse -> rerank -> context -> answer."""

    def __init__(
        self,
        preprocessor: PreprocessorProtocol,
        retriever_pool: RetrieverPoolProtocol,
        fusion_engine: FusionEngineProtocol,
        reranker: RerankerProtocol,
        context_builder: ContextBuilderProtocol,
        answer_generator: AnswerGeneratorProtocol,
        *,
        ladder: TopKLadder,
        min_rerank_score: float | None = None,
        qdrant=None,
        qdrant_collection: str = "",
    ) -> None:
        self._preprocessor = preprocessor
        self._retriever_pool = retriever_pool
        self._fusion_engine = fusion_engine
        self._reranker = reranker
        self._context_builder = context_builder
        self._answer_generator = answer_generator
        self._ladder = ladder
        self._min_rerank_score = min_rerank_score
        self._qdrant = qdrant
        self._qdrant_collection = qdrant_collection

    def run(
        self,
        raw_query: str,
        *,
        filters: QueryFilters | None = None,
        business_type: str = "",
        enable_rewrite: bool = False,
        ladder_override: TopKLadder | None = None,
    ) -> QueryPipelineResult:
        ladder = ladder_override or self._ladder

        processed = self._preprocessor.process(
            raw_query,
            filters=filters,
            business_type=business_type,
            enable_rewrite=enable_rewrite,
        )
        retriever_results = self._retriever_pool.retrieve_all(
            processed,
            top_k_override=ladder.recall_top_k,
        )
        retriever_counts = {
            result.retriever_id: len(result.candidates)
            for result in retriever_results
        }

        fused = self._fusion_engine.fuse(
            retriever_results,
            pool_top_k=ladder.rrf_pool_k,
        )
        rerank_result = self._reranker.rerank(
            processed.effective_query,
            fused,
            top_k=ladder.rerank_top_k,
        )
        reranked_candidates = self._filter_by_min_rerank_score(
            rerank_result.candidates,
            reranked=rerank_result.reranked,
        )
        context: BuiltContext = self._context_builder.build(
            processed.effective_query,
            reranked_candidates,
            reranked=rerank_result.reranked,
            qdrant=self._qdrant,
            qdrant_collection=self._qdrant_collection,
        )
        answer: GeneratedAnswer = self._answer_generator.generate(
            processed.effective_query,
            context,
        )

        result = QueryPipelineResult(
            query=raw_query,
            processed_query_text=processed.effective_query,
            answer=answer,
            reranked=rerank_result.reranked,
            retriever_candidate_counts=retriever_counts,
            fused_count=len(fused),
            rerank_input_count=len(rerank_result.candidates),
        )
        logger.info("QueryPipeline: %s", result.summary())
        return result

    def _filter_by_min_rerank_score(
        self,
        candidates: list[RetrievalCandidate],
        *,
        reranked: bool,
    ) -> list[RetrievalCandidate]:
        if self._min_rerank_score is None or not reranked:
            return candidates

        kept = [
            candidate
            for candidate in candidates
            if candidate.rerank_score is not None
            and candidate.rerank_score >= self._min_rerank_score
        ]
        dropped = len(candidates) - len(kept)
        if dropped:
            logger.info(
                "QueryPipeline: dropped %d/%d candidate(s) below "
                "min_rerank_score=%.4f",
                dropped,
                len(candidates),
                self._min_rerank_score,
            )
        return kept

    def component_names(self) -> dict[str, str]:
        return {
            "preprocessor": type(self._preprocessor).__name__,
            "retriever_pool": type(self._retriever_pool).__name__,
            "fusion_engine": type(self._fusion_engine).__name__,
            "reranker": type(self._reranker).__name__,
            "context_builder": type(self._context_builder).__name__,
            "answer_generator": type(self._answer_generator).__name__,
        }
