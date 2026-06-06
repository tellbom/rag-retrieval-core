from __future__ import annotations

from core.pipeline.query_pipeline import QueryPipeline, TopKLadder
from core.query.answer_generator import AnswerGenerator, GeneratedAnswer
from core.query.context_builder import BuiltContext
from core.query.processed_query import ProcessedQuery, QueryFilters
from core.query.reranker import RerankResult
from core.query.retrieval_candidate import RetrievalCandidate
from core.query.retriever_pool import RetrieverResult


class _Preprocessor:
    def process(
        self,
        raw_query: str,
        *,
        filters: QueryFilters | None,
        business_type: str,
        enable_rewrite: bool,
    ) -> ProcessedQuery:
        return ProcessedQuery(
            original_query=raw_query,
            normalized_query=raw_query,
            filters=filters or QueryFilters(),
            business_type=business_type,
        )


class _RetrieverPool:
    def retrieve_all(
        self,
        query: ProcessedQuery,
        *,
        top_k_override: int | None,
    ) -> list[RetrieverResult]:
        return [
            RetrieverResult(
                retriever_id="fake",
                retriever_type="lexical",
                weight=1.0,
                candidates=[],
            )
        ]


class _FusionEngine:
    def __init__(self, candidates: list[RetrievalCandidate]) -> None:
        self._candidates = candidates

    def fuse(
        self,
        retriever_results: list[RetrieverResult],
        *,
        pool_top_k: int | None,
    ) -> list[RetrievalCandidate]:
        return self._candidates


class _Reranker:
    def __init__(self, candidates: list[RetrievalCandidate]) -> None:
        self._candidates = candidates

    def rerank(
        self,
        query_text: str,
        fused_candidates: list[RetrievalCandidate],
        *,
        top_k: int | None,
    ) -> RerankResult:
        return RerankResult(reranked=True, candidates=self._candidates)


class _ContextBuilder:
    def __init__(self) -> None:
        self.seen_candidates: list[RetrievalCandidate] | None = None

    def build(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        *,
        reranked: bool,
        qdrant,
        qdrant_collection: str,
    ) -> BuiltContext:
        self.seen_candidates = candidates
        return BuiltContext(
            context_text="",
            citations=[],
            is_empty=not candidates,
            candidate_count=len(candidates),
            reranked=reranked,
        )


class _AnswerGenerator:
    def generate(self, query: str, context: BuiltContext) -> GeneratedAnswer:
        if context.is_empty:
            return GeneratedAnswer(
                answer="根据现有资料无法回答该问题",
                citations=[],
                grounded=False,
                context_used=0,
                reranked=context.reranked,
            )
        return GeneratedAnswer(
            answer="ok",
            citations=context.citations,
            grounded=True,
            context_used=context.candidate_count,
            reranked=context.reranked,
        )


def test_min_rerank_score_filters_low_scored_candidates_to_empty_context() -> None:
    candidate = RetrievalCandidate(
        chunk_id="chunk-1",
        doc_id="doc-1",
        text="irrelevant",
        rerank_score=0.12,
    )
    context_builder = _ContextBuilder()
    pipeline = QueryPipeline(
        preprocessor=_Preprocessor(),
        retriever_pool=_RetrieverPool(),
        fusion_engine=_FusionEngine([candidate]),
        reranker=_Reranker([candidate]),
        context_builder=context_builder,
        answer_generator=_AnswerGenerator(),
        ladder=TopKLadder(recall_top_k=10, rrf_pool_k=10, rerank_top_k=10, context_top_k=8),
        min_rerank_score=0.5,
    )

    result = pipeline.run("question", business_type="news")

    assert context_builder.seen_candidates == []
    assert result.answer.answer == "根据现有资料无法回答该问题"
    assert result.answer.citations == []
    assert result.answer.grounded is False


def test_answer_generator_empty_context_uses_fixed_insufficient_answer() -> None:
    class _LLMClient:
        _model = "fake"

        def chat(self, *args, **kwargs) -> str:
            raise AssertionError("empty context should not call the LLM")

    generator = AnswerGenerator(_LLMClient())
    answer = generator.generate(
        "question",
        BuiltContext(context_text="", citations=[], is_empty=True),
    )

    assert answer.answer == "根据现有资料无法回答该问题"
    assert answer.citations == []
    assert answer.grounded is False
