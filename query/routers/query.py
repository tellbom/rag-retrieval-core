"""Main query endpoint for the online query service."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/query", tags=["query"])

_state: "_QueryRouterState | None" = None


class _QueryRouterState:
    def __init__(self, query_pipeline, iterative_retriever=None) -> None:
        self.pipeline = query_pipeline
        self.iterative_retriever = iterative_retriever


def init_query_router(query_pipeline, iterative_retriever=None) -> None:
    global _state
    _state = _QueryRouterState(
        query_pipeline,
        iterative_retriever=iterative_retriever,
    )


def _require_state() -> _QueryRouterState:
    if _state is None:
        raise HTTPException(status_code=503, detail="Query pipeline not initialised")
    return _state


class QueryFiltersRequest(BaseModel):
    business_type: str | None = None
    category: str | None = None
    doc_id: str | None = None
    created_after: str | None = None
    created_before: str | None = None
    extra: dict[str, str] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    business_type: str = ""
    filters: QueryFiltersRequest = Field(default_factory=QueryFiltersRequest)
    enable_rewrite: bool = False
    enable_iterative: bool = False


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    title: str
    source: str
    bm25_score: float | None
    dense_scores: dict[str, float]
    rrf_score: float | None
    rerank_score: float | None
    extra: dict


class QueryResponse(BaseModel):
    query: str
    effective_query: str
    answer: str
    grounded: bool
    reranked: bool
    citations: list[CitationOut]
    context_blocks_used: int
    llm_model: str
    retriever_candidate_counts: dict[str, int]
    fused_count: int
    rerank_input_count: int
    iterations: int = 1
    sub_queries: list[str] = Field(default_factory=list)
    iterative_enabled: bool = False
    self_eval_sufficient: bool | None = None
    self_eval_confidence: str = ""
    self_eval_missing: str = ""
    topic_absent: bool = False


@router.post("", response_model=QueryResponse, summary="Query the knowledge base")
def run_query(req: QueryRequest) -> QueryResponse:
    from core.query.processed_query import QueryFilters

    state = _require_state()
    filters = QueryFilters(
        business_type=req.filters.business_type,
        category=req.filters.category,
        doc_id=req.filters.doc_id,
        created_after=req.filters.created_after,
        created_before=req.filters.created_before,
        extra=req.filters.extra,
    )

    try:
        if req.enable_iterative and state.iterative_retriever is not None:
            iterative_result = state.iterative_retriever.run(
                req.query,
                filters=filters,
                business_type=req.business_type,
                enable_rewrite=req.enable_rewrite,
            )
            return QueryResponse(
                query=iterative_result.query,
                effective_query=iterative_result.processed_query_text,
                answer=iterative_result.answer_text,
                grounded=iterative_result.grounded,
                reranked=iterative_result.reranked,
                citations=[
                    CitationOut(
                        index=citation.index,
                        chunk_id=citation.chunk_id,
                        doc_id=citation.doc_id,
                        title=citation.title,
                        source=citation.source,
                        bm25_score=citation.bm25_score,
                        dense_scores=citation.dense_scores,
                        rrf_score=citation.rrf_score,
                        rerank_score=citation.rerank_score,
                        extra=citation.extra,
                    )
                    for citation in iterative_result.citations
                ],
                context_blocks_used=iterative_result.context_blocks_used,
                llm_model=iterative_result.llm_model,
                retriever_candidate_counts=iterative_result.retriever_candidate_counts,
                fused_count=iterative_result.fused_count,
                rerank_input_count=iterative_result.rerank_input_count,
                iterations=iterative_result.iterations,
                sub_queries=iterative_result.sub_queries,
                iterative_enabled=True,
                self_eval_sufficient=iterative_result.self_eval_sufficient,
                self_eval_confidence=iterative_result.self_eval_confidence,
                self_eval_missing=iterative_result.self_eval_missing,
                topic_absent=iterative_result.topic_absent,
            )

        pipeline_result = state.pipeline.run(
            req.query,
            filters=filters,
            business_type=req.business_type,
            enable_rewrite=req.enable_rewrite,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    answer = pipeline_result.answer
    return QueryResponse(
        query=pipeline_result.query,
        effective_query=pipeline_result.processed_query_text,
        answer=answer.answer,
        grounded=answer.grounded,
        reranked=pipeline_result.reranked,
        citations=[
            CitationOut(
                index=citation.index,
                chunk_id=citation.chunk_id,
                doc_id=citation.doc_id,
                title=citation.title,
                source=citation.source,
                bm25_score=citation.bm25_score,
                dense_scores=citation.dense_scores,
                rrf_score=citation.rrf_score,
                rerank_score=citation.rerank_score,
                extra=citation.extra,
            )
            for citation in answer.citations
        ],
        context_blocks_used=answer.context_used,
        llm_model=answer.llm_model,
        retriever_candidate_counts=pipeline_result.retriever_candidate_counts,
        fused_count=pipeline_result.fused_count,
        rerank_input_count=pipeline_result.rerank_input_count,
        iterations=1,
        sub_queries=[],
        iterative_enabled=False,
    )
