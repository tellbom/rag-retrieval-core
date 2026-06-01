"""core.query — query pipeline components."""

from core.query.processed_query import ProcessedQuery, QueryFilters
from core.query.normalizer import QueryNormalizer
from core.query.rewriter import QueryRewriter
from core.query.preprocessor import QueryPreprocessor
from core.query.retrieval_candidate import RetrievalCandidate
from core.query.filter_builder import build_es_filter_clauses, build_qdrant_filter
from core.query.es_retriever import ESRetriever
from core.query.qdrant_retriever import QdrantRetriever
from core.query.retriever_pool import RetrieverPool, RetrieverResult
from core.query.fusion_engine import FusionEngine
from core.query.reranker import Reranker, RerankResult
from core.query.context_builder import BuiltContext, Citation, ContextBuilder
from core.query.answer_generator import (
    AnswerGenerationError,
    AnswerGenerator,
    GeneratedAnswer,
)

__all__ = [
    "ProcessedQuery",
    "QueryFilters",
    "QueryNormalizer",
    "QueryRewriter",
    "QueryPreprocessor",
    "RetrievalCandidate",
    "build_es_filter_clauses",
    "build_qdrant_filter",
    "ESRetriever",
    "QdrantRetriever",
    "RetrieverPool",
    "RetrieverResult",
    "FusionEngine",
    "Reranker",
    "RerankResult",
    "BuiltContext",
    "Citation",
    "ContextBuilder",
    "AnswerGenerationError",
    "AnswerGenerator",
    "GeneratedAnswer",
]
