"""Component protocols for ingestion and query pipelines."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from core.ingestion.chunk import ChunkingResult
from core.ingestion.cleaning_record import CleanedDocument
from core.ingestion.enhanced_document import EnhancedDocument
from core.query.answer_generator import GeneratedAnswer
from core.query.context_builder import BuiltContext
from core.query.processed_query import ProcessedQuery, QueryFilters
from core.query.reranker import RerankResult
from core.query.retrieval_candidate import RetrievalCandidate
from core.query.retriever_pool import RetrieverResult
from core.storage.indexer import IndexResult


@runtime_checkable
class CleanerProtocol(Protocol):
    def clean(
        self,
        doc_id: str,
        raw_text: str,
        *,
        business_type: str,
        source_metadata: dict | None,
    ) -> CleanedDocument: ...


@runtime_checkable
class EnhancerProtocol(Protocol):
    def enhance(self, cleaned: CleanedDocument) -> EnhancedDocument: ...


@runtime_checkable
class StructuralChunkerProtocol(Protocol):
    def chunk(
        self,
        doc: EnhancedDocument,
        config_version: str,
    ) -> ChunkingResult: ...


@runtime_checkable
class SemanticChunkerProtocol(Protocol):
    def split(self, result: ChunkingResult) -> ChunkingResult: ...


@runtime_checkable
class EmbedderProtocol(Protocol):
    def embed(self, result: ChunkingResult) -> ChunkingResult: ...


@runtime_checkable
class IndexerProtocol(Protocol):
    def index(
        self,
        result: ChunkingResult,
        *,
        enhanced: bool,
    ) -> IndexResult: ...


@runtime_checkable
class PreprocessorProtocol(Protocol):
    def process(
        self,
        raw_query: str,
        *,
        filters: QueryFilters | None,
        business_type: str,
        enable_rewrite: bool,
    ) -> ProcessedQuery: ...


@runtime_checkable
class RetrieverPoolProtocol(Protocol):
    def retrieve_all(
        self,
        query: ProcessedQuery,
        *,
        top_k_override: int | None,
    ) -> list[RetrieverResult]: ...


@runtime_checkable
class FusionEngineProtocol(Protocol):
    def fuse(
        self,
        retriever_results: list[RetrieverResult],
        *,
        pool_top_k: int | None,
    ) -> list[RetrievalCandidate]: ...


@runtime_checkable
class RerankerProtocol(Protocol):
    def rerank(
        self,
        query_text: str,
        fused_candidates: list[RetrievalCandidate],
        *,
        top_k: int | None,
    ) -> RerankResult: ...


@runtime_checkable
class ContextBuilderProtocol(Protocol):
    def build(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        *,
        reranked: bool,
        qdrant: Any,
        qdrant_collection: str,
    ) -> BuiltContext: ...


@runtime_checkable
class AnswerGeneratorProtocol(Protocol):
    def generate(
        self,
        query: str,
        context: BuiltContext,
    ) -> GeneratedAnswer: ...


class ComponentAdapter:
    """Base marker for future third-party component adapters."""

    def component_name(self) -> str:
        return self.__class__.__name__
