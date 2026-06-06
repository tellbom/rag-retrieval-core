"""Factories for assembling ingestion and query pipelines."""

from __future__ import annotations

import logging

from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient

from core.config.models import AppConfig
from core.ingestion.cleaner import Cleaner
from core.ingestion.embedder import Embedder
from core.ingestion.enhancer import EnhancerFactory
from core.ingestion.semantic_chunker import SemanticChunker
from core.ingestion.structural_chunker import StructuralChunker
from core.pipeline.ingestion_pipeline import IngestionPipeline
from core.pipeline.query_pipeline import QueryPipeline, TopKLadder
from core.query.answer_generator import AnswerGenerator
from core.query.context_builder import ContextBuilder
from core.query.fusion_engine import FusionEngine
from core.query.preprocessor import QueryPreprocessor
from core.query.reranker import Reranker
from core.query.retriever_pool import RetrieverPool
from core.serving.registry import ServingRegistry
from core.storage.failed_index_store import FailedIndexStore
from core.storage.indexer import Indexer

logger = logging.getLogger(__name__)


class PipelineFactory:
    """Stateless factory for pipeline assembly."""

    @staticmethod
    def build_ingestion_pipeline(
        cfg: AppConfig,
        *,
        es: Elasticsearch,
        qdrant: QdrantClient,
        fail_store: FailedIndexStore,
        serving_registry: ServingRegistry,
        es_index: str,
        qdrant_collection: str,
        component_overrides: dict | None = None,
    ) -> IngestionPipeline:
        overrides = component_overrides or {}

        cleaner = overrides.get("cleaner") or Cleaner()
        enhancer = overrides.get("enhancer") or EnhancerFactory.from_config(cfg)
        structural_chunker = (
            overrides.get("structural_chunker") or StructuralChunker(cfg.chunking)
        )
        semantic_chunker = (
            overrides.get("semantic_chunker") or SemanticChunker(cfg.chunking)
        )
        embedder = overrides.get("embedder") or Embedder.from_config(
            cfg,
            serving_registry,
        )
        indexer = overrides.get("indexer") or Indexer(
            es=es,
            qdrant=qdrant,
            fail_store=fail_store,
            es_index=es_index,
            qdrant_collection=qdrant_collection,
        )

        pipeline = IngestionPipeline(
            cleaner=cleaner,
            enhancer=enhancer,
            structural_chunker=structural_chunker,
            semantic_chunker=semantic_chunker,
            embedder=embedder,
            indexer=indexer,
            config_version=cfg.version,
        )
        logger.info("IngestionPipeline built: %s", pipeline.component_names())
        return pipeline

    @staticmethod
    def build_query_pipeline(
        cfg: AppConfig,
        *,
        es: Elasticsearch,
        qdrant: QdrantClient,
        serving_registry: ServingRegistry,
        es_index: str,
        qdrant_collection: str,
        component_overrides: dict | None = None,
    ) -> QueryPipeline:
        overrides = component_overrides or {}

        preprocessor = (
            overrides.get("preprocessor") or QueryPreprocessor.from_config(cfg)
        )
        retriever_pool = overrides.get("retriever_pool") or RetrieverPool.from_config(
            cfg,
            es_client=es,
            qdrant_client=qdrant,
            serving_registry=serving_registry,
            es_index=es_index,
            qdrant_collection=qdrant_collection,
        )
        fusion_engine = (
            overrides.get("fusion_engine")
            or FusionEngine.from_retrieval_config(cfg.retrieval)
        )
        reranker = overrides.get("reranker") or Reranker.from_config(
            cfg,
            serving_registry,
        )
        context_builder = (
            overrides.get("context_builder") or ContextBuilder.from_config(cfg)
        )
        answer_generator = (
            overrides.get("answer_generator") or AnswerGenerator.from_config(cfg)
        )

        ladder = TopKLadder(
            recall_top_k=cfg.retrieval.top_k_ladder.recall_top_k,
            rrf_pool_k=cfg.retrieval.top_k_ladder.rrf_pool_k,
            rerank_top_k=cfg.retrieval.top_k_ladder.rerank_top_k,
            context_top_k=cfg.retrieval.top_k_ladder.context_top_k,
        )

        pipeline = QueryPipeline(
            preprocessor=preprocessor,
            retriever_pool=retriever_pool,
            fusion_engine=fusion_engine,
            reranker=reranker,
            context_builder=context_builder,
            answer_generator=answer_generator,
            ladder=ladder,
            min_rerank_score=cfg.retrieval.rerank.min_score,
            qdrant=qdrant,
            qdrant_collection=qdrant_collection,
        )
        logger.info("QueryPipeline built: %s", pipeline.component_names())
        return pipeline
