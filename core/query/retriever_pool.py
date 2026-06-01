"""Concurrent execution pool for configured retrievers."""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass

from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient

from core.config.models import AppConfig
from core.query.es_retriever import ESRetriever
from core.query.processed_query import ProcessedQuery
from core.query.qdrant_retriever import QdrantRetriever
from core.query.retrieval_candidate import RetrievalCandidate
from core.serving.registry import ServingRegistry

logger = logging.getLogger(__name__)

AnyRetriever = ESRetriever | QdrantRetriever


@dataclass
class RetrieverResult:
    """Candidates returned by one retriever."""

    retriever_id: str
    retriever_type: str
    weight: float
    candidates: list[RetrievalCandidate]


class RetrieverPool:
    """Run all configured retrievers concurrently while preserving config order."""

    def __init__(
        self,
        retrievers: list[AnyRetriever],
        *,
        max_workers: int | None = None,
    ) -> None:
        self._retrievers = retrievers
        self._max_workers = max_workers or max(len(retrievers), 1)

    @property
    def retriever_ids(self) -> list[str]:
        return [retriever.retriever_id for retriever in self._retrievers]

    def retrieve_all(
        self,
        query: ProcessedQuery,
        *,
        top_k_override: int | None = None,
    ) -> list[RetrieverResult]:
        """Run every retriever for a processed query."""

        def run_one(retriever: AnyRetriever) -> RetrieverResult:
            retriever_type = "lexical" if isinstance(retriever, ESRetriever) else "dense"
            try:
                candidates = retriever.retrieve(
                    query.effective_query,
                    query.filters,
                    top_k=top_k_override,
                )
            except Exception as exc:
                logger.error(
                    "RetrieverPool: retriever %s raised unexpectedly: %s",
                    retriever.retriever_id,
                    exc,
                )
                candidates = []

            return RetrieverResult(
                retriever_id=retriever.retriever_id,
                retriever_type=retriever_type,
                weight=retriever.weight,
                candidates=candidates,
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="retriever",
        ) as pool:
            futures = [pool.submit(run_one, retriever) for retriever in self._retrievers]
            return [future.result() for future in futures]

    @classmethod
    def from_config(
        cls,
        cfg: AppConfig,
        es_client: Elasticsearch,
        qdrant_client: QdrantClient,
        serving_registry: ServingRegistry,
        *,
        es_index: str,
        qdrant_collection: str,
    ) -> "RetrieverPool":
        """Build the pool from ordered ``retrieval.retrievers`` config."""
        retrievers: list[AnyRetriever] = []

        for retriever_cfg in cfg.retrieval.retrievers:
            if retriever_cfg.type == "lexical" and retriever_cfg.engine == "elasticsearch":
                retrievers.append(
                    ESRetriever(es=es_client, cfg=retriever_cfg, index=es_index)
                )
                continue

            if retriever_cfg.type == "dense" and retriever_cfg.engine == "qdrant":
                if retriever_cfg.model_id is None:
                    raise ValueError(
                        f"Dense retriever '{retriever_cfg.id}' is missing model_id"
                    )
                retrievers.append(
                    QdrantRetriever(
                        qdrant=qdrant_client,
                        embed_client=serving_registry.get_embedding_client(
                            retriever_cfg.model_id
                        ),
                        cfg=retriever_cfg,
                        collection=qdrant_collection,
                    )
                )
                continue

            logger.warning(
                "Skipping unsupported retriever %s (%s/%s)",
                retriever_cfg.id,
                retriever_cfg.type,
                retriever_cfg.engine,
            )

        if not retrievers:
            raise ValueError("No valid retrievers could be built from config")

        return cls(retrievers=retrievers)
