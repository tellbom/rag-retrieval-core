"""
core/query/retriever_pool.py

RetrieverPool: runs all configured retrievers concurrently and returns
a list of per-retriever result lists, preserving config order.

Design
------
- ES BM25 and each Qdrant dense retriever run in parallel via
  ThreadPoolExecutor.  On CPU-only infrastructure this is IO-bound
  (HTTP calls), so threading gives real concurrency.
- A retriever that fails returns an empty list (logged); the pool
  never raises.  A single retriever failure does not abort the pipeline.
- Config order is preserved — the list index matches the retriever
  position in `retrieval.retrievers`, which matters for deterministic
  tie-breaking in fusion.
- Adding/removing a retriever = one config change; RetrieverPool rebuilds
  from config at startup.

Factory
-------
RetrieverPool.from_config(cfg, es_client, qdrant_client, serving_registry)
"""

from __future__ import annotations

import logging
import concurrent.futures
from dataclasses import dataclass

from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient

from core.config.models import AppConfig, RetrieverConfig
from core.query.es_retriever import ESRetriever
from core.query.processed_query import ProcessedQuery
from core.query.qdrant_retriever import QdrantRetriever
from core.query.retrieval_candidate import RetrievalCandidate
from core.serving.registry import ServingRegistry

logger = logging.getLogger(__name__)

# Type alias for a single retriever (either ES or Qdrant)
AnyRetriever = ESRetriever | QdrantRetriever


@dataclass
class RetrieverResult:
    """Result from one retriever."""
    retriever_id: str
    retriever_type: str          # 'lexical' | 'dense'
    weight: float
    candidates: list[RetrievalCandidate]


class RetrieverPool:
    """
    Manages and concurrently executes all configured retrievers.

    Parameters
    ----------
    retrievers:
        Ordered list of retriever instances matching the config order.
    max_workers:
        Thread pool size.  Defaults to number of retrievers (all run in parallel).
    """

    def __init__(
        self,
        retrievers: list[AnyRetriever],
        *,
        max_workers: int | None = None,
    ) -> None:
        self._retrievers = retrievers
        self._max_workers = max_workers or max(len(retrievers), 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve_all(
        self,
        query: ProcessedQuery,
        *,
        top_k_override: int | None = None,
    ) -> list[RetrieverResult]:
        """
        Run all retrievers concurrently for the given ProcessedQuery.

        Parameters
        ----------
        query:          ProcessedQuery from the preprocessor.
        top_k_override: If set, overrides each retriever's configured top_k.
                        Used when the ladder's `recall_top_k` differs from
                        per-retriever defaults.

        Returns
        -------
        List of RetrieverResult in config order (same order as retrievers list).
        A retriever that errors returns an empty-candidates RetrieverResult.
        """
        query_text = query.effective_query
        filters = query.filters

        def run_one(retriever: AnyRetriever) -> RetrieverResult:
            try:
                candidates = retriever.retrieve(
                    query_text,
                    filters,
                    top_k=top_k_override,
                )
                retriever_type = (
                    "lexical" if isinstance(retriever, ESRetriever) else "dense"
                )
                return RetrieverResult(
                    retriever_id=retriever.retriever_id,
                    retriever_type=retriever_type,
                    weight=retriever.weight,
                    candidates=candidates,
                )
            except Exception as exc:
                logger.error(
                    "RetrieverPool: retriever %s raised unexpectedly: %s",
                    getattr(retriever, "retriever_id", "?"),
                    exc,
                )
                return RetrieverResult(
                    retriever_id=getattr(retriever, "retriever_id", "?"),
                    retriever_type="unknown",
                    weight=1.0,
                    candidates=[],
                )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="retriever",
        ) as pool:
            # Submit in config order, preserve that order in results
            futures = [pool.submit(run_one, r) for r in self._retrievers]
            results = [f.result() for f in futures]

        total = sum(len(r.candidates) for r in results)
        logger.debug(
            "RetrieverPool: %d retriever(s), %d total candidates for query=%r",
            len(results),
            total,
            query_text[:60],
        )
        return results

    @property
    def retriever_ids(self) -> list[str]:
        return [r.retriever_id for r in self._retrievers]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

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
        """
        Build a RetrieverPool from AppConfig.

        Adding/removing a retriever in config automatically changes the pool
        at next service startup — no code changes required.
        """
        retrievers: list[AnyRetriever] = []

        for r_cfg in cfg.retrieval.retrievers:
            if r_cfg.type == "lexical" and r_cfg.engine == "elasticsearch":
                retrievers.append(
                    ESRetriever(
                        es=es_client,
                        cfg=r_cfg,
                        index=es_index,
                    )
                )
            elif r_cfg.type == "dense" and r_cfg.engine == "qdrant":
                embed_client = serving_registry.get_embedding_client(r_cfg.model_id)  # type: ignore[arg-type]
                retrievers.append(
                    QdrantRetriever(
                        qdrant=qdrant_client,
                        embed_client=embed_client,
                        cfg=r_cfg,
                        collection=qdrant_collection,
                    )
                )
            else:
                logger.warning(
                    "RetrieverPool: unknown retriever type/engine combo "
                    "(%s/%s) for id=%s — skipping",
                    r_cfg.type, r_cfg.engine, r_cfg.id,
                )

        if not retrievers:
            raise ValueError(
                "RetrieverPool: no valid retrievers could be built from config. "
                "Check retrieval.retrievers in the config file."
            )

        logger.info(
            "RetrieverPool built: %d retriever(s) %s",
            len(retrievers),
            [r.retriever_id for r in retrievers],
        )
        return cls(retrievers=retrievers)
