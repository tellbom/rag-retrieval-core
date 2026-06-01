"""Dense retriever backed by Qdrant named vectors."""

from __future__ import annotations

import logging
from typing import cast

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from core.config.models import RetrieverConfig
from core.query.filter_builder import build_qdrant_filter
from core.query.processed_query import QueryFilters
from core.query.retrieval_candidate import RetrievalCandidate
from core.serving.embed import EmbeddingClient, EmbeddingError

logger = logging.getLogger(__name__)

_PAYLOAD_FIELDS = [
    "chunk_id",
    "doc_id",
    "parent_id",
    "hierarchy_level",
    "position",
    "text",
    "business_type",
    "category",
    "config_version",
    "embedding_model_versions",
    "title",
    "source",
    "created_time",
    "updated_time",
    "author",
    "_enhanced",
]


class QdrantRetriever:
    """Dense semantic retriever for one configured Qdrant named vector."""

    def __init__(
        self,
        qdrant: QdrantClient,
        embed_client: EmbeddingClient,
        cfg: RetrieverConfig,
        collection: str,
    ) -> None:
        self._qdrant = qdrant
        self._embed_client = embed_client
        self._cfg = cfg
        self._collection = collection

    @property
    def retriever_id(self) -> str:
        return self._cfg.id

    @property
    def vector_name(self) -> str:
        return cast(str, self._cfg.vector_name)

    @property
    def weight(self) -> float:
        return self._cfg.weight

    def retrieve(
        self,
        query_text: str,
        filters: QueryFilters,
        *,
        top_k: int | None = None,
    ) -> list[RetrievalCandidate]:
        """Embed the query and run ANN search with Qdrant filter pushdown."""
        k = top_k if top_k is not None else self._cfg.top_k
        if not query_text.strip():
            return []

        try:
            query_vector = self._embed_client.embed([query_text])[0]
        except (EmbeddingError, IndexError) as exc:
            logger.error("QdrantRetriever %s embedding failed: %s", self.retriever_id, exc)
            return []

        try:
            results = self._qdrant.search(
                collection_name=self._collection,
                query_vector=(self.vector_name, query_vector),
                query_filter=build_qdrant_filter(filters),
                limit=k,
                with_payload=_PAYLOAD_FIELDS,
                with_vectors=False,
            )
        except Exception as exc:
            logger.error("QdrantRetriever %s search failed: %s", self.retriever_id, exc)
            return []

        return self._parse_results(results)

    def _parse_results(self, results: list[qmodels.ScoredPoint]) -> list[RetrievalCandidate]:
        candidates: list[RetrievalCandidate] = []

        for rank, point in enumerate(results, start=1):
            payload = point.payload or {}
            chunk_id = str(payload.get("chunk_id") or point.id)
            score = float(point.score)

            candidates.append(
                RetrievalCandidate(
                    chunk_id=chunk_id,
                    doc_id=str(payload.get("doc_id", "")),
                    text=str(payload.get("text", "")),
                    payload=dict(payload),
                    dense_scores={self.vector_name: score},
                    source_retriever_ids={self.retriever_id},
                    rank_in_retriever={self.retriever_id: rank},
                )
            )

        return candidates
