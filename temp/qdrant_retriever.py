"""
core/query/qdrant_retriever.py

QdrantRetriever: dense semantic retrieval from Qdrant using named vectors.

One QdrantRetriever instance per configured dense retriever entry.
Each instance uses a specific named vector space (one embedding model).

Query strategy
--------------
1. Embed the query text using the embedding client for this model.
2. Call `qdrant_client.search` with the query vector against the named
   vector space, with filter pushdown via `query_filter`.
3. Return top_k results as RetrievalCandidates with `dense_scores` set.

Filter pushdown
---------------
Qdrant filters are passed as `query_filter` to the search call.
They run BEFORE the ANN search narrows candidates, not after.
This is the correct behaviour per plan.md — no post-fusion filtering.

Named vector isolation
----------------------
Each retriever targets exactly one `vector_name` (e.g. "bge_base").
Multiple dense models each get their own QdrantRetriever instance.
Adding a model = adding one config entry + one QdrantRetriever — no code change.
"""

from __future__ import annotations

import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from core.config.models import RetrieverConfig
from core.query.filter_builder import build_qdrant_filter
from core.query.processed_query import QueryFilters
from core.query.retrieval_candidate import RetrievalCandidate
from core.serving.embed import EmbeddingClient, EmbeddingError

logger = logging.getLogger(__name__)

# Payload fields to retrieve from Qdrant (enough for context + citations)
_PAYLOAD_FIELDS = [
    "chunk_id", "doc_id", "parent_id", "hierarchy_level", "position",
    "text", "business_type", "category", "config_version",
    "embedding_model_versions", "title", "source", "created_time",
    "updated_time", "author", "_enhanced",
]


class QdrantRetriever:
    """
    Dense semantic retriever backed by Qdrant named vectors.

    Parameters
    ----------
    qdrant:        Raw QdrantClient.
    embed_client:  EmbeddingClient for this retriever's model.
    cfg:           RetrieverConfig (type=dense, engine=qdrant).
    collection:    Qdrant collection alias name.
    """

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
        return self._cfg.vector_name  # type: ignore[return-value]

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
        """
        Embed query_text and run ANN search in Qdrant.

        Parameters
        ----------
        query_text: effective_query from ProcessedQuery.
        filters:    Pushed into Qdrant search as query_filter.
        top_k:      Override config top_k.

        Returns
        -------
        List of RetrievalCandidates in cosine similarity order (highest first).
        Empty list on any error (logged, never raises).
        """
        k = top_k if top_k is not None else self._cfg.top_k

        if not query_text.strip():
            logger.debug(
                "QdrantRetriever %s: empty query, returning empty", self.retriever_id
            )
            return []

        # --- Embed query ---
        try:
            vectors = self._embed_client.embed([query_text])
            query_vector = vectors[0]
        except EmbeddingError as exc:
            logger.error(
                "QdrantRetriever %s: embedding failed: %s", self.retriever_id, exc
            )
            return []

        # --- Build filter ---
        qdrant_filter = build_qdrant_filter(filters)

        # --- ANN search ---
        try:
            results = self._qdrant.search(
                collection_name=self._collection,
                query_vector=(self.vector_name, query_vector),
                query_filter=qdrant_filter,
                limit=k,
                with_payload=_PAYLOAD_FIELDS,
                with_vectors=False,   # vectors not needed downstream
            )
        except Exception as exc:
            logger.error(
                "QdrantRetriever %s: search failed: %s", self.retriever_id, exc
            )
            return []

        return self._parse_results(results)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _parse_results(
        self, results: list[qmodels.ScoredPoint]
    ) -> list[RetrievalCandidate]:
        candidates: list[RetrievalCandidate] = []

        for rank, point in enumerate(results, start=1):
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id", str(point.id))
            doc_id   = payload.get("doc_id", "")
            score    = float(point.score)

            candidates.append(
                RetrievalCandidate(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=payload.get("text", ""),
                    payload=dict(payload),
                    dense_scores={self.vector_name: score},
                    source_retriever_ids={self.retriever_id},
                    rank_in_retriever={self.retriever_id: rank},
                )
            )

        logger.debug(
            "QdrantRetriever %s (%s): %d hits (top score=%.4f)",
            self.retriever_id,
            self.vector_name,
            len(candidates),
            candidates[0].dense_scores[self.vector_name] if candidates else 0.0,
        )
        return candidates
