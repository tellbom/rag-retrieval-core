"""
core/query/es_retriever.py

ESRetriever: BM25 lexical retrieval from Elasticsearch 7.x.

Query strategy
--------------
Uses a `bool` query with:
  - `must`: `multi_match` on the `text` field (IK-analyzed, BM25 scoring).
  - `filter`: zero-score filter clauses from QueryFilters (pushed in, not post-applied).

Highlight is requested on the `text` field so the caller can surface the
matching fragment without a separate round-trip.

Returns top_k results as RetrievalCandidates with `bm25_score` set and
`source_retriever_ids` containing the retriever id from config.

ES 7.x caveats honoured
------------------------
- `track_total_hits: true` is set in the index mapping (P1-03); we rely on it.
- No `dense_vector` / kNN usage (not production-ready in 7.x).
- `keyword` fields (doc_id, business_type) are exact-match terms — the IK
  analyzer is not applied to them.
"""

from __future__ import annotations

import logging
from typing import Any

from elasticsearch import Elasticsearch

from core.config.models import RetrieverConfig
from core.query.filter_builder import build_es_filter_clauses
from core.query.processed_query import QueryFilters
from core.query.retrieval_candidate import RetrievalCandidate

logger = logging.getLogger(__name__)

# Fields to return in _source (enough to build context + citations)
_SOURCE_FIELDS = [
    "chunk_id", "doc_id", "parent_id", "hierarchy_level", "position",
    "text", "business_type", "category", "config_version",
    "embedding_model_versions", "title", "source", "created_time",
    "updated_time", "author",
]

_HIGHLIGHT_CONFIG: dict[str, Any] = {
    "fields": {"text": {}},
    "pre_tags": ["<em>"],
    "post_tags": ["</em>"],
    "number_of_fragments": 3,
    "fragment_size": 150,
}


class ESRetriever:
    """
    BM25 retriever backed by Elasticsearch 7.x.

    Parameters
    ----------
    es:       Raw Elasticsearch client.
    cfg:      RetrieverConfig (type=lexical, engine=elasticsearch).
    index:    ES index alias name.
    """

    def __init__(
        self,
        es: Elasticsearch,
        cfg: RetrieverConfig,
        index: str,
    ) -> None:
        self._es = es
        self._cfg = cfg
        self._index = index

    @property
    def retriever_id(self) -> str:
        return self._cfg.id

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
        Run BM25 retrieval for `query_text`.

        Parameters
        ----------
        query_text: The effective_query from ProcessedQuery.
        filters:    QueryFilters — pushed into `bool.filter` clauses.
        top_k:      Override the config top_k (used by ladder logic).

        Returns
        -------
        List of RetrievalCandidates in BM25 score order (highest first).
        Empty list on any error (logged, never raises).
        """
        k = top_k if top_k is not None else self._cfg.top_k

        if not query_text.strip():
            logger.debug("ESRetriever %s: empty query, returning empty", self.retriever_id)
            return []

        query_body = self._build_query(query_text, filters, k)

        try:
            resp = self._es.search(
                index=self._index,
                body=query_body,
                _source=_SOURCE_FIELDS,
            )
        except Exception as exc:
            logger.error("ESRetriever %s: search failed: %s", self.retriever_id, exc)
            return []

        return self._parse_response(resp)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_query(
        self,
        query_text: str,
        filters: QueryFilters,
        top_k: int,
    ) -> dict[str, Any]:
        filter_clauses = build_es_filter_clauses(filters)

        bool_query: dict[str, Any] = {
            "must": [
                {
                    "multi_match": {
                        "query": query_text,
                        "fields": ["text", "title^2"],   # boost title matches
                        "type": "best_fields",
                    }
                }
            ]
        }
        if filter_clauses:
            bool_query["filter"] = filter_clauses

        return {
            "size": top_k,
            "query": {"bool": bool_query},
            "highlight": _HIGHLIGHT_CONFIG,
            "_source": _SOURCE_FIELDS,
        }

    def _parse_response(self, resp: dict[str, Any]) -> list[RetrievalCandidate]:
        candidates: list[RetrievalCandidate] = []
        hits = resp.get("hits", {}).get("hits", [])

        for rank, hit in enumerate(hits, start=1):
            source = hit.get("_source", {})
            chunk_id = source.get("chunk_id") or hit.get("_id", "")
            doc_id   = source.get("doc_id", "")
            score    = float(hit.get("_score") or 0.0)

            # Extract highlight fragments
            hl_data  = hit.get("highlight", {})
            highlight = hl_data.get("text") or []

            candidates.append(
                RetrievalCandidate(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    text=source.get("text", ""),
                    payload=dict(source),
                    highlight=highlight if highlight else None,
                    bm25_score=score,
                    source_retriever_ids={self.retriever_id},
                    rank_in_retriever={self.retriever_id: rank},
                )
            )

        logger.debug(
            "ESRetriever %s: %d hits (top score=%.4f)",
            self.retriever_id,
            len(candidates),
            candidates[0].bm25_score if candidates else 0.0,
        )
        return candidates
