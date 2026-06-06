"""BM25 retriever backed by Elasticsearch 7.x."""

from __future__ import annotations

import logging
from typing import Any

from elasticsearch import Elasticsearch

from core.config.models import RetrieverConfig
from core.query.filter_builder import build_es_filter_clauses
from core.query.processed_query import QueryFilters
from core.query.retrieval_candidate import RetrievalCandidate

logger = logging.getLogger(__name__)

_SOURCE_FIELDS = [
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
    "derived_keywords",
    "derived_entities",
    "derived_questions",
]

_HIGHLIGHT_CONFIG: dict[str, Any] = {
    "fields": {"text": {}},
    "pre_tags": ["<em>"],
    "post_tags": ["</em>"],
    "number_of_fragments": 3,
    "fragment_size": 150,
}


class ESRetriever:
    """BM25 lexical retriever with filter pushdown and highlight support."""

    def __init__(self, es: Elasticsearch, cfg: RetrieverConfig, index: str) -> None:
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
        """Run BM25 retrieval and return candidates in ES score order."""
        k = top_k if top_k is not None else self._cfg.top_k
        if not query_text.strip():
            return []

        try:
            response = self._es.search(
                index=self._index,
                body=self._build_query(query_text=query_text, filters=filters, top_k=k),
            )
        except Exception as exc:
            logger.error("ESRetriever %s failed: %s", self.retriever_id, exc)
            return []

        return self._parse_response(response)

    def _build_query(
        self,
        *,
        query_text: str,
        filters: QueryFilters,
        top_k: int,
    ) -> dict[str, Any]:
        bool_query: dict[str, Any] = {
            "must": [
                {
                    "multi_match": {
                        "query": query_text,
                        "fields": [
                            "text",
                            "title^3",
                            "derived_questions^2",
                            "derived_keywords^1.5",
                            "derived_entities^1.5",
                        ],
                        "type": "best_fields",
                    }
                }
            ]
        }

        filter_clauses = build_es_filter_clauses(filters)
        if filter_clauses:
            bool_query["filter"] = filter_clauses

        return {
            "size": top_k,
            "track_total_hits": True,
            "query": {"bool": bool_query},
            "highlight": _HIGHLIGHT_CONFIG,
            "_source": _SOURCE_FIELDS,
        }

    def _parse_response(self, response: dict[str, Any]) -> list[RetrievalCandidate]:
        candidates: list[RetrievalCandidate] = []

        for rank, hit in enumerate(response.get("hits", {}).get("hits", []), start=1):
            source = hit.get("_source", {})
            chunk_id = source.get("chunk_id") or hit.get("_id", "")
            highlight = hit.get("highlight", {}).get("text") or []

            candidates.append(
                RetrievalCandidate(
                    chunk_id=chunk_id,
                    doc_id=source.get("doc_id", ""),
                    text=source.get("text", ""),
                    payload=dict(source),
                    highlight=highlight or None,
                    bm25_score=float(hit.get("_score") or 0.0),
                    source_retriever_ids={self.retriever_id},
                    rank_in_retriever={self.retriever_id: rank},
                )
            )

        return candidates
