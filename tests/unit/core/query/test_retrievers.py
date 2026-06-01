from __future__ import annotations

import sys
import types


def _install_client_stubs() -> None:
    if "elasticsearch" not in sys.modules:
        elasticsearch = types.ModuleType("elasticsearch")
        elasticsearch.Elasticsearch = object
        sys.modules["elasticsearch"] = elasticsearch

    if "qdrant_client" not in sys.modules:
        qdrant_client = types.ModuleType("qdrant_client")
        qdrant_client.QdrantClient = object

        http = types.ModuleType("qdrant_client.http")
        models = types.ModuleType("qdrant_client.http.models")

        class MatchValue:
            def __init__(self, value):
                self.value = value

        class DatetimeRange:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FieldCondition:
            def __init__(self, key, match=None, range=None):
                self.key = key
                self.match = match
                self.range = range

        class Filter:
            def __init__(self, must):
                self.must = must

        class ScoredPoint:
            pass

        models.MatchValue = MatchValue
        models.DatetimeRange = DatetimeRange
        models.FieldCondition = FieldCondition
        models.Filter = Filter
        models.ScoredPoint = ScoredPoint
        http.models = models
        qdrant_client.http = http

        sys.modules["qdrant_client"] = qdrant_client
        sys.modules["qdrant_client.http"] = http
        sys.modules["qdrant_client.http.models"] = models


_install_client_stubs()

from core.config.models import RetrieverConfig
from core.query.es_retriever import ESRetriever
from core.query.filter_builder import build_es_filter_clauses, build_qdrant_filter
from core.query.processed_query import ProcessedQuery, QueryFilters
from core.query.qdrant_retriever import QdrantRetriever
from core.query.retrieval_candidate import RetrievalCandidate
from core.query.retriever_pool import RetrieverPool


def test_retrieval_candidate_keeps_score_tiers_explicit() -> None:
    candidate = RetrievalCandidate(
        chunk_id="c1",
        doc_id="d1",
        dense_scores={"bge_base": 0.72},
    )

    assert candidate.bm25_score is None
    assert candidate.rrf_score is None
    assert candidate.rerank_score is None
    assert candidate.primary_dense_score() == 0.72
    assert "bm25=None" in candidate.summary()


def test_build_es_filter_clauses_pushes_all_filters() -> None:
    filters = QueryFilters(
        business_type="policy",
        category="hr",
        doc_id="d1",
        created_after="2026-01-01T00:00:00Z",
        created_before="2026-02-01T00:00:00Z",
        extra={"owner": "legal"},
    )

    assert build_es_filter_clauses(filters) == [
        {"term": {"business_type": "policy"}},
        {"term": {"category": "hr"}},
        {"term": {"doc_id": "d1"}},
        {
            "range": {
                "created_time": {
                    "gte": "2026-01-01T00:00:00Z",
                    "lte": "2026-02-01T00:00:00Z",
                }
            }
        },
        {"term": {"owner": "legal"}},
    ]


def test_build_qdrant_filter_returns_none_when_empty() -> None:
    assert build_qdrant_filter(QueryFilters()) is None


def test_es_retriever_uses_multi_match_highlight_and_filter_pushdown() -> None:
    class FakeES:
        def __init__(self) -> None:
            self.calls = []

        def search(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "hits": {
                    "hits": [
                        {
                            "_id": "c1",
                            "_score": 3.5,
                            "_source": {
                                "chunk_id": "c1",
                                "doc_id": "d1",
                                "text": "matched text",
                            },
                            "highlight": {"text": ["<em>matched</em> text"]},
                        }
                    ]
                }
            }

    cfg = RetrieverConfig(
        id="es_bm25",
        type="lexical",
        engine="elasticsearch",
        top_k=100,
        weight=1.0,
    )
    es = FakeES()
    retriever = ESRetriever(es=es, cfg=cfg, index="chunks")

    candidates = retriever.retrieve(
        "safety policy",
        QueryFilters(business_type="policy"),
        top_k=5,
    )

    body = es.calls[0]["body"]
    multi_match = body["query"]["bool"]["must"][0]["multi_match"]
    assert body["size"] == 5
    assert multi_match["fields"] == ["text", "title^2"]
    assert body["highlight"]["pre_tags"] == ["<em>"]
    assert body["query"]["bool"]["filter"] == [
        {"term": {"business_type": "policy"}}
    ]
    assert candidates[0].bm25_score == 3.5
    assert candidates[0].highlight == ["<em>matched</em> text"]
    assert candidates[0].rank_in_retriever == {"es_bm25": 1}


def test_qdrant_retriever_uses_named_vector_and_filter_pushdown() -> None:
    class FakeEmbedder:
        def embed(self, texts):
            assert texts == ["safety policy"]
            return [[0.1, 0.2, 0.3]]

    class FakePoint:
        id = "point-1"
        score = 0.81
        payload = {
            "chunk_id": "c1",
            "doc_id": "d1",
            "text": "semantic match",
        }

    class FakeQdrant:
        def __init__(self) -> None:
            self.calls = []

        def search(self, **kwargs):
            self.calls.append(kwargs)
            return [FakePoint()]

    cfg = RetrieverConfig(
        id="qd_bge_base",
        type="dense",
        engine="qdrant",
        model_id="bge_base",
        vector_name="bge_base",
        top_k=100,
        weight=1.0,
    )
    qdrant = FakeQdrant()
    retriever = QdrantRetriever(
        qdrant=qdrant,
        embed_client=FakeEmbedder(),
        cfg=cfg,
        collection="chunks",
    )

    candidates = retriever.retrieve(
        "safety policy",
        QueryFilters(category="hr"),
        top_k=7,
    )

    call = qdrant.calls[0]
    assert call["collection_name"] == "chunks"
    assert call["query_vector"] == ("bge_base", [0.1, 0.2, 0.3])
    assert call["query_filter"].must[0].key == "category"
    assert call["limit"] == 7
    assert call["with_vectors"] is False
    assert candidates[0].dense_scores == {"bge_base": 0.81}
    assert candidates[0].rank_in_retriever == {"qd_bge_base": 1}


def test_retriever_pool_preserves_config_order_and_isolates_failures() -> None:
    class FakeRetriever:
        def __init__(self, retriever_id, weight, fail=False):
            self.retriever_id = retriever_id
            self.weight = weight
            self.fail = fail

        def retrieve(self, query_text, filters, *, top_k=None):
            if self.fail:
                raise RuntimeError("boom")
            return [
                RetrievalCandidate(
                    chunk_id=f"{self.retriever_id}-c1",
                    doc_id="d1",
                    source_retriever_ids={self.retriever_id},
                )
            ]

    pool = RetrieverPool(
        retrievers=[
            FakeRetriever("first", 1.0),
            FakeRetriever("second", 0.5, fail=True),
            FakeRetriever("third", 0.8),
        ],
        max_workers=3,
    )
    query = ProcessedQuery(original_query="Raw", normalized_query="raw")

    results = pool.retrieve_all(query, top_k_override=10)

    assert [result.retriever_id for result in results] == ["first", "second", "third"]
    assert [len(result.candidates) for result in results] == [1, 0, 1]
