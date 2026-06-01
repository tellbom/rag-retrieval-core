"""
core/query/filter_builder.py

Translates QueryFilters into the native filter formats of ES 7.x and Qdrant.

These filters are pushed INTO each engine query (not applied post-fusion).
This is mandatory for:
  - Correctness: permission / business_type filters must not leak across types.
  - Precision: wasted recall slots on filtered docs reduce the useful pool.
  - Performance: engine-side filtering is indexed and fast.

ES filter → list of `bool.filter` term/range clauses (injected into the query body).
Qdrant filter → `qdrant_client.http.models.Filter` object.

Both builders are pure functions: same input → same output, no side effects.
"""

from __future__ import annotations

from typing import Any

from qdrant_client.http import models as qmodels

from core.query.processed_query import QueryFilters


# ---------------------------------------------------------------------------
# ES filter builder
# ---------------------------------------------------------------------------

def build_es_filter_clauses(filters: QueryFilters) -> list[dict[str, Any]]:
    """
    Build a list of ES `bool.filter` clauses from QueryFilters.

    Returns an empty list when no filters are active (no-op).
    Caller injects the list into the ES query body:

        {
          "query": {
            "bool": {
              "must": [{"match": {"text": query_text}}],
              "filter": build_es_filter_clauses(filters)   ← here
            }
          }
        }
    """
    clauses: list[dict[str, Any]] = []

    if filters.business_type:
        clauses.append({"term": {"business_type": filters.business_type}})

    if filters.category:
        clauses.append({"term": {"category": filters.category}})

    if filters.doc_id:
        clauses.append({"term": {"doc_id": filters.doc_id}})

    # Date range filter on created_time
    range_clause: dict[str, Any] = {}
    if filters.created_after:
        range_clause["gte"] = filters.created_after
    if filters.created_before:
        range_clause["lte"] = filters.created_before
    if range_clause:
        clauses.append({"range": {"created_time": range_clause}})

    # Extra filters: each key→value becomes a term clause
    for field_name, value in (filters.extra or {}).items():
        clauses.append({"term": {field_name: value}})

    return clauses


# ---------------------------------------------------------------------------
# Qdrant filter builder
# ---------------------------------------------------------------------------

def build_qdrant_filter(filters: QueryFilters) -> qmodels.Filter | None:
    """
    Build a Qdrant Filter from QueryFilters.

    Returns None when no filters are active (Qdrant skips filtering entirely).
    Caller passes the result as the `query_filter` parameter of a Qdrant search.
    """
    must_conditions: list[qmodels.Condition] = []

    if filters.business_type:
        must_conditions.append(
            qmodels.FieldCondition(
                key="business_type",
                match=qmodels.MatchValue(value=filters.business_type),
            )
        )

    if filters.category:
        must_conditions.append(
            qmodels.FieldCondition(
                key="category",
                match=qmodels.MatchValue(value=filters.category),
            )
        )

    if filters.doc_id:
        must_conditions.append(
            qmodels.FieldCondition(
                key="doc_id",
                match=qmodels.MatchValue(value=filters.doc_id),
            )
        )

    # Date range: stored as ISO-8601 string in payload
    if filters.created_after or filters.created_before:
        range_kwargs: dict[str, Any] = {}
        if filters.created_after:
            range_kwargs["gte"] = filters.created_after
        if filters.created_before:
            range_kwargs["lte"] = filters.created_before
        must_conditions.append(
            qmodels.FieldCondition(
                key="created_time",
                range=qmodels.DatetimeRange(**range_kwargs),
            )
        )

    # Extra filters
    for field_name, value in (filters.extra or {}).items():
        must_conditions.append(
            qmodels.FieldCondition(
                key=field_name,
                match=qmodels.MatchValue(value=value),
            )
        )

    if not must_conditions:
        return None

    return qmodels.Filter(must=must_conditions)
