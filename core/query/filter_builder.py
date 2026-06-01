"""Pure builders for pushing query filters into ES and Qdrant."""

from __future__ import annotations

from typing import Any

from qdrant_client.http import models as qmodels

from core.query.processed_query import QueryFilters


def build_es_filter_clauses(filters: QueryFilters) -> list[dict[str, Any]]:
    """Build Elasticsearch ``bool.filter`` clauses from QueryFilters."""
    clauses: list[dict[str, Any]] = []

    if filters.business_type:
        clauses.append({"term": {"business_type": filters.business_type}})
    if filters.category:
        clauses.append({"term": {"category": filters.category}})
    if filters.doc_id:
        clauses.append({"term": {"doc_id": filters.doc_id}})

    range_clause: dict[str, Any] = {}
    if filters.created_after:
        range_clause["gte"] = filters.created_after
    if filters.created_before:
        range_clause["lte"] = filters.created_before
    if range_clause:
        clauses.append({"range": {"created_time": range_clause}})

    for field_name, value in filters.extra.items():
        clauses.append({"term": {field_name: value}})

    return clauses


def build_qdrant_filter(filters: QueryFilters) -> qmodels.Filter | None:
    """Build a Qdrant query filter from QueryFilters."""
    must_conditions: list[qmodels.Condition] = []

    if filters.business_type:
        must_conditions.append(_match("business_type", filters.business_type))
    if filters.category:
        must_conditions.append(_match("category", filters.category))
    if filters.doc_id:
        must_conditions.append(_match("doc_id", filters.doc_id))

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

    for field_name, value in filters.extra.items():
        must_conditions.append(_match(field_name, value))

    if not must_conditions:
        return None
    return qmodels.Filter(must=must_conditions)


def _match(field_name: str, value: Any) -> qmodels.FieldCondition:
    return qmodels.FieldCondition(
        key=field_name,
        match=qmodels.MatchValue(value=value),
    )
