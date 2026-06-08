"""
core/storage/chunk_review_store.py

Persists chunk quality audit reports to a dedicated Elasticsearch index.

Index design
------------
- Index name: {base_name}_chunk_reviews  (e.g. "rag_chunks_chunk_reviews")
- No alias switching / versioned rebuild — the review index is audit metadata,
  not retrieval data.  It is rebuilt in-place if needed.
- Mapping uses dynamic=false: declared fields are indexed for querying;
  undeclared fields (e.g. groups array) are stored but not indexed.
  This avoids strict-mode rejections on nested array content while keeping
  doc_id / audited_at queryable.
- One document per (doc_id, audited_at) — multiple audit runs for the same
  doc_id are all preserved.  The latest is returned by get_latest().

Idempotency
-----------
ensure_index() checks existence before creating.  Safe to call at every
service startup.

"Pending" detection
-------------------
A doc_id is considered "pending audit" when there is no review document for
it in the review index.  list_pending_doc_ids() performs a terms-aggregation
over the chunk index to get all known doc_ids, then subtracts those already
reviewed.

Marking reviewed / clearing the reviewed flag
----------------------------------------------
The reviewed flag lives on the review document itself (_audited=True).
To force a re-audit of a doc_id, call clear_reviewed(doc_id) which deletes
all review documents for that doc_id — next scheduler run will re-audit it.

Public API
----------
    store = ChunkReviewStore(es_raw, base_name="rag_chunks")
    store.ensure_index()

    store.save(report_dict)                      # upsert latest report
    report = store.get_latest(doc_id)            # None if not audited
    reports = store.list_for_doc(doc_id)         # all runs, newest first
    pending = store.list_pending_doc_ids(        # doc_ids not yet audited
        chunk_index="rag_chunks", limit=200
    )
    store.clear_reviewed(doc_id)                 # delete all reviews → re-audit
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from elasticsearch import Elasticsearch, NotFoundError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Index naming
# ---------------------------------------------------------------------------

def review_index_name(base_name: str) -> str:
    """Derive the review index name from the chunk index base name."""
    return f"{base_name}_chunk_reviews"


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

_REVIEW_MAPPING: dict[str, Any] = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "30s",
    },
    "mappings": {
        # dynamic=false: unknown fields stored but not indexed.
        # Avoids strict-mode rejections on the `groups` array payload.
        "dynamic": False,
        "properties": {
            # Queryable fields — explicitly declared
            "doc_id":      {"type": "keyword"},
            "audited_at":  {"type": "date", "format": "strict_date_optional_time"},
            "total_groups": {"type": "integer"},
            "skipped_groups": {"type": "integer"},
            # audit_summary sub-fields kept flat for easy aggregation
            "summary_ok":             {"type": "integer"},
            "summary_should_merge":   {"type": "integer"},
            "summary_should_split":   {"type": "integer"},
            "summary_boundary_issue": {"type": "integer"},
            "summary_audit_failed":   {"type": "integer"},
            # System flag: True once a non-empty audit has been saved
            "_audited": {"type": "boolean"},
        },
    },
}


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ChunkReviewStore:
    """
    Manages chunk quality audit reports in a dedicated ES index.

    Parameters
    ----------
    es:
        Raw elasticsearch.Elasticsearch client.
    base_name:
        Chunk index base name (e.g. "rag_chunks").  The review index will be
        named "{base_name}_chunk_reviews".
    """

    def __init__(self, es: Elasticsearch, base_name: str = "rag_chunks") -> None:
        self._es = es
        self._index = review_index_name(base_name)
        self._chunk_index_base = base_name

    # ------------------------------------------------------------------
    # Provisioning
    # ------------------------------------------------------------------

    def ensure_index(self) -> bool:
        """
        Create the review index if it does not exist.

        Returns True if the index was freshly created, False if it already
        existed.  Safe to call on every service startup (idempotent).
        """
        if self._es.indices.exists(index=self._index):
            logger.debug("Review index already exists: %s", self._index)
            return False
        self._es.indices.create(index=self._index, body=_REVIEW_MAPPING)
        logger.info("Created review index: %s", self._index)
        return True

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, report: dict[str, Any]) -> str:
        """
        Persist an audit report.  Returns the ES document _id.

        The document _id is derived from doc_id + audited_at so that
        re-saving the same report is idempotent.

        Parameters
        ----------
        report:
            The dict produced by ChunkQualityAuditor.audit().  Must contain
            "doc_id" and "audited_at" keys.
        """
        doc_id: str = report["doc_id"]
        audited_at: str = report["audited_at"]

        summary: dict[str, int] = report.get("audit_summary", {})

        # Flatten audit_summary for queryable sub-fields
        flat: dict[str, Any] = {
            "doc_id":      doc_id,
            "audited_at":  audited_at,
            "total_groups":  report.get("total_groups", 0),
            "skipped_groups": report.get("skipped_groups", 0),
            "summary_ok":             summary.get("ok", 0),
            "summary_should_merge":   summary.get("should_merge", 0),
            "summary_should_split":   summary.get("should_split", 0),
            "summary_boundary_issue": summary.get("boundary_issue", 0),
            "summary_audit_failed":   summary.get("audit_failed", 0),
            "_audited": True,
            # Full report stored as-is (dynamic=false means it is stored
            # but not field-indexed; retrievable via _source)
            "groups": report.get("groups", []),
        }

        es_id = _make_review_id(doc_id, audited_at)
        self._es.index(index=self._index, id=es_id, body=flat)
        logger.debug("Saved audit report for doc_id=%s id=%s", doc_id, es_id)
        return es_id

    def clear_reviewed(self, doc_id: str) -> int:
        """
        Delete all review documents for doc_id.

        After this call, doc_id will appear in list_pending_doc_ids() and
        the scheduler will re-audit it on the next run.

        Returns the number of documents deleted.
        """
        resp = self._es.delete_by_query(
            index=self._index,
            body={"query": {"term": {"doc_id": doc_id}}},
            refresh=True,
        )
        deleted: int = resp.get("deleted", 0)
        logger.info(
            "Cleared %d review doc(s) for doc_id=%s", deleted, doc_id
        )
        return deleted

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_latest(self, doc_id: str) -> dict[str, Any] | None:
        """
        Return the most recent audit report for doc_id, or None if not audited.

        The returned dict is the full report (including the groups list).
        """
        resp = self._es.search(
            index=self._index,
            body={
                "query": {"term": {"doc_id": doc_id}},
                "sort": [{"audited_at": {"order": "desc"}}],
                "size": 1,
            },
        )
        hits = resp.get("hits", {}).get("hits", [])
        if not hits:
            return None
        return hits[0]["_source"]

    def list_for_doc(self, doc_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """
        Return all audit reports for doc_id, newest first.

        Parameters
        ----------
        limit:
            Maximum number of reports to return (default 20).
        """
        resp = self._es.search(
            index=self._index,
            body={
                "query": {"term": {"doc_id": doc_id}},
                "sort": [{"audited_at": {"order": "desc"}}],
                "size": limit,
            },
        )
        return [h["_source"] for h in resp.get("hits", {}).get("hits", [])]

    def list_pending_doc_ids(
        self,
        *,
        chunk_index: str,
        limit: int = 200,
    ) -> list[str]:
        """
        Return doc_ids present in the chunk index that have no review document.

        Algorithm
        ---------
        1. Aggregate all doc_ids from the chunk index (terms agg, up to limit*2
           to give headroom after subtraction).
        2. Aggregate all reviewed doc_ids from the review index.
        3. Return the difference, capped at limit.

        Parameters
        ----------
        chunk_index:
            ES alias/index for the main chunk data (e.g. "rag_chunks").
        limit:
            Maximum number of pending doc_ids to return.
        """
        # Step 1: all doc_ids in chunk index
        chunk_resp = self._es.search(
            index=chunk_index,
            body={
                "size": 0,
                "aggs": {
                    "all_docs": {
                        "terms": {
                            "field": "doc_id",
                            "size": limit * 4,
                        }
                    }
                },
            },
        )
        all_doc_ids: set[str] = {
            b["key"]
            for b in chunk_resp.get("aggregations", {})
            .get("all_docs", {})
            .get("buckets", [])
        }

        if not all_doc_ids:
            return []

        # Step 2: already-reviewed doc_ids
        reviewed_resp = self._es.search(
            index=self._index,
            body={
                "size": 0,
                "aggs": {
                    "reviewed_docs": {
                        "terms": {
                            "field": "doc_id",
                            "size": len(all_doc_ids) + 1,
                        }
                    }
                },
            },
        )
        reviewed_doc_ids: set[str] = {
            b["key"]
            for b in reviewed_resp.get("aggregations", {})
            .get("reviewed_docs", {})
            .get("buckets", [])
        }

        pending = sorted(all_doc_ids - reviewed_doc_ids)
        return pending[:limit]

    def is_reviewed(self, doc_id: str) -> bool:
        """Return True if at least one audit report exists for doc_id."""
        resp = self._es.search(
            index=self._index,
            body={
                "query": {"term": {"doc_id": doc_id}},
                "size": 0,
                "track_total_hits": True,
            },
        )
        total = resp.get("hits", {}).get("total", {})
        # ES 7.x returns {"value": N, "relation": "eq"} when track_total_hits=True
        count = total.get("value", 0) if isinstance(total, dict) else int(total)
        return count > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_review_id(doc_id: str, audited_at: str) -> str:
    """
    Deterministic ES document ID for a review: "{doc_id}::{audited_at}".
    Using :: as separator because doc_id is typically a plain string without
    colons; audited_at is ISO-8601.  Idempotent re-save of the same report
    will overwrite the same ES document.
    """
    return f"{doc_id}::{audited_at}"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
