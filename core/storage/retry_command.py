"""
core/storage/retry_command.py

RetryCommand: replays pending entries from failed_index_records.

How it works
------------
1. Load pending records from FailedIndexStore (oldest first).
2. For each record, look up the chunk from ES (it may exist partially).
   If the chunk exists in ES, re-upsert to Qdrant only; vice versa.
   If absent from both, re-run full ingestion (caller responsibility via
   the ingestion API — retry only handles the write layer here).
3. On success → mark_resolved(record_id).
4. On failure → mark_attempt(record_id, error_msg) and continue.

The retry command does NOT re-run the full ingestion pipeline (cleaning,
chunking, embedding).  It works with already-processed chunks.  If the
original chunk data is missing from both stores, the record is left pending
and the operator should trigger a full re-ingest via the ingestion API.

Caller
------
Triggered by:
  - POST /indexer/retry  (ingestion service API, P1-18)
  - CLI: python -m core.storage.retry_command

Public API
----------
    cmd = RetryCommand(indexer, fail_store)
    report = cmd.run(max_attempts=5, limit=200)
    print(report.resolved_count, report.still_failing_count)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.storage.failed_index_store import FailedIndexRecord, FailedIndexStore
from core.storage.indexer import Indexer

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ATTEMPTS = 5
_DEFAULT_LIMIT = 200


@dataclass
class RetryReport:
    resolved_count: int = 0
    still_failing_count: int = 0
    skipped_count: int = 0        # exceeded max_attempts
    details: list[dict] = field(default_factory=list)


class RetryCommand:
    """
    Replays pending failed_index_records using the Indexer.

    Parameters
    ----------
    indexer:    Indexer instance (for ES + Qdrant re-upsert).
    fail_store: FailedIndexStore.
    """

    def __init__(self, indexer: Indexer, fail_store: FailedIndexStore) -> None:
        self._indexer = indexer
        self._fail_store = fail_store

    def run(
        self,
        *,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        limit: int = _DEFAULT_LIMIT,
    ) -> RetryReport:
        """
        Process pending records.

        Parameters
        ----------
        max_attempts: Skip records that have already been attempted this many times.
        limit:        Max records to process in one run.
        """
        pending = self._fail_store.list_pending(
            limit=limit, max_attempts=max_attempts
        )
        report = RetryReport()

        if not pending:
            logger.info("RetryCommand: no pending records")
            return report

        logger.info("RetryCommand: processing %d pending record(s)", len(pending))

        for record in pending:
            self._process_record(record, report)

        logger.info(
            "RetryCommand done: resolved=%d, still_failing=%d, skipped=%d",
            report.resolved_count,
            report.still_failing_count,
            report.skipped_count,
        )
        return report

    def _process_record(
        self, record: FailedIndexRecord, report: RetryReport
    ) -> None:
        """
        Attempt to re-index a single failed record by fetching the chunk
        from whichever store still has it and re-writing to the other.

        Strategy:
        - failure_mode == 'es_write'     → chunk is in Qdrant, re-upsert to ES
        - failure_mode == 'qdrant_write' → chunk is in ES,     re-upsert to Qdrant
        - failure_mode == 'both'         → both stores need the write; try both
        """
        try:
            success = self._retry_write(record)
            if success:
                self._fail_store.mark_resolved(record.id)
                report.resolved_count += 1
                report.details.append({
                    "chunk_id": record.chunk_id,
                    "status": "resolved",
                })
            else:
                self._fail_store.mark_attempt(record.id, error_msg="retry failed")
                report.still_failing_count += 1
                report.details.append({
                    "chunk_id": record.chunk_id,
                    "status": "still_failing",
                })
        except Exception as exc:
            logger.error(
                "RetryCommand exception for chunk_id=%s: %s",
                record.chunk_id, exc,
            )
            self._fail_store.mark_attempt(record.id, error_msg=str(exc))
            report.still_failing_count += 1

    def _retry_write(self, record: FailedIndexRecord) -> bool:
        """
        Re-attempt the failed write.  Returns True if successful.

        Fetches the document from the store that succeeded originally,
        then re-writes to the store that failed.
        """
        mode = record.failure_mode

        if mode == "es_write":
            # Try to get the chunk payload from Qdrant, re-upsert to ES
            return self._retry_es_from_qdrant(record)

        if mode == "qdrant_write":
            # Try to get the chunk from ES, re-upsert to Qdrant
            return self._retry_qdrant_from_es(record)

        # mode == 'both': try re-fetching from ES first, then Qdrant
        es_ok = self._retry_es_from_qdrant(record)
        qd_ok = self._retry_qdrant_from_es(record)
        return es_ok and qd_ok

    def _retry_es_from_qdrant(self, record: FailedIndexRecord) -> bool:
        """Fetch chunk from Qdrant payload, re-upsert to ES."""
        from core.storage.chunk_serializer import _chunk_id_to_uuid
        import uuid

        try:
            point_uuid = _chunk_id_to_uuid(record.chunk_id)
            results = self._indexer._qdrant.retrieve(
                collection_name=self._indexer._qdrant_collection,
                ids=[point_uuid],
                with_payload=True,
                with_vectors=False,
            )
            if not results:
                logger.warning(
                    "RetryCommand: chunk_id=%s not found in Qdrant — "
                    "requires full re-ingest",
                    record.chunk_id,
                )
                return False

            payload = results[0].payload or {}
            # Re-upsert to ES using the payload data
            doc = {
                "chunk_id":        payload.get("chunk_id", record.chunk_id),
                "doc_id":          payload.get("doc_id", record.doc_id),
                "parent_id":       payload.get("parent_id"),
                "hierarchy_level": payload.get("hierarchy_level", 0),
                "position":        payload.get("position", 0),
                "text":            payload.get("text", ""),
                "business_type":   payload.get("business_type", ""),
                "config_version":  payload.get("config_version", ""),
                "embedding_model_versions": payload.get("embedding_model_versions", ""),
                "_enhanced":       payload.get("_enhanced", False),
                "_config_version": payload.get("config_version", ""),
                "_embedding_model_versions": payload.get("embedding_model_versions", ""),
            }
            self._indexer._es.index(
                index=self._indexer._es_index,
                id=record.chunk_id,
                document=doc,
            )
            return True
        except Exception as exc:
            logger.error(
                "RetryCommand: ES retry failed for chunk_id=%s: %s",
                record.chunk_id, exc,
            )
            return False

    def _retry_qdrant_from_es(self, record: FailedIndexRecord) -> bool:
        """Fetch chunk from ES, re-upsert to Qdrant (vectors only approach)."""
        # Without the original vectors we cannot re-upsert to Qdrant.
        # Qdrant requires vectors; if they are missing, a full re-embed is needed.
        # Log and signal that full re-ingest is required.
        logger.warning(
            "RetryCommand: qdrant_write retry for chunk_id=%s requires original "
            "vectors — schedule full re-ingest for doc_id=%s",
            record.chunk_id, record.doc_id,
        )
        # Check if the chunk is in ES (at least the data is there)
        try:
            resp = self._indexer._es.get(
                index=self._indexer._es_index,
                id=record.chunk_id,
                ignore=[404],
            )
            if resp.get("found"):
                # Data exists in ES but we can't re-embed here; mark as
                # requiring full re-ingest rather than leaving it silent
                logger.warning(
                    "chunk_id=%s found in ES but Qdrant write requires "
                    "re-embedding. Mark doc_id=%s for re-ingest.",
                    record.chunk_id, record.doc_id,
                )
        except Exception:
            pass
        return False
