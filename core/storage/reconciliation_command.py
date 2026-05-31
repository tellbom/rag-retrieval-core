"""
core/storage/reconciliation_command.py

ReconciliationCommand: diffs chunk_id sets between ES and Qdrant, then
repairs detected drift.

When drift occurs
-----------------
- A chunk exists in ES but not Qdrant: Qdrant write failed silently.
- A chunk exists in Qdrant but not ES: ES write failed silently.
- Both cases are recorded in failed_index_records for retry.

Repair strategy
---------------
- ES-only chunks:     record as failure_mode='qdrant_write' → retry will
                      try to re-embed and re-upsert to Qdrant.
- Qdrant-only chunks: record as failure_mode='es_write' → retry will
                      re-upsert to ES from Qdrant payload.
- Already in failed_index_records: skip (avoid duplicate records).

Scope
-----
The reconciliation scans all chunks in both stores for the configured
alias.  For large corpora this is an offline batch operation — schedule
it periodically (e.g. nightly or after bulk ingestion).  It uses scroll
(ES) and scroll/list (Qdrant) for memory-efficient enumeration.

Caller
------
Triggered by:
  - POST /indexer/reconcile (ingestion service API, P1-18)
  - CLI: python -m core.storage.reconciliation_command

Public API
----------
    cmd = ReconciliationCommand(es_client, qdrant_client, fail_store,
                                es_index, qdrant_collection)
    report = cmd.run()
    print(report.es_only_count, report.qdrant_only_count, report.in_sync_count)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from core.storage.chunk_serializer import _chunk_id_to_uuid
from core.storage.failed_index_store import FailedIndexStore

logger = logging.getLogger(__name__)

_ES_SCROLL_SIZE   = 1000
_ES_SCROLL_TTL    = "2m"
_QD_SCROLL_LIMIT  = 1000


@dataclass
class ReconciliationReport:
    es_only_count: int = 0          # in ES but not Qdrant
    qdrant_only_count: int = 0      # in Qdrant but not ES
    in_sync_count: int = 0          # present in both
    newly_recorded: int = 0         # new failed_index_records inserted
    already_pending: int = 0        # skipped (already in failed_index_records)
    errors: list[str] = field(default_factory=list)


class ReconciliationCommand:
    """
    Diffs ES and Qdrant chunk_id sets and records any drift.

    Parameters
    ----------
    es:                 Raw Elasticsearch client.
    qdrant:             Raw QdrantClient.
    fail_store:         FailedIndexStore (to record drift as failures).
    es_index:           ES index alias name.
    qdrant_collection:  Qdrant collection alias name.
    """

    def __init__(
        self,
        es: Elasticsearch,
        qdrant: QdrantClient,
        fail_store: FailedIndexStore,
        *,
        es_index: str,
        qdrant_collection: str,
    ) -> None:
        self._es = es
        self._qdrant = qdrant
        self._fail_store = fail_store
        self._es_index = es_index
        self._qdrant_collection = qdrant_collection

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> ReconciliationReport:
        """
        Enumerate all chunk_ids from both stores, compute the diff,
        record any drift in failed_index_records.
        """
        report = ReconciliationReport()

        logger.info(
            "ReconciliationCommand: scanning ES index=%s, Qdrant collection=%s",
            self._es_index, self._qdrant_collection,
        )

        try:
            es_chunks = self._scan_es_chunk_ids()
        except Exception as exc:
            msg = f"ES scan failed: {exc}"
            logger.error(msg)
            report.errors.append(msg)
            return report

        try:
            qd_chunks = self._scan_qdrant_chunk_ids()
        except Exception as exc:
            msg = f"Qdrant scan failed: {exc}"
            logger.error(msg)
            report.errors.append(msg)
            return report

        logger.info(
            "ReconciliationCommand: ES=%d chunks, Qdrant=%d chunks",
            len(es_chunks), len(qd_chunks),
        )

        # --- Compute diff ---
        es_only    = es_chunks.keys() - qd_chunks.keys()
        qdrant_only = qd_chunks.keys() - es_chunks.keys()
        in_sync    = es_chunks.keys() & qd_chunks.keys()

        report.in_sync_count    = len(in_sync)
        report.es_only_count    = len(es_only)
        report.qdrant_only_count = len(qdrant_only)

        # --- Already-pending chunk_ids (avoid duplicates) ---
        already_pending_ids = self._fail_store.get_pending_chunk_ids()

        # --- Record ES-only chunks as qdrant_write failures ---
        for chunk_id in es_only:
            if chunk_id in already_pending_ids:
                report.already_pending += 1
                continue
            doc_id = es_chunks.get(chunk_id, "unknown")
            self._fail_store.record_failure(
                chunk_id=chunk_id,
                doc_id=doc_id,
                failure_mode="qdrant_write",
                error_msg="detected by reconciliation: missing from Qdrant",
            )
            report.newly_recorded += 1

        # --- Record Qdrant-only chunks as es_write failures ---
        for chunk_id in qdrant_only:
            if chunk_id in already_pending_ids:
                report.already_pending += 1
                continue
            doc_id = qd_chunks.get(chunk_id, "unknown")
            self._fail_store.record_failure(
                chunk_id=chunk_id,
                doc_id=doc_id,
                failure_mode="es_write",
                error_msg="detected by reconciliation: missing from ES",
            )
            report.newly_recorded += 1

        logger.info(
            "ReconciliationCommand done: in_sync=%d, es_only=%d, qdrant_only=%d, "
            "newly_recorded=%d, already_pending=%d",
            report.in_sync_count,
            report.es_only_count,
            report.qdrant_only_count,
            report.newly_recorded,
            report.already_pending,
        )
        return report

    # ------------------------------------------------------------------
    # Internal: ES scroll
    # ------------------------------------------------------------------

    def _scan_es_chunk_ids(self) -> dict[str, str]:
        """
        Return {chunk_id: doc_id} for all documents in the ES index.
        Uses scroll API for memory-efficient enumeration.
        """
        chunk_map: dict[str, str] = {}

        resp = self._es.search(
            index=self._es_index,
            body={
                "query": {"match_all": {}},
                "_source": ["chunk_id", "doc_id"],
                "size": _ES_SCROLL_SIZE,
            },
            scroll=_ES_SCROLL_TTL,
        )
        scroll_id = resp.get("_scroll_id")

        try:
            while True:
                hits = resp.get("hits", {}).get("hits", [])
                if not hits:
                    break
                for hit in hits:
                    src = hit.get("_source", {})
                    cid = src.get("chunk_id") or hit.get("_id")
                    did = src.get("doc_id", "unknown")
                    if cid:
                        chunk_map[cid] = did

                resp = self._es.scroll(scroll_id=scroll_id, scroll=_ES_SCROLL_TTL)
                scroll_id = resp.get("_scroll_id")
        finally:
            if scroll_id:
                try:
                    self._es.clear_scroll(scroll_id=scroll_id)
                except Exception:
                    pass

        return chunk_map

    # ------------------------------------------------------------------
    # Internal: Qdrant scroll
    # ------------------------------------------------------------------

    def _scan_qdrant_chunk_ids(self) -> dict[str, str]:
        """
        Return {chunk_id: doc_id} for all points in the Qdrant collection.
        Uses scroll API for memory-efficient enumeration.
        """
        chunk_map: dict[str, str] = {}
        offset = None

        while True:
            resp = self._qdrant.scroll(
                collection_name=self._qdrant_collection,
                scroll_filter=None,
                limit=_QD_SCROLL_LIMIT,
                offset=offset,
                with_payload=["chunk_id", "doc_id"],
                with_vectors=False,
            )
            points, next_offset = resp

            for point in points:
                payload = point.payload or {}
                cid = payload.get("chunk_id")
                did = payload.get("doc_id", "unknown")
                if cid:
                    chunk_map[cid] = did

            if next_offset is None:
                break
            offset = next_offset

        return chunk_map
