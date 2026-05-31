"""
core/storage/indexer.py

Indexer: idempotent dual-write of embedded chunks to ES 7.x and Qdrant.

Write flow per chunk
--------------------
1. Serialize chunk → ES document body + Qdrant PointStruct.
2. Upsert to ES (index with _id=chunk_id → idempotent).
3. Upsert to Qdrant (upsert with deterministic UUID → idempotent).
4a. Both succeed → done.
4b. Either fails → record chunk_id in failed_index_records with failure_mode.
    Ingestion continues with remaining chunks (no abort).

Idempotency
-----------
ES:    `index` call with explicit `_id` — if the document exists it is
       replaced; if absent it is created. Same result on re-run.
Qdrant: `upsert_points` with the same deterministic UUID — Qdrant replaces
        the existing point. Same result on re-run.

Batching
--------
Chunks are upserted to ES in configurable batches (default 50) and to Qdrant
in batches (default 100).  This reduces HTTP overhead while keeping memory
bounded for large documents.

Public API
----------
    indexer = Indexer(es_client, qdrant_client, fail_store, cfg,
                      es_index="rag_chunks", qdrant_collection="rag_chunks")
    result = indexer.index(chunking_result, enhanced=False)
    # result.indexed_count, result.failed_count, result.failed_chunk_ids
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from core.ingestion.chunk import Chunk, ChunkingResult
from core.storage.chunk_serializer import chunk_to_es_doc, chunk_to_qdrant_point
from core.storage.failed_index_store import FailedIndexStore

logger = logging.getLogger(__name__)

_DEFAULT_ES_BATCH   = 50
_DEFAULT_QD_BATCH   = 100


@dataclass
class IndexResult:
    """Summary of one index() call."""
    doc_id: str
    indexed_count: int = 0
    failed_count: int = 0
    failed_chunk_ids: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.failed_count == 0


class Indexer:
    """
    Dual-writes chunks to Elasticsearch + Qdrant.

    Parameters
    ----------
    es:                 Raw elasticsearch.Elasticsearch client.
    qdrant:             Raw qdrant_client.QdrantClient.
    fail_store:         FailedIndexStore for recording partial failures.
    es_index:           ES index alias name (from provisioner.alias_name).
    qdrant_collection:  Qdrant collection alias name.
    es_batch_size:      Chunks per ES bulk request.
    qdrant_batch_size:  Chunks per Qdrant upsert request.
    """

    def __init__(
        self,
        es: Elasticsearch,
        qdrant: QdrantClient,
        fail_store: FailedIndexStore,
        *,
        es_index: str,
        qdrant_collection: str,
        es_batch_size: int = _DEFAULT_ES_BATCH,
        qdrant_batch_size: int = _DEFAULT_QD_BATCH,
    ) -> None:
        self._es = es
        self._qdrant = qdrant
        self._fail_store = fail_store
        self._es_index = es_index
        self._qdrant_collection = qdrant_collection
        self._es_batch = es_batch_size
        self._qd_batch = qdrant_batch_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(
        self,
        result: ChunkingResult,
        *,
        enhanced: bool = False,
    ) -> IndexResult:
        """
        Upsert all chunks from a ChunkingResult into ES + Qdrant.

        Parameters
        ----------
        result:   ChunkingResult with embedded chunks (named_vectors populated).
        enhanced: Whether the document went through LLM enhancement.

        Returns
        -------
        IndexResult with counts and failed chunk_ids.
        """
        if not result.chunks:
            return IndexResult(doc_id=result.doc_id)

        index_result = IndexResult(doc_id=result.doc_id)
        chunks = result.chunks

        # --- ES bulk upsert ---
        es_failed: set[str] = set()
        for batch in _batches(chunks, self._es_batch):
            failed_ids = self._es_upsert_batch(batch, enhanced=enhanced)
            es_failed.update(failed_ids)

        # --- Qdrant bulk upsert ---
        qd_failed: set[str] = set()
        for batch in _batches(chunks, self._qd_batch):
            failed_ids = self._qdrant_upsert_batch(batch, enhanced=enhanced)
            qd_failed.update(failed_ids)

        # --- Record failures ---
        all_failed = es_failed | qd_failed
        for chunk in chunks:
            cid = chunk.chunk_id
            if cid in all_failed:
                mode = _failure_mode(cid, es_failed, qd_failed)
                self._fail_store.record_failure(
                    chunk_id=cid,
                    doc_id=chunk.doc_id,
                    failure_mode=mode,
                    error_msg=None,   # detailed error already logged
                )
                index_result.failed_count += 1
                index_result.failed_chunk_ids.append(cid)
            else:
                index_result.indexed_count += 1

        if index_result.failed_count:
            logger.warning(
                "Indexed %d/%d chunks for doc_id=%s (%d failed, recorded in failed_index_records)",
                index_result.indexed_count,
                len(chunks),
                result.doc_id,
                index_result.failed_count,
            )
        else:
            logger.info(
                "Indexed %d chunks for doc_id=%s",
                index_result.indexed_count, result.doc_id,
            )

        return index_result

    def delete_by_doc_id(self, doc_id: str) -> tuple[int, int]:
        """
        Delete all chunks belonging to `doc_id` from both stores.
        Returns (es_deleted, qdrant_deleted).
        Used by CRUD delete and update (delete-then-insert).
        """
        es_deleted = self._es_delete_by_doc_id(doc_id)
        qd_deleted = self._qdrant_delete_by_doc_id(doc_id)
        logger.info(
            "Deleted doc_id=%s: ES=%d, Qdrant=%d",
            doc_id, es_deleted, qd_deleted,
        )
        return es_deleted, qd_deleted

    def delete_by_chunk_id(self, chunk_id: str, doc_id: str) -> None:
        """Delete a single chunk from both stores."""
        self._es_delete_chunk(chunk_id)
        self._qdrant_delete_chunk(chunk_id)
        logger.debug("Deleted chunk_id=%s doc_id=%s", chunk_id, doc_id)

    # ------------------------------------------------------------------
    # ES operations
    # ------------------------------------------------------------------

    def _es_upsert_batch(
        self, chunks: list[Chunk], *, enhanced: bool
    ) -> set[str]:
        """
        Bulk upsert a batch of chunks to ES.
        Returns set of chunk_ids that failed.
        """
        failed: set[str] = set()
        # Build bulk body: alternating action + doc lines
        body: list[dict] = []
        chunk_map: dict[int, str] = {}   # bulk item index → chunk_id

        for i, chunk in enumerate(chunks):
            body.append({"index": {"_index": self._es_index, "_id": chunk.chunk_id}})
            body.append(chunk_to_es_doc(chunk, enhanced=enhanced))
            chunk_map[i] = chunk.chunk_id

        try:
            resp = self._es.bulk(body=body, refresh=False)
            if resp.get("errors"):
                for item_idx, item in enumerate(resp.get("items", [])):
                    action = item.get("index", {})
                    if action.get("status", 200) >= 400:
                        cid = chunks[item_idx].chunk_id
                        logger.error(
                            "ES upsert failed for chunk_id=%s: %s",
                            cid, action.get("error"),
                        )
                        failed.add(cid)
        except Exception as exc:
            logger.error("ES bulk upsert exception: %s", exc)
            failed.update(c.chunk_id for c in chunks)

        return failed

    def _es_delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all ES documents with the given doc_id. Returns count."""
        try:
            resp = self._es.delete_by_query(
                index=self._es_index,
                body={"query": {"term": {"doc_id": doc_id}}},
                refresh=True,
            )
            return resp.get("deleted", 0)
        except Exception as exc:
            logger.error("ES delete_by_query failed for doc_id=%s: %s", doc_id, exc)
            return 0

    def _es_delete_chunk(self, chunk_id: str) -> None:
        try:
            self._es.delete(index=self._es_index, id=chunk_id, ignore=[404])
        except Exception as exc:
            logger.error("ES delete failed for chunk_id=%s: %s", chunk_id, exc)

    # ------------------------------------------------------------------
    # Qdrant operations
    # ------------------------------------------------------------------

    def _qdrant_upsert_batch(
        self, chunks: list[Chunk], *, enhanced: bool
    ) -> set[str]:
        """Upsert a batch of chunks to Qdrant. Returns set of failed chunk_ids."""
        failed: set[str] = set()
        points = [chunk_to_qdrant_point(c, enhanced=enhanced) for c in chunks]
        try:
            self._qdrant.upsert(
                collection_name=self._qdrant_collection,
                points=points,
                wait=True,
            )
        except Exception as exc:
            logger.error("Qdrant upsert exception (batch of %d): %s", len(chunks), exc)
            failed.update(c.chunk_id for c in chunks)
        return failed

    def _qdrant_delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all Qdrant points with doc_id in payload. Returns count."""
        try:
            result = self._qdrant.delete(
                collection_name=self._qdrant_collection,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="doc_id",
                                match=qmodels.MatchValue(value=doc_id),
                            )
                        ]
                    )
                ),
                wait=True,
            )
            # Qdrant doesn't return count on delete; return 0 as unknown
            return 0
        except Exception as exc:
            logger.error(
                "Qdrant delete failed for doc_id=%s: %s", doc_id, exc
            )
            return 0

    def _qdrant_delete_chunk(self, chunk_id: str) -> None:
        """Delete a single Qdrant point by chunk_id."""
        from core.storage.chunk_serializer import _chunk_id_to_uuid
        point_uuid = _chunk_id_to_uuid(chunk_id)
        try:
            self._qdrant.delete(
                collection_name=self._qdrant_collection,
                points_selector=qmodels.PointIdsList(points=[point_uuid]),
                wait=True,
            )
        except Exception as exc:
            logger.error("Qdrant delete failed for chunk_id=%s: %s", chunk_id, exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batches(items: list, size: int):
    """Yield successive batches of `size` from `items`."""
    for i in range(0, len(items), size):
        yield items[i: i + size]


def _failure_mode(
    chunk_id: str, es_failed: set[str], qd_failed: set[str]
) -> str:
    in_es = chunk_id in es_failed
    in_qd = chunk_id in qd_failed
    if in_es and in_qd:
        return "both"
    if in_es:
        return "es_write"
    return "qdrant_write"
