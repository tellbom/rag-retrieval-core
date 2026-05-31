"""
core/storage/crud_service.py

CrudService: coordinates add / delete / update at the document level.

Operations
----------
add(doc_id, raw_text, ...)
    Full pipeline: store original → clean → enhance → chunk → embed → index.
    Idempotent: if doc_id already exists, behaves like update.

delete(doc_id)
    Remove all chunks belonging to doc_id from ES + Qdrant + original store.
    Removes ALL child chunks (hierarchy is stored flat in ES/Qdrant; deleting
    by doc_id removes every chunk regardless of hierarchy_level).

update(doc_id, new_raw_text, ...)
    Delete-then-insert.  Chunk boundaries may shift on edit — the only
    correct update strategy.  Original store is updated first.

Design rules
------------
- CrudService orchestrates the full pipeline for a single document.
  It is NOT responsible for batching across documents (the ingestion API
  or pipeline runner handles that).
- Every add/update calls the full ingestion stack in sequence.
- Errors in any stage propagate to the caller; partial states (stored
  original but failed index) are recoverable via failed_index_records.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from core.ingestion.cleaner import Cleaner
from core.ingestion.cleaning_profile import CleaningProfile
from core.ingestion.embedder import Embedder
from core.ingestion.enhancer import Enhancer
from core.ingestion.semantic_chunker import SemanticChunker
from core.ingestion.structural_chunker import StructuralChunker
from core.storage.indexer import IndexResult, Indexer
from core.storage.original_text_store import OriginalTextStore

logger = logging.getLogger(__name__)


@dataclass
class CrudResult:
    doc_id: str
    operation: str           # 'add' | 'delete' | 'update'
    success: bool
    indexed_count: int = 0
    failed_count: int = 0
    message: str = ""


class CrudService:
    """
    Coordinates single-document CRUD over the ingestion pipeline.

    All pipeline components are injected so they can be tested/mocked
    independently.  In production they are built once at service startup
    and shared across requests.

    Parameters
    ----------
    original_store:   OriginalTextStore for persistence of raw texts.
    cleaner:          Cleaner instance (profile pre-loaded).
    enhancer:         Enhancer (may be no-op if enhancement disabled).
    structural_chunker: StructuralChunker.
    semantic_chunker:   SemanticChunker.
    embedder:           Embedder (wraps TEI clients).
    indexer:            Indexer (dual-write ES + Qdrant).
    config_version:     Stamped onto every chunk for provenance.
    """

    def __init__(
        self,
        original_store: OriginalTextStore,
        cleaner: Cleaner,
        enhancer: Enhancer,
        structural_chunker: StructuralChunker,
        semantic_chunker: SemanticChunker,
        embedder: Embedder,
        indexer: Indexer,
        config_version: str = "",
    ) -> None:
        self._original_store = original_store
        self._cleaner = cleaner
        self._enhancer = enhancer
        self._struct_chunker = structural_chunker
        self._sem_chunker = semantic_chunker
        self._embedder = embedder
        self._indexer = indexer
        self._config_version = config_version

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        doc_id: str,
        raw_text: str,
        *,
        business_type: str = "",
        source_metadata: dict | None = None,
    ) -> CrudResult:
        """
        Ingest a new document end-to-end.
        Idempotent: re-adding the same doc_id replaces it (ES/Qdrant upsert).
        """
        logger.info("CrudService.add: doc_id=%s", doc_id)

        # 1. Persist original
        self._original_store.put(
            doc_id, raw_text,
            business_type=business_type,
            source_metadata=source_metadata,
        )

        # 2. Run pipeline
        return self._run_pipeline(
            doc_id, raw_text,
            business_type=business_type,
            source_metadata=source_metadata,
            operation="add",
        )

    def delete(self, doc_id: str) -> CrudResult:
        """
        Remove a document and ALL its chunks from ES + Qdrant + original store.
        Returns success even if the document was not found (idempotent).
        """
        logger.info("CrudService.delete: doc_id=%s", doc_id)

        es_deleted, qd_deleted = self._indexer.delete_by_doc_id(doc_id)
        self._original_store.delete(doc_id)

        return CrudResult(
            doc_id=doc_id,
            operation="delete",
            success=True,
            message=(
                f"Deleted from ES ({es_deleted} docs) "
                f"and Qdrant ({qd_deleted} points)"
            ),
        )

    def update(
        self,
        doc_id: str,
        new_raw_text: str,
        *,
        business_type: str = "",
        source_metadata: dict | None = None,
    ) -> CrudResult:
        """
        Update a document: delete all existing chunks, then re-ingest.
        Chunk boundaries shift on edit — delete-then-insert is the only
        correct strategy.

        If source_metadata is None, the existing metadata from the original
        store is preserved.
        """
        logger.info("CrudService.update: doc_id=%s", doc_id)

        # Preserve metadata if not provided
        if source_metadata is None:
            existing = self._original_store.get(doc_id)
            if existing:
                business_type = business_type or existing.business_type
                source_metadata = existing.source_metadata

        # Persist updated original text (before deleting; if re-index fails,
        # the original store still has the latest version for rebuild/retry)
        self._original_store.put(
            doc_id, new_raw_text,
            business_type=business_type,
            source_metadata=source_metadata,
        )

        # Delete all existing chunks from both stores
        self._indexer.delete_by_doc_id(doc_id)

        # Re-ingest with new text
        return self._run_pipeline(
            doc_id, new_raw_text,
            business_type=business_type,
            source_metadata=source_metadata,
            operation="update",
        )

    # ------------------------------------------------------------------
    # Internal pipeline runner
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        doc_id: str,
        raw_text: str,
        *,
        business_type: str,
        source_metadata: dict | None,
        operation: str,
    ) -> CrudResult:
        try:
            # Clean
            cleaned = self._cleaner.clean(
                doc_id, raw_text,
                business_type=business_type,
                source_metadata=source_metadata or {},
            )

            # Enhance
            enhanced = self._enhancer.enhance(cleaned)

            # Chunk (structural → semantic)
            struct_result = self._struct_chunker.chunk(
                enhanced, config_version=self._config_version
            )
            sem_result = self._sem_chunker.split(struct_result)

            # Embed
            embedded_result = self._embedder.embed(sem_result)

            # Index
            index_result: IndexResult = self._indexer.index(
                embedded_result, enhanced=enhanced.enhanced
            )

            return CrudResult(
                doc_id=doc_id,
                operation=operation,
                success=index_result.success,
                indexed_count=index_result.indexed_count,
                failed_count=index_result.failed_count,
                message=(
                    f"{operation} complete: "
                    f"{index_result.indexed_count} chunks indexed"
                    + (
                        f", {index_result.failed_count} failed"
                        if index_result.failed_count else ""
                    )
                ),
            )

        except Exception as exc:
            logger.error(
                "CrudService.%s failed for doc_id=%s: %s",
                operation, doc_id, exc,
            )
            return CrudResult(
                doc_id=doc_id,
                operation=operation,
                success=False,
                message=f"{operation} failed: {exc}",
            )
