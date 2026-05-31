"""
core/storage/rebuild_service.py

RebuildService: full corpus rebuild into a new versioned index/collection,
followed by atomic alias switch and cleanup of the old index.

Rebuild algorithm
-----------------
1. Create NEW versioned ES index + Qdrant collection (from new AppConfig).
   Name pattern: {base}_{config_version}_{primary_model_version}
2. Build a temporary Indexer that writes to the NEW names (not the alias).
3. Iterate all documents from OriginalTextStore, run each through the full
   pipeline (clean → enhance → chunk → embed), write to NEW stores.
4. On completion:
   a. Atomically switch ES alias → new index.
   b. Switch Qdrant alias → new collection.
   c. Drop the old index/collection.
5. Update the Indexer used by the live service to point to the new names.

Zero-downtime guarantee
-----------------------
During rebuild, the live alias continues to serve the OLD index/collection.
Queries are unaffected.  Only after BOTH alias switches succeed does the old
store get dropped.  If the rebuild fails mid-way, the alias is never touched
and the old store remains intact.

Rebuild is triggered:
  - When config version changes (chunking, cleaning, models).
  - When an embedding model version changes.
  - Manually via POST /crud/rebuild.

Progress reporting
------------------
RebuildService yields RebuildProgress objects so the caller can stream
progress to the operator (logged + returned in API response).

Public API
----------
    svc = RebuildService(es_provisioner, qdrant_provisioner,
                         original_store, crud_service_factory,
                         old_es_index, old_qdrant_collection)
    for progress in svc.rebuild():
        print(progress)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generator

from core.config.models import AppConfig
from core.ingestion.cleaner import Cleaner
from core.ingestion.embedder import Embedder
from core.ingestion.enhancer import Enhancer
from core.ingestion.semantic_chunker import SemanticChunker
from core.ingestion.structural_chunker import StructuralChunker
from core.storage.es.client import ESClient
from core.storage.es.provisioner import ESProvisioner
from core.storage.failed_index_store import FailedIndexStore
from core.storage.indexer import Indexer
from core.storage.original_text_store import OriginalTextStore
from core.storage.qdrant.client import QdrantClientWrapper
from core.storage.qdrant.provisioner import QdrantProvisioner

logger = logging.getLogger(__name__)


@dataclass
class RebuildProgress:
    phase: str              # 'provision' | 'index' | 'switch' | 'cleanup' | 'done' | 'error'
    docs_processed: int = 0
    docs_total: int = 0
    chunks_indexed: int = 0
    chunks_failed: int = 0
    message: str = ""
    error: str = ""

    @property
    def complete(self) -> bool:
        return self.phase in ("done", "error")

    def __str__(self) -> str:
        if self.phase == "index":
            pct = (
                f"{self.docs_processed}/{self.docs_total} docs"
                if self.docs_total else f"{self.docs_processed} docs"
            )
            return (
                f"[{self.phase}] {pct} — "
                f"{self.chunks_indexed} chunks indexed, "
                f"{self.chunks_failed} failed"
            )
        return f"[{self.phase}] {self.message or self.error}"


@dataclass
class RebuildResult:
    success: bool
    new_es_index: str = ""
    new_qdrant_collection: str = ""
    docs_processed: int = 0
    chunks_indexed: int = 0
    chunks_failed: int = 0
    error: str = ""
    progress_log: list[str] = field(default_factory=list)


class RebuildService:
    """
    Orchestrates zero-downtime full corpus rebuild.

    Parameters
    ----------
    cfg:                   New AppConfig (with updated versions/chunking).
    es_client:             ESClient for provisioning new index.
    qdrant_client:         QdrantClientWrapper for provisioning new collection.
    fail_store:            FailedIndexStore for tracking rebuild failures.
    original_store:        Source of all documents to re-ingest.
    cleaner:               Cleaner instance.
    enhancer:              Enhancer instance.
    structural_chunker:    StructuralChunker.
    semantic_chunker:      SemanticChunker.
    embedder:              Embedder (already has correct model clients).
    current_es_index:      Current live ES index name (to drop post-switch).
    current_qdrant_collection: Current live Qdrant collection (to drop post-switch).
    base_name:             Index/collection base name.
    """

    def __init__(
        self,
        cfg: AppConfig,
        es_client: ESClient,
        qdrant_client: QdrantClientWrapper,
        fail_store: FailedIndexStore,
        original_store: OriginalTextStore,
        cleaner: Cleaner,
        enhancer: Enhancer,
        structural_chunker: StructuralChunker,
        semantic_chunker: SemanticChunker,
        embedder: Embedder,
        *,
        current_es_index: str,
        current_qdrant_collection: str,
        base_name: str = "rag_chunks",
    ) -> None:
        self._cfg = cfg
        self._es_client = es_client
        self._qdrant_client = qdrant_client
        self._fail_store = fail_store
        self._original_store = original_store
        self._cleaner = cleaner
        self._enhancer = enhancer
        self._struct_chunker = structural_chunker
        self._sem_chunker = semantic_chunker
        self._embedder = embedder
        self._current_es_index = current_es_index
        self._current_qdrant_collection = current_qdrant_collection
        self._base_name = base_name

        # Build provisioners for the NEW names
        self._es_provisioner = ESProvisioner(
            es_client, cfg, base_name=base_name
        )
        self._qdrant_provisioner = QdrantProvisioner(
            qdrant_client, cfg, base_name=base_name
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rebuild(self) -> Generator[RebuildProgress, None, RebuildResult]:
        """
        Execute rebuild as a generator yielding progress at each phase.
        The final yielded value is also returned (StopIteration.value).

        Usage:
            result = None
            for progress in svc.rebuild():
                log(progress)
                if progress.phase == 'done':
                    result = progress
        """
        result = RebuildResult(success=False)
        progress_log: list[str] = []

        def emit(p: RebuildProgress) -> RebuildProgress:
            msg = str(p)
            logger.info("Rebuild: %s", msg)
            progress_log.append(msg)
            return p

        # ── Phase 1: provision new stores ──────────────────────────────
        new_es_index = self._es_provisioner.index_name
        new_qdrant_col = self._qdrant_provisioner.collection_name

        yield emit(RebuildProgress(
            phase="provision",
            message=f"Creating new ES index '{new_es_index}' "
                    f"and Qdrant collection '{new_qdrant_col}'",
        ))

        try:
            self._es_provisioner.provision()
            self._qdrant_provisioner.provision()
        except Exception as exc:
            p = emit(RebuildProgress(
                phase="error",
                error=f"Provision failed: {exc}",
            ))
            yield p
            result.error = p.error
            result.progress_log = progress_log
            return result

        # ── Phase 2: re-index all documents ────────────────────────────
        # Temporary indexer pointing at NEW index/collection (not alias)
        rebuild_indexer = Indexer(
            es=self._es_client.raw,
            qdrant=self._qdrant_client.raw,
            fail_store=self._fail_store,
            es_index=new_es_index,
            qdrant_collection=new_qdrant_col,
        )

        doc_ids = self._original_store.list_all()
        total = len(doc_ids)
        processed = 0
        total_indexed = 0
        total_failed = 0

        yield emit(RebuildProgress(
            phase="index",
            docs_total=total,
            message=f"Re-indexing {total} document(s)",
        ))

        for doc_id in doc_ids:
            entry = self._original_store.get(doc_id)
            if entry is None:
                logger.warning("Rebuild: skipping unreadable doc_id=%s", doc_id)
                processed += 1
                total_failed += 1
                continue

            try:
                cleaned = self._cleaner.clean(
                    entry.doc_id, entry.raw_text,
                    business_type=entry.business_type,
                    source_metadata=entry.source_metadata,
                )
                enhanced = self._enhancer.enhance(cleaned)
                struct_result = self._struct_chunker.chunk(
                    enhanced, config_version=self._cfg.version
                )
                sem_result = self._sem_chunker.split(struct_result)
                embedded_result = self._embedder.embed(sem_result)
                index_result = rebuild_indexer.index(
                    embedded_result, enhanced=enhanced.enhanced
                )
                total_indexed += index_result.indexed_count
                total_failed += index_result.failed_count
            except Exception as exc:
                logger.error(
                    "Rebuild: pipeline error for doc_id=%s: %s", doc_id, exc
                )
                total_failed += 1

            processed += 1
            # Emit progress every 10 docs
            if processed % 10 == 0 or processed == total:
                yield emit(RebuildProgress(
                    phase="index",
                    docs_processed=processed,
                    docs_total=total,
                    chunks_indexed=total_indexed,
                    chunks_failed=total_failed,
                ))

        if total_failed:
            p = emit(RebuildProgress(
                phase="error",
                docs_processed=processed,
                docs_total=total,
                chunks_indexed=total_indexed,
                chunks_failed=total_failed,
                error=(
                    f"Rebuild produced {total_failed} failed document/chunk item(s). "
                    "Aliases were not switched; old stores remain live."
                ),
            ))
            yield p
            result.error = p.error
            result.progress_log = progress_log
            return result

        # ── Phase 3: atomic alias switch ───────────────────────────────
        yield emit(RebuildProgress(
            phase="switch",
            docs_processed=processed,
            message=(
                f"Switching aliases: "
                f"ES {self._current_es_index} → {new_es_index}, "
                f"Qdrant {self._current_qdrant_collection} → {new_qdrant_col}"
            ),
        ))

        switched_es = False
        switched_qdrant = False
        try:
            # ES alias switch: atomic remove + add
            if self._current_es_index != new_es_index:
                self._es_provisioner.alias_switch(
                    self._current_es_index, new_es_index
                )
                switched_es = True
            else:
                logger.info("Rebuild: ES index unchanged, no switch needed")

            # Qdrant alias switch
            if self._current_qdrant_collection != new_qdrant_col:
                self._qdrant_provisioner.alias_switch(
                    self._current_qdrant_collection, new_qdrant_col
                )
                switched_qdrant = True
            else:
                logger.info("Rebuild: Qdrant collection unchanged, no switch needed")

        except Exception as exc:
            rollback_errors: list[str] = []
            if switched_es:
                try:
                    self._es_provisioner.alias_switch(
                        new_es_index, self._current_es_index
                    )
                except Exception as rollback_exc:
                    rollback_errors.append(f"ES rollback failed: {rollback_exc}")
            if switched_qdrant:
                try:
                    self._qdrant_provisioner.alias_switch(
                        new_qdrant_col, self._current_qdrant_collection
                    )
                except Exception as rollback_exc:
                    rollback_errors.append(f"Qdrant rollback failed: {rollback_exc}")

            rollback_msg = (
                " Rollback attempted."
                if not rollback_errors
                else " Rollback errors: " + "; ".join(rollback_errors)
            )
            p = emit(RebuildProgress(
                phase="error",
                error=f"Alias switch failed: {exc}. "
                      f"Old stores are still live. New stores: "
                      f"ES={new_es_index}, Qdrant={new_qdrant_col}."
                      f"{rollback_msg}",
            ))
            yield p
            result.error = p.error
            result.progress_log = progress_log
            return result

        # ── Phase 4: cleanup old stores ────────────────────────────────
        yield emit(RebuildProgress(
            phase="cleanup",
            message=(
                f"Dropping old ES index '{self._current_es_index}' "
                f"and Qdrant collection '{self._current_qdrant_collection}'"
            ),
        ))

        if self._current_es_index != new_es_index:
            try:
                self._es_provisioner.delete_index(self._current_es_index)
            except Exception as exc:
                logger.warning(
                    "Rebuild: could not delete old ES index '%s': %s",
                    self._current_es_index, exc,
                )

        if self._current_qdrant_collection != new_qdrant_col:
            try:
                self._qdrant_provisioner.delete_collection(
                    self._current_qdrant_collection
                )
            except Exception as exc:
                logger.warning(
                    "Rebuild: could not delete old Qdrant collection '%s': %s",
                    self._current_qdrant_collection, exc,
                )

        # ── Done ───────────────────────────────────────────────────────
        result = RebuildResult(
            success=True,
            new_es_index=new_es_index,
            new_qdrant_collection=new_qdrant_col,
            docs_processed=processed,
            chunks_indexed=total_indexed,
            chunks_failed=total_failed,
            progress_log=progress_log,
        )

        yield emit(RebuildProgress(
            phase="done",
            docs_processed=processed,
            docs_total=total,
            chunks_indexed=total_indexed,
            chunks_failed=total_failed,
            message=(
                f"Rebuild complete: {processed} docs, "
                f"{total_indexed} chunks indexed, "
                f"{total_failed} failed"
            ),
        ))

        return result
