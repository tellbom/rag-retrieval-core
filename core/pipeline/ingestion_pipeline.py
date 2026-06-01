"""Ordered ingestion pipeline runner."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from core.pipeline.protocols import (
    CleanerProtocol,
    EmbedderProtocol,
    EnhancerProtocol,
    IndexerProtocol,
    SemanticChunkerProtocol,
    StructuralChunkerProtocol,
)
from core.storage.indexer import IndexResult

logger = logging.getLogger(__name__)


@dataclass
class IngestionPipelineResult:
    doc_id: str
    success: bool
    indexed_count: int = 0
    failed_count: int = 0
    enhanced: bool = False
    message: str = ""


class IngestionPipeline:
    """Run clean -> enhance -> chunk -> semantic split -> embed -> index."""

    def __init__(
        self,
        cleaner: CleanerProtocol,
        enhancer: EnhancerProtocol,
        structural_chunker: StructuralChunkerProtocol,
        semantic_chunker: SemanticChunkerProtocol,
        embedder: EmbedderProtocol,
        indexer: IndexerProtocol,
        config_version: str = "",
    ) -> None:
        self._cleaner = cleaner
        self._enhancer = enhancer
        self._struct_chunker = structural_chunker
        self._sem_chunker = semantic_chunker
        self._embedder = embedder
        self._indexer = indexer
        self._config_version = config_version

    def run(
        self,
        doc_id: str,
        raw_text: str,
        *,
        business_type: str = "",
        source_metadata: dict | None = None,
    ) -> IngestionPipelineResult:
        logger.debug("IngestionPipeline.run: doc_id=%s", doc_id)

        cleaned = self._cleaner.clean(
            doc_id,
            raw_text,
            business_type=business_type,
            source_metadata=source_metadata,
        )
        enhanced_doc = self._enhancer.enhance(cleaned)
        structural_result = self._struct_chunker.chunk(
            enhanced_doc,
            self._config_version,
        )
        semantic_result = self._sem_chunker.split(structural_result)
        embedded_result = self._embedder.embed(semantic_result)
        index_result: IndexResult = self._indexer.index(
            embedded_result,
            enhanced=enhanced_doc.enhanced,
        )

        message = f"{index_result.indexed_count} chunks indexed"
        if index_result.failed_count:
            message += f", {index_result.failed_count} failed"

        return IngestionPipelineResult(
            doc_id=doc_id,
            success=index_result.success,
            indexed_count=index_result.indexed_count,
            failed_count=index_result.failed_count,
            enhanced=enhanced_doc.enhanced,
            message=message,
        )

    def run_batch(
        self,
        documents: list[tuple[str, str]],
        *,
        business_type: str = "",
    ) -> list[IngestionPipelineResult]:
        results: list[IngestionPipelineResult] = []
        for doc_id, raw_text in documents:
            try:
                results.append(
                    self.run(doc_id, raw_text, business_type=business_type)
                )
            except Exception as exc:
                logger.error(
                    "IngestionPipeline.run_batch: doc_id=%s failed: %s",
                    doc_id,
                    exc,
                )
                results.append(
                    IngestionPipelineResult(
                        doc_id=doc_id,
                        success=False,
                        message=str(exc),
                    )
                )
        return results

    def component_names(self) -> dict[str, str]:
        return {
            "cleaner": type(self._cleaner).__name__,
            "enhancer": type(self._enhancer).__name__,
            "structural_chunker": type(self._struct_chunker).__name__,
            "semantic_chunker": type(self._sem_chunker).__name__,
            "embedder": type(self._embedder).__name__,
            "indexer": type(self._indexer).__name__,
        }
