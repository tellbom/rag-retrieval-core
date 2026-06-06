"""Build LLM-ready context blocks from reranked retrieval candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from core.query.retrieval_candidate import RetrievalCandidate

logger = logging.getLogger(__name__)

_BLOCK_DELIMITER = "\n\n---\n\n"


@dataclass
class Citation:
    """Traceability metadata for one context block."""

    index: int
    chunk_id: str
    doc_id: str
    title: str = ""
    source: str = ""
    bm25_score: float | None = None
    dense_scores: dict[str, float] = field(default_factory=dict)
    rrf_score: float | None = None
    rerank_score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuiltContext:
    """Output of ContextBuilder.build()."""

    context_text: str
    citations: list[Citation]
    is_empty: bool
    candidate_count: int = 0
    reranked: bool = False

    def summary(self) -> str:
        return (
            f"blocks={len(self.citations)} "
            f"chars={len(self.context_text)} "
            f"reranked={self.reranked} "
            f"empty={self.is_empty}"
        )


class ContextBuilder:
    """Assemble final prompt context from reranked candidates."""

    def __init__(
        self,
        context_top_k: int = 8,
        *,
        include_scores_in_context: bool = False,
    ) -> None:
        self._context_top_k = context_top_k
        self._include_scores = include_scores_in_context

    def build(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        *,
        reranked: bool = False,
        qdrant=None,
        qdrant_collection: str = "",
    ) -> BuiltContext:
        """Build context text and citations from an ordered candidate list."""
        if not candidates:
            logger.debug("ContextBuilder: no candidates, returning empty context")
            return BuiltContext(
                context_text="",
                citations=[],
                is_empty=True,
                candidate_count=0,
                reranked=reranked,
            )

        top = candidates[: self._context_top_k]
        top = self._backfill_parents(top, qdrant=qdrant, collection=qdrant_collection)
        top = self._dedup(top)

        if not top:
            return BuiltContext(
                context_text="",
                citations=[],
                is_empty=True,
                candidate_count=0,
                reranked=reranked,
            )

        blocks: list[str] = []
        citations: list[Citation] = []

        for index, candidate in enumerate(top, start=1):
            blocks.append(self._format_block(index, candidate))
            citations.append(self._make_citation(index, candidate))

        context_text = _BLOCK_DELIMITER.join(blocks)

        logger.debug(
            "ContextBuilder: built %d block(s) (%d chars) for query=%r",
            len(blocks),
            len(context_text),
            query[:60],
        )

        return BuiltContext(
            context_text=context_text,
            citations=citations,
            is_empty=False,
            candidate_count=len(top),
            reranked=reranked,
        )

    def _backfill_parents(
        self,
        candidates: list[RetrievalCandidate],
        *,
        qdrant,
        collection: str,
    ) -> list[RetrievalCandidate]:
        present_ids = {candidate.chunk_id for candidate in candidates}
        result: list[RetrievalCandidate] = []
        inserted_parent_ids: set[str] = set()

        for candidate in candidates:
            parent_id = candidate.payload.get("parent_id") or None
            if not parent_id or parent_id in present_ids:
                result.append(candidate)
                continue

            if qdrant is not None and collection and parent_id not in inserted_parent_ids:
                parent_candidate = self._fetch_parent_from_qdrant(
                    parent_id,
                    candidate,
                    qdrant,
                    collection,
                )
                if parent_candidate is not None:
                    result.append(parent_candidate)
                    inserted_parent_ids.add(parent_id)
                    present_ids.add(parent_id)
                    continue

            result.append(candidate)

        return result

    def _fetch_parent_from_qdrant(
        self,
        parent_id: str,
        child: RetrievalCandidate,
        qdrant,
        collection: str,
    ) -> RetrievalCandidate | None:
        from core.storage.chunk_serializer import _chunk_id_to_uuid

        parent_uuid = _chunk_id_to_uuid(parent_id)
        try:
            points = qdrant.retrieve(
                collection_name=collection,
                ids=[parent_uuid],
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                return None
            payload = points[0].payload or {}
            return RetrievalCandidate(
                chunk_id=parent_id,
                doc_id=payload.get("doc_id", child.doc_id),
                text=payload.get("text", ""),
                payload=dict(payload),
                bm25_score=child.bm25_score,
                dense_scores=child.dense_scores,
                rrf_score=child.rrf_score,
                rerank_score=child.rerank_score,
                source_retriever_ids=child.source_retriever_ids,
                rank_in_retriever=child.rank_in_retriever,
            )
        except Exception as exc:
            logger.debug(
                "ContextBuilder: parent fetch failed for parent_id=%s: %s",
                parent_id,
                exc,
            )
            return None

    @staticmethod
    def _dedup(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
        seen: set[str] = set()
        result: list[RetrievalCandidate] = []
        for candidate in candidates:
            if candidate.chunk_id in seen:
                continue
            seen.add(candidate.chunk_id)
            result.append(candidate)
        return result

    def _format_block(self, index: int, candidate: RetrievalCandidate) -> str:
        title = candidate.payload.get("title") or candidate.doc_id
        lines = [f"[{index}] {title}", candidate.text]

        if self._include_scores:
            score_parts: list[str] = []
            if candidate.rerank_score is not None:
                score_parts.append(f"rerank={candidate.rerank_score:.4f}")
            if candidate.rrf_score is not None:
                score_parts.append(f"rrf={candidate.rrf_score:.4f}")
            if score_parts:
                lines.append(f"({', '.join(score_parts)})")

        return "\n".join(lines)

    @staticmethod
    def _make_citation(index: int, candidate: RetrievalCandidate) -> Citation:
        payload = candidate.payload
        return Citation(
            index=index,
            chunk_id=candidate.chunk_id,
            doc_id=candidate.doc_id,
            title=payload.get("title", ""),
            source=payload.get("source", ""),
            bm25_score=candidate.bm25_score,
            dense_scores=dict(candidate.dense_scores),
            rrf_score=candidate.rrf_score,
            rerank_score=candidate.rerank_score,
            extra={
                key: payload[key]
                for key in (
                    "text",
                    "category",
                    "created_time",
                    "author",
                    "config_version",
                    "hierarchy_level",
                    "position",
                    "parent_id",
                )
                if key in payload
            },
        )

    @classmethod
    def from_config(cls, cfg) -> "ContextBuilder":
        context_top_k = cfg.retrieval.rerank.context_top_k or 8
        return cls(context_top_k=context_top_k)
