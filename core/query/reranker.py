"""Cross-encoder reranking stage for fused retrieval candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from core.config.models import AppConfig, RerankConfig
from core.query.retrieval_candidate import RetrievalCandidate
from core.serving.registry import ServingRegistry
from core.serving.rerank import RerankUnavailableError, RerankerClient

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result returned by the query reranking stage."""

    reranked: bool
    candidates: list[RetrievalCandidate]
    degradation_reason: str = ""

    @property
    def top_k(self) -> int:
        return len(self.candidates)


class Reranker:
    """Apply cross-encoder scores to the top candidates from the fused pool."""

    def __init__(
        self,
        client: RerankerClient,
        cfg: RerankConfig,
    ) -> None:
        self._client = client
        self._cfg = cfg

    def rerank(
        self,
        query_text: str,
        fused_candidates: list[RetrievalCandidate],
        *,
        top_k: int | None = None,
    ) -> RerankResult:
        """Rerank the top-K fused candidates, degrading to RRF order on failure."""
        if not self._cfg.enabled:
            return RerankResult(
                reranked=False,
                candidates=fused_candidates,
                degradation_reason="rerank.enabled=false",
            )

        if not fused_candidates:
            return RerankResult(reranked=False, candidates=[])

        k = top_k if top_k is not None else (self._cfg.top_k or len(fused_candidates))
        to_rerank = fused_candidates[:k]
        texts = [candidate.text for candidate in to_rerank]

        try:
            scores = self._client.rerank(query_text, texts)
        except RerankUnavailableError as exc:
            logger.warning("Reranker unavailable, degrading to RRF order: %s", exc)
            return RerankResult(
                reranked=False,
                candidates=to_rerank,
                degradation_reason=str(exc),
            )

        if len(scores) != len(to_rerank):
            reason = (
                f"Reranker returned {len(scores)} scores for "
                f"{len(to_rerank)} candidates; degrading to RRF order"
            )
            logger.warning(reason)
            return RerankResult(
                reranked=False,
                candidates=to_rerank,
                degradation_reason=reason,
            )

        scored: list[RetrievalCandidate] = []
        for score in scores:
            index = score.index
            if index < 0 or index >= len(to_rerank):
                logger.warning("Reranker returned out-of-range index %d", index)
                continue
            scored.append(replace(to_rerank[index], rerank_score=score.score))

        if not scored:
            return RerankResult(
                reranked=False,
                candidates=to_rerank,
                degradation_reason="All reranker score indices were invalid",
            )

        # Must use explicit `is not None`; 0.0 is a valid cross-encoder score
        # and must not be treated as an absent score.
        scored.sort(
            key=lambda candidate: candidate.rerank_score
            if candidate.rerank_score is not None
            else float("-inf"),
            reverse=True,
        )

        # Apply min_score threshold when configured.
        # Candidates below the threshold are dropped before context building,
        # which lets the answer generator return "insufficient context" for
        # queries where no retrieved chunk meets the minimum relevance bar
        # (e.g. topic-absent or factual-negation queries).
        # 0.0 is a valid score, so compare with `is not None` guard.
        min_score = self._cfg.min_score
        if min_score is not None:
            before = len(scored)
            scored = [
                c for c in scored
                if c.rerank_score is not None and c.rerank_score >= min_score
            ]
            if len(scored) < before:
                logger.debug(
                    "Reranker: min_score=%.4f filtered %d → %d candidates (query=%r)",
                    min_score, before, len(scored), query_text[:60],
                )

        if not scored:
            logger.debug(
                "Reranker: all candidates filtered by min_score=%.4f (query=%r)",
                min_score if min_score is not None else 0.0,
                query_text[:60],
            )
            return RerankResult(reranked=True, candidates=[])

        logger.debug(
            "Reranker: %d candidates, top rerank_score=%.4f (query=%r)",
            len(scored),
            scored[0].rerank_score if scored[0].rerank_score is not None else 0.0,
            query_text[:60],
        )
        return RerankResult(reranked=True, candidates=scored)

    @classmethod
    def from_config(
        cls,
        cfg: AppConfig,
        registry: ServingRegistry,
    ) -> "Reranker":
        return cls(
            client=registry.reranker_client,
            cfg=cfg.retrieval.rerank,
        )

    @classmethod
    def disabled(cls) -> "Reranker":
        return cls(
            client=None,  # type: ignore[arg-type]
            cfg=RerankConfig(enabled=False),
        )
