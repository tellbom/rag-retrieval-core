"""
core/query/preprocessor.py

QueryPreprocessor: first stage of the query pipeline.

Orchestrates:
    raw query
        → QueryNormalizer  (always)
        → QueryRewriter    (optional, switch per request)
        → ProcessedQuery

Design rules
------------
- Normalisation always runs. A blank normalised query returns a ProcessedQuery
  with effective_query="" — callers decide how to handle empty queries.
- Rewrite is controlled per-call by the `enable_rewrite` parameter, not
  globally forced. This lets callers opt in based on request context
  (e.g. complex natural-language questions vs short keyword queries).
- If the rewrite produces the same text as normalized_query, it is discarded
  (rewrite_used=False) to avoid redundant embedding calls downstream.
- Filters are attached as-is from the caller — no filter inference from text.

Public API
----------
    preprocessor = QueryPreprocessor(normalizer, rewriter)
    # or:
    preprocessor = QueryPreprocessor.from_config(cfg)

    pq = preprocessor.process(
        raw_query="用户的原始查询",
        filters=QueryFilters(business_type="policy"),
        business_type="policy",
        enable_rewrite=True,
    )
    # pq.effective_query  → use for embedding + BM25
    # pq.filters          → push into ES/Qdrant
"""

from __future__ import annotations

import logging

from core.config.models import AppConfig
from core.query.normalizer import QueryNormalizer
from core.query.processed_query import ProcessedQuery, QueryFilters
from core.query.rewriter import QueryRewriter

logger = logging.getLogger(__name__)


class QueryPreprocessor:
    """
    Orchestrates query normalisation and optional LLM rewriting.

    Parameters
    ----------
    normalizer: QueryNormalizer instance (stateless, shared).
    rewriter:   QueryRewriter instance (stateless client, shared).
    """

    def __init__(
        self,
        normalizer: QueryNormalizer,
        rewriter: QueryRewriter,
    ) -> None:
        self._normalizer = normalizer
        self._rewriter = rewriter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        raw_query: str,
        *,
        filters: QueryFilters | None = None,
        business_type: str = "",
        enable_rewrite: bool = False,
    ) -> ProcessedQuery:
        """
        Process a raw query string into a ProcessedQuery.

        Parameters
        ----------
        raw_query:      The verbatim user query.
        filters:        Pre-built hard filters. If None, an empty QueryFilters
                        is used.
        business_type:  Caller-provided tag; also set on filters if not already.
        enable_rewrite: Whether to attempt LLM rewriting for this request.
                        The rewriter still degrades gracefully if LLM is down.

        Returns
        -------
        ProcessedQuery — always succeeds; never raises.
        """
        # --- Step 1: normalise (always) ---
        normalized = self._normalizer.normalize(raw_query)

        # --- Step 2: build filters ---
        resolved_filters = filters or QueryFilters()
        if business_type and resolved_filters.business_type is None:
            resolved_filters.business_type = business_type

        # --- Step 3: optional rewrite ---
        rewritten: str | None = None
        rewrite_used = False

        if enable_rewrite and normalized:
            rewritten = self._rewriter.rewrite(normalized)
            # Discard rewrite if identical to normalized (no-op rewrite)
            if rewritten and rewritten.strip() == normalized.strip():
                logger.debug(
                    "QueryPreprocessor: rewrite identical to normalized, discarding"
                )
                rewritten = None

            if rewritten:
                rewrite_used = True

        pq = ProcessedQuery(
            original_query=raw_query,
            normalized_query=normalized,
            rewritten_query=rewritten,
            rewrite_used=rewrite_used,
            filters=resolved_filters,
            business_type=business_type,
        )
        logger.debug("QueryPreprocessor: %s", pq.summary())
        return pq

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "QueryPreprocessor":
        """
        Build a QueryPreprocessor from AppConfig.
        If an LLM is configured, the rewriter is available; callers still pass
        enable_rewrite=True per request when they want LLM rewriting.
        """
        normalizer = QueryNormalizer()
        rewriter = QueryRewriter.from_config(cfg, enabled=True)
        return cls(normalizer=normalizer, rewriter=rewriter)

    @classmethod
    def from_config_with_rewrite(cls, cfg: AppConfig) -> "QueryPreprocessor":
        """
        Build with the rewriter enabled (requires models.enhancement_llm).
        Use when the caller always wants LLM rewriting available.
        """
        normalizer = QueryNormalizer()
        rewriter = QueryRewriter.from_config(cfg, enabled=True)
        return cls(normalizer=normalizer, rewriter=rewriter)
