"""
core/query/processed_query.py

ProcessedQuery: the canonical DTO flowing through the query pipeline
from QueryPreprocessor to Retrievers.

Design rules
------------
- `original_query`  — the raw string exactly as received from the caller.
  Never mutated. Used in logs, citations, and answer attribution.
- `normalized_query` — the cleaned, normalised form. Always present.
  Used as the BM25 search text and the fallback embedding input.
- `rewritten_query`  — LLM-rewritten form. None if rewrite is disabled
  or the LLM was unavailable (degraded). When present, this is what
  gets embedded for dense retrieval.
- `effective_query`  — the query that pipeline stages should USE:
  `rewritten_query` if available, otherwise `normalized_query`.
  This is the single string every downstream component reads.
- `rewrite_used`     — True only when a successful LLM rewrite was applied.
- `filters`          — pre-parsed hard filters extracted from the query
  or passed in by the caller. Pushed directly into ES/Qdrant queries.
- `business_type`    — caller-provided tag for per-business retrieval tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QueryFilters:
    """
    Hard filters to push into ES and Qdrant queries.
    All fields are optional; None means "no filter on this dimension".

    Callers set these explicitly — the query preprocessor does NOT
    infer filters from query text (that is a Phase 2 query-router concern).
    """
    business_type: str | None = None
    category: str | None = None
    doc_id: str | None = None
    # Date range: ISO-8601 strings or None
    created_after: str | None = None
    created_before: str | None = None
    # Arbitrary extra filters from the caller (field_name → value)
    extra: dict[str, str] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return all([
            self.business_type is None,
            self.category is None,
            self.doc_id is None,
            self.created_after is None,
            self.created_before is None,
            not self.extra,
        ])


@dataclass
class ProcessedQuery:
    """
    Output of QueryPreprocessor; input to Retrievers (P1-12).

    Attributes
    ----------
    original_query:
        Verbatim user input. Never modified.
    normalized_query:
        Cleaned, whitespace-normalised form. Always populated.
    rewritten_query:
        LLM-rewritten form (None if disabled/failed).
    rewrite_used:
        True when rewritten_query is available and will be used.
    filters:
        Hard filters to push into engine queries.
    business_type:
        Routing/tuning tag. Copied to filters.business_type if not
        already set there.
    """
    original_query: str
    normalized_query: str
    rewritten_query: str | None = None
    rewrite_used: bool = False
    filters: QueryFilters = field(default_factory=QueryFilters)
    business_type: str = ""

    @property
    def effective_query(self) -> str:
        """
        The query text that pipeline stages should USE.
        Returns rewritten_query if a successful rewrite was applied,
        otherwise normalized_query.
        """
        if self.rewrite_used and self.rewritten_query:
            return self.rewritten_query
        return self.normalized_query

    def summary(self) -> str:
        return (
            f"original={self.original_query!r} "
            f"rewrite_used={self.rewrite_used} "
            f"effective={self.effective_query!r} "
            f"filters_empty={self.filters.is_empty()}"
        )
