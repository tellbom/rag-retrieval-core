"""
query/routers/preprocess.py

FastAPI router exposing the query preprocessor.

  POST /query/preprocess   — normalise + optional rewrite a query

This endpoint is useful for:
- Operator testing (verify normalisation and rewrite behaviour).
- Business adapters that want to pre-process before sending to retrieval.
- Debugging unexpected retrieval results (inspect what effective_query was).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/query", tags=["query"])

_state: "_PreprocessRouterState | None" = None


class _PreprocessRouterState:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor


def init_router(preprocessor) -> None:
    global _state
    _state = _PreprocessRouterState(preprocessor)


def _require_state() -> _PreprocessRouterState:
    if _state is None:
        raise HTTPException(status_code=503, detail="Query preprocessor not initialised")
    return _state


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PreprocessRequest(BaseModel):
    """
    Request body for POST /query/preprocess.

    Callers MUST supply the raw query text.
    All other fields are optional — the retrieval core uses them to
    narrow the search scope and to apply the correct per-business tuning.
    """
    query: str = Field(
        min_length=1,
        description="Raw user query (verbatim, pre-normalisation)",
    )
    business_type: str = Field(
        default="",
        description=(
            "Business-defined document type tag. "
            "Used as a filter field in ES and Qdrant queries. "
            "Leave empty to search across all business types."
        ),
    )
    enable_rewrite: bool = Field(
        default=False,
        description=(
            "Whether to attempt LLM query rewriting. "
            "Requires models.enhancement_llm to be configured. "
            "If the LLM is unavailable, the normalised query is used as fallback."
        ),
    )
    # Hard filters (optional)
    filter_category: str | None = Field(
        default=None,
        description="Filter by document category (exact match, keyword field)",
    )
    filter_doc_id: str | None = Field(
        default=None,
        description="Restrict search to a single document by doc_id",
    )
    filter_created_after: str | None = Field(
        default=None,
        description="ISO-8601 date — only return documents created after this date",
    )
    filter_created_before: str | None = Field(
        default=None,
        description="ISO-8601 date — only return documents created before this date",
    )


class PreprocessResponse(BaseModel):
    original_query: str = Field(description="Verbatim input (unchanged)")
    normalized_query: str = Field(description="After rule-based normalisation")
    rewritten_query: str | None = Field(
        description="LLM-rewritten form (null if rewrite disabled or failed)"
    )
    effective_query: str = Field(
        description=(
            "The query text that will be used for retrieval: "
            "rewritten_query if available, otherwise normalized_query"
        )
    )
    rewrite_used: bool = Field(
        description="True when a successful LLM rewrite was applied"
    )
    filters_applied: dict = Field(
        description="The filter set that will be pushed into engine queries"
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/preprocess",
    response_model=PreprocessResponse,
    summary="Normalise and optionally rewrite a query",
    description=(
        "Applies rule-based normalisation (always) and optional LLM rewriting "
        "(when enable_rewrite=true). Returns the effective_query that will be "
        "used for retrieval. Useful for debugging and adapter validation."
    ),
)
def preprocess_query(req: PreprocessRequest) -> PreprocessResponse:
    from core.query.processed_query import QueryFilters

    state = _require_state()

    filters = QueryFilters(
        business_type=req.business_type or None,
        category=req.filter_category,
        doc_id=req.filter_doc_id,
        created_after=req.filter_created_after,
        created_before=req.filter_created_before,
    )

    pq = state.preprocessor.process(
        req.query,
        filters=filters,
        business_type=req.business_type,
        enable_rewrite=req.enable_rewrite,
    )

    # Serialise filters for the response
    filters_out: dict = {}
    if pq.filters.business_type:
        filters_out["business_type"] = pq.filters.business_type
    if pq.filters.category:
        filters_out["category"] = pq.filters.category
    if pq.filters.doc_id:
        filters_out["doc_id"] = pq.filters.doc_id
    if pq.filters.created_after:
        filters_out["created_after"] = pq.filters.created_after
    if pq.filters.created_before:
        filters_out["created_before"] = pq.filters.created_before
    filters_out.update(pq.filters.extra)

    return PreprocessResponse(
        original_query=pq.original_query,
        normalized_query=pq.normalized_query,
        rewritten_query=pq.rewritten_query,
        effective_query=pq.effective_query,
        rewrite_used=pq.rewrite_used,
        filters_applied=filters_out,
    )
