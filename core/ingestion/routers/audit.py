"""
core/ingestion/routers/audit.py

FastAPI router for chunk quality audit operations.

  POST  /audit/doc/{doc_id}   — trigger audit for one document (synchronous)
  GET   /audit/doc/{doc_id}   — fetch the latest audit report for a document
  GET   /audit/pending        — list doc_ids that have not yet been audited

Injection pattern
-----------------
Follows the same module-level _state + init_router() pattern used by all
other routers in this service (crud, ingest, indexer).  init_router() is
called once from ingestion/app.py lifespan before the server starts serving.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/audit", tags=["audit"])

_state: "_AuditRouterState | None" = None


class _AuditRouterState:
    def __init__(self, auditor, review_store, chunk_index: str) -> None:
        self.auditor = auditor
        self.review_store = review_store
        self.chunk_index = chunk_index


def init_router(auditor, review_store, chunk_index: str) -> None:
    """
    Initialise the audit router with its dependencies.

    Parameters
    ----------
    auditor:
        ChunkQualityAuditor instance.
    review_store:
        ChunkReviewStore instance (already provisioned via ensure_index()).
    chunk_index:
        ES alias/index name for the main chunk data.  Used by
        GET /audit/pending to enumerate known doc_ids.
    """
    global _state
    _state = _AuditRouterState(auditor, review_store, chunk_index)


def _require_state() -> _AuditRouterState:
    if _state is None:
        raise HTTPException(status_code=503, detail="Audit service not initialised")
    return _state


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class AuditSummary(BaseModel):
    ok: int = 0
    should_merge: int = 0
    should_split: int = 0
    boundary_issue: int = 0
    audit_failed: int = 0


class AuditReportResponse(BaseModel):
    """
    Full audit report for one document.

    The `groups` field contains the per-group LLM verdicts including
    the raw chunk_ids, reason text, and affected_chunk_ids.
    """
    doc_id: str
    audited_at: str
    total_groups: int
    skipped_groups: int
    audit_summary: dict[str, int]
    groups: list[dict[str, Any]]


class TriggerAuditRequest(BaseModel):
    dry_run: bool = Field(
        default=False,
        description=(
            "If true, build and log LLM prompts without calling the LLM. "
            "The report will be returned but NOT saved to the review index."
        ),
    )


class TriggerAuditResponse(BaseModel):
    doc_id: str
    audited_at: str
    total_groups: int
    skipped_groups: int
    audit_summary: dict[str, int]
    saved: bool = Field(
        description="True if the report was persisted to the review index."
    )
    groups: list[dict[str, Any]]


class PendingAuditResponse(BaseModel):
    pending_doc_ids: list[str]
    count: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/doc/{doc_id}",
    response_model=TriggerAuditResponse,
    summary="Trigger chunk quality audit for a document",
    description=(
        "Fetches all chunks for doc_id from ES, groups them by lc_group_id "
        "(Phase 2) or parent_id (Phase 1 fallback), and calls the intranet LLM "
        "once per group to audit semantic boundary quality.\n\n"
        "The report is saved to the review index unless dry_run=true.\n\n"
        "This call is synchronous and blocks until all LLM calls complete. "
        "For documents with many chunk groups this may take 10–60 seconds."
    ),
)
def trigger_audit(doc_id: str, req: TriggerAuditRequest) -> TriggerAuditResponse:
    state = _require_state()

    report = state.auditor.audit(doc_id, dry_run=req.dry_run)

    saved = False
    if not req.dry_run:
        try:
            state.review_store.save(report)
            saved = True
        except Exception as exc:
            logger.error(
                "Failed to save audit report for doc_id=%s: %s", doc_id, exc
            )
            # Return the report even if persistence failed; caller can see
            # saved=False and retry.

    return TriggerAuditResponse(
        doc_id=report["doc_id"],
        audited_at=report["audited_at"],
        total_groups=report["total_groups"],
        skipped_groups=report["skipped_groups"],
        audit_summary=report.get("audit_summary", {}),
        saved=saved,
        groups=report.get("groups", []),
    )


@router.get(
    "/doc/{doc_id}",
    response_model=AuditReportResponse,
    summary="Get the latest audit report for a document",
    description=(
        "Returns the most recent audit report stored in the review index "
        "for the given doc_id.  Returns 404 if no audit has been run yet."
    ),
)
def get_audit_report(doc_id: str) -> AuditReportResponse:
    state = _require_state()

    report = state.review_store.get_latest(doc_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"No audit report found for doc_id={doc_id!r}. "
                   "Run POST /audit/doc/{doc_id} first.",
        )

    return AuditReportResponse(
        doc_id=report["doc_id"],
        audited_at=report["audited_at"],
        total_groups=report.get("total_groups", 0),
        skipped_groups=report.get("skipped_groups", 0),
        audit_summary=report.get("audit_summary", {}),
        groups=report.get("groups", []),
    )


@router.get(
    "/pending",
    response_model=PendingAuditResponse,
    summary="List doc_ids pending audit",
    description=(
        "Returns doc_ids that exist in the chunk index but have no audit "
        "report in the review index.  Used by the scheduler (CQR-04) and "
        "for operator visibility.\n\n"
        "To force re-audit of an already-reviewed doc_id, call "
        "DELETE /audit/doc/{doc_id}/reviews to clear its review history, "
        "then it will reappear here."
    ),
)
def list_pending(
    limit: int = Query(default=100, ge=1, le=1000, description="Max doc_ids to return"),
) -> PendingAuditResponse:
    state = _require_state()

    pending = state.review_store.list_pending_doc_ids(
        chunk_index=state.chunk_index,
        limit=limit,
    )
    return PendingAuditResponse(pending_doc_ids=pending, count=len(pending))


@router.delete(
    "/doc/{doc_id}/reviews",
    summary="Clear audit history for a document",
    description=(
        "Deletes all review documents for doc_id from the review index. "
        "After this call, doc_id will reappear in GET /audit/pending "
        "and the scheduler will re-audit it on the next run."
    ),
)
def clear_reviews(doc_id: str) -> dict[str, Any]:
    state = _require_state()

    deleted = state.review_store.clear_reviewed(doc_id)
    return {"doc_id": doc_id, "deleted_reviews": deleted}
