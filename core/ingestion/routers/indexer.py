"""
ingestion/routers/indexer.py

FastAPI router exposing indexer operations to operators and callers.

  GET  /indexer/status             — pending failure count + summary
  GET  /indexer/failures           — list pending failed_index_records
  POST /indexer/retry              — trigger retry command
  POST /indexer/reconcile          — trigger reconciliation command
  DELETE /indexer/failures/resolved — purge resolved records
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/indexer", tags=["indexer"])

_state: "_IndexerRouterState | None" = None


class _IndexerRouterState:
    def __init__(self, indexer, fail_store, retry_cmd, reconcile_cmd):
        self.indexer = indexer
        self.fail_store = fail_store
        self.retry_cmd = retry_cmd
        self.reconcile_cmd = reconcile_cmd


def init_router(indexer, fail_store, retry_cmd, reconcile_cmd) -> None:
    global _state
    _state = _IndexerRouterState(indexer, fail_store, retry_cmd, reconcile_cmd)


def _require_state() -> _IndexerRouterState:
    if _state is None:
        raise HTTPException(status_code=503, detail="Indexer not initialised")
    return _state


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class IndexerStatusResponse(BaseModel):
    pending_failure_count: int = Field(
        description="Number of unresolved failed_index_records"
    )
    message: str = Field(
        description="Human-readable summary"
    )


class FailedRecordOut(BaseModel):
    id: int
    chunk_id: str
    doc_id: str
    failure_mode: str
    error_msg: str | None
    attempt_count: int
    created_at: str
    last_attempt: str | None


class RetryRequest(BaseModel):
    max_attempts: int = Field(
        default=5,
        ge=1, le=20,
        description="Skip records that have already failed this many times",
    )
    limit: int = Field(
        default=200,
        ge=1, le=1000,
        description="Max records to process in this run",
    )


class RetryResponse(BaseModel):
    resolved_count: int
    still_failing_count: int
    skipped_count: int


class ReconcileResponse(BaseModel):
    es_only_count: int = Field(description="Chunks in ES but missing from Qdrant")
    qdrant_only_count: int = Field(description="Chunks in Qdrant but missing from ES")
    in_sync_count: int = Field(description="Chunks present in both stores")
    newly_recorded: int = Field(description="New failed_index_records created")
    already_pending: int = Field(description="Skipped — already in failed_index_records")
    errors: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/status", response_model=IndexerStatusResponse, summary="Indexer health")
def indexer_status() -> IndexerStatusResponse:
    state = _require_state()
    count = state.fail_store.pending_count()
    return IndexerStatusResponse(
        pending_failure_count=count,
        message=(
            f"{count} pending failure(s) in failed_index_records. "
            "Use POST /indexer/retry to replay, or POST /indexer/reconcile "
            "to detect drift."
        ) if count else "All chunks indexed successfully.",
    )


@router.get(
    "/failures",
    response_model=list[FailedRecordOut],
    summary="List pending failed_index_records",
)
def list_failures(
    limit: int = Query(default=100, ge=1, le=1000),
    max_attempts: int | None = Query(default=None, ge=1),
) -> list[FailedRecordOut]:
    state = _require_state()
    records = state.fail_store.list_pending(limit=limit, max_attempts=max_attempts)
    return [
        FailedRecordOut(
            id=r.id,
            chunk_id=r.chunk_id,
            doc_id=r.doc_id,
            failure_mode=r.failure_mode,
            error_msg=r.error_msg,
            attempt_count=r.attempt_count,
            created_at=r.created_at,
            last_attempt=r.last_attempt,
        )
        for r in records
    ]


@router.post(
    "/retry",
    response_model=RetryResponse,
    summary="Replay pending failed_index_records",
    description=(
        "Attempts to re-write chunks that previously failed to index. "
        "Marks successfully retried records as resolved. "
        "Records that still fail are left pending with incremented attempt_count."
    ),
)
def run_retry(req: RetryRequest) -> RetryResponse:
    state = _require_state()
    report = state.retry_cmd.run(
        max_attempts=req.max_attempts,
        limit=req.limit,
    )
    return RetryResponse(
        resolved_count=report.resolved_count,
        still_failing_count=report.still_failing_count,
        skipped_count=report.skipped_count,
    )


@router.post(
    "/reconcile",
    response_model=ReconcileResponse,
    summary="Reconcile ES and Qdrant chunk sets",
    description=(
        "Scans all chunk_ids in both stores and compares them. "
        "Any drift (chunk in one store but not the other) is recorded "
        "in failed_index_records for the retry command to repair. "
        "This is an offline operation — for large corpora it may take minutes."
    ),
)
def run_reconcile() -> ReconcileResponse:
    state = _require_state()
    report = state.reconcile_cmd.run()
    return ReconcileResponse(
        es_only_count=report.es_only_count,
        qdrant_only_count=report.qdrant_only_count,
        in_sync_count=report.in_sync_count,
        newly_recorded=report.newly_recorded,
        already_pending=report.already_pending,
        errors=report.errors,
    )


@router.delete(
    "/failures/resolved",
    summary="Purge resolved failed_index_records",
    description="Remove all records where resolved=1. Returns count deleted.",
)
def purge_resolved() -> dict:
    state = _require_state()
    count = state.fail_store.delete_resolved()
    return {"deleted_count": count}
