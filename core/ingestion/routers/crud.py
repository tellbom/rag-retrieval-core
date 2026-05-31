"""
ingestion/routers/crud.py

FastAPI router for document CRUD and corpus rebuild operations.

  POST   /crud/add                 — ingest a new document
  GET    /crud/documents/{doc_id}  — check existence in original-text store
  DELETE /crud/documents/{doc_id}  — delete document + all its chunks
  PUT    /crud/documents/{doc_id}  — update document (delete-then-insert)
  POST   /crud/rebuild             — trigger full corpus rebuild
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/crud", tags=["crud"])

_state: "_CrudRouterState | None" = None


class _CrudRouterState:
    def __init__(self, crud_svc, rebuild_svc, original_store):
        self.crud_svc = crud_svc
        self.rebuild_svc = rebuild_svc
        self.original_store = original_store


def init_router(crud_svc, rebuild_svc, original_store) -> None:
    global _state
    _state = _CrudRouterState(crud_svc, rebuild_svc, original_store)


def _require_state() -> _CrudRouterState:
    if _state is None:
        raise HTTPException(status_code=503, detail="CRUD service not initialised")
    return _state


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AddDocumentRequest(BaseModel):
    """
    Request body for POST /crud/add.

    The retrieval core is business-agnostic.  The caller provides:
    - doc_id:          a stable, unique identifier owned by the business system.
    - raw_text:        the original document text (pre-cleaning).
    - business_type:   an arbitrary string tag for filter pushdown.
    - source_metadata: any key-value pairs the business system wants stored
                       (persisted and indexed as filter/display fields).
    """
    doc_id: str = Field(description="Stable unique document identifier from the business system")
    raw_text: str = Field(min_length=1, description="Original document text (pre-cleaning)")
    business_type: str = Field(default="", description="Business-defined document type tag")
    source_metadata: dict = Field(
        default_factory=dict,
        description=(
            "Arbitrary metadata (title, source URL, dates, category, etc.). "
            "Keys matching standard_fields in config will be indexed as filters."
        ),
    )


class UpdateDocumentRequest(BaseModel):
    raw_text: str = Field(min_length=1, description="New document text")
    business_type: str = Field(default="", description="Updated business type tag (optional)")
    source_metadata: dict = Field(
        default_factory=dict,
        description="Updated metadata (empty dict = keep existing metadata)",
    )


class CrudOperationResponse(BaseModel):
    doc_id: str
    operation: str
    success: bool
    indexed_count: int
    failed_count: int
    message: str


class DocumentExistsResponse(BaseModel):
    doc_id: str
    exists: bool
    business_type: str = ""
    stored_at: str = ""


class RebuildResponse(BaseModel):
    """
    Full result of a corpus rebuild operation.
    Rebuild runs synchronously (blocking); for large corpora consider
    triggering it via a background job and polling /indexer/status.
    """
    success: bool
    new_es_index: str
    new_qdrant_collection: str
    docs_processed: int
    chunks_indexed: int
    chunks_failed: int
    error: str
    progress_log: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post(
    "/add",
    response_model=CrudOperationResponse,
    summary="Ingest a document",
    description=(
        "Runs the full ingestion pipeline: clean → enhance → chunk → embed → index. "
        "Idempotent: re-adding the same doc_id replaces existing chunks. "
        "The original text is persisted in the original-text store for future rebuilds."
    ),
)
def add_document(req: AddDocumentRequest) -> CrudOperationResponse:
    state = _require_state()
    result = state.crud_svc.add(
        req.doc_id,
        req.raw_text,
        business_type=req.business_type,
        source_metadata=req.source_metadata,
    )
    if not result.success and result.indexed_count == 0:
        raise HTTPException(status_code=500, detail=result.message)
    return CrudOperationResponse(
        doc_id=result.doc_id,
        operation=result.operation,
        success=result.success,
        indexed_count=result.indexed_count,
        failed_count=result.failed_count,
        message=result.message,
    )


@router.get(
    "/documents/{doc_id}",
    response_model=DocumentExistsResponse,
    summary="Check if a document exists in the original-text store",
)
def get_document(doc_id: str) -> DocumentExistsResponse:
    state = _require_state()
    entry = state.original_store.get(doc_id)
    if entry is None:
        return DocumentExistsResponse(doc_id=doc_id, exists=False)
    return DocumentExistsResponse(
        doc_id=doc_id,
        exists=True,
        business_type=entry.business_type,
        stored_at=entry.stored_at,
    )


@router.delete(
    "/documents/{doc_id}",
    response_model=CrudOperationResponse,
    summary="Delete a document and all its chunks",
    description=(
        "Removes all chunks for this doc_id from ES and Qdrant, "
        "and deletes the original text from the original-text store. "
        "Idempotent: deleting a non-existent doc_id returns success."
    ),
)
def delete_document(doc_id: str) -> CrudOperationResponse:
    state = _require_state()
    result = state.crud_svc.delete(doc_id)
    return CrudOperationResponse(
        doc_id=result.doc_id,
        operation=result.operation,
        success=result.success,
        indexed_count=0,
        failed_count=0,
        message=result.message,
    )


@router.put(
    "/documents/{doc_id}",
    response_model=CrudOperationResponse,
    summary="Update a document",
    description=(
        "Delete-then-insert: removes all existing chunks for this doc_id "
        "and re-runs the full ingestion pipeline with the new text. "
        "Chunk boundaries may change — this is the only correct update strategy. "
        "If source_metadata is empty dict, the existing metadata is preserved."
    ),
)
def update_document(doc_id: str, req: UpdateDocumentRequest) -> CrudOperationResponse:
    state = _require_state()
    result = state.crud_svc.update(
        doc_id,
        req.raw_text,
        business_type=req.business_type or "",
        source_metadata=req.source_metadata or None,
    )
    if not result.success and result.indexed_count == 0:
        raise HTTPException(status_code=500, detail=result.message)
    return CrudOperationResponse(
        doc_id=result.doc_id,
        operation=result.operation,
        success=result.success,
        indexed_count=result.indexed_count,
        failed_count=result.failed_count,
        message=result.message,
    )


@router.post(
    "/rebuild",
    response_model=RebuildResponse,
    summary="Trigger full corpus rebuild",
    description=(
        "Re-ingests ALL documents from the original-text store into a NEW versioned "
        "ES index and Qdrant collection. After completion, atomically switches "
        "the aliases to the new stores and drops the old ones. "
        "Zero-downtime: the live alias is not touched until the rebuild succeeds. "
        "This is a LONG-RUNNING BLOCKING operation. "
        "Trigger when: chunking config changed, embedding model version changed."
    ),
)
def trigger_rebuild() -> RebuildResponse:
    state = _require_state()

    # Drive the generator manually to capture the RebuildResult returned
    # via StopIteration.value — Python generators return their value through
    # the StopIteration exception, not through yield.
    gen = state.rebuild_svc.rebuild()
    rebuild_result = None

    try:
        while True:
            progress = next(gen)
            if progress.phase == "error":
                raise HTTPException(
                    status_code=500,
                    detail=f"Rebuild failed: {progress.error}",
                )
    except StopIteration as stop:
        # The generator's return value is the RebuildResult
        rebuild_result = stop.value

    if rebuild_result is None or not rebuild_result.success:
        error_msg = rebuild_result.error if rebuild_result else "Rebuild produced no result"
        raise HTTPException(status_code=500, detail=error_msg)

    return RebuildResponse(
        success=rebuild_result.success,
        new_es_index=rebuild_result.new_es_index,
        new_qdrant_collection=rebuild_result.new_qdrant_collection,
        docs_processed=rebuild_result.docs_processed,
        chunks_indexed=rebuild_result.chunks_indexed,
        chunks_failed=rebuild_result.chunks_failed,
        error=rebuild_result.error,
        progress_log=rebuild_result.progress_log,
    )
