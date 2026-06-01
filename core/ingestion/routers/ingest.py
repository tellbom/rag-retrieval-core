"""FastAPI routes for document ingestion through IngestionPipeline."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/ingest", tags=["ingest"])

_state: "_IngestRouterState | None" = None


class _IngestRouterState:
    def __init__(self, ingestion_pipeline, original_store) -> None:
        self.pipeline = ingestion_pipeline
        self.original_store = original_store


def init_router(ingestion_pipeline, original_store) -> None:
    global _state
    _state = _IngestRouterState(ingestion_pipeline, original_store)


def _require_state() -> _IngestRouterState:
    if _state is None:
        raise HTTPException(status_code=503, detail="Ingestion pipeline not initialised")
    return _state


class IngestRequest(BaseModel):
    doc_id: str = Field(description="Stable document identifier")
    raw_text: str = Field(min_length=1, description="Original document text")
    business_type: str = Field(default="", description="Business type tag")
    source_metadata: dict = Field(default_factory=dict)


class BatchIngestRequest(BaseModel):
    documents: list[IngestRequest] = Field(min_length=1, max_length=50)
    business_type: str = Field(default="")


class IngestResponse(BaseModel):
    doc_id: str
    success: bool
    indexed_count: int
    failed_count: int
    enhanced: bool
    message: str


class BatchIngestResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[IngestResponse]


@router.post("", response_model=IngestResponse, summary="Ingest a single document")
def ingest_document(req: IngestRequest) -> IngestResponse:
    state = _require_state()
    state.original_store.put(
        req.doc_id,
        req.raw_text,
        business_type=req.business_type,
        source_metadata=req.source_metadata,
    )
    try:
        result = state.pipeline.run(
            req.doc_id,
            req.raw_text,
            business_type=req.business_type,
            source_metadata=req.source_metadata,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(
        doc_id=result.doc_id,
        success=result.success,
        indexed_count=result.indexed_count,
        failed_count=result.failed_count,
        enhanced=result.enhanced,
        message=result.message,
    )


@router.post("/batch", response_model=BatchIngestResponse, summary="Ingest documents")
def ingest_batch(req: BatchIngestRequest) -> BatchIngestResponse:
    state = _require_state()
    results: list[IngestResponse] = []

    for document in req.documents:
        business_type = document.business_type or req.business_type
        state.original_store.put(
            document.doc_id,
            document.raw_text,
            business_type=business_type,
            source_metadata=document.source_metadata,
        )
        try:
            result = state.pipeline.run(
                document.doc_id,
                document.raw_text,
                business_type=business_type,
                source_metadata=document.source_metadata,
            )
            results.append(
                IngestResponse(
                    doc_id=result.doc_id,
                    success=result.success,
                    indexed_count=result.indexed_count,
                    failed_count=result.failed_count,
                    enhanced=result.enhanced,
                    message=result.message,
                )
            )
        except Exception as exc:
            results.append(
                IngestResponse(
                    doc_id=document.doc_id,
                    success=False,
                    indexed_count=0,
                    failed_count=0,
                    enhanced=False,
                    message=str(exc),
                )
            )

    succeeded = sum(1 for result in results if result.success)
    return BatchIngestResponse(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
    )
