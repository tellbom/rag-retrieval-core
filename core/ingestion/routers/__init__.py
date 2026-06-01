"""FastAPI routers for the ingestion service."""

from core.ingestion.routers import crud, embedding, indexer, ingest

__all__ = ["crud", "embedding", "indexer", "ingest"]
