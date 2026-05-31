"""FastAPI routers for the ingestion service."""

from core.ingestion.routers import crud, embedding, indexer

__all__ = ["crud", "embedding", "indexer"]
