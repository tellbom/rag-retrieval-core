"""FastAPI routers for the ingestion service."""

from core.ingestion.routers import embedding, indexer

__all__ = ["embedding", "indexer"]
