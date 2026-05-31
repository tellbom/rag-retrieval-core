"""Model serving clients and warm-up registry."""

from core.serving.embed import EmbeddingClient, EmbeddingError
from core.serving.health import ServiceNotReadyError, is_healthy, wait_until_ready
from core.serving.registry import ServingRegistry
from core.serving.rerank import RerankScore, RerankUnavailableError, RerankerClient

__all__ = [
    "EmbeddingClient",
    "EmbeddingError",
    "RerankScore",
    "RerankUnavailableError",
    "RerankerClient",
    "ServiceNotReadyError",
    "ServingRegistry",
    "is_healthy",
    "wait_until_ready",
]
