"""
core/serving/registry.py

ServingRegistry: single point of truth for all model service clients.

Responsibilities
----------------
1. Build one EmbeddingClient per configured embedding model.
2. Build one RerankerClient for the configured reranker.
3. Run the warm-up gate (wait_until_ready) for each endpoint at startup.
   No traffic is routed until all configured services are healthy.
4. Expose clients to the rest of the pipeline by model id / role.

Usage (in FastAPI lifespan)
---------------------------
    registry = ServingRegistry.from_config(cfg)
    registry.wait_all_ready()   # blocks until all TEI services are warm
    # store on app state, pass to pipeline components

Design rules
------------
- Registry is constructed once at service startup; it is read-only after that.
- Adding a new embedding model = one new entry in config; no code changes.
- If any service fails the warm-up gate the process aborts (fail-fast).
  A partially-warm pipeline would produce silent recall degradation.
"""

from __future__ import annotations

import logging
import concurrent.futures
from dataclasses import dataclass, field

from core.config.models import AppConfig
from core.serving.embed import EmbeddingClient
from core.serving.rerank import RerankerClient
from core.serving.health import ServiceNotReadyError, wait_until_ready

logger = logging.getLogger(__name__)


@dataclass
class ServingRegistry:
    """
    Holds all model service clients, keyed for fast lookup.

    Attributes
    ----------
    embedding_clients : dict[str, EmbeddingClient]
        Keyed by model id (matches EmbeddingModelConfig.id).
    reranker_client : RerankerClient
    """

    embedding_clients: dict[str, EmbeddingClient] = field(default_factory=dict)
    reranker_client: RerankerClient = field(default=None)  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "ServingRegistry":
        """Build registry from the resolved AppConfig. Does NOT block on warm-up yet."""
        embedding_clients = {
            emb.id: EmbeddingClient(emb)
            for emb in cfg.models.embeddings
        }
        reranker_client = RerankerClient(cfg.models.reranker)
        return cls(
            embedding_clients=embedding_clients,
            reranker_client=reranker_client,
        )

    # ------------------------------------------------------------------
    # Warm-up gate
    # ------------------------------------------------------------------

    def wait_all_ready(
        self,
        *,
        timeout_seconds: float = 300.0,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        """
        Block until every configured model service reports healthy.
        Probes all endpoints concurrently (they load in parallel on startup).

        Raises ServiceNotReadyError if any service does not become ready
        within timeout_seconds. The process should abort on this error.
        """
        endpoints: list[tuple[str, str]] = []  # (name, endpoint)

        for client in self.embedding_clients.values():
            endpoints.append((
                f"embed:{client.model_id}",
                client._config.endpoint,
            ))
        endpoints.append((
            "reranker",
            self.reranker_client._config.endpoint,
        ))

        logger.info(
            "Waiting for %d model service(s) to become ready ...", len(endpoints)
        )

        errors: list[str] = []

        # Probe all endpoints concurrently to minimise startup latency
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(endpoints), thread_name_prefix="serving-warmup"
        ) as pool:
            futures = {
                pool.submit(
                    wait_until_ready,
                    endpoint,
                    timeout_seconds=timeout_seconds,
                    poll_interval_seconds=poll_interval_seconds,
                    service_name=name,
                ): name
                for name, endpoint in endpoints
            }
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                    logger.info("  ✓ %s ready", name)
                except ServiceNotReadyError as exc:
                    errors.append(str(exc))
                    logger.error("  ✗ %s NOT ready: %s", name, exc)

        if errors:
            raise ServiceNotReadyError(
                f"{len(errors)} model service(s) failed warm-up:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

        logger.info("All model services ready.")

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_embedding_client(self, model_id: str) -> EmbeddingClient:
        """
        Return the EmbeddingClient for the given model id.
        Raises KeyError if the model_id is not in config.
        """
        try:
            return self.embedding_clients[model_id]
        except KeyError:
            available = list(self.embedding_clients)
            raise KeyError(
                f"No embedding client for model_id='{model_id}'. "
                f"Available: {available}"
            ) from None

    def all_embedding_clients(self) -> list[EmbeddingClient]:
        """Return all embedding clients, in config order."""
        return list(self.embedding_clients.values())
