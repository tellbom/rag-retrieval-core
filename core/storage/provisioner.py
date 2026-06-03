"""
core/storage/provisioner.py

StorageProvisioner: unified entry point for provisioning both ES and Qdrant.

This is what app lifespan hooks call.  It:
  1. Reads connection settings from AppConfig.storage or environment fallback.
  2. Verifies connectivity to both stores.
  3. Runs ES provisioning (index + alias).
  4. Runs Qdrant provisioning (collection + payload indexes + alias).

StorageSettings
---------------
Connection parameters live here — separate from AppConfig which is purely
about RAG behaviour (chunking, retrieval, models).  Connection settings are
environment-specific (dev/prod host:port differ); RAG behaviour settings are
the same across environments for the same business type.

Environment variables (all optional, have defaults for local dev)
-----------------------------------------------------------------
    RAG_ES_HOSTS          — comma-separated, e.g. "http://localhost:9200"
    RAG_ES_TIMEOUT        — seconds, default 30
    RAG_QDRANT_URL        — e.g. "http://localhost:6333"
    RAG_QDRANT_API_KEY    — optional, default empty
    RAG_QDRANT_TIMEOUT    — seconds, default 30
    RAG_STORAGE_BASE_NAME — index/collection base name, default "rag_chunks"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from core.config.models import AppConfig
from core.storage.es.client import ESClient
from core.storage.es.provisioner import ESProvisioner, ProvisionResult
from core.storage.qdrant.client import QdrantClientWrapper
from core.storage.qdrant.provisioner import QdrantProvisioner, QdrantProvisionResult

logger = logging.getLogger(__name__)


@dataclass
class StorageSettings:
    """
    Connection settings for ES and Qdrant.
    Loaded from AppConfig.storage or environment fallback; can also be
    constructed directly in tests.
    """
    es_hosts: list[str] = field(
        default_factory=lambda: _split_env("RAG_ES_HOSTS", "http://localhost:9200")
    )
    es_timeout: int = field(
        default_factory=lambda: int(os.environ.get("RAG_ES_TIMEOUT", "30"))
    )
    es_max_retries: int = field(
        default_factory=lambda: int(os.environ.get("RAG_ES_MAX_RETRIES", "3"))
    )
    qdrant_url: str = field(
        default_factory=lambda: os.environ.get("RAG_QDRANT_URL", "http://localhost:6333")
    )
    qdrant_api_key: str | None = field(
        default_factory=lambda: os.environ.get("RAG_QDRANT_API_KEY") or None
    )
    qdrant_timeout: int = field(
        default_factory=lambda: int(os.environ.get("RAG_QDRANT_TIMEOUT", "30"))
    )
    base_name: str = field(
        default_factory=lambda: os.environ.get("RAG_STORAGE_BASE_NAME", "rag_chunks")
    )

    @classmethod
    def from_env(cls) -> "StorageSettings":
        """Explicitly load from environment (same as default_factory, but more readable)."""
        return cls()

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "StorageSettings":
        """Load from AppConfig.storage, falling back to environment for old configs."""
        storage = getattr(cfg, "storage", None)
        if storage is None:
            return cls.from_env()

        qdrant_api_key = None
        if storage.qdrant.api_key_env:
            qdrant_api_key = os.environ.get(storage.qdrant.api_key_env)
            if not qdrant_api_key:
                raise ValueError(
                    f"Qdrant api_key_env '{storage.qdrant.api_key_env}' is configured "
                    "but the environment variable is not set"
                )

        return cls(
            es_hosts=storage.elasticsearch.hosts,
            es_timeout=storage.elasticsearch.timeout_seconds,
            es_max_retries=storage.elasticsearch.max_retries,
            qdrant_url=storage.qdrant.url,
            qdrant_api_key=qdrant_api_key,
            qdrant_timeout=storage.qdrant.timeout_seconds,
            base_name=storage.base_name,
        )


def _split_env(key: str, default: str) -> list[str]:
    val = os.environ.get(key, default)
    return [h.strip() for h in val.split(",") if h.strip()]


@dataclass
class FullProvisionResult:
    es: ProvisionResult
    qdrant: QdrantProvisionResult


class StorageProvisioner:
    """
    Coordinates ES + Qdrant provisioning.

    Usage
    -----
        settings = StorageSettings.from_config(cfg)
        provisioner = StorageProvisioner(settings, cfg)
        result = provisioner.provision()
        # result.es.alias_name  — use this for all ES operations
        # result.qdrant.alias_name — use this for all Qdrant operations
    """

    def __init__(self, settings: StorageSettings, cfg: AppConfig) -> None:
        self._settings = settings
        self._cfg = cfg

        self._es_client = ESClient.from_settings(
            hosts=settings.es_hosts,
            timeout=settings.es_timeout,
            max_retries=settings.es_max_retries,
        )
        self._qdrant_client = QdrantClientWrapper.from_settings(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout,
        )
        self._es_provisioner = ESProvisioner(
            self._es_client, cfg, base_name=settings.base_name
        )
        self._qdrant_provisioner = QdrantProvisioner(
            self._qdrant_client, cfg, base_name=settings.base_name
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def es_client(self) -> ESClient:
        return self._es_client

    @property
    def qdrant_client(self) -> QdrantClientWrapper:
        return self._qdrant_client

    @property
    def es_alias(self) -> str:
        return self._es_provisioner.alias_name

    @property
    def qdrant_alias(self) -> str:
        return self._qdrant_provisioner.alias_name

    def verify_connections(self) -> None:
        """Verify ES and Qdrant are reachable. Raises on failure."""
        self._es_client.verify_connection()
        self._qdrant_client.verify_connection()

    def provision(self) -> FullProvisionResult:
        """
        Verify connectivity then provision both stores idempotently.
        Raises on any failure — caller should abort startup.
        """
        self.verify_connections()
        es_result = self._es_provisioner.provision()
        qdrant_result = self._qdrant_provisioner.provision()

        logger.info(
            "Storage provisioned — ES: %s (alias=%s), Qdrant: %s (alias=%s)",
            es_result.index_name, es_result.alias_name,
            qdrant_result.collection_name, qdrant_result.alias_name,
        )
        return FullProvisionResult(es=es_result, qdrant=qdrant_result)
