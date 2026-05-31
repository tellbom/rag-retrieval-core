"""
core/storage/es/client.py

Thin wrapper around the elasticsearch-py 7.x client.

Responsibilities
----------------
- Construction from config (hosts, timeout, retry).
- Ping / connection verification.
- Expose the raw client for provisioner and pipeline use.

Design rule: this class does NOT own index logic.  Provisioner owns create/alias;
pipeline components own search/index/delete.  The raw client is accessible for
components that need direct ES API access.
"""

from __future__ import annotations

import logging
from typing import Any

from elasticsearch import Elasticsearch, ConnectionError as ESConnectionError

logger = logging.getLogger(__name__)


class ESClient:
    """
    Wraps an Elasticsearch 7.x connection.

    Parameters
    ----------
    hosts:
        List of ES host strings, e.g. ["http://localhost:9200"].
    timeout:
        Request timeout in seconds.
    max_retries:
        Number of retries on transient failures.
    """

    def __init__(
        self,
        hosts: list[str],
        *,
        timeout: int = 30,
        max_retries: int = 3,
        retry_on_timeout: bool = True,
    ) -> None:
        self._hosts = hosts
        self._client = Elasticsearch(
            hosts,
            timeout=timeout,
            max_retries=max_retries,
            retry_on_timeout=retry_on_timeout,
            # ES 7.x: suppress the version-mismatch warning if client is 7.x too
            http_compress=True,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def raw(self) -> Elasticsearch:
        """Direct access to the underlying elasticsearch.Elasticsearch client."""
        return self._client

    def ping(self) -> bool:
        """Return True if the ES cluster is reachable."""
        try:
            return self._client.ping()
        except ESConnectionError:
            return False

    def verify_connection(self) -> None:
        """
        Raise ESConnectionError with a descriptive message if ES is unreachable.
        Called at startup before any provisioning or indexing.
        """
        if not self.ping():
            raise ESConnectionError(
                f"Cannot reach Elasticsearch at {self._hosts}. "
                "Verify the cluster is running and hosts are correct."
            )
        info = self._client.info()
        version = info.get("version", {}).get("number", "unknown")
        logger.info(
            "Connected to Elasticsearch %s at %s", version, self._hosts
        )
        # Warn if not 7.x — our mapping and provisioning assume 7.x behaviour
        if not version.startswith("7."):
            logger.warning(
                "Elasticsearch version %s detected; this codebase targets 7.x. "
                "Some behaviours (mapping, alias API) may differ.",
                version,
            )

    @classmethod
    def from_settings(
        cls,
        hosts: list[str],
        timeout: int = 30,
        max_retries: int = 3,
    ) -> "ESClient":
        """Convenience factory used by provisioner and app startup."""
        return cls(hosts, timeout=timeout, max_retries=max_retries)
